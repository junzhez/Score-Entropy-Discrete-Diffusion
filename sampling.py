import abc
import torch
import losses
import numpy as np
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils

from transformers import GPT2TokenizerFast

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size, p):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, p=None):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        
        return x, p

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, p=None):
        return x, p

@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, p=None):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs), p

@register_predictor(name="hamiltonian")
class HamiltonianPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, p):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size/2 * dsigma[..., None] * self.graph.reverse_rate(x, score)
        p = self.graph.sample_rate(p, rev_rate)

        x = self.graph.sample_rate(p, step_size * rev_rate)

        score = score_fn(x, sigma)
                
        rev_rate = step_size / 2 * dsigma[..., None] * self.graph.reverse_rate(x, score)
        p = self.graph.sample_rate(p, rev_rate)
        
        return x, p

class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x,  threshold=False, N=10, temp=0.002):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        log_score_fn = mutils.get_score_fn(model, train=False)
        x = graph.sample_limit(*batch_dims).to(device)
        p = graph.sample_limit(*batch_dims).to(device)

        if threshold:
            timesteps = torch.linspace(1, eps, N * steps + 1, device=device)
            dt = (1 - eps) / (N * steps)
            for j in range(N):
                p_ = p.clone()
                x_ = x.clone()
                for i in range(steps):
                    t = timesteps[i + steps*j] * torch.ones(x.shape[0], 1, device=device)
                    x = projector(x)
                    x, p = predictor.update_fn(sampling_score_fn, x, t, dt, p)

                d = (x != x_)
                di = torch.arange(0, batch_dims[1], device=device).view(1, batch_dims[1])[d]
                dv = x[d]

                for i in range(d.sum()):
                    sigma = noise(torch.tensor(temp, device=device))[0]
                    score = log_score_fn(x_, sigma)
            
                    alpha1 = score[0, di[i], dv[i]]
                    alpha2 = 0.5*(p[0, di[i], :].pow(2) - p_[0, di[i], :].pow(2)).sum()

                    alpha = torch.exp(alpha1 - alpha2).clamp(max=1.0)
                    u = torch.rand(1, device=device)             

                    if u > alpha:
                        print(alpha, u)
                        x[0, di[i]] = x_[0, di[i]]
                        p[0, di[i]] = p_[0, di[i]]
                        
                    del sigma, score

        else:
            timesteps = torch.linspace(1, eps, steps + 1, device=device)
            dt = (1 - eps) / steps
            for i in range(steps):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
                x = projector(x)
                x, p = predictor.update_fn(sampling_score_fn, x, t, dt, p)
        
        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        return x
    
    return pc_sampler

