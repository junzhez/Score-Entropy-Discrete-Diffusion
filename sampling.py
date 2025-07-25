import abc
import torch
import losses
import numpy as np
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils

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
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, p=None):
        return x

@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, p=None):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)

@register_predictor(name="hamiltonian")
class HamiltonianPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, p):
        """
        Hamiltonian-inspired update using auxiliary momentum variable p.
        Args:
            score_fn: score function
            x: current state tensor
            t: current time tensor
            step_size: integration step size
            p: momentum tensor (should be same length as x)
        Returns:
            Updated state tensor x.
        """
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        p = p - step_size/2 * score

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate) + step_size * p

        sigma, dsigma = self.noise(t + step_size)
        score = score_fn(x, sigma)
        p = p - step_size/2 * score

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
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        p = torch.randn_like(x, dtype=torch.float32)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        x_prev = x.clone()

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            if type(predictor) is HamiltonianPredictor:
                # Hamiltonian update
                x, p = predictor.update_fn(sampling_score_fn, x, t, dt, p)
            else:
                x = predictor.update_fn(sampling_score_fn, x, t, dt)
        
        loss_fun = losses.get_loss_fn(noise, graph, train=False)
        alpha = torch.clamp(torch.exp(loss_fun(model, x)) / torch.exp(loss_fun(model, x_prev)), max=1.0).item()

        print(alpha)
        u = np.random.randn()

        if u < alpha:
            x = x_prev
        
        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        return x
    
    return pc_sampler

