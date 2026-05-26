import abc
import torch
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


_CORRECTORS = {}


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        local_name = cls.__name__ if name is None else name
        if local_name in _CORRECTORS:
            raise ValueError(
                f'Already registered corrector with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_corrector(name):
    return _CORRECTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
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
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t):
        """One corrector update at a fixed noise level t. Returns the next state."""
        pass


@register_corrector(name="none")
class NoneCorrector(Corrector):
    def update_fn(self, score_fn, x, t):
        return x


@register_corrector(name="lb_mean_field")
class LBMeanFieldCorrector(Corrector):
    """Unified locally-balanced (Zanella) Metropolis-Hastings corrector.

    At a fixed noise level, proposes a token flip at eligible positions
    simultaneously and accepts/rejects each position independently under
    SEDD's mean-field factorization (the same factorization the tau-leap
    predictor already assumes). The variant (Uniform vs Absorb) is detected
    from ``graph.absorb`` and selects both the proposal mask (which flips
    Q^tok allows) and the acceptance form:

    - Uniform: any token may flip to any other; Zanella Z_x/Z_y acceptance
      (two score-function calls per step).
    - Absorb: only masked positions flip to data tokens (unmasked positions
      are frozen); noising-kernel reverse acceptance (one score call).

    Both variants use Barker acceptance. See
    .claude/skills/lb_corrector_unified_recipe.md for the derivation.
    """

    def __init__(self, graph, noise, *, balancing="barker", update_fraction=1.0, eps=1e-30):
        super().__init__(graph, noise)
        self.update_fraction = update_fraction
        self.eps = eps
        self.last_info = None       # per-step diagnostics, stashed for inspection
        if balancing == "barker":
            self._g = lambda r: r / (1.0 + r)
            self._g_self = 0.5          # g(1)
        elif balancing == "sqrt":
            self._g = lambda r: torch.sqrt(r.clamp(min=0.0))
            self._g_self = 1.0          # g(1)
        else:
            raise ValueError(f"Unknown balancing function: {balancing}")

    def update_fn(self, score_fn, x, t):
        # Bridge t -> sigma exactly like the predictors; score_fn expects sigma.
        sigma = self.noise(t)[0]
        g, g_self, eps = self._g, self._g_self, self.eps
        absorb = self.graph.absorb
        B, L = x.shape
        device = x.device
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        # First score call at x. Zero the diagonal, clamp, build the LB proposal.
        # Cast to float32: score_fn runs under bf16 autocast and the small eps /
        # log below would underflow at bf16 precision.
        score_x = score_fn(x, sigma).float()
        score_x[batch_idx, pos_idx, x] = 0.0
        score_x = score_x.clamp(min=eps)
        K = score_x.shape[-1]

        # Transition mask from Q^tok: which flips x_i -> k are allowed.
        g_x = g(score_x)
        if absorb:
            mask_id = self.graph.dim - 1
            is_masked = (x == mask_id)                                  # [B, L]
            not_mask_token = (torch.arange(K, device=device) != mask_id)
            transition_mask = is_masked.unsqueeze(-1) & not_mask_token.view(1, 1, K)
            full_g_x = torch.where(transition_mask, g_x, torch.zeros_like(g_x))
        else:
            full_g_x = g_x.clone()
        full_g_x[batch_idx, pos_idx, x] = g_self    # "stay" weight = g(1)
        Z_x = full_g_x.sum(dim=-1)                  # [B, L]
        probs = full_g_x / Z_x.unsqueeze(-1)

        proposed = torch.multinomial(probs.view(B * L, K), 1).squeeze(-1).view(B, L)

        # Eligible positions: where a flip is even possible. Uniform -> all;
        # Absorb -> masked positions only. update_fraction subsamples within these.
        eligible = is_masked if absorb else torch.ones_like(x, dtype=torch.bool)
        if self.update_fraction < 1.0:
            do_propose = eligible & (torch.rand(B, L, device=device) < self.update_fraction)
        else:
            do_propose = eligible
        y = torch.where(do_propose, proposed, x)

        flipped = (y != x)
        if not flipped.any():
            self.last_info = {
                "acceptance_rate": torch.tensor(1.0),
                "flip_rate": torch.tensor(0.0),
                "n_proposed_flips": torch.tensor(0.0),
                "mean_log_rho": torch.tensor(0.0),
            }
            return x                                # skip the second score call

        # Per-position acceptance ratio rho_i, variant-specific.
        if absorb:
            # Noising-kernel reverse: rho_i = s(x)_{i,y_i} * Z_x_i / g(s(x)_{i,y_i}).
            # No second score call needed.
            s_at_y = score_x.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            g_s_at_y = g_x.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            log_rho = (
                torch.log(s_at_y.clamp(min=eps))
                + torch.log(Z_x.clamp(min=eps))
                - torch.log(g_s_at_y.clamp(min=eps))
            )
        else:
            # Zanella clean cancellation: rho_i = Z_x_i / Z_y_i.
            # Second score call at the proposed state y.
            score_y = score_fn(y, sigma).float()
            score_y[batch_idx, pos_idx, y] = 0.0
            score_y = score_y.clamp(min=eps)
            full_g_y = g(score_y)
            full_g_y[batch_idx, pos_idx, y] = g_self
            Z_y = full_g_y.sum(dim=-1)
            log_rho = torch.log(Z_x.clamp(min=eps)) - torch.log(Z_y.clamp(min=eps))

        # Barker acceptance per position: log alpha = log sigmoid(log_rho).
        log_alpha = -F.softplus(-log_rho)
        accept = torch.log(torch.rand_like(log_alpha).clamp(min=eps)) < log_alpha

        do_commit = flipped & accept
        new_x = torch.where(do_commit, y, x)

        n_proposed = flipped.float().sum()
        n_accepted = do_commit.float().sum()
        self.last_info = {
            "acceptance_rate": (n_accepted / n_proposed.clamp(min=1)).detach(),
            "flip_rate": (n_accepted / float(B * L)).detach(),
            "n_proposed_flips": n_proposed.detach(),
            "mean_log_rho": log_rho[flipped].mean().detach(),
        }
        return new_x


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
                                 device=device,
                                 corrector=config.sampling.get("corrector", "none"),
                                 corrector_steps=config.sampling.get("corrector_steps", 0),
                                 balancing=config.sampling.get("balancing", "barker"),
                                 update_fraction=config.sampling.get("update_fraction", 1.0),
                                 corrector_t_threshold=config.sampling.get("corrector_t_threshold", 0.0))

    return sampling_fn


def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x,
                   corrector="none", corrector_steps=0, balancing="barker", update_fraction=1.0, corrector_t_threshold=0.0):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    use_corrector = corrector != "none" and corrector_steps > 0
    corrector_obj = (
        get_corrector(corrector)(graph, noise, balancing=balancing, update_fraction=update_fraction)
        if use_corrector else None
    )

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)

            # Corrector at the new (lower) noise level t_next, optionally gated by t.
            if use_corrector and timesteps[i + 1].item() >= corrector_t_threshold:
                t_next = timesteps[i + 1] * torch.ones(x.shape[0], 1, device=device)
                for _ in range(corrector_steps):
                    x = projector(x)
                    x = corrector_obj.update_fn(sampling_score_fn, x, t_next)


        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)
            
        return x
    
    return pc_sampler

