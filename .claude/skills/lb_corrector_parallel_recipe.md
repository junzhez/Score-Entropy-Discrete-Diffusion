# Parallel Locally Balanced Corrector for SEDD

A predictor-corrector sampler for SEDD where the corrector proposes flips at *every* position simultaneously and accepts/rejects each position independently. Uses a locally balanced (Zanella) Metropolis-Hastings step under the mean-field factorization that SEDD's tau-leap already assumes.

## Why this design

SEDD's tau-leap sampler treats positions as conditionally independent given the current state:

```
p_t(y) / p_t(x) ≈ ∏_i s_θ(x, t)_{i, y_i}
```

This factorization is built into SEDD. Once we accept it, joint MH at fixed `t` decomposes into independent per-position MH updates. The chain targets the mean-field factorization `p̃_t(x) = ∏_i p_t(x_i | x_{-i}, t)` rather than the full `p_t(x)`, but tau-leap *already* targets this factorization — the corrector adds no new bias relative to the predictor.

The payoff: 2 SEDD calls per corrector step, up to `L` accepted position-flips per step. For `L = 1024` this is up to 1024× more flips per SEDD call than sequential single-token MH.

## Mathematical specification

**Target:** `p̃_t(x) = ∏_i p_t(x_i | x_{-i}, t)`. Inherited from SEDD's tau-leap factorization.

**Per-position proposal:** at position `i`, draw the proposed token `y_i ∈ {1, ..., K}` from the locally balanced categorical:

```
q_i(y_i = k | x) ∝ g(s_θ(x, t)_{i, k})       for k ≠ x_i
q_i(y_i = x_i | x) ∝ g(1)                    (the "stay" option)
```

Normalize over all K options including the stay weight. Sample one `y_i` per position, independently across positions.

**Per-position acceptance:** under the LB balancing condition `g(r) = r · g(1/r)` and SEDD's mean-field factorization, the per-position acceptance ratio collapses to:

```
α_i(x_i → y_i) = min(1, Z_{x,i} / Z_{y,i})
```

where

```
Z_{x,i} = Σ_k g(s_θ(x, t)_{i, k})
Z_{y,i} = Σ_k g(s_θ(y, t)_{i, k})
```

The second normalizer requires `s_θ(y, t)` evaluated at the *full proposed state* y, which is one SEDD call regardless of how many positions are accepted.

**Choice of g:**
- `g(r) = r / (1 + r)` — Barker. Default. Robust to the extreme dynamic range of SEDD scores near `t = 0` (where one neighbor may have score ~10³ and others ~10⁻³).
- `g(r) = sqrt(r)` — Zanella. Try if Barker mixes too slowly.

## Cost per noise level

- **Predictor (Euler step):** 1 SEDD call (or 0 with cross-stage cache).
- **Per corrector step:** 2 SEDD calls (one at `x`, one at proposed `y`). Cannot cache across corrector iterations because per-position acceptance produces a mixed state.
- **K corrector steps:** 2K SEDD calls per noise level.

Total per noise level: `1 + 2K` SEDD calls.

## Caveat on the mean-field assumption

The factorization is exact for SEDD's tau-leap but is an *approximation* to the true `p_t`. Specifically, it ignores correlations between positions that flip simultaneously. For text, adjacent positions are strongly correlated; flipping two adjacent positions independently violates the factorization more than flipping two distant positions.

If you observe quality degradation at low `t` (near data), reduce the per-step update density: at each corrector step, sample a random subset of positions (e.g., 50%) to update, leaving the rest at `x_i`. This bounds the factorization error per step at the cost of needing more steps to sweep all positions.

## Implementation

```python
"""
lb_corrector_parallel.py

Parallel locally balanced Metropolis-Hastings corrector for SEDD.
Propose flips at every position; accept/reject each position independently
under SEDD's mean-field factorization.
"""

import math
from typing import Callable, Optional, Tuple
import torch
from torch import LongTensor, FloatTensor


def make_balancing_function(name: str = "barker") -> Callable[[FloatTensor], FloatTensor]:
    """Returns a numerically stable elementwise g function."""
    if name == "barker":
        return lambda r: r / (1.0 + r)
    elif name == "sqrt":
        return lambda r: torch.sqrt(r.clamp(min=0.0))
    else:
        raise ValueError(f"Unknown balancing function: {name}")


def lb_corrector_parallel_step(
    x: LongTensor,                              # [B, L]
    t: float,
    score_fn: Callable,                         # (x, t) -> [B, L, K]
    g: Callable[[FloatTensor], FloatTensor],
    update_fraction: float = 1.0,
    eps: float = 1e-30,
) -> LongTensor:
    """
    One parallel LB corrector step at noise level t.

    Proposes a flip at every position (or a random subset if update_fraction < 1),
    accepts/rejects each position independently.

    Args:
        x: current state, [B, L] of token indices.
        t: current noise level.
        score_fn: SEDD scoring function, returns [B, L, K].
        g: balancing function (elementwise).
        update_fraction: fraction of positions to update per step. 1.0 = all
            positions. Lower values bound mean-field error at the cost of
            slower per-step coverage. Recommended: 0.5 for text at low t.
        eps: numerical floor for score values.

    Returns:
        new_x: [B, L] updated state.
    """
    B, L = x.shape
    device = x.device

    # First SEDD call: s_θ(x, t)
    score_x = score_fn(x, t).clone()

    K = score_x.shape[-1]
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
    pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    # Zero diagonal — SEDD conventionally returns 0 at self-transitions,
    # but we enforce it explicitly to be safe.
    score_x[batch_idx, pos_idx, x] = 0.0
    score_x = score_x.clamp(min=eps)

    # Build per-position proposal weights including a "stay" weight at the
    # diagonal. The stay weight is g(1) (since s_θ at the diagonal is
    # conceptually 1: p_t(x)/p_t(x) = 1).
    g_x = g(score_x)                                    # [B, L, K]
    g_self = g(torch.tensor(1.0, device=device)).item()
    full_g_x = g_x.clone()
    full_g_x[batch_idx, pos_idx, x] = g_self            # set diagonal to g(1)

    Z_x_per_pos = full_g_x.sum(dim=-1)                  # [B, L]
    probs_per_pos = full_g_x / Z_x_per_pos.unsqueeze(-1)  # [B, L, K]

    # Sample proposed token per position
    flat_probs = probs_per_pos.view(B * L, K)
    proposed = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
    proposed = proposed.view(B, L)                      # [B, L]

    # Optional: restrict updates to a random subset of positions
    if update_fraction < 1.0:
        mask = torch.rand(B, L, device=device) < update_fraction
        proposed = torch.where(mask, proposed, x)

    # Proposed full state y
    y = proposed

    # Positions that actually changed (where y differs from x)
    flipped = (y != x)                                  # [B, L] bool

    # Early-exit: if no positions flipped anywhere, skip the second SEDD call
    if not flipped.any():
        return x

    # Second SEDD call: s_θ(y, t) at the proposed full state
    score_y = score_fn(y, t).clone()
    score_y[batch_idx, pos_idx, y] = 0.0
    score_y = score_y.clamp(min=eps)

    g_y = g(score_y)
    full_g_y = g_y.clone()
    full_g_y[batch_idx, pos_idx, y] = g_self
    Z_y_per_pos = full_g_y.sum(dim=-1)                  # [B, L]

    # Per-position acceptance: α_i = min(1, Z_{x,i} / Z_{y,i})
    log_alpha_per_pos = torch.log(Z_x_per_pos) - torch.log(Z_y_per_pos)
    log_u = torch.log(torch.rand_like(log_alpha_per_pos))
    accept_per_pos = log_u < log_alpha_per_pos          # [B, L] bool

    # Positions where we did NOT propose a flip get accepted trivially
    # (proposing y_i = x_i has α_i = 1; the formula gives this automatically
    # since Z_{x,i} = Z_{y,i} when y agrees with x at all positions, but with
    # per-position updates we need to handle this carefully).
    # The clean rule: only override x_i with y_i when (a) flipped, AND (b) accepted.
    do_update = flipped & accept_per_pos                # [B, L]
    new_x = torch.where(do_update, y, x)

    return new_x


def lb_corrector_parallel_loop(
    x: LongTensor,
    t: float,
    score_fn: Callable,
    g: Callable,
    K_steps: int,
    update_fraction: float = 1.0,
) -> LongTensor:
    """Run K_steps parallel LB corrector iterations at fixed t."""
    for _ in range(K_steps):
        x = lb_corrector_parallel_step(
            x, t, score_fn, g,
            update_fraction=update_fraction,
        )
    return x


# -----------------------------------------------------------------------------
# Integration into predictor-corrector loop
# -----------------------------------------------------------------------------


def predictor_corrector_sample(
    score_fn: Callable,
    euler_step: Callable,                       # (x, t, dt) -> new_x
    x_init: LongTensor,
    schedule: list,                             # decreasing list of t values
    K_corrector: int = 1,
    balancing: str = "barker",
    update_fraction: float = 1.0,
    corrector_predicate: Optional[Callable[[float], bool]] = None,
) -> LongTensor:
    """
    Predictor-corrector sampler with parallel LB corrector.

    Args:
        score_fn: SEDD scoring function.
        euler_step: predictor; advances x from t to t - dt.
        x_init: initial state at t = schedule[0], typically drawn from p_T.
        schedule: time grid, strictly decreasing.
        K_corrector: number of corrector iterations per noise level.
        balancing: "barker" or "sqrt".
        update_fraction: per-step fraction of positions to update. 1.0 = all.
        corrector_predicate: optional fn(t) -> bool to gate corrector by level.

    Returns:
        Final sample at t = schedule[-1].
    """
    g = make_balancing_function(balancing)
    x = x_init

    for i in range(len(schedule) - 1):
        t_now = schedule[i]
        t_next = schedule[i + 1]
        dt = t_now - t_next

        # Predictor — unchanged
        x = euler_step(x, t_now, dt)

        # Corrector at the new noise level t_next
        run_corrector = corrector_predicate(t_next) if corrector_predicate else True
        if K_corrector > 0 and run_corrector:
            x = lb_corrector_parallel_loop(
                x, t_next, score_fn, g, K_corrector,
                update_fraction=update_fraction,
            )

    return x


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def lb_corrector_parallel_step_with_diagnostics(
    x: LongTensor,
    t: float,
    score_fn: Callable,
    g: Callable,
    update_fraction: float = 1.0,
    eps: float = 1e-30,
) -> Tuple[LongTensor, dict]:
    """
    Parallel LB corrector step with diagnostics for monitoring.

    Returned diagnostics:
        - per_position_accept_rate: fraction of positions where a proposed
          flip was accepted (averaged over batch)
        - proposal_flip_rate: fraction of positions where a flip was proposed
          (i.e., y_i != x_i)
        - effective_flip_rate: per_position_accept_rate * proposal_flip_rate
          (fraction of positions actually updated)
        - mean_log_alpha: mean of log(Z_x_i / Z_y_i) across flipped positions
    """
    B, L = x.shape
    device = x.device

    score_x = score_fn(x, t).clone()
    K = score_x.shape[-1]
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
    pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    score_x[batch_idx, pos_idx, x] = 0.0
    score_x = score_x.clamp(min=eps)

    g_x = g(score_x)
    g_self = g(torch.tensor(1.0, device=device)).item()
    full_g_x = g_x.clone()
    full_g_x[batch_idx, pos_idx, x] = g_self
    Z_x_per_pos = full_g_x.sum(dim=-1)
    probs_per_pos = full_g_x / Z_x_per_pos.unsqueeze(-1)

    flat_probs = probs_per_pos.view(B * L, K)
    proposed = torch.multinomial(flat_probs, num_samples=1).squeeze(-1).view(B, L)

    if update_fraction < 1.0:
        mask = torch.rand(B, L, device=device) < update_fraction
        proposed = torch.where(mask, proposed, x)

    y = proposed
    flipped = (y != x)

    if not flipped.any():
        return x, {
            "per_position_accept_rate": torch.tensor(0.0),
            "proposal_flip_rate": torch.tensor(0.0),
            "effective_flip_rate": torch.tensor(0.0),
            "mean_log_alpha": torch.tensor(0.0),
        }

    score_y = score_fn(y, t).clone()
    score_y[batch_idx, pos_idx, y] = 0.0
    score_y = score_y.clamp(min=eps)
    g_y = g(score_y)
    full_g_y = g_y.clone()
    full_g_y[batch_idx, pos_idx, y] = g_self
    Z_y_per_pos = full_g_y.sum(dim=-1)

    log_alpha_per_pos = torch.log(Z_x_per_pos) - torch.log(Z_y_per_pos)
    log_u = torch.log(torch.rand_like(log_alpha_per_pos))
    accept_per_pos = log_u < log_alpha_per_pos

    do_update = flipped & accept_per_pos
    new_x = torch.where(do_update, y, x)

    diagnostics = {
        "per_position_accept_rate": (accept_per_pos & flipped).float().sum() / flipped.float().sum().clamp(min=1),
        "proposal_flip_rate": flipped.float().mean(),
        "effective_flip_rate": do_update.float().mean(),
        "mean_log_alpha": log_alpha_per_pos[flipped].mean(),
    }
    return new_x, diagnostics
```

## How to wire into your existing predictor-corrector

If your current loop is:

```python
def sample(score_fn, euler_step, x_init, schedule, K):
    x = x_init
    for i in range(len(schedule) - 1):
        t = schedule[i]; t_next = schedule[i + 1]
        x = euler_step(x, t, t - t_next)
        for _ in range(K):
            x = my_corrector(x, t_next, score_fn)   # <-- replace this
    return x
```

Replace the corrector call:

```python
g = make_balancing_function("barker")
...
for _ in range(K):
    x = lb_corrector_parallel_step(x, t_next, score_fn, g)
```

The Euler step and the surrounding loop structure are unchanged.

## Cost vs vanilla SEDD

For `K = 1` corrector step per noise level: 2 extra SEDD calls per level. Total cost is roughly 3× vanilla SEDD at the same number of noise levels.

For a fair comparison, match total NFE. If vanilla SEDD uses N noise levels with 1 SEDD call each (N total calls), the parallel LB corrector with `K = 1` should use `N / 3` noise levels (also N total calls). The question is whether `N/3` levels with `K = 1` LB-MH refinement beats `N` levels of pure Euler.

The expected win: at low NFE budgets, where Euler is under-resolved, LB-MH refinement at each level recovers per-position marginal accuracy that vanilla SEDD would only achieve with many more levels. At high NFE budgets, vanilla SEDD's Euler is already accurate and corrector becomes redundant.

## Validation checklist

Before reporting perplexity numbers, verify:

1. **`K = 0` matches vanilla SEDD.** With `K_corrector = 0`, the sampler must produce identical samples to your baseline. If not, the corrector is leaking.

2. **`per_position_accept_rate` is in [0.3, 0.8].** Lower means the proposal is too aggressive — try `g = sqrt` if using Barker, or vice versa, or lower `update_fraction`. Higher (near 1) means the proposal is too timid — possibly all positions are proposing `y_i = x_i`. Check `proposal_flip_rate`.

3. **`proposal_flip_rate` is in [0.1, 0.5].** This is the fraction of positions where the LB sampler proposed a flip (vs the stay option). If it's near 0, the `g(1)` self-weight is dominating; check that scores are nonzero in expected places. If it's near 1, the stay weight is too small relative to score magnitudes.

4. **Matched-NFE comparison.** Pick a total NFE budget. Run vanilla SEDD with `N` levels. Run parallel LB-corrector with `N // 3` levels and `K = 1`. Compare generative perplexity (e.g., under GPT-2 or Llama as external evaluator). Report only matched-NFE wins.

5. **Update-fraction ablation.** Try `update_fraction ∈ {0.25, 0.5, 1.0}`. Lower values bound mean-field error per step; higher values mix faster per step. The sweet spot is task-dependent.

## Pitfalls

- **Diagonal not zeroed.** SEDD wrappers sometimes return nonzero diagonal scores. The recipe zeroes them explicitly with `score_x[batch_idx, pos_idx, x] = 0.0`. Do not remove this line.

- **`g(1)` self-weight choice.** The recipe sets the diagonal to `g(1)` (Barker: 0.5; sqrt: 1.0). This is the "natural" choice because at the diagonal `r = p_t(x)/p_t(x) = 1`. If you observe `proposal_flip_rate` saturating near 0 or 1, you can rescale the diagonal weight to tune the stay/flip balance (multiply `g_self` by a temperature factor).

- **Mean-field error at low t.** Near the data distribution, SEDD scores become highly peaked and positions are strongly correlated. Parallel updates can produce inconsistent token combinations (e.g., flipping two adjacent positions to forms that don't go together). Mitigations: (a) set `update_fraction < 1.0` to update fewer positions per step; (b) gate the parallel corrector to only run at intermediate `t`, falling back to vanilla SEDD or sequential MH for `t < t_threshold`; (c) accept the bias and verify empirically that perplexity still improves.

- **No cross-step caching.** Unlike the sequential variant, the parallel corrector cannot cache `score_y` for the next iteration's `score_x`, because the new state is a mix of `x` and `y` per-position. Each corrector iteration pays 2 fresh SEDD calls. If this is prohibitive, reduce `K_corrector` rather than trying to cache.

- **Batch dimension.** The recipe assumes `[B, L]` input. If your sampler operates on single sequences, add a singleton batch dim before calling.

## References

- Zanella, G. (2020). Informed Proposals for Local MCMC in Discrete Spaces. *JASA* 115(530), 852-865. arXiv:1711.07424.
- Lou, A., Meng, C., Ermon, S. (2024). Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution. *ICML*.
- Sun, H., Dai, H., Xia, W., Ramamurthy, A. (2021). Path Auxiliary Proposal for MCMC in Discrete Space. *ICLR 2022*. — Discusses parallel multi-site updates and their mean-field cost.
