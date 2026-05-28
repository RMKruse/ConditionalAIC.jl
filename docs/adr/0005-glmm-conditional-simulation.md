# ADR-0005 — GLMM conditional bootstrap: bypass `simulate!`, draw from f(y | μ̂) directly

**Date:** 2026-05-29
**Status:** Accepted

## Context

The GLMM conditional bootstrap (df correction for non-Poisson, non-Bernoulli families) follows
cAIC4's `conditionalBootstrap`, which calls `simulate(object, nsim=B, use.u=TRUE)`. The
`use.u=TRUE` flag holds the random effects **fixed** at their estimated values, so the draws come
from the conditional distribution f(y | b̂), not the marginal.

`MixedModels.jl` v5.5.1 has no `use_u` parameter. Its `simulate!(rng, m::GeneralizedLinearMixedModel)`
always calls `unscaledre!(rng, η, trm)` — drawing **new** random effects for every simulation.
That is a marginal draw, not a conditional one.

## Decision

For the GLMM conditional bootstrap, draw bootstrap samples directly from the conditional
distribution:

```
y_i^(b) ~ f(μ̂_i)    for i = 1…n, b = 1…B
```

where μ̂_i = `m.resp.mu[i]` (accessed via `MMInternals`), and `f` is the GLMM response family
(e.g. `Poisson(μ̂_i)`, `Bernoulli(μ̂_i)`).

This is equivalent to `use.u=TRUE` simulation: μ̂ already encodes b̂ through η̂ = Xβ̂ + Zb̂
(which has been computed at the fitted parameter values). There is no need to re-enter the
PIRLS loop.

## Alternatives considered

1. **Hack `setθ!` to pin random effects, then call `simulate!`.** Complex, depends on internals
   not in the quarantine table, fragile against version bumps, and still wouldn't produce a
   conditional draw — `unscaledre!` would still draw new u.

2. **Override `m.u` in the deep copy before calling `simulate!`.** Mutating u without re-running
   `updateη!` would leave η inconsistent. Risky.

3. **Add a `use_u` keyword upstream (PR to MixedModels.jl).** Not feasible as a pinned-version
   dependency; adds an uncontrolled upstream dependency.

## Consequences

- The conditional bootstrap for GLMMs is correct and simple: allocate a matrix of draws, one
  column per bootstrap replicate, fill via `rand(rng, Family(μ̂_i))`.
- `m.resp.mu` must be in the `mm_internals.jl` quarantine table before this path is implemented.
- This approach works only for families without a free dispersion parameter (Poisson, Binomial,
  Bernoulli, NegativeBinomial with fixed r). For families with a free dispersion, σ ≠ 1 and the
  Efron covariance formula would need adaptation — matches cAIC4's warning
  (`"Families with a dispersion parameter not yet supported"`).
