```@meta
CurrentModule = ConditionalAIC
```

# ConditionalAIC.jl

Conditional Akaike Information Criterion and conditional model selection for
mixed-effects models fitted with
[`MixedModels.jl`](https://github.com/JuliaStats/MixedModels.jl).

`ConditionalAIC.jl` is a Julia re-platforming of R's
[`cAIC4`](https://cran.r-project.org/package=cAIC4). Where `cAIC4` sits on
`lme4`/`merMod`, `ConditionalAIC.jl` sits on `MixedModels.jl` — its
`LinearMixedModel` and `GeneralizedLinearMixedModel`. The work is to reconstruct
`cAIC4`'s bias-correction and degrees-of-freedom logic on top of the fitted
objects `MixedModels.jl` produces, validated against `cAIC4` as ground truth.

## What it computes

The conditional AIC has the schematic form

```math
\mathrm{cAIC} = -2\,\ell_{\mathrm{cond}}(y \mid \hat b, \hat\beta, \hat\theta) + 2\rho
```

where ``\ell_{\mathrm{cond}}`` is the *conditional* log-likelihood given the
predicted random effects ``\hat b``, and ``\rho`` is the bias-corrected effective
degrees of freedom. The naive plug-in df ``\rho_0 = \mathrm{tr}(H_1)``
understates ``\rho`` because the variance parameters ``\theta`` are estimated; the
correction for that estimation uncertainty is the mathematical core of the
package.

**References** (the package's formulas match these):

- Greven, S. & Kneib, T. (2010). On the behaviour of marginal and conditional AIC
  in linear mixed models. *Biometrika*.
- Säfken, B., Rügamer, D., Kneib, T. & Greven, S. (2021). Conditional Model
  Selection in Mixed-Effects Models with `cAIC4`. *Journal of Statistical
  Software*.

## Installation

The package is not yet registered in the General registry. Install it directly
from the repository:

```julia
using Pkg
Pkg.add(url="https://github.com/RMKruse/ConditionalAIC.jl")
```

Requires Julia ≥ 1.10 (LTS). `MixedModels` and `GLM` are pinned to exact versions
(`MixedModels = "=5.5.1"`, `GLM = "=1.9.5"`) because the package depends on
`MixedModels.jl` internals by design — a version bump is a deliberate, reviewed
event.

## Quick start

```julia
using MixedModels, ConditionalAIC

m = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy); REML=false, progress=false)

r = caic(m)
r.caic          # the conditional AIC
r.dof           # ρ, the effective degrees of freedom
r.condloglik    # ℓ_cond
```

See the [Usage guide](guide.md) for scoring, comparison, search, and averaging,
and the [API reference](api.md) for the full public surface.

## Public API

| Function | Purpose |
|----------|---------|
| [`caic`](@ref) | **Score** a single fitted model by its conditional AIC → [`CAICResult`](@ref). |
| [`anocaic`](@ref) | **Compare** a user-supplied set of fitted models, ranked by cAIC → [`AnocaicTable`](@ref). |
| [`stepcaic`](@ref) | **Search** the random-effects structure (backward / forward / both) → [`StepcaicResult`](@ref). |
| [`modelavg`](@ref) | **Average** several models into one weighted combination → [`ModelAvgResult`](@ref). |
| [`getweights`](@ref) | Zhang-optimal model-averaging weights → [`WeightResult`](@ref). |
| [`predictma`](@ref) | Weighted conditional prediction from an averaged model. |

## Status

`ConditionalAIC.jl` is under active development toward feature parity with
`cAIC4`. `PARITY.md` in the repository is the authoritative scope matrix; current
state:

| Capability | Milestone | Status |
|------------|-----------|--------|
| Gaussian LMM scoring (Greven–Kneib) | M2 | ✅ done, validated vs `cAIC4` |
| Comparison (`anocaic`) | M2.5 | ✅ done |
| GLMM scoring — Poisson (Chen–Stein), Bernoulli (Efron), bootstrap | M3 | ✅ done |
| Stepwise search (`stepcaic`, backward / forward / both; LMM + GLMM) | M4 | ✅ done |
| Model averaging (`modelavg` / `getweights` / `predictma`) | M4.5 | ✅ done |
| Additive models (`gamm4` analogue) | M5 | 🚫 deferred (no direct Julia analogue) |
| v1.0 / registration | M6 | ⬜ planned |

Every estimator is validated against `cAIC4` on two levels: Level 1 feeds
identical synthetic inputs into `cAIC4`'s internal functions and the
`ConditionalAIC.jl` kernels (tight tolerance); Level 2 fits the same model in both
stacks and compares the full cAIC within a fit-discrepancy-justified tolerance.
```
