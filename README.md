# ConditionalAIC.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://RMKruse.github.io/ConditionalAIC.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://RMKruse.github.io/ConditionalAIC.jl/dev)

Conditional Akaike Information Criterion and conditional model selection for
mixed-effects models fitted with [`MixedModels.jl`](https://github.com/JuliaStats/MixedModels.jl).

`ConditionalAIC.jl` is a Julia re-platforming of R's
[`cAIC4`](https://cran.r-project.org/package=cAIC4). Where `cAIC4` sits on
`lme4`/`merMod`, `ConditionalAIC.jl` sits on `MixedModels.jl` — its
`LinearMixedModel` and `GeneralizedLinearMixedModel`. The work is to
reconstruct `cAIC4`'s bias-correction and degrees-of-freedom logic on top of the
fitted objects `MixedModels.jl` produces, validated against `cAIC4` as ground
truth.

## What it computes

The conditional AIC has the schematic form

```
cAIC = -2 · ℓ_cond(y | b̂, β̂, θ̂) + 2 · ρ
```

where `ℓ_cond` is the *conditional* log-likelihood given the predicted random
effects `b̂`, and `ρ` is the bias-corrected effective degrees of freedom. The
naive plug-in df `ρ₀ = tr(H₁)` understates `ρ` because the variance parameters
`θ` are estimated; the correction for that estimation uncertainty is the
mathematical core of the package.

**References** (the project's formulas match these):

- Greven, S. & Kneib, T. (2010). On the behaviour of marginal and conditional
  AIC in linear mixed models. *Biometrika*.
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

Requires Julia ≥ 1.10 (LTS). `MixedModels` and `GLM` are pinned to exact
versions (`MixedModels = "=5.5.1"`, `GLM = "=1.9.5"`) because the package depends
on `MixedModels.jl` internals by design — a version bump is a deliberate,
reviewed event.

## Public API

| Function | Purpose |
|----------|---------|
| `caic(m; …)` | **Score** a single fitted model by its conditional AIC → `CAICResult`. |
| `anocaic(ms…; …)` | **Compare** a user-supplied set of fitted models, ranked by cAIC → `AnocaicTable`. |
| `stepcaic(m, data; …)` | **Search** the random-effects structure (backward / forward / both) → `StepcaicResult`. |
| `modelavg(ms…; …)` | **Average** several models into one weighted combination → `ModelAvgResult`. |
| `getweights(res)` | Zhang-optimal model-averaging weights → `WeightResult`. |
| `predictma(res, newdata; …)` | Weighted conditional prediction from an averaged model. |

### Scoring — `caic`

A Gaussian `LinearMixedModel` is scored via the analytic Greven–Kneib bias
correction; a `GeneralizedLinearMixedModel` via its family-specific df route.

```julia
using MixedModels, ConditionalAIC

m = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy); REML=false, progress=false)

r = caic(m)
r.caic          # the conditional AIC
r.dof           # ρ, the effective degrees of freedom
r.condloglik    # ℓ_cond
```

Key keywords: `method` (`:auto`, `:steinian`, `:bootstrap`), `hessian`
(the B-source for the steinian route — `:analytic`, `:finitediff`,
`:forwarddiff`), `nboot`, `sigmapenalty`, and `rng`.

### Comparison — `anocaic`

```julia
m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), data; REML=false, progress=false)
m2 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)),       data; REML=false, progress=false)

t = anocaic(m1, m2)   # ranked ascending by cAIC
```

### Search — `stepcaic`

Greedy stepwise selection of the **random-effects** structure (the fixed-effects
part is held constant, matching `cAIC4`'s `(g)lmer` path). A backward search can
bottom out at a fixed-effects-only `lm`/`glm` terminal, scored directly.

```julia
m   = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), sleepstudy; progress=false)
res = stepcaic(m, sleepstudy; direction=:forward, slopecandidates=[:days])
res.selected.caic
res.path          # the per-step search record
```

### Averaging — `modelavg`

cAIC-weighted combination of candidate models (Gaussian LMM only). Two weight
schemes: **Zhang-optimal** (`weights=:zhang`, the default) and **Buckland
smoothed** (`weights=:smoothed`).

```julia
res = modelavg(m1, m2)            # Zhang-optimal weights by default
sum(res.weights)                  # ≈ 1.0
res.fixeff                        # model-averaged fixed effects (name-keyed)

yhat = predictma(res, data)       # weighted conditional prediction

display(res)                      # the full report (the summaryMA port)
```

## Status

`ConditionalAIC.jl` is under active development toward feature parity with `cAIC4`.
[`PARITY.md`](PARITY.md) is the authoritative scope matrix; current state:

| Capability | Milestone | Status |
|------------|-----------|--------|
| Gaussian LMM scoring (Greven–Kneib) | M2 | ✅ done, validated vs `cAIC4` |
| Comparison (`anocaic`) | M2.5 | ✅ done |
| GLMM scoring — Poisson (Chen–Stein), Bernoulli (Efron), bootstrap | M3 | ✅ done |
| Stepwise search (`stepcaic`, backward / forward / both; LMM + GLMM) | M4 | ✅ done |
| Model averaging (`modelavg` / `getweights` / `predictma`) | M4.5 | ✅ done |
| Additive models (`gamm4` analogue) | M5 | 🚫 deferred (no direct Julia analogue) |
| v1.0 / registration | M6 | ⬜ planned |

Every estimator is validated against `cAIC4` on **two levels**: Level 1 feeds
identical synthetic inputs into `cAIC4`'s internal functions and the `ConditionalAIC.jl`
kernels (tight tolerance); Level 2 fits the same model in both stacks and
compares the full cAIC within a fit-discrepancy-justified tolerance. Divergences
from R are recorded in [`DECISIONS.md`](DECISIONS.md); the mathematical
specification for each estimator lives in [`docs/math/`](docs/math/) and
architectural choices in [`docs/adr/`](docs/adr/).

## Architecture

All access to `MixedModels.jl` internals is quarantined in a single file,
`src/mm_internals.jl`, which carries an auditable table of every internal field
and function it touches against the pinned `MixedModels` version. No other source
file reaches into a `MixedModels` object's internals.

```
src/
  ConditionalAIC.jl          # module entry point + public exports
  mm_internals.jl  # the only MixedModels-internals touchpoint
  numerics.jl      # numerically-stable primitives
  loglik.jl        # conditional log-likelihood
  dof_lmm.jl       # Gaussian Greven–Kneib df correction
  dof_glmm.jl      # GLMM df routes (Chen–Stein, Efron, bootstrap)
  components.jl    # fit → components bridge
  types.jl         # CAICResult, AnocaicTable, StepcaicResult, ModelAvgResult, …
  scoring.jl       # the caic methods (scoring assembly)
  comparison.jl    # anocaic (comparison)
  respec.jl        # RE-structure spec (the cnms analogue)
  stepcaic.jl      # backward/forward/both candidate enumeration + controller
  averaging.jl     # modelavg / getweights / predictma
```

Computations stay in the numerically stable formulation throughout: log-space
likelihoods, Cholesky-based solves, no explicit inverses, `logdet` over
triangular factors.

## Testing

The suite is built with [`TestItems.jl`](https://github.com/julia-vscode/TestItems.jl)
and run via `Pkg.test()`. CI runs the matrix `{1.10, 1.11, nightly}` and enforces
three gates: `JuliaFormatter` (Blue style), `JET.jl` static analysis, and the full
test suite (including `@inferred` type-stability checks and `Aqua.jl`).

R reference values are committed as HDF5 fixtures under `test/fixtures/`,
regenerated by the `test/generate_fixtures_*.{R,jl}` scripts that drive `cAIC4`.
CI runs against fixtures only (no R required); live `RCall.jl` re-validation runs
in a separate, gated job. `RCall` is a test-only dependency, never a runtime one.
