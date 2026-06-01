```@meta
CurrentModule = ConditionalAIC
```

# Usage guide

The public surface is six functions organised around four verbs — **score** a
model, **compare** a set of models, **search** a random-effects structure, and
**average** models into one prediction.

## Scoring — [`caic`](@ref)

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

Key keywords: `method` (`:auto`, `:steinian`, `:bootstrap`), `hessian` (the
B-source for the steinian route — `:analytic`, `:finitediff`, `:forwarddiff`),
`nboot`, `sigmapenalty`, and `rng`.

## Comparison — [`anocaic`](@ref)

Score several fitted models with **identical** keyword arguments and rank them
ascending by cAIC (best first).

```julia
m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), data; REML=false, progress=false)
m2 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)),        data; REML=false, progress=false)

t = anocaic(m1, m2)   # ranked ascending by cAIC
```

## Search — [`stepcaic`](@ref)

Greedy stepwise selection of the **random-effects** structure (the fixed-effects
part is held constant, matching `cAIC4`'s `(g)lmer` path). A backward search can
bottom out at a fixed-effects-only `lm`/`glm` terminal, scored directly.

```julia
m   = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), sleepstudy; progress=false)
res = stepcaic(m, sleepstudy; direction=:forward, slopecandidates=[:days])
res.selected.caic
res.path          # the per-step search record
```

## Averaging — [`modelavg`](@ref)

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

The optimal weights themselves are available without forming a prediction via
[`getweights`](@ref).
```
