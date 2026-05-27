"""
    cAIC

Conditional Akaike Information Criterion and conditional model selection for
mixed-effects models fitted with [`MixedModels.jl`](https://github.com/JuliaStats/MixedModels.jl)
— a re-platforming of R's `cAIC4` onto `LinearMixedModel` / `GeneralizedLinearMixedModel`.

This is the **walking-skeleton** state (milestone M1): the public verbs are declared
as stubs and carry no methods yet. All access to `MixedModels.jl` internals is
quarantined in the [`cAIC.MMInternals`](@ref) submodule.
"""
module cAIC

include("mm_internals.jl")

# ── Public surface ──────────────────────────────────────────────────────────
# Declared here as zero-method stubs so `using cAIC` exposes a stable export
# surface before the estimators land. Methods are added in their own milestones:
# `caic` (M2 scoring), `anocaic` (M2.5 comparison), `stepcaic` (M4 search).
# See CONTEXT.md for the Scoring / Comparison / Search vocabulary.

"""
    caic(model)

Score a single fitted mixed-effects model by its conditional AIC.

Walking-skeleton stub — carries no methods yet (M2).
"""
function caic end

"""
    anocaic(models...)

Rank a user-supplied set of fitted models by conditional AIC (port of `cAIC4`'s
`anocAIC`).

Walking-skeleton stub — carries no methods yet (M2.5).
"""
function anocaic end

"""
    stepcaic(model)

Conditional stepwise selection: search a candidate space of random-/fixed-effects
terms guided by the conditional AIC.

Walking-skeleton stub — carries no methods yet (M4).
"""
function stepcaic end

export caic, anocaic, stepcaic

end # module cAIC
