"""
    cAIC

Conditional Akaike Information Criterion and conditional model selection for
mixed-effects models fitted with [`MixedModels.jl`](https://github.com/JuliaStats/MixedModels.jl)
— a re-platforming of R's `cAIC4` onto `LinearMixedModel` / `GeneralizedLinearMixedModel`.

[`caic`](@ref) scores a fitted Gaussian `LinearMixedModel` by its conditional AIC (M2): the
analytic Greven–Kneib bias correction assembled end-to-end into a [`CAICResult`](@ref). The
comparison (`anocaic`, M2.5) and search (`stepcaic`, M4) verbs remain stubs. All access to
`MixedModels.jl` internals is quarantined in the [`cAIC.MMInternals`](@ref) submodule.
"""
module cAIC

using MixedModels: MixedModel, LinearMixedModel, GeneralizedLinearMixedModel

include("numerics.jl")
include("loglik.jl")
include("dof_lmm.jl")        # defines DofLMM.GaussianComponents (used by Components)
include("mm_internals.jl")   # QUARANTINE: the only MixedModels-internals touchpoint
include("components.jl")     # fit-extracted arrays → GaussianComponents (uses ..DofLMM)
include("types.jl")          # CAICResult (public, returned by caic)
include("scoring.jl")        # the caic methods (the scoring assembly)

# ── Public surface ──────────────────────────────────────────────────────────
# `caic` (M2 scoring) is implemented in `scoring.jl`. `anocaic` (M2.5 comparison) and
# `stepcaic` (M4 search) remain zero-method stubs so `using cAIC` exposes a stable export
# surface before those estimators land. See CONTEXT.md for the Scoring / Comparison /
# Search vocabulary.

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

export caic, anocaic, stepcaic, CAICResult

end # module cAIC
