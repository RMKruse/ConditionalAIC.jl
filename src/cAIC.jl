"""
    cAIC

Conditional Akaike Information Criterion and conditional model selection for
mixed-effects models fitted with [`MixedModels.jl`](https://github.com/JuliaStats/MixedModels.jl)
— a re-platforming of R's `cAIC4` onto `LinearMixedModel` / `GeneralizedLinearMixedModel`.

[`caic`](@ref) scores a fitted Gaussian `LinearMixedModel` by its conditional AIC (M2): the
analytic Greven–Kneib bias correction assembled end-to-end into a [`CAICResult`](@ref).
[`anocaic`](@ref) ranks a user-supplied set of models by cAIC (M2.5), returning an
[`AnocaicTable`](@ref). The search verb (`stepcaic`, M4) remains a stub. All access to
`MixedModels.jl` internals is quarantined in the [`cAIC.MMInternals`](@ref) submodule.
"""
module cAIC

using MixedModels:
    MixedModel, LinearMixedModel, GeneralizedLinearMixedModel, Poisson, Bernoulli
using Random: AbstractRNG, default_rng, randn

include("numerics.jl")
include("loglik.jl")
include("dof_lmm.jl")        # defines DofLMM.GaussianComponents (used by Components)
include("mm_internals.jl")   # QUARANTINE: the only MixedModels-internals touchpoint
include("dof_glmm.jl")       # GLMM df routes: Poisson Chen-Stein + Bernoulli Efron (M3)
include("components.jl")     # fit-extracted arrays → GaussianComponents (uses ..DofLMM)
include("types.jl")          # CAICResult, AnocaicTable (public result types)
include("scoring.jl")        # the caic methods (the scoring assembly)
include("comparison.jl")     # the anocaic method (comparison table, M2.5)

# ── Public surface ──────────────────────────────────────────────────────────
# `caic` (M2 scoring) is implemented in `scoring.jl`. `anocaic` (M2.5 comparison) is
# implemented in `comparison.jl`. `stepcaic` (M4 search) remains a zero-method stub.
# See CONTEXT.md for the Scoring / Comparison / Search vocabulary.

"""
    anocaic(m::LinearMixedModel, rest::LinearMixedModel...; kwargs...) -> AnocaicTable

Rank a user-supplied set of fitted models by conditional AIC (port of `cAIC4`'s
`anocAIC`). See [`comparison.jl`] for the full signature and examples.
"""
function anocaic end

"""
    stepcaic(model)

Conditional stepwise selection: search a candidate space of random-/fixed-effects
terms guided by the conditional AIC.

Walking-skeleton stub — carries no methods yet (M4).
"""
function stepcaic end

export caic, anocaic, stepcaic, CAICResult, AnocaicTable

end # module cAIC
