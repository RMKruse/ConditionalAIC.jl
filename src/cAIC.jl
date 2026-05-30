"""
    cAIC

Conditional Akaike Information Criterion and conditional model selection for
mixed-effects models fitted with [`MixedModels.jl`](https://github.com/JuliaStats/MixedModels.jl)
— a re-platforming of R's `cAIC4` onto `LinearMixedModel` / `GeneralizedLinearMixedModel`.

[`caic`](@ref) scores a fitted model by its conditional AIC: a Gaussian `LinearMixedModel`
via the analytic Greven–Kneib bias correction (M2), a `GeneralizedLinearMixedModel`
via Poisson Chen–Stein / Bernoulli Efron Steinian / conditional-bootstrap df (M3), or — as the
terminal node a backward `stepcaic` search reaches when the last random-effects term is dropped —
a plain `GLM.jl` `lm`/`glm` fit scored directly (`df = rank + 1`, M4 / ADR-0006). All assemble into
a [`CAICResult`](@ref). [`anocaic`](@ref) ranks a user-supplied set of models by cAIC (M2.5),
returning an [`AnocaicTable`](@ref). The search verb (`stepcaic`, M4) remains a stub. All access to
`MixedModels.jl` internals is quarantined in the [`cAIC.MMInternals`](@ref) submodule.
"""
module cAIC

using MixedModels:
    MixedModel, LinearMixedModel, GeneralizedLinearMixedModel, Poisson, Bernoulli, Binomial
using GLM:
    GLM,
    RegressionModel,
    LinearModel,
    GeneralizedLinearModel,
    coef,
    response,
    predict,
    deviance
# `TableRegressionModel` (the `lm`/`glm` formula-fit wrapper) is not exported by GLM; it lives
# in StatsModels, reachable through GLM's loaded copy. Aliased here so the terminal-scoring
# dispatch (`src/scoring.jl`) can name the type without taking a separate StatsModels dependency.
const TableRegressionModel = GLM.StatsModels.TableRegressionModel
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
include("respec.jl")         # RESpec extract/render — the M4 RE-structure representation
include("stepcaic.jl")       # backward/forward candidate enumeration (M4 stepwise search)

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
