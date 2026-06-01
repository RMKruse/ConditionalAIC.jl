"""
    ConditionalAIC.GLMInternals

**Quarantine module — the single, auditable touchpoint for `GLM.jl` internals.**

The backward `stepcaic` search's fixed-effects-only **terminal node** is fit and scored as a
plain `GLM.jl` `lm`/`glm` (ADR-0006): `MixedModels.jl` v5.5.1 cannot represent or fit a no-RE
model. That terminal scoring (`src/scoring.jl`) runs on `GLM.jl`'s **public** surface —
`lm`/`glm`, `response`, `predict`, `deviance`, `coef`, the `LinearModel`/`GeneralizedLinearModel`
types — with **two** exceptions that reach into a fitted `glm`'s internal `GlmResp`: the response
family and the prior weights. Those two field accesses live here and nowhere else, mirroring the
[`MMInternals`](@ref ConditionalAIC.MMInternals) quarantine for `MixedModels.jl`. This module
performs **no** translation or abstraction; it reaches into the fitted-model object directly and
exists solely to make every `GLM` internal touchpoint auditable in one place. Each accessor
shape-asserts what it extracts so a silent upstream change surfaces as a clear error rather than a
wrong number downstream.

# Internal-access table

Pinned against **`GLM = "=1.9.5"`**. On a version bump, walking this table is the required
checklist; accessing a `GLM` internal not listed here is forbidden — add the row first.

| Touchpoint       | Kind  | Used by             | Extracted quantity                                                        |
|:-----------------|:------|:--------------------|:--------------------------------------------------------------------------|
| `m.model.rr.d`   | field | [`glmfamily`]       | response distribution family D of the fitted `glm`; drives terminal-scoring dispatch |
| `m.model.rr.wts` | field | [`glmpriorweights`] | per-observation prior weights — the binomial trial counts nᵢ; consumed only by the Binomial terminal kernel |

`m.model` is the `GeneralizedLinearModel` wrapped by the `glm` formula-fit `TableRegressionModel`;
`m.model.rr` is its `GLM.GlmResp{V,D,L}` response object, whose pinned field layout is recorded in
`_GLMRESP_FIELDS` for the version-bump checklist. Both accessors reach `GlmResp` **by name** (`.d`,
`.wts`), so a removed/renamed field fails loud at access, and each accessor `_drift`-asserts the
extracted *type* so a meaning change surfaces too. The Gaussian `lm` (`LinearModel`) terminal
touches **no** `GLM` internals — its scoring is entirely on the public
`response`/`predict`/`deviance`/`coef` surface.
"""
module GLMInternals

using GLM: GLM, GeneralizedLinearModel

# `TableRegressionModel` (the `lm`/`glm` formula-fit wrapper) lives in StatsModels, reached through
# `GLM`'s loaded copy — the same alias `ConditionalAIC` uses for terminal-scoring dispatch.
# `Distribution` is reached through `GLM`'s loaded `Distributions` for the family shape-assert.
const TableRegressionModel = GLM.StatsModels.TableRegressionModel
const Distribution = GLM.Distributions.Distribution

const PINNED_VERSION = "1.9.5"

# Raised when an internal touchpoint yields a value of an unexpected type/shape — i.e. `GLM` has
# drifted from the pinned version. Failing loud here turns a silent upstream change into a clear
# error instead of a wrong number downstream (mirrors `MMInternals._drift`).
@noinline function _drift(touchpoint::AbstractString, expected, got)
    return error(
        "GLM internal `$touchpoint` produced $(typeof(got)); expected $expected. \
         This indicates drift from the pinned GLM v$PINNED_VERSION — reconcile the \
         internal-access table in `GLMInternals` against the new version before use."
    )
end

# The exact field-name layout of `GLM.GlmResp` (pinned GLM v1.9.5). The accessors below reach it by
# name (`.d`, `.wts`), so a removed/renamed field already fails loud; this constant records the full
# pinned layout for the version-bump checklist.
const _GLMRESP_FIELDS = (
    :y, :d, :link, :devresid, :eta, :mu, :offset, :wts, :wrkwt, :wrkresid
)

"""
    glmfamily(m::TableRegressionModel{<:GeneralizedLinearModel}) -> Distribution

The response distribution family D of a fitted `glm` (`m.model.rr.d`) — e.g. `Poisson()`,
`Bernoulli()`, `Binomial()`. `m.model` is the wrapped `GeneralizedLinearModel`, `m.model.rr` its
`GlmResp`. Returned as the family **instance** (not the type) so the terminal scorer can dispatch on
it through a function barrier; an unsupported family is returned unchanged for the caller to reject
with a clear `ArgumentError`.
"""
function glmfamily(m::TableRegressionModel{<:GeneralizedLinearModel})
    d = m.model.rr.d
    d isa Distribution || _drift("m.model.rr.d", "a Distributions.Distribution", d)
    return d
end

"""
    glmpriorweights(m::TableRegressionModel{<:GeneralizedLinearModel}) -> AbstractVector

The per-observation prior weights of a fitted `glm` (`m.model.rr.wts`); for a multi-trial Binomial
fit these are the trial counts nᵢ. Consumed only by the Binomial terminal kernel
(`condloglik_binomial`) — ignored by the Poisson/Bernoulli paths, which carry an empty/uniform
weight vector.
"""
function glmpriorweights(m::TableRegressionModel{<:GeneralizedLinearModel})
    w = m.model.rr.wts
    w isa AbstractVector{<:Real} || _drift("m.model.rr.wts", "an AbstractVector{<:Real}", w)
    return w
end

end # module GLMInternals
