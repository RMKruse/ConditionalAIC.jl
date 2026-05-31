# Result type for the scoring path. Included directly into the `cAIC` module (not a
# submodule): `CAICResult` is part of the public surface and is returned by `caic`.
# `RegressionModel` is imported at the module top.

"""
    CAICResult{T<:AbstractFloat,M<:RegressionModel}

The result of scoring a fitted mixed-effects model by its conditional AIC
(`cAIC = −2 ℓ_cond + 2 ρ`). Returned by [`caic`](@ref).

The fields are the public, typed accessors:

- `caic::T` — the conditional AIC value.
- `dof::T` — the bias-corrected effective degrees of freedom ρ (the penalty term).
- `condloglik::T` — the conditional log-likelihood ℓ(y | b̂, β̂, θ̂) (the `cAIC4`
  `getcondLL` quantity).
- `reducedmodel::Union{Nothing,M}` — the reduced model a singular fit was scored on, or
  `nothing` when the fit was non-singular and scored as given (always `nothing` for the
  never-singular `lm`/`glm` terminal).
- `refit::Bool` — whether scoring was performed on a refitted reduced model.
- `method::Symbol` — the degrees-of-freedom **method** actually used (provenance), e.g.
  `:steinian`.
- `bsource::Symbol` — the Hessian **B-source** actually used (provenance), e.g.
  `:analytic`.

`method`/`bsource` record what was *resolved and run* (e.g. `method = :auto` resolves to
`:steinian` for the Gaussian family), so candidates can be checked for consistent scoring.

The model bound is `M <: RegressionModel` (the common supertype of `LinearMixedModel`,
`GeneralizedLinearMixedModel`, and the `GLM.jl` `lm`/`glm` terminal a backward `stepcaic`
search reaches when the last random-effects term is dropped). Every candidate — mixed or
terminal — therefore scores into this one result type.
"""
struct CAICResult{T<:AbstractFloat,M<:RegressionModel}
    caic::T
    dof::T
    condloglik::T
    reducedmodel::Union{Nothing,M}
    refit::Bool
    method::Symbol
    bsource::Symbol
end

"""
    AnocaicTable{T<:AbstractFloat,M<:RegressionModel}

The result of comparing a user-supplied set of fitted models by conditional AIC
(port of `cAIC4`'s `anocAIC`). Returned by [`anocaic`](@ref).

Models are sorted ascending by `cAIC` — entry `1` is the best-fitting model (lowest cAIC).

- `results::Vector{CAICResult{T,M}}` — per-model scoring results, sorted ascending by
  `cAIC` (best first). Each [`CAICResult`](@ref) carries the `cAIC`, effective degrees of
  freedom `ρ` (`dof`), conditional log-likelihood, and scoring provenance.
- `inputorder::Vector{Int}` — original 1-based position of each result in the argument
  list passed to `anocaic`. `inputorder[k]` is the input-list index of the k-th ranked
  model.

The Δcaic for rank `k` is `results[k].caic - results[1].caic` (≥ 0 for all `k`).
"""
struct AnocaicTable{T<:AbstractFloat,M<:RegressionModel}
    results::Vector{CAICResult{T,M}}
    inputorder::Vector{Int}
end

function Base.show(io::IO, ::MIME"text/plain", t::AnocaicTable)
    n = length(t.results)
    best = t.results[1].caic
    noun = n == 1 ? "model" : "models"
    println(io, "Conditional AIC comparison (anocaic): $n $noun")
    println(io, " Rank  Input        cAIC           ρ      condloglik       Δcaic")
    for k in 1:n
        r = t.results[k]
        Δ = r.caic - best
        println(
            io,
            "  ",
            lpad(k, 3),
            "    ",
            lpad(t.inputorder[k], 3),
            "  ",
            lpad(round(r.caic; digits=4), 12),
            "  ",
            lpad(round(r.dof; digits=4), 8),
            "  ",
            lpad(round(r.condloglik; digits=4), 12),
            "  ",
            lpad(round(Δ; digits=4), 10),
        )
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", r::CAICResult)
    println(io, "Conditional AIC (cAIC)")
    println(io, "  cAIC               = ", r.caic)
    println(io, "  effective df (ρ)   = ", r.dof)
    println(io, "  conditional logLik = ", r.condloglik)
    print(io, "  method = :", r.method, "   B-source = :", r.bsource)
    r.refit && print(io, "   (scored on reduced model)")
    return nothing
end
