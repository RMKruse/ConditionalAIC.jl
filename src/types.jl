# Result type for the scoring path. Included directly into the `ConditionalAIC` module (not a
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
  conditional log-likelihood quantity).
- `reducedmodel::Union{Nothing,M}` — the reduced model a singular fit was scored on, or
  `nothing` when the fit was non-singular and scored as given (always `nothing` for the
  never-singular `lm`/`glm` terminal).
- `refit::Bool` — whether scoring was performed on a refitted reduced model.
- `method::Symbol` — the degrees-of-freedom **method** actually used (provenance), one of
  the closed set `:steinian` | `:bootstrap` | `:terminal`. `:steinian` is the analytic
  covariance-penalty correction (Greven–Kneib for the LMM; the family-dispatched Stein-type
  df — Poisson Chen–Stein / Bernoulli Efron — for the GLMM); `:bootstrap` the
  parametric/conditional bootstrap penalty; `:terminal` the deterministic `(g)lm` terminal.
- `bsource::Symbol` — the Hessian **B-source** actually used (provenance), one of
  `:analytic` | `:forwarddiff` | `:finitediff` | `:na` (`:na` whenever no Hessian B is
  formed — every `:bootstrap` and `:terminal` result, and the GLMM paths).

`method`/`bsource` record what was *resolved and run* — the request-level `method = :auto`
never appears, resolving to `:steinian` on both the Gaussian and the analytic GLMM paths — so
candidates can be checked for consistent scoring. These two closed Symbol domains are the
stable provenance surface; only values from the sets above are ever stored.

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

"""
    NamedEffects{K,T<:AbstractFloat}

A name-keyed vector of effect values: `keys::Vector{K}` (name-sorted) paired with
`values::Vector{T}`. The storage mirrors the StatsAPI ecosystem convention of parallel
name/value vectors (`coef`/`coefnames`, `fixef`/`fixefnames`), so the ordering the report
wants is baked into the storage; keyed lookup is provided via `getindex` for ergonomics.

`K` is `String` for the model-averaged fixed effects (the coefficient name) and
`Tuple{String,String,String}` — `(grouping factor, level, RE term)` — for the random
effects. Used by [`ModelAvgResult`](@ref).
"""
struct NamedEffects{K,T<:AbstractFloat}
    keys::Vector{K}
    values::Vector{T}
end

Base.length(e::NamedEffects) = length(e.keys)
Base.keys(e::NamedEffects) = e.keys
Base.values(e::NamedEffects) = e.values
Base.haskey(e::NamedEffects, k) = any(isequal(k), e.keys)
Base.pairs(e::NamedEffects) = (k => v for (k, v) in zip(e.keys, e.values))

function Base.getindex(e::NamedEffects{K}, k) where {K}
    i = findfirst(isequal(k), e.keys)
    i === nothing && throw(KeyError(k))
    return e.values[i]
end

"""
    WeightResult{T<:AbstractFloat}

The result of Zhang-optimal weight optimization over a set of Gaussian
`LinearMixedModel{T}` candidates (port of `cAIC4`'s weight optimizer).
Returned by [`getweights`](@ref).

The Zhang (2014) Mallows criterion J(w) = (y − μw)ᵀ(y − μw) + 2σ̂²(ρᵀw) is minimised
over the unit simplex 𝒲 = {w ≥ 0, Σwᵢ = 1} via the transcribed `solnp` augmented-
Lagrangian SQP of `cAIC4`'s weight optimizer.

- `weights::Vector{T}` — the optimal model-averaging weights; non-negative and summing to 1.
- `objective::T` — the minimised Mallows criterion J(ŵ) (the objective value of the weight optimizer).
- `duration::Float64` — wall-clock seconds for the optimization (excluded from reproducibility
  assertions).
"""
struct WeightResult{T<:AbstractFloat}
    weights::Vector{T}
    objective::T
    duration::Float64
end

# The `caic` scoring options a `modelavg` call used, captured so the Zhang slow path of
# `getweights` re-scores ρ with the *same* options the candidates were scored under (rather
# than `caic`'s defaults). Concrete field types — `nboot`'s small `Union{Int,Nothing}` mirrors
# `caic`'s own signature; `sigmapenalty` is narrowed to `Int` on construction. `rng` is not
# captured: `modelavg` exposes no `rng` kwarg, so both paths use `default_rng()`.
const CAICKwargs = NamedTuple{
    (:method, :hessian, :nboot, :sigmapenalty),Tuple{Symbol,Symbol,Union{Int,Nothing},Int}
}

"""
    ModelAvgResult{T<:AbstractFloat}

The result of cAIC-weighted model averaging over a set of Gaussian `LinearMixedModel{T}`
candidates (port of `cAIC4`'s `modelAvg`). Returned by [`modelavg`](@ref).

- `fixeff::NamedEffects{String,T}` — the model-averaged fixed effects, keyed on coefficient
  name over the union of candidate coefficients (a name absent from a candidate contributes
  0), name-sorted.
- `raneff::NamedEffects{Tuple{String,String,String},T}` — the model-averaged random effects,
  keyed on `(grouping factor, level, RE term)` over the union across candidates, sorted.
- `weights::Vector{T}` — the per-candidate averaging weights, in **input order**; non-negative
  and summing to 1.
- `caics::Vector{T}` — the per-candidate conditional AIC, in **input order** (the unsorted
  `anocAIC` analogue).
- `models::Vector{LinearMixedModel{T}}` — the candidate models, in input order.
- `weighttype::Symbol` — the weight scheme used: `:zhang` (Zhang-optimal, the default) or
  `:smoothed` (Buckland 1997 exponential-cAIC weights).
- `weightresult::Union{Nothing,WeightResult{T}}` — the full [`WeightResult`](@ref) from the
  Zhang-optimal optimizer (weights, objective `J(ŵ)`, duration) when `weighttype == :zhang`;
  `nothing` on the `:smoothed` path.
- `caickwargs::CAICKwargs` — the `caic` scoring options (`method`, `hessian`, `nboot`,
  `sigmapenalty`) this `modelavg` call used. Captured so the Zhang slow path of
  [`getweights`](@ref) re-scores ρ under the *same* options the candidates were scored with,
  not `caic`'s defaults.

The result is **not** itself a fitted model — it is a pair of name-keyed averaged-coefficient
vectors plus the weight provenance.

Displaying a `ModelAvgResult` (`show`/REPL auto-display) prints the full averaging report —
the candidate models, a per-candidate cAIC + weight table, and the model-averaged fixed and
random effects — the port of `cAIC4`'s `summaryMA` (there is no standalone summary function).
"""
struct ModelAvgResult{T<:AbstractFloat}
    fixeff::NamedEffects{String,T}
    raneff::NamedEffects{Tuple{String,String,String},T}
    weights::Vector{T}
    caics::Vector{T}
    models::Vector{LinearMixedModel{T}}
    weighttype::Symbol
    weightresult::Union{Nothing,WeightResult{T}}
    caickwargs::CAICKwargs
end

# The full model-averaging report (port of `cAIC4`'s `summaryMA`, folded into the result's
# REPL display — there is no standalone `summaryma` function; PARITY.md M4.5, DECISIONS
# 2026-05-31). Sections: a header, the candidate models in input order, a per-candidate
# cAIC + weight table, the model-averaged fixed effects, and the model-averaged random
# effects. `formula` is the public StatsAPI accessor imported at the module top.
function Base.show(io::IO, ::MIME"text/plain", r::ModelAvgResult)
    n = length(r.models)
    noun = n == 1 ? "candidate" : "candidates"
    scheme = if r.weighttype == :smoothed
        "Buckland smoothed"
    elseif r.weighttype == :zhang
        "Zhang optimal"
    else
        string(r.weighttype)
    end
    println(io, "Model-averaged mixed model (modelavg): $n $noun, $scheme weights")

    # Candidate models in input order. The formulas stand in for `summaryMA`'s `z$call`,
    # which `ModelAvgResult` does not retain (DECISIONS 2026-05-31).
    println(io, "Candidate models:")
    for (i, m) in enumerate(r.models)
        println(io, "  ", lpad(i, 3), ": ", string(formula(m)))
    end

    # Per-candidate cAIC + weight table. The cAIC column extends `summaryMA`'s weights-only
    # listing; weights are `round(·; digits=6)` to match `summaryMA`.
    println(io, " Cand          cAIC       weight")
    for i in 1:n
        println(
            io,
            "  ",
            lpad(i, 3),
            "  ",
            lpad(round(r.caics[i]; digits=4), 12),
            "  ",
            lpad(round(r.weights[i]; digits=6), 10),
        )
    end

    println(io, "Model Averaged Fixed Effects:")
    for (k, v) in zip(r.fixeff.keys, r.fixeff.values)
        println(io, "  ", rpad(k, 16), round(v; digits=6))
    end

    # Heading corrected from `summaryMA`'s copy-pasted "...Fixed Effects" label (an upstream
    # bug, not transcribed; ADR-0007 decision 3). Keyed on (grouping, level, term).
    println(io, "Model Averaged Random Effects:")
    for (k, v) in zip(r.raneff.keys, r.raneff.values)
        g, lev, term = k
        println(io, "  ", rpad("$g[$lev] $term", 28), round(v; digits=6))
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
