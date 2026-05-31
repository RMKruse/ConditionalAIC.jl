# The model-averaging assembly (the `modelavg` method). Included directly into the `cAIC`
# module. Port of `cAIC4`'s `modelAvg` restricted to Gaussian `LinearMixedModel` candidates
# (docs/math/0009-model-averaging.md): score each candidate with `caic`, form the weights,
# and combine the candidates' coefficients into name-keyed model-averaged effects.
#
# This file touches only the public `MixedModels`/StatsAPI surface (`response`, `fixef`,
# `fixefnames`, `raneftables`) plus the in-package `MMInternals.reml` and `Numerics`; it
# does not reach into a `MixedModels` object's internal fields (mm_internals.jl unchanged).

const _WEIGHTTYPES = (:smoothed,)

"""
    modelavg(m1::LinearMixedModel{T}, rest::LinearMixedModel{T}...; weights=:smoothed,
             method=:auto, hessian=:analytic, nboot=nothing, sigmapenalty=1)
        -> ModelAvgResult{T}

cAIC-weighted model averaging over a set of Gaussian linear mixed models (port of `cAIC4`'s
`modelAvg`). Each candidate is scored with [`caic`](@ref) (in **input order**), the
per-candidate cAICs are turned into averaging weights, and the candidates' fixed and random
effects are combined into a name-keyed weighted sum returned in a [`ModelAvgResult`](@ref).

Candidates **may** differ in both fixed- and random-effects structure, but must share one
response `y`/observation count `n` and one REML setting (validated; `ArgumentError`
otherwise — the fail-loud strengthening of `cAIC4`'s unchecked `getME(m[[1]], "y")`).

# Arguments
- `m1, rest...`: one or more fitted Gaussian `LinearMixedModel` objects of the same float
  type.
- `weights`: the weight scheme. `:smoothed` (the default) is the Buckland (1997)
  exponential-cAIC smoothed weights `wᵢ = exp(−Δᵢ/2)/Σ exp(−Δ/2)`, `Δᵢ = cAICᵢ − min cAIC`,
  computed in log-space.
- `method, hessian, nboot, sigmapenalty`: forwarded unchanged to [`caic`](@ref) for every
  candidate, so all are scored consistently.

# Returns
- A [`ModelAvgResult`](@ref) carrying the name-keyed averaged `fixeff`/`raneff`, the
  `weights` and per-candidate `caics` (both in input order), the candidate `models`, and the
  `weighttype`.

# Throws
- `ArgumentError` if the candidates differ in response/observation count or REML setting, if
  any candidate is not a Gaussian `LinearMixedModel` of the shared float type, or for an
  unsupported `weights` scheme.

# Example
```jldoctest
julia> using MixedModels, cAIC

julia> data = MixedModels.dataset(:sleepstudy);

julia> m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), data; REML=false, progress=false);

julia> m2 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), data; REML=false, progress=false);

julia> res = modelavg(m1, m2; weights=:smoothed);

julia> sum(res.weights) ≈ 1.0
true
```
"""
function modelavg(
    m1::LinearMixedModel{T},
    rest::Vararg{LinearMixedModel{T}};
    weights::Symbol=:smoothed,
    method::Symbol=:auto,
    hessian::Symbol=:analytic,
    nboot::Union{Int,Nothing}=nothing,
    sigmapenalty::Integer=1,
) where {T}
    ms = (m1, rest...)
    _validate_candidates(ms)
    weights in _WEIGHTTYPES || throw(
        ArgumentError(
            "unknown weights=:$weights; supported: :smoothed (Buckland smoothed weights). " *
            "The Zhang-optimal path is a separate milestone step and is not yet implemented.",
        ),
    )

    # Per-candidate cAIC in INPUT order (the unsorted anocAIC analogue, docs/math/0009 §0).
    n = length(ms)
    caics = Vector{T}(undef, n)
    for (i, m) in enumerate(ms)
        caics[i] =
            caic(m; method=method, hessian=hessian, nboot=nboot, sigmapenalty=sigmapenalty).caic
    end

    w = _bucklandweights(caics)
    fixeff = _avgfixeff(ms, w)
    raneff = _avgraneff(ms, w)
    return ModelAvgResult{T}(
        fixeff, raneff, w, caics, collect(LinearMixedModel{T}, ms), weights
    )
end

# Fail-loud fallback: any other mix of MixedModel candidates (a GLMM, or differing float
# types) cannot be averaged. The homogeneous `LinearMixedModel{T}` method above is strictly
# more specific and wins when every candidate matches it, so this fires only on a genuine
# contract violation. `cAIC4` averaging is Gaussian-LMM only (docs/math/0009 §0/§1).
function modelavg(m1::MixedModel, rest::Vararg{MixedModel}; kwargs...)
    throw(
        ArgumentError(
            "modelavg requires every candidate to be a Gaussian LinearMixedModel sharing " *
            "one float type; got $(map(typeof, (m1, rest...))). Model averaging is not " *
            "defined for generalized (GLMM) candidates (docs/math/0009 §0).",
        ),
    )
end

# ── Candidate-set contract ───────────────────────────────────────────────────
function _validate_candidates(ms::Tuple{Vararg{LinearMixedModel}})
    m1 = ms[1]
    y1 = response(m1)
    n1 = length(y1)
    reml1 = MMInternals.reml(m1)
    for i in 2:length(ms)
        m = ms[i]
        yi = response(m)
        length(yi) == n1 || throw(
            ArgumentError(
                "candidate $i has $(length(yi)) observations but candidate 1 has $n1; " *
                "model averaging requires one common response.",
            ),
        )
        yi == y1 || throw(
            ArgumentError(
                "candidate $i has a different response vector than candidate 1; model " *
                "averaging requires one common response.",
            ),
        )
        MMInternals.reml(m) == reml1 || throw(
            ArgumentError(
                "candidate $i has REML=$(MMInternals.reml(m)) but candidate 1 has " *
                "REML=$reml1; all candidates must share the REML setting.",
            ),
        )
    end
    return nothing
end

# ── Buckland smoothed weights (docs/math/0009 §3), log-space ─────────────────
# wᵢ = exp(−Δᵢ/2)/Σ exp(−Δ/2) with Δᵢ = cAICᵢ − min cAIC. This is softmax(−cAIC/2): the
# min-subtraction is absorbed by the logsumexp normalisation (CLAUDE §9 — the vetted
# log-space entry point), exact-equivalent to the R `exp(-delta/2)/sum(exp(-delta/2))`.
function _bucklandweights(caics::AbstractVector{T}) where {T}
    x = caics ./ (-2)
    return exp.(x .- Numerics.logsumexp(x))
end

# ── Name-keyed model-averaged effects (docs/math/0009 §4) ────────────────────
# Fixed effects keyed on coefficient name over the union across candidates; a candidate
# lacking a term contributes 0; reported name-sorted (the `tapply(..., sum)` analogue).
function _avgfixeff(ms, w::AbstractVector{T}) where {T}
    acc = Dict{String,T}()
    for (i, m) in enumerate(ms)
        names = fixefnames(m)
        vals = fixef(m)
        for (nm, v) in zip(names, vals)
            acc[nm] = get(acc, nm, zero(T)) + w[i] * v
        end
    end
    ks = sort!(collect(keys(acc)))
    vs = T[acc[k] for k in ks]
    return NamedEffects{String,T}(ks, vs)
end

# Random effects keyed on (grouping factor, level, RE term) over the union across
# candidates. `raneftables(m)` is a NamedTuple keyed by grouping symbol; each table's first
# column holds the levels and the remaining columns the per-term conditional modes.
function _avgraneff(ms, w::AbstractVector{T}) where {T}
    acc = Dict{Tuple{String,String,String},T}()
    for (i, m) in enumerate(ms)
        rt = raneftables(m)
        for g in keys(rt)
            tbl = rt[g]
            cols = propertynames(tbl)
            levels = getproperty(tbl, cols[1])
            for c in cols[2:end]
                term = String(c)
                colvals = getproperty(tbl, c)
                for (lev, v) in zip(levels, colvals)
                    key = (String(g), string(lev), term)
                    acc[key] = get(acc, key, zero(T)) + w[i] * v
                end
            end
        end
    end
    ks = sort!(collect(keys(acc)))
    vs = T[acc[k] for k in ks]
    return NamedEffects{Tuple{String,String,String},T}(ks, vs)
end
