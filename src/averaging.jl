# The model-averaging assembly (`modelavg` + `getweights`). Included directly into the
# `ConditionalAIC` module. Port of `cAIC4`'s `modelAvg` / `getWeights` restricted to Gaussian
# `LinearMixedModel` candidates (docs/math/0009-model-averaging.md).
#
# Public surface:
#   modelavg   — score + weight + combine (Buckland :smoothed or Zhang :zhang)
#   getweights — Zhang-optimal weight optimizer (port of getWeights/.weightOptim)
#
# This file touches the public `MixedModels`/StatsAPI surface (`response`, `fixef`,
# `fixefnames`, `raneftables`) plus `MMInternals.reml`, `MMInternals.conditionalmean`,
# `MMInternals.sigmahat`, and `Numerics`. It does not access `MixedModels` internal
# fields directly.

const _WEIGHTTYPES = (:smoothed, :zhang)

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
- `weights`: the weight scheme. `:zhang` (the default) is the Zhang-optimal Mallows-criterion
  weights via the transcribed `solnp` SQP. `:smoothed` is the
  Buckland (1997) exponential-cAIC smoothed weights `wᵢ = exp(−Δᵢ/2)/Σ exp(−Δ/2)`,
  `Δᵢ = cAICᵢ − min cAIC`, computed in log-space.
- `method, hessian, nboot, sigmapenalty`: forwarded unchanged to [`caic`](@ref) for every
  candidate, so all are scored consistently.

# Returns
- A [`ModelAvgResult`](@ref) carrying the name-keyed averaged `fixeff`/`raneff`, the
  `weights` and per-candidate `caics` (both in input order), the candidate `models`, the
  `weighttype`, and — when `weighttype == :zhang` — the full [`WeightResult`](@ref) in
  `weightresult` (objective `J(ŵ)`, duration).

# Throws
- `ArgumentError` if the candidates differ in response/observation count or REML setting, if
  any candidate is not a Gaussian `LinearMixedModel` of the shared float type, or for an
  unsupported `weights` scheme.

# Example
```jldoctest
julia> using MixedModels, ConditionalAIC

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
    weights::Symbol=:zhang,
    method::Symbol=:auto,
    hessian::Symbol=:analytic,
    nboot::Union{Int,Nothing}=nothing,
    sigmapenalty::Integer=1,
) where {T}
    ms = (m1, rest...)
    _validate_candidates(ms)
    weights in _WEIGHTTYPES || throw(
        ArgumentError(
            "unknown weights=:$weights; supported: :zhang (Zhang-optimal) and :smoothed (Buckland).",
        ),
    )

    # Per-candidate scoring in INPUT order (the unsorted anocAIC analogue, docs/math/0009 §0).
    # The scoring options are captured into `caickwargs` so `getweights`'s Zhang slow path can
    # re-score ρ under the same options rather than `caic`'s defaults.
    caickwargs = CAICKwargs((method, hessian, nboot, Int(sigmapenalty)))
    n = length(ms)
    caic_results = Vector{CAICResult{T,LinearMixedModel{T}}}(undef, n)
    caics = Vector{T}(undef, n)
    for (i, m) in enumerate(ms)
        r = caic(m; caickwargs...)
        caic_results[i] = r
        caics[i] = r.caic
    end

    local wr::Union{Nothing,WeightResult{T}}
    if weights == :smoothed
        w = _bucklandweights(caics)
        wr = nothing
    else  # :zhang
        wr = _zhangweightresult(ms, T[r.dof for r in caic_results])
        w = wr.weights
    end
    fixeff = _avgfixeff(ms, w)
    raneff = _avgraneff(ms, w)
    return ModelAvgResult{T}(
        fixeff, raneff, w, caics, collect(LinearMixedModel{T}, ms), weights, wr, caickwargs
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
# min-subtraction is absorbed by the logsumexp normalisation (the vetted log-space
# entry point), exact-equivalent to the R `exp(-delta/2)/sum(exp(-delta/2))`.
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

# ── Model-averaged prediction (docs/math/0009 §5) ────────────────────────────────────

"""
    predictma(res::ModelAvgResult{T}, newdata; new_re_levels::Symbol=:error) -> Vector{T}

Model-averaged conditional prediction over new data (port of `cAIC4`'s `predictMA`). Each
candidate predicts conditionally on `newdata` and the per-candidate predictions are combined
with the averaging weights of `res`:

```math
\\hat y^{\\mathrm{MA}}(D^*) \\;=\\; \\sum_{i=1}^{M} w_i\\,\\hat y_i(D^*),
\\qquad
\\hat y_i(D^*) \\;=\\; \\texttt{predict}(m_i, D^*),
```

mirroring `cAIC4`'s `w %*% t(sapply(candidates, predict, newdata = D*))`. `predict(mᵢ, D*)`
is **conditional** — `Xβ̂ + Zb̂`, including the predicted random effects for grouping levels
seen in training — matching `lme4`'s default `re.form = NULL`.

The model-averaged prediction is a **stable functional** of the weights: when the candidates'
conditional-mean vectors are collinear (`MᵀM` singular) the weight vector `ŵ` is non-unique,
but `ŷ^MA` is invariant across the flat optimum directions.

# Arguments
- `res`: a [`ModelAvgResult`](@ref) (from [`modelavg`](@ref)); its `weights` and candidate
  `models` are reused, no refitting.
- `newdata`: a `Tables.jl`-compatible table with the same schema the candidates were fit on,
  including a (non-missing, numeric) response column — required by `MixedModels.predict` to
  construct the prediction model.
- `new_re_levels`: how previously-unobserved grouping levels are handled, forwarded to
  `MixedModels.predict`. **Default `:error`** — mirrors `lme4`'s `allow.new.levels = FALSE`
  (this overrides `MixedModels`' own `:missing` default). `:population`
  (treat the random effect as 0) and `:missing` are opt-in.

# Returns
- `Vector{T}` — the model-averaged conditional prediction `ŷ^MA`, one entry per row of `newdata`,
  in input-row order.

# Throws
- `ArgumentError` (from `MixedModels.predict`) if `new_re_levels == :error` and `newdata`
  contains a grouping level not seen in training, or if the response column is missing.

# Example
```jldoctest
julia> using MixedModels, ConditionalAIC

julia> data = MixedModels.dataset(:sleepstudy);

julia> m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), data; REML=false, progress=false);

julia> m2 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), data; REML=false, progress=false);

julia> res = modelavg(m1, m2);

julia> yhat = predictma(res, data);

julia> length(yhat) == length(data.reaction)
true
```
"""
function predictma(res::ModelAvgResult{T}, newdata; new_re_levels::Symbol=:error) where {T}
    w = res.weights
    ms = res.models
    yhat = w[1] .* predict(ms[1], newdata; new_re_levels=new_re_levels)
    for i in 2:length(ms)
        yhat .+= w[i] .* predict(ms[i], newdata; new_re_levels=new_re_levels)
    end
    return yhat
end

# ── Zhang-optimal weight optimizer (docs/math/0009 §1–2, ADR-0007) ──────────────────

"""
    getweights(res::ModelAvgResult{T}) -> WeightResult{T}

Zhang-optimal weight optimization for a set of Gaussian `LinearMixedModel` candidates
(port of `cAIC4`'s `getWeights`). Minimises the Mallows-type criterion

```math
J(w) = (y - \\mu w)^{\\!\\top}(y - \\mu w) + 2\\hat\\sigma^2(\\rho^{\\!\\top} w)
```

over the unit simplex 𝒲 = {w ≥ 0, Σwᵢ = 1} via the transcribed `solnp` augmented-
Lagrangian SQP of `cAIC4`'s weight optimizer.

`σ̂²` is taken from the candidate with the largest effective df (full-precision ρᵢ from
`caic`; cf. cAIC4's selection of the maximum-df candidate).

# Arguments
- `res`: a [`ModelAvgResult`](@ref) (from [`modelavg`](@ref)). The candidate models,
  response, and cAIC scoring are re-used; no new model fitting is performed. When `res` is a
  `:zhang` result the stored [`WeightResult`](@ref) is returned directly; on the `:smoothed`
  slow path ρ is re-scored with [`caic`](@ref) under `res.caickwargs` — the same options
  `modelavg` scored the candidates with, not `caic`'s defaults.

# Returns
- A [`WeightResult`](@ref) with the optimal `weights`, the minimised `objective` J(ŵ),
  and the wall-clock `duration` (seconds).

# Example
```jldoctest
julia> using MixedModels, ConditionalAIC

julia> data = MixedModels.dataset(:sleepstudy);

julia> m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), data; REML=false, progress=false);

julia> m2 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), data; REML=false, progress=false);

julia> res = modelavg(m1, m2; weights=:smoothed);

julia> wr = getweights(res);

julia> sum(wr.weights) ≈ 1.0
true
```
"""
function getweights(res::ModelAvgResult{T}) where {T}
    # Fast path: :zhang result already carries the WeightResult — return it directly.
    if res.weighttype == :zhang && res.weightresult !== nothing
        return res.weightresult::WeightResult{T}
    end
    # Slow path (e.g. a :smoothed result): re-score with caic to get full-precision ρ
    # (docs/math/0009 §6.1: not rounded), then run the optimizer. Re-score under the SAME
    # options `modelavg` scored the candidates with (`res.caickwargs`) — not `caic`'s defaults —
    # so the optimizer's ρ matches the ρ behind `res.caics`.
    ms = res.models
    rho = T[caic(m; res.caickwargs...).dof for m in ms]
    return _zhangweightresult(ms, rho)
end

# Shared Zhang-weight assembly for both `modelavg`'s :zhang branch and `getweights`'s slow
# path. σ̂² is taken from the max-df candidate (docs/math/0009 §6.1); the candidate
# conditional means are stacked column-wise and the optimizer is run on the common response.
# `stack` (not `hcat(…...)`) keeps this type-stable whether `ms` is a tuple (modelavg) or a
# runtime-length Vector (getweights), and yields the same n×M μ column-for-column.
function _zhangweightresult(ms, rho::Vector{T}) where {T}
    sigma_sq = MMInternals.sigmahat(ms[argmax(rho)])^2
    mu = stack(MMInternals.conditionalmean.(ms))
    return _getweights_raw(collect(response(ms[1])), mu, rho, sigma_sq)
end
