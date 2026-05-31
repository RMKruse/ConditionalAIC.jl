# The model-averaging assembly (`modelavg` + `getweights`). Included directly into the
# `cAIC` module. Port of `cAIC4`'s `modelAvg` / `getWeights` restricted to Gaussian
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

using LinearAlgebra: Diagonal, I, Symmetric, cholesky, diag, dot, inv

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
  weights via the transcribed `solnp` SQP (ADR-0007; docs/math/0009 §1–2). `:smoothed` is the
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
    n = length(ms)
    caic_results = Vector{CAICResult{T,LinearMixedModel{T}}}(undef, n)
    caics = Vector{T}(undef, n)
    for (i, m) in enumerate(ms)
        r = caic(m; method=method, hessian=hessian, nboot=nboot, sigmapenalty=sigmapenalty)
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
        fixeff, raneff, w, caics, collect(LinearMixedModel{T}, ms), weights, wr
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
but `ŷ^MA` is invariant across the flat optimum directions (docs/math/0009 §7).

# Arguments
- `res`: a [`ModelAvgResult`](@ref) (from [`modelavg`](@ref)); its `weights` and candidate
  `models` are reused, no refitting.
- `newdata`: a `Tables.jl`-compatible table with the same schema the candidates were fit on,
  including a (non-missing, numeric) response column — required by `MixedModels.predict` to
  construct the prediction model.
- `new_re_levels`: how previously-unobserved grouping levels are handled, forwarded to
  `MixedModels.predict`. **Default `:error`** — mirrors `lme4`'s `allow.new.levels = FALSE`
  (DECISIONS 2026-05-31; this overrides `MixedModels`' own `:missing` default). `:population`
  (treat the random effect as 0) and `:missing` are opt-in.

# Returns
- `Vector{T}` — the model-averaged conditional prediction `ŷ^MA`, one entry per row of `newdata`,
  in input-row order.

# Throws
- `ArgumentError` (from `MixedModels.predict`) if `new_re_levels == :error` and `newdata`
  contains a grouping level not seen in training, or if the response column is missing.

# Example
```jldoctest
julia> using MixedModels, cAIC

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

# ── Model-averaging summary report (docs/math/0009 §5, port of summaryMA) ────────────

"""
    summaryma([io::IO=stdout,] res::ModelAvgResult; randeff::Bool=false) -> Nothing

Print a result summary of a model-averaging fit (port of `cAIC4`'s `summaryMA`). The report
lists the candidate models, the model-averaged fixed effects, the candidate weights, and —
when `randeff = true` — the model-averaged random effects.

# Arguments
- `io`: the output stream (default `stdout`).
- `res`: a [`ModelAvgResult`](@ref) (from [`modelavg`](@ref)).
- `randeff`: whether to also print the model-averaged random effects (default `false`,
  matching `summaryMA`'s `randeff = FALSE`).

# Returns
- `nothing`; the summary is written to `io`.

# Example
```jldoctest
julia> using MixedModels, cAIC

julia> data = MixedModels.dataset(:sleepstudy);

julia> m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), data; REML=false, progress=false);

julia> m2 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), data; REML=false, progress=false);

julia> res = modelavg(m1, m2; weights=:smoothed);

julia> io = IOBuffer();

julia> summaryma(io, res)  # write the candidate models, averaged fixed effects, and weights

julia> occursin("Model Averaged Fixed Effects", String(take!(io)))
true
```
"""
function summaryma(io::IO, res::ModelAvgResult; randeff::Bool=false)
    # In place of cAIC4's `z$call` (not retained), list the candidate model formulas in
    # input order — a recorded divergence (docs/math/0009 §5).
    println(io, "Candidate models:")
    for (i, m) in enumerate(res.models)
        println(io, "  ", lpad(i, 3), ": ", string(formula(m)))
    end
    println(io)
    println(io, "Model Averaged Fixed Effects:")
    for (k, v) in zip(res.fixeff.keys, res.fixeff.values)
        println(io, "  ", rpad(k, 16), v)
    end
    if randeff
        # Heading corrected from summaryMA's copy-pasted "...Fixed Effects" label (an upstream
        # bug not transcribed; ADR-0007 decision 3). Keyed on (grouping, level, term).
        println(io, "\nModel Averaged Random Effects:")
        for (k, v) in zip(res.raneff.keys, res.raneff.values)
            g, lev, term = k
            println(io, "  ", rpad("$g[$lev] $term", 28), v)
        end
    end
    println(io, "\nWeights for underlying Candidate Models:")
    for (i, w) in enumerate(res.weights)
        println(io, "  ", lpad(i, 3), "  ", round(w; digits=6))
    end
    return nothing
end

summaryma(res::ModelAvgResult; kwargs...) = summaryma(stdout, res; kwargs...)

# ── Zhang-optimal weight optimizer (docs/math/0009 §1–2, ADR-0007) ──────────────────

"""
    getweights(res::ModelAvgResult{T}) -> WeightResult{T}

Zhang-optimal weight optimization for a set of Gaussian `LinearMixedModel` candidates
(port of `cAIC4`'s `getWeights`). Minimises the Mallows-type criterion

```math
J(w) = (y - \\mu w)^{\\!\\top}(y - \\mu w) + 2\\hat\\sigma^2(\\rho^{\\!\\top} w)
```

over the unit simplex 𝒲 = {w ≥ 0, Σwᵢ = 1} via the transcribed `solnp` augmented-
Lagrangian SQP of `cAIC4`'s `.weightOptim` (ADR-0007; docs/math/0009 §2).

`σ̂²` is taken from the candidate with the largest effective df (full-precision ρᵢ from
`caic`; cf. cAIC4's `which.max(modelcAIC\$df)`, docs/math/0009 §6.1).

# Arguments
- `res`: a [`ModelAvgResult`](@ref) (from [`modelavg`](@ref)). The candidate models,
  response, and cAIC scoring are re-used; no new model fitting is performed.

# Returns
- A [`WeightResult`](@ref) with the optimal `weights`, the minimised `objective` J(ŵ),
  and the wall-clock `duration` (seconds).

# Example
```jldoctest
julia> using MixedModels, cAIC

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
    # (docs/math/0009 §6.1: not rounded), then run the optimizer.
    ms = res.models
    rho = T[caic(m).dof for m in ms]
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

# Pure optimizer — the getWeights body with model-fitting bypassed (Level-1 testable).
# Transcribes getWeights.R lines 62-122 (initialization + outer loop) using the supplied
# (y, mu, rho, sigma_sq). Variable names shadow R where possible (ADR-0007 decision 1).
function _getweights_raw(
    y::Vector{T}, mu::Matrix{T}, rho::Vector{T}, sigma_sq::T
) where {T<:AbstractFloat}
    nw = length(rho)
    equB = one(T)
    lowb = zeros(T, nw)
    uppb = ones(T, nw)

    # M=1 degenerate case (docs/math/0009 §2.3): ŵ = (1), short-circuit the optimizer.
    if nw == 1
        resid = y - mu[:, 1]
        J = T(dot(resid, resid)) + 2 * sigma_sq * rho[1]
        return WeightResult{T}(ones(T, 1), J, 0.0)
    end

    # R: find_weights <- function(w){ t(y - mu %*% w) %*% (y - mu %*% w) + 2*varDF*(w %*% df) }
    find_weights = let y = y, mu = mu, sigma_sq = sigma_sq, rho = rho
        function (w::AbstractVector)
            resid = y - mu * w
            return T(dot(resid, resid)) + 2 * sigma_sq * T(dot(rho, w))
        end
    end

    # Initialization (getWeights.R lines 62-85)
    p = fill(one(T) / nw, nw)    # R: weights <- rep(1/M, M); p <- c(weights)
    funv = find_weights(p)
    eqv = sum(p) - equB
    maxit = 400
    tol = T(1e-8)
    j = funv                      # R: j <- jh <- funv
    lambda = zero(T)                   # R: lambda <- c(0) (Lagrange multiplier)
    hess = Matrix{T}(I, nw, nw)      # R: hess <- diag(nw)
    mue = T(nw)                     # R: mue <- nw (augmented-Lagrangian penalty)
    iters = 0
    targets = T[funv, eqv]

    t0 = time_ns()
    while iters < maxit
        iters += 1
        # Build scaler (getWeights.R lines 88-91)
        sc1 = min(max(abs(targets[1]), tol), one(T) / tol)
        sc2 = min(max(abs(targets[2]), tol), one(T) / tol)
        scaler = vcat(T[sc1, sc2], ones(T, nw))

        res = _weightoptim(
            p, lambda, targets, hess, mue, scaler, find_weights, equB, lowb, uppb
        )
        p = res.p
        lambda = res.y
        hess = res.hess
        mue = res.lambda

        funv = find_weights(p)
        eqv = sum(p) - equB
        targets = T[funv, eqv]

        tt = (j - targets[1]) / max(targets[1], one(T))   # R: tt <- (j - targets[1])/max(targets[1],1)
        j = targets[1]
        if abs(targets[2]) < 10 * tol      # R: if abs(constraint) < 10*tol
            mue = min(mue, tol)            #    rho <- 0 (stays 0); mue <- min(mue, tol)
        end
        if (tol + tt) <= zero(T)           # R: if (tol + tt) <= 0
            lambda = zero(T)               #    lambda <- 0
            hess = Matrix(Diagonal(diag(hess)))  # R: hess <- diag(diag(hess))
        end
        if sqrt(tt^2 + eqv^2) <= tol      # R: if sqrt(sum((c(tt,eqv))^2)) <= tol
            maxit = iters                  #    maxit <- .iters  (break)
        end
    end
    duration = (time_ns() - t0) / 1e9

    # Renormalize onto the unit simplex (DECISIONS.md 2026-05-31, deliberate divergence from
    # cAIC4). The transcribed `solnp` SQP enforces Σwᵢ = 1 only to its convergence tolerance
    # (the outer break gates on |Σp − 1| ≤ tol = 1e-8, larger if `maxit` is hit without
    # convergence), and — like `cAIC4`'s `getWeights` — returns that raw iterate. Dividing by
    # the sum makes the public weights sum to 1 to machine precision, so the model-averaged
    # effects are an exact convex combination. The objective is re-evaluated at the projected
    # weights so `WeightResult.objective == find_weights(weights)` stays exactly consistent.
    s = sum(p)
    s > zero(T) || throw(
        DomainError(
            s,
            "getweights: optimized weights sum to $s (≤ 0); cannot project onto the unit " *
            "simplex. The weight optimization did not converge to a feasible point.",
        ),
    )
    p ./= s
    j = find_weights(p)
    return WeightResult{T}(p, j, duration)
end

# Inner step of the SQP — faithful Julia transcription of cAIC4's `.weightOptim`
# (weightOptim.R). Variable names shadow R names directly (ADR-0007 decision 1).
# `find_weights` is the Mallows objective closure; `equB`, `lowb`, `uppb` are the
# feasibility data. Returns (p, y, hess, lambda) unscaled, mirroring R's `ans` list.
#
# NOTE: `rho` (augmented-penalty coefficient inside this function) is always 0 in this
# implementation — it is a dead variable in both getWeights.R and weightOptim.R.
# Kept for auditability (ADR-0007 decision 1).
function _weightoptim(
    weights_in::AbstractVector{T},   # R: weights (= p0 on entry)
    lm_in::T,                # R: lm (Lagrange multiplier)
    targets_in::Vector{T},   # R: targets [funv, eqv] (unscaled)
    hess_in::Matrix{T},      # R: hess
    lambda_in::T,            # R: lambda (augmented-Lagrangian penalty)
    scaler::Vector{T},       # R: scaler (nw+2 vector)
    find_weights,            # R: find_weights closure
    equB::T,                 # R: equB (= 1.0)
    lowb::Vector{T},         # R: lowb
    uppb::Vector{T},         # R: uppb
) where {T<:AbstractFloat}
    # ── Local constants (weightOptim.R lines 11-14) ───────────────────────────────────
    rho_aug = zero(T)       # R: rho <- 0  (augmented penalty; stays 0, see NOTE above)
    inner_maxit = 800        # R: maxit (inner loop limit)
    delta = T(1e-7)       # R: delta
    tol = T(1e-8)       # R: tol
    nw = length(weights_in)   # R: numw = length(m)
    mm = nw            # R: mm = numw

    # ── Mutable working copies (R: p0 <- weights; hess and targets are passed by value) ─
    p0 = Vector{T}(weights_in)
    hess = copy(hess_in)
    lambda = lambda_in
    targets = copy(targets_in)

    l = zeros(T, 3)
    ab = hcat(lowb, uppb)    # M×2: [lowb  uppb], R: ab <- cbind(lowb, uppb)
    st = zeros(T, 3)
    sc = zeros(T, 2)

    # ── Scale (weightOptim.R lines 26-32) ─────────────────────────────────────────────
    targets ./= scaler[1:2]
    p_sc = scaler[3:(nw + 2)]                       # p-scalers (always 1.0)
    p0 ./= p_sc
    ab ./= reshape(p_sc, :, 1)                  # divide each row i by p_sc[i]
    lm = scaler[2] * lm_in / scaler[1]          # R: lm <- scaler[2]*lm/scaler[1]
    hess .*= (p_sc * p_sc') ./ scaler[1]        # R: hess <- hess*(outer(p_sc,p_sc))/scaler[1]

    # ── Gradient and Jacobian via finite differences (lines 34-48) ────────────────────
    j = targets[1]
    a = zeros(T, 1, nw)                 # R: a <- matrix(0, 1, numw)
    g = zeros(T, nw)                    # R: g <- rep(0, numw)
    p = copy(p0)                        # R: p <- p0[1:numw]
    constraint = targets[2]                      # R: constraint <- targets[2]

    for i in 1:nw
        p0[i] += delta
        tmpv = p0 .* p_sc                       # R: p0[1:numw] * scaler[3:(numw+2)]
        funv = find_weights(tmpv)
        eqv = sum(tmpv) - equB
        tv = T[funv, eqv] ./ scaler[1:2]
        g[i] = (tv[1] - j) / delta
        a[1, i] = (tv[2] - constraint) / delta
        p0[i] -= delta
    end

    b = dot(vec(a), p0) - constraint          # R: b <- a %*% p0 - constraint (scalar)
    ind = -1
    l[1] = tol - abs(constraint)                 # R: l[1] <- tol - max(abs(constraint))

    # ── Feasibility restoration (lines 50-100) ────────────────────────────────────────
    if l[1] <= zero(T)
        ind = 1
        # Extend p0 and a by one element/column (slack variable)
        p0_ext = vcat(p0, one(T))               # R: p0[numw+1] <- 1
        a_ext = hcat(a, T(-constraint))        # R: a <- cbind(a, -constraint)
        cx = hcat(zeros(T, 1, nw), ones(T, 1, 1))  # R: cx <- cbind(matrix(0,1,numw), 1)
        dx_ext = ones(T, nw + 1)               # R: dx <- rep(1, numw+1)
        go = one(T)
        minit_f = 0

        while go >= tol
            minit_f += 1
            gap = hcat(p0_ext[1:mm] - ab[:, 1], ab[:, 2] - p0_ext[1:mm])  # M×2
            _sort_rows2!(gap)
            dx_ext[1:mm] = gap[:, 1]
            dx_ext[nw + 1] = p0_ext[nw + 1]

            # R: y <- try(qr.solve(t(a %*% diag(dx)), dx * t(cx)), silent=TRUE)
            A_f = (a_ext * Diagonal(dx_ext))'    # (nw+1)×1 matrix
            rhs_f = dx_ext .* vec(cx')           # nw+1 vector
            y_f = try
                A_f \ rhs_f
            catch
                @warn "getweights: feasibility restoration (qr.solve) failed — ill-conditioned weight problem; optimum may be non-unique."
                p_ret = p0_ext[1:nw] .* p_sc
                hess_ret = scaler[1] .* hess ./ (p_sc * p_sc')
                return (p=p_ret, y=zero(T), hess=hess_ret, lambda=lambda)
            end

            # R: v <- dx * (dx * (t(cx) - t(a) %*% y))
            v_f = dx_ext .* (dx_ext .* (vec(cx') .- vec(a_ext' * y_f)))
            if v_f[nw + 1] > zero(T)
                z = p0_ext[nw + 1] / v_f[nw + 1]
                for i in 1:mm
                    if v_f[i] < zero(T)
                        z = min(z, -(ab[i, 2] - p0_ext[i]) / v_f[i])
                    elseif v_f[i] > zero(T)
                        z = min(z, (p0_ext[i] - ab[i, 1]) / v_f[i])
                    end
                end
                if z >= p0_ext[nw + 1] / v_f[nw + 1]
                    p0_ext .-= z .* v_f
                else
                    p0_ext .-= T(0.9) * z .* v_f
                end
                go = p0_ext[nw + 1]
                if minit_f >= 10
                    go = zero(T)
                end
            else
                go = zero(T)
            end
        end

        a = a_ext[:, 1:nw]                       # R: a <- matrix(a[,1:numw], ncol=numw)
        b = dot(vec(a), p0_ext[1:nw])            # R: b <- a %*% p0[1:numw] (scalar)
        p = p0_ext[1:nw]
    else
        p = copy(p0)
    end

    # ── Recompute targets after feasibility (lines 102-111) ───────────────────────────
    y = zero(T)
    if ind > 0
        tmpv = p .* p_sc
        funv = find_weights(tmpv)
        eqv = sum(tmpv) - equB
        targets .= T[funv, eqv] ./ scaler[1:2]
    end

    j = targets[1]
    targets[2] -= dot(vec(a), p) - b             # R: targets[2] <- targets[2] - a%*%p + b
    j = targets[1] - lm * targets[2] + rho_aug * targets[2]^2

    # ── Inner loop (BFGS + bisection line search, lines 113-262) ─────────────────────
    sx = copy(p)
    yg = copy(g)

    y_val = zero(T)   # initialized here so it is in scope after the inner loop
    minit = 0
    while minit < inner_maxit
        minit += 1

        # Gradient of augmented Lagrangian (lines 115-127)
        if ind > 0
            for i in 1:nw
                p[i] += delta
                tmpv = p .* p_sc
                funv = find_weights(tmpv)
                eqv = sum(tmpv) - equB
                tv = T[funv, eqv] ./ scaler[1:2]
                tv[2] -= dot(vec(a), p) - b
                tv_aug = tv[1] - lm * tv[2] + rho_aug * tv[2]^2
                g[i] = (tv_aug - j) / delta
                p[i] -= delta
            end
        end

        # BFGS Hessian update (lines 128-136)
        if minit > 1
            yg_d = g .- yg                       # gradient difference
            sx_d = p .- sx                       # step
            sc[1] = dot(sx_d, hess * sx_d)
            sc[2] = dot(sx_d, yg_d)
            if sc[1] * sc[2] > zero(T)
                Hsx = hess * sx_d
                hess .-= (Hsx * Hsx') ./ sc[1]
                hess .+= (yg_d * yg_d') ./ sc[2]
            end
        end
        sx = copy(p)
        yg = copy(g)

        # Barrier diagonal (lines 138-142)
        dx = fill(T(0.1), nw)
        gap = hcat(p[1:mm] - ab[:, 1], ab[:, 2] - p[1:mm])   # M×2
        _sort_rows2!(gap)
        gap1 = gap[:, 1] .+ sqrt(eps(T))        # R: gap[,1] + sqrt(.Machine$double.eps)
        dx[1:mm] = one(T) ./ gap1

        go_lm = T(-1)
        lambda /= 10                             # R: lambda <- lambda/10

        # Levenberg–Marquardt feasibility ramp (lines 145-175)
        p_trial = similar(p)
        y_val = zero(T)
        while go_lm <= zero(T)
            # R: cz <- try(chol(hess + lambda * diag(dx*dx, nw, nw)), silent=TRUE)
            H_reg = Symmetric(hess .+ lambda .* Diagonal(dx .* dx))
            cz_U = try
                cholesky(H_reg).U
            catch
                @warn "getweights: Cholesky decomposition failed (ill-conditioned). Weights may be non-unique."
                p_ret = p .* p_sc
                hess_ret = scaler[1] .* hess ./ (p_sc * p_sc')
                return (p=p_ret, y=zero(T), hess=hess_ret, lambda=lambda)
            end

            # R: cz <- try(solve(cz), silent=TRUE);  yg <- t(cz) %*% g
            # §9-compliant transcription: the inverse Cholesky factor cz = inv(cz_U) is
            # never materialised. Every downstream use is a triangular solve against the
            # factor cz_U — provably equivalent (cz' * v == cz_U' \ v, cz * v == cz_U \ v)
            # and matching cAIC4 to the same roundoff. Supersedes ADR-0007 decision (2);
            # see DECISIONS 2026-05-31.
            yg_kkt, A_kkt = try
                (cz_U' \ g, cz_U' \ a')          # R: yg <- t(cz)%*%g ;  t(cz)%*%t(a)
            catch
                @warn "getweights: triangular solve of the Cholesky factor failed (ill-conditioned). Weights may be non-unique."
                p_ret = p .* p_sc
                hess_ret = scaler[1] .* hess ./ (p_sc * p_sc')
                return (p=p_ret, y=zero(T), hess=hess_ret, lambda=lambda)
            end
            y_kkt = try
                A_kkt \ yg_kkt
            catch
                @warn "getweights: KKT multiplier solve failed (ill-conditioned). Weights may be non-unique."
                p_ret = p .* p_sc
                hess_ret = scaler[1] .* hess ./ (p_sc * p_sc')
                return (p=p_ret, y=zero(T), hess=hess_ret, lambda=lambda)
            end
            y_val = only(y_kkt)

            # R: u <- -cz %*% (yg - (t(cz) %*% t(a)) %*% y)
            u_step = -(cz_U \ (yg_kkt .- A_kkt .* y_val))   # cz * v == cz_U \ v
            p_trial .= u_step[1:nw] .+ p
            go_lm = minimum(vcat(p_trial[1:mm] - ab[:, 1], ab[:, 2] - p_trial[1:mm]))
            lambda *= 3                         # R: lambda <- 3*lambda
        end

        # ── Three-point bisection line search (lines 176-232) ─────────────────────────
        l[1] = zero(T)
        targets1 = copy(targets)
        targets2 = copy(targets)
        st[1] = j
        st[2] = j
        p1 = copy(p)              # ptt[:,1]
        p2 = copy(p)              # ptt[:,2] (midpoint, updated each bisection step)
        l[3] = one(T)
        p3 = copy(p_trial)        # ptt[:,3] (trial point)

        tmpv = p3 .* p_sc
        funv = find_weights(tmpv)
        eqv = sum(tmpv) - equB
        targets3 = T[funv, eqv] ./ scaler[1:2]
        st[3] = targets3[1]
        targets3[2] -= dot(vec(a), p3) - b
        st[3] = targets3[1] - lm * targets3[2] + rho_aug * targets3[2]^2

        go_bs = one(T)
        while go_bs > tol
            l[2] = (l[1] + l[3]) / 2
            p2 = (one(T) - l[2]) .* p .+ l[2] .* p_trial   # R: ptt[,2] <- (1-l[2])*p + l[2]*p0
            tmpv = p2 .* p_sc
            funv = find_weights(tmpv)
            eqv = sum(tmpv) - equB
            targets2 .= T[funv, eqv] ./ scaler[1:2]
            st[2] = targets2[1]
            targets2[2] -= dot(vec(a), p2) - b
            st[2] = targets2[1] - lm * targets2[2] + rho_aug * targets2[2]^2

            targetsm = maximum(st)
            if targetsm < j
                targetsn = minimum(st)
                go_bs = tol * (targetsm - targetsn) / (j - targetsm)
            end

            con1 = st[2] >= st[1]
            con2 = st[1] <= st[3] && st[2] < st[1]
            con3 = st[2] < st[1] && st[1] > st[3]

            if con1
                st[3] = st[2];
                targets3 = copy(targets2);
                l[3] = l[2];
                p3 = copy(p2)
            end
            if con2
                st[3] = st[2];
                targets3 = copy(targets2);
                l[3] = l[2];
                p3 = copy(p2)
            end
            if con3
                st[1] = st[2];
                targets1 = copy(targets2);
                l[1] = l[2];
                p1 = copy(p2)
            end

            if go_bs >= tol
                go_bs = l[3] - l[1]
            end
        end

        # ── Select best of three-point bracket (lines 233-261) ────────────────────────
        ind = 1
        targetsn = minimum(st)
        if j <= targetsn
            inner_maxit = minit                  # converged: no progress, break
        end
        reduce = (j - targetsn) / (one(T) + abs(j))
        if reduce < tol
            inner_maxit = minit
        end

        con1 = st[1] < st[2]
        con2 = st[3] < st[2] && st[1] >= st[2]
        con3 = st[1] >= st[2] && st[3] >= st[2]

        if con1
            j = st[1];
            p = copy(p1);
            targets = copy(targets1)
        end
        if con2
            j = st[3];
            p = copy(p3);
            targets = copy(targets3)
        end
        if con3
            j = st[2];
            p = copy(p2);
            targets = copy(targets2)
        end
    end  # inner loop

    # ── Unscale and return (weightOptim.R lines 263-267) ─────────────────────────────
    p_out = p .* p_sc                         # = p * 1.0 (p_sc = ones)
    y_out = scaler[1] * y_val / scaler[2]     # R: y <- scaler[1]*y/scaler[2]
    hess_out = scaler[1] .* hess ./ (p_sc * p_sc')  # = scaler[1] * hess
    return (p=p_out, y=y_out, hess=hess_out, lambda=lambda)
end

# Sort the two columns of an M×2 matrix in-place so each row is in ascending order.
# Transcribes R's `t(apply(gap, 1, FUN=function(x) sort(x)))`.
function _sort_rows2!(m::Matrix)
    for i in axes(m, 1)
        if m[i, 1] > m[i, 2]
            m[i, 1], m[i, 2] = m[i, 2], m[i, 1]
        end
    end
    return m
end
