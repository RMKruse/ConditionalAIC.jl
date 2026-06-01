"""
    ConditionalAIC.MMInternals

**Quarantine module — the single, auditable touchpoint for `MixedModels.jl` internals.**

*All* access to `MixedModels.jl` internal fields and unexported functions lives here
and nowhere else. This module performs **no** translation or abstraction; it reaches
into the fitted-model object directly and exists solely to make every internal
touchpoint auditable in one place. Each accessor shape-asserts what it extracts so a
silent upstream change surfaces as a clear error rather than a wrong number downstream.

# Internal-access table

Pinned against **`MixedModels = "=5.5.1"`**. On a version bump, walking this table is the
required checklist; accessing an internal not listed here is forbidden — add the row
first.

| Touchpoint        | Kind        | Used by             | Extracted quantity                                       |
|:------------------|:------------|:--------------------|:---------------------------------------------------------|
| `m.optsum.REML`   | field       | [`reml`]            | REML flag (`Bool`); which objective was fitted           |
| `mb.optsum.REML =`| field (mutating write) | [`bootstrapfit`] | set the bootstrap LMM's REML flag (from [`reml`](@ref)) so it optimises the same objective as the original before refitting |
| `m.optsum.returnvalue` | field  | [`converged`]       | optimizer return code (`Symbol`); non-converged when in the failure modes |
| `m.sigma`         | property    | [`sigmahat`]        | residual standard deviation σ̂                            |
| `ranef(m)`        | exported fn | [`bhat`]            | predicted random effects b̂ = λu, per grouping            |
| `m.X`             | field       | [`fixedeffects`]    | n×p fixed-effects design X                                |
| `response(m)`     | exported fn | [`responsevec`]     | response vector y                                        |
| `fitted(m)`       | exported fn | [`conditionalmean`] | conditional fitted mean ŷ = Xβ̂ + Zb̂                      |
| `leverage(m)`     | exported fn | [`rho0`]            | per-observation hat-matrix diagonal; ρ₀ = its sum       |
| `m.reterms`       | field       | [`retermdesigns`], [`reduceboundary`] | the per-grouping `ReMat`s                  |
| `Matrix(re)`      | constructor | [`retermdesigns`]   | dense random-effects design Z block (n×qₜ) per reterm    |
| `re.λ`            | field       | [`retermlambdas`], [`reduceboundary`] | relative covariance factor λ block (kₜ×kₜ) |
| `m.parmap`        | field       | [`parmap`]          | θ → (reterm, row, col) map — the `lme4` `Lind` analogue  |
| `issingular(m)`   | exported fn | [`issingular`]      | is a variance component on the boundary (some `λ[d,d]=0`)|
| `m.feterm`        | field       | [`reduceboundary`]  | the fixed-effects term `FeTerm`, reused by the reduced fit|
| `m.formula`       | field       | [`reduceboundary`]  | the model formula (bookkeeping for the reduced-fit ctor) |
| `re.trm/refs/levels/cnames/z/scratch` | fields | [`reduceboundary`] | `ReMat` design pieces, column-subset to rebuild a reduced `ReMat` |
| `ReMat{T,S}(…)`   | constructor (10-slot positional) | [`reduceboundary`], [`bootstrapfit`] | rebuild a boundary-reduced random-effects term; the field-name layout is pinned via `_REMAT_FIELDS` (drift-guarded in [`_subsetreterm`]) and the reduced design is passed into **both** the `z` and `wtz` slots — valid because they alias for an unweighted term |
| `adjA(refs, z)`   | fn          | [`reduceboundary`]  | the `ReMat` adjoint sparse block for the subset design    |
| `LinearMixedModel(y, feterm, reterms, form)` | constructor | [`reduceboundary`] | assemble the reduced model from reused design objects |
| `fit!(m)`         | exported fn | [`reduceboundary`], [`bootstrapfit`] | refit the reduced / bootstrap model       |
| `m.θ`             | property    | [`bhessian`]        | fitted θ̂ — the FD evaluation point and restoration check |
| `ForwardDiff.hessian(m)` | ext fn (experimental) | [`bhessian`] | s×s deviance Hessian (`:forwarddiff`; frozen-σ — see below) |
| `objective!(m)`   | unexported  | [`bhessian`]        | curried θ→deviance closure (`Base.Fix1`); FD driver target |
| `setθ!(m, θ)`     | unexported  | [`bhessian`]        | set variance parameters θ — restore θ̂ after FD perturbation |
| `updateL!(m)`     | unexported  | [`bhessian`]        | refactorise `L` after `setθ!` — completes the restore    |
| `m.η`             | property    | [`glmmlinpred`]     | linear predictor η (GLMM, n-vector); aliases `m.resp.eta`|
| `m.resp.mu`       | field       | [`glmmfittedmu`]    | fitted mean μ on the response scale (GLMM, n-vector)     |
| `m.resp.y`        | field       | [`glmmresponse`]    | response vector y (GLMM, n-vector, on the μ scale)       |
| `m.resp.d`        | field       | [`glmmdist`]        | GLM distribution family D (from `GeneralizedLinearMixedModel{T,D}`)|
| `m.LMM.feterm.rank` | field     | [`glmmfixedefrank`] | rank of fixed-effects design in the working LMM          |
| `m.LMM`           | field       | [`reduceboundary`]  | the working `LinearMixedModel` of a GLMM — source of its `reterms`/`feterm`/`formula` for the reduced fit |
| `m.LMM.reterms`   | field       | [`glmmisfullysingular`], [`reduceboundary`] | random-effects terms of the working LMM; each `re.λ` diagonal checked for boundary directions (all-zero ⇒ fully singular; some-zero ⇒ reduced) |
| `m.LMM.feterm` / `m.LMM.formula` | fields | [`reduceboundary`] | fixed-effects term + formula of the working LMM, reused by the reduced GLMM's working LMM |
| `m.β`             | property    | [`reduceboundary`]  | fitted fixed-effects β̂ (GLMM), reused as the refit's β/β₀ start |
| `m.wt`            | field       | [`reduceboundary`]  | GLMM prior weights; empty for an unweighted fit (then a length-`n` ones vector is supplied) |
| `m.resp`          | field       | [`reduceboundary`]  | the `GlmResp` (family/link/response), deep-copied into the reduced GLMM |
| `vsize(t)` / `nlevs(t)` | unexported fns | [`reduceboundary`] | per-reterm random-effect width and group count — size the reduced `u` scratch |
| `rlmm.reterms`    | field       | [`reduceboundary`]  | reterms of the freshly-built reduced working LMM — iterated (with `vsize`/`nlevs`) to allocate the reduced GLMM's `u`/`u₀`/`b` scratch |
| `rlmm.θ`          | property    | [`reduceboundary`]  | initial θ of the reduced working LMM (identity λ) — passed into the reduced GLMM's `θ` constructor slot |
| `GeneralizedLinearMixedModel{T,D}(…)` | constructor | [`reduceboundary`] | assemble the reduced GLMM around the reduced working LMM |
| `refit!(m, y)`    | exported fn | [`bootstrapglmmfit`], [`refitglmm_eta`], [`bernoulliflipmu`] | refit a GLMM copy to a new response vector y |
| `m.η` (post-refit)| property    | [`refitglmm_eta`]   | linear predictor η̂ of the refitted GLMM copy (Chen–Stein refit loop) |
| `m.resp.wts`      | field       | [`glmmpriorweights`]| prior weights (binomial denominators nᵢ); empty `T[]` for unweighted (Poisson, Bernoulli) fits |
| `m.formula.rhs`   | field       | [`reterminfo`], [`fixedterm`] | formula RHS tuple: leading `MatrixTerm` (fixed) + the RE terms (RE-structure read) |
| `MixedModels.schematize(f, data, contrasts)` | unexported fn | [`reterminfo`] | apply the model schema to a `keep` formula fragment so its `|` bars become RE terms |
| `m.formula.lhs`   | field       | [`responseterm`]    | formula response term; the `lhs` of the rendered candidate formula             |
| `RandomEffectsTerm` (`.lhs`/`.rhs`) | type/fields | [`reterminfo`] | a correlated RE term: `.lhs` directions `MatrixTerm`, `.rhs` grouping `CategoricalTerm` |
| `MixedModels.ZeroCorr` (`.term`) | type/field | [`reterminfo`] | an uncorrelated `zerocorr` term; unwrapped to its inner `RandomEffectsTerm` |
| `MatrixTerm` (`.terms`), `InterceptTerm{B}`, `CategoricalTerm.sym`, `termvars` | StatsModels types/fns | [`reterminfo`], [`fixedterm`] | RE-direction labels (`"(Intercept)"`/slope names), grouping symbol, fixed-part identification |

**Experimental surface.** `ForwardDiff.hessian(::LinearMixedModel)` (the
`MixedModelsForwardDiffExt` extension, used by [`bhessian`]) is the one touchpoint on
`MixedModels`' *experimental* AD surface; the docs warn that which parameters are
differentiated alongside θ may change, which would silently alter B's dimension, so
[`bhessian`] shape-asserts the `s×s` result against the `=5.5.1` pin. The companion
`FiniteDiff.finite_difference_hessian(::LinearMixedModel)` extension is **deliberately not
accessed** — the `:finitediff` source self-drives `FiniteDiff` over the stable
`objective!`/`setθ!`/`updateL!` trio instead.
"""
module MMInternals

using LinearAlgebra: Diagonal, LowerTriangular, I
using MixedModels:
    MixedModel,
    LinearMixedModel,
    GeneralizedLinearMixedModel,
    RandomEffectsTerm,
    AbstractReMat,
    ReMat,
    vsize,
    nlevs,
    Poisson,
    Binomial,
    Bernoulli,
    ranef,
    response,
    fitted,
    leverage,
    adjA,
    fit!,
    refit!,
    objective!,
    setθ!,
    updateL!
using MixedModels: MixedModels
using FiniteDiff: finite_difference_hessian
using ForwardDiff: ForwardDiff
using Random: AbstractRNG

const PINNED_VERSION = "5.5.1"

# StatsModels term types/functions (`MatrixTerm`, `InterceptTerm`, `termvars`, the
# `FormulaTerm` field layout) are the upstream term representation `m.formula` is built
# from; interpreting them is a quarantine concern (M4 `reterminfo`/`responseterm`/`fixedterm`).
const SM = MixedModels.StatsModels

# Raised when an internal touchpoint yields a value of an unexpected type/shape —
# i.e. `MixedModels` has drifted from the pinned version. Failing loud here turns a
# silent upstream change into a clear error instead of a wrong number downstream.
@noinline function _drift(touchpoint::AbstractString, expected, got)
    return error(
        "MixedModels internal `$touchpoint` produced $(typeof(got)); expected $expected. \
         This indicates drift from the pinned MixedModels v$PINNED_VERSION — reconcile \
         the internal-access table in `MMInternals` against the new version before use."
    )
end

# The exact field-name layout of `GeneralizedLinearMixedModel` (pinned MixedModels
# v5.5.1), against which [`reduceboundary`](@ref) constructs a reduced GLMM
# positionally. Guarding the names — not just the per-field types the parametric
# constructor checks — is what catches a reorder among the many same-typed fields.
const _GLMM_FIELDS = (
    :LMM, :β, :β₀, :θ, :b, :u, :u₀, :resp, :η, :wt, :devc, :devc0, :sd, :mult
)

# The exact field-name layout of `ReMat` (pinned MixedModels v5.5.1), against which
# [`_subsetreterm`](@ref) rebuilds a boundary-reduced random-effects term via the 10-slot
# positional constructor. The reduced design `znew` is passed into BOTH the `z` (slot 5) and
# `wtz` (slot 6) slots — valid only because they alias for an unweighted term (the `ReMat`
# docstring: "z and wtz are the same object for unweighted cases"). A reorder among the
# same-typed slots (`z`/`wtz`, both `Matrix{T}`; `refs`/`inds`; the design/scratch matrices)
# would be silently accepted by the positional constructor, yielding a malformed `ReMat` with
# no error — so pin the names and fail loud on drift, exactly as `_GLMM_FIELDS` guards the
# GLMM constructor.
const _REMAT_FIELDS = (:trm, :refs, :levels, :cnames, :z, :wtz, :λ, :inds, :adjA, :scratch)

"""
    reml(m::LinearMixedModel) -> Bool

The REML flag the model was fitted under (`m.optsum.REML`): `true` for restricted
maximum likelihood, `false` for maximum likelihood. The conditional-AIC machinery
dispatches on this to use the matching objective for θ̂, b̂, and the Hessian.
"""
function reml(m::LinearMixedModel)
    flag = m.optsum.REML
    flag isa Bool || _drift("m.optsum.REML", Bool, flag)
    return flag
end

# The optimizer return codes that signal a failed (non-converged) fit, mirroring
# `MixedModels`' own `_NLOPT_FAILURE_MODES` (the NLopt backend's failure classification). A
# return value outside this set — `:FTOL_REACHED`, `:XTOL_REACHED`, `:SUCCESS`,
# `:STOPVAL_REACHED`, the tolerance-reached successes, and the soft `:ROUNDOFF_LIMITED` — is a
# converged fit. Held locally (not imported from the unexported backend constant) so the
# convergence test does not reach past the documented touchpoint.
const _NONCONVERGED_RETURNS = (
    :FAILURE,
    :INVALID_ARGS,
    :OUT_OF_MEMORY,
    :FORCED_STOP,
    :MAXEVAL_REACHED,
    :MAXTIME_REACHED,
)

"""
    converged(m::MixedModel) -> Bool

Whether the model's variance-parameter optimization converged (`m.optsum.returnvalue` is not
one of the optimizer failure codes). The convergence signal `stepcaic`'s `skipnonconverged`
option (the `cAIC4` `calcNonOptimMod` analogue) tests to exclude a non-converged candidate from
the comparison.

`lme4` flags non-convergence with a richer gradient/Hessian check (its `optinfo` convergence
code);
`MixedModels.jl` exposes only the optimizer return code, so this is the faithful analogue
available — a documented divergence. Works for both `LinearMixedModel` and
`GeneralizedLinearMixedModel` (each carries its own `optsum`).
"""
function converged(m::MixedModel)
    ret = m.optsum.returnvalue
    ret isa Symbol || _drift("m.optsum.returnvalue", Symbol, ret)
    return !(ret in _NONCONVERGED_RETURNS)
end

"""
    sigmahat(m::LinearMixedModel{T}) -> T

The estimated residual standard deviation σ̂ (`m.sigma`), in the model's float type
`T`. It scales the conditional log-likelihood and enters the Gaussian bias correction.
"""
function sigmahat(m::LinearMixedModel{T}) where {T}
    s = m.sigma
    s isa T || _drift("m.sigma", T, s)
    return s
end

"""
    bhat(m::LinearMixedModel{T}) -> Vector{Matrix{T}}

The predicted random effects b̂ = λu (`ranef(m)`): one matrix per grouping factor,
each shaped `(n random-effect coefficients) × (n groups)`. These are the conditional
modes on which the conditional log-likelihood ℓ(y | b̂, β̂, θ̂) is evaluated.
"""
function bhat(m::LinearMixedModel{T}) where {T}
    b = ranef(m)
    b isa Vector{Matrix{T}} || _drift("ranef(m)", Vector{Matrix{T}}, b)
    return b
end

"""
    fixedeffects(m::LinearMixedModel{T}) -> Matrix{T}

The `n×p` fixed-effects design matrix `X` (`m.X`). Enters the bias correction through the
fixed-effects-adjusted projector `A` and (via `p = ncol(X)`) the REML degrees `nθ = n−p`.
"""
function fixedeffects(m::LinearMixedModel{T}) where {T}
    X = m.X
    X isa Matrix{T} || _drift("m.X", Matrix{T}, X)
    return X
end

"""
    responsevec(m::LinearMixedModel{T}) -> Vector{T}

The response vector `y` (`response(m)`), materialised as a dense `Vector{T}`. The
conditional log-likelihood and the residual `e = y − ŷ` are built from it.
"""
function responsevec(m::LinearMixedModel{T}) where {T}
    return collect(response(m))::Vector{T}
end

"""
    conditionalmean(m::LinearMixedModel{T}) -> Vector{T}

The conditional fitted mean `ŷ = X β̂ + Z b̂` (`fitted(m)`) — the mean the conditional
log-likelihood is evaluated about and the source of the conditional residual `e = y − ŷ`.
"""
function conditionalmean(m::LinearMixedModel{T}) where {T}
    μ = fitted(m)
    μ isa Vector{T} || _drift("fitted(m)", Vector{T}, μ)
    return μ
end

"""
    rho0(m::LinearMixedModel{T}) -> T

The naive plug-in effective degrees of freedom `ρ₀ = tr(H₁) = sum(leverage(m))` — the
trace of the hat matrix `y ↦ ŷ` at the fitted, fixed variance parameters.
`leverage(m)` returns the per-observation hat-matrix *diagonal*; ρ₀
is its sum. This is the `MixedModels`-native ρ₀ (computed via triangular solves against the
fit's Cholesky `L`), used to cross-check the bias correction (`ρ ≥ ρ₀`).
"""
function rho0(m::LinearMixedModel{T}) where {T}
    lev = leverage(m)
    lev isa Vector{T} || _drift("leverage(m)", Vector{T}, lev)
    return sum(lev)
end

"""
    retermdesigns(m::LinearMixedModel{T}) -> Vector{Matrix{T}}

The dense random-effects design `Z` block for each grouping factor (`Matrix(re)` over
`m.reterms`), each `n×qₜ`. `MixedModels` amalgamates random-effects terms sharing a
grouping factor into a single `ReMat`, so the blocks are indexed by *reterm*, matching the
first field of [`parmap`](@ref).
"""
function retermdesigns(m::LinearMixedModel{T}) where {T}
    return Matrix{T}[Matrix(re) for re in m.reterms]
end

"""
    retermlambdas(m::LinearMixedModel{T}) -> Vector{Matrix{T}}

The relative covariance factor `λ` block (`re.λ`, `kₜ×kₜ` lower-triangular, dense) for each
reterm. The per-group relative covariance is `λ λᵀ`; together with [`retermdesigns`](@ref)
and [`parmap`](@ref) it fixes the scaled marginal variance `V₀ = Iₙ + Z λλᵀ Zᵀ` and the
derivative matrices `Wⱼ`.
"""
function retermlambdas(m::LinearMixedModel{T}) where {T}
    return Matrix{T}[Matrix(re.λ) for re in m.reterms]
end

"""
    parmap(m::LinearMixedModel) -> Vector{NTuple{3,Int}}

The free-covariance-parameter map `m.parmap` — the `lme4` `Lind` analogue. Entry `s` is
`(t, i, j)`: the `s`-th component `θₛ` occupies position `(i, j)` of reterm `t`'s `λ`
block. This drives the `Wⱼ` derivative-pattern construction.
"""
function parmap(m::LinearMixedModel)
    pm = m.parmap
    pm isa Vector{NTuple{3,Int}} || _drift("m.parmap", Vector{NTuple{3,Int}}, pm)
    return pm
end

"""
    issingular(m::LinearMixedModel) -> Bool

Whether the fit sits on the boundary of the parameter space — a variance component
estimated at zero (`MixedModels.issingular`). For a `LinearMixedModel` this is exactly the
condition that some reterm's relative-covariance diagonal `λ[d, d]` is zero, i.e. a
random-effect direction has collapsed. It is the trigger for the drop-and-refit path
([`reduceboundary`]); it is the analogue of `cAIC4`'s `theta == 0` test in
`deleteZeroComponents`. A boundary fit is a first-class, supported case, never an error.
"""
function issingular(m::LinearMixedModel)
    s = MixedModels.issingular(m)
    s isa Bool || _drift("issingular(m)", Bool, s)
    return s
end

"""
    issingular(m::GeneralizedLinearMixedModel) -> Bool

Whether the GLMM's working `LinearMixedModel` sits on the boundary — *some* random-effect
direction has collapsed (`λ[d, d] = 0` for at least one `d`). True for both partial and
full singularity; [`glmmisfullysingular`] distinguishes the all-collapsed case. This is the
trigger for the GLMM drop-and-refit cascade ([`reduceboundary`]).
"""
function issingular(m::GeneralizedLinearMixedModel)
    s = MixedModels.issingular(m)
    s isa Bool || _drift("issingular(m)", Bool, s)
    return s
end

"""
    reduceboundary(m::LinearMixedModel{T}) -> Union{Nothing,LinearMixedModel{T}}

Perform **one** structural reduction of a boundary (singular) fit: drop every
random-effect direction whose relative-covariance diagonal `λ[d, d]` is zero, then refit
the resulting reduced model. This is the `MixedModels.jl` analogue of one level of
`cAIC4`'s `deleteZeroComponents` — the columns on the boundary are removed and the model
is re-estimated on the surviving random-effects structure.

Per reterm the surviving directions are `keep = {d : λ[d, d] ≠ 0}`:
- a *partial* drop (some but not all directions kept) column-subsets the `ReMat`
  (e.g. `(1 + x | g)` with a boundary slope → `(1 + x | g)` reduced to the intercept);
- a reterm with no surviving direction is dropped whole (e.g. `(1 | g₁) + (1 | g₂)` with
  `g₂` on the boundary → `(1 | g₁)`).

Each surviving reterm is rebuilt **fresh** (reusing the stored grouping `trm`/`refs`/
`levels` and the kept design rows, with `λ` reset to the identity for re-estimation), so
the returned model shares no mutable state with `m`. The fixed-effects term and formula
are reused; refitting uses the reterms and feterm, not the (now stale) formula, so the
reduced fit matches a native fit of the reduced model. The reduction may itself land on
the boundary — the caller iterates ([`caic`](@ref ConditionalAIC.caic) cascades until non-singular).

Returns the refitted reduced `LinearMixedModel`, or `nothing` when **every**
random-effect direction is on the boundary (no random-effects model remains — the caller
falls back to the fixed-effects-only score, mirroring `cAIC4`'s `lm` branch).
"""
function reduceboundary(m::LinearMixedModel{T}) where {T}
    reduced = _reducedreterms(m.reterms, T, "m.reterms element")
    reduced === nothing && return nothing
    mr = LinearMixedModel(response(m), m.feterm, reduced, m.formula)
    fit!(mr; progress=false)
    return mr
end

# Rebuild a single `ReMat` keeping only the random-effect directions `keep`, with `λ` reset
# to the identity (its structure — `Diagonal` for an uncorrelated term, `LowerTriangular`
# for a correlated one — preserved) so the reduced model re-estimates θ from scratch. The
# grouping (`trm`/`refs`/`levels`) and the kept design rows are reused; `inds` is the linear
# index pattern θ fills in the reduced `λ` (its lower triangle, or its diagonal).
function _subsetreterm(re::ReMat{T}, keep::Vector{Int}) where {T}
    Snew = length(keep)
    znew = re.z[keep, :]
    # A `Diagonal` λ (uncorrelated/`zerocorr` term) keeps its diagonal structure only while
    # more than one direction survives; a single-direction survivor must use the `1×1`
    # `LowerTriangular` that `MixedModels`' `ReMat{T,1}` stores (its `copyscaleinflate!`
    # reaches into `λ.data`, which `Diagonal` lacks) — see docs/math/0007 §3.
    if re.λ isa Diagonal && Snew > 1
        λnew = Diagonal{T}(I, Snew)
        indsnew = collect(range(1; step=Snew + 1, length=Snew))
    else
        λnew = LowerTriangular(Matrix{T}(I, Snew, Snew))
        indsnew = [i + (j - 1) * Snew for j in 1:Snew for i in j:Snew]
    end
    scratchnew = Matrix{T}(undef, Snew, size(re.scratch, 2))
    # The 10-arg construction below is purely positional; `znew` lands in both the `z` and
    # `wtz` slots, and several same-typed fields neighbour each other — so a field reorder in
    # a future MixedModels would be silently accepted, yielding a malformed `ReMat`. Pin the
    # exact field-name layout so any reorder/rename/addition fails loud here.
    fieldnames(ReMat) === _REMAT_FIELDS ||
        _drift("ReMat field layout", _REMAT_FIELDS, fieldnames(ReMat))
    return ReMat{T,Snew}(
        re.trm,
        re.refs,
        re.levels,
        re.cnames[keep],
        znew,
        znew,
        λnew,
        indsnew,
        adjA(re.refs, znew),
        scratchnew,
    )
end

# The surviving (non-boundary) directions of one reterm: the 1-based indices `d` whose
# relative-covariance diagonal `λ[d, d]` is off the boundary (`≠ 0`) — the `cAIC4` `theta != 0`
# direction filter shared by the boundary-reduction builders and the full-singularity query. An
# empty result marks a reterm with every direction on the boundary. `driftlabel` names the
# reterm source for the drift error.
function _keptdirections(re, driftlabel::AbstractString)
    re isa ReMat || _drift(driftlabel, "ReMat", re)
    S = size(re.λ, 1)
    return [d for d in 1:S if re.λ[d, d] != 0]
end

# One structural reduction of a reterm collection: drop every boundary direction from each
# reterm and rebuild the survivors fresh (via [`_subsetreterm`]), returning the reduced reterm
# vector — or `nothing` when no direction survives anywhere (the model collapses to
# fixed-effects only). The shared core of both [`reduceboundary`] methods (the Gaussian fit and
# the GLMM's working LMM); `driftlabel` names the reterm source for the drift error.
function _reducedreterms(reterms, ::Type{T}, driftlabel::AbstractString) where {T}
    reduced = AbstractReMat{T}[]
    for re in reterms
        keep = _keptdirections(re, driftlabel)
        isempty(keep) && continue
        push!(reduced, _subsetreterm(re, keep))
    end
    return isempty(reduced) ? nothing : reduced
end

"""
    bhessian(m::LinearMixedModel{T}, source::Symbol) -> Matrix{T}

The `s×s` numeric Hessian **B** of the (restricted) profile objective with respect to the
variance parameters θ, evaluated at the fitted θ̂, on the **deviance scale** (−2·profile
log-likelihood for ML, the REML criterion for REML) — the scale `cAIC4`'s `analytic = FALSE`
path consumes. `s = length(m.θ)`. Dispatches on `source`:

- `:finitediff` — **self-driven** finite differences over `MixedModels`' *stable*
  `objective!`/`setθ!`/`updateL!` API, **not** `MixedModelsFiniteDiffExt`.
  `objective!(m, θ)` mutates `m`, so `FiniteDiff` leaves it parked at its last probe; the
  driver restores θ̂ in a `finally` and **fails loud** if the restoration did not take — a
  Hessian computed against a silently-mutated fit is a defect.
- `:forwarddiff` — rides the **experimental** `MixedModelsForwardDiffExt`
  (`ForwardDiff.hessian(m)`), the only B-source on experimental surface. It
  differentiates a *frozen-σ* deviance, so it diverges from `:finitediff` by the σ-freezing;
  the result type is shape-asserted against the `=5.5.1` pin.

The `s×s` result is shape-asserted: the experimental AD surface may change which parameters
are differentiated alongside θ, which would silently alter B's dimension — the assertion
turns that drift into a loud error against the pinned version.

# Throws
- `ArgumentError` for a `source` other than `:finitediff` / `:forwarddiff`.
- `ErrorException` if the finite-difference driver leaves `m` perturbed, or if the Hessian's
  shape drifts from `s×s`.
"""
function bhessian(m::LinearMixedModel{T}, source::Symbol) where {T}
    s = length(m.θ)
    H = if source === :finitediff
        _bhessian_finitediff(m)
    elseif source === :forwarddiff
        # MixedModelsForwardDiffExt — the only experimental-surface touchpoint (ADR-0002).
        # Out-of-place (it copies A/L/reterms), so it does not mutate `m`.
        ForwardDiff.hessian(m)
    else
        throw(
            ArgumentError(
                "bhessian source must be :finitediff or :forwarddiff; got :$(source)"
            ),
        )
    end
    size(H) == (s, s) || error(
        "bhessian(:$source) produced a $(size(H)) Hessian; expected $s×$s (s = length(θ̂)). \
         The experimental MixedModels AD surface may have changed which parameters are \
         differentiated — reconcile against the pinned MixedModels v$PINNED_VERSION.",
    )
    return H::Matrix{T}
end

# Self-driven finite differences over the stable in-place objective (ADR-0002). The curried
# `objective!(m)` (= `Base.Fix1(objective!, m)`) re-profiles σ²(θ) at every probe — so this
# is the *profiled*-deviance Hessian `lme4`/`cAIC4` use — but it also mutates `m`. Restore
# θ̂ in the `finally`, then assert it took: never return a Hessian against a mutated fit.
function _bhessian_finitediff(m::LinearMixedModel{T}) where {T}
    θ̂ = copy(m.θ)
    H = try
        finite_difference_hessian(objective!(m), θ̂)   # Symmetric{T}; materialised below
    finally
        updateL!(setθ!(m, θ̂))
    end
    m.θ == θ̂ || error(
        "the :finitediff B-source left the model perturbed (m.θ = $(m.θ) ≠ θ̂ = $θ̂); \
         refusing to return a Hessian computed against a silently-mutated fit."
    )
    return Matrix{T}(H)
end

"""
    bootstrapfit(m::LinearMixedModel{T}, y_star::Vector{T}) -> Vector{T}

Fit a fresh `LinearMixedModel` to the bootstrap response `y_star` (same design as `m`,
covariance parameters re-estimated from scratch) and return the conditional fitted mean
`ŷ* = Xβ̂* + Zb̂*` of the new fit. The REML flag of `m` is preserved so the bootstrap
objective matches the original.

Used by the `:bootstrap` df path in [`caic`](@ref ConditionalAIC.caic): each bootstrap draw refits
with full θ re-estimation, so the covariance penalty captures the estimation-uncertainty
correction (not just the naive ρ₀).

# Arguments
- `m`: the original fitted `LinearMixedModel`; supplies the design (`feterm`, `reterms`,
  `formula`) and REML flag.
- `y_star`: a bootstrap response vector of length `n = length(response(m))`.

# Returns
- `Vector{T}` — the conditional fitted mean of the bootstrap fit.

# Throws
- `ArgumentError` if `length(y_star) ≠ n`.
"""
function bootstrapfit(m::LinearMixedModel{T}, y_star::Vector{T}) where {T}
    length(y_star) == length(response(m)) ||
        throw(ArgumentError("y_star length $(length(y_star)) ≠ n = $(length(response(m)))"))
    fresh_reterms = AbstractReMat{T}[
        _subsetreterm(re, collect(1:size(re.λ, 1))) for re in m.reterms
    ]
    mb = LinearMixedModel(y_star, m.feterm, fresh_reterms, m.formula)
    mb.optsum.REML = reml(m)   # mutating write — propagate the original's objective (table)
    fit!(mb; progress=false)
    return conditionalmean(mb)
end

# ── GLMM singularity (M3) ─────────────────────────────────────────────────────

"""
    glmmisfullysingular(m::GeneralizedLinearMixedModel) -> Bool

Whether **every** random-effect variance direction in the GLMM is on the boundary
(`λ[d, d] = 0` for all `d` in every reterm of the working LMM `m.LMM`). This is the
GLMM analogue of `reduceboundary(m.LMM) === nothing` for the Gaussian path: when
fully singular, the GLMM collapses to a plain GLM and the cAIC df is `rank(X)` with
no σ-penalty.

Returns `false` for partial singularity (some but not all directions on the boundary)
or for a non-singular fit — those cases are handled by the general GLMM influence paths
(not yet implemented).
"""
function glmmisfullysingular(m::GeneralizedLinearMixedModel)
    for re in m.LMM.reterms
        isempty(_keptdirections(re, "m.LMM.reterms element")) || return false
    end
    return true
end

"""
    reduceboundary(m::GeneralizedLinearMixedModel{T,D})
        -> Union{Nothing,GeneralizedLinearMixedModel{T,D}}

Perform **one** structural reduction of a boundary (singular) GLMM fit: drop every
random-effect direction whose relative-covariance diagonal `λ[d, d]` is zero, rebuild the
reduced GLMM, and refit it under the Laplace approximation (`fast=false, nAGQ=1`). This is
the `MixedModels.jl` analogue of one level of `cAIC4`'s `deleteZeroComponents.merMod` for a
`glmerMod` — the same single drop-and-refit operation as the Gaussian
[`reduceboundary(::LinearMixedModel)`](@ref), with the reduced *working* `LinearMixedModel`
re-wrapped in a `GeneralizedLinearMixedModel` so the refit maximises the GLMM likelihood.

Per reterm the surviving directions are `keep = {d : λ[d, d] ≠ 0}`: a *partial* drop
column-subsets the `ReMat` (e.g. `zerocorr(1 + x | g)` with a boundary slope → `(1 | g)`),
while a reterm with no surviving direction is dropped whole. The fixed-effects term `feterm`,
family/link (`m.resp`), response `y`, and prior weights are reused; an unweighted GLMM is
given an explicit length-`n` ones weight vector (matching `MixedModels`' own GLMM
constructor, whose empty `sqrtwts` would otherwise produce a different working response).
Each surviving reterm is rebuilt **fresh** (via [`_subsetreterm`], `λ` reset to identity), so
the returned model shares no mutable state with `m`.

The reduced fit may itself land on the boundary — the caller iterates ([`caic`](@ref ConditionalAIC.caic)
cascades until non-singular). Returns `nothing` when **every** random-effect direction is on
the boundary (no random-effects model remains — the caller falls back to the
fixed-effects-only score ρ = rank(X)), mirroring
the `LinearMixedModel` method and `cAIC4`'s `glm` branch.
"""
function reduceboundary(m::GeneralizedLinearMixedModel{T,D}) where {T,D}
    lmm = m.LMM
    reduced = _reducedreterms(lmm.reterms, T, "m.LMM.reterms element")
    reduced === nothing && return nothing

    # The 14-arg construction below is purely positional, and eight fields share
    # `Vector{T}` (β, β₀, η, wt, devc, devc0, sd, mult) while three share
    # `Vector{Matrix{T}}` (b, u, u₀) — so a field *reorder* among same-typed fields in
    # a future MixedModels would be silently accepted by the parametric constructor,
    # yielding a wrong reduced model with no error. Pin the exact field-name layout so
    # any reorder/rename/addition/removal fails loud here rather than downstream.
    fieldnames(GeneralizedLinearMixedModel) === _GLMM_FIELDS || _drift(
        "GeneralizedLinearMixedModel field layout",
        _GLMM_FIELDS,
        fieldnames(GeneralizedLinearMixedModel),
    )

    y = glmmresponse(m)                       # m.resp.y, shape-asserted to Vector{T}
    β = m.β
    β isa Vector{T} || _drift("m.β", Vector{T}, β)
    wt = m.wt
    wt isa Vector{T} || _drift("m.wt", Vector{T}, wt)
    lwts = isempty(wt) ? fill(one(T), length(y)) : copy(wt)

    rlmm = LinearMixedModel(copy(y), lmm.feterm, reduced, lmm.formula, lwts)
    u = [fill(zero(T), vsize(t), nlevs(t)) for t in rlmm.reterms]
    vv = length(u) == 1 ? vec(first(u)) : similar(y, 0)
    mr = GeneralizedLinearMixedModel{T,D}(
        rlmm,
        copy(β),
        copy(β),
        rlmm.θ,
        copy.(u),
        u,
        zero.(u),
        deepcopy(m.resp),
        similar(y),
        copy(wt),
        similar(vv),
        similar(vv),
        similar(vv),
        similar(vv),
    )
    # Sanity-check that the positional args landed in dimensionally coherent fields: the
    # fixed-effects vector is length rank(X) and the per-observation buffers are length n.
    # A reorder that swapped a length-p field with a length-n one (undetectable by the
    # name check if upstream also renamed) surfaces here before the refit consumes it.
    (length(mr.β) == length(β) && length(mr.η) == length(y)) || _drift(
        "reduced GLMM dimensions (β: $(length(mr.β))≠$(length(β)), η: \
         $(length(mr.η))≠$(length(y)))",
        "β length $(length(β)), η length $(length(y))",
        mr,
    )
    fit!(mr; fast=false, nAGQ=1, progress=false)
    return mr
end

# ── GLMM accessors (M3) ────────────────────────────────────────────────────────

"""
    glmmlinpred(m::GeneralizedLinearMixedModel{T}) -> Vector{T}

The linear predictor `η` (`m.η`, an alias for `m.resp.eta`) — the `n`-vector on the
link scale satisfying `g(μ) = η` where `g` is the link function and `μ` is the fitted
mean. Enters the GLMM conditional log-likelihood and the bias-correction routines.
"""
function glmmlinpred(m::GeneralizedLinearMixedModel{T}) where {T}
    η = m.η
    η isa Vector{T} || _drift("m.η", Vector{T}, η)
    return η
end

"""
    glmmfittedmu(m::GeneralizedLinearMixedModel{T}) -> Vector{T}

The fitted mean `μ` on the response scale (`m.resp.mu`) — the `n`-vector satisfying
`μ = g⁻¹(η)` where `g` is the link function. For a binomial GLMM, `μ` is a vector of
probabilities; for a Poisson GLMM, the conditional Poisson rates. Used for conditional
log-likelihood evaluation and as the return value of [`bootstrapglmmfit`](@ref).
"""
function glmmfittedmu(m::GeneralizedLinearMixedModel{T}) where {T}
    μ = m.resp.mu
    μ isa Vector{T} || _drift("m.resp.mu", Vector{T}, μ)
    return μ
end

"""
    glmmresponse(m::GeneralizedLinearMixedModel{T}) -> Vector{T}

The response vector `y` (`m.resp.y`) — the `n`-vector of observed values on the μ
scale (proportions for binomial-with-weights, raw counts for Poisson, etc.). Used to
feed the refit loop in [`bootstrapglmmfit`](@ref).
"""
function glmmresponse(m::GeneralizedLinearMixedModel{T}) where {T}
    y = m.resp.y
    y isa Vector{T} || _drift("m.resp.y", Vector{T}, y)
    return y
end

"""
    glmmdist(m::GeneralizedLinearMixedModel{T, D}) -> D

The GLM distribution family `D` (`m.resp.d`) — the distribution type parameter of the
`GeneralizedLinearMixedModel{T, D}`. Dispatches the conditional log-likelihood
(`loglik.jl`) and the bootstrap draw in [`bootstrapglmmfit`](@ref) to the correct
density/sampler.
"""
function glmmdist(m::GeneralizedLinearMixedModel{T,D}) where {T,D}
    d = m.resp.d
    d isa D || _drift("m.resp.d", D, d)
    return d::D
end

"""
    glmmfixedefrank(m::GeneralizedLinearMixedModel) -> Int

The rank `p` of the fixed-effects design matrix in the working linear mixed model
(`m.LMM.feterm.rank`). Used as the full-singularity fallback: when every random-effect
direction collapses, the working LMM's `p` enters the fixed-effects-only cAIC score.
"""
function glmmfixedefrank(m::GeneralizedLinearMixedModel)
    p = m.LMM.feterm.rank
    p isa Int || _drift("m.LMM.feterm.rank", Int, p)
    return p
end

"""
    bootstrapglmmfit(m::GeneralizedLinearMixedModel{T}, y_star::Vector{T}) -> Vector{T}

Refit a deep copy of the GLMM `m` to the bootstrap response `y_star` (same design and
distribution family as `m`, variance parameters re-estimated from scratch via
`refit!`) and return the conditional fitted mean `μ* = g⁻¹(η*)` of the new fit. The
original model `m` is not mutated.

# Arguments
- `m`: the original fitted `GeneralizedLinearMixedModel`; supplies the design, link, and
  distribution.
- `y_star`: a bootstrap response vector of length `n = length(glmmresponse(m))`, on the
  same scale as `m.resp.y` (proportions for binomial-with-weights, counts for Poisson,
  etc.).

# Returns
- `Vector{T}` — the conditional fitted mean of the bootstrap fit.

# Throws
- `ArgumentError` if `length(y_star) ≠ n`.
"""
function bootstrapglmmfit(m::GeneralizedLinearMixedModel{T}, y_star::Vector{T}) where {T}
    n = length(m.resp.y)
    length(y_star) == n || throw(ArgumentError("y_star length $(length(y_star)) ≠ n = $n"))
    m_copy = deepcopy(m)
    refit!(m_copy, y_star; progress=false)
    return glmmfittedmu(m_copy)
end

"""
    refitglmm_eta(m::GeneralizedLinearMixedModel{T}, y_new::Vector{T}) -> Vector{T}

Refit a deep copy of the GLMM `m` to the response `y_new` and return the linear
predictor `η̂` of the refitted model. The original model `m` is not mutated.

Used by the Poisson Chen–Stein refit loop (`DofGLMM.dof_glmm_poisson`): for each
nonzero observation `i`, the response is decremented (`yᵢ − 1`) and this function
returns the new `η̂` so the caller can extract the `i`-th component.

Distinct from [`bootstrapglmmfit`](@ref) which returns the fitted mean `μ̂`; the
Chen–Stein formula requires the link-scale `η̂`.

# Arguments
- `m`: the original fitted `GeneralizedLinearMixedModel`.
- `y_new`: new response vector, length `n = length(glmmresponse(m))`.

# Returns
- `Vector{T}` — the linear predictor `η̂ = Xβ̂ + Zb̂` of the refitted model.

# Throws
- `ArgumentError` if `length(y_new) ≠ n`.
"""
function refitglmm_eta(m::GeneralizedLinearMixedModel{T}, y_new::Vector{T}) where {T}
    n = length(m.resp.y)
    length(y_new) == n || throw(ArgumentError("y_new length $(length(y_new)) ≠ n = $n"))
    m_copy = deepcopy(m)
    refit!(m_copy, y_new; progress=false)
    return glmmlinpred(m_copy)
end

"""
    bernoulliflipmu(m::GeneralizedLinearMixedModel{T}) -> Vector{T}

For each observation `i`, refit the model on the response with `yᵢ → 1 − yᵢ` (all
other entries unchanged) and return the `i`-th fitted mean of that refit.

One deepcopy of `m` is made as a working buffer; `n` sequential `refit!` calls are
performed on it. The original `m` is not mutated.

This is the refit loop underlying `DofGLMM.dof_glmm_bernoulli` / `cAIC4`'s
`biasCorrectionBernoulli` (`R/biasCorrectionBernoulli.R:19–21`).

# Returns
- `Vector{T}` — length `n`; entry `i` is `μ̂ᵢ^{flip}` (the fitted mean at position `i`
  after the `i`-th label flip), in `(0, 1)`.
"""
function bernoulliflipmu(m::GeneralizedLinearMixedModel{T}) where {T}
    y = glmmresponse(m)
    n = length(y)
    m_work = deepcopy(m)
    y_work = copy(y)
    μ_flip = Vector{T}(undef, n)
    for i in 1:n
        y_work[i] = one(T) - y_work[i]
        refit!(m_work, y_work; progress=false)
        μ_flip[i] = glmmfittedmu(m_work)[i]
        y_work[i] = one(T) - y_work[i]
    end
    return μ_flip
end

"""
    glmmpriorweights(m::GeneralizedLinearMixedModel{T}) -> Vector{T}

The prior-weights vector `m.resp.wts` — the per-observation binomial denominators for
a GLMM fitted with `weights=`. Empty (`T[]`) for unweighted fits (Poisson, Bernoulli);
non-empty (`T[n₁, …, nₙ]`) for binomial-with-counts fits.

Used by [`glmmconddraw`](@ref) to reconstruct the per-observation `Binomial(nᵢ, μ̂ᵢ)`
distribution for conditional bootstrap draws.
"""
function glmmpriorweights(m::GeneralizedLinearMixedModel{T}) where {T}
    wts = m.resp.wts
    wts isa Vector{T} || _drift("m.resp.wts", Vector{T}, wts)
    return wts
end

"""
    glmmconddraw(rng::AbstractRNG, m::GeneralizedLinearMixedModel{T}, B::Int) -> Matrix{T}

Draw `B` conditional bootstrap samples from the GLMM response distribution, holding the
random effects fixed at their estimated values `b̂` (i.e. using the fitted `μ̂`). Returns
an `n × B` matrix whose `b`-th column is the `b`-th bootstrap response vector.

Draws directly from `f(μ̂ᵢ)`:
- **Poisson:** `yᵢ^{(b)} = rand(Poisson(μ̂ᵢ))` (float count)
- **Binomial:** `yᵢ^{(b)} = rand(Binomial(nᵢ, μ̂ᵢ)) / nᵢ` (proportion); `nᵢ` from
  [`glmmpriorweights`](@ref).
- **Bernoulli:** `yᵢ^{(b)} = rand(Bernoulli(μ̂ᵢ))` (0.0 or 1.0)

Unsupported families (free-dispersion etc.) raise `ArgumentError`.

# Throws
- `ArgumentError` for unsupported distribution families.
- `ArgumentError` if the Binomial model has no prior weights.
"""
function glmmconddraw(rng::AbstractRNG, m::GeneralizedLinearMixedModel{T}, B::Int) where {T}
    μ = glmmfittedmu(m)
    n = length(μ)
    Ystar = Matrix{T}(undef, n, B)
    _fill_conddraw!(rng, Ystar, μ, glmmdist(m), m)
    return Ystar
end

function _fill_conddraw!(
    rng::AbstractRNG, Ystar::Matrix{T}, μ::Vector{T}, ::Poisson, _m
) where {T}
    n, B = size(Ystar)
    for b in 1:B, i in 1:n
        Ystar[i, b] = T(rand(rng, Poisson(μ[i])))
    end
    return Ystar
end

function _fill_conddraw!(
    rng::AbstractRNG, Ystar::Matrix{T}, μ::Vector{T}, ::Bernoulli, _m
) where {T}
    n, B = size(Ystar)
    for b in 1:B, i in 1:n
        Ystar[i, b] = T(rand(rng, Bernoulli(μ[i])))
    end
    return Ystar
end

function _fill_conddraw!(
    rng::AbstractRNG, Ystar::Matrix{T}, μ::Vector{T}, ::Binomial, m
) where {T}
    wts = glmmpriorweights(m)
    isempty(wts) && throw(
        ArgumentError(
            "glmmconddraw: conditional bootstrap for Binomial GLMM requires prior weights " *
            "(number of trials per observation). Refit the model with `weights=ntrials`.",
        ),
    )
    n, B = size(Ystar)
    for b in 1:B, i in 1:n
        ni = Int(wts[i])
        Ystar[i, b] = T(rand(rng, Binomial(ni, μ[i]))) / T(ni)
    end
    return Ystar
end

function _fill_conddraw!(rng, Ystar, μ, d, _m)
    throw(
        ArgumentError(
            "glmmconddraw: family $(typeof(d)) is not supported by the conditional " *
            "bootstrap. Supported: Poisson (log link), Bernoulli (logit link), Binomial " *
            "(logit link, with prior weights). Free-dispersion families are outside M3 " *
            "scope — matches cAIC4's \"not yet supported\" warning.",
        ),
    )
end

# ── RE-structure interpretation (M4) ───────────────────────────────────────────

"""
    reterminfo(m::MixedModel) -> Vector{Tuple{Symbol,Vector{String},Bool}}

Interpret the random-effects terms of `m.formula` (the structural truth) into the raw
pieces of the `cAIC4` `cnms`-analogue `RESpec` — one entry `(grouping, directions,
correlated)` per RE term, in formula order:

- `grouping` — the grouping-factor symbol (`ret.rhs.sym` of the `RandomEffectsTerm`);
- `directions` — the `cnms`-style column labels: `"(Intercept)"` for a random intercept
  (`InterceptTerm{true}`), then each slope's variable name. A suppressed intercept
  (`InterceptTerm{false}`, the `0 + …` form) contributes no entry;
- `correlated` — `false` for a `zerocorr(… | g)` term (`MixedModels.ZeroCorr`), `true` for
  a plain `(… | g)` (`RandomEffectsTerm`).

This is the **quarantine** read of MixedModels'/StatsModels' term representation
(`RandomEffectsTerm`, `MixedModels.ZeroCorr`, `MatrixTerm`, `InterceptTerm`, `termvars`):
the wrapping into `RESpec`/`REGroup` is fit-independent and lives outside the quarantine
(`src/respec.jl`). Every touched type/field is shape-asserted against the pinned version.

# Throws
- `ErrorException` (drift) for an unexpected `rhs` element type, a non-`Symbol` grouping, or
  a slope direction that is not a single-variable term.
"""
function reterminfo(m::MixedModel)
    rhs = m.formula.rhs
    rhs isa Tuple || _drift("m.formula.rhs", "Tuple", rhs)
    return _reterminfo(rhs)
end

"""
    reterminfo(keep::FormulaTerm, data) -> Vector{Tuple{Symbol,Vector{String},Bool}}

Interpret the random-effects terms of a **schema-less** `keep` formula fragment (the `cAIC4`
`keep\$random` analogue) into the same `(grouping, directions, correlated)` tuples as the model
method. `keep` is first run through `MixedModels.schematize(keep, data, …)` — the same schema
application `fit(MixedModel, …)` uses — so its `|` `FunctionTerm` bars become the
`RandomEffectsTerm`/`ZeroCorr` terms `reterminfo` reads. The formula's left-hand side and any
fixed-effects terms are ignored; only the RE structure is extracted.
"""
function reterminfo(keep::SM.FormulaTerm, data)
    sf = MixedModels.schematize(keep, data, Dict{Symbol,Any}())
    # A fixed-only RHS schematizes to a bare `MatrixTerm` (not a tuple); normalise so the
    # no-random-effects case flows to an empty info list (and `extractkeep`'s `ArgumentError`).
    rhs = sf.rhs isa Tuple ? sf.rhs : (sf.rhs,)
    return _reterminfo(rhs)
end

# Shared RE-term reader for both `reterminfo` methods: walk a schema-applied formula RHS tuple,
# skip the fixed-effects `MatrixTerm`, and read each `RandomEffectsTerm`/`ZeroCorr` into a
# `(grouping, directions, correlated)` tuple.
function _reterminfo(rhs::Tuple)
    info = Tuple{Symbol,Vector{String},Bool}[]
    for t in rhs
        if t isa SM.MatrixTerm
            continue                                  # the fixed-effects part
        elseif t isa MixedModels.ZeroCorr
            push!(info, _reterm_entry(t.term, false))
        elseif t isa RandomEffectsTerm
            push!(info, _reterm_entry(t, true))
        else
            _drift("formula RHS element", "MatrixTerm/RandomEffectsTerm/ZeroCorr", t)
        end
    end
    return info
end

# Read one `RandomEffectsTerm` into `(grouping, directions, correlated)`. `ret.rhs` is the
# grouping `CategoricalTerm` (its `.sym` is the factor name); `ret.lhs` is the directions
# `MatrixTerm` (or a bare term for a single direction).
function _reterm_entry(ret::RandomEffectsTerm, correlated::Bool)
    grouping = ret.rhs.sym
    grouping isa Symbol || _drift("RE term grouping `.rhs.sym`", Symbol, grouping)
    terms = ret.lhs isa SM.MatrixTerm ? ret.lhs.terms : (ret.lhs,)
    directions = String[]
    for d in terms
        if d isa SM.InterceptTerm{true}
            push!(directions, "(Intercept)")
        elseif d isa SM.InterceptTerm{false}
            # suppressed intercept (`0 + …`): contributes no `cnms` column
        else
            vars = SM.termvars(d)
            length(vars) == 1 ||
                _drift("RE slope direction `termvars`", "a single-variable term", d)
            push!(directions, string(only(vars)))
        end
    end
    return (grouping, directions, correlated)
end

"""
    responseterm(m::MixedModel)

The response (left-hand-side) term of `m.formula` (`m.formula.lhs`) — supplied to
[`render`](@ref ConditionalAIC.render) as the `lhs` of the rebuilt formula. Read from the structural
truth `m.formula` (quarantine), no shape assertion: any `StatsModels` term is a valid `lhs`.
"""
responseterm(m::MixedModel) = m.formula.lhs

"""
    fixedterm(m::MixedModel) -> StatsModels.MatrixTerm

The fixed-effects `MatrixTerm` of `m.formula` (the leading `m.formula.rhs[1]`) — reattached
unchanged by [`render`](@ref ConditionalAIC.render) (the fixed part is held constant across `stepcaic`
candidates). Shape-asserted to be a `MatrixTerm`.
"""
function fixedterm(m::MixedModel)
    fe = m.formula.rhs[1]
    fe isa SM.MatrixTerm || _drift("m.formula.rhs[1]", "MatrixTerm", fe)
    return fe
end

end # module MMInternals
