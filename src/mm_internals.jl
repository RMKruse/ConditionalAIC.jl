"""
    cAIC.MMInternals

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
| `m.sigma`         | property    | [`sigmahat`]        | residual standard deviation σ̂                            |
| `ranef(m)`        | exported fn | [`bhat`]            | predicted random effects b̂ = λu, per grouping            |
| `m.X`             | field       | [`fixedeffects`]    | n×p fixed-effects design X                                |
| `response(m)`     | exported fn | [`responsevec`]     | response vector y                                        |
| `fitted(m)`       | exported fn | [`conditionalmean`] | conditional fitted mean ŷ = Xβ̂ + Zb̂                      |
| `leverage(m)`     | exported fn | [`rho0`]            | per-observation hat-matrix diagonal; ρ₀ = its sum (§2)   |
| `m.reterms`       | field       | [`retermdesigns`], [`reduceboundary`] | the per-grouping `ReMat`s                  |
| `Matrix(re)`      | constructor | [`retermdesigns`]   | dense random-effects design Z block (n×qₜ) per reterm    |
| `re.λ`            | field       | [`retermlambdas`], [`reduceboundary`] | relative covariance factor λ block (kₜ×kₜ) |
| `m.parmap`        | field       | [`parmap`]          | θ → (reterm, row, col) map — the `lme4` `Lind` analogue  |
| `issingular(m)`   | exported fn | [`issingular`]      | is a variance component on the boundary (some `λ[d,d]=0`)|
| `m.feterm`        | field       | [`reduceboundary`]  | the fixed-effects term `FeTerm`, reused by the reduced fit|
| `m.formula`       | field       | [`reduceboundary`]  | the model formula (bookkeeping for the reduced-fit ctor) |
| `re.trm/refs/levels/cnames/z/scratch` | fields | [`reduceboundary`] | `ReMat` design pieces, column-subset to rebuild a reduced `ReMat` |
| `ReMat{T,S}(…)`   | constructor | [`reduceboundary`]  | rebuild a boundary-reduced random-effects term            |
| `adjA(refs, z)`   | fn          | [`reduceboundary`]  | the `ReMat` adjoint sparse block for the subset design    |
| `LinearMixedModel(y, feterm, reterms, form)` | constructor | [`reduceboundary`] | assemble the reduced model from reused design objects |
| `fit!(m)`         | exported fn | [`reduceboundary`]  | refit the reduced model to its MLE/REML estimate          |
| `m.θ`             | property    | [`bhessian`]        | fitted θ̂ — the FD evaluation point and restoration check |
| `ForwardDiff.hessian(m)` | ext fn (experimental) | [`bhessian`] | s×s deviance Hessian (`:forwarddiff`; frozen-σ — see below) |
| `objective!(m)`   | unexported  | [`bhessian`]        | curried θ→deviance closure (`Base.Fix1`); FD driver target |
| `setθ!(m, θ)`     | unexported  | [`bhessian`]        | set variance parameters θ — restore θ̂ after FD perturbation |
| `updateL!(m)`     | unexported  | [`bhessian`]        | refactorise `L` after `setθ!` — completes the restore    |

**Experimental surface (ADR-0002).** `ForwardDiff.hessian(::LinearMixedModel)` (the
`MixedModelsForwardDiffExt` extension, used by [`bhessian`]) is the one touchpoint on
`MixedModels`' *experimental* AD surface; the docs warn that which parameters are
differentiated alongside θ may change, which would silently alter B's dimension, so
[`bhessian`] shape-asserts the `s×s` result against the `=5.5.1` pin. The companion
`FiniteDiff.finite_difference_hessian(::LinearMixedModel)` extension is **deliberately not
accessed** — the `:finitediff` source self-drives `FiniteDiff` over the stable
`objective!`/`setθ!`/`updateL!` trio instead (ADR-0002).
"""
module MMInternals

using LinearAlgebra: Diagonal, LowerTriangular, I
using MixedModels:
    LinearMixedModel,
    AbstractReMat,
    ReMat,
    ranef,
    response,
    fitted,
    leverage,
    adjA,
    fit!,
    objective!,
    setθ!,
    updateL!
using MixedModels: MixedModels
using FiniteDiff: finite_difference_hessian
using ForwardDiff: ForwardDiff

const PINNED_VERSION = "5.5.1"

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
trace of the hat matrix `y ↦ ŷ` at the fitted, fixed variance parameters
(`docs/math/0002` §2). `leverage(m)` returns the per-observation hat-matrix *diagonal*; ρ₀
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
derivative matrices `Wⱼ` (`docs/math/0002` §3, §6).
"""
function retermlambdas(m::LinearMixedModel{T}) where {T}
    return Matrix{T}[Matrix(re.λ) for re in m.reterms]
end

"""
    parmap(m::LinearMixedModel) -> Vector{NTuple{3,Int}}

The free-covariance-parameter map `m.parmap` — the `lme4` `Lind` analogue. Entry `s` is
`(t, i, j)`: the `s`-th component `θₛ` occupies position `(i, j)` of reterm `t`'s `λ`
block. This drives the `Wⱼ` derivative-pattern construction (`docs/math/0002` §6).
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
the boundary — the caller iterates ([`caic`](@ref) cascades until non-singular).

Returns the refitted reduced [`LinearMixedModel`](@ref), or `nothing` when **every**
random-effect direction is on the boundary (no random-effects model remains — the caller
falls back to the fixed-effects-only score, mirroring `cAIC4`'s `lm` branch).
"""
function reduceboundary(m::LinearMixedModel{T}) where {T}
    reduced = AbstractReMat{T}[]
    for re in m.reterms
        re isa ReMat || _drift("m.reterms element", "ReMat", re)
        S = size(re.λ, 1)
        keep = [d for d in 1:S if re.λ[d, d] != 0]
        isempty(keep) && continue
        push!(reduced, _subsetreterm(re, keep))
    end
    isempty(reduced) && return nothing
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
    if re.λ isa Diagonal
        λnew = Diagonal{T}(I, Snew)
        indsnew = collect(range(1; step=Snew + 1, length=Snew))
    else
        λnew = LowerTriangular(Matrix{T}(I, Snew, Snew))
        indsnew = [i + (j - 1) * Snew for j in 1:Snew for i in j:Snew]
    end
    scratchnew = Matrix{T}(undef, Snew, size(re.scratch, 2))
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

"""
    bhessian(m::LinearMixedModel{T}, source::Symbol) -> Matrix{T}

The `s×s` numeric Hessian **B** of the (restricted) profile objective with respect to the
variance parameters θ, evaluated at the fitted θ̂, on the **deviance scale** (−2·profile
log-likelihood for ML, the REML criterion for REML) — the scale `cAIC4`'s `analytic = FALSE`
path consumes (`docs/math/0004` §1, §3). `s = length(m.θ)`. Dispatches on `source`:

- `:finitediff` — **self-driven** finite differences over `MixedModels`' *stable*
  `objective!`/`setθ!`/`updateL!` API (ADR-0002), **not** `MixedModelsFiniteDiffExt`.
  `objective!(m, θ)` mutates `m`, so `FiniteDiff` leaves it parked at its last probe; the
  driver restores θ̂ in a `finally` and **fails loud** if the restoration did not take — a
  Hessian computed against a silently-mutated fit is a defect (`docs/math/0004` §3b).
- `:forwarddiff` — rides the **experimental** `MixedModelsForwardDiffExt`
  (`ForwardDiff.hessian(m)`), the only B-source on experimental surface (ADR-0002). It
  differentiates a *frozen-σ* deviance, so it diverges from `:finitediff` by the σ-freezing
  of `docs/math/0004` §3a; the result type is shape-asserted against the `=5.5.1` pin.

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

end # module MMInternals
