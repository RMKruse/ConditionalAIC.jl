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
| `m.reterms`       | field       | [`retermdesigns`]   | the per-grouping `ReMat`s                                |
| `Matrix(re)`      | constructor | [`retermdesigns`]   | dense random-effects design Z block (n×qₜ) per reterm    |
| `re.λ`            | field       | [`retermlambdas`]   | relative covariance factor λ block (kₜ×kₜ) per reterm    |
| `m.parmap`        | field       | [`parmap`]          | θ → (reterm, row, col) map — the `lme4` `Lind` analogue  |
"""
module MMInternals

using MixedModels: LinearMixedModel, ranef, response, fitted, leverage

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

end # module MMInternals
