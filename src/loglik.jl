"""
    cAIC.Loglik

**Conditional log-likelihoods** — the first term of the conditional AIC
(`cAIC = −2 ℓ + 2 ρ`). Pure functions of the extracted quantities `(y, ŷ)`, with **no
`MixedModels` dependency**, importable and testable in isolation and generic over
`T <: AbstractFloat`.

| Function | Family | Signature |
|:---------|:-------|:----------|
| [`condloglik`](@ref) | Gaussian | `(y, ŷ, σ̂)` |
| [`condloglik_poisson`](@ref) | Poisson (log link) | `(y, μ̂)` |
| [`condloglik_bernoulli`](@ref) | Bernoulli (logit link) | `(y, μ̂)` |

All estimands are recorded in `docs/math/0003-conditional-loglik.md` (Gaussian) and
`docs/math/0006-glmm-bias-correction.md §1` (Poisson, Bernoulli); each is the `cAIC.jl`
analogue of `cAIC4`'s `getcondLL`.
"""
module Loglik

using LogExpFunctions: xlogy, xlog1py
using SpecialFunctions: loggamma

"""
    condloglik(y::AbstractVector, yhat::AbstractVector, sigma::Real) -> eltype

Gaussian **conditional log-likelihood** `ℓ(y | b̂, β̂, θ̂)` — the first term of the
conditional AIC (`cAIC = −2 ℓ + 2 ρ`) — of a response `y` about the conditional fitted
mean `ŷ = X β̂ + Z b̂` (`yhat`) with residual standard deviation `σ̂` (`sigma`).

Conditional on the predicted random effects, `y | b̂, β̂ ~ N(ŷ, σ̂² Iₙ)`, so the
log-likelihood is the sum of independent univariate Gaussian log-densities

```math
ℓ = Σᵢ \\log φ(yᵢ; ŷᵢ, σ̂²)
  = -\\tfrac{n}{2} \\log(2π) - n \\log σ̂ - \\frac{1}{2 σ̂²} Σᵢ (yᵢ - ŷᵢ)².
```

This is the `cAIC.jl` analogue of `cAIC4`'s `getcondLL` (which evaluates
`sum(dnorm(y, fitted, sigma, log = TRUE))`); the conditional covariance `σ̂² Iₙ` is the
unweighted residual covariance (all residual weights 1). It is computed in the stable
log-space form of CLAUDE.md §9 — densities enter as `log φ`, no explicit inverse and no
determinant are formed (the diagonal `σ̂² Iₙ` collapses `logdet`/`invquad` to scalars),
and `Σᵢ (yᵢ - ŷᵢ)²` is accumulated without materialising `y - ŷ`. The estimand is recorded
in `docs/math/0003-conditional-loglik.md`.

# Arguments
- `y`: the response, length `n`.
- `yhat`: the conditional fitted mean `ŷ`, length `n`.
- `sigma`: the residual **standard deviation** `σ̂ > 0` (e.g. `MixedModels.sigma(m)`), not
  the variance.

# Returns
- The scalar `ℓ`, in the promoted floating element type of `y`, `yhat`, and `sigma` (a
  `Float32` input yields a `Float32`). An empty `y` gives the empty sum `0`; a perfect fit
  (`ŷ = y`) gives the finite maximum `−(n/2) log(2π) − n log σ̂`; non-finite data
  propagates.

# Throws
- `DomainError` if `sigma ≤ 0` (or `NaN`): `σ̂` is a standard deviation.
- `DimensionMismatch` if `y` and `yhat` do not index alike.

# Example
```jldoctest
julia> cAIC.Loglik.condloglik([0.0], [0.0], 1.0)   # perfect fit, n = 1, σ̂ = 1
-0.9189385332046727
```
"""
function condloglik(y::AbstractVector, yhat::AbstractVector, sigma::Real)
    sigma > 0 || throw(
        DomainError(sigma, "condloglik requires a positive residual standard deviation σ̂"),
    )
    T = float(promote_type(eltype(y), eltype(yhat), typeof(sigma)))
    ss = zero(T)
    # `eachindex(y, yhat)` throws `DimensionMismatch` unless the axes match.
    @inbounds for i in eachindex(y, yhat)
        d = T(y[i]) - T(yhat[i])
        ss += d * d
    end
    n = T(length(y))
    σ = T(sigma)
    return -(n / 2) * log(2 * T(π)) - n * log(σ) - ss / (2 * σ * σ)
end

"""
    condloglik_poisson(y::AbstractVector, μhat::AbstractVector) -> eltype

Poisson **conditional log-likelihood** `ℓ(y | b̂, β̂, θ̂)` — the GLMM analogue of
`cAIC4`'s `getcondLL` for the Poisson family (`dpois(y, lambda = μ̂, log = TRUE)`).

Conditional on the predicted random effects, each `yᵢ | b̂, β̂ ~ Poisson(μ̂ᵢ)`, so the
log-likelihood is the sum of independent Poisson log-densities

```math
ℓ = Σᵢ [yᵢ \\log μ̂ᵢ - μ̂ᵢ - \\log(yᵢ!)] ,
```

where `log(yᵢ!) = loggamma(yᵢ + 1)`. The `log(yᵢ!)` constant is kept so the absolute
cAIC value matches `cAIC4`. `xlogy(0, μ̂ᵢ) = 0` handles `yᵢ = 0` without NaN.
The estimand is specified in `docs/math/0006-glmm-bias-correction.md §1`.

# Arguments
- `y`: the count response, length `n` (non-negative reals; in practice, integer counts).
- `μhat`: the conditional Poisson rate `μ̂`, length `n` (strictly positive).

# Returns
- The scalar `ℓ`, in the promoted floating element type of `y` and `μhat` (a `Float32`
  input yields a `Float32`). An empty `y` gives `0`; non-finite counts propagate.

# Throws
- `DomainError` if any `μhat[i] ≤ 0`: Poisson rates must be strictly positive.
- `DimensionMismatch` if `y` and `μhat` do not index alike.

# Example
```jldoctest
julia> cAIC.Loglik.condloglik_poisson([1.0], [1.0])   # y=1, μ̂=1 → 0 − 1 − 0 = −1
-1.0
```
"""
function condloglik_poisson(y::AbstractVector, μhat::AbstractVector)
    all(>(0), μhat) ||
        throw(DomainError(μhat, "condloglik_poisson requires all Poisson rates μ̂ᵢ > 0"))
    T = float(promote_type(eltype(y), eltype(μhat)))
    ℓ = zero(T)
    @inbounds for i in eachindex(y, μhat)
        yi, μi = T(y[i]), T(μhat[i])
        ℓ += xlogy(yi, μi) - μi - loggamma(yi + one(T))
    end
    return ℓ
end

"""
    condloglik_bernoulli(y::AbstractVector, μhat::AbstractVector) -> eltype

Bernoulli **conditional log-likelihood** `ℓ(y | b̂, β̂, θ̂)` — the GLMM analogue of
`cAIC4`'s `getcondLL` for the Bernoulli/binomial family (`dbinom(y, size=1, prob=μ̂, log=TRUE)`).

Conditional on the predicted random effects, each `yᵢ | b̂, β̂ ~ Bernoulli(μ̂ᵢ)`, so the
log-likelihood is the sum of independent Bernoulli log-densities

```math
ℓ = Σᵢ [yᵢ \\log μ̂ᵢ + (1 - yᵢ) \\log(1 - μ̂ᵢ)] .
```

`xlogy` and `xlog1py` handle the `yᵢ ∈ {0,1}` boundary cases without NaN
(`xlogy(0, μ̂) = 0`, `xlog1py(0, −μ̂) = 0`). The estimand is specified in
`docs/math/0006-glmm-bias-correction.md §1`.

# Arguments
- `y`: the binary response, length `n` (values in `{0, 1}` in practice, but the formula
  accepts any real in `[0, 1]`).
- `μhat`: the conditional Bernoulli probability `μ̂`, length `n` (strictly in `(0, 1)`).

# Returns
- The scalar `ℓ`, in the promoted floating element type of `y` and `μhat` (a `Float32`
  input yields a `Float32`). An empty `y` gives `0`; non-finite responses propagate.

# Throws
- `DomainError` if any `μhat[i] ∉ (0, 1)`: Bernoulli probabilities must be strictly in
  the open unit interval.
- `DimensionMismatch` if `y` and `μhat` do not index alike.

# Example
```jldoctest
julia> cAIC.Loglik.condloglik_bernoulli([1.0], [0.5])   # y=1, μ̂=0.5 → log(0.5) = −log 2
-0.6931471805599453
```
"""
function condloglik_bernoulli(y::AbstractVector, μhat::AbstractVector)
    all(x -> 0 < x < 1, μhat) || throw(
        DomainError(
            μhat,
            "condloglik_bernoulli requires all Bernoulli probabilities μ̂ᵢ ∈ (0, 1)",
        ),
    )
    T = float(promote_type(eltype(y), eltype(μhat)))
    ℓ = zero(T)
    @inbounds for i in eachindex(y, μhat)
        yi, μi = T(y[i]), T(μhat[i])
        ℓ += xlogy(yi, μi) + xlog1py(one(T) - yi, -μi)
    end
    return ℓ
end

end # module Loglik
