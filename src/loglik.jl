"""
    cAIC.Loglik

The **conditional log-likelihoods** — the first term of the conditional AIC
(`cAIC = −2 ℓ + 2 ρ`). Pure functions of the extracted quantities with **no `MixedModels`
dependency**, importable and testable in isolation and generic over `T <: AbstractFloat`.

- Gaussian: [`condloglik`](@ref) — a function of `(y, ŷ, σ̂)`.
- Poisson: [`condloglik_poisson`](@ref) — a function of `(y, μ̂)`.
- Bernoulli: [`condloglik_bernoulli`](@ref) — a function of `(y, μ̂)`.

The estimands and their numerically-stable forms are recorded in
`docs/math/0003-conditional-loglik.md` (Gaussian) and `0006-glmm-bias-correction.md §1`
(Poisson, Bernoulli); they are the `cAIC.jl` analogues of `cAIC4`'s `getcondLL`.
"""
module Loglik

using LogExpFunctions: xlogy

# log(k!) = log(1) + log(2) + ... + log(k), accumulated from scratch. Used by
# condloglik_poisson for the normalising constant; avoids a SpecialFunctions dependency.
function _logfactorial(yi::T) where {T<:AbstractFloat}
    k = round(Int, yi)
    r = 0.0
    for j in 2:k
        r += log(j)
    end
    return T(r)
end

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
    condloglik_poisson(y::AbstractVector, mu::AbstractVector) -> T

Poisson **conditional log-likelihood** `ℓ(y | μ̂)` — the per-family first term of the
conditional AIC for a Poisson GLMM (math spec: `0006-glmm-bias-correction.md §1`).
Evaluated at the conditional fitted mean `μ̂ = exp(η̂)` from `cAIC4`'s `getcondLL.merMod`
(`dpois(y, lambda = μ̂, log = TRUE)` summed over observations):

```math
\\ell^{\\mathrm{Pois}}(y \\mid \\hat{\\mu})
  = \\sum_{i=1}^{n} \\bigl[\\, y_i \\log \\hat{\\mu}_i - \\hat{\\mu}_i
    - \\log(y_i!) \\,\\bigr].
```

The `y_i log μ̂_i` term is computed via `xlogy` (`= 0` when `y_i = 0`) so that
zero-count observations (`y_i = 0`) contribute only `−μ̂_i`, not `NaN`.

# Arguments
- `y`: the count response, length `n`. Values must be non-negative (the Poisson support).
- `mu`: the conditional fitted mean `μ̂`, length `n`; each element must be strictly positive.

# Returns
- The scalar `ℓ`, in the promoted floating element type of `y` and `mu`.

# Throws
- `DomainError` if any `mu[i] ≤ 0`.
- `DimensionMismatch` if `y` and `mu` do not index alike.
"""
function condloglik_poisson(y::AbstractVector, mu::AbstractVector)
    T = float(promote_type(eltype(y), eltype(mu)))
    s = zero(T)
    for i in eachindex(y, mu)
        yi = T(y[i])
        μi = T(mu[i])
        μi > 0 || throw(DomainError(μi, "condloglik_poisson requires μᵢ > 0"))
        s += xlogy(yi, μi) - μi - _logfactorial(yi)
    end
    return s
end

"""
    condloglik_bernoulli(y::AbstractVector, mu::AbstractVector) -> T

Bernoulli **conditional log-likelihood** `ℓ(y | μ̂)` — the per-family first term of the
conditional AIC for a Bernoulli GLMM (math spec: `0006-glmm-bias-correction.md §1`).
Evaluated at the conditional fitted probability `μ̂ = logit⁻¹(η̂)` from `cAIC4`'s
`getcondLL.merMod` (`dbinom(y, size=1, prob=μ̂, log=TRUE)` summed over observations):

```math
\\ell^{\\mathrm{Bern}}(y \\mid \\hat{\\mu})
  = \\sum_{i=1}^{n} \\bigl[\\, y_i \\log \\hat{\\mu}_i
    + (1 - y_i)\\log(1 - \\hat{\\mu}_i) \\,\\bigr].
```

Both log terms are computed via `xlogy` so that boundary labels (`y_i ∈ {0,1}`) yield
`0` (not `NaN`) for the corresponding inactive term.

# Arguments
- `y`: the binary response, length `n`. Values in `{0, 1}`.
- `mu`: the conditional fitted probability `μ̂`, length `n`; each element must satisfy
  `0 < mu[i] < 1` (open interval — boundary probabilities yield infinite log-likelihood).

# Returns
- The scalar `ℓ`, in the promoted floating element type of `y` and `mu`.

# Throws
- `DomainError` if any `mu[i] ∉ (0, 1)`.
- `DimensionMismatch` if `y` and `mu` do not index alike.
"""
function condloglik_bernoulli(y::AbstractVector, mu::AbstractVector)
    T = float(promote_type(eltype(y), eltype(mu)))
    s = zero(T)
    for i in eachindex(y, mu)
        yi = T(y[i])
        μi = T(mu[i])
        0 < μi < 1 || throw(DomainError(μi, "condloglik_bernoulli requires μᵢ ∈ (0,1)"))
        s += xlogy(yi, μi) + xlogy(1 - yi, 1 - μi)
    end
    return s
end

end # module Loglik
