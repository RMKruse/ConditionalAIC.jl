"""
    ConditionalAIC.Loglik

**Conditional log-likelihoods** — the first term of the conditional AIC
(`cAIC = −2 ℓ + 2 ρ`). Pure functions of the extracted quantities `(y, ŷ)`, with **no
`MixedModels` dependency**, importable and testable in isolation and generic over
`T <: AbstractFloat`.

| Function | Family | Signature |
|:---------|:-------|:----------|
| [`condloglik`](@ref) | Gaussian | `(y, ŷ, σ̂)` |
| [`condloglik_poisson`](@ref) | Poisson (log link) | `(y, μ̂)` |
| [`condloglik_bernoulli`](@ref) | Bernoulli (logit link) | `(y, μ̂)` |
| [`condloglik_binomial`](@ref) | multi-trial binomial (logit link) | `(y, μ̂, n)` |

Each is the analogue of `cAIC4`'s conditional log-likelihood — except `condloglik_binomial`,
which **deviates** from `cAIC4`'s defective binomial branch (a documented `−∞` bug; see its
own docstring).
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

This is the analogue of `cAIC4`'s conditional log-likelihood (which evaluates
`sum(dnorm(y, fitted, sigma, log = TRUE))`); the conditional covariance `σ̂² Iₙ` is the
unweighted residual covariance (all residual weights 1). It is computed in the stable
log-space form — densities enter as `log φ`, no explicit inverse and no
determinant are formed (the diagonal `σ̂² Iₙ` collapses `logdet`/`invquad` to scalars),
and `Σᵢ (yᵢ - ŷᵢ)²` is accumulated without materialising `y - ŷ`.

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
julia> ConditionalAIC.Loglik.condloglik([0.0], [0.0], 1.0)   # perfect fit, n = 1, σ̂ = 1
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
`cAIC4`'s conditional log-likelihood for the Poisson family (`dpois(y, lambda = μ̂, log = TRUE)`).

Conditional on the predicted random effects, each `yᵢ | b̂, β̂ ~ Poisson(μ̂ᵢ)`, so the
log-likelihood is the sum of independent Poisson log-densities

```math
ℓ = Σᵢ [yᵢ \\log μ̂ᵢ - μ̂ᵢ - \\log(yᵢ!)] ,
```

where `log(yᵢ!) = loggamma(yᵢ + 1)`. The `log(yᵢ!)` constant is kept so the absolute
cAIC value matches `cAIC4`. `xlogy(0, μ̂ᵢ) = 0` handles `yᵢ = 0` without NaN.

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
julia> ConditionalAIC.Loglik.condloglik_poisson([1.0], [1.0])   # y=1, μ̂=1 → 0 − 1 − 0 = −1
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
`cAIC4`'s conditional log-likelihood for the Bernoulli/binomial family (`dbinom(y, size=1, prob=μ̂, log=TRUE)`).

Conditional on the predicted random effects, each `yᵢ | b̂, β̂ ~ Bernoulli(μ̂ᵢ)`, so the
log-likelihood is the sum of independent Bernoulli log-densities

```math
ℓ = Σᵢ [yᵢ \\log μ̂ᵢ + (1 - yᵢ) \\log(1 - μ̂ᵢ)] .
```

`xlogy` and `xlog1py` handle the `yᵢ ∈ {0,1}` boundary cases without NaN
(`xlogy(0, μ̂) = 0`, `xlog1py(0, −μ̂) = 0`).

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
julia> ConditionalAIC.Loglik.condloglik_bernoulli([1.0], [0.5])   # y=1, μ̂=0.5 → log(0.5) = −log 2
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

"""
    condloglik_binomial(y::AbstractVector, μhat::AbstractVector, n::AbstractVector) -> eltype

Multi-trial binomial **conditional log-likelihood** `ℓ(y | b̂, β̂, θ̂)` for a GLMM whose
response is stored as the success **proportion** `yᵢ = kᵢ/nᵢ` with per-observation trial
counts `nᵢ` (the prior weights, `m.resp.wts`).

Conditional on the predicted random effects, each success count `kᵢ = nᵢ yᵢ` satisfies
`kᵢ | b̂, β̂ ~ Binomial(nᵢ, μ̂ᵢ)`, so

```math
ℓ = Σᵢ [ \\log \\binom{nᵢ}{kᵢ} + kᵢ \\log μ̂ᵢ + (nᵢ - kᵢ) \\log(1 - μ̂ᵢ) ] ,
    \\qquad kᵢ = nᵢ yᵢ ,
```

with `log C(nᵢ, kᵢ) = loggamma(nᵢ+1) − loggamma(kᵢ+1) − loggamma(nᵢ−kᵢ+1)`. `xlogy` and
`xlog1py` handle the `kᵢ ∈ {0, nᵢ}` boundaries without NaN. For `nᵢ ≡ 1` the coefficient
vanishes and this equals [`condloglik_bernoulli`](@ref).

!!! note "Deviation from `cAIC4`"
    This is **not** the analogue of `cAIC4`'s conditional log-likelihood. `cAIC4`'s binomial
    branch (`dbinom(y, size = length(unique(y)) − 1, prob = μ̂)`) is correct only for
    Bernoulli; for multi-trial data it passes a non-integer `x` and a wrong `size`, returns
    `0`, and yields `ℓ = −∞`. `condloglik_binomial` evaluates the **correct** binomial
    density at the true trial counts `nᵢ`. The reference is base-R
    `dbinom(kᵢ, nᵢ, μ̂ᵢ, log = TRUE)`.

# Arguments
- `y`: the success-proportion response `kᵢ/nᵢ`, length `n` (each in `[0, 1]`).
- `μhat`: the conditional success probability `μ̂`, length `n` (strictly in `(0, 1)`).
- `n`: the per-observation trial counts `nᵢ`, length `n` (strictly positive). Each
  `kᵢ = nᵢ yᵢ` must be a (near-)integer in `[0, nᵢ]`.

# Returns
- The scalar `ℓ`, in the promoted floating element type of `y`, `μhat`, and `n` (a `Float32`
  input yields a `Float32`). An empty `y` gives the empty sum `0`.

# Throws
- `DomainError` if any `μhat[i] ∉ (0, 1)`, any `n[i] ≤ 0`, or any reconstructed success
  count `kᵢ = nᵢ yᵢ` is not a (near-)integer in `[0, nᵢ]`.
- `DimensionMismatch` if `y`, `μhat`, and `n` do not index alike.

# Example
```jldoctest
julia> ConditionalAIC.Loglik.condloglik_binomial([0.5], [0.5], [2.0])   # k=1, n=2, μ̂=0.5 → log 0.5
-0.6931471805599453
```
"""
function condloglik_binomial(y::AbstractVector, μhat::AbstractVector, n::AbstractVector)
    all(x -> 0 < x < 1, μhat) || throw(
        DomainError(
            μhat, "condloglik_binomial requires all binomial probabilities μ̂ᵢ ∈ (0, 1)"
        ),
    )
    all(>(0), n) ||
        throw(DomainError(n, "condloglik_binomial requires all trial counts nᵢ > 0"))
    T = float(promote_type(eltype(y), eltype(μhat), eltype(n)))
    ℓ = zero(T)
    # `eachindex(y, μhat, n)` throws `DimensionMismatch` unless all three axes match.
    @inbounds for i in eachindex(y, μhat, n)
        yi, μi, ni = T(y[i]), T(μhat[i]), T(n[i])
        # The success count kᵢ = nᵢ yᵢ is an integer in exact arithmetic; guard the float
        # round-trip and the binomial support 0 ≤ kᵢ ≤ nᵢ (fail loud — §4, no silent wrong number).
        ki = yi * ni
        kr = round(ki)
        abs(ki - kr) ≤ sqrt(eps(T)) * (one(T) + ni) || throw(
            DomainError(
                ki,
                "condloglik_binomial: nᵢ·yᵢ must be a (near-)integer success count; got $(ki) (nᵢ=$(ni), yᵢ=$(yi))",
            ),
        )
        (zero(T) ≤ kr ≤ ni) || throw(
            DomainError(
                kr, "condloglik_binomial: success count kᵢ=$(kr) outside [0, nᵢ=$(ni)]"
            ),
        )
        logcoef = loggamma(ni + one(T)) - loggamma(kr + one(T)) - loggamma(ni - kr + one(T))
        ℓ += logcoef + xlogy(kr, μi) + xlog1py(ni - kr, -μi)
    end
    return ℓ
end

end # module Loglik
