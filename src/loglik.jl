"""
    cAIC.Loglik

The Gaussian **conditional log-likelihood** — the first term of the conditional AIC
(`cAIC = −2 ℓ + 2 ρ`). A pure function of the extracted quantities `(y, ŷ, σ̂)`, with **no
`MixedModels` dependency**, importable and testable in isolation and generic over
`T <: AbstractFloat`.

The estimand and its numerically-stable form are recorded in
`docs/math/0003-conditional-loglik.md`; it is the `cAIC.jl` analogue of `cAIC4`'s
`getcondLL`.
"""
module Loglik

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

end # module Loglik
