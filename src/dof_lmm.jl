"""
    ConditionalAIC.DofLMM

Greven–Kneib bias-corrected **effective degrees of freedom** ρ for a Gaussian linear
mixed model — the port of `cAIC4`'s analytic Gaussian bias correction.

This module is a *pure, fit-independent, parametrisation-neutral* map from a component
set ([`GaussianComponents`](@ref)) to the scalar ρ. It touches **no** `MixedModels`
object; it consumes dense components in `cAIC4`'s `getModelComponents.merMod` layout and
reproduces the exact arithmetic of `cAIC4`'s analytic Gaussian bias correction.

Every kernel uses the numerically-stable [`ConditionalAIC.Numerics`](@ref) primitives: the Fisher
trace term `tr(Wⱼ M Wₖ M)` is formed by `traceprod` without materialising the product,
and `Λ̂ʸ = B⁻¹C` is a factorisation-based solve with no explicit inverse.
"""
module DofLMM

using LinearAlgebra: Symmetric, cholesky, dot, issuccess, tr
using Statistics: mean

using ..Numerics: traceprod

"""
    GaussianComponents{T<:AbstractFloat}

The Gaussian-LMM bias-correction component set, in `cAIC4`'s `getModelComponents.merMod`
layout. All matrices are dense and parametrisation-neutral — this type carries *no*
`θ`-vector and no fitted model, so the correction arithmetic is tested in isolation from
any fit. Targets the **unweighted** Gaussian path (`R = Iₙ`, so `R A = A`); weighted
Gaussian is deferred, matching `cAIC4`.

The number of observations is `n = length(e)`, the number of free covariance components
is `s = length(Wlist)`, and the fixed-effects rank is `p = size(X, 2)`.

# Fields
- `X::Matrix{T}`: the `n×p` fixed-effects design. Only its column count `p` enters the
  correction (the REML degrees `nθ = n − p`); it is carried to match `cAIC4`'s `model\$X`.
- `e::Vector{T}`: the `n` conditional residual `e = y − ŷ = A y`.
- `A::Matrix{T}`: the `n×n` fixed-effects-adjusted projector
  `A = V₀⁻¹ − V₀⁻¹X(XᵀV₀⁻¹X)⁻¹XᵀV₀⁻¹`.
- `V0inv::Matrix{T}`: the `n×n` inverse scaled marginal variance `V₀⁻¹`.
- `Wlist::Vector{Matrix{T}}`: the `s` derivative matrices `Wⱼ = Z Dⱼ Zᵀ` (each `n×n`).
- `eWelist::Vector{T}`: the `s` residual quadratic forms `eᵀ Wⱼ e`.
- `tye::T`: the scalar `tʸᵉ = yᵀe = yᵀ A y`.
- `isREML::Bool`: whether the fit used REML (selects `Wⱼ A` and `nθ = n − p`) or ML
  (`Wⱼ V₀⁻¹` and `nθ = n`).
"""
struct GaussianComponents{T<:AbstractFloat}
    X::Matrix{T}
    e::Vector{T}
    A::Matrix{T}
    V0inv::Matrix{T}
    Wlist::Vector{Matrix{T}}
    eWelist::Vector{T}
    tye::T
    isREML::Bool

    # Validate that every component has a mutually-consistent shape; an inconsistent set
    # would otherwise produce a silently-wrong ρ downstream. Fail loudly.
    function GaussianComponents{T}(
        X::Matrix{T},
        e::Vector{T},
        A::Matrix{T},
        V0inv::Matrix{T},
        Wlist::Vector{Matrix{T}},
        eWelist::Vector{T},
        tye::T,
        isREML::Bool,
    ) where {T<:AbstractFloat}
        n = length(e)
        size(X, 1) == n ||
            throw(ArgumentError("size(X, 1) = $(size(X, 1)) must equal n = length(e) = $n"))
        size(A) == (n, n) ||
            throw(ArgumentError("A must be $n×$n (n = length(e)); got $(size(A))"))
        size(V0inv) == (n, n) ||
            throw(ArgumentError("V0inv must be $n×$n (n = length(e)); got $(size(V0inv))"))
        all(W -> size(W) == (n, n), Wlist) ||
            throw(ArgumentError("every Wⱼ must be $n×$n (n = length(e))"))
        length(eWelist) == length(Wlist) || throw(
            ArgumentError(
                "length(eWelist) = $(length(eWelist)) must equal length(Wlist) = $(length(Wlist))",
            ),
        )
        return new{T}(X, e, A, V0inv, Wlist, eWelist, tye, isREML)
    end
end

# Infer the element type `T` from the (concrete, mutually-consistent) field types.
function GaussianComponents(
    X::Matrix{T},
    e::Vector{T},
    A::Matrix{T},
    V0inv::Matrix{T},
    Wlist::Vector{Matrix{T}},
    eWelist::Vector{T},
    tye::T,
    isREML::Bool,
) where {T<:AbstractFloat}
    return GaussianComponents{T}(X, e, A, V0inv, Wlist, eWelist, tye, isREML)
end

# Λ̂ʸ = B⁻¹C as a factorisation-based solve — never an explicit inverse.
# B is the positive-definite negative profile-(restricted-)likelihood Hessian (doc 0002
# §5), so a Cholesky solve is the stable primary path; a symmetric (Bunch–Kaufman) solve
# is the fallback when B is not numerically positive-definite (θ near the boundary).
# Both compute the same B⁻¹C as `cAIC4`'s `solve(B) %*% C`.
function _lambday(B::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T}
    Bsym = Symmetric(B)
    fac = cholesky(Bsym; check=false)
    return issuccess(fac) ? fac \ C : Bsym \ C
end

# Shared ρ-assembly tail of the analytic and numeric Gaussian bias corrections. Given the
# solved Λ̂ʸ = B⁻¹C and the reused A Wⱼ e vectors, form
#   ρ = ρ₀ + Σⱼ Λ̂ʸ[j,:]·(A Wⱼ e) + sigmapenalty,   ρ₀ = n − tr(A)   (unweighted, R A = A).
# Both paths reach this *identical* assembly (only the source of B and the scaling of C
# differ upstream); factoring it guarantees the two penalties can never drift in the tail.
function _assemble_rho(
    A::AbstractMatrix{T},
    Λy::AbstractMatrix{T},
    AWje::AbstractVector{<:AbstractVector{T}},
    sigmapenalty::Integer,
) where {T}
    ρ = T(size(A, 1)) - tr(A)              # ρ₀ = n − tr(R A), unweighted R A = A
    @inbounds for j in eachindex(AWje)
        ρ += dot(view(Λy, j, :), AWje[j])  # Λ̂ʸ[j,:] · (A Wⱼ e)
    end
    return ρ + T(sigmapenalty)
end

# Shared spine of the analytic and numeric Gaussian bias corrections. Computes the REML/ML
# degrees `nθ` and the reused `A Wⱼ e` vectors, defers the `(B, C)` pair to `buildBC` (the only
# divergence — the C-row formula and the B-source), then solves `Λ̂ʸ = B⁻¹C` (factorisation, no
# inverse) and assembles ρ. Both [`dof_lmm`](@ref) and [`dof_lmm_numeric`](@ref) route through
# here, so the solve and the ρ tail exist once and the two penalties cannot drift outside `buildBC`.
function _dof_lmm_spine(
    buildBC::F, c::GaussianComponents{T}, sigmapenalty::Integer
) where {T,F}
    e, A, Wlist = c.e, c.A, c.Wlist
    n = length(e)
    p = size(c.X, 2)
    nθ = c.isREML ? (n - p) : n            # REML uses n − p; ML uses n
    AWje = [A * (W * e) for W in Wlist]    # A Wⱼ e — reused by C, B's cross term, and ρ
    B, C = buildBC(nθ, AWje)
    Λy = _lambday(B, C)
    return _assemble_rho(A, Λy, AWje, sigmapenalty)
end

"""
    dof_lmm(c::GaussianComponents{T}; sigmapenalty::Integer = 1) -> T

The Greven–Kneib bias-corrected effective degrees of freedom ρ of a Gaussian LMM — the
penalty term of the conditional AIC `cAIC = −2 ℓ_cond + 2ρ`. A faithful port of
`cAIC4`'s analytic Gaussian bias correction (`analytic = TRUE`).

# Mathematical background

With `n` observations, `s` free covariance components, fixed-effects rank `p`, residual
`e`, projector `A`, inverse scaled marginal variance `V₀⁻¹`, derivative matrices
`Wⱼ = Z Dⱼ Zᵀ`, quadratic forms `eᵀWⱼe`, and `tʸᵉ = yᵀe`, define `M = V₀⁻¹` (ML) or
`M = A` (REML) and `nθ = n` (ML) or `nθ = n − p` (REML). Build

```math
C_{j,:} = A W_j e - \\frac{e^{\\mathsf T} W_j e}{2\\,t^{ye}}\\, e^{\\mathsf T},
\\qquad
B_{jk} = -\\frac{t^{ye}\\,\\operatorname{tr}(W_j M W_k M)}{2 n_\\theta}
        - \\frac{(e^{\\mathsf T}W_j e)(e^{\\mathsf T}W_k e)}{2\\,t^{ye}}
        + e^{\\mathsf T} W_k A W_j e,
```

solve `Λ̂ʸ = B⁻¹ C` (factorisation, no inverse), and assemble

```math
\\rho = \\underbrace{n - \\operatorname{tr}(A)}_{\\rho_0}
     + \\sum_{j=1}^{s} \\hat\\Lambda^{y}_{j,:} \\cdot (A W_j e)
     + \\texttt{sigmapenalty}.
```

The Greven–Kneib term `Σⱼ …` corrects `ρ₀ = tr(H₁)` for the estimation of `θ`, giving
`ρ ≥ ρ₀`.

# Arguments
- `c`: the [`GaussianComponents`](@ref) (unweighted Gaussian path, `R = Iₙ`).
- `sigmapenalty`: number of estimated residual-variance parameters added to ρ — `1` for
  one estimated σ² (the default and `cAIC4`'s default), `0` if the error variance is known.

# Returns
- The scalar effective degrees of freedom `ρ::T`.

# Example
```jldoctest
julia> using ConditionalAIC.DofLMM: GaussianComponents, dof_lmm

julia> using LinearAlgebra: I, tr

julia> c = GaussianComponents(
           reshape([1.0, 1.0, 1.0], 3, 1),            # X (intercept)
           [0.5, -0.5, 0.0],                          # e
           Matrix(0.5I, 3, 3),                        # A
           Matrix(1.0I, 3, 3),                        # V₀⁻¹
           [Matrix(1.0I, 3, 3)],                      # Wlist (one component)
           [0.5],                                     # eᵀW₁e
           0.5,                                       # tye
           false,                                     # ML
       );

julia> dof_lmm(c) > 3 - tr(c.A)   # ρ exceeds the naive plug-in ρ₀
true
```
"""
function dof_lmm(c::GaussianComponents{T}; sigmapenalty::Integer=1) where {T}
    return _dof_lmm_spine(c, sigmapenalty) do nθ, AWje
        _analytic_bc(c, nθ, AWje)
    end
end

# The analytic `(B, C)`: the closed-form Greven–Kneib Hessian B — the Fisher trace term
# `tr(Wⱼ M Wₖ M)` (via `traceprod`, no materialised product), the residual-quadratic square, and
# the quartic cross term `eᵀ Wₖ A Wⱼ e` — together with the analytic cross-product C, in one
# j-loop. `M = A` (REML) or `V₀⁻¹` (ML). The reused `A Wⱼ e` vectors come from the spine.
function _analytic_bc(c::GaussianComponents{T}, nθ::Int, AWje) where {T}
    e, A, Wlist, eWe, tye = c.e, c.A, c.Wlist, c.eWelist, c.tye
    n = length(e)
    s = length(Wlist)
    M = c.isREML ? A : c.V0inv            # WAⱼ = Wⱼ M : REML uses A, ML uses V₀⁻¹
    WA = [W * M for W in Wlist]           # the only materialised products (reused over k)
    C = Matrix{T}(undef, s, n)
    B = Matrix{T}(undef, s, s)
    @inbounds for j in 1:s
        C[j, :] = AWje[j] .- (eWe[j] / (2 * tye)) .* e
        for k in j:s
            traceterm = -tye * traceprod(WA[j], WA[k]) / (2 * nθ)
            sqterm = -eWe[j] * eWe[k] / (2 * tye)
            quartic = dot(e, Wlist[k] * AWje[j])          # eᵀ Wₖ A Wⱼ e
            B[j, k] = B[k, j] = traceterm + sqterm + quartic
        end
    end
    return B, C
end

"""
    dof_lmm_numeric(c::GaussianComponents{T}, B::AbstractMatrix{T};
                    sigmapenalty::Integer = 1) -> T

The Greven–Kneib bias-corrected effective degrees of freedom ρ with an **externally
supplied** Hessian `B` — the port of `cAIC4`'s numeric Gaussian bias correction
(`analytic = FALSE`). This is the assembly behind the `:forwarddiff` and `:finitediff`
B-sources of [`caic`](@ref ConditionalAIC.caic): the curvature `B` of the (restricted) profile log-likelihood
is obtained numerically rather than from the closed form, and only the cross-product `C`
and the final ρ assembly are recomputed here.

# Mathematical background

With the notation of [`dof_lmm`](@ref) and `nθ = n` (ML) or `nθ = n − p` (REML), the
numeric path leaves `B` external and rescales the cross-product (matching `cAIC4`'s
numeric Gaussian bias correction):

```math
C_{j,:} = \\frac{2\\,n_\\theta}{t^{ye}}
          \\left( (A W_j e)^{\\mathsf T} - \\frac{e^{\\mathsf T} W_j e}{t^{ye}}\\, e^{\\mathsf T} \\right),
```

using `eᵀWⱼA = (A Wⱼ e)ᵀ` (both `A` and `Wⱼ` symmetric). The solve `Λ̂ʸ = B⁻¹C`
(factorisation, no inverse) and the ρ assembly

```math
\\rho = n - \\operatorname{tr}(A)
     + \\sum_{j=1}^{s} \\hat\\Lambda^{y}_{j,:} \\cdot (A W_j e)
     + \\texttt{sigmapenalty}
```

are **identical** to the analytic path — only the source of `B` and the scaling of `C`
differ. `B` must be supplied on the deviance scale (−2·log-lik for ML, the REML criterion
for REML), matching the objective the optimiser differentiates.

# Arguments
- `c`: the [`GaussianComponents`](@ref) (unweighted Gaussian path, `R = Iₙ`).
- `B`: the `s×s` numeric Hessian of the (restricted) profile objective at `θ̂`.
- `sigmapenalty`: number of estimated residual-variance parameters added to ρ (default `1`).

# Returns
- The scalar effective degrees of freedom `ρ::T`.

# Throws
- `ArgumentError` if `B` is not `s×s`, where `s = length(c.Wlist)`.
"""
function dof_lmm_numeric(
    c::GaussianComponents{T}, B::AbstractMatrix{T}; sigmapenalty::Integer=1
) where {T}
    s = length(c.Wlist)
    size(B) == (s, s) ||
        throw(ArgumentError("B must be $s×$s (s = length(Wlist)); got $(size(B))"))
    return _dof_lmm_spine(c, sigmapenalty) do nθ, AWje
        _numeric_c(c, B, nθ, AWje)
    end
end

# The numeric `(B, C)`: B is supplied externally (numerically differentiated upstream); only the
# rescaled `analytic = FALSE` cross-product C is formed here, using eᵀWⱼA = (A Wⱼ e)ᵀ (both A and
# Wⱼ symmetric). The external B is returned unchanged alongside C so the spine's solve and ρ
# assembly are shared with the analytic path. `nθ` is `np` in the `analytic = FALSE` branch.
function _numeric_c(c::GaussianComponents{T}, B::AbstractMatrix{T}, nθ::Int, AWje) where {T}
    e, eWe, tye = c.e, c.eWelist, c.tye
    n = length(e)
    s = length(c.Wlist)
    C = Matrix{T}(undef, s, n)
    @inbounds for j in 1:s
        C[j, :] = (2 * nθ / tye) .* (AWje[j] .- (eWe[j] / tye) .* e)
    end
    return B, C
end

"""
    efron_penalty(yhat, sigma, Ystar, Yhatstar, sigmapenalty=0) -> T

Efron's covariance penalty (the parametric-bootstrap effective degrees of freedom) —
the faithful port of `cAIC4`'s conditional-bootstrap df estimator. Pure, fit-independent,
and testable without any `MixedModels` object.

# Mathematical background

With `n` observations, `B ≥ 2` bootstrap draws, residual standard deviation `σ̂`,
bootstrap responses `Y*` (`n×B`) with row means `ȳ*ᵢ = (1/B) Σ_b y*(b)ᵢ`, and
bootstrap conditional means `Ŷ*` (`n×B`), `cAIC4`'s estimator is

```math
\\rho = \\frac{1}{(B - 1)\\,\\hat\\sigma^{2}}
        \\sum_{b = 1}^{B} \\sum_{i = 1}^{n}
          \\hat y^{*}(b)_{i} \\, \\bigl(y^{*}(b)_{i} - \\bar y^{*}_{i}\\bigr)
        + \\texttt{sigmapenalty}.
```

The centring is on the **bootstrap row mean** `ȳ*ᵢ` (not the original fit `ŷᵢ`) and
the divisor is the **unbiased** `B − 1`, making the assembly the standard sample-
covariance estimator of `cov(y, ŷ) / σ²`. The `yhat` argument is *unused*
arithmetically — it is carried in the signature for symmetry with the analytic /
numeric units (and for caller readability), and exists to match the original
fit's conditional mean that the spine constructs `Y*` around.

Each draw `y*(b) = ŷ + σ̂ ε(b)`, `ε ~ N(0,I)` is a parametric bootstrap sample; the
corresponding `ŷ*(b)` is the conditional mean of a fresh model fit to `y*(b)`. The
`sigmapenalty` term is the package's σ²-parameter count, added by the spine for
interface symmetry with the analytic path; `cAIC4`'s conditional-bootstrap estimator
itself does not add one (`cAIC4` routes `sigma.penalty` only to its Gaussian bias
correction). The default here is therefore `0` — matching `cAIC4`'s bare arithmetic —
and the fixture compares against the same.

# Arguments
- `yhat`: the `n`-vector conditional fitted mean `ŷ` of the original fit (carried for
  signature symmetry; unused arithmetically — see above).
- `sigma`: the residual standard deviation `σ̂ > 0` of the original fit.
- `Ystar`: an `n×B` matrix whose `b`-th column is `y*(b)`; `B ≥ 2`.
- `Yhatstar`: an `n×B` matrix whose `b`-th column is `ŷ*(b)`.
- `sigmapenalty`: non-negative integer added to the penalty (default `0` — matches
  `cAIC4`'s arithmetic; the bootstrap *spine* adds the package's σ²-parameter count).

# Returns
- The scalar Efron penalty `ρ::T`.

# Throws
- `ArgumentError` if `Ystar` or `Yhatstar` have the wrong shape, `B < 2`, or
  `sigmapenalty < 0`.
- `DomainError` if `sigma ≤ 0`.
"""
function efron_penalty(
    yhat::AbstractVector{T},
    sigma::T,
    Ystar::AbstractMatrix{T},
    Yhatstar::AbstractMatrix{T},
    sigmapenalty::Integer=0,
) where {T<:AbstractFloat}
    n = length(yhat)
    nstar, B = size(Ystar)
    n == nstar || throw(ArgumentError("Ystar has $nstar rows but yhat has length $n"))
    size(Yhatstar) == (nstar, B) ||
        throw(ArgumentError("Yhatstar shape $(size(Yhatstar)) ≠ Ystar shape ($nstar, $B)"))
    sigma > 0 || throw(DomainError(sigma, "sigma must be positive"))
    sigmapenalty >= 0 || throw(ArgumentError("sigmapenalty must be ≥ 0; got $sigmapenalty"))
    B >= 2 || throw(
        ArgumentError(
            "B = size(Ystar, 2) must be ≥ 2 (cAIC4 uses the unbiased (B−1) divisor); got B = $B",
        ),
    )
    rowmean = vec(mean(Ystar; dims=2))                 # n-vector ȳ*ᵢ
    S = zero(T)
    for b in 1:B
        S += dot(view(Yhatstar, :, b), view(Ystar, :, b) .- rowmean)
    end
    return S / ((B - one(T)) * sigma^2) + T(sigmapenalty)
end

end # module DofLMM
