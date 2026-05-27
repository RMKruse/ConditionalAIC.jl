"""
    cAIC.DofLMM

Greven–Kneib bias-corrected **effective degrees of freedom** ρ for a Gaussian linear
mixed model — the port of `cAIC4`'s `calculateGaussianBc` (`analytic = TRUE`).

This module is the **Level-1 isolation unit** (ADR-0003): a *pure, fit-independent,
parametrisation-neutral* map from a component set ([`GaussianComponents`](@ref)) to the
scalar ρ. It touches **no** `MixedModels` object; it consumes dense components in
`cAIC4`'s `getModelComponents.merMod` layout and reproduces the exact arithmetic of
`calculateGaussianBc`. The mathematics is pinned in `docs/math/0002-gaussian-bias-correction.md`
(§3 the component layout, §4 the closed-form B/C and the ρ assembly).

Every kernel uses the numerically-stable [`cAIC.Numerics`](@ref) primitives: the Fisher
trace term `tr(Wⱼ M Wₖ M)` is formed by `traceprod` without materialising the product,
and `Λ̂ʸ = B⁻¹C` is a factorisation-based solve with no explicit inverse (CLAUDE §9).
"""
module DofLMM

using LinearAlgebra: Symmetric, cholesky, dot, issuccess, tr

using ..Numerics: traceprod

"""
    GaussianComponents{T<:AbstractFloat}

The Gaussian-LMM bias-correction component set, in `cAIC4`'s `getModelComponents.merMod`
layout (`docs/math/0002` §3). All matrices are dense and parametrisation-neutral — this
type carries *no* `θ`-vector and no fitted model, so the correction arithmetic is tested
in isolation from any fit (ADR-0003). Targets the **unweighted** Gaussian path
(`R = Iₙ`, so `R A = A`), the M2 scope; weighted Gaussian is deferred, matching `cAIC4`.

The number of observations is `n = length(e)`, the number of free covariance components
is `s = length(Wlist)`, and the fixed-effects rank is `p = size(X, 2)`.

# Fields
- `X::Matrix{T}`: the `n×p` fixed-effects design. Only its column count `p` enters the
  correction (the REML degrees `nθ = n − p`); it is carried to match `cAIC4`'s `model\$X`.
- `e::Vector{T}`: the `n` conditional residual `e = y − ŷ = A y` (§0).
- `A::Matrix{T}`: the `n×n` fixed-effects-adjusted projector
  `A = V₀⁻¹ − V₀⁻¹X(XᵀV₀⁻¹X)⁻¹XᵀV₀⁻¹`.
- `V0inv::Matrix{T}`: the `n×n` inverse scaled marginal variance `V₀⁻¹`.
- `Wlist::Vector{Matrix{T}}`: the `s` derivative matrices `Wⱼ = Z Dⱼ Zᵀ` (each `n×n`; §6).
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
    # would otherwise produce a silently-wrong ρ downstream. Fail loudly (CLAUDE §4).
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

# Λ̂ʸ = B⁻¹C as a factorisation-based solve — never an explicit inverse (CLAUDE §9).
# B is the positive-definite negative profile-(restricted-)likelihood Hessian (doc 0002
# §5), so a Cholesky solve is the stable primary path; a symmetric (Bunch–Kaufman) solve
# is the fallback when B is not numerically positive-definite (θ near the boundary).
# Both compute the same B⁻¹C as `cAIC4`'s `solve(B) %*% C`.
function _lambday(B::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T}
    Bsym = Symmetric(B)
    fac = cholesky(Bsym; check=false)
    return issuccess(fac) ? fac \ C : Bsym \ C
end

"""
    dof_lmm(c::GaussianComponents{T}; sigmapenalty::Integer = 1) -> T

The Greven–Kneib bias-corrected effective degrees of freedom ρ of a Gaussian LMM — the
penalty term of the conditional AIC `cAIC = −2 ℓ_cond + 2ρ`. A faithful port of
`cAIC4::calculateGaussianBc(model, sigma.penalty, analytic = TRUE)`.

# Mathematical background

With `n` observations, `s` free covariance components, fixed-effects rank `p`, residual
`e`, projector `A`, inverse scaled marginal variance `V₀⁻¹`, derivative matrices
`Wⱼ = Z Dⱼ Zᵀ`, quadratic forms `eᵀWⱼe`, and `tʸᵉ = yᵀe`, define `M = V₀⁻¹` (ML) or
`M = A` (REML) and `nθ = n` (ML) or `nθ = n − p` (REML). Build (doc 0002 §4)

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
`ρ ≥ ρ₀` (doc 0002 §5).

# Arguments
- `c`: the [`GaussianComponents`](@ref) (unweighted Gaussian path, `R = Iₙ`).
- `sigmapenalty`: number of estimated residual-variance parameters added to ρ — `1` for
  one estimated σ² (the default and `cAIC4`'s default), `0` if the error variance is known.

# Returns
- The scalar effective degrees of freedom `ρ::T`.

# Example
```jldoctest
julia> using cAIC.DofLMM: GaussianComponents, dof_lmm

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
    e, A, Wlist, eWe, tye = c.e, c.A, c.Wlist, c.eWelist, c.tye
    n = length(e)
    s = length(Wlist)
    p = size(c.X, 2)

    M = c.isREML ? A : c.V0inv            # WAⱼ = Wⱼ M : REML uses A, ML uses V₀⁻¹
    nθ = c.isREML ? (n - p) : n

    WA = [W * M for W in Wlist]            # the only materialised products (reused over k)
    AWje = [A * (W * e) for W in Wlist]    # A Wⱼ e — reused by C, B's cross term, and ρ

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

    Λy = _lambday(B, C)

    ρ = T(n) - tr(A)                       # ρ₀ = n − tr(R A), unweighted R A = A
    @inbounds for j in 1:s
        ρ += dot(view(Λy, j, :), AWje[j])  # Λ̂ʸ[j,:] · (A Wⱼ e)
    end
    return ρ + T(sigmapenalty)
end

end # module DofLMM
