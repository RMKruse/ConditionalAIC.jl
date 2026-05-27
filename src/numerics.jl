"""
    cAIC.Numerics

Pure, numerically-stable numerical primitives the conditional AIC is assembled from
(CLAUDE.md §9). **No `MixedModels` dependency** — this module is importable and testable
in isolation, and every kernel is generic over `T <: AbstractFloat`.

Each kernel computes its quantity via the *stable* formulation (log-space, Cholesky-based
solves, traces without materialising products, no explicit inverse and no `det` of a
large/near-singular matrix) and is validated against the *naive* reference on synthetic
inputs at the Level-1 tolerance. The mathematical identities are recorded in
`docs/math/0001-numerics-primitives.md`.

These kernels are consumed by `loglik`, the Level-1 df port, and the scoring spine; they
are internal helpers and are not part of the public `cAIC` export surface.
"""
module Numerics

using LinearAlgebra: Cholesky, Symmetric, checksquare, cholesky, dot, issuccess, logdet
using LogExpFunctions: logsumexp as _logsumexp

"""
    traceprod(A::AbstractMatrix, B::AbstractMatrix) -> eltype

Trace of the product `AB` computed **without materialising** the product, via the
identity

```math
\\operatorname{tr}(AB) = \\sum_i \\sum_k A_{ik} B_{ki}
```

(equivalently `Σ A .* Bᵀ`; CLAUDE.md §9). The accumulating double sum touches each entry
of `A` and `B` once and allocates nothing, where `tr(A * B)` would form the `m×m`
product first. `A` is `m×n` and `B` is `n×m`; the empty contraction returns zero.

# Arguments
- `A`: an `m×n` matrix.
- `B`: an `n×m` matrix (so that `AB` is square).

# Returns
- The scalar `tr(AB)`, in the promoted element type of `A` and `B`.

# Throws
- `DimensionMismatch` if `size(A) != reverse(size(B))`.

# Example
```jldoctest
julia> cAIC.Numerics.traceprod([1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0])
70.0
```
"""
function traceprod(A::AbstractMatrix, B::AbstractMatrix)
    (axes(A, 1) == axes(B, 2) && axes(A, 2) == axes(B, 1)) || throw(
        DimensionMismatch(
            "traceprod requires size(A) == reverse(size(B)); got $(size(A)) and $(size(B))",
        ),
    )
    # Allocation-free reduction of the trace identity. A manual loop is the justified
    # form here (CLAUDE.md §12): the "vectorized" `sum(A .* transpose(B))` would
    # allocate an m×n temporary, defeating the §9 "without materialising" mandate.
    s = zero(eltype(A)) * zero(eltype(B))
    @inbounds for k in axes(A, 2), i in axes(A, 1)
        s += A[i, k] * B[k, i]
    end
    return s
end

"""
    logdetpd(A::AbstractMatrix) -> eltype
    logdetpd(C::Cholesky) -> eltype

Log-determinant of a symmetric positive-definite matrix, computed from its Cholesky
factor `A = L Lᵀ` via

```math
\\log\\det(A) = 2 \\sum_i \\log L_{ii},
```

in log-space and **never** as `log(det(A))` (CLAUDE.md §9: `logdet` on a
Cholesky/triangular factor, never `det` of a large or near-singular matrix). The
factor is obtained with `cholesky(Symmetric(A))`; pass a `Cholesky` directly to reuse
an existing factor without refactorising (§9).

# Arguments
- `A`: a square, symmetric positive-definite matrix (its upper triangle is used), or
- `C`: an already-computed `Cholesky` factor.

# Returns
- The scalar `logdet(A)`, in the matrix' element type.

# Throws
- `DimensionMismatch` if `A` is not square.
- `DomainError` if `A` is not positive-definite (the Cholesky factorisation fails);
  a silently-wrong determinant is never returned.

# Example
```jldoctest
julia> cAIC.Numerics.logdetpd([4.0 1.0; 1.0 3.0])
2.3978952727983707
```
"""
# Cholesky factor of a symmetric positive-definite `A`, or a loud failure: a
# `DimensionMismatch` if `A` is not square, a `DomainError` (naming the caller `who`) if
# `A` is not positive-definite. `check=false` lets us convert the failed factorisation
# into a domain error rather than surfacing a bare `PosDefException`.
function _cholpd(A::AbstractMatrix, who::AbstractString)
    checksquare(A)
    C = cholesky(Symmetric(A); check=false)
    issuccess(C) || throw(
        DomainError(
            A, "$who requires a positive-definite matrix; Cholesky factorisation failed"
        ),
    )
    return C
end

logdetpd(A::AbstractMatrix) = logdetpd(_cholpd(A, "logdetpd"))
logdetpd(C::Cholesky) = logdet(C)

"""
    invquad(A::AbstractMatrix, x::AbstractVector) -> eltype
    invquad(C::Cholesky, x::AbstractVector) -> eltype

Quadratic form `xᵀ A⁻¹ x` for symmetric positive-definite `A = L Lᵀ`, computed by a
Cholesky **solve** and **never** by forming an explicit inverse (CLAUDE.md §9/§12):

```math
x^{\\mathsf T} A^{-1} x = \\lVert L^{-1} x \\rVert^2,
```

where `L⁻¹ x` is the triangular solve `L \\ x`. The result is a sum of squares, hence
`≥ 0` for real `x`. Pass a `Cholesky` directly to reuse an existing factor without
refactorising (§9).

# Arguments
- `A`: a square, symmetric positive-definite matrix (its upper triangle is used), or
  `C`: an already-computed `Cholesky` factor.
- `x`: a vector of length `size(A, 1)`.

# Returns
- The scalar `xᵀ A⁻¹ x`, in the promoted element type.

# Throws
- `DimensionMismatch` if `A` is not square or `length(x) != size(A, 1)`.
- `DomainError` if `A` is not positive-definite (the Cholesky factorisation fails).

# Example
```jldoctest
julia> cAIC.Numerics.invquad([4.0 0.0; 0.0 2.0], [2.0, 2.0])
3.0
```
"""
invquad(A::AbstractMatrix, x::AbstractVector) = invquad(_cholpd(A, "invquad"), x)

function invquad(C::Cholesky, x::AbstractVector)
    size(C, 1) == length(x) || throw(
        DimensionMismatch(
            "invquad requires length(x) == size(C, 1); got $(length(x)) and $(size(C, 1))",
        ),
    )
    y = C.L \ x          # L y = x  ⟹  y = L⁻¹ x
    return dot(y, y)      # ‖L⁻¹ x‖² = xᵀ (L Lᵀ)⁻¹ x
end

"""
    logsumexp(x::AbstractArray{<:Real}) -> eltype

Numerically-stable log of a sum of exponentials,

```math
\\log \\sum_i \\exp(x_i) = m + \\log \\sum_i \\exp(x_i - m), \\quad m = \\max_i x_i,
```

which keeps every exponent `≤ 0` so the sum neither overflows nor underflows to zero,
where the naive `log(sum(exp, x))` overflows once any `xᵢ` is large (CLAUDE.md §9:
log-space, never sum raw products of small numbers). Delegates to
`LogExpFunctions.logsumexp`; this is the package's vetted log-space entry point, used by
the GLMM log-space accumulations.

# Arguments
- `x`: an array of reals.

# Returns
- The scalar `log Σ exp(xᵢ)`, in `x`'s (floating) element type. An empty `x` returns
  `-Inf` (the log of an empty sum); a `NaN` entry propagates.

# Example
```jldoctest
julia> cAIC.Numerics.logsumexp([0.0, 0.0])
0.6931471805599453
```
"""
logsumexp(x::AbstractArray{<:Real}) = _logsumexp(x)

end # module Numerics
