"""
    cAIC.Components

Build the Gaussian-LMM bias-correction component set ([`cAIC.DofLMM.GaussianComponents`](@ref))
from the raw quantities extracted from a fitted `LinearMixedModel` — the `cAIC.jl` analogue
of `cAIC4`'s `getModelComponents.merMod`.

This module performs the **construction** (the math of `docs/math/0002-gaussian-bias-correction.md`
§3 and §6); it touches **no** `MixedModels` object — only the dense arrays the quarantine
module [`cAIC.MMInternals`](@ref) extracts (`X`, `y`, `ŷ`, the per-reterm `Z`/`λ` blocks,
and the `θ`-parametrisation map). It is therefore the fit-dependent bridge that ADR-0003
places at **Level-2**: it is exercised end-to-end through the assembled [`caic`](@ref), not
in Level-1 isolation.

Every linear solve goes through a Cholesky factorisation; no explicit inverse and no `det`
is formed (CLAUDE §9). The scaled inverse marginal variance `V₀⁻¹` is built from the
Woodbury identity through a Cholesky of the `q×q` capacitance matrix `I + (ZΛ)ᵀ(ZΛ)`
(`docs/math/0002` §3), and `A`'s fixed-effects adjustment from a Cholesky of `Xᵀ V₀⁻¹ X`.
"""
module Components

using LinearAlgebra: I, Symmetric, cholesky, dot

using ..DofLMM: GaussianComponents

"""
    gaussiancomponents(X, y, μ, Zblocks, λblocks, parmap, isREML) -> GaussianComponents

Assemble the [`GaussianComponents`](@ref) of a fitted Gaussian LMM from its extracted
pieces. `X` is the `n×p` fixed-effects design, `y` the response, `μ = ŷ` the conditional
fitted mean, `Zblocks`/`λblocks` the per-reterm dense `Z` and `λ` blocks, and `parmap` the
`θ → (reterm, row, col)` map (`docs/math/0002` §3, §6). `isREML` selects the objective the
downstream bias correction uses.

Builds, all via Cholesky solves (no explicit inverse; CLAUDE §9):

- `ZΛ` block-by-block (`Λ` is block-diagonal across reterms and group copies);
- `V₀⁻¹ = Iₙ − (ZΛ)(Iₖ + (ZΛ)ᵀ(ZΛ))⁻¹(ZΛ)ᵀ` (Woodbury);
- `A = V₀⁻¹ − G(Xᵀ V₀⁻¹ X)⁻¹ Gᵀ` with `G = V₀⁻¹ X`;
- the conditional residual `e = y − μ`, `tʸᵉ = yᵀe`;
- the derivative matrices `Wⱼ = Z Dⱼ Zᵀ` from the `parmap` positions, and `eᵀWⱼe`.
"""
function gaussiancomponents(
    X::Matrix{T},
    y::Vector{T},
    μ::Vector{T},
    Zblocks::Vector{Matrix{T}},
    λblocks::Vector{Matrix{T}},
    parmap::Vector{NTuple{3,Int}},
    isREML::Bool,
) where {T<:AbstractFloat}
    n = length(y)

    # ZΛ, built per reterm and per group copy: within reterm t, each k-column group slice
    # of Z is right-multiplied by the shared λ block (Λ is block-diagonal; doc 0002 §3/§6).
    ZΛcols = Matrix{T}[]
    for (Zt, λt) in zip(Zblocks, λblocks)
        k = size(λt, 1)
        ng = size(Zt, 2) ÷ k
        ZΛt = similar(Zt)
        for g in 0:(ng - 1)
            cols = (g * k + 1):(g * k + k)
            @views ZΛt[:, cols] = Zt[:, cols] * λt
        end
        push!(ZΛcols, ZΛt)
    end
    ZΛ = reduce(hcat, ZΛcols)
    q = size(ZΛ, 2)

    # V₀⁻¹ = Iₙ − (ZΛ) S⁻¹ (ZΛ)ᵀ via Woodbury, S = Iₖ + (ZΛ)ᵀ(ZΛ) SPD (doc 0002 §3).
    # U = L⁻¹(ZΛ)ᵀ from the Cholesky S = L Lᵀ; V₀⁻¹ = Iₙ − UᵀU. No explicit inverse.
    S = Symmetric(Matrix{T}(I, q, q) + transpose(ZΛ) * ZΛ)
    U = cholesky(S).L \ transpose(ZΛ)
    V0inv = Matrix{T}(I, n, n) - transpose(U) * U
    V0inv = (V0inv + transpose(V0inv)) / 2          # discard round-off asymmetry

    # A = V₀⁻¹ − G (Xᵀ V₀⁻¹ X)⁻¹ Gᵀ, G = V₀⁻¹ X; the p×p solve via Cholesky (no inverse).
    G = V0inv * X
    A = V0inv - G * (cholesky(Symmetric(transpose(X) * G)) \ transpose(G))
    A = (A + transpose(A)) / 2

    e = y .- μ                                       # conditional residual (cAIC4: y − μ)
    tye = dot(y, e)

    Wlist = Matrix{T}[_buildw(pm, Zblocks, λblocks) for pm in parmap]
    eWelist = T[dot(e, W * e) for W in Wlist]

    return GaussianComponents(X, e, A, V0inv, Wlist, eWelist, tye, isREML)
end

# Wⱼ = Z Dⱼ Zᵀ for the θ-component at parmap position `(t, i, j)` (doc 0002 §6): Dⱼ is the
# symmetric 0/1 pattern with ones at (i, j) and (j, i) within each group copy of reterm t.
# Since Dⱼ touches only reterm t's columns, Wⱼ = Zₜ Dⱼ,ₜ Zₜᵀ.
function _buildw(pm::NTuple{3,Int}, Zblocks::Vector{Matrix{T}}, λblocks) where {T}
    t, i, j = pm
    Zt = Zblocks[t]
    k = size(λblocks[t], 1)
    qt = size(Zt, 2)
    ng = qt ÷ k
    D = zeros(T, qt, qt)
    for g in 0:(ng - 1)
        base = g * k
        D[base + i, base + j] = one(T)
        D[base + j, base + i] = one(T)               # symmetric pattern (no-op if i == j)
    end
    return Zt * D * transpose(Zt)
end

end # module Components
