"""
    cAIC.DofGLMM

Family-specific **effective degrees of freedom** ρ for generalised linear mixed models —
the GLMM-side analogue of [`cAIC.DofLMM`](@ref) for the Gaussian path.

The estimand and all family-specific formulae are pinned in
`docs/math/0006-glmm-bias-correction.md`. This module implements two df routes in M3
scope:

- **Poisson (Chen–Stein):** [`dof_glmm_poisson`](@ref) / §3 of the math spec.
  Influence-based: one full-model refit per nonzero observation (`yᵢ → yᵢ − 1`).

Each route follows the same Level-1 / Level-2 isolation pattern as `DofLMM`
(ADR-0003): a [`PoissonInfluenceComponents`](@ref) struct carries the pure arithmetic
inputs so the df formula is testable without any model fitting; the
`GeneralizedLinearMixedModel` dispatch builds those components via the refit loop and
delegates.

All access to `MixedModels.jl` internals goes through [`cAIC.MMInternals`](@ref)
(CLAUDE.md §3).
"""
module DofGLMM

using MixedModels: GeneralizedLinearMixedModel

using ..MMInternals: glmmresponse, glmmlinpred, refitglmm_eta

# ── PoissonInfluenceComponents ─────────────────────────────────────────────────

"""
    PoissonInfluenceComponents{T<:AbstractFloat}

The influence-function component set for the Poisson Chen–Stein df
(`docs/math/0006` §3). Parametrisation-neutral — this struct carries **no** fitted
model, so the df arithmetic is testable in isolation from any fit (ADR-0003).

# Fields
- `y::Vector{T}`: the `n`-vector of observed counts (the fitted model's response).
- `eta0::Vector{T}`: the fitted linear predictor `η̂ = Xβ̂ + Zb̂`, length `n`.
- `ind::Vector{Int}`: 1-based indices of the nonzero observations (`y[i] ≠ 0`);
  the loop only iterates over these (decrementing `y[i] = 0` is out of domain).
- `eta_dec::Vector{T}`: for each `k`-th entry in `ind`, the *k*-th linear predictor
  `η̂ᵢ^{(−i)}` — the `ind[k]`-th component of the linear predictor after refitting
  the model on `y` with `y[ind[k]]` decremented by one (`yᵢ − 1`).
"""
struct PoissonInfluenceComponents{T<:AbstractFloat}
    y::Vector{T}
    eta0::Vector{T}
    ind::Vector{Int}
    eta_dec::Vector{T}

    function PoissonInfluenceComponents(
        y::Vector{T}, eta0::Vector{T}, ind::Vector{Int}, eta_dec::Vector{T}
    ) where {T<:AbstractFloat}
        length(y) == length(eta0) ||
            throw(DimensionMismatch("y and eta0 must have the same length"))
        length(ind) == length(eta_dec) ||
            throw(DimensionMismatch("ind and eta_dec must have the same length"))
        all(i -> 1 <= i <= length(y), ind) ||
            throw(ArgumentError("all ind entries must be valid indices into y"))
        return new{T}(y, eta0, ind, eta_dec)
    end
end

# ── dof_glmm_poisson — Level-1 arithmetic ─────────────────────────────────────

"""
    dof_glmm_poisson(c::PoissonInfluenceComponents{T}) -> T

**Level-1 arithmetic dispatch** — the Chen–Stein influence df computed from
pre-assembled components `c`, with no model fitting.

Implements `docs/math/0006` §3:

```math
ρ_{Pois} = ∑_{i : y_i ≠ 0} y_i (η̂_i - η̂_i^{(-i)})
```

where `η̂_i^{(-i)}` is the `i`-th fitted linear predictor after refitting the model
on `y` with its `i`-th count decremented by one (the Chen–Stein / Hudson unit
decrement for the Poisson).

# Arguments
- `c`: pre-assembled [`PoissonInfluenceComponents`](@ref); `c.ind` must be the
  1-based indices of all nonzero `y` entries.

# Returns
- The scalar `ρ`, type `T`. Returns `zero(T)` when `c.ind` is empty (all
  observations have `y = 0`; no terms contribute).

# Example
```jldoctest
julia> using cAIC: DofGLMM
julia> c = DofGLMM.PoissonInfluenceComponents(
           [2.0, 0.0, 1.0], [1.0, 0.5, 1.5], [1, 3], [0.9, 1.4]
       );
julia> DofGLMM.dof_glmm_poisson(c)  # 2*(1.0-0.9) + 1*(1.5-1.4) = 0.3
0.30000000000000004
```
"""
function dof_glmm_poisson(c::PoissonInfluenceComponents{T}) where {T}
    bc = zero(T)
    @inbounds for k in eachindex(c.ind)
        i = c.ind[k]
        bc += c.y[i] * (c.eta0[i] - c.eta_dec[k])
    end
    return bc
end

# ── dof_glmm_poisson — Level-2 model dispatch ─────────────────────────────────

"""
    dof_glmm_poisson(m::GeneralizedLinearMixedModel{T}) -> T

**Level-2 model dispatch** — the Chen–Stein influence df for a fitted Poisson
`GeneralizedLinearMixedModel`.

Builds a [`PoissonInfluenceComponents`](@ref) by performing one full-model refit per
nonzero observation (`y_i → y_i − 1`, the Chen–Stein / Hudson unit decrement) and
collecting the `i`-th fitted linear predictor from each refit. Delegates the final
arithmetic to the Level-1 dispatch.

The model `m` is assumed to already be boundary-reduced (i.e. not singular); the
caller is responsible for applying `MMInternals.reduceboundary` / the full-singularity
fallback before invoking this function (consistent with the Gaussian path and
`cAIC4::biasCorrectionPoisson`'s `deleteZeroComponents` pre-step).

# Arguments
- `m`: a fitted `GeneralizedLinearMixedModel` with Poisson family. The original
  model is not mutated; all refits operate on deep copies (via
  [`MMInternals.refitglmm_eta`](@ref)).

# Returns
- The scalar `ρ_{Pois}`, type `T`.

# Example
```julia
using MixedModels, cAIC
m = fit(MixedModel, @formula(y ~ x + (1|group)), dat, Poisson(); progress=false)
ρ = DofGLMM.dof_glmm_poisson(m)
```
"""
function dof_glmm_poisson(m::GeneralizedLinearMixedModel{T}) where {T}
    y = glmmresponse(m)
    eta0 = glmmlinpred(m)
    ind = findall(!=(zero(T)), y)
    isempty(ind) && return zero(T)

    eta_dec = Vector{T}(undef, length(ind))
    @inbounds for (k, i) in enumerate(ind)
        y_dec = copy(y)
        y_dec[i] -= one(T)
        eta_dec[k] = refitglmm_eta(m, y_dec)[i]
    end
    c = PoissonInfluenceComponents(y, eta0, ind, eta_dec)
    return dof_glmm_poisson(c)
end

end # module DofGLMM
