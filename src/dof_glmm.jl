"""
    ConditionalAIC.DofGLMM

Family-specific **effective degrees of freedom** ρ for generalised linear mixed models —
the GLMM-side analogue of [`ConditionalAIC.DofLMM`](@ref) for the Gaussian path.

This module implements three df routes:

- **Poisson (Chen–Stein):** [`dof_glmm_poisson`](@ref).
  Influence-based: one full-model refit per nonzero observation (`yᵢ → yᵢ − 1`).
- **Bernoulli (Efron's Steinian):** [`dof_glmm_bernoulli`](@ref).
  Per-observation label flip (`yᵢ → 1 − yᵢ`): `n` full-model refits, accumulated as a
  weighted logit difference.
- **Other families — conditional bootstrap:** [`dof_glmm_bootstrap`](@ref).
  Binomial with `|unique(y)|>2` and any other canonical-link family. `B` conditional
  draws `y*(b) ~ f(μ̂)` directly from the conditional response distribution, each refitted;
  the link-scale covariance
  penalty is [`DofLMM.efron_penalty`](@ref ConditionalAIC.DofLMM.efron_penalty) with σ̂²=1.

Each route follows the same fit-independent / model-dispatch pattern as `DofLMM`:
a pure arithmetic kernel ([`PoissonInfluenceComponents`](@ref) +
[`dof_glmm_poisson`](@ref) for Poisson, [`_bernoulli_df`](@ref) for Bernoulli) carries
the formula so it is testable without any model fitting; the
`GeneralizedLinearMixedModel` dispatch builds those inputs via the refit loop and
delegates.

All access to `MixedModels.jl` internals is quarantined in
[`ConditionalAIC.MMInternals`](@ref).
"""
module DofGLMM

using LogExpFunctions: logit
using MixedModels: GeneralizedLinearMixedModel, Poisson, Bernoulli, Binomial
using Random: AbstractRNG, Xoshiro
using ..DofLMM: efron_penalty
using ..MMInternals
using ..MMInternals: glmmresponse, glmmlinpred, refitglmm_eta

# ── PoissonInfluenceComponents ─────────────────────────────────────────────────

"""
    PoissonInfluenceComponents{T<:AbstractFloat}

The influence-function component set for the Poisson Chen–Stein df.
Parametrisation-neutral — this struct carries **no** fitted
model, so the df arithmetic is testable in isolation from any fit.

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

**Arithmetic dispatch** — the Chen–Stein influence df computed from
pre-assembled components `c`, with no model fitting.

Implements the Chen–Stein influence df:

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
julia> using ConditionalAIC: DofGLMM

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

**Model dispatch** — the Chen–Stein influence df for a fitted Poisson
`GeneralizedLinearMixedModel`.

Builds a [`PoissonInfluenceComponents`](@ref) by performing one full-model refit per
nonzero observation (`y_i → y_i − 1`, the Chen–Stein / Hudson unit decrement) and
collecting the `i`-th fitted linear predictor from each refit. Delegates the final
arithmetic to the arithmetic dispatch.

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
using MixedModels, ConditionalAIC
m = fit(MixedModel, @formula(y ~ x + (1|group)), dat, Poisson(); progress=false)
ρ = DofGLMM.dof_glmm_poisson(m)
```
"""
function dof_glmm_poisson(m::GeneralizedLinearMixedModel{T}) where {T}
    y = glmmresponse(m)
    eta0 = glmmlinpred(m)
    ind = findall(!=(zero(T)), y)
    isempty(ind) && return zero(T)

    # One full-model refit per nonzero observation, each from a fresh copy of the ORIGINAL fit
    # (`refitglmm_eta` deep-copies `m` per call): decrement yᵢ by one (the Chen–Stein / Hudson
    # unit decrement) and read the i-th refitted linear predictor η̂ᵢ^{(−i)}. The per-i fresh copy
    # is deliberate and *not* interchangeable with a single reused buffer: reusing one buffer
    # warm-starts the fixed-effects β from the previous perturbation's optimum (θ is reset to the
    # canonical initial either way), which shifts the Poisson df past the Level-2 parity tolerance.
    # This loop therefore is NOT unified with the Bernoulli flip loop (which does reuse one
    # buffer) — their buffering differs for a numerical reason, not by oversight.
    eta_dec = Vector{T}(undef, length(ind))
    @inbounds for (k, i) in enumerate(ind)
        y_dec = copy(y)
        y_dec[i] -= one(T)
        eta_dec[k] = refitglmm_eta(m, y_dec)[i]
    end
    c = PoissonInfluenceComponents(y, eta0, ind, eta_dec)
    return dof_glmm_poisson(c)
end

# ── Bernoulli / binary logistic GLMM (Efron's Steinian estimator) ─────────────

"""
    dof_glmm_bernoulli(m::GeneralizedLinearMixedModel{T}) -> T

Efron's Steinian bias-corrected effective degrees of freedom for a fitted Bernoulli
(binary logistic) GLMM. This is the `ConditionalAIC.jl` analogue of `cAIC4`'s
`biasCorrectionBernoulli` (`R/biasCorrectionBernoulli.R`).

For each observation `i`, the whole model is refitted on the response with
`yᵢ → 1 − yᵢ` (all other entries unchanged); the change in the conditional
fitted mean at position `i` accumulates as a weighted logit difference:

```math
\\rho = \\sum_{i=1}^{n} \\hat\\mu_i(1 - \\hat\\mu_i)\\,(-2y_i + 1)\\,
          \\bigl(\\operatorname{logit}(\\hat\\mu_i^{\\mathrm{flip}}) -
                 \\operatorname{logit}(\\hat\\mu_i)\\bigr),
```

where `μ̂ᵢ^{flip}` is the `i`-th fitted mean after refitting the model on the
label-flipped response. `n` refits are performed — one per observation; every binary
point is flippable (no `yᵢ = 0` skipping, unlike the Poisson Chen–Stein route).

The ground-truth R
function is `cAIC4::biasCorrectionBernoulli`.

# Arguments
- `m`: a fitted `GeneralizedLinearMixedModel` with a Bernoulli / binary logistic
  response (`y ∈ {0, 1}`). Partial boundary reduction (some `θ = 0`) is the
  caller's responsibility; this function scores the model as given.

# Returns
- `T` — the scalar effective df `ρ`.
"""
function dof_glmm_bernoulli(m::GeneralizedLinearMixedModel{T}) where {T}
    y = MMInternals.glmmresponse(m)
    μhat = MMInternals.glmmfittedmu(m)
    μhat_flip = MMInternals.bernoulliflipmu(m)
    return _bernoulli_df(y, μhat, μhat_flip)
end

"""
    _bernoulli_df(y, μhat, μhat_flip) -> T

Pure Efron Steinian formula kernel for the Bernoulli GLMM effective df.
Given pre-computed per-flip fitted means `μhat_flip`,
the result is a deterministic function of `(y, μhat, μhat_flip)`.

This kernel is fit-independent and can be
driven directly with synthetic inputs for tight-tolerance formula verification.

# Arguments
- `y`: binary response vector (`0.0` or `1.0`), length `n`.
- `μhat`: original fitted mean probabilities, length `n`, elements in `(0, 1)`.
- `μhat_flip`: length-`n` vector; entry `i` is the fitted mean at position `i` after
  refitting the model with `yᵢ → 1 − yᵢ`, elements in `(0, 1)`.

# Returns
`ρ = Σ μ̂ᵢ(1−μ̂ᵢ)(−2yᵢ+1)(logit(μ̂_flipᵢ)−logit(μ̂ᵢ))` as type `T`.
"""
function _bernoulli_df(
    y::AbstractVector{T}, μhat::AbstractVector{T}, μhat_flip::AbstractVector{T}
) where {T<:AbstractFloat}
    ρ = zero(T)
    @inbounds for i in eachindex(y, μhat, μhat_flip)
        sign_i = -2 * y[i] + one(T)
        weight_i = μhat[i] * (one(T) - μhat[i])
        logit_diff = logit(μhat_flip[i]) - logit(μhat[i])
        ρ += weight_i * sign_i * logit_diff
    end
    return ρ
end

# ── dof_glmm_bootstrap — conditional bootstrap (other families) ───────────────

"""
    glmmconddraw(rng::AbstractRNG, m::GeneralizedLinearMixedModel{T}, B::Int) -> Matrix{T}

Draw `B` conditional bootstrap samples from the GLMM response distribution, holding the
random effects fixed at their estimated values `b̂` (i.e. using the fitted `μ̂`). Returns
an `n × B` matrix whose `b`-th column is the `b`-th bootstrap response vector.

Draws directly from `f(μ̂ᵢ)`:
- **Poisson:** `yᵢ^{(b)} = rand(Poisson(μ̂ᵢ))` (float count)
- **Binomial:** `yᵢ^{(b)} = rand(Binomial(nᵢ, μ̂ᵢ)) / nᵢ` (proportion); `nᵢ` from
  [`MMInternals.glmmpriorweights`](@ref).
- **Bernoulli:** `yᵢ^{(b)} = rand(Bernoulli(μ̂ᵢ))` (0.0 or 1.0)

The fitted mean `μ̂`, distribution family, and prior weights are read through the
[`MMInternals`](@ref ConditionalAIC.MMInternals) accessors; the family-dispatched sampling
itself ([`_fill_conddraw!`](@ref)) touches no `MixedModels` internals.

Unsupported families (free-dispersion etc.) raise `ArgumentError`.

# Throws
- `ArgumentError` for unsupported distribution families.
- `ArgumentError` if the Binomial model has no prior weights.
"""
function glmmconddraw(rng::AbstractRNG, m::GeneralizedLinearMixedModel{T}, B::Int) where {T}
    μ = MMInternals.glmmfittedmu(m)
    n = length(μ)
    Ystar = Matrix{T}(undef, n, B)
    _fill_conddraw!(rng, Ystar, μ, MMInternals.glmmdist(m), MMInternals.glmmpriorweights(m))
    return Ystar
end

# Family-dispatched conditional sampler: fill `Ystar` in place with draws from `f(μ̂)`.
# Pure — receives the fitted mean `μ`, the distribution family (for dispatch), and the
# prior weights `wts` as plain data, so it touches no `MixedModels` internals. Poisson and
# Bernoulli ignore `wts`; the Binomial branch reads the per-observation trial counts from it.
function _fill_conddraw!(
    rng::AbstractRNG, Ystar::Matrix{T}, μ::Vector{T}, ::Poisson, _wts::Vector{T}
) where {T}
    n, B = size(Ystar)
    for b in 1:B, i in 1:n
        Ystar[i, b] = T(rand(rng, Poisson(μ[i])))
    end
    return Ystar
end

function _fill_conddraw!(
    rng::AbstractRNG, Ystar::Matrix{T}, μ::Vector{T}, ::Bernoulli, _wts::Vector{T}
) where {T}
    n, B = size(Ystar)
    for b in 1:B, i in 1:n
        Ystar[i, b] = T(rand(rng, Bernoulli(μ[i])))
    end
    return Ystar
end

function _fill_conddraw!(
    rng::AbstractRNG, Ystar::Matrix{T}, μ::Vector{T}, ::Binomial, wts::Vector{T}
) where {T}
    isempty(wts) && throw(
        ArgumentError(
            "glmmconddraw: conditional bootstrap for Binomial GLMM requires prior weights " *
            "(number of trials per observation). Refit the model with `weights=ntrials`.",
        ),
    )
    n, B = size(Ystar)
    for b in 1:B, i in 1:n
        ni = Int(wts[i])
        Ystar[i, b] = T(rand(rng, Binomial(ni, μ[i]))) / T(ni)
    end
    return Ystar
end

function _fill_conddraw!(rng, Ystar, μ, d, _wts)
    throw(
        ArgumentError(
            "glmmconddraw: family $(typeof(d)) is not supported by the conditional " *
            "bootstrap. Supported: Poisson (log link), Bernoulli (logit link), Binomial " *
            "(logit link, with prior weights). Free-dispersion families are outside M3 " *
            "scope — matches cAIC4's \"not yet supported\" warning.",
        ),
    )
end

"""
    dof_glmm_bootstrap(m::GeneralizedLinearMixedModel{T}; nboot, rng) -> T

Conditional bootstrap effective degrees of freedom for a fitted GLMM with a family
outside the Poisson Chen–Stein and Bernoulli Efron paths. The primary use case is
**binomial with `|unique(y)| > 2`** (multiple-trials binomial) and any other
canonical-link family.

```math
\\rho_{\\mathrm{boot}}
  = \\frac{1}{(B-1)\\,\\hat\\sigma^{2}}
    \\sum_{b=1}^{B} \\sum_{i=1}^{n}
      \\hat\\eta_i^{(b)}\\,\\bigl(y_i^{(b)} - \\bar y^{*}_i\\bigr),
\\quad \\hat\\sigma^2 = 1 \\text{ (canonical-link families).}
```

Each `y^{(b)} ~ f(\\hat\\mu)` is drawn directly from the conditional response
distribution: `Poisson(μ̂ᵢ)`, `Binomial(nᵢ, μ̂ᵢ)`, or `Bernoulli(μ̂ᵢ)`. The η̂^{(b)} are
the link-scale fitted values after refitting on `y^{(b)}` — one full GLMM refit per draw,
via [`MMInternals.refitglmm_eta`](@ref). The bias-correction arithmetic is the shared
[`DofLMM.efron_penalty`](@ref ConditionalAIC.DofLMM.efron_penalty) kernel with σ=1.

The ground-truth R function is the cAIC4 conditional bootstrap routine.

# Arguments
- `m`: a fitted `GeneralizedLinearMixedModel`. The original model is not mutated; all
  refits operate on deep copies (via [`MMInternals.refitglmm_eta`](@ref)).
- `nboot`: number of bootstrap draws `B ≥ 2`; default `max(n, 100)`, matching
  `cAIC4`'s `bcMer.R:54–56`.
- `rng`: random-number generator for the conditional draws; default `Xoshiro()`
  (platform-seeded, unpredictable). Pass a seeded `Xoshiro(seed)` for reproducibility.

# Returns
- `ρ::T` — the effective df. Returns `T(rank(X))` immediately if the model is fully
  singular (all variance components on the boundary), consistent with
  `cAIC4::biasCorrectionPoisson` and `biasCorrectionBernoulli` (both call
  `deleteZeroComponents` first and fall back to `zeroLessModel\$rank`).

# Throws
- `ArgumentError` for unsupported families (free-dispersion families outside the supported scope).
- `ArgumentError` if a Binomial GLMM has no prior weights.
"""
function dof_glmm_bootstrap(
    m::GeneralizedLinearMixedModel{T};
    nboot::Int=max(length(MMInternals.glmmresponse(m)), 100),
    rng::AbstractRNG=Xoshiro(),
) where {T}
    MMInternals.glmmisfullysingular(m) && return T(MMInternals.glmmfixedefrank(m))

    μhat = MMInternals.glmmfittedmu(m)
    n = length(μhat)
    B = nboot

    Ystar = glmmconddraw(rng, m, B)                  # n×B conditional draws
    Etastar = Matrix{T}(undef, n, B)
    for b in 1:B
        Etastar[:, b] = MMInternals.refitglmm_eta(m, Ystar[:, b])
    end

    return efron_penalty(μhat, one(T), Ystar, Etastar)
end

end # module DofGLMM
