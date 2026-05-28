"""
    cAIC.DofGLMM

Bias-corrected effective degrees of freedom for the conditional AIC of generalised
linear mixed models. Implements the family-specific df estimators from
`docs/math/0006`:
  - **Bernoulli** logistic-link GLMM: Efron's Steinian per-observation estimator
    (`dof_glmm_bernoulli`, §4 of the math spec).

All `MixedModels` object access goes through [`cAIC.MMInternals`](@ref); this module
touches only pre-extracted arrays and the pure kernels.
"""
module DofGLMM

using LogExpFunctions: logit
using MixedModels: GeneralizedLinearMixedModel
using ..MMInternals

# ── Bernoulli / binary logistic GLMM (Efron's Steinian estimator) ─────────────

"""
    dof_glmm_bernoulli(m::GeneralizedLinearMixedModel{T}) -> T

Efron's Steinian bias-corrected effective degrees of freedom for a fitted Bernoulli
(binary logistic) GLMM. This is the `cAIC.jl` analogue of `cAIC4`'s
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

The estimand and algorithm are pinned in `docs/math/0006` §4. The ground-truth R
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

Pure Efron Steinian formula kernel for the Bernoulli GLMM effective df
(`docs/math/0006` §4). Given pre-computed per-flip fitted means `μhat_flip`,
the result is a deterministic function of `(y, μhat, μhat_flip)`.

This kernel is a Level-1 isolation unit (ADR-0003): it is fit-independent and can be
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

end # module DofGLMM
