# The conditional-AIC scoring assembly (the `caic` methods). Included directly into the
# `cAIC` module: these methods extend the `caic` generic and wire the spine together —
#   MMInternals (extract) → Components (build) → DofLMM (ρ) + Loglik (ℓ) → CAICResult.
# All `MixedModels`-object access is via `MMInternals`; this file touches only its
# extracted arrays and the pure kernels.
#
# (Named `scoring.jl`, not `caic.jl`: the latter collides with the module entry `cAIC.jl`
# on a case-insensitive filesystem.)

const _METHODS = (:auto, :steinian, :bootstrap)
const _BSOURCES = (:analytic, :forwarddiff, :finitediff)

"""
    caic(m::LinearMixedModel; method=:auto, hessian=:analytic, nboot=nothing,
         sigmapenalty=1, rng=default_rng()) -> CAICResult

Score a fitted Gaussian linear mixed model by its **conditional AIC**

```math
\\mathrm{cAIC} = -2\\,\\ell_{\\mathrm{cond}}(y \\mid \\hat b, \\hat\\beta, \\hat\\theta) + 2\\rho,
```

the analogue of `cAIC4`'s `cAIC`. The conditional log-likelihood `ℓ_cond` is the Gaussian
density of `y` about the conditional fitted mean `ŷ = X β̂ + Z b̂`
([`condloglik`](@ref cAIC.Loglik.condloglik)); `ρ` is the bias-corrected effective degrees
of freedom, computed by the selected `method`. The mathematics is pinned in
`docs/math/0002-gaussian-bias-correction.md` (Greven–Kneib correction) and
`0003-conditional-loglik.md` (the log-likelihood).

The computation is performed on the fit *as given*, dispatching on `m.optsum.REML` (no
force-refit).

# Arguments
- `m`: a fitted Gaussian `LinearMixedModel`.
- `method`: the degrees-of-freedom method.
  - `:auto` (the default) resolves to `:steinian` for the Gaussian family — the analytic
    Greven–Kneib correction.
  - `:steinian` — the analytic Greven–Kneib bias correction.
  - `:bootstrap` — Efron's parametric-bootstrap covariance penalty; requires `nboot`.
- `hessian`: the Hessian **B**-source for `:steinian` — how the Greven–Kneib Hessian B is
  obtained. `:analytic` (the default) is the closed-form B. `:finitediff` self-drives finite
  differences; `:forwarddiff` rides the experimental `MixedModelsForwardDiffExt`. Ignored
  when `method = :bootstrap` (no Hessian B used; `bsource` is `:na` in the result).
- `nboot`: the number of parametric-bootstrap draws; required (and valid only) when
  `method = :bootstrap`.
- `sigmapenalty`: the number of estimated residual-variance parameters added to ρ — `1`
  (the default) for one estimated σ², `0` if the error variance is known.
- `rng`: an `AbstractRNG` for the bootstrap draws; defaults to `Random.default_rng()`.
  Pass a seeded `Xoshiro` (e.g. `rng=Xoshiro(42)`) for reproducibility.

# Returns
- A [`CAICResult`](@ref) carrying the cAIC, ρ (`dof`), the conditional log-likelihood, and
  provenance (the `method` and B-`source` actually used; `bsource = :na` for `:bootstrap`).

# Throws
- `ArgumentError` for an unknown `method`/`hessian`, a negative `sigmapenalty`, or `nboot`
  misuse (supplied without `method = :bootstrap`, or non-positive).

# Example
```jldoctest
julia> using MixedModels, cAIC

julia> m = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)),
               MixedModels.dataset(:sleepstudy); REML=false, progress=false);

julia> r = caic(m);

julia> r.caic ≈ -2 * r.condloglik + 2 * r.dof
true
```
"""
function caic(
    m::LinearMixedModel{T};
    method::Symbol=:auto,
    hessian::Symbol=:analytic,
    nboot::Union{Int,Nothing}=nothing,
    sigmapenalty::Integer=1,
    rng::AbstractRNG=default_rng(),
) where {T}
    # ── option validation (fail loudly; CLAUDE §4) ──────────────────────────────────────
    method in _METHODS ||
        throw(ArgumentError("method must be one of $(_METHODS); got :$(method)"))
    hessian in _BSOURCES ||
        throw(ArgumentError("hessian must be one of $(_BSOURCES); got :$(hessian)"))
    sigmapenalty >= 0 ||
        throw(ArgumentError("sigmapenalty must be ≥ 0; got $(sigmapenalty)"))
    if nboot !== nothing
        method === :bootstrap || throw(
            ArgumentError(
                "nboot is only valid with method = :bootstrap; got method = :$(method)"
            ),
        )
        nboot > 0 || throw(ArgumentError("nboot must be positive; got $(nboot)"))
    end

    resolved = method === :auto ? :steinian : method
    ndraws = resolved === :bootstrap ? (nboot !== nothing ? nboot : 500) : 0
    actual_bsource = resolved === :bootstrap ? :na : hessian

    # ── singular fit: drop the boundary components and score the reduced refit ───────────
    # A variance component estimated on the boundary makes the bias-correction spine
    # degenerate (a nonsensical, even negative, ρ). Mirroring `cAIC4`'s drop-and-refit
    # (`biasCorrectionGaussian` → `deleteZeroComponents`), the boundary directions are
    # removed and the cAIC is computed on the reduced model. The reduction cascades — a
    # reduced refit may itself be singular — until a non-singular model is reached.
    if MMInternals.issingular(m)
        mr = m
        while MMInternals.issingular(mr)
            next = MMInternals.reduceboundary(mr)
            # All random-effect directions on the boundary → no random-effects model remains.
            # Mirror `cAIC4`'s `lm` branch: score the fixed-effects-only fit. At b̂ = 0 the
            # conditional mean is ŷ = Xβ̂ (so `condloglik` on the original fit is exactly
            # `cAIC4`'s `getcondLL(original)`), and ρ = p + sigmapenalty (rank of the fixed
            # effects plus the estimated σ²). No reduced model is carried (`refit = false`).
            if next === nothing
                p = size(MMInternals.fixedeffects(m), 2)
                ρ = T(p + sigmapenalty)
                ℓ = Loglik.condloglik(
                    MMInternals.responsevec(m),
                    MMInternals.conditionalmean(m),
                    MMInternals.sigmahat(m),
                )
                return CAICResult{T,LinearMixedModel{T}}(
                    -2ℓ + 2ρ, ρ, ℓ, nothing, false, resolved, actual_bsource
                )
            end
            mr = next
        end
        ρ, ℓ = if resolved === :bootstrap
            _bootstrap(mr, ndraws, sigmapenalty, rng)
        else
            _steinian(mr, sigmapenalty, hessian)
        end
        return CAICResult{T,LinearMixedModel{T}}(
            -2ℓ + 2ρ, ρ, ℓ, mr, true, resolved, actual_bsource
        )
    end

    # ── non-singular fit: score it as given ──────────────────────────────────────────────
    ρ, ℓ = if resolved === :bootstrap
        _bootstrap(m, ndraws, sigmapenalty, rng)
    else
        _steinian(m, sigmapenalty, hessian)
    end
    return CAICResult{T,LinearMixedModel{T}}(
        -2ℓ + 2ρ, ρ, ℓ, nothing, false, resolved, actual_bsource
    )
end

# The bootstrap Gaussian scoring spine: draw `ndraws` parametric bootstrap samples from the
# fitted mean and sigma, refit each, and return the Efron penalty ρ and the conditional
# log-likelihood ℓ of the original fit.
function _bootstrap(
    m::LinearMixedModel{T}, ndraws::Int, sigmapenalty::Integer, rng::AbstractRNG
) where {T}
    y = MMInternals.responsevec(m)
    μ = MMInternals.conditionalmean(m)
    σ = MMInternals.sigmahat(m)
    n = length(y)
    Ystar = μ .+ σ .* randn(rng, T, n, ndraws)
    Yhatstar = Matrix{T}(undef, n, ndraws)
    for b in 1:ndraws
        Yhatstar[:, b] = MMInternals.bootstrapfit(m, Ystar[:, b])
    end
    ρ = DofLMM.efron_penalty(μ, σ, Ystar, Yhatstar, sigmapenalty)
    ℓ = Loglik.condloglik(y, μ, σ)
    return ρ, ℓ
end

# The steinian Gaussian scoring spine: extract the fit's quantities via the `MMInternals`
# quarantine, build the Gaussian components, and return the bias-corrected effective degrees
# of freedom ρ and the conditional log-likelihood ℓ. Shared by the non-singular path and the
# reduced-model (singular) path. The B-source selects how the Greven–Kneib Hessian B is
# obtained: `:analytic` from the closed form ([`dof_lmm`](@ref cAIC.DofLMM.dof_lmm)),
# `:forwarddiff` / `:finitediff` numerically ([`bhessian`](@ref cAIC.MMInternals.bhessian),
# fed to [`dof_lmm_numeric`](@ref cAIC.DofLMM.dof_lmm_numeric)). All feed the *same* assembly.
function _steinian(m::LinearMixedModel{T}, sigmapenalty::Integer, hessian::Symbol) where {T}
    y = MMInternals.responsevec(m)
    μ = MMInternals.conditionalmean(m)
    comps = Components.gaussiancomponents(
        MMInternals.fixedeffects(m),
        y,
        μ,
        MMInternals.retermdesigns(m),
        MMInternals.retermlambdas(m),
        MMInternals.parmap(m),
        MMInternals.reml(m),
    )
    ρ = if hessian === :analytic
        DofLMM.dof_lmm(comps; sigmapenalty=sigmapenalty)
    else
        B = MMInternals.bhessian(m, hessian)
        DofLMM.dof_lmm_numeric(comps, B; sigmapenalty=sigmapenalty)
    end
    ℓ = Loglik.condloglik(y, μ, MMInternals.sigmahat(m))
    return ρ, ℓ
end

"""
    caic(m::GeneralizedLinearMixedModel) -> CAICResult

Score a fitted **generalized** linear mixed model by its **conditional AIC**.

This method currently implements **only the full-singularity fallback** (milestone M3,
issue #27): when every random-effect variance component is on the boundary (`θ = 0`),
the GLMM collapses to a plain GLM and the effective degrees of freedom is

```math
\\rho = \\operatorname{rank}(X),
```

with **no σ-penalty** (canonical-link Poisson and Bernoulli have fixed dispersion = 1).
This mirrors `cAIC4`'s `biasCorrectionPoisson.R:14–16` and
`biasCorrectionBernoulli.R:11–13` (both return `zeroLessModel\$rank` when
`deleteZeroComponents` reduces the model to a plain GLM).

For non-fully-singular fits (partial or non-singular) this method raises `ArgumentError`
— the Poisson Chen–Stein influence path, the Bernoulli Efron estimator, and the
bootstrap fallback are later M3 issues.

# Supported families (M3 scope)
- **Poisson** (log link): ρ = rank(X).
- **Bernoulli / binomial (binary)** (logit link): ρ = rank(X).

# Throws
- `ArgumentError` for non-fully-singular GLMM fits (general M3 path not yet implemented).
- `ArgumentError` for unsupported distribution families (free-dispersion families are
  outside M3 scope, matching `cAIC4`'s own "not yet supported" warning).
"""
function caic(m::GeneralizedLinearMixedModel{T,D}; kwargs...) where {T,D}
    MMInternals.glmmisfullysingular(m) || throw(
        ArgumentError(
            "caic: GLMM scoring for non-fully-singular fits is not yet implemented (M3). \
             Only the full-singularity fallback (all θ = 0 → ρ = rank(X)) is supported.",
        ),
    )
    p = MMInternals.glmmfixedefrank(m)
    ρ = T(p)
    μ = MMInternals.glmmfittedmu(m)
    y = MMInternals.glmmresponse(m)
    d = MMInternals.glmmdist(m)
    ℓ = _glmm_condloglik_dispatch(d, y, μ)
    return CAICResult{T,GeneralizedLinearMixedModel{T,D}}(
        -2ℓ + 2ρ, ρ, ℓ, nothing, false, :auto, :na
    )
end

# Family dispatch for the GLMM conditional log-likelihood (full-singularity path).
# Only Poisson and Bernoulli are in M3 scope; all other families raise ArgumentError.
function _glmm_condloglik_dispatch(::Poisson, y, μ)
    return Loglik.condloglik_poisson(y, μ)
end
function _glmm_condloglik_dispatch(::Bernoulli, y, μ)
    return Loglik.condloglik_bernoulli(y, μ)
end
function _glmm_condloglik_dispatch(d, y, μ)
    throw(
        ArgumentError(
            "caic: unsupported GLMM family $(typeof(d)). Only Poisson (log link) and \
             Bernoulli (logit link) are in M3 scope."
        ),
    )
end
