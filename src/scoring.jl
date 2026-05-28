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
    caic(m::LinearMixedModel; method=:auto, hessian=:analytic, nboot=nothing, sigmapenalty=1)
        -> CAICResult

Score a fitted Gaussian linear mixed model by its **conditional AIC**

```math
\\mathrm{cAIC} = -2\\,\\ell_{\\mathrm{cond}}(y \\mid \\hat b, \\hat\\beta, \\hat\\theta) + 2\\rho,
```

the analogue of `cAIC4`'s `cAIC`. The conditional log-likelihood `ℓ_cond` is the Gaussian
density of `y` about the conditional fitted mean `ŷ = X β̂ + Z b̂`
([`condloglik`](@ref cAIC.Loglik.condloglik)); `ρ` is the Greven–Kneib bias-corrected
effective degrees of freedom ([`dof_lmm`](@ref cAIC.DofLMM.dof_lmm)), which exceeds the
naive plug-in `ρ₀ = tr(H₁)` because the variance parameters `θ` are estimated. The
mathematics is pinned in `docs/math/0002-gaussian-bias-correction.md` (the bias correction)
and `0003-conditional-loglik.md` (the log-likelihood).

The computation is performed on the fit *as given*, dispatching on `m.optsum.REML` (no
force-refit).

# Arguments
- `m`: a fitted Gaussian `LinearMixedModel`.
- `method`: the degrees-of-freedom method. `:auto` (the default) resolves to `:steinian`
  for the Gaussian family — the analytic Greven–Kneib correction. `:bootstrap` is parsed
  and validated but not yet implemented.
- `hessian`: the Hessian **B**-source — how the Greven–Kneib Hessian B is obtained.
  `:analytic` (the default) is the closed-form B, with no derivative dependency.
  `:finitediff` self-drives finite differences over `MixedModels`' stable objective;
  `:forwarddiff` rides the experimental `MixedModelsForwardDiffExt`. The numeric sources are
  three estimators of the same ρ that diverge as documented in `DECISIONS.md`
  (`:finitediff` reproduces `cAIC4`'s `analytic = FALSE`; `:forwarddiff` differs by holding
  σ̂² fixed). See `docs/math/0004-numeric-hessian-bsources.md` and ADR-0002.
- `nboot`: the number of bootstrap draws; valid only with `method = :bootstrap`.
- `sigmapenalty`: the number of estimated residual-variance parameters added to ρ — `1`
  (the default) for one estimated σ², `0` if the error variance is known.

# Returns
- A [`CAICResult`](@ref) carrying the cAIC, ρ (`dof`), the conditional log-likelihood, and
  provenance (the `method` and B-`source` actually used).

# Throws
- `ArgumentError` for an unknown `method`/`hessian`, a negative `sigmapenalty`, or `nboot`
  misuse (supplied without `method = :bootstrap`, or non-positive).
- An error stating the feature is *not yet implemented* for `method = :bootstrap`.

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

    # ── method resolution; the not-yet-implemented bootstrap path errors clearly ────────
    resolved = method === :auto ? :steinian : method
    resolved === :bootstrap && error(
        "method = :bootstrap (conditional bootstrap) is not yet implemented (delivered in #12)",
    )

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
                    -2ℓ + 2ρ, ρ, ℓ, nothing, false, resolved, hessian
                )
            end
            mr = next
        end
        ρ, ℓ = _steinian(mr, sigmapenalty, hessian)
        return CAICResult{T,LinearMixedModel{T}}(
            -2ℓ + 2ρ, ρ, ℓ, mr, true, resolved, hessian
        )
    end

    # ── non-singular fit: score it as given ──────────────────────────────────────────────
    ρ, ℓ = _steinian(m, sigmapenalty, hessian)
    return CAICResult{T,LinearMixedModel{T}}(
        -2ℓ + 2ρ, ρ, ℓ, nothing, false, resolved, hessian
    )
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
    caic(m::GeneralizedLinearMixedModel; kwargs...)

Scoring a generalised (non-Gaussian) mixed model is not yet supported — the GLMM bias
correction is milestone M3. Raises `ArgumentError`.
"""
function caic(m::GeneralizedLinearMixedModel; kwargs...)
    return throw(
        ArgumentError(
            "caic currently supports only Gaussian LinearMixedModel fits; GLMM scoring is \
             not yet implemented (M3)"
        ),
    )
end
