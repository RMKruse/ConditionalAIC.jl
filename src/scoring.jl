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
- `hessian`: the Hessian **B**-source. `:analytic` (the default) is the closed-form B, with
  no derivative dependency. `:forwarddiff` / `:finitediff` are parsed and validated but not
  yet implemented.
- `nboot`: the number of bootstrap draws; valid only with `method = :bootstrap`.
- `sigmapenalty`: the number of estimated residual-variance parameters added to ρ — `1`
  (the default) for one estimated σ², `0` if the error variance is known.

# Returns
- A [`CAICResult`](@ref) carrying the cAIC, ρ (`dof`), the conditional log-likelihood, and
  provenance (the `method` and B-`source` actually used).

# Throws
- `ArgumentError` for an unknown `method`/`hessian`, a negative `sigmapenalty`, or `nboot`
  misuse (supplied without `method = :bootstrap`, or non-positive).
- An error stating the feature is *not yet implemented* for `method = :bootstrap` and for
  the `:forwarddiff` / `:finitediff` B-sources.

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

    # ── method / B-source resolution; not-yet-implemented paths error clearly ───────────
    resolved = method === :auto ? :steinian : method
    resolved === :bootstrap && error(
        "method = :bootstrap (conditional bootstrap) is not yet implemented (delivered in #12)",
    )
    hessian === :analytic ||
        error("hessian = :$(hessian) B-source is not yet implemented (delivered in #11); \
               only :analytic is available")

    # ── the steinian / analytic-B Gaussian scoring spine ────────────────────────────────
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
    ρ = DofLMM.dof_lmm(comps; sigmapenalty=sigmapenalty)
    ℓ = Loglik.condloglik(y, μ, MMInternals.sigmahat(m))
    return CAICResult{T,typeof(m)}(-2ℓ + 2ρ, ρ, ℓ, nothing, false, resolved, hessian)
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
