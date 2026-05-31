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
    caic(m::GeneralizedLinearMixedModel; method=:auto, nboot=nothing, rng=default_rng()) -> CAICResult

Score a fitted **generalized** linear mixed model by its **conditional AIC**

```math
\\mathrm{cAIC} = -2\\,\\ell_{\\mathrm{cond}}(y \\mid \\hat b, \\hat\\beta, \\hat\\theta) + 2\\rho.
```

The conditional log-likelihood `ℓ_cond` is the log-probability of `y` under the
conditional response distribution `f(μ̂)` (Poisson: [`condloglik_poisson`](@ref
cAIC.Loglik.condloglik_poisson); Bernoulli: [`condloglik_bernoulli`](@ref
cAIC.Loglik.condloglik_bernoulli); multi-trial Binomial: [`condloglik_binomial`](@ref
cAIC.Loglik.condloglik_binomial), which deviates from `cAIC4`'s defective binomial
`getcondLL` — see `DECISIONS.md`). The effective df `ρ` is estimated by the method
selected by `method`:

- **`:auto`** (the default) dispatches by family:
  - **Poisson** → Chen–Stein influence df ([`dof_glmm_poisson`](@ref
    cAIC.DofGLMM.dof_glmm_poisson)), the `cAIC4` `biasCorrectionPoisson` analogue.
  - **Bernoulli** → Efron's Steinian df ([`dof_glmm_bernoulli`](@ref
    cAIC.DofGLMM.dof_glmm_bernoulli)), the `cAIC4` `biasCorrectionBernoulli` analogue.
  - Other families: `ArgumentError` — use `method = :bootstrap`.
- **`:bootstrap`** → conditional bootstrap df ([`dof_glmm_bootstrap`](@ref
  cAIC.DofGLMM.dof_glmm_bootstrap)). Works for every bootstrap-supported family (Poisson,
  Bernoulli, multi-trial Binomial — the families `glmmconddraw` can simulate). `nboot` sets
  the draw count (default `max(n, 100)`).

**Full-singularity shortcut.** When every variance component is on the boundary (θ = 0),
the GLMM collapses to a plain GLM: `ρ = rank(X)` is returned directly with no refit,
mirroring `cAIC4`'s `deleteZeroComponents → zeroLessModel\$rank` in both
`biasCorrectionPoisson` and `biasCorrectionBernoulli`. The `method` kwarg has no effect
on this path.

The estimand and algorithm are pinned in `docs/math/0006-glmm-bias-correction.md`.

# Arguments
- `m`: a fitted `GeneralizedLinearMixedModel`.
- `method`: df estimation method — `:auto` (default, family-dispatch) or `:bootstrap`.
- `nboot`: bootstrap draw count; valid only with `method = :bootstrap`; default
  `max(n, 100)` (matching `cAIC4::bcMer.R:54–56`).
- `rng`: random-number generator for the bootstrap draws; default `Random.default_rng()`.

# Returns
- A [`CAICResult`](@ref) carrying the cAIC, ρ (`dof`), the conditional log-likelihood,
  and provenance (`method` as given; `bsource = :na` — GLMM paths carry no Hessian
  B-source).

# Throws
- `ArgumentError` for unsupported `method`, `nboot` misuse, or an unsupported family
  under `method = :auto`.

# Example
```jldoctest
julia> using MixedModels, cAIC

julia> y = Float64[1,1,2,1, 8,9,8,9, 3,4,3,4]; g = repeat(1:3, inner=4);

julia> m = fit(MixedModel, @formula(y ~ 1 + (1|g)), (; y, g), Poisson(); progress=false);

julia> r = caic(m); r.caic ≈ -2 * r.condloglik + 2 * r.dof
true
```
"""
function caic(
    m::GeneralizedLinearMixedModel{T,D};
    method::Symbol=:auto,
    nboot::Union{Int,Nothing}=nothing,
    rng::AbstractRNG=default_rng(),
) where {T,D}
    method in (:auto, :bootstrap) || throw(
        ArgumentError("method for GLMM caic must be :auto or :bootstrap; got :$(method)"),
    )
    if nboot !== nothing
        method === :bootstrap || throw(
            ArgumentError(
                "nboot is only valid with method = :bootstrap; got method = :$(method)"
            ),
        )
        nboot > 0 || throw(ArgumentError("nboot must be positive; got $(nboot)"))
    end

    # Full-singularity: every θ = 0 → GLMM collapses to plain GLM; ρ = rank(X), no σ-penalty.
    # No refit (b̂ = 0 ⇒ μ̂ = Xβ̂ already), mirroring `cAIC4`'s `deleteZeroComponents → glm`.
    if MMInternals.glmmisfullysingular(m)
        ℓ = _glmm_condll(m)
        ρ = T(MMInternals.glmmfixedefrank(m))
        return CAICResult{T,GeneralizedLinearMixedModel{T,D}}(
            -2ℓ + 2ρ, ρ, ℓ, nothing, false, method, :na
        )
    end

    # Partial-singularity: SOME directions on the boundary. Drop them, refit the reduced GLMM,
    # and cascade — a reduced refit may itself be singular — until a non-singular model is
    # reached; score THAT (the singular fit's df is degenerate). Mirrors the LMM cascade and
    # `cAIC4`'s `deleteZeroComponents` recursion. If the reduction fully collapses (no direction
    # survives), fall back to the rank(X) plain-GLM df on the collapsed fit (its b̂ = 0 ⇒
    # μ̂ = Xβ̂), with no reduced model carried.
    if MMInternals.issingular(m)
        mr = m
        while MMInternals.issingular(mr)
            next = MMInternals.reduceboundary(mr)
            if next === nothing
                ℓ = _glmm_condll(mr)
                ρ = T(MMInternals.glmmfixedefrank(mr))
                return CAICResult{T,GeneralizedLinearMixedModel{T,D}}(
                    -2ℓ + 2ρ, ρ, ℓ, nothing, false, method, :na
                )
            end
            mr = next
        end
        ℓ = _glmm_condll(mr)
        ρ = _glmm_score_df(mr, method, nboot, rng)
        return CAICResult{T,GeneralizedLinearMixedModel{T,D}}(
            -2ℓ + 2ρ, ρ, ℓ, mr, true, method, :na
        )
    end

    # Non-singular: score it as given.
    ℓ = _glmm_condll(m)
    ρ = _glmm_score_df(m, method, nboot, rng)
    return CAICResult{T,GeneralizedLinearMixedModel{T,D}}(
        -2ℓ + 2ρ, ρ, ℓ, nothing, false, method, :na
    )
end

# Conditional log-likelihood of a (possibly reduced) GLMM fit, via family dispatch.
# `glmmpriorweights` carries the per-observation binomial trial counts (empty for
# Poisson/Bernoulli); only the Binomial branch consumes them.
function _glmm_condll(m::GeneralizedLinearMixedModel)
    return _glmm_condloglik_dispatch(
        MMInternals.glmmdist(m),
        MMInternals.glmmresponse(m),
        MMInternals.glmmfittedmu(m),
        MMInternals.glmmpriorweights(m),
    )
end

# Effective df of a non-singular GLMM fit under the requested method.
function _glmm_score_df(
    m::GeneralizedLinearMixedModel,
    method::Symbol,
    nboot::Union{Int,Nothing},
    rng::AbstractRNG,
)
    if method === :bootstrap
        ndraws = nboot !== nothing ? nboot : max(length(MMInternals.glmmresponse(m)), 100)
        return DofGLMM.dof_glmm_bootstrap(m; nboot=ndraws, rng=rng)
    end
    return _glmm_df_auto(m, MMInternals.glmmdist(m))
end

# Family dispatch for the GLMM conditional log-likelihood. `wts` are the binomial trial
# counts (`glmmpriorweights`); ignored by Poisson/Bernoulli, consumed by Binomial.
function _glmm_condloglik_dispatch(::Poisson, y, μ, wts)
    return Loglik.condloglik_poisson(y, μ)
end
function _glmm_condloglik_dispatch(::Bernoulli, y, μ, wts)
    return Loglik.condloglik_bernoulli(y, μ)
end
function _glmm_condloglik_dispatch(::Binomial, y, μ, wts)
    # Multi-trial binomial — the correct binomial density, deviating from cAIC4's defective
    # getcondLL (DECISIONS.md 2026-05-29; docs/math/0006 §1.1). nᵢ are the prior weights.
    return Loglik.condloglik_binomial(y, μ, wts)
end
function _glmm_condloglik_dispatch(d, y, μ, wts)
    throw(
        ArgumentError(
            "caic: unsupported GLMM family $(typeof(d)). Supported conditional \
             log-likelihoods: Poisson (log link), Bernoulli and multi-trial Binomial \
             (logit link)."
        ),
    )
end

# Family dispatch for the GLMM df estimator (method = :auto path).
_glmm_df_auto(m::GeneralizedLinearMixedModel, ::Poisson) = DofGLMM.dof_glmm_poisson(m)
_glmm_df_auto(m::GeneralizedLinearMixedModel, ::Bernoulli) = DofGLMM.dof_glmm_bernoulli(m)
function _glmm_df_auto(::GeneralizedLinearMixedModel, d)
    throw(
        ArgumentError(
            "caic: GLMM family $(typeof(d)) has no analytic df estimator for \
             method=:auto. Use method=:bootstrap for non-Poisson/Bernoulli families."
        ),
    )
end

# ── `lm`/`glm` terminal scoring (M4, ADR-0006) ────────────────────────────────────────────
# A backward `stepcaic` search drops random-effects terms one at a time; dropping the last RE
# term yields a fixed-effects-only model `MixedModels.jl` cannot represent, so the terminal of
# the search is a plain `GLM.jl` `lm`/`glm` fit, scored here exactly as `cAIC4`'s `(g)lm` branch
# (`cAIC4:::cAIC`): `df = rank + 1`, `cll = Σ` family log-density at μ̂, `caic = −2·cll + 2·df`,
# reusing the `Loglik` kernels. This is **not** a Greven–Kneib / Steinian / bootstrap path — no
# bias correction is applied — so the provenance is recorded as `method = :terminal`,
# `bsource = :na`. Fitting/scoring the terminal touches no `MixedModels` internals (public
# `GLM.jl` accessors only), so `MMInternals` is not involved. The estimand is pinned in
# `docs/math/0008-stepcaic-search.md §0`.

"""
    caic(m::RegressionModel) -> CAICResult

Score a fixed-effects-only `GLM.jl` `lm`/`glm` fit — the **terminal node** a backward
[`stepcaic`](@ref) search reaches once the last random-effects term is dropped — by the same
conditional AIC `cAIC4` assigns its `(g)lm` branch:

```math
\\mathrm{cAIC} = -2\\,\\ell + 2\\,(\\mathrm{rank} + 1),
```

with `ℓ` the marginal (here = conditional, no random effects) log-likelihood of `y` under the
fitted family at the fitted mean μ̂, and the penalty `rank + 1` (the fixed-effect rank plus one
estimated dispersion/σ²). No Greven–Kneib / Steinian / bootstrap bias correction is involved —
the terminal is a deterministic closed form — so the result carries `method = :terminal` and
`bsource = :na`, and `reducedmodel = nothing` / `refit = false` (the terminal is never singular).

Supported terminals mirror M2/M3: the Gaussian `lm` (σ̂ the MLE rescaling
`summary(·)\$sigma·√((n−p)/n)`, i.e. `√(RSS/n)`), and the Poisson / Bernoulli / multi-trial
Binomial `glm`. The Binomial branch reuses [`condloglik_binomial`](@ref
cAIC.Loglik.condloglik_binomial) (the corrected density, deviating from `cAIC4`'s defective
multi-trial `getcondLL`; see `DECISIONS.md`); for the Bernoulli case it reduces to and matches
`cAIC4` exactly. The estimand is pinned in `docs/math/0008-stepcaic-search.md §0`.

# Arguments
- `m`: a fitted `GLM.jl` `lm`/`glm` model (a `TableRegressionModel`).

# Returns
- A [`CAICResult`](@ref) carrying the cAIC, the penalty `ρ = rank + 1` (`dof`), the
  log-likelihood (`condloglik`), and provenance (`method = :terminal`, `bsource = :na`).

# Throws
- `ArgumentError` for a `glm` family with no supported conditional log-likelihood (only
  Gaussian `lm`, Poisson, Bernoulli, and Binomial `glm` are supported).

# Example
```jldoctest
julia> using GLM, cAIC

julia> data = (; x=[-1.0, -0.3, 0.2, 0.8, 1.4], y=[0.1, 0.9, 1.6, 2.1, 3.0]);

julia> r = caic(lm(@formula(y ~ 1 + x), data));

julia> r.caic ≈ -2 * r.condloglik + 2 * r.dof && r.dof == 3
true
```
"""
function caic(m::TableRegressionModel{<:LinearModel})
    y = response(m)
    μ = predict(m)
    T = float(eltype(y))
    n = length(y)
    ρ = T(_terminalrank(m) + 1)               # cAIC4: df = rank + 1
    σ = sqrt(T(deviance(m)) / n)               # MLE σ̂ = √(RSS/n); deviance(lm) = RSS
    ℓ = Loglik.condloglik(y, μ, σ)
    return CAICResult{T,typeof(m)}(-2ℓ + 2ρ, ρ, ℓ, nothing, false, :terminal, :na)
end

# Fixed-effect rank of a GLM.jl terminal — the `cAIC4` `object$rank`. For the full-rank
# fixed-effects designs a `stepcaic` terminal carries, this is the number of coefficients
# (`docs/math/0008 §0`); a rank-deficient design is out of scope for the search terminal.
_terminalrank(m::TableRegressionModel) = length(coef(m))

# Response family of a fitted `glm`, for terminal-scoring dispatch. `m.model` is the wrapped
# `GeneralizedLinearModel`; its `rr.d` is the response distribution instance. GLM is exact-pinned
# (DECISIONS 2026-05-30), so this field is stable for the supported version.
_glmfamily(m::TableRegressionModel{<:GeneralizedLinearModel}) = m.model.rr.d

# `glm` terminal: score by family, reusing the `Loglik` GLMM kernels at the fitted mean μ̂.
# df = rank + 1 (cAIC4's `(g)lm` branch), as for the Gaussian `lm`.
caic(m::TableRegressionModel{<:GeneralizedLinearModel}) = _glm_terminal(m, _glmfamily(m))

function _glm_terminal(m::TableRegressionModel, ::Poisson)
    y = response(m)
    μ = predict(m)
    T = float(eltype(y))
    ρ = T(_terminalrank(m) + 1)
    ℓ = Loglik.condloglik_poisson(y, μ)
    return CAICResult{T,typeof(m)}(-2ℓ + 2ρ, ρ, ℓ, nothing, false, :terminal, :na)
end

function _glm_terminal(m::TableRegressionModel, ::Bernoulli)
    y = response(m)
    μ = predict(m)
    T = float(eltype(y))
    ρ = T(_terminalrank(m) + 1)
    ℓ = Loglik.condloglik_bernoulli(y, μ)
    return CAICResult{T,typeof(m)}(-2ℓ + 2ρ, ρ, ℓ, nothing, false, :terminal, :na)
end

function _glm_terminal(m::TableRegressionModel, ::Binomial)
    # Multi-trial Binomial — the documented DEVIATION (DECISIONS 2026-05-29 / 2026-05-30): cAIC4's
    # binomial getcondLL is −∞ for nᵢ > 1, so this reuses the corrected `condloglik_binomial` at the
    # true per-observation trial counts, exactly as the M3 GLMM binomial path does. The response is
    # the success proportion kᵢ/nᵢ and the trial counts nᵢ are the fit's prior weights.
    y = response(m)
    μ = predict(m)
    n = m.model.rr.wts
    T = float(eltype(y))
    ρ = T(_terminalrank(m) + 1)
    ℓ = Loglik.condloglik_binomial(y, μ, n)
    return CAICResult{T,typeof(m)}(-2ℓ + 2ρ, ρ, ℓ, nothing, false, :terminal, :na)
end

function _glm_terminal(m::TableRegressionModel, d)
    throw(
        ArgumentError(
            "caic: unsupported glm terminal family $(typeof(d)). Supported terminals: the \
             Gaussian `lm`, and Poisson / Bernoulli / multi-trial Binomial `glm`."
        ),
    )
end
