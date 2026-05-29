# Tests for caic(::GeneralizedLinearMixedModel) — general scoring assembly (issue #31, M3).
#
# Verifies that the scoring assembly correctly wires the GLMM df estimators
# (dof_glmm_poisson, dof_glmm_bernoulli, dof_glmm_bootstrap) into CAICResult for
# non-singular fits. Correctness of the df estimators themselves is in dof_glmm_tests.jl;
# these tests cover the assembly contract (correct fields, cAIC identity, method dispatch).
#
# Math spec: docs/math/0006-glmm-bias-correction.md.
# Ground truth: cAIC4 v1.1, biasCorrectionPoisson / biasCorrectionBernoulli / conditionalBootstrap.

# ── synthetic datasets ────────────────────────────────────────────────────────────────────
#
# Poisson: 5 groups × 4 obs, strong between-group variation → non-singular fit.
#   Log-group-means ≈ 0.22, 2.14, 1.25, 1.70, 0.22 (SD ≈ 0.76 on log scale).
# Bernoulli: 4 groups × 8 obs, strong between-group variation → non-singular fit.
#   Logit-group-proportions ≈ -1.95, +1.95, 0.00, +1.10.

# ── Poisson: non-singular scoring ─────────────────────────────────────────────

@testitem "caic on non-singular Poisson GLMM: no throw, cAIC identity, correct metadata" tags = [
    :glmm, :level2
] begin
    # Tracer bullet (issue #31): caic must no longer throw for a non-singular Poisson GLMM.
    # The cAIC identity −2ℓ + 2ρ must hold and all provenance fields must be correct.
    using MixedModels
    using cAIC

    y = Float64[1, 1, 2, 1, 8, 9, 8, 9, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 1]
    g = repeat(1:5, inner=4)
    m = fit(MixedModel, @formula(y ~ 1 + (1 | g)), (; y, g), Poisson(); progress=false)
    @test !issingular(m)

    r = caic(m)
    @test r isa CAICResult{Float64,<:GeneralizedLinearMixedModel}
    @test r.method == :auto
    @test r.bsource == :na
    @test !r.refit
    @test r.reducedmodel === nothing
    @test r.dof > 0
    @test isfinite(r.condloglik)
    @test isfinite(r.caic)
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof rtol = 1e-12
end

@testitem "caic on non-singular Poisson GLMM: dof wired to dof_glmm_poisson" tags = [
    :glmm, :level2
] begin
    # Assembly wiring test: caic must delegate df to DofGLMM.dof_glmm_poisson (Chen-Stein)
    # and the log-lik to Loglik.condloglik_poisson.
    using MixedModels
    using cAIC
    using cAIC: DofGLMM, Loglik, MMInternals

    y = Float64[1, 1, 2, 1, 8, 9, 8, 9, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 1]
    g = repeat(1:5, inner=4)
    m = fit(MixedModel, @formula(y ~ 1 + (1 | g)), (; y, g), Poisson(); progress=false)
    @test !issingular(m)

    ρ_ref = DofGLMM.dof_glmm_poisson(m)
    ℓ_ref = Loglik.condloglik_poisson(
        MMInternals.glmmresponse(m), MMInternals.glmmfittedmu(m)
    )

    r = caic(m)
    @test r.dof ≈ ρ_ref rtol = 1e-12
    @test r.condloglik ≈ ℓ_ref rtol = 1e-12
end

# ── Bernoulli: non-singular scoring ───────────────────────────────────────────

@testitem "caic on non-singular Bernoulli GLMM: no throw, cAIC identity, correct metadata" tags = [
    :glmm, :level2
] begin
    # caic must no longer throw for a non-singular Bernoulli GLMM.
    using MixedModels
    using cAIC

    y = Float64[
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
    ]
    g = repeat(1:4, inner=8)
    m = fit(MixedModel, @formula(y ~ 1 + (1 | g)), (; y, g), Bernoulli(); progress=false)
    @test !issingular(m)

    r = caic(m)
    @test r isa CAICResult{Float64,<:GeneralizedLinearMixedModel}
    @test r.method == :auto
    @test r.bsource == :na
    @test !r.refit
    @test r.reducedmodel === nothing
    @test r.dof > 0
    @test isfinite(r.condloglik)
    @test isfinite(r.caic)
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof rtol = 1e-12
end

@testitem "caic on non-singular Bernoulli GLMM: dof wired to dof_glmm_bernoulli" tags = [
    :glmm, :level2
] begin
    # Assembly wiring test: caic must delegate df to DofGLMM.dof_glmm_bernoulli (Efron).
    using MixedModels
    using cAIC
    using cAIC: DofGLMM, Loglik, MMInternals

    y = Float64[
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
    ]
    g = repeat(1:4, inner=8)
    m = fit(MixedModel, @formula(y ~ 1 + (1 | g)), (; y, g), Bernoulli(); progress=false)
    @test !issingular(m)

    ρ_ref = DofGLMM.dof_glmm_bernoulli(m)
    ℓ_ref = Loglik.condloglik_bernoulli(
        MMInternals.glmmresponse(m), MMInternals.glmmfittedmu(m)
    )

    r = caic(m)
    @test r.dof ≈ ρ_ref rtol = 1e-12
    @test r.condloglik ≈ ℓ_ref rtol = 1e-12
end

# ── method = :bootstrap override ──────────────────────────────────────────────

@testitem "caic GLMM method=:bootstrap: dispatches to bootstrap df, cAIC identity holds" tags = [
    :glmm, :level2, :bootstrap
] begin
    # method=:bootstrap must override family dispatch and use dof_glmm_bootstrap.
    # nboot=20 for speed; correctness of bootstrap df is tested in dof_glmm_tests.jl.
    using MixedModels
    using Random: Xoshiro
    using cAIC

    y = Float64[1, 1, 2, 1, 8, 9, 8, 9, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 1]
    g = repeat(1:5, inner=4)
    m = fit(MixedModel, @formula(y ~ 1 + (1 | g)), (; y, g), Poisson(); progress=false)
    @test !issingular(m)

    r = caic(m; method=:bootstrap, nboot=20, rng=Xoshiro(42))
    @test r isa CAICResult{Float64,<:GeneralizedLinearMixedModel}
    @test r.method == :bootstrap
    @test r.bsource == :na
    @test r.dof > 0
    @test isfinite(r.caic)
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof rtol = 1e-12
end

@testitem "caic GLMM method=:bootstrap on multi-trial Binomial: finite cAIC, identity holds" tags = [
    :glmm, :level2, :bootstrap
] begin
    # Multi-trial Binomial (CBPP) has no analytic df and cAIC4's getcondLL is defective
    # for it (DECISIONS.md 2026-05-29). The bootstrap path must nonetheless produce a
    # FINITE conditional log-likelihood (via condloglik_binomial) and a finite cAIC
    # satisfying the identity — i.e. the path is reachable end-to-end through caic, not
    # only through DofGLMM.dof_glmm_bootstrap. nboot=20 for speed.
    using MixedModels
    using Random: Xoshiro
    using cAIC

    cbpp = MixedModels.dataset(:cbpp)
    m = fit(
        MixedModel,
        @formula(incid / hsz ~ period + (1 | herd)),
        cbpp,
        Binomial();
        weights=float.(cbpp.hsz),
        progress=false,
    )
    @test !issingular(m)

    r = caic(m; method=:bootstrap, nboot=20, rng=Xoshiro(42))
    @test r isa CAICResult{Float64,<:GeneralizedLinearMixedModel}
    @test r.method == :bootstrap
    @test r.bsource == :na
    @test r.dof > 0
    @test isfinite(r.condloglik)   # the bug: this used to throw before any df was computed
    @test isfinite(r.caic)
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof rtol = 1e-12
end

# ── argument validation ────────────────────────────────────────────────────────

@testitem "caic GLMM raises ArgumentError for unsupported method" tags = [:glmm, :level2] begin
    using MixedModels
    using cAIC

    y = Float64[1, 1, 2, 1, 8, 9, 8, 9, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 1]
    g = repeat(1:5, inner=4)
    m = fit(MixedModel, @formula(y ~ 1 + (1 | g)), (; y, g), Poisson(); progress=false)

    @test_throws ArgumentError caic(m; method=:steinian)
    @test_throws ArgumentError caic(m; method=:analytic)
    @test_throws ArgumentError caic(m; method=:unknown)
end

@testitem "caic GLMM raises ArgumentError for nboot misuse" tags = [:glmm, :level2] begin
    using MixedModels
    using cAIC

    y = Float64[1, 1, 2, 1, 8, 9, 8, 9, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 1]
    g = repeat(1:5, inner=4)
    m = fit(MixedModel, @formula(y ~ 1 + (1 | g)), (; y, g), Poisson(); progress=false)

    @test_throws ArgumentError caic(m; nboot=100)                     # nboot without :bootstrap
    @test_throws ArgumentError caic(m; method=:bootstrap, nboot=0)   # non-positive nboot
    @test_throws ArgumentError caic(m; method=:bootstrap, nboot=-5)  # negative nboot
end

@testitem "caic GLMM raises ArgumentError for unsupported family with method=:auto" tags = [
    :glmm, :level2
] begin
    # Binomial GLMM (multiple-trial) has no analytic df estimator for method=:auto;
    # the user must pass method=:bootstrap explicitly.
    using MixedModels
    using cAIC

    cbpp = MixedModels.dataset(:cbpp)
    m = fit(
        MixedModel,
        @formula(incid / hsz ~ period + (1 | herd)),
        cbpp,
        Binomial();
        weights=float.(cbpp.hsz),
        progress=false,
    )
    @test !issingular(m)
    @test_throws ArgumentError caic(m)               # :auto on Binomial → unsupported
    @test_throws ArgumentError caic(m; method=:auto) # same explicitly
end

# ── type stability ────────────────────────────────────────────────────────────

@testitem "caic on GLMM is type-stable (Poisson, default kwargs)" tags = [
    :glmm, :type_stability
] begin
    # Type instability in the GLMM scoring path is a defect (CLAUDE §4); @inferred asserts
    # the compiler resolves caic to a concrete CAICResult{Float64,…} with no Any/Union.
    using MixedModels
    using cAIC: caic, CAICResult

    y = Float64[1, 1, 2, 1, 8, 9, 8, 9, 3, 4, 3, 4, 5, 6, 5, 6, 1, 2, 1, 1]
    g = repeat(1:5, inner=4)
    m = fit(MixedModel, @formula(y ~ 1 + (1 | g)), (; y, g), Poisson(); progress=false)
    @test !issingular(m)

    scoreit(model) = caic(model)
    r = @inferred scoreit(m)
    @test r isa CAICResult{Float64}
end
