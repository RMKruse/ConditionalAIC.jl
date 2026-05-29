# Tests for the full-singularity GLMM fallback (issue #27, M3).
#
# The full-singularity path triggers when every random-effect variance component is on the
# boundary (θ=0). The GLMM collapses to a plain GLM and the df is rank(X) — the
# fixed-effects count — with NO σ-penalty (canonical-link families have fixed dispersion).
# This file tests:
#   (a) The GLMM conditional log-likelihood kernels (Level-1: pure math, no MixedModels).
#   (b) The full-singularity scoring path end-to-end (Level-2: real GLMM fits).
#
# Math spec: docs/math/0006-glmm-bias-correction.md §1 (log-lik) and §5 (full-singularity
# fallback). Ground truth: cAIC4 v1.1, `biasCorrectionPoisson.R:14–16` and
# `biasCorrectionBernoulli.R:11–13`.

# ── Level-1: Poisson conditional log-likelihood ──────────────────────────────

@testitem "condloglik_poisson matches the Poisson log-probability sum" tags = [
    :glmm, :loglik
] begin
    # ℓ_cond^Pois = Σᵢ [y_i log(μ_i) - μ_i - log(y_i!)]
    # Reference: hand-computed. y=[1,2,3], mu=[1,2,3]:
    #   log(1!) = 0, log(2!) = log(2) ≈ 0.6931, log(3!) = log(6) ≈ 1.7918
    #   = [0 - 1 - 0] + [2·log(2) - 2 - log(2)] + [3·log(3) - 3 - log(6)]
    #   ≈ -1 - 1.3069 - 1.4959 = -3.8028
    using cAIC
    _logfact(k) = sum(log(j) for j in 2:k; init=0.0)  # log(k!) via log-sum (no special functions)
    y = [1.0, 2.0, 3.0]
    μ = [1.0, 2.0, 3.0]
    ref = sum(y[i] * log(μ[i]) - μ[i] - _logfact(round(Int, y[i])) for i in eachindex(y))
    @test cAIC.Loglik.condloglik_poisson(y, μ) ≈ ref rtol = 1e-6 atol = 1e-10
end

@testitem "condloglik_poisson: zero count (y_i=0) contributes zero log term" tags = [
    :glmm, :loglik
] begin
    # y_i = 0 → y_i·log(μ_i) = 0 (xlogy convention, not NaN), and log(0!) = 0.
    # So the zero-count term is -μ_i only.
    using cAIC
    _logfact(k) = sum(log(j) for j in 2:k; init=0.0)
    y = [0.0, 1.0]
    μ = [2.0, 2.0]
    ref = sum(y[i] * log(μ[i]) - μ[i] - _logfact(round(Int, y[i])) for i in eachindex(y))
    @test cAIC.Loglik.condloglik_poisson(y, μ) ≈ ref rtol = 1e-6 atol = 1e-10
    @test isfinite(cAIC.Loglik.condloglik_poisson(y, μ))
end

@testitem "condloglik_poisson is type-stable and generic over T" tags = [:glmm, :loglik] begin
    using cAIC
    y = [1.0, 2.0]
    μ = [1.5, 2.5]
    @test (@inferred cAIC.Loglik.condloglik_poisson(y, μ)) isa Float64
    y32, μ32 = Float32[1.0, 2.0], Float32[1.5, 2.5]
    @test (@inferred cAIC.Loglik.condloglik_poisson(y32, μ32)) isa Float32
end

@testitem "condloglik_poisson rejects non-positive μ and mismatched lengths" tags = [
    :glmm, :loglik
] begin
    using cAIC
    @test_throws DomainError cAIC.Loglik.condloglik_poisson([1.0], [0.0])
    @test_throws DomainError cAIC.Loglik.condloglik_poisson([1.0], [-1.0])
    @test_throws DimensionMismatch cAIC.Loglik.condloglik_poisson([1.0, 2.0], [1.0])
end

# ── Level-1: Bernoulli conditional log-likelihood ────────────────────────────

@testitem "condloglik_bernoulli matches the binary cross-entropy sum" tags = [
    :glmm, :loglik
] begin
    # ℓ_cond^Bern = Σᵢ [y_i log(μ_i) + (1-y_i) log(1-μ_i)]
    using cAIC
    y = [1.0, 0.0, 1.0]
    μ = [0.8, 0.3, 0.6]
    ref = sum(y[i] * log(μ[i]) + (1 - y[i]) * log(1 - μ[i]) for i in eachindex(y))
    @test cAIC.Loglik.condloglik_bernoulli(y, μ) ≈ ref rtol = 1e-6 atol = 1e-10
end

@testitem "condloglik_bernoulli handles boundary labels (y=0 and y=1)" tags = [
    :glmm, :loglik
] begin
    # y=0 → only the (1-y)·log(1-μ) term; y=1 → only y·log(μ). Both handled without NaN.
    using cAIC
    y = [0.0, 1.0]
    μ = [0.3, 0.7]
    ref = log(1 - μ[1]) + log(μ[2])
    @test cAIC.Loglik.condloglik_bernoulli(y, μ) ≈ ref rtol = 1e-6 atol = 1e-10
    @test isfinite(cAIC.Loglik.condloglik_bernoulli(y, μ))
end

@testitem "condloglik_bernoulli is type-stable and generic over T" tags = [:glmm, :loglik] begin
    using cAIC
    y = [1.0, 0.0]
    μ = [0.7, 0.4]
    @test (@inferred cAIC.Loglik.condloglik_bernoulli(y, μ)) isa Float64
    y32, μ32 = Float32[1.0, 0.0], Float32[0.7, 0.4]
    @test (@inferred cAIC.Loglik.condloglik_bernoulli(y32, μ32)) isa Float32
end

@testitem "condloglik_bernoulli rejects μ outside (0,1) and mismatched lengths" tags = [
    :glmm, :loglik
] begin
    using cAIC
    @test_throws DomainError cAIC.Loglik.condloglik_bernoulli([1.0], [0.0])
    @test_throws DomainError cAIC.Loglik.condloglik_bernoulli([0.0], [1.0])
    @test_throws DomainError cAIC.Loglik.condloglik_bernoulli([1.0], [-0.1])
    @test_throws DomainError cAIC.Loglik.condloglik_bernoulli([1.0], [1.1])
    @test_throws DimensionMismatch cAIC.Loglik.condloglik_bernoulli([1.0, 0.0], [0.5])
end

# ── Level-2: full-singularity GLMM scoring ───────────────────────────────────

@testitem "caic on fully-singular Poisson GLMM: ρ = rank(X), no σ-penalty, cAIC identity" tags = [
    :glmm, :level2
] begin
    # Alternating [2,4] within each group → all group means = 3.0 → no between-group variation
    # → θ=0 → fully singular. cAIC4's `biasCorrectionPoisson.R:14–16` fallback: return
    # rank(X), no +1 for σ (canonical-link Poisson has fixed dispersion).
    using cAIC
    using MixedModels
    y = repeat([2, 4], 10)    # 20 obs, all group means = 3 → zero between-group variance
    g = repeat(1:10, inner=2) # 10 groups of 2, each group has y=[2,4]
    data = (; y, g)
    m = fit(MixedModel, @formula(y ~ 1 + (1 | g)), data, Poisson(); progress=false)

    @test issingular(m)   # precondition: optimizer must have found θ=0

    r = caic(m)
    @test r.dof == 1.0              # rank(X) = 1 for intercept-only; no +1 σ-penalty
    @test !r.refit                  # full-singularity path does not refit
    @test r.reducedmodel === nothing
    @test r.method == :auto
    @test r.bsource == :na
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof rtol = 1e-10  # cAIC identity
end

@testitem "caic on fully-singular GLMM: type is CAICResult{Float64, GeneralizedLinearMixedModel}" tags = [
    :glmm, :level2
] begin
    using cAIC
    using MixedModels
    y = repeat([2, 4], 10)
    g = repeat(1:10, inner=2)
    data = (; y, g)
    m = fit(MixedModel, @formula(y ~ 1 + (1 | g)), data, Poisson(); progress=false)
    @test issingular(m)

    r = caic(m)
    @test r isa CAICResult{Float64,<:GeneralizedLinearMixedModel}
    @test r.caic isa Float64
    @test r.dof isa Float64
    @test r.condloglik isa Float64
end

@testitem "caic on Binomial GLMM with method=:auto throws ArgumentError (no analytic df)" tags = [
    :glmm, :level2
] begin
    # A Binomial GLMM with multiple-trial data has no analytic df estimator for method=:auto
    # (neither Poisson Chen-Stein nor Bernoulli Efron applies). caic must throw ArgumentError
    # directing the user to method=:bootstrap. This is consistent with cAIC4's own scope:
    # biasCorrectionPoisson and biasCorrectionBernoulli do not handle multi-trial binomial.
    using cAIC
    using MixedModels
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
    @test_throws ArgumentError caic(m)          # :auto + Binomial → unsupported family
    @test_throws ArgumentError caic(m; method=:auto)
end

@testitem "caic on fully-singular Binomial GLMM (method=:bootstrap): ρ = rank(X), finite cAIC" tags = [
    :glmm, :level2, :bootstrap
] begin
    # Every group carries identical (incid, hsz) data → no between-group variation → θ=0
    # → fully singular. The full-singularity shortcut returns ρ = rank(X) directly (no
    # refit, no bootstrap), but it must still evaluate the binomial conditional
    # log-likelihood (condloglik_binomial) — the path that used to throw for Binomial.
    # `fast=true` (PIRLS-only) reaches the θ=0 boundary here; the default nAGQ optimiser is
    # roundoff-limited on this degenerate design and stalls at the θ=1 init.
    using cAIC
    using MixedModels
    using Random: Xoshiro
    incid = repeat([1, 3], 10)     # each group: proportions 1/4 and 3/4
    hsz = fill(4, 20)
    g = repeat(1:10, inner=2)      # 10 identical groups of 2
    data = (; incid, hsz, g)
    m = fit(
        MixedModel,
        @formula(incid / hsz ~ 1 + (1 | g)),
        data,
        Binomial();
        weights=float.(hsz),
        progress=false,
        fast=true,
    )

    @test issingular(m)   # precondition: optimizer must have found θ=0

    r = caic(m; method=:bootstrap, nboot=20, rng=Xoshiro(42))
    @test r.dof == 1.0              # rank(X) = 1 for intercept-only; no +1 σ-penalty
    @test !r.refit                  # full-singularity path does not refit/bootstrap
    @test r.reducedmodel === nothing
    @test isfinite(r.condloglik)    # the previously-dead binomial loglik path
    @test isfinite(r.caic)
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof rtol = 1e-10
end
