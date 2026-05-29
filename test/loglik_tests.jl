@testitem "condloglik matches the hand-computed Gaussian conditional log-likelihood" tags = [
    :loglik
] begin
    # ℓ = Σᵢ log φ(yᵢ; ŷᵢ, σ̂²). A perfect fit (e = 0) with n = 1, σ̂ = 1 is the
    # hand-checkable anchor: ℓ = −½ log(2π) ≈ −0.9189385332046727.
    @test cAIC.Loglik.condloglik([3.0], [3.0], 1.0) ≈ -0.5 * log(2π) rtol = 1e-6 atol =
        1e-10

    # A general synthetic case, cross-checked against the per-observation form of the
    # estimand (a different arrangement from the shipped aggregated closed form).
    y = [1.0, -2.0, 0.5, 3.0]
    ŷ = [1.2, -1.5, 0.0, 2.0]
    σ = 0.8
    ref = sum(-0.5 * log(2π) - log(σ) - (y[i] - ŷ[i])^2 / (2σ^2) for i in eachindex(y))
    @test cAIC.Loglik.condloglik(y, ŷ, σ) ≈ ref rtol = 1e-6 atol = 1e-10
end

@testitem "condloglik is type-stable and generic over T" tags = [:loglik] begin
    y = [1.0, -2.0, 0.5, 3.0]
    ŷ = [1.2, -1.5, 0.0, 2.0]
    σ = 0.8
    ref = sum(-0.5 * log(2π) - log(σ) - (y[i] - ŷ[i])^2 / (2σ^2) for i in eachindex(y))

    @test (@inferred cAIC.Loglik.condloglik(y, ŷ, σ)) ≈ ref

    # Float32 in ⟹ Float32 out (no silent promotion to Float64).
    y32, ŷ32, σ32 = Float32.(y), Float32.(ŷ), 0.8f0
    @test cAIC.Loglik.condloglik(y32, ŷ32, σ32) isa Float32
    @test (@inferred Float32 cAIC.Loglik.condloglik(y32, ŷ32, σ32)) ≈ ref rtol = 1e-5
end

@testitem "condloglik rejects σ̂ outside its domain and mismatched lengths" tags = [:loglik] begin
    y = [1.0, 2.0, 3.0]
    ŷ = [0.5, 2.5, 2.0]

    # σ̂ is a standard deviation: must be a positive real.
    @test_throws DomainError cAIC.Loglik.condloglik(y, ŷ, 0.0)
    @test_throws DomainError cAIC.Loglik.condloglik(y, ŷ, -1.0)
    @test_throws DomainError cAIC.Loglik.condloglik(y, ŷ, NaN)

    # y and ŷ must index alike.
    @test_throws DimensionMismatch cAIC.Loglik.condloglik(y, [1.0, 2.0], 1.0)
    @test_throws DimensionMismatch cAIC.Loglik.condloglik([1.0, 2.0], ŷ, 1.0)
end

@testitem "condloglik handles empty input, single obs, perfect fit, and non-finite data" tags = [
    :loglik
] begin
    # The conditional log-likelihood of no data is the empty sum, 0.
    @test cAIC.Loglik.condloglik(Float64[], Float64[], 1.0) == 0.0

    # A single observation is the univariate normal log-density.
    @test cAIC.Loglik.condloglik([2.0], [0.5], 1.5) ≈
        -0.5 * log(2π) - log(1.5) - (2.0 - 0.5)^2 / (2 * 1.5^2) rtol = 1e-6 atol = 1e-10

    # A perfect fit (e = 0) is the finite maximum −(n/2)·log(2π) − n·log σ̂.
    n, σ = 5, 2.0
    @test cAIC.Loglik.condloglik(fill(3.0, n), fill(3.0, n), σ) ≈
        -(n / 2) * log(2π) - n * log(σ) rtol = 1e-6 atol = 1e-10

    # Non-finite data propagates rather than being silently dropped.
    @test isnan(cAIC.Loglik.condloglik([NaN, 1.0], [0.0, 1.0], 1.0))
    @test isinf(cAIC.Loglik.condloglik([Inf, 1.0], [0.0, 1.0], 1.0))
end

# ── GLMM Poisson ─────────────────────────────────────────────────────────────

@testitem "condloglik_poisson matches hand-computed Poisson conditional log-likelihood" tags = [
    :loglik
] begin
    # Single-obs anchor: y=1, μ=1 → 1·log(1) − 1 − log(1!) = 0 − 1 − 0 = −1.
    @test cAIC.Loglik.condloglik_poisson([1.0], [1.0]) ≈ -1.0 rtol = 1e-6 atol = 1e-10

    # General case cross-checked against the per-observation estimand. y[2]=0 is included
    # to exercise the xlogy(0, ·) = 0 branch (zero count, nonzero rate). For integer y the
    # log-factorial is computed as log(factorial(·)) to avoid importing SpecialFunctions.
    y = [2.0, 0.0, 3.0]
    μ = [1.5, 2.0, 3.0]
    logfact = [log(factorial(Int(yi))) for yi in y]  # [log(2), 0, log(6)]
    ref = sum(y[i] * log(μ[i]) - μ[i] - logfact[i] for i in eachindex(y))
    @test cAIC.Loglik.condloglik_poisson(y, μ) ≈ ref rtol = 1e-6 atol = 1e-10
end

@testitem "condloglik_poisson is type-stable and generic over T" tags = [:loglik] begin
    y = [2.0, 0.0, 3.0]
    μ = [1.5, 2.0, 3.0]
    logfact = [log(factorial(Int(yi))) for yi in y]
    ref = sum(y[i] * log(μ[i]) - μ[i] - logfact[i] for i in eachindex(y))

    @test (@inferred cAIC.Loglik.condloglik_poisson(y, μ)) ≈ ref

    y32, μ32 = Float32.(y), Float32.(μ)
    @test cAIC.Loglik.condloglik_poisson(y32, μ32) isa Float32
    @test (@inferred Float32 cAIC.Loglik.condloglik_poisson(y32, μ32)) ≈ Float32(ref) rtol =
        1e-4
end

@testitem "condloglik_poisson rejects invalid μ̂ and mismatched lengths" tags = [:loglik] begin
    y = [1.0, 2.0]
    # μ̂ is a Poisson rate and must be strictly positive.
    @test_throws DomainError cAIC.Loglik.condloglik_poisson(y, [0.0, 1.0])
    @test_throws DomainError cAIC.Loglik.condloglik_poisson(y, [-0.5, 1.0])
    @test_throws DimensionMismatch cAIC.Loglik.condloglik_poisson(y, [1.0])
    @test_throws DimensionMismatch cAIC.Loglik.condloglik_poisson([1.0], y)
end

@testitem "condloglik_poisson handles empty input, y=0 entries, and non-finite data" tags = [
    :loglik
] begin
    # Empty input is the empty sum.
    @test cAIC.Loglik.condloglik_poisson(Float64[], Float64[]) == 0.0

    # y_i = 0 with μ̂_i = c: xlogy(0, c) = 0, so ℓ = −c − log(0!) = −c.
    @test cAIC.Loglik.condloglik_poisson([0.0], [2.5]) ≈ -2.5 rtol = 1e-6 atol = 1e-10

    # Non-finite counts propagate.
    @test isnan(cAIC.Loglik.condloglik_poisson([NaN, 1.0], [1.0, 1.0]))
end

# ── GLMM Bernoulli ───────────────────────────────────────────────────────────

@testitem "condloglik_bernoulli matches hand-computed Bernoulli conditional log-likelihood" tags = [
    :loglik
] begin
    # Single-obs anchor: y=1, μ=0.5 → log(0.5) = −log 2 ≈ −0.6931471805599453.
    @test cAIC.Loglik.condloglik_bernoulli([1.0], [0.5]) ≈ -log(2.0) rtol = 1e-6 atol =
        1e-10

    # General case cross-checked per-observation. y ∈ {0,1} so log(1−μ) is base Julia.
    y = [0.0, 1.0, 1.0]
    μ = [0.3, 0.7, 0.9]
    ref = sum(y[i] * log(μ[i]) + (1 - y[i]) * log(1 - μ[i]) for i in eachindex(y))
    @test cAIC.Loglik.condloglik_bernoulli(y, μ) ≈ ref rtol = 1e-6 atol = 1e-10
end

@testitem "condloglik_bernoulli is type-stable and generic over T" tags = [:loglik] begin
    y = [0.0, 1.0, 1.0]
    μ = [0.3, 0.7, 0.9]
    ref = sum(y[i] * log(μ[i]) + (1 - y[i]) * log(1 - μ[i]) for i in eachindex(y))

    @test (@inferred cAIC.Loglik.condloglik_bernoulli(y, μ)) ≈ ref

    y32, μ32 = Float32.(y), Float32.(μ)
    @test cAIC.Loglik.condloglik_bernoulli(y32, μ32) isa Float32
    @test (@inferred Float32 cAIC.Loglik.condloglik_bernoulli(y32, μ32)) ≈ Float32(ref) rtol =
        1e-4
end

@testitem "condloglik_bernoulli rejects invalid μ̂ and mismatched lengths" tags = [:loglik] begin
    y = [0.0, 1.0]
    # μ̂ is a Bernoulli probability and must be strictly in (0, 1).
    @test_throws DomainError cAIC.Loglik.condloglik_bernoulli(y, [0.0, 0.5])
    @test_throws DomainError cAIC.Loglik.condloglik_bernoulli(y, [0.5, 1.0])
    @test_throws DomainError cAIC.Loglik.condloglik_bernoulli(y, [-0.1, 0.5])
    @test_throws DomainError cAIC.Loglik.condloglik_bernoulli(y, [0.5, 1.1])
    @test_throws DimensionMismatch cAIC.Loglik.condloglik_bernoulli(y, [0.5])
    @test_throws DimensionMismatch cAIC.Loglik.condloglik_bernoulli([0.0], [0.5, 0.7])
end

@testitem "condloglik_bernoulli handles empty input and non-finite data" tags = [:loglik] begin
    # Empty input is the empty sum.
    @test cAIC.Loglik.condloglik_bernoulli(Float64[], Float64[]) == 0.0

    # Non-finite data propagates.
    @test isnan(cAIC.Loglik.condloglik_bernoulli([NaN, 1.0], [0.5, 0.5]))
end

# ── GLMM multi-trial binomial ──────────────────────────────────────────────────

@testitem "condloglik_binomial matches base-R dbinom (multi-trial)" tags = [:loglik] begin
    # Single-obs anchor: k=1, n=2, μ̂=0.5 → log C(2,1) + log(0.5) + log(0.5)
    #   = log 2 − 2 log 2 = −log 2.  base R: dbinom(1, 2, 0.5, log=TRUE) = −0.6931472.
    # The interface takes the proportion y = k/n, so y = 1/2.
    @test cAIC.Loglik.condloglik_binomial([0.5], [0.5], [2.0]) ≈ -log(2.0) rtol = 1e-6 atol =
        1e-10

    # base R: dbinom(2, 3, 0.5, log=TRUE) = −0.9808293 (k=2, n=3, μ̂=0.5; y = 2/3).
    @test cAIC.Loglik.condloglik_binomial([2 / 3], [0.5], [3.0]) ≈ -0.9808292530117262 rtol =
        1e-6 atol = 1e-10

    # General case. k=[1,2,5], n=[2,4,10], μ̂=[0.5,0.3,0.6]; y = k/n.
    # base R: sum(dbinom(c(1,2,5), c(2,4,10), c(0.5,0.3,0.6), log=TRUE)) = −3.628836.
    k = [1.0, 2.0, 5.0]
    n = [2.0, 4.0, 10.0]
    μ = [0.5, 0.3, 0.6]
    y = k ./ n
    @test cAIC.Loglik.condloglik_binomial(y, μ, n) ≈ -3.628836 rtol = 1e-6 atol = 1e-5

    # Cross-check against the per-observation estimand (a different arrangement than the
    # kernel's aggregated xlogy/loggamma form, so it checks the aggregation).
    ref = sum(
        log(binomial(Int(n[i]), Int(k[i]))) +
        k[i] * log(μ[i]) +
        (n[i] - k[i]) * log(1 - μ[i]) for i in eachindex(k)
    )
    @test cAIC.Loglik.condloglik_binomial(y, μ, n) ≈ ref rtol = 1e-6 atol = 1e-10
end

@testitem "condloglik_binomial reduces to Bernoulli when all nᵢ = 1" tags = [:loglik] begin
    # nᵢ ≡ 1 ⇒ the binomial coefficient vanishes ⇒ identical to condloglik_bernoulli.
    y = [0.0, 1.0, 1.0]
    μ = [0.3, 0.7, 0.9]
    n = ones(3)
    @test cAIC.Loglik.condloglik_binomial(y, μ, n) ≈ cAIC.Loglik.condloglik_bernoulli(y, μ) rtol =
        1e-12
end

@testitem "condloglik_binomial is type-stable and generic over T" tags = [:loglik] begin
    k = [1.0, 2.0, 5.0]
    n = [2.0, 4.0, 10.0]
    μ = [0.5, 0.3, 0.6]
    y = k ./ n
    ref = -3.628836

    @test (@inferred cAIC.Loglik.condloglik_binomial(y, μ, n)) ≈ ref rtol = 1e-5

    y32, μ32, n32 = Float32.(y), Float32.(μ), Float32.(n)
    @test cAIC.Loglik.condloglik_binomial(y32, μ32, n32) isa Float32
    @test (@inferred Float32 cAIC.Loglik.condloglik_binomial(y32, μ32, n32)) ≈ Float32(ref) rtol =
        1e-3
end

@testitem "condloglik_binomial rejects invalid inputs" tags = [:loglik] begin
    y = [0.5, 0.5]
    n = [2.0, 2.0]
    # μ̂ must be strictly in (0, 1).
    @test_throws DomainError cAIC.Loglik.condloglik_binomial(y, [0.0, 0.5], n)
    @test_throws DomainError cAIC.Loglik.condloglik_binomial(y, [0.5, 1.0], n)
    # Trial counts must be strictly positive.
    @test_throws DomainError cAIC.Loglik.condloglik_binomial(y, [0.5, 0.5], [0.0, 2.0])
    @test_throws DomainError cAIC.Loglik.condloglik_binomial(y, [0.5, 0.5], [-1.0, 2.0])
    # kᵢ = nᵢ yᵢ must be a (near-)integer: y = 0.3 with n = 2 → k = 0.6, not integer.
    @test_throws DomainError cAIC.Loglik.condloglik_binomial([0.3, 0.5], [0.5, 0.5], n)
    # kᵢ must lie in [0, nᵢ]: y = 1.5 with n = 2 → k = 3 > n.
    @test_throws DomainError cAIC.Loglik.condloglik_binomial([1.5, 0.5], [0.5, 0.5], n)
    # Mismatched lengths across any of the three vectors.
    @test_throws DimensionMismatch cAIC.Loglik.condloglik_binomial(y, [0.5], n)
    @test_throws DimensionMismatch cAIC.Loglik.condloglik_binomial(y, [0.5, 0.5], [2.0])
end

@testitem "condloglik_binomial handles empty input and boundary counts" tags = [:loglik] begin
    # Empty input is the empty sum.
    @test cAIC.Loglik.condloglik_binomial(Float64[], Float64[], Float64[]) == 0.0

    # kᵢ = 0 (all failures) and kᵢ = nᵢ (all successes): xlogy/xlog1py keep the 0·log 0
    # boundary finite. k=0,n=4,μ̂=0.25 → 4·log(0.75); k=3,n=3,μ̂=0.5 → log C(3,3)+3log(0.5).
    @test cAIC.Loglik.condloglik_binomial([0.0], [0.25], [4.0]) ≈ 4 * log(0.75) rtol = 1e-10
    @test cAIC.Loglik.condloglik_binomial([1.0], [0.5], [3.0]) ≈ 3 * log(0.5) rtol = 1e-10
end
