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
