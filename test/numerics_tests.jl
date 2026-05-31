@testitem "traceprod computes tr(AB) without forming the product" tags = [:numerics] begin
    using LinearAlgebra: tr

    A = [1.0 2.0 3.0; 4.0 5.0 6.0]        # 2×3
    B = [7.0 8.0; 9.0 10.0; 11.0 12.0]    # 3×2
    @test ConditionalAIC.Numerics.traceprod(A, B) ≈ tr(A * B) rtol = 1e-6 atol = 1e-10
end

@testitem "traceprod is type-stable and generic over T" tags = [:numerics] begin
    using LinearAlgebra: tr

    A = [1.0 2.0 3.0; 4.0 5.0 6.0]
    B = [7.0 8.0; 9.0 10.0; 11.0 12.0]
    @test (@inferred ConditionalAIC.Numerics.traceprod(A, B)) ≈ tr(A * B)

    A32, B32 = Float32.(A), Float32.(B)
    @test ConditionalAIC.Numerics.traceprod(A32, B32) isa Float32
    @test (@inferred Float32 ConditionalAIC.Numerics.traceprod(A32, B32)) ≈ tr(A32 * B32) rtol =
        1e-5
end

@testitem "traceprod guards dimensions, empty inputs, and propagates NaN/Inf" tags = [
    :numerics
] begin
    # size(A) must equal reverse(size(B)) for tr(AB) to be defined.
    @test_throws DimensionMismatch ConditionalAIC.Numerics.traceprod(
        [1.0 2.0; 3.0 4.0], [1.0 2.0 3.0]
    )

    # The empty contraction is zero, not an error.
    @test ConditionalAIC.Numerics.traceprod(zeros(2, 0), zeros(0, 2)) == 0.0
    @test ConditionalAIC.Numerics.traceprod(zeros(0, 3), zeros(3, 0)) == 0.0

    # Non-finite entries propagate rather than being silently dropped.
    @test isnan(ConditionalAIC.Numerics.traceprod([NaN 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]))
    @test isinf(ConditionalAIC.Numerics.traceprod([Inf 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0]))
end

@testitem "logdetpd matches log(det(A)) for an SPD matrix" tags = [:numerics] begin
    using LinearAlgebra: det

    A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]   # SPD
    @test ConditionalAIC.Numerics.logdetpd(A) ≈ log(det(A)) rtol = 1e-6 atol = 1e-10
end

@testitem "logdetpd accepts a precomputed Cholesky factor without refactorising" tags = [
    :numerics
] begin
    using LinearAlgebra: Symmetric, cholesky, det

    A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
    C = cholesky(Symmetric(A))
    @test ConditionalAIC.Numerics.logdetpd(C) ≈ log(det(A)) rtol = 1e-6 atol = 1e-10
    @test ConditionalAIC.Numerics.logdetpd(C) == ConditionalAIC.Numerics.logdetpd(A)
end

@testitem "logdetpd is type-stable and generic over T" tags = [:numerics] begin
    using LinearAlgebra: det

    A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
    @test (@inferred ConditionalAIC.Numerics.logdetpd(A)) ≈ log(det(A))

    A32 = Float32.(A)
    @test ConditionalAIC.Numerics.logdetpd(A32) isa Float32
    @test (@inferred Float32 ConditionalAIC.Numerics.logdetpd(A32)) ≈ log(det(A32)) rtol =
        1e-5
end

@testitem "logdetpd guards non-square and non-positive-definite inputs" tags = [:numerics] begin
    @test_throws DimensionMismatch ConditionalAIC.Numerics.logdetpd(zeros(2, 3))
    @test_throws DomainError ConditionalAIC.Numerics.logdetpd([1.0 2.0; 2.0 1.0])  # indefinite
    @test_throws DomainError ConditionalAIC.Numerics.logdetpd([1.0 1.0; 1.0 1.0])  # singular (PSD boundary)
end

@testitem "invquad matches xᵀA⁻¹x via a Cholesky solve" tags = [:numerics] begin
    using LinearAlgebra: dot, inv

    A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]   # SPD
    x = [1.0, -2.0, 0.5]
    @test ConditionalAIC.Numerics.invquad(A, x) ≈ dot(x, inv(A) * x) rtol = 1e-6 atol =
        1e-10
    @test ConditionalAIC.Numerics.invquad(A, x) ≥ 0          # sum of squares ‖L⁻¹x‖²
end

@testitem "invquad accepts a precomputed Cholesky factor without refactorising" tags = [
    :numerics
] begin
    using LinearAlgebra: Symmetric, cholesky, dot, inv

    A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
    x = [1.0, -2.0, 0.5]
    C = cholesky(Symmetric(A))
    @test ConditionalAIC.Numerics.invquad(C, x) ≈ dot(x, inv(A) * x) rtol = 1e-6 atol =
        1e-10
    @test ConditionalAIC.Numerics.invquad(C, x) == ConditionalAIC.Numerics.invquad(A, x)
end

@testitem "invquad is type-stable and generic over T" tags = [:numerics] begin
    using LinearAlgebra: dot, inv

    A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
    x = [1.0, -2.0, 0.5]
    @test (@inferred ConditionalAIC.Numerics.invquad(A, x)) ≈ dot(x, inv(A) * x)

    A32, x32 = Float32.(A), Float32.(x)
    @test ConditionalAIC.Numerics.invquad(A32, x32) isa Float32
    @test (@inferred Float32 ConditionalAIC.Numerics.invquad(A32, x32)) ≈
        dot(x32, inv(A32) * x32) rtol = 1e-5
end

@testitem "invquad guards dimensions and non-positive-definite inputs" tags = [:numerics] begin
    A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
    @test_throws DimensionMismatch ConditionalAIC.Numerics.invquad(A, [1.0, 2.0])          # wrong length
    @test_throws DimensionMismatch ConditionalAIC.Numerics.invquad(
        zeros(2, 3), [1.0, 2.0, 3.0]
    )  # non-square
    @test_throws DomainError ConditionalAIC.Numerics.invquad([1.0 2.0; 2.0 1.0], [1.0, 1.0])       # indefinite
end

@testitem "logsumexp matches log(Σexp) and stays stable where the naive form overflows" tags = [
    :numerics
] begin
    x = [0.5, -1.2, 2.3, 0.0]
    @test ConditionalAIC.Numerics.logsumexp(x) ≈ log(sum(exp, x)) rtol = 1e-6 atol = 1e-10

    # The stable form factors out the max; the naive log(Σexp) overflows to Inf here.
    @test ConditionalAIC.Numerics.logsumexp([1000.0, 1000.0]) ≈ 1000.0 + log(2.0) rtol =
        1e-12
    @test isinf(log(sum(exp, [1000.0, 1000.0])))
end

@testitem "logsumexp is type-stable, generic over T, and guards edge inputs" tags = [
    :numerics
] begin
    x = [0.5, -1.2, 2.3, 0.0]
    @test (@inferred ConditionalAIC.Numerics.logsumexp(x)) ≈ log(sum(exp, x))

    x32 = Float32.(x)
    @test ConditionalAIC.Numerics.logsumexp(x32) isa Float32
    @test (@inferred Float32 ConditionalAIC.Numerics.logsumexp(x32)) ≈ log(sum(exp, x32)) rtol =
        1e-5

    @test ConditionalAIC.Numerics.logsumexp(Float64[]) == -Inf       # log of an empty sum
    @test isnan(ConditionalAIC.Numerics.logsumexp([0.0, NaN, 1.0]))  # NaN propagates
end
