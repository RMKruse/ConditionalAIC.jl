@testitem "caic scores a Gaussian lm terminal end-to-end matching cAIC4 (Level-2)" tags = [
    :level2
] begin
    # The `lm`/`glm` terminal of a backward `stepcaic` search (ADR-0006, issue #36): when the
    # last random-effects term is dropped there is no mixed model left, so the candidate is a
    # plain `GLM.jl` fit scored directly. Here the Gaussian `lm` terminal — the endpoint an LMM
    # backward search reaches (e.g. `cAIC4`'s `Pastes` example bottoms out at an `lm`).
    #
    # `cAIC4`'s `(g)lm` branch: df = rank + 1, σ̂ the MLE rescaling summary$sigma·√((n−p)/n),
    # cll = Σ dnorm(y, μ̂, σ̂), caic = −2·cll + 2·df. References read from the committed fixture
    # (written by `generate_fixtures_glm_terminal.R`); no R runs here. The terminal is a
    # deterministic OLS solve, so R and Julia land on the same β̂ to ~machine precision and the
    # discrepancy is far inside the Level-2 band (see DECISIONS.md).
    using HDF5
    using GLM: lm, @formula
    using cAIC: caic

    asscalar(x) = x isa AbstractArray ? only(x) : x
    L2_ATOL = 1e-3   # the Level-2 gate (DECISIONS.md); the OLS terminal sits far inside it

    fixture = joinpath(@__DIR__, "fixtures", "caic_glm_terminal_level2.h5")
    @test isfile(fixture)

    h5open(fixture, "r") do f
        g = f["gaussian_lm"]
        data = (; y=read(g["y"]), x=read(g["x"]))   # the embedded shared sample
        m = lm(@formula(y ~ 1 + x), data)
        r = caic(m)

        @test r.reducedmodel === nothing            # the terminal is never singular
        @test !r.refit
        @test r.caic ≈ -2 * r.condloglik + 2 * r.dof
        @test r.dof ≈ asscalar(read(g["df"])) atol = L2_ATOL
        @test r.condloglik ≈ asscalar(read(g["cll"])) atol = L2_ATOL
        @test r.caic ≈ asscalar(read(g["caic"])) atol = L2_ATOL
    end
end

@testitem "caic scores a Poisson glm terminal end-to-end matching cAIC4 (Level-2)" tags = [
    :level2
] begin
    # The Poisson-GLMM backward terminal (ADR-0006): a log-link Poisson `glm`. cAIC4 poisson
    # branch: df = rank + 1, cll = Σ dpois(y, λ = μ̂), reusing `condloglik_poisson`. References
    # from the committed fixture; the counts are embedded so Julia's IRLS and R's land on the
    # same Poisson MLE (discrepancy far inside the Level-2 band).
    using HDF5
    using GLM: glm, @formula, Poisson, LogLink
    using cAIC: caic

    asscalar(x) = x isa AbstractArray ? only(x) : x
    L2_ATOL = 1e-3

    fixture = joinpath(@__DIR__, "fixtures", "caic_glm_terminal_level2.h5")
    h5open(fixture, "r") do f
        g = f["poisson_glm"]
        data = (; y=read(g["y"]), x=read(g["x"]))
        m = glm(@formula(y ~ 1 + x), data, Poisson(), LogLink())
        r = caic(m)

        @test r.reducedmodel === nothing
        @test !r.refit
        @test r.method === :terminal
        @test r.caic ≈ -2 * r.condloglik + 2 * r.dof
        @test r.dof ≈ asscalar(read(g["df"])) atol = L2_ATOL
        @test r.condloglik ≈ asscalar(read(g["cll"])) atol = L2_ATOL
        @test r.caic ≈ asscalar(read(g["caic"])) atol = L2_ATOL
    end
end

@testitem "caic scores a Bernoulli glm terminal end-to-end matching cAIC4 (Level-2)" tags = [
    :level2
] begin
    # The Bernoulli-GLMM backward terminal (ADR-0006): a logit-link binary `glm`. cAIC4's binomial
    # branch with y ∈ {0,1} uses size = |unique(y)| − 1 = 1, reducing to Bernoulli, so
    # `condloglik_bernoulli` matches cAIC4 *exactly* (the size convention only deviates for
    # multi-trial data — that case is scored by the corrected kernel and validated separately).
    using HDF5
    using GLM: glm, @formula, Bernoulli, LogitLink
    using cAIC: caic

    asscalar(x) = x isa AbstractArray ? only(x) : x
    L2_ATOL = 1e-3

    fixture = joinpath(@__DIR__, "fixtures", "caic_glm_terminal_level2.h5")
    h5open(fixture, "r") do f
        g = f["bernoulli_glm"]
        data = (; y=read(g["y"]), x=read(g["x"]))
        m = glm(@formula(y ~ 1 + x), data, Bernoulli(), LogitLink())
        r = caic(m)

        @test r.reducedmodel === nothing
        @test !r.refit
        @test r.method === :terminal
        @test r.caic ≈ -2 * r.condloglik + 2 * r.dof
        @test r.dof ≈ asscalar(read(g["df"])) atol = L2_ATOL
        @test r.condloglik ≈ asscalar(read(g["cll"])) atol = L2_ATOL
        @test r.caic ≈ asscalar(read(g["caic"])) atol = L2_ATOL
    end
end

@testitem "caic scores a multi-trial Binomial glm terminal via the corrected kernel (deviation)" tags = [
    :level2
] begin
    # The multi-trial-Binomial-GLMM backward terminal (ADR-0006) — the documented DEVIATION case
    # (DECISIONS 2026-05-29 / 2026-05-30). `cAIC4`'s binomial branch evaluates the density on the
    # proportion with size = |unique(y)| − 1 and returns cll = −∞, so there is no finite
    # `cAIC4::cAIC` reference. The terminal instead reuses `condloglik_binomial` — the correct
    # binomial density at the true per-observation trial counts nᵢ — exactly as the M3 GLMM
    # binomial path does. Ground truth is base-R `dbinom(kᵢ, nᵢ, μ̂ᵢ)`, embedded in the fixture;
    # the trial counts are recovered from the fit's prior weights.
    using HDF5
    using GLM: glm, @formula, Binomial, LogitLink
    using cAIC: caic

    asscalar(x) = x isa AbstractArray ? only(x) : x
    L2_ATOL = 1e-3

    fixture = joinpath(@__DIR__, "fixtures", "caic_glm_terminal_level2.h5")
    h5open(fixture, "r") do f
        g = f["binomial_glm"]
        y = read(g["y"])          # proportion kᵢ/nᵢ
        x = read(g["x"])
        n = read(g["n"])          # trial counts
        m = glm(@formula(y ~ 1 + x), (; y, x), Binomial(), LogitLink(); wts=n)
        r = caic(m)

        @test r.reducedmodel === nothing
        @test !r.refit
        @test r.method === :terminal
        @test r.caic ≈ -2 * r.condloglik + 2 * r.dof
        @test r.dof ≈ asscalar(read(g["df"])) atol = L2_ATOL
        @test r.condloglik ≈ asscalar(read(g["cll"])) atol = L2_ATOL
        @test r.condloglik > -Inf     # finite, unlike cAIC4's defective binomial getcondLL
        @test r.caic ≈ asscalar(read(g["caic"])) atol = L2_ATOL
    end
end

@testitem "caic on a Gaussian lm terminal is type-stable (@inferred)" begin
    using Test
    using GLM: lm, @formula
    using cAIC: caic, CAICResult

    data = (; x=[-1.0, -0.3, 0.2, 0.8, 1.4], y=[0.1, 0.9, 1.6, 2.1, 3.0])
    m = lm(@formula(y ~ 1 + x), data)
    r = @inferred caic(m)
    @test r isa CAICResult{Float64}
    @test r.method === :terminal
    @test r.bsource === :na
end

@testitem "caic on Poisson / Bernoulli / Binomial glm terminals is type-stable (@inferred)" begin
    # Each supported `glm` terminal family must infer to a concrete `CAICResult{Float64}` — the
    # family dispatch (`_glm_terminal(m, family)`) resolves the response distribution from the fit,
    # so the inner kernel call is statically known once the family method is selected.
    using Test
    using GLM: glm, @formula, Poisson, LogLink, Bernoulli, Binomial, LogitLink
    using cAIC: caic, CAICResult

    x = [-1.0, -0.3, 0.2, 0.8, 1.4]

    mp = glm(@formula(y ~ 1 + x), (; x, y=[0.0, 1.0, 2.0, 3.0, 5.0]), Poisson(), LogLink())
    rp = @inferred caic(mp)
    @test rp isa CAICResult{Float64}
    @test rp.method === :terminal

    # Non-separable labels so μ̂ stays interior (separable data drives μ̂ → 0/1, a log domain error).
    mb = glm(
        @formula(y ~ 1 + x), (; x, y=[1.0, 0.0, 1.0, 0.0, 1.0]), Bernoulli(), LogitLink()
    )
    rb = @inferred caic(mb)
    @test rb isa CAICResult{Float64}

    n = [4.0, 4.0, 4.0, 4.0, 4.0]
    mm = glm(
        @formula(y ~ 1 + x),
        (; x, y=[0.25, 0.5, 0.5, 0.75, 1.0]),
        Binomial(),
        LogitLink();
        wts=n,
    )
    rm = @inferred caic(mm)
    @test rm isa CAICResult{Float64}
end

@testitem "caic on an unsupported glm terminal family raises ArgumentError" begin
    # The supported terminals are the Gaussian `lm` and Poisson / Bernoulli / Binomial `glm`; any
    # other family (here a Gamma `glm`) must fail loudly rather than return a silently-wrong number.
    using Test
    using GLM: glm, @formula, Gamma, InverseLink
    using cAIC: caic

    x = [-1.0, -0.3, 0.2, 0.8, 1.4]
    m = glm(@formula(y ~ 1 + x), (; x, y=[0.5, 1.0, 1.5, 2.0, 3.0]), Gamma(), InverseLink())
    @test_throws ArgumentError caic(m)
end
