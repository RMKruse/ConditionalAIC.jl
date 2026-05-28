@testitem "_bernoulli_df: hand-computed two-observation arithmetic" tags = [:level1, :glmm] begin
    # Level-1 correctness gate for the pure Efron formula kernel
    # (`docs/math/0006` §4). Synthetic inputs, expected output computed by hand.
    #
    # n=2:  y=[0,1],  μhat=[0.3,0.7],  μhat_flip=[0.4,0.6]
    #
    # i=1:  sign  = -2*0+1 = +1
    #        w    = 0.3*0.7 = 0.21
    #        Δlogit = logit(0.4) - logit(0.3) = log(2/3) - log(3/7) = log(14/9)
    #        contrib = 0.21 * 1 * log(14/9)
    #
    # i=2:  sign  = -2*1+1 = -1
    #        w    = 0.7*0.3 = 0.21
    #        Δlogit = logit(0.6) - logit(0.7) = log(3/2) - log(7/3) = log(9/14) = -log(14/9)
    #        contrib = 0.21 * (-1) * (-log(14/9)) = 0.21*log(14/9)
    #
    # ρ = 2 * 0.21 * log(14/9) = 0.42 * log(14/9)
    using cAIC: DofGLMM

    y = Float64[0.0, 1.0]
    μhat = Float64[0.3, 0.7]
    μhat_flip = Float64[0.4, 0.6]

    ρ = DofGLMM._bernoulli_df(y, μhat, μhat_flip)
    ρ_expected = 0.42 * log(14.0 / 9.0)
    @test ρ ≈ ρ_expected atol = 1e-14
end

@testitem "_bernoulli_df: type stability over Float64 and Float32" tags = [:level1, :glmm] begin
    using cAIC: DofGLMM

    y64 = Float64[0.0, 1.0]
    μ64 = Float64[0.3, 0.7]
    μf64 = Float64[0.4, 0.6]

    @test (@inferred DofGLMM._bernoulli_df(y64, μ64, μf64)) isa Float64

    y32 = Float32[0.0f0, 1.0f0]
    μ32 = Float32[0.3f0, 0.7f0]
    μf32 = Float32[0.4f0, 0.6f0]

    ρ32 = @inferred DofGLMM._bernoulli_df(y32, μ32, μf32)
    @test ρ32 isa Float32
    @test ρ32 ≈ DofGLMM._bernoulli_df(y64, μ64, μf64) rtol = 1e-4
end

@testitem "_bernoulli_df: sign correction is +1 for y=0 and -1 for y=1" tags = [
    :level1, :glmm
] begin
    # A single-observation sanity: the formula's sign term (-2y+1) ensures that
    # when the flip-refit raises μ̂ (positive logit diff), a y=0 observation contributes
    # positively to ρ, and a y=1 observation also contributes positively. Both cases
    # reflect sensitivity: the model's prediction at i changes when y_i is flipped.
    using cAIC: DofGLMM

    # y=0 case: sign=+1, flip pushes μ̂_flip > μ̂  → logit_diff > 0 → contrib > 0
    μ = 0.4
    μ_flip = 0.5   # after flipping y=0→1, model raises probability
    ρ_y0 = DofGLMM._bernoulli_df([0.0], [μ], [μ_flip])
    @test ρ_y0 > 0

    # y=1 case: sign=-1, flip pushes μ̂_flip < μ̂  → logit_diff < 0 → contrib > 0
    μ2 = 0.6
    μ_flip2 = 0.5   # after flipping y=1→0, model lowers probability
    ρ_y1 = DofGLMM._bernoulli_df([1.0], [μ2], [μ_flip2])
    @test ρ_y1 > 0

    # Reference values:
    # y=0: ρ = 0.4*0.6*1*(logit(0.5)-logit(0.4)) = 0.24*(0 - log(2/3)) = 0.24*log(3/2)
    @test ρ_y0 ≈ 0.24 * log(3.0 / 2.0) atol = 1e-14
    # y=1: ρ = 0.6*0.4*(-1)*(logit(0.5)-logit(0.6)) = -0.24*(0 - log(3/2)) = 0.24*log(3/2)
    @test ρ_y1 ≈ 0.24 * log(3.0 / 2.0) atol = 1e-14
end

@testitem "_bernoulli_df: boundary-logit inputs propagate without error" tags = [
    :level1, :glmm
] begin
    # μ̂ = 0.5 → logit = 0; μ̂_flip = 0.5 → zero contribution (model insensitive to flip).
    # Degenerate case, but not an error.
    using cAIC: DofGLMM

    y = Float64[0.0, 1.0]
    μhat = Float64[0.5, 0.5]
    μhat_flip = Float64[0.5, 0.5]

    ρ = DofGLMM._bernoulli_df(y, μhat, μhat_flip)
    @test ρ == 0.0
end

@testitem "dof_glmm_bernoulli reproduces cAIC4 biasCorrectionBernoulli (Level-2 fixture)" tags = [
    :level2, :glmm
] begin
    # Level-2 correctness gate: `dof_glmm_bernoulli` must reproduce the effective df
    # returned by `cAIC4`'s `biasCorrectionBernoulli` on the *same* data, within a
    # tolerance that accounts for fit discrepancies between lme4 and MixedModels.jl.
    # The tolerance atol=1e-3 matches the LMM Level-2 band (DECISIONS.md).
    using HDF5
    using MixedModels
    using cAIC: DofGLMM
    using CategoricalArrays

    fixture = joinpath(@__DIR__, "fixtures", "dof_glmm_bernoulli_level2.h5")
    @test isfile(fixture)

    y, x, group_ids, rho_ref = h5open(fixture, "r") do f
        asscalar(x) = x isa AbstractArray ? only(x) : x
        (
            Float64.(read(f["y"])),
            Float64.(read(f["x"])),
            Int.(read(f["group"])),
            asscalar(Float64.(read(f["rho_ref"]))),
        )
    end

    dat = (y=y, x=x, group=CategoricalArray(string.(group_ids)))
    m = fit(MixedModel, @formula(y ~ x + (1 | group)), dat, Bernoulli(); progress=false)

    ρ = DofGLMM.dof_glmm_bernoulli(m)

    @test isfinite(ρ)
    @test ρ > 0
    @test ρ ≈ rho_ref atol = 1e-3
end

@testitem "dof_glmm_bernoulli: type stability on a fitted Bernoulli GLMM" tags = [
    :level2, :glmm
] begin
    using MixedModels
    using cAIC: DofGLMM
    using CategoricalArrays

    # Tiny Bernoulli GLMM with 3 groups of 4 obs — just enough for a stable GLMM fit
    # and a fast (12-refit) @inferred check without a fixture.
    y = Float64[0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    x = Float64[-1, 1, -0.5, 0.5, -1, 1, -0.5, 0.5, -1, 1, -0.5, 0.5]
    group = CategoricalArray(repeat(["a", "b", "c"], inner=4))

    dat = (y=y, x=x, group=group)
    m = fit(MixedModel, @formula(y ~ x + (1 | group)), dat, Bernoulli(); progress=false)

    @test (@inferred DofGLMM.dof_glmm_bernoulli(m)) isa Float64
end

@testitem "dof_glmm_bernoulli: leaves the original model untouched" tags = [:level2, :glmm] begin
    # The refit loop must not mutate `m`: it deepcopies once as a working buffer
    # (docs/math/0006 §4 — the algorithm description in issue #29).
    using MixedModels
    using cAIC: DofGLMM
    using CategoricalArrays

    y = Float64[0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    x = Float64[-1, 1, -0.5, 0.5, -1, 1, -0.5, 0.5, -1, 1, -0.5, 0.5]
    group = CategoricalArray(repeat(["a", "b", "c"], inner=4))

    dat = (y=y, x=x, group=group)
    m = fit(MixedModel, @formula(y ~ x + (1 | group)), dat, Bernoulli(); progress=false)

    η_before = copy(m.η)
    μ_before = copy(cAIC.MMInternals.glmmfittedmu(m))

    DofGLMM.dof_glmm_bernoulli(m)

    @test m.η == η_before
    @test cAIC.MMInternals.glmmfittedmu(m) == μ_before
end
