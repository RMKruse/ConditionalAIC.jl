# Tests for the partial-singularity GLMM reduction (issue #32, M3).
#
# Partial singularity: SOME but not all random-effect variance components are on the
# boundary (λ[d,d]=0). `MMInternals.reduceboundary(::GeneralizedLinearMixedModel)` drops
# the boundary directions, rebuilds the reduced GLMM, and refits it (Laplace, fast=false) —
# the GLMM analogue of one level of cAIC4's `deleteZeroComponents.merMod`. The caller
# (`caic`) cascades until non-singular or fully collapsed.
#
# Math spec: docs/math/0007-glmm-partial-singularity-reduction.md.
# Ground truth: cAIC4 v1.1, R/deleteZeroComponents.R (the single merMod method).

# ── Level-1: bit-for-bit reconstruction (tracer) ───────────────────────────────

@testitem "reduceboundary reconstructs a partially-singular Bernoulli reduced model" tags = [
    :glmm, :level1
] begin
    # A `zerocorr(1 + x | g)` Bernoulli fit where the random slope variance collapses to the
    # boundary (λ[2,2]=0) but the intercept survives (λ[1,1]≠0). reduceboundary must drop the
    # slope direction and refit the reduced `(1 | g)` model. The reconstruction is validated
    # bit-for-bit against a native MixedModels fit of the reduced model: the optimized
    # objective agrees to 1e-6 (the minimized quantity — proof the reconstruction defines the
    # identical optimization problem), and the parameters θ/β agree to 1e-5 (GLMM Laplace
    # optimizer resolution on a flat objective; see DECISIONS.md 2026-05-29).
    using ConditionalAIC
    using ConditionalAIC: MMInternals
    using MixedModels
    using Random: Xoshiro

    # seed-35 design (the Level-2 fixture design): 24 groups × 14 obs, random intercept+slope,
    # the slope variance lands exactly on the boundary in both ConditionalAIC.jl and cAIC4.
    rng = Xoshiro(35)
    ng, npg = 24, 14
    g = repeat(1:ng, inner=npg)
    xg = randn(rng, ng)
    x = repeat(xg, inner=npg)
    b0 = repeat(randn(rng, ng), inner=npg) .* 0.7
    y = Float64[rand(rng, Bernoulli(1 / (1 + exp(-e)))) for e in (0.3 .+ 0.4 .* x .+ b0)]
    data = (; y, x, g)

    m = fit(
        MixedModel,
        @formula(y ~ 1 + x + zerocorr(1 + x | g)),
        data,
        Bernoulli();
        progress=false,
        fast=true,
    )
    @test issingular(m)                          # precondition: must reach the boundary
    λ = only(m.reterms).λ
    @test λ[1, 1] != 0 && λ[2, 2] == 0           # partial: intercept survives, slope on boundary

    mr = MMInternals.reduceboundary(m)
    @test mr isa GeneralizedLinearMixedModel     # tracer: reconstruction returns a GLMM

    # structurally reduced to a single scalar random-intercept term
    @test length(mr.reterms) == 1
    @test only(mr.reterms).cnames == ["(Intercept)"]
    @test length(mr.θ) == 1

    # bit-for-bit vs a native fit of the reduced model
    nat = fit(MixedModel, @formula(y ~ 1 + x + (1 | g)), data, Bernoulli(); progress=false)
    @test objective(mr) ≈ objective(nat) atol = 1e-6
    @test mr.θ ≈ nat.θ atol = 1e-5
    @test mr.β ≈ nat.β atol = 1e-5
end

@testitem "reduceboundary returns nothing when every component is on the boundary" tags = [
    :glmm, :level1
] begin
    # Full collapse: a single scalar random-intercept term whose variance lands exactly on the
    # boundary (θ=0). There is no surviving direction to keep, so reduceboundary returns
    # `nothing` — the signal that the caller (`caic`) must route to the rank(X) full-singularity
    # fallback rather than refit a degenerate reduced model. (Mirrors reduceboundary(::LMM)
    # returning nothing, and cAIC4's deleteZeroComponents dropping to a plain glm.)
    using ConditionalAIC
    using ConditionalAIC: MMInternals
    using MixedModels

    # Alternating [2,4] within each group → all group means equal → zero between-group variance
    # → θ=0 → fully singular (same degenerate design as the Poisson full-singularity test).
    y = repeat([2, 4], 10)
    g = repeat(1:10, inner=2)
    data = (; y, g)
    m = fit(MixedModel, @formula(y ~ 1 + (1 | g)), data, Poisson(); progress=false)
    @test issingular(m)                          # precondition: optimizer found θ=0

    @test MMInternals.reduceboundary(m) === nothing
end

# ── Level-1: caic cascades the partial-singularity reduction ────────────────────

@testitem "caic on a partially-singular GLMM scores the reduced refit" tags = [
    :glmm, :level1
] begin
    # The public `caic` path must detect partial singularity, drop the boundary direction,
    # refit the reduced `(1 | g)` model, and score THAT — not the degenerate singular fit
    # (whose Efron df is nonsensical). The result carries the reduced model (`refit == true`,
    # `reducedmodel` a non-singular GLMM with one reterm), and the cAIC identity holds.
    using ConditionalAIC
    using MixedModels
    using Random: Xoshiro

    rng = Xoshiro(35)
    ng, npg = 24, 14
    g = repeat(1:ng, inner=npg)
    xg = randn(rng, ng)
    x = repeat(xg, inner=npg)
    b0 = repeat(randn(rng, ng), inner=npg) .* 0.7
    y = Float64[rand(rng, Bernoulli(1 / (1 + exp(-e)))) for e in (0.3 .+ 0.4 .* x .+ b0)]
    data = (; y, x, g)

    m = fit(
        MixedModel,
        @formula(y ~ 1 + x + zerocorr(1 + x | g)),
        data,
        Bernoulli();
        progress=false,
        fast=true,
    )
    @test issingular(m)   # precondition: partial boundary (intercept survives, slope on boundary)

    r = caic(m)
    @test r isa CAICResult{Float64,<:GeneralizedLinearMixedModel}
    @test r.refit                                    # reduced model was scored
    @test r.reducedmodel isa GeneralizedLinearMixedModel
    @test length(r.reducedmodel.reterms) == 1        # slope direction dropped
    @test !issingular(r.reducedmodel)                # cascade ran until non-singular
    @test isfinite(r.dof) && r.dof > 0
    @test isfinite(r.caic)
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof rtol = 1e-10
end

# ── Level-2: caic matches cAIC4 end-to-end on a partially-singular Bernoulli fit ────────────

@testitem "caic matches cAIC4 on a partially-singular Bernoulli GLMM (Level-2)" tags = [
    :glmm, :level2
] begin
    # The correctness gate (CLAUDE §6 Level-2) for the partial-singularity GLMM path: reproduce
    # the conditional AIC that `cAIC4`'s public `cAIC()` returns on a boundary glmer fit. The
    # seed-35 sample (a zerocorr(1 + x | g) Bernoulli design whose slope variance collapses to
    # the boundary) is embedded in the fixture so both ecosystems score the *identical* data;
    # cAIC4 drops the slope via `deleteZeroComponents`, refits `(1 | g)` and scores it
    # (`new = TRUE`), exactly as `caic` cascades. References are read from the committed fixture
    # (`generate_fixtures_glmm_singular.{jl,R}`); no R runs here.
    #
    # atol = 1e-3 (the Level-2 tolerance; see DECISIONS.md). The observed gap is far inside it:
    # |Δcaic| ≈ 8e-5, |Δdf| ≈ 5e-5, |Δcll| ≈ 1e-5 — a (1|g) Efron-Steinian score with
    # near-identical θ̂ across ecosystems. A machinery error shifts caic by ≥ O(0.1).
    using HDF5
    using MixedModels
    using ConditionalAIC: caic

    asscalar(x) = x isa AbstractArray ? only(x) : x
    L2_ATOL = 1e-3   # derived Level-2 tolerance; see DECISIONS.md

    fixture = joinpath(@__DIR__, "fixtures", "caic_glmm_singular_level2.h5")
    @test isfile(fixture)

    h5open(fixture, "r") do f
        g = f["partial_bernoulli"]
        @test Bool(asscalar(read(g["new"])))           # cAIC4 took the boundary-refit path
        data = (;
            y=read(g["y"]),
            x=read(g["x"]),
            g=string.(Int.(read(g["g"]))),  # embedded shared sample; integer codes → factor
        )
        m = fit(
            MixedModel,
            @formula(y ~ 1 + x + zerocorr(1 + x | g)),
            data,
            Bernoulli();
            progress=false,
            fast=true,
        )
        @test issingular(m)

        r = caic(m)
        @test r.refit                                  # reduction occurred, matching new = TRUE
        @test r.reducedmodel isa GeneralizedLinearMixedModel
        @test !issingular(r.reducedmodel)
        @test r.caic ≈ asscalar(read(g["caic"])) atol = L2_ATOL
        @test r.dof ≈ asscalar(read(g["df"])) atol = L2_ATOL
        @test r.condloglik ≈ asscalar(read(g["cll"])) atol = L2_ATOL
    end
end

# ── Type stability of the partial-singularity GLMM scoring path ─────────────────

@testitem "the partial-singularity GLMM path is type-stable" tags = [:glmm, :level1] begin
    # Type-stability gate (CLAUDE §6): the cascade-and-refit branch must infer to the *same*
    # concrete result type as the ordinary GLMM path. A reduced refit is a
    # `GeneralizedLinearMixedModel{T,D}` (same concrete type — the family `D` is preserved by
    # `reduceboundary`), so `caic` stays type-stable on a partially-singular fit.
    using ConditionalAIC
    using ConditionalAIC: CAICResult
    using MixedModels
    using Random: Xoshiro

    rng = Xoshiro(35)
    ng, npg = 24, 14
    g = repeat(1:ng, inner=npg)
    xg = randn(rng, ng)
    x = repeat(xg, inner=npg)
    b0 = repeat(randn(rng, ng), inner=npg) .* 0.7
    y = Float64[rand(rng, Bernoulli(1 / (1 + exp(-e)))) for e in (0.3 .+ 0.4 .* x .+ b0)]
    data = (; y, x, g)

    m = fit(
        MixedModel,
        @formula(y ~ 1 + x + zerocorr(1 + x | g)),
        data,
        Bernoulli();
        progress=false,
        fast=true,
    )
    @test issingular(m)

    scoreit(model) = caic(model)
    @test (@inferred scoreit(m)) isa
        CAICResult{Float64,GeneralizedLinearMixedModel{Float64,Bernoulli{Float64}}}
end
