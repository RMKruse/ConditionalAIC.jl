# Singular-fit handling (#10): a variance component estimated on the boundary is dropped
# and the cAIC is computed on the **reduced** model, mirroring `cAIC4`'s drop-and-refit
# (`biasCorrectionGaussian` → `deleteZeroComponents`). The reduction itself is the
# `caic`-assembly / singular-fit path that `docs/math/0002` §1 designates out of the bias-
# correction note and into here.

# Reproducible singular-fit data builders. A fitted `LinearMixedModel` retains no source
# table, so each case is a self-contained NamedTuple column-table, fitted fresh per testitem.
@testsnippet SingularData begin
    using Random

    # (1 + x | g): the data carry intercept variation but NO slope variation, so the slope
    # variance is estimated on the boundary (λ[2,2] = 0) → a *partial* drop to (1 | g).
    function partialcorr_data()
        Random.seed!(20260528)
        ng, npg = 12, 8
        g = repeat(1:ng; inner=npg)
        x = randn(ng * npg)
        b0 = randn(ng)[g] .* 2.0
        y = 3.0 .+ 1.5 .* x .+ b0 .+ randn(ng * npg) .* 0.5
        return (; y, x, g=string.(g))
    end

    # (1 | g1) + (1 | g2): only g1 carries variation, so g2's variance is on the boundary
    # → the whole g2 term is dropped, reducing to (1 | g1).
    function twoterm_data()
        Random.seed!(11)
        g1 = repeat(1:10; inner=10)
        g2 = repeat(1:5; outer=20)
        a1 = randn(10)[g1] .* 2.0
        y = 2.0 .+ a1 .+ randn(100) .* 0.7
        return (; y, g1=string.(g1), g2=string.(g2))
    end

    # (1 + x + z | g): only the intercept varies, but the 3×3 correlated factor collapses one
    # direction at a time — the first reduction (to 2×2) is *itself* still on the boundary, so
    # the drop must cascade (3 → 2 → 1) before a non-singular fit is reached.
    function cascade_data()
        Random.seed!(16)
        ng, npg = 15, 6
        g = repeat(1:ng; inner=npg)
        x = randn(ng * npg)
        z = randn(ng * npg)
        b0 = randn(ng)[g] .* 2.0
        y = 1.0 .+ 0.5 .* x .+ 0.3 .* z .+ b0 .+ randn(ng * npg) .* 0.5
        return (; y, x, z, g=string.(g))
    end
end

@testitem "caic scores a singular LMM by reducing and refitting" setup = [SingularData] tags = [
    :level2
] begin
    # Tracer: the whole singular path end-to-end. A correlated (1 + x | g) fit whose slope
    # variance is on the boundary must not be scored as-is (the bias-correction spine on a
    # singular model yields a nonsensical, even negative, ρ); instead the boundary component
    # is dropped, the reduced model is refitted, and the cAIC is computed on *that*. The
    # result records the reduction (`refit`) and carries the reduced model.
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel, @formula(y ~ 1 + x + (1 + x | g)), partialcorr_data(); progress=false
    )
    @test issingular(m)                          # precondition: the fit is on the boundary

    r = caic(m)
    @test r.refit                                # a reduction occurred
    @test r.reducedmodel isa LinearMixedModel    # the cAIC was computed on the reduced model
    @test !issingular(r.reducedmodel)            # and that reduced model is no longer singular
    @test isfinite(r.caic)                       # a finite, sane score …
    @test r.dof > 0                              # … with ρ > 0 (the singular spine gave ρ < 0)
end

@testitem "the reduced model matches a native fit of the reduced formula" setup = [
    SingularData
] tags = [:level2] begin
    # Reconstruction fidelity: dropping the boundary slope of (1 + x | g) and refitting must
    # reproduce — bit-for-bit — what fitting (1 + x + (1 | g)) from scratch on the same data
    # gives. The reduced model is rebuilt from the stored design objects (no source table is
    # retained), so this guards that reconstruction against any drift: the objective and the
    # covariance parameters θ must agree to optimiser tolerance.
    using MixedModels
    using cAIC: caic

    data = partialcorr_data()
    m = fit(MixedModel, @formula(y ~ 1 + x + (1 + x | g)), data; progress=false)
    reduced = caic(m).reducedmodel

    native = fit(MixedModel, @formula(y ~ 1 + x + (1 | g)), data; progress=false)

    @test reduced.θ ≈ native.θ atol = 1e-7
    @test objective(reduced) ≈ objective(native) atol = 1e-7
    @test reduced.feterm.cnames == native.feterm.cnames   # fixed effects intact
    @test only(reduced.reterms).cnames == ["(Intercept)"]  # slope direction dropped
end

@testitem "a whole boundary term is dropped from a multi-term model" setup = [SingularData] tags = [
    :level2
] begin
    # A grouping factor carrying no variation sits entirely on the boundary, so the *whole*
    # term is removed rather than column-subset: (1 | g₁) + (1 | g₂) with g₂ on the boundary
    # reduces to (1 | g₁). The surviving term must be g₁ — its grouping, levels, and the
    # fixed effects untouched.
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel, @formula(y ~ 1 + (1 | g1) + (1 | g2)), twoterm_data(); progress=false
    )
    @test issingular(m)
    @test length(m.reterms) == 2                          # both terms present in the fit

    reduced = caic(m).reducedmodel
    @test length(reduced.reterms) == 1                    # g₂ dropped whole
    surviving = only(reduced.reterms)
    @test MixedModels.fname(surviving) == :g1             # …and it is g₁ that survives
    @test length(surviving.levels) == 10                  # g₁'s 10 groups, not g₂'s 5
    @test !issingular(reduced)
end

@testitem "a non-singular fit is scored as given, with no reduction" tags = [:level2] begin
    # Regression guard: the singular drop-and-refit path must not perturb an ordinary,
    # non-singular fit. The well-conditioned sleepstudy fit is scored as given — no reduced
    # model, no refit flag — and the result still satisfies the cAIC identity.
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    @test !issingular(m)                          # precondition: the fit is interior

    r = caic(m)
    @test !r.refit                                # no reduction occurred
    @test r.reducedmodel === nothing              # …and no reduced model is carried
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof  # the cAIC identity still holds
    @test r.dof > 0
end

@testitem "the drop cascades until the reduced fit is non-singular" setup = [SingularData] tags = [
    :level2
] begin
    # A reduced refit may itself land on the boundary; the drop must then recurse. Here a 3×3
    # correlated factor collapses one direction at a time: a *single* reduction still leaves a
    # singular 2×2 fit, so only the cascade reaches a non-singular model. This drives the
    # while-loop in `caic`, mirroring `cAIC4`'s recursive `deleteZeroComponents`.
    using MixedModels
    using cAIC: caic
    using cAIC: cAIC

    m = fit(
        MixedModel,
        @formula(y ~ 1 + x + z + (1 + x + z | g)),
        cascade_data();
        progress=false,
    )
    @test issingular(m)

    # one reduction is *not* enough — the intermediate model is still on the boundary
    once = cAIC.MMInternals.reduceboundary(m)
    @test issingular(once)

    reduced = caic(m).reducedmodel
    @test !issingular(reduced)                    # the cascade reached a non-singular fit
    @test size(only(reduced.reterms).λ, 1) == 1   # collapsed all the way to the intercept
end

@testitem "an all-boundary fit falls back to the fixed-effects-only score" tags = [:level2] begin
    # When *every* random-effect direction is on the boundary no random-effects model
    # remains, so there is nothing to reduce to. Mirroring `cAIC4`'s `lm` branch
    # (`biasCorrectionGaussian` returning `df = rank + sigma.penalty`, `new = FALSE`,
    # `reducedModel = NULL`, `cll = getcondLL(original)`), the score is the fixed-effects-only
    # one: ρ = p + sigmapenalty, the conditional log-likelihood is that of the original fit at
    # b̂ = 0 (μ = Xβ̂), and no reduced model is carried.
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(yield ~ 1 + (1 | batch)),
        MixedModels.dataset(:dyestuff2);
        progress=false,
    )
    @test issingular(m)                           # every component is on the boundary

    r = caic(m)
    @test !r.refit                                # cAIC4 new = FALSE
    @test r.reducedmodel === nothing              # cAIC4 reducedModel = NULL
    @test r.dof == size(m.X, 2) + 1               # ρ = rank(FE) + sigmapenalty
    @test caic(m; sigmapenalty=0).dof == size(m.X, 2)  # the penalty count is honoured
    @test isfinite(r.caic)

    # the conditional log-likelihood is the original fit's Gaussian density at b̂ = 0
    y = response(m)
    n = length(y)
    σ = m.sigma
    expected = -0.5 * (n * log(2π) + 2n * log(σ) + sum(abs2, y .- fitted(m)) / σ^2)
    @test r.condloglik ≈ expected
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof
end

@testitem "the singular path is type-stable" setup = [SingularData] tags = [:level2] begin
    # Type-stability gate (CLAUDE §6): the drop-and-refit branches must infer to the *same*
    # concrete result type as the ordinary path. A reduced refit is a `LinearMixedModel{T}`
    # (same concrete type as the original — `LinearMixedModel` carries only the float param),
    # and the all-boundary fallback returns `reducedmodel = nothing`; both unify to
    # `CAICResult{Float64,LinearMixedModel{Float64}}`, so `caic` stays type-stable on a
    # singular fit just as on an interior one.
    using MixedModels
    using cAIC: caic, CAICResult

    scoreit(model) = caic(model)

    # the reduce-and-refit path
    mr = fit(
        MixedModel, @formula(y ~ 1 + x + (1 + x | g)), partialcorr_data(); progress=false
    )
    @test issingular(mr)
    @test (@inferred scoreit(mr)) isa CAICResult{Float64,LinearMixedModel{Float64}}

    # the all-boundary lm-fallback path
    md = fit(
        MixedModel,
        @formula(yield ~ 1 + (1 | batch)),
        MixedModels.dataset(:dyestuff2);
        progress=false,
    )
    @test issingular(md)
    @test (@inferred scoreit(md)) isa CAICResult{Float64,LinearMixedModel{Float64}}
end

@testitem "caic matches cAIC4 end-to-end on singular fits (Level-2)" tags = [:level2] begin
    # The correctness gate (CLAUDE §6 Level-2) for the singular path: reproduce the conditional
    # AIC that `cAIC4`'s public `cAIC()` returns on boundary fits, across both code paths of
    # `biasCorrectionGaussian`. References are read from the committed fixture (written by
    # `generate_fixtures_singular.R`); no R runs here.
    #
    #   reduce_ml      a (1 + x | g) fit with x constant within group — the slope is
    #                  unidentifiable and collapses to the boundary in lme4 *and* MixedModels,
    #                  so both drop to (1 | g) and refit (`new = TRUE`). The synthetic sample is
    #                  embedded in the fixture so both ecosystems score the *identical* data.
    #   dyestuff2_*    the canonical `Dyestuff2` fit whose batch variance is zero: every
    #                  component is on the boundary, so cAIC4 takes the `lm` fallback
    #                  (`new = FALSE`, df = rank + sigma.penalty, cll = the original fit at b̂=0).
    #
    # Same fit-discrepancy-derived tolerance as the non-singular Level-2 gate (atol = 1e-3; see
    # DECISIONS.md). The worst observed |Δcaic| here is far inside it: reduce_ml ≈ 3.2e-8
    # (a (1|g) refit, near-identical θ̂ across ecosystems), dyestuff2 ≈ 3e-11 (the fixed-
    # effects-only score depends on no boundary refit at all). A machinery error shifts caic by
    # ≥ O(0.1) — well outside the band.
    using HDF5
    using MixedModels
    using cAIC: caic

    # rhdf5 stores an R length-1 numeric as a 1-element array; coerce before comparing.
    asscalar(x) = x isa AbstractArray ? only(x) : x
    L2_ATOL = 1e-3   # derived Level-2 tolerance; see DECISIONS.md

    fixture = joinpath(@__DIR__, "fixtures", "caic_singular_level2.h5")
    @test isfile(fixture)

    h5open(fixture, "r") do f
        # ── reduce_ml — partial drop, refit (cAIC4 new = TRUE) ────────────────────────────
        g = f["reduce_ml"]
        @test !Bool(asscalar(read(g["reml"])))         # the fixture case is ML
        @test Bool(asscalar(read(g["new"])))           # cAIC4 took the boundary-refit path
        data = (;
            y=read(g["y"]),
            x=read(g["x"]),
            g=string.(Int.(read(g["g"]))),  # embedded shared sample; integer codes → factor
        )
        m = fit(
            MixedModel, @formula(y ~ 1 + x + (1 + x | g)), data; REML=false, progress=false
        )
        @test issingular(m)
        r = caic(m)
        @test r.refit                                  # reduction occurred, matching new = TRUE
        @test !issingular(r.reducedmodel)
        @test r.caic ≈ asscalar(read(g["caic"])) atol = L2_ATOL
        @test r.dof ≈ asscalar(read(g["df"])) atol = L2_ATOL
        @test r.condloglik ≈ asscalar(read(g["cll"])) atol = L2_ATOL

        # ── dyestuff2_{ml,reml} — all-boundary lm fallback (cAIC4 new = FALSE) ─────────────
        for (name, reml) in (("dyestuff2_ml", false), ("dyestuff2_reml", true))
            gc = f[name]
            @test Bool(asscalar(read(gc["reml"]))) == reml
            @test !Bool(asscalar(read(gc["new"])))     # cAIC4 took the lm fallback
            md = fit(
                MixedModel,
                @formula(yield ~ 1 + (1 | batch)),
                MixedModels.dataset(:dyestuff2);
                REML=reml,
                progress=false,
            )
            @test issingular(md)
            rd = caic(md)
            @test !rd.refit                            # no reduced model, matching new = FALSE
            @test rd.reducedmodel === nothing
            @test rd.caic ≈ asscalar(read(gc["caic"])) atol = L2_ATOL
            @test rd.dof ≈ asscalar(read(gc["df"])) atol = L2_ATOL
            @test rd.condloglik ≈ asscalar(read(gc["cll"])) atol = L2_ATOL
        end
    end
end
