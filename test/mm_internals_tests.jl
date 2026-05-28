@testitem "package loads with the stubbed public export surface" begin
    # `using cAIC` having succeeded, the three public verbs exist as zero-method
    # stubs — the M1 surface, before any estimator methods are added.
    for name in (:caic, :anocaic, :stepcaic)
        @test isdefined(cAIC, name)
        @test getproperty(cAIC, name) isa Function
    end
end

@testitem "reml reports which objective the model was fitted under" begin
    using MixedModels
    f = @formula(yield ~ 1 + (1 | batch))
    data = MixedModels.dataset(:dyestuff)
    m_ml = fit(MixedModel, f, data; REML=false, progress=false)
    m_reml = fit(MixedModel, f, data; REML=true, progress=false)

    @test cAIC.MMInternals.reml(m_ml) === false
    @test cAIC.MMInternals.reml(m_reml) === true
    @inferred Bool cAIC.MMInternals.reml(m_ml)
end

@testitem "sigmahat extracts the residual standard deviation σ̂" begin
    using MixedModels
    m = fit(
        MixedModel,
        @formula(yield ~ 1 + (1 | batch)),
        MixedModels.dataset(:dyestuff);
        REML=false,
        progress=false,
    )
    s = cAIC.MMInternals.sigmahat(m)

    @test s isa Float64
    @test s > 0
    @test isfinite(s)
    @test s ≈ 49.5101 atol = 1e-3   # known ML residual SD for the dyestuff fit
    @inferred Float64 cAIC.MMInternals.sigmahat(m)
end

@testitem "bhat extracts the predicted random effects b̂ = λu, one block per grouping factor" begin
    using MixedModels

    # Single grouping factor: 1 random intercept × 6 batches.
    m = fit(
        MixedModel,
        @formula(yield ~ 1 + (1 | batch)),
        MixedModels.dataset(:dyestuff);
        progress=false,
    )
    b = cAIC.MMInternals.bhat(m)
    @test b isa Vector{Matrix{Float64}}
    @test length(b) == 1
    @test size(b[1]) == (1, 6)
    @inferred Vector{Matrix{Float64}} cAIC.MMInternals.bhat(m)

    # Crossed grouping factors: one block per factor.
    mc = fit(
        MixedModel,
        @formula(diameter ~ 1 + (1 | plate) + (1 | sample)),
        MixedModels.dataset(:penicillin);
        progress=false,
    )
    bc = cAIC.MMInternals.bhat(mc)
    @test length(bc) == 2
    @test all(blk -> blk isa Matrix{Float64}, bc)
end

@testitem "bhessian(:finitediff) returns the s×s deviance Hessian and restores θ̂" begin
    using cAIC
    using MixedModels

    # Self-driven finite differences over the *stable* objective (ADR-0002): the Hessian is
    # s×s on the deviance scale, and — the non-negotiable mutation contract — the model is
    # left at its fitted θ̂ afterwards (FiniteDiff parks `m` at its last probe; the driver
    # must restore). Exercised on s = 1 (random intercept) and s = 3 (correlated slope).
    for (form, s) in (
        (@formula(reaction ~ 1 + days + (1 | subj)), 1),
        (@formula(reaction ~ 1 + days + (1 + days | subj)), 3),
    )
        m = fit(
            MixedModel, form, MixedModels.dataset(:sleepstudy); REML=false, progress=false
        )
        θ̂ = copy(m.θ)
        obĵ = objective(m)

        B = cAIC.MMInternals.bhessian(m, :finitediff)

        @test B isa Matrix{Float64}
        @test size(B) == (s, s)
        @test B ≈ B' atol = 1e-6                  # a Hessian is symmetric
        @test m.θ == θ̂                            # restoration contract: fit untouched
        @test objective(m) ≈ obĵ                  # …and re-evaluable to the fitted objective
    end
end

@testitem "bhessian(:forwarddiff) returns the s×s deviance Hessian via the experimental ext" begin
    using cAIC
    using MixedModels

    # Rides MixedModelsForwardDiffExt (ADR-0002): the s×s Hessian on the deviance scale,
    # evaluated at θ̂. The extension differentiates an out-of-place objective (it copies A/L/
    # reterms), so unlike the self-driven FD path it does not mutate the fit. Exercised on
    # s = 1 (random intercept) and s = 3 (correlated slope).
    for (form, s) in (
        (@formula(reaction ~ 1 + days + (1 | subj)), 1),
        (@formula(reaction ~ 1 + days + (1 + days | subj)), 3),
    )
        m = fit(
            MixedModel, form, MixedModels.dataset(:sleepstudy); REML=false, progress=false
        )
        θ̂ = copy(m.θ)

        B = cAIC.MMInternals.bhessian(m, :forwarddiff)

        @test B isa Matrix{Float64}
        @test size(B) == (s, s)
        @test B ≈ B' atol = 1e-8           # a Hessian is symmetric
        @test m.θ == θ̂                      # the ext is out-of-place: fit untouched
    end
end

@testitem "bhessian rejects an unknown B-source" begin
    using cAIC
    using MixedModels
    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    @test_throws ArgumentError cAIC.MMInternals.bhessian(m, :analytic)
    @test_throws ArgumentError cAIC.MMInternals.bhessian(m, :nonsense)
end

@testitem "drift guard fails loud with a clear, version-pinned message" begin
    # The accessors' type/shape guards fire only when MixedModels drifts from the
    # pin — which cannot be triggered without mocking the quarantined dependency. So
    # we exercise the guard directly: it must raise (never return) a clear error
    # naming the touchpoint and the pinned version.
    drift = cAIC.MMInternals._drift
    @test_throws ErrorException drift("m.sigma", Float64, "not-a-float")
    @test_throws "m.sigma" drift("m.sigma", Float64, "not-a-float")
    @test_throws "5.5.1" drift("m.sigma", Float64, "not-a-float")
end
