@testitem "modelavg returns a ModelAvgResult with Buckland weights summing to 1" tags = [
    :level2
] begin
    # Tracer: the first end-to-end run of modelavg on the smoothed (Buckland) path.
    # Two sleepstudy candidates differing in RE structure are scored and combined; the
    # exponential-cAIC weights must form a probability vector (non-negative, sum to 1).
    using MixedModels
    using cAIC: modelavg, ModelAvgResult, caic

    data = MixedModels.dataset(:sleepstudy)
    m_slope = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        data;
        REML=false,
        progress=false,
    )
    m_int = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        data;
        REML=false,
        progress=false,
    )

    res = modelavg(m_slope, m_int; weights=:smoothed)
    @test res isa ModelAvgResult{Float64}
    @test length(res.weights) == 2
    @test all(≥(0), res.weights)
    @test sum(res.weights) ≈ 1.0

    # Buckland: wᵢ = exp(−Δᵢ/2)/Σ exp(−Δ/2), Δᵢ = cAICᵢ − min cAIC, scored in INPUT order.
    c1 = caic(m_slope).caic
    c2 = caic(m_int).caic
    @test res.caics ≈ [c1, c2]
    expw = exp.(-([c1, c2] .- min(c1, c2)) ./ 2)
    expw ./= sum(expw)
    @test res.weights ≈ expw
end

@testitem "modelavg fixeff is a name-keyed weighted sum over the union of FE terms" tags = [
    :level2
] begin
    # Candidates with DIFFERING fixed-effects structure: m1 has (Intercept)+days, m2 has
    # only (Intercept). The averaged fixeff is keyed on the union of coefficient names; a
    # term present in only one candidate equals that candidate's weight × coefficient.
    using MixedModels
    using cAIC: modelavg

    data = MixedModels.dataset(:sleepstudy)
    m1 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        data;
        REML=false,
        progress=false,
    )
    m2 = fit(
        MixedModel, @formula(reaction ~ 1 + (1 | subj)), data; REML=false, progress=false
    )

    res = modelavg(m1, m2; weights=:smoothed)
    w = res.weights

    # union of coefficient names, name-sorted
    @test res.fixeff.keys == ["(Intercept)", "days"]

    b1 = Dict(zip(fixefnames(m1), fixef(m1)))   # ("(Intercept)", "days")
    b2 = Dict(zip(fixefnames(m2), fixef(m2)))   # ("(Intercept)",)

    # (Intercept): present in both → weighted sum across candidates
    @test res.fixeff["(Intercept)"] ≈ w[1] * b1["(Intercept)"] + w[2] * b2["(Intercept)"]
    # days: present only in m1 → exactly that candidate's weight × coefficient (m2 ⇒ 0)
    @test res.fixeff["days"] ≈ w[1] * b1["days"]
    @test !haskey(b2, "days")
end

@testitem "modelavg raneff is keyed on (grouping, level, term) over the RE-term union" tags = [
    :level2
] begin
    # Candidates with DIFFERING random-effects structure: m1 has (1 + days | subj) — both
    # an (Intercept) and a days mode per subject — while m2 has (1 | subj) — (Intercept)
    # only. The averaged raneff is keyed on (grouping factor, level, RE term); a (level,
    # term) present in only one candidate equals that candidate's weight × mode.
    using MixedModels
    using cAIC: modelavg

    data = MixedModels.dataset(:sleepstudy)
    m1 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        data;
        REML=false,
        progress=false,
    )
    m2 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        data;
        REML=false,
        progress=false,
    )

    res = modelavg(m1, m2; weights=:smoothed)
    w = res.weights

    t1 = raneftables(m1).subj          # columns: subj, (Intercept), days
    t2 = raneftables(m2).subj          # columns: subj, (Intercept)
    lev = string(t1.subj[1])           # first subject level, e.g. "S308"
    int1 = t1.var"(Intercept)"[1]
    days1 = t1.days[1]
    int2 = t2.var"(Intercept)"[1]

    # the keys exist over the union and the grouping/term labels are as expected
    @test ("subj", lev, "(Intercept)") in res.raneff.keys
    @test ("subj", lev, "days") in res.raneff.keys

    # (Intercept) mode: present in both candidates → weighted sum
    @test res.raneff[("subj", lev, "(Intercept)")] ≈ w[1] * int1 + w[2] * int2
    # days mode: present only in m1 → exactly its weight × mode (m2 has no days mode ⇒ 0)
    @test res.raneff[("subj", lev, "days")] ≈ w[1] * days1
    @test !(("subj", lev, "days") in propertynames(t2))
end

@testitem "modelavg rejects an inconsistent candidate set (response/n, REML)" tags = [
    :level2
] begin
    # The fail-loud candidate-set contract (docs/math/0009 §0): mismatched observation count,
    # a differing response on the same n, and a mixed REML setting each raise ArgumentError.
    using MixedModels
    using cAIC: modelavg

    sleep = MixedModels.dataset(:sleepstudy)
    m_ml = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        sleep;
        REML=false,
        progress=false,
    )

    # mixed REML setting
    m_reml = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        sleep;
        REML=true,
        progress=false,
    )
    @test_throws ArgumentError modelavg(m_ml, m_reml)

    # mismatched observation count (different dataset, n = 30 ≠ 180)
    dye = MixedModels.dataset(:dyestuff)
    m_dye = fit(
        MixedModel, @formula(yield ~ 1 + (1 | batch)), dye; REML=false, progress=false
    )
    @test_throws ArgumentError modelavg(m_ml, m_dye)

    # same n, different response values
    shifted = (
        reaction=collect(sleep.reaction) .+ 1.0,
        days=collect(sleep.days),
        subj=collect(sleep.subj),
    )
    m_shift = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        shifted;
        REML=false,
        progress=false,
    )
    @test_throws ArgumentError modelavg(m_ml, m_shift)
end

@testitem "modelavg rejects non-Gaussian (GLMM) candidates with ArgumentError" tags = [
    :level2
] begin
    # cAIC4 model averaging is Gaussian-LMM only (docs/math/0009 §0/§1): a GLMM candidate
    # hits the fail-loud MixedModel fallback rather than a MethodError.
    using MixedModels
    using cAIC: modelavg

    y = repeat([0, 1, 1, 0, 1, 0, 1, 1], outer=4)
    g = repeat(1:4, inner=8)
    glmm = fit(MixedModel, @formula(y ~ 1 + (1 | g)), (; y, g), Bernoulli(); progress=false)
    @test_throws ArgumentError modelavg(glmm)
end

@testitem "modelavg is type-stable and shows a ModelAvgResult summary" tags = [:level2] begin
    # Type stability of the assembly (CLAUDE §8) despite the heterogeneous raneftables walk
    # being behind a function barrier, plus a Base.show smoke check.
    using MixedModels, Test
    using cAIC: modelavg, ModelAvgResult

    data = MixedModels.dataset(:sleepstudy)
    m1 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        data;
        REML=false,
        progress=false,
    )
    m2 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        data;
        REML=false,
        progress=false,
    )

    res = @inferred ModelAvgResult{Float64} modelavg(m1, m2; weights=:smoothed)
    @test res isa ModelAvgResult{Float64}

    str = sprint(show, MIME("text/plain"), res)
    @test occursin("Model-averaged mixed model", str)
    @test occursin("Buckland", str)
end

@testitem "modelavg Buckland weights match cAIC4 modelAvg(opt=FALSE) end-to-end (Level-2)" tags = [
    :level2
] begin
    # The end-to-end gate for the smoothed (Buckland) path against cAIC4's modelAvg(opt=FALSE)
    # (fixture written by generate_fixtures_modelavg.R; no R runs here). The candidate set —
    # correlated slope / uncorrelated slope / intercept-only RE on sleepstudy — yields a
    # non-degenerate weight vector that exercises the exp(-Δ/2) shape. Per docs/math/0009 §7
    # the Buckland weights are a deterministic map of the (Level-2-validated) cAICs, so they
    # inherit the M2 band; modelAvg(opt=FALSE) additionally rounds cAIC to 2 digits before
    # weighting (anocAIC, methods.R:63), absorbed by the measured band (DECISIONS 2026-05-31).
    using MixedModels, HDF5
    using cAIC: modelavg

    asscalar(x) = x isa AbstractArray ? only(x) : x
    L2_ATOL = 1e-3

    data = MixedModels.dataset(:sleepstudy)
    # input order MUST match the R generator's `forms` (corr, uncorr, int)
    forms = [
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
        @formula(reaction ~ 1 + days + (1 | subj)),
    ]
    ms = [fit(MixedModel, f, data; REML=false, progress=false) for f in forms]
    res = modelavg(ms...; weights=:smoothed)

    fixture = joinpath(@__DIR__, "fixtures", "modelavg_level2.h5")
    @test isfile(fixture)
    h5open(fixture, "r") do f
        Rcaic = read(f["caic"])        # full-precision cAIC4 cAIC, input order
        Rweights = read(f["weights"])  # modelAvg(opt=FALSE) weights, input order
        Rfix = read(f["fixeff_vals"])  # averaged fixed effects (name-sorted)

        # per-candidate cAIC vs full-precision cAIC4 (M2 band)
        for i in eachindex(Rcaic)
            @test res.caics[i] ≈ Rcaic[i] atol = L2_ATOL
        end
        # Buckland weights end-to-end vs modelAvg(opt=FALSE), input order
        @test res.weights ≈ Rweights atol = L2_ATOL
        @test sum(res.weights) ≈ 1.0
        # model-averaged fixed effects (both name-sorted; the Days/days rename keeps order)
        @test res.fixeff.keys == ["(Intercept)", "days"]
        @test res.fixeff.values ≈ Rfix atol = L2_ATOL
    end
end

@testitem "live R re-validation of the modelavg Level-2 fixture (gated by CAIC_LIVE_RCALL)" tags = [
    :live_rcall
] begin
    # Fixture-rot guard (CLAUDE §6) for the model-averaging Level-2 fixture: regenerate the
    # references with live lme4 + cAIC4 and check the committed reference has not drifted.
    # Skipped in the default (no-R) job; enabled by CAIC_LIVE_RCALL=1.
    using HDF5

    if get(ENV, "CAIC_LIVE_RCALL", "0") == "1"
        here = @__DIR__
        committed = joinpath(here, "fixtures", "modelavg_level2.h5")
        comm_w = h5read(committed, "weights")
        comm_c = h5read(committed, "caic")

        tmp = joinpath(mktempdir(), "modelavg_level2.h5")
        run(
            addenv(
                `Rscript $(joinpath(here, "generate_fixtures_modelavg.R"))`,
                "FIXTURE" => tmp,
            ),
        )
        live_w = h5read(tmp, "weights")
        live_c = h5read(tmp, "caic")
        # deterministic lmer+cAIC4 recompute: committed and fresh agree to machine precision
        @test live_w ≈ comm_w rtol = 1e-8 atol = 1e-8
        @test live_c ≈ comm_c rtol = 1e-8 atol = 1e-8
    else
        @info "Skipping modelavg Level-2 live-RCall re-validation (set CAIC_LIVE_RCALL=1)"
    end
end
