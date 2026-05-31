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

# ── M4.5 Zhang-optimal weight optimizer (issue #50) ─────────────────────────────────

@testitem "WeightResult is a concrete parametric type with the right fields" tags = [
    :level1
] begin
    # Tracer bullet (CLAUDE §7 step 3): WeightResult{T} is defined, has the documented
    # fields, and the M=1 degenerate case returns weights=[1.0] (docs/math/0009 §2.3).
    using MixedModels
    using cAIC: cAIC

    @test isconcretetype(cAIC.WeightResult{Float64})
    @test fieldnames(cAIC.WeightResult) == (:weights, :objective, :duration)

    # M=1 degenerate: ŵ = (1), J = (y-μ₁)ᵀ(y-μ₁) + 2σ²ρ₁ — no optimizer needed
    y = [1.0, 2.0, 3.0]
    mu = reshape([1.5, 2.0, 2.8], 3, 1)    # 3×1 matrix
    rho = [2.5]
    sigma_sq = 1.2
    res = cAIC._getweights_raw(y, mu, rho, sigma_sq)
    @test res isa cAIC.WeightResult{Float64}
    @test res.weights ≈ [1.0]
    expected_J = sum((y - mu[:, 1]) .^ 2) + 2 * sigma_sq * rho[1]
    @test res.objective ≈ expected_J
    @test res.duration isa Float64
end

@testitem "Zhang Level-1: _getweights_raw matches cAIC4/.weightOptim on synthetic inputs (case 1, M=3)" tags = [
    :level1
] begin
    # Level-1 isolation (CLAUDE §6 / ADR-0003 / docs/math/0009 §7): feeds IDENTICAL
    # synthetic (y, mu, rho, sigma_sq) to Julia's _getweights_raw and the R-pre-computed
    # fixture. No model fitting — pure optimizer transcription check. Both sides converge
    # to the same unique minimiser (MᵀM ≻ 0, well-conditioned) so weight vectors agree.
    # Tolerance rtol=1e-6 (docs/math/0009 §7 target; relaxed band recorded in DECISIONS
    # at implementation if the iterative stopping band forces it).
    using HDF5
    using cAIC: cAIC

    L1_RTOL = 1e-6
    L1_ATOL = 1e-10

    fixture = joinpath(@__DIR__, "fixtures", "zhang_weights_level1.h5")
    @test isfile(fixture)

    y = h5read(fixture, "case1/inputs/y")
    mu = h5read(fixture, "case1/inputs/mu")       # HDF5 stores as column-major
    rho = h5read(fixture, "case1/inputs/rho")
    sigma_sq = only(h5read(fixture, "case1/inputs/sigma_sq"))
    r_weights = h5read(fixture, "case1/outputs_r/weights")
    r_objective = only(h5read(fixture, "case1/outputs_r/objective"))

    res = cAIC._getweights_raw(y, mu, rho, sigma_sq)

    @test res.weights ≈ r_weights rtol = L1_RTOL atol = L1_ATOL
    # Renormalized onto the unit simplex: sums to 1 to machine precision, not just the
    # SQP convergence tolerance (DECISIONS.md 2026-05-31, simplex projection).
    @test abs(sum(res.weights) - 1) ≤ 1e-12
    @test all(≥(-1e-10), res.weights)
    @test res.objective ≈ r_objective rtol = L1_RTOL atol = L1_ATOL
end

@testitem "Zhang Level-1: _getweights_raw matches cAIC4/.weightOptim on synthetic inputs (case 2, M=2)" tags = [
    :level1
] begin
    using HDF5
    using cAIC: cAIC

    L1_RTOL = 1e-6
    L1_ATOL = 1e-10

    fixture = joinpath(@__DIR__, "fixtures", "zhang_weights_level1.h5")
    @test isfile(fixture)

    y = h5read(fixture, "case2/inputs/y")
    mu = h5read(fixture, "case2/inputs/mu")
    rho = h5read(fixture, "case2/inputs/rho")
    sigma_sq = only(h5read(fixture, "case2/inputs/sigma_sq"))
    r_weights = h5read(fixture, "case2/outputs_r/weights")
    r_objective = only(h5read(fixture, "case2/outputs_r/objective"))

    res = cAIC._getweights_raw(y, mu, rho, sigma_sq)

    @test res.weights ≈ r_weights rtol = L1_RTOL atol = L1_ATOL
    # Renormalized onto the unit simplex: sums to 1 to machine precision, not just the
    # SQP convergence tolerance (DECISIONS.md 2026-05-31, simplex projection).
    @test abs(sum(res.weights) - 1) ≤ 1e-12
    @test all(≥(-1e-10), res.weights)
    @test res.objective ≈ r_objective rtol = L1_RTOL atol = L1_ATOL
end

@testitem "getweights(ModelAvgResult) returns a WeightResult with weights summing to 1" tags = [
    :level2
] begin
    # End-to-end: fit two sleepstudy candidates, build ModelAvgResult (Buckland path),
    # call getweights to optimize Zhang weights. The result must be a WeightResult{Float64}
    # with non-negative weights summing to 1.
    using MixedModels
    using cAIC: modelavg, getweights, WeightResult, ModelAvgResult

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

    res_buckland = modelavg(m1, m2; weights=:smoothed)
    wr = getweights(res_buckland)

    @test wr isa WeightResult{Float64}
    @test length(wr.weights) == 2
    @test all(≥(-1e-10), wr.weights)
    @test sum(wr.weights) ≈ 1.0 atol = 1e-8
    @test isfinite(wr.objective)
    @test wr.objective >= 0
    @test wr.duration isa Float64
end

@testitem "getweights is type-stable" tags = [:level1] begin
    using MixedModels, Test
    using cAIC: modelavg, getweights, WeightResult

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
    res_buckland = modelavg(m1, m2; weights=:smoothed)
    wr = @inferred WeightResult{Float64} getweights(res_buckland)
    @test wr isa WeightResult{Float64}
end

@testitem "modelavg with weights=:zhang calls getweights and returns a ModelAvgResult" tags = [
    :level2
] begin
    # modelavg(...; weights=:zhang) uses the Zhang optimizer internally and stores
    # the WeightResult in the returned ModelAvgResult (weighttype=:zhang).
    using MixedModels
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

    res = modelavg(m1, m2; weights=:zhang)
    @test res isa ModelAvgResult{Float64}
    @test res.weighttype == :zhang
    @test length(res.weights) == 2
    @test all(≥(-1e-10), res.weights)
    @test sum(res.weights) ≈ 1.0 atol = 1e-8
end

@testitem "live R re-validation of the Zhang Level-1 fixture (gated by CAIC_LIVE_RCALL)" tags = [
    :live_rcall
] begin
    # Fixture-rot guard: regenerate the Level-1 Zhang fixture with live cAIC4
    # and verify the committed fixture has not drifted. Skipped unless CAIC_LIVE_RCALL=1.
    using HDF5
    using cAIC: cAIC

    if get(ENV, "CAIC_LIVE_RCALL", "0") == "1"
        here = @__DIR__
        tmp = joinpath(mktempdir(), "zhang_weights_level1.h5")
        run(
            addenv(
                `Rscript $(joinpath(here, "generate_fixtures_zhang_level1.R"))`,
                "FIXTURE" => tmp,
            ),
        )
        for case_id in ("case1", "case2")
            committed_w = h5read(
                joinpath(here, "fixtures", "zhang_weights_level1.h5"),
                "$case_id/outputs_r/weights",
            )
            live_w = h5read(tmp, "$case_id/outputs_r/weights")
            @test live_w ≈ committed_w rtol = 1e-8 atol = 1e-8
        end
    else
        @info "Skipping Zhang Level-1 live-RCall re-validation (set CAIC_LIVE_RCALL=1)"
    end
end

# ── M4.5 issue #51: wire optimal weights as default + Level-2 anchor ────────────────────

@testitem "modelavg default weights scheme is :zhang (Zhang-optimal)" tags = [:level2] begin
    # The default weight scheme for modelavg is Zhang-optimal (:zhang), mirroring
    # cAIC4's modelAvg(opt=TRUE) default. Calling modelavg without the weights kwarg
    # must produce weighttype == :zhang, not :smoothed.
    using MixedModels
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

    res = modelavg(m1, m2)
    @test res isa ModelAvgResult{Float64}
    @test res.weighttype == :zhang
    @test length(res.weights) == 2
    @test all(≥(-1e-10), res.weights)
    @test sum(res.weights) ≈ 1.0 atol = 1e-8
end

@testitem "modelavg :zhang stores a WeightResult in res.weightresult" tags = [:level2] begin
    # Wire: ModelAvgResult from the :zhang path carries the full WeightResult
    # (weights, objective, duration) so callers can inspect J(ŵ) without re-scoring.
    # The :smoothed path stores nothing (weightresult === nothing).
    using MixedModels
    using cAIC: modelavg, ModelAvgResult, WeightResult

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

    res_zhang = modelavg(m1, m2; weights=:zhang)
    @test hasproperty(res_zhang, :weightresult)
    @test res_zhang.weightresult isa WeightResult{Float64}
    @test res_zhang.weightresult.weights == res_zhang.weights
    @test isfinite(res_zhang.weightresult.objective)
    @test res_zhang.weightresult.objective ≥ 0

    res_buckland = modelavg(m1, m2; weights=:smoothed)
    @test res_buckland.weightresult === nothing
end

@testitem "getweights on :zhang ModelAvgResult returns cached weightresult" tags = [:level2] begin
    # When modelavg was called with :zhang, getweights should return the already-computed
    # WeightResult directly (no re-running caic). Result is identical to res.weightresult.
    using MixedModels
    using cAIC: modelavg, getweights, WeightResult

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

    res = modelavg(m1, m2; weights=:zhang)
    wr = getweights(res)
    @test wr isa WeightResult{Float64}
    @test wr.weights ≈ res.weightresult.weights
    @test wr.objective ≈ res.weightresult.objective
end

@testitem "modelavg Zhang Level-2: optimal weights match cAIC4 modelAvg(opt=TRUE) on well-conditioned set" tags = [
    :level2
] begin
    # Level-2 anchor (docs/math/0009 §7, ADR-0007): fits a WELL-CONDITIONED candidate set
    # (reaction ~ days FE present vs absent → genuinely different conditional means →
    # MᵀM ≻ 0, unique QP minimiser) in MixedModels.jl, runs modelavg (default :zhang), and
    # compares the weight vector to the cAIC4::modelAvg(opt=TRUE) reference from the
    # committed fixture. Band = max(lme4↔MM fit discrepancy, §6.1 df-rounding perturbation);
    # measured and recorded in DECISIONS.md.
    using MixedModels, HDF5
    using cAIC: modelavg

    asscalar(x) = x isa AbstractArray ? only(x) : x
    L2_ATOL = 1e-2  # wider than Buckland (1e-3): absorbs df-rounding perturbation

    data = MixedModels.dataset(:sleepstudy)
    # input order MUST match the R generator (full-slope, intercept-only) — two candidates
    # with DIFFERENT fixed-effects structure ensure genuinely different conditional means.
    forms = [
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        @formula(reaction ~ 1 + (1 | subj)),
    ]
    ms = [fit(MixedModel, f, data; REML=false, progress=false) for f in forms]
    res = modelavg(ms...)  # default :zhang

    fixture = joinpath(@__DIR__, "fixtures", "zhang_modelavg_level2.h5")
    @test isfile(fixture)
    h5open(fixture, "r") do f
        Rcaic = read(f["caic"])      # full-precision cAIC4 cAIC, input order
        Rweights = read(f["weights"])   # modelAvg(opt=TRUE) weights, input order
        Robj = asscalar(read(f["objective"]))  # J(ŵ) from cAIC4

        # per-candidate cAIC within M2 band
        for i in eachindex(Rcaic)
            @test res.caics[i] ≈ Rcaic[i] atol = 1e-3
        end
        # Zhang weight vector end-to-end
        @test res.weights ≈ Rweights atol = L2_ATOL
        @test sum(res.weights) ≈ 1.0
        # objective value J(ŵ): magnitude O(n·σ̂²) ≈ 140000; relative band absorbs fit discrepancy.
        # Observed deviation: |ΔJ| ≈ 1.04, rtol ≈ 7.4e-6 (see DECISIONS.md).
        @test res.weightresult.objective ≈ Robj rtol = 1e-4
    end
end

# ── M4.5 issue #54: edge-case hardening ─────────────────────────────────────────────────

@testitem "modelavg M=1 :zhang short-circuit: single model returns weights=[1.0] (Level-2)" tags = [
    :level2
] begin
    # docs/math/0009 §2.3: a single candidate is the trivially unique minimiser — _getweights_raw
    # hits the nw==1 branch and returns ŵ=(1), J=(y−μ₁)ᵀ(y−μ₁)+2σ̂²ρ₁, skipping the SQP.
    # This exercises the full modelavg(:zhang) pipeline with M=1, ensuring the short-circuit
    # propagates correctly through the scoring, sigma_sq, and effect-averaging steps.
    using MixedModels
    using cAIC: modelavg, ModelAvgResult, WeightResult

    data = MixedModels.dataset(:sleepstudy)
    m1 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        data;
        REML=false,
        progress=false,
    )

    res = modelavg(m1; weights=:zhang)

    @test res isa ModelAvgResult{Float64}
    @test res.weighttype == :zhang
    @test length(res.weights) == 1
    @test res.weights ≈ [1.0]
    @test sum(res.weights) ≈ 1.0

    # WeightResult is stored; the objective formula (y-μ)ᵀ(y-μ)+2σ²ρ is already validated
    # at Level-1 (WeightResult tracer test). Here we just verify the result is finite.
    @test res.weightresult isa WeightResult{Float64}
    @test isfinite(res.weightresult.objective)
    @test res.weightresult.objective ≥ 0
    @test res.weightresult.duration == 0.0
end

@testitem "modelavg M=1 :smoothed short-circuit: single model returns weights=[1.0] (Level-2)" tags = [
    :level2
] begin
    # _bucklandweights([c]) = exp(0)/1 = [1.0]: the Δᵢ=0 term gives weight 1 to the sole
    # candidate regardless of its absolute cAIC value.
    using MixedModels
    using cAIC: modelavg, ModelAvgResult

    data = MixedModels.dataset(:sleepstudy)
    m1 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        data;
        REML=false,
        progress=false,
    )

    res = modelavg(m1; weights=:smoothed)

    @test res isa ModelAvgResult{Float64}
    @test res.weighttype == :smoothed
    @test length(res.weights) == 1
    @test res.weights ≈ [1.0]
    @test res.weightresult === nothing
end

@testitem "_weightoptim with negative-definite Hessian emits @warn and returns valid fallback (Level-1)" tags = [
    :level1
] begin
    # Regression guard for the ill-conditioned fallback paths (averaging.jl). A negative-
    # definite Hessian with lambda=0 forces cholesky(Symmetric(hess)) to throw PosDefException
    # on the first LM-ramp step, exercising the @warn + early-return branch. The fallback
    # must: (a) emit exactly the documented warning, and (b) return a NamedTuple with finite p.
    using LinearAlgebra, Test
    using cAIC: cAIC

    T = Float64
    nw = 2
    y = T[1.0, 2.0, 3.0]
    mu_mat = T[1.5 1.4; 2.0 2.1; 2.8 2.9]
    rho_v = T[2.5, 3.0]
    sigma_sq = T(1.2)
    equB = one(T)
    lowb = zeros(T, nw)
    uppb = ones(T, nw)

    find_weights = let y = y, mu = mu_mat, σ² = sigma_sq, ρ = rho_v
        w -> let r = y .- mu * w
            dot(r, r) + 2 * σ² * dot(ρ, w)
        end
    end

    w0 = fill(one(T) / nw, nw)
    funv = find_weights(w0)
    eqv = sum(w0) - equB                   # = 0.0 (feasible start)
    tol = T(1e-8)
    sc1 = min(max(abs(funv), tol), one(T) / tol)
    sc2 = min(max(abs(eqv), tol), one(T) / tol)
    scaler = vcat(T[sc1, sc2], ones(T, nw))

    # Negative-definite Hessian: cholesky(Symmetric(-I + 0·D)) throws immediately
    hess_nd = -Matrix{T}(I, nw, nw)

    res = @test_logs (:warn, r"Cholesky decomposition failed") cAIC._weightoptim(
        w0, zero(T), T[funv, eqv], hess_nd, zero(T), scaler, find_weights, equB, lowb, uppb
    )
    @test length(res.p) == nw
    @test all(isfinite, res.p)
end

@testitem "_bucklandweights with uniform cAIC gives uniform weights (Level-1)" tags = [
    :level1
] begin
    # When all candidates carry the same cAIC, Δᵢ=0 for every i, so wᵢ=1/M (maximum
    # entropy). The log-space computation must not introduce rounding asymmetries.
    using cAIC: cAIC

    for M in (2, 3, 5)
        w = cAIC._bucklandweights(fill(42.7, M))
        @test length(w) == M
        @test w ≈ fill(1.0 / M, M)
        @test sum(w) ≈ 1.0
    end
end

@testitem "_getweights_raw with all-zero rho: penalty vanishes, returns valid WeightResult (Level-1)" tags = [
    :level1
] begin
    # When every ρᵢ=0, the penalty 2σ²(ρᵀw)=0 regardless of w and the Mallows criterion
    # reduces to pure RSS minimisation. The optimizer must still converge to a weight vector
    # on the unit simplex (non-negative, sum=1) with a finite, non-negative objective.
    using cAIC: cAIC

    T = Float64
    y = T[1.0, 2.0, 3.0]
    mu = T[1.5 1.4; 2.0 2.1; 2.8 2.9]
    rho = zeros(T, 2)           # all-zero effective-df: penalty term drops out
    sigma_sq = T(1.2)

    res = cAIC._getweights_raw(y, mu, rho, sigma_sq)

    @test res isa cAIC.WeightResult{T}
    @test length(res.weights) == 2
    @test all(≥(-1e-10), res.weights)
    @test sum(res.weights) ≈ 1.0 atol = 1e-8
    @test isfinite(res.objective)
    @test res.objective ≥ 0
end

@testitem "modelavg :zhang on duplicate (collinear) candidates converges cleanly to a simplex-valid weight (Level-2)" tags = [
    :level2
] begin
    # docs/math/0009 §2.3 / ADR-0007 decision 4 say duplicate/collinear candidates make MᵀM
    # singular so a try-error fallback *may* fire (warned). Empirically, on a natural fit it
    # does NOT: with identical μ columns and ρ₁=ρ₂ the residual (y−μw) is constant on the
    # simplex and the penalty is symmetric, so J is flat — the SQP stays at its w⁰=(1/M,…)
    # start, and the Levenberg ramp (hess+λ·D², λ×3) keeps the Cholesky PD throughout. The
    # @warn fallback is thus unreachable from a real collinear fit; it is locked in separately
    # at Level-1 by forcing a negative-definite Hessian (DECISIONS 2026-05-31). This test pins
    # the honest end-to-end behavior: a simplex-valid weight is returned with NO warning fired.
    using MixedModels, Test, Logging
    using cAIC: modelavg, ModelAvgResult, WeightResult

    data = MixedModels.dataset(:sleepstudy)
    f = @formula(reaction ~ 1 + days + (1 + days | subj))
    m1 = fit(MixedModel, f, data; REML=false, progress=false)
    m2 = fit(MixedModel, f, data; REML=false, progress=false)   # identical fit ⇒ collinear μ

    # No ill-conditioned fallback @warn is emitted on the natural collinear set.
    res = @test_logs min_level = Logging.Warn modelavg(m1, m2; weights=:zhang)

    @test res isa ModelAvgResult{Float64}
    @test res.weighttype == :zhang
    @test length(res.weights) == 2
    @test all(≥(-1e-10), res.weights)          # non-negative
    @test sum(res.weights) ≈ 1.0 atol = 1e-8    # on the unit simplex
    # The flat objective ⇒ the optimizer returns its symmetric start (one valid, non-unique
    # minimiser of the many on the simplex); §7 anchors stable functionals, not ŵ itself.
    @test res.weights ≈ [0.5, 0.5] atol = 1e-6
    @test res.weightresult isa WeightResult{Float64}
    @test isfinite(res.weightresult.objective)
    @test res.weightresult.objective ≥ 0
end

@testitem "modelavg rejects an unknown weights= scheme with ArgumentError (Level-2)" tags = [
    :level2
] begin
    # Degenerate-input guard (CLAUDE §4 fail-loud): only :zhang and :smoothed are supported
    # weight schemes. Any other symbol must raise ArgumentError, not silently fall through.
    using MixedModels, Test
    using cAIC: modelavg

    data = MixedModels.dataset(:sleepstudy)
    m1 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        data;
        REML=false,
        progress=false,
    )

    @test_throws ArgumentError modelavg(m1; weights=:bogus)
    @test_throws ArgumentError modelavg(m1; weights=:optimal)   # plausible-but-wrong name
end

@testitem "modelavg is type-stable on the M=1 :zhang degenerate path (Level-2)" tags = [
    :level2
] begin
    # Acceptance criterion 4 (#54): @inferred type-stability is preserved on the degenerate
    # branch. M=1 routes through the nw==1 short-circuit in _getweights_raw, a different code
    # path than the M≥2 optimizer; it must still infer to a concrete ModelAvgResult{Float64}.
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

    res = @inferred ModelAvgResult{Float64} modelavg(m1; weights=:zhang)
    @test res isa ModelAvgResult{Float64}
    @test res.weights ≈ [1.0]
end

# ── M4.5 issue #52: predictma — weighted conditional prediction ─────────────────────────

@testitem "predictma weight-combines per-candidate conditional predictions (tracer)" tags = [
    :level2
] begin
    # Tracer (CLAUDE §7 step 3): the model-averaged prediction on the TRAINING data is the
    # weighted sum of each candidate's conditional prediction, ŷ^MA = Σ wᵢ predict(mᵢ, D*)
    # (docs/math/0009 §5; port of cAIC4's `w %*% t(sapply(models, predict, newdata))`).
    using MixedModels
    using cAIC: modelavg, predictma

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
    yhat = predictma(res, data)

    @test yhat isa Vector{Float64}
    @test length(yhat) == length(data.reaction)

    w = res.weights
    expected = w[1] .* predict(m1, data) .+ w[2] .* predict(m2, data)
    @test yhat ≈ expected
end

@testitem "predictma default new_re_levels=:error raises on an unseen grouping level" tags = [
    :level2
] begin
    # docs/math/0009 §5/§6.3: the default new_re_levels=:error mirrors lme4's
    # allow.new.levels=FALSE — a grouping level absent from training raises ArgumentError
    # (overriding MixedModels' own :missing default). The error originates in
    # MixedModels.predict and propagates through the weighted combination.
    using MixedModels
    using cAIC: modelavg, predictma

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

    # a row carrying a subject level never seen in sleepstudy
    newdata = (reaction=[300.0], days=[3.0], subj=["BRAND_NEW_SUBJ"])
    @test_throws ArgumentError predictma(res, newdata)
    # explicit :error is identical to the default
    @test_throws ArgumentError predictma(res, newdata; new_re_levels=:error)
end

@testitem "predictma new_re_levels=:population forwards through to a population prediction" tags = [
    :level2
] begin
    # docs/math/0009 §5: the opt-in new_re_levels=:population treats an unseen grouping
    # level's random effect as 0, so the model-averaged prediction on a brand-new subject is
    # the weighted combination of each candidate's population (fixed-effects-only for that row)
    # prediction — and must NOT error.
    using MixedModels
    using cAIC: modelavg, predictma

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

    newdata = (reaction=[300.0], days=[3.0], subj=["BRAND_NEW_SUBJ"])
    yhat = predictma(res, newdata; new_re_levels=:population)

    @test yhat isa Vector{Float64}
    @test length(yhat) == 1
    w = res.weights
    expected =
        w[1] .* predict(m1, newdata; new_re_levels=:population) .+
        w[2] .* predict(m2, newdata; new_re_levels=:population)
    @test yhat ≈ expected
    # population prediction for a new subject = fixed effects only (RE = 0). For m1/m2 the
    # FE are (Intercept)+days, so the per-candidate value is β̂₀ + β̂₁·3; verify it is finite
    # and not accidentally NaN/missing-poisoned.
    @test all(isfinite, yhat)
end

@testitem "predictma is type-stable (CLAUDE §8)" tags = [:level2] begin
    # The weighted-combination loop must infer to a concrete Vector{Float64} despite
    # MixedModels.predict's new_re_levels-dependent return type (constant-propagated default
    # :error keeps the eltype Float64, not Union{Float64,Missing}).
    using MixedModels, Test
    using cAIC: modelavg, predictma

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

    yhat = @inferred Vector{Float64} predictma(res, data)
    @test yhat isa Vector{Float64}
end

@testitem "predictma Level-2: ŷ^MA matches cAIC4 predictMA on well-conditioned + nested sets" tags = [
    :level2
] begin
    # Level-2 stable-functional anchor (docs/math/0009 §5/§7, issue #52): the model-averaged
    # PREDICTION ŷ^MA = Σ wᵢ predict(mᵢ, D*) is the functional anchored on EVERY scenario,
    # including the nested set — it is stable under a non-unique weight vector (the M4.5
    # analogue of stepcaic's path-only-on-well-separated-cases rule). The fixture is written by
    # generate_fixtures_predictma.R (cAIC4 modelAvg(opt=TRUE) + predictMA); no R runs here.
    #
    # Two scenarios, both predicting on the TRAINING data (every level seen → :error path):
    #   wc      — well-conditioned, distinct FE (MᵀM ≻ 0): anchor BOTH the prediction and the
    #             weight vector (the minimiser is unique).
    #   nested  — three nested candidates (Orthodont-style): anchor ONLY the prediction (the
    #             stable functional); the weight vector is not pinned (§7 discipline).
    #
    # Band atol = 5e-3 (~3× the measured worst-case |Δŷ^MA| = 1.63e-3, driven by the lme4↔MM
    # fit discrepancy on the response scale; DECISIONS 2026-05-31). A wrong combination shifts
    # predictions by O(1)+ ms — orders of magnitude above this band.
    using MixedModels, HDF5
    using cAIC: modelavg, predictma

    L2_ATOL_PRED = 5e-3
    L2_ATOL_CAIC = 1e-3
    L2_ATOL_W = 1e-2

    data = MixedModels.dataset(:sleepstudy)
    fixture = joinpath(@__DIR__, "fixtures", "predictma_level2.h5")
    @test isfile(fixture)

    # ── Scenario wc: well-conditioned, anchor prediction AND weights ──────────────────────
    wc_forms = [
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        @formula(reaction ~ 1 + (1 | subj)),
    ]
    wc_ms = [fit(MixedModel, f, data; REML=false, progress=false) for f in wc_forms]
    wc_res = modelavg(wc_ms...)   # default :zhang
    wc_pred = predictma(wc_res, data)

    h5open(fixture, "r") do f
        Rcaic = read(f["wc/caic"])
        Rweights = read(f["wc/weights"])
        Rpred = read(f["wc/prediction"])

        for i in eachindex(Rcaic)
            @test wc_res.caics[i] ≈ Rcaic[i] atol = L2_ATOL_CAIC
        end
        @test wc_res.weights ≈ Rweights atol = L2_ATOL_W
        @test length(wc_pred) == length(Rpred)
        # per-observation band (elementwise), not the aggregated L2 norm over n = 180
        @test maximum(abs.(wc_pred .- Rpred)) <= L2_ATOL_PRED
    end

    # ── Scenario nested: anchor the prediction (stable functional) only ───────────────────
    nested_forms = [
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        @formula(reaction ~ 1 + days + (1 | subj)),
        @formula(reaction ~ 1 + (1 | subj)),
    ]
    nested_ms = [fit(MixedModel, f, data; REML=false, progress=false) for f in nested_forms]
    nested_res = modelavg(nested_ms...)   # default :zhang
    nested_pred = predictma(nested_res, data)

    h5open(fixture, "r") do f
        Rcaic = read(f["nested/caic"])
        Rpred = read(f["nested/prediction"])

        for i in eachindex(Rcaic)
            @test nested_res.caics[i] ≈ Rcaic[i] atol = L2_ATOL_CAIC
        end
        # the stable functional: prediction matches even though the weight vector is not pinned
        @test length(nested_pred) == length(Rpred)
        # per-observation band (elementwise), not the aggregated L2 norm over n = 180
        @test maximum(abs.(nested_pred .- Rpred)) <= L2_ATOL_PRED
        # weights remain a valid probability vector
        @test sum(nested_res.weights) ≈ 1.0 atol = 1e-8
        @test all(≥(-1e-10), nested_res.weights)
    end
end

@testitem "live R re-validation of the predictma Level-2 fixture (gated by CAIC_LIVE_RCALL)" tags = [
    :live_rcall
] begin
    # Fixture-rot guard (CLAUDE §6) for the predictMA Level-2 fixture: regenerate the
    # references with live lme4 + cAIC4 and check the committed reference has not drifted.
    # Skipped in the default (no-R) job; enabled by CAIC_LIVE_RCALL=1.
    using HDF5

    if get(ENV, "CAIC_LIVE_RCALL", "0") == "1"
        here = @__DIR__
        committed = joinpath(here, "fixtures", "predictma_level2.h5")
        tmp = joinpath(mktempdir(), "predictma_level2.h5")
        run(
            addenv(
                `Rscript $(joinpath(here, "generate_fixtures_predictma.R"))`,
                "FIXTURE" => tmp,
            ),
        )
        for s in ("wc", "nested")
            comm_p = h5read(committed, "$s/prediction")
            live_p = h5read(tmp, "$s/prediction")
            comm_w = h5read(committed, "$s/weights")
            live_w = h5read(tmp, "$s/weights")
            @test live_p ≈ comm_p rtol = 1e-8 atol = 1e-8
            @test live_w ≈ comm_w rtol = 1e-8 atol = 1e-8
        end
    else
        @info "Skipping predictMA Level-2 live-RCall re-validation (set CAIC_LIVE_RCALL=1)"
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
