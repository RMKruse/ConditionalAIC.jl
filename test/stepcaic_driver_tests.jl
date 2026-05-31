# Level-2 end-to-end tests for the greedy `stepcaic` driver (M4 §4.1, #40). Each fits the same
# model `cAIC4`'s `stepcAIC` was driven on (test/generate_fixtures_stepcaic_driver.R) with
# `MixedModels.jl`, runs `cAIC.stepcaic`, and asserts the selected RE structure (authored from
# cAIC4's decision — the fixture's `finalformula` is its provenance, grouping names differ across
# packages) plus `selected.caic ≈ bestCAIC` within the Level-2 fit-discrepancy band.

@testitem "stepcaic backward keeps the incumbent when no candidate improves (sleepstudy, Level-2)" tags = [
    :level2
] begin
    using HDF5
    using MixedModels
    using cAIC: caic, extract, RESpec, REGroup, stepcaic, StepcaicResult

    # rhdf5 stores an R length-1 numeric as a 1-element array; coerce before comparing.
    asscalar(x) = x isa AbstractArray ? only(x) : x

    data = MixedModels.dataset(:sleepstudy)
    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        data;
        REML=false,
        progress=false,
    )

    res = stepcaic(m, data; direction=:backward)
    @test res isa StepcaicResult

    # cAIC4's decision: backward selects the FULL model (no drop lowers the cAIC). Provenance:
    # fixture sleepstudy_slope/finalformula = "Reaction ~ 1 + Days + (1 + Days | Subject)".
    expected = RESpec([REGroup(:subj, ["(Intercept)", "days"], true)])
    @test extract(res.model) == expected

    # No step accepted ⇒ the selected score is exactly our own caic of the input (internal identity:
    # the driver scored the incumbent with the same kwargs and accepted nothing).
    @test res.selected.caic == caic(m).caic

    # … and matches cAIC4's bestCAIC within the Level-2 fit-discrepancy band (DECISIONS 2026-05-27;
    # the selected model IS the input full fit, so the band is the `caic` slope_ml band, atol=1e-3).
    L2_ATOL = 1e-3
    fixture = joinpath(@__DIR__, "fixtures", "stepcaic_driver_level2.h5")
    @test isfile(fixture)
    bestCAIC = h5open(fixture, "r") do f
        asscalar(read(f["sleepstudy_slope"]["bestCAIC"]))
    end
    @test res.selected.caic ≈ bestCAIC atol = L2_ATOL

    # the path recorded the single scoring round, and it was a rejection (no acceptance)
    @test length(res.path) == 1
    rec = only(res.path)
    @test rec.direction === :backward
    @test rec.accepted == false
    @test !isempty(rec.candidates)
    @test rec.incumbentcaic == res.selected.caic
end

@testitem "stepcaic backward descends to the lm terminal and rejects it (sleepstudy_int, Level-2)" tags = [
    :level2
] begin
    using HDF5
    using MixedModels
    using GLM: lm
    using cAIC: caic, extract, RESpec, REGroup, stepcaic

    asscalar(x) = x isa AbstractArray ? only(x) : x

    data = MixedModels.dataset(:sleepstudy)
    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        data;
        REML=false,
        progress=false,
    )

    res = stepcaic(m, data; direction=:backward)

    # cAIC4's decision: the only backward neighbour of a single random intercept is the `lm`/`glm`
    # terminal (`reaction ~ 1 + days`, no random effects). It is scored and REJECTED (its cAIC is
    # worse), so the search keeps `(1 | subj)`. Provenance: fixture sleepstudy_int/finalformula =
    # "Reaction ~ 1 + Days + (1 | Subject)", finalclass lmerMod.
    @test extract(res.model) == RESpec([REGroup(:subj, ["(Intercept)"], true)])
    @test res.model isa LinearMixedModel
    @test res.selected.caic == caic(m).caic

    L2_ATOL = 1e-3
    fixture = joinpath(@__DIR__, "fixtures", "stepcaic_driver_level2.h5")
    bestCAIC = h5open(fixture, "r") do f
        asscalar(read(f["sleepstudy_int"]["bestCAIC"]))
    end
    @test res.selected.caic ≈ bestCAIC atol = L2_ATOL

    # The terminal WAS scored: one rejected step, its sole candidate the `lm` node (`spec === nothing`),
    # scored by the same `caic(::TableRegressionModel)` the controller drives. cAIC4's terminal cAIC for
    # this lm is 1906.293 (`lm(Reaction ~ Days)`); the OLS fit is package-identical so it matches tightly.
    @test length(res.path) == 1
    rec = only(res.path)
    @test rec.direction === :backward
    @test rec.accepted == false
    @test length(rec.candidates) == 1
    cand = only(rec.candidates)
    @test cand.spec === nothing
    lmterm = caic(lm(@formula(reaction ~ 1 + days), data))
    @test cand.caic == lmterm.caic
    @test cand.caic ≈ 1906.293 atol = 1e-2
    @test cand.caic > res.selected.caic       # rejected: terminal is worse than the incumbent
end

@testitem "stepcaic backward descends to AND accepts the lm terminal when it wins (unsupported grouping)" begin
    using MixedModels
    using GLM: lm, StatsModels
    using Random: Xoshiro
    using cAIC: caic, stepcaic

    # An unsupported single-grouping model: `y ~ 1 + x + (1|g)` over data with NO true group
    # effect. The backward search reaches the single random intercept, descends to the `lm`
    # terminal (`y ~ 1 + x`), and — because the random intercept is not supported — the terminal
    # WINS the `≤` rule and is accepted-and-stopped, the `cAIC4` `bestModel ∈ {lm,glm} & ≤`
    # arc (`R/stepcAIC.R:223–229`). cAIC4 takes the same structural decision (its `finalModel`
    # carries class c("glm","lm")); the numeric `bestCAIC` is NOT anchored here because cAIC4's
    # stepCAIC scores that terminal as a glm with the dispersion σ̂, diverging from this lm/MLE
    # terminal (DECISIONS 2026-05-31). The score is instead pinned to the project's own (Level-2-
    # validated) `caic(lm)`. Deterministic via a fixed `Xoshiro` seed; the win margin (≈2.5) is far
    # wider than any fit perturbation.
    rng = Xoshiro(555)
    n = 150
    g = repeat(1:25, inner=6)
    x = randn(rng, n)
    y = 1.0 .+ 2.0 .* x .+ 0.15 .* randn(rng, 25)[g] .+ randn(rng, n)
    data = (; y, x, g)

    m = fit(MixedModel, @formula(y ~ 1 + x + (1 | g)), data; REML=false, progress=false)
    res = stepcaic(m, data; direction=:backward)

    # The selected model IS the lm terminal — a `TableRegressionModel`, not a `MixedModel`.
    @test res.model isa StatsModels.TableRegressionModel
    @test res.selected.method === :terminal
    # It strictly improved on the input mixed model, and equals the project's own lm-terminal score.
    @test res.selected.caic < caic(m).caic
    @test res.selected.caic == caic(lm(@formula(y ~ 1 + x), data)).caic
    # The saved k-best carries the selected (terminal) score, consistently typed.
    @test length(res.saved) == 1
    @test res.saved[1].caic == res.selected.caic

    # The final step descended to the terminal (sole candidate `spec === nothing`) and accepted it.
    rec = res.path[end]
    @test rec.direction === :backward
    @test rec.accepted == true
    @test length(rec.candidates) == 1
    @test only(rec.candidates).spec === nothing
    @test only(rec.candidates).caic == res.selected.caic
end

@testitem "stepcaic backward honours the keep floor: the pinned term is never dropped (Pastes, Level-2)" tags = [
    :level2
] begin
    using HDF5
    using MixedModels
    using cAIC: caic, extract, RESpec, REGroup, stepcaic

    asscalar(x) = x isa AbstractArray ? only(x) : x

    data = MixedModels.dataset(:pastes)
    m = fit(
        MixedModel,
        @formula(strength ~ 1 + (1 | batch) + (1 | cask)),
        data;
        REML=false,
        progress=false,
    )

    # `keep = (1 | batch)` pins the batch intercept: backward may drop `cask` but never `batch`,
    # and — crucially — the search does NOT descend to the `lm` terminal (the keep floor is the
    # smallest reachable model). cAIC4's decision: select `(1 | batch)` and stop (keep-minimal).
    # Provenance: fixture pastes_keepbatch/finalformula = "strength ~ (1 | batch)", keep "~(1 | batch)".
    # `keep` is supplied as a `FormulaTerm` RE fragment (the `cAIC4` `keep$random` analogue),
    # parsed to a `RESpec` floor against `data`.
    keep = @formula(strength ~ (1 | batch))
    res = stepcaic(m, data; direction=:backward, keep=keep)

    @test extract(res.model) == RESpec([REGroup(:batch, ["(Intercept)"], true)])
    @test res.model isa LinearMixedModel                       # never the lm terminal under keep

    L2_ATOL = 1e-3
    fixture = joinpath(@__DIR__, "fixtures", "stepcaic_driver_level2.h5")
    bestCAIC = h5open(fixture, "r") do f
        asscalar(read(f["pastes_keepbatch"]["bestCAIC"]))
    end
    @test res.selected.caic ≈ bestCAIC atol = L2_ATOL

    # Faithful path: cAIC4 stops in ONE step. `backwardStep`'s `keep` re-add reconstitutes the
    # unchanged crossed model as a candidate, but `makeBackward`→`mergeChanges` discards every
    # candidate equal to the incumbent (the drop-original step), leaving only the `cask`-drop —
    # a single candidate, accepted, single-candidate stop. No redundant self-rescore step.
    @test length(res.path) == 1

    incumbent = RESpec([
        REGroup(:batch, ["(Intercept)"], true), REGroup(:cask, ["(Intercept)"], true)
    ])
    # The keep floor held at EVERY step: every scored candidate keeps a `batch` intercept term,
    # none is the `lm` terminal (`spec === nothing`), and none is the unchanged incumbent itself
    # (mergeChanges drop-original).
    for rec in res.path
        for cand in rec.candidates
            @test cand.spec !== nothing
            @test cand.spec != incumbent
            @test any(
                g -> g.grouping === :batch && "(Intercept)" in g.directions,
                cand.spec.groups,
            )
        end
    end
end

@testitem "stepcaic backward saves the k-best scored models (savedmodels=2, Pastes, Level-2)" tags = [
    :level2
] begin
    using HDF5
    using MixedModels
    using cAIC: caic, extract, RESpec, REGroup, stepcaic

    asscalar(x) = x isa AbstractArray ? only(x) : x

    data = MixedModels.dataset(:pastes)
    m = fit(
        MixedModel,
        @formula(strength ~ 1 + (1 | batch) + (1 | cask)),
        data;
        REML=false,
        progress=false,
    )

    # `savedmodels = 2` asks the search to retain the two best DISTINCT scored models across the
    # whole walk (cAIC4's `numberOfSavedModels` — accumulate every step's candidates, dedup by
    # structure, keep the top-k by cAIC). The crossed model's single backward step scores both
    # `(1|cask)` and `(1|batch)`; the k-best are those two mixed fits, ranked ascending. Provenance:
    # fixture pastes_saved2/savedcaics = c(301.4828311, 314.2642667), both `lmerMod` (the `lm`
    # terminal enters only at k=3 and is excluded — DECISIONS 2026-05-31).
    res = stepcaic(m, data; direction=:backward, savedmodels=2)

    L2_ATOL = 1e-3
    fixture = joinpath(@__DIR__, "fixtures", "stepcaic_driver_level2.h5")
    savedcaics = h5open(fixture, "r") do f
        Float64.(read(f["pastes_saved2"]["savedcaics"]))
    end
    @test length(savedcaics) == 2

    # The saved set is the ranked k-best, ascending, the selected (global best) first.
    @test length(res.saved) == 2
    @test issorted([s.caic for s in res.saved])
    @test res.saved[1].caic == res.selected.caic           # the selected IS the best saved
    @test res.saved[1].caic ≈ savedcaics[1] atol = L2_ATOL  # (1|batch), 301.4828311
    @test res.saved[2].caic ≈ savedcaics[2] atol = L2_ATOL  # (1|cask),  314.2642667

    # The selected model is `(1|batch)` (the lower-cAIC drop), unchanged by the savedmodels request.
    @test extract(res.model) == RESpec([REGroup(:batch, ["(Intercept)"], true)])
end

@testitem "stepcaic backward keeps the incumbent on a Poisson GLMM and threads scoring kwargs (Level-2)" tags = [
    :level2
] begin
    using HDF5
    using MixedModels
    using cAIC: caic, extract, RESpec, REGroup, stepcaic, StepcaicResult

    asscalar(x) = x isa AbstractArray ? only(x) : x

    # Crossed 2-RE Poisson GLMM. Data shared bit-for-bit with cAIC4 via `raw_data` (lme4 and
    # MixedModels.jl do not share datasets); the Julia driver re-fits the SAME columns. cAIC4's
    # decision: backward keeps the FULL model — both random intercepts are supported (dropping
    # `sub` → 666, dropping `it` → 457, both worse than the incumbent 448.21). Provenance: fixture
    # glmm_poisson_keep/finalformula = "y ~ x + (1 | sub) + (1 | it)", finalclass glmerMod.
    fixture = joinpath(@__DIR__, "fixtures", "stepcaic_driver_level2.h5")
    y, x, sub, it, bestCAIC = h5open(fixture, "r") do f
        g = f["glmm_poisson_keep"]
        rd = g["raw_data"]
        (
            Float64.(read(rd["y"])),
            Float64.(read(rd["x"])),
            Int.(read(rd["sub"])),
            Int.(read(rd["it"])),
            asscalar(read(g["bestCAIC"])),
        )
    end
    data = (; y, x, sub, it)
    m = fit(
        MixedModel,
        @formula(y ~ 1 + x + (1 | sub) + (1 | it)),
        data,
        Poisson();
        progress=false,
    )

    res = stepcaic(m, data; direction=:backward)
    @test res isa StepcaicResult
    @test res.model isa GeneralizedLinearMixedModel

    expected = RESpec([
        REGroup(:sub, ["(Intercept)"], true), REGroup(:it, ["(Intercept)"], true)
    ])
    @test extract(res.model) == expected

    # No step accepted ⇒ the selected score is exactly our own caic of the input (the driver scored
    # the incumbent with the same forwarded kwargs and accepted nothing).
    @test res.selected.caic == caic(m).caic

    # … and matches cAIC4's bestCAIC within the GLMM end-to-end Level-2 band (atol=1e-3; the measured
    # lme4↔MixedModels discrepancy on this data is 9.6e-4 — DECISIONS 2026-05-29/30).
    L2_ATOL = 1e-3
    @test res.selected.caic ≈ bestCAIC atol = L2_ATOL

    # The path recorded one scoring round of the two GLMM drops, rejected (incumbent kept).
    @test length(res.path) == 1
    rec = only(res.path)
    @test rec.direction === :backward
    @test rec.accepted == false
    @test length(rec.candidates) == 2
    @test all(c -> c.spec !== nothing, rec.candidates)   # both are GLMM specs, not the glm terminal
    @test rec.incumbentcaic == res.selected.caic

    # Scoring kwargs are forwarded UNCHANGED into every candidate score: `nboot` without
    # `method = :bootstrap` is rejected by the GLMM `caic`, so if the driver threads it the whole
    # search fails loud at the very first score — observable proof the kwarg reaches `caic`.
    @test_throws ArgumentError stepcaic(m, data; direction=:backward, nboot=5)
end

@testitem "stepcaic GLMM backward descends to and scores the Poisson glm terminal (glmm_poisson_terminal, Level-2)" tags = [
    :level2
] begin
    using HDF5
    using MixedModels
    using GLM: glm, Poisson
    using cAIC: caic, extract, RESpec, REGroup, stepcaic, StepcaicResult

    asscalar(x) = x isa AbstractArray ? only(x) : x

    # A SINGLE random-intercept Poisson GLMM `y ~ x + (1 | g)` — the Poisson analogue of the Gaussian
    # `sleepstudy_int` scenario. The only backward neighbour of one random intercept is the no-RE
    # `glm` terminal (§0.1), so the search descends to `glm(y ~ x, Poisson())`, SCORES it, and —
    # because the group effect is supported — REJECTS it (the terminal cAIC is worse), keeping
    # `(1 | g)`. Data shared bit-for-bit with cAIC4 via `raw_data` (lme4 and MixedModels.jl share no
    # datasets); the Julia driver re-fits the SAME columns. Provenance: fixture
    # glmm_poisson_terminal/finalformula = "y ~ x + (1 | g)", finalclass glmerMod.
    fixture = joinpath(@__DIR__, "fixtures", "stepcaic_driver_level2.h5")
    y, x, g, bestCAIC, glmTermCAIC = h5open(fixture, "r") do f
        gr = f["glmm_poisson_terminal"]
        rd = gr["raw_data"]
        (
            Float64.(read(rd["y"])),
            Float64.(read(rd["x"])),
            Int.(read(rd["g"])),
            asscalar(read(gr["bestCAIC"])),
            asscalar(read(gr["glmTermCAIC"])),
        )
    end
    data = (; y, x, g)
    m = fit(MixedModel, @formula(y ~ 1 + x + (1 | g)), data, Poisson(); progress=false)

    res = stepcaic(m, data; direction=:backward)
    @test res isa StepcaicResult
    @test res.model isa GeneralizedLinearMixedModel        # incumbent kept, not the glm terminal

    # cAIC4's decision: keep `(1 | g)` (the terminal is worse). The selected score is exactly our own
    # caic of the input (no step accepted) and matches cAIC4's bestCAIC within the GLMM Level-2 band.
    @test extract(res.model) == RESpec([REGroup(:g, ["(Intercept)"], true)])
    @test res.selected.caic == caic(m).caic
    # Per-scenario Level-2 band (DECISIONS 2026-05-31): the single-grouping 20-level Poisson fit
    # diverges lme4↔MixedModels by a measured 7.57e-3 (relative 1.04e-5) — a pure Laplace-fit/θ̂
    # discrepancy the Chen–Stein df (ρ≈18.04) reads, larger than the crossed-2RE `glmm_poisson_keep`
    # (9.6e-4). The terminal sits ≈117 cAIC units away, so the keep decision is never in doubt.
    L2_ATOL = 1e-2
    @test res.selected.caic ≈ bestCAIC atol = L2_ATOL

    # The terminal WAS reached and scored: one rejected backward step whose SOLE candidate is the
    # `glm` terminal (`spec === nothing`), scored by the same `caic(::TableRegressionModel)` the
    # controller drives. Its score equals the project's own Poisson-glm-terminal caic, and matches
    # cAIC4's glm-terminal cAIC tightly (Poisson has no dispersion σ̂ — no Gaussian σ̂ divergence).
    @test length(res.path) == 1
    rec = only(res.path)
    @test rec.direction === :backward
    @test rec.accepted == false
    @test length(rec.candidates) == 1
    cand = only(rec.candidates)
    @test cand.spec === nothing
    glmterm = caic(glm(@formula(y ~ 1 + x), data, Poisson()))
    @test cand.caic == glmterm.caic
    @test cand.caic ≈ glmTermCAIC atol = 1e-2
    @test cand.caic > res.selected.caic                    # rejected: terminal is worse
    @test cand.dof == glmterm.dof                          # df = rank + 1 = 3
end

@testitem "stepcaic GLMM bootstrap-family search is reproducible under a fixed rng (crossed multi-trial Binomial)" begin
    using MixedModels
    using Random: Xoshiro
    using cAIC: caic, stepcaic, StepcaicResult

    # Multi-trial Binomial has no analytic df (cAIC4's getcondLL is defective for nᵢ > 1; the M3
    # path uses the corrected `condloglik_binomial` — DECISIONS 2026-05-29). Under `method=:auto`
    # the family is unsupported; the user must pass `method=:bootstrap`. The search then scores each
    # candidate with a SINGLE forwarded `rng`, serially — so a whole run is reproducible: two runs
    # from identically-seeded `Xoshiro`s land on bit-identical scores at every candidate of every
    # step (#42 acceptance criterion). A crossed 2-RE model makes the backward step score TWO GLMM
    # drops, so reproducibility is exercised across a sequence of bootstrap scores (not just the
    # incumbent) — the rng state must advance identically through both. Self-contained synthetic
    # data: candidate REFITS are deterministic (GLMM optimisation), only the bootstrap draws consume
    # the rng, so identical seeds ⇒ identical run.
    rng0 = Xoshiro(11)
    ns, ni, nrep = 10, 8, 2
    rows = ns * ni * nrep
    sub = repeat(1:ns, inner=ni * nrep)
    item = repeat(repeat(1:ni, inner=nrep), outer=ns)
    us = 0.6 .* randn(rng0, ns)
    ui = 0.5 .* randn(rng0, ni)
    ntri = rand(rng0, 10:20, rows)
    p = 1 ./ (1 .+ exp.(-(-0.2 .+ us[sub] .+ ui[item])))
    incid = [Float64(sum(rand(rng0, ntri[i]) .< p[i])) for i in 1:rows]   # successes out of nᵢ trials
    data = (; prop=incid ./ ntri, ntri=Float64.(ntri), sub, item)

    m = fit(
        MixedModel,
        @formula(prop ~ 1 + (1 | sub) + (1 | item)),
        data,
        Binomial();
        weights=data.ntri,
        progress=false,
    )
    @test !issingular(m)

    # `method=:auto` on a multi-trial Binomial is unsupported: the incumbent score fails loud at the
    # very first step (the kwargs reach `caic`), so the whole search raises ArgumentError.
    @test_throws ArgumentError stepcaic(m, data; direction=:backward)

    # A comparable signature of a whole run: the selected score plus, per step, every candidate's
    # cAIC, the argmin, and the accept flag. Two identically-seeded runs must produce equal signatures.
    sig(res) = (
        res.selected.caic,
        [
            ([c.caic for c in rec.candidates], rec.bestindex, rec.accepted) for
            rec in res.path
        ],
    )

    run() = stepcaic(
        m, data; direction=:backward, method=:bootstrap, nboot=25, rng=Xoshiro(20240531)
    )
    r1 = run()
    r2 = run()
    @test r1 isa StepcaicResult
    @test length(r1.path[1].candidates) == 2          # both crossed drops bootstrap-scored
    @test sig(r1) == sig(r2)                            # bit-identical across the whole run

    # A different seed perturbs the bootstrap draws — observable proof the forwarded rng actually
    # drives the scoring (the reproducibility above is not a degenerate no-op).
    r3 = stepcaic(
        m, data; direction=:backward, method=:bootstrap, nboot=25, rng=Xoshiro(99)
    )
    @test sig(r3) != sig(r1)
end

@testitem "stepcaic GLMM backward reaches and scores the multi-trial Binomial glm terminal (CBPP)" begin
    using MixedModels
    using GLM: glm, Binomial
    using Random: Xoshiro
    using cAIC: caic, extract, RESpec, REGroup, stepcaic

    # CBPP: a single random-intercept multi-trial Binomial GLMM `incid/hsz ~ period + (1 | herd)`
    # (trial counts `hsz` are the prior weights). Backward's only neighbour is the no-RE `glm`
    # terminal (§0.1) — `glm(incid/hsz ~ period, Binomial(); wts = hsz)`. Reaching it in-search
    # requires the driver to thread the model's prior weights into the terminal fit; without them the
    # terminal is a different (unweighted) model and the corrected `condloglik_binomial` mismatches
    # the trial counts. The terminal is scored by the same `caic(::TableRegressionModel)` the
    # controller drives, so the in-search terminal candidate equals a standalone `caic(glm)` exactly.
    # `method=:bootstrap` is mandatory (multi-trial Binomial has no `:auto` df); the terminal score
    # itself is deterministic (the glm terminal ignores `method`).
    cbpp = MixedModels.dataset(:cbpp)
    m = fit(
        MixedModel,
        @formula(incid / hsz ~ 1 + period + (1 | herd)),
        cbpp,
        Binomial();
        weights=float.(cbpp.hsz),
        progress=false,
    )

    res = stepcaic(
        m, cbpp; direction=:backward, method=:bootstrap, nboot=25, rng=Xoshiro(7)
    )

    # One backward step whose SOLE candidate is the `glm` terminal (`spec === nothing`).
    @test length(res.path) == 1
    rec = only(res.path)
    @test rec.direction === :backward
    @test length(rec.candidates) == 1
    cand = only(rec.candidates)
    @test cand.spec === nothing

    # The terminal was scored WITH the trial counts: its in-search score equals a standalone
    # weighted `glm` terminal scored by the project's own `caic`, to the bit. df = rank + 1.
    terminal = caic(
        glm(@formula(incid / hsz ~ 1 + period), cbpp, Binomial(); wts=float.(cbpp.hsz))
    )
    @test cand.caic == terminal.caic
    @test cand.dof == terminal.dof
    @test isfinite(cand.caic)                          # the corrected condloglik_binomial is finite

    # The group effect is supported, so cAIC4-style the terminal is rejected and `(1 | herd)` kept.
    @test extract(res.model) == RESpec([REGroup(:herd, ["(Intercept)"], true)])
    @test res.model isa GeneralizedLinearMixedModel
    @test cand.caic > res.selected.caic
end

@testitem "stepcaic backward accepts a term drop that lowers the cAIC (Pastes, Level-2)" tags = [
    :level2
] begin
    using HDF5
    using MixedModels
    using cAIC: caic, extract, RESpec, REGroup, stepcaic

    asscalar(x) = x isa AbstractArray ? only(x) : x

    data = MixedModels.dataset(:pastes)
    m = fit(
        MixedModel,
        @formula(strength ~ 1 + (1 | batch) + (1 | cask)),
        data;
        REML=false,
        progress=false,
    )

    res = stepcaic(m, data; direction=:backward)

    # cAIC4's decision: backward drops `cask`, selecting `(1 | batch)` (a singular fit, scored via
    # the reduce-and-refit collapse). Provenance: fixture pastes_crossed/finalformula =
    # "strength ~ (1 | batch)". Well-separated (cask candidate ≈ 314.3 vs batch ≈ 301.5).
    @test extract(res.model) == RESpec([REGroup(:batch, ["(Intercept)"], true)])

    # A real step was accepted: the selected cAIC strictly improves on the input's. The faithful
    # path is TWO steps: step 1 drops `cask` (accepted), step 2 descends to the `lm` terminal
    # (`strength ~ 1`) and REJECTS it (worse), so the search keeps `(1 | batch)`.
    @test res.selected.caic < caic(m).caic
    @test length(res.path) == 2
    @test res.path[1].accepted == true                 # the cask drop
    @test res.path[2].accepted == false                # the lm terminal, rejected
    @test only(res.path[2].candidates).spec === nothing
    @test res.model isa LinearMixedModel               # incumbent kept, not the lm terminal

    L2_ATOL = 1e-3
    fixture = joinpath(@__DIR__, "fixtures", "stepcaic_driver_level2.h5")
    bestCAIC = h5open(fixture, "r") do f
        asscalar(read(f["pastes_crossed"]["bestCAIC"]))
    end
    @test res.selected.caic ≈ bestCAIC atol = L2_ATOL
end

@testitem "stepcaic path records each candidate's effective df ρ (Pastes, Level-2)" tags = [
    :level2
] begin
    using MixedModels
    using GLM: lm
    using cAIC: caic, stepcaic

    # Cycle 6 enrichment: every scored candidate in the path carries its effective degrees of
    # freedom ρ (`dof`) alongside its cAIC — the `df` column `cAIC4`'s `stepcAIC` trace/`aicTab`
    # prints per candidate. The driver already computes it (the candidate's `CAICResult.dof`); the
    # record now exposes it. Anchored to already-public quantities, so no re-fit and refactor-proof.
    data = MixedModels.dataset(:pastes)
    m = fit(
        MixedModel,
        @formula(strength ~ 1 + (1 | batch) + (1 | cask)),
        data;
        REML=false,
        progress=false,
    )

    res = stepcaic(m, data; direction=:backward)

    # Every scored candidate exposes a positive, correctly-typed ρ.
    for rec in res.path
        for cand in rec.candidates
            @test cand.dof isa Float64
            @test cand.dof > 0
        end
    end

    # Step 1 accepts the `cask`-drop: its best candidate IS (becomes) the selected model `(1|batch)`,
    # so its recorded ρ must equal the selected model's ρ — the enrichment is the same number the
    # search acted on, not a re-derivation.
    step1 = res.path[1]
    @test step1.accepted == true
    @test step1.candidates[step1.bestindex].dof == res.selected.dof

    # Step 2's sole candidate is the `lm` terminal (`strength ~ 1`); its recorded ρ equals the
    # project's own `caic(lm)` penalty (rank + 1 = 2 for an intercept-only OLS fit).
    termcand = only(res.path[2].candidates)
    @test termcand.spec === nothing
    @test termcand.dof == caic(lm(@formula(strength ~ 1), data)).dof
    @test termcand.dof == 2
end

@testitem "stepcaic skipnonconverged is inert when every candidate converges (Pastes)" begin
    using MixedModels
    using cAIC: caic, extract, RESpec, REGroup, stepcaic

    data = MixedModels.dataset(:pastes)
    m = fit(
        MixedModel,
        @formula(strength ~ 1 + (1 | batch) + (1 | cask)),
        data;
        REML=false,
        progress=false,
    )

    # All backward candidates of the crossed model converge, so dropping non-converged ones is a
    # no-op: the `skipnonconverged = true` run must match the default `false` run exactly.
    base = stepcaic(m, data; direction=:backward, savedmodels=2)
    skip = stepcaic(m, data; direction=:backward, savedmodels=2, skipnonconverged=true)

    @test extract(skip.model) == extract(base.model)
    @test skip.selected.caic == base.selected.caic
    @test length(skip.path) == length(base.path)
    @test [s.caic for s in skip.saved] == [s.caic for s in base.saved]
    @test skip.options.skipnonconverged === true
    @test base.options.skipnonconverged === false
end

@testitem "stepcaic skipnonconverged excludes a non-converged candidate (selection + saved)" begin
    using MixedModels
    using GLM: lm, StatsModels
    using cAIC: caic, extract, render, RESpec, REGroup, backwardcandidates, StepcaicOptions
    # The driver is exercised directly: no public `stepcaic` knob can deterministically force a
    # *candidate refit* to report non-convergence, so the test injects a refit closure that taints
    # one candidate's optimizer return code. Everything else (greedy walk, `converged`, scoring) is
    # the real machinery — this isolates the `skipnonconverged` exclusion branch.
    using cAIC: _runstepcaic, MMInternals

    data = MixedModels.dataset(:pastes)
    m = fit(
        MixedModel,
        @formula(strength ~ 1 + (1 | batch) + (1 | cask)),
        data;
        REML=false,
        progress=false,
    )

    batchspec = RESpec([REGroup(:batch, ["(Intercept)"], true)])
    BATCH_CAIC = 301.4828311   # the lower-cAIC drop — the global minimum, would win if included

    fixed = MMInternals.fixedterm(m)
    lhs = MMInternals.responseterm(m)
    score(model) = caic(model)
    # Refit each candidate normally, but force the `(1|batch)` drop to look non-converged.
    function tainted_refit(c)
        cm = fit(MixedModel, render(c, fixed, lhs), data; REML=false, progress=false)
        extract(cm) == batchspec && (cm.optsum.returnvalue = :MAXEVAL_REACHED)
        return cm
    end
    terminalfit() = (tm=lm(StatsModels.FormulaTerm(lhs, fixed), data); (tm, caic(tm)))
    gencands(spec, _) = backwardcandidates(
        spec; keep=nothing, selectcorrelation=false, allownointercept=false
    )

    # savedmodels = 0 keeps all distinct scored models; the rest are a plain backward run.
    opts(skip) =
        StepcaicOptions(:backward, false, false, 50, 0, skip, Symbol[], Symbol[], 2, false)

    noskip = _runstepcaic(
        Float64, m, score, tainted_refit, terminalfit, gencands, opts(false); keep=nothing
    )
    skip = _runstepcaic(
        Float64, m, score, tainted_refit, terminalfit, gencands, opts(true); keep=nothing
    )

    # Without the flag the non-converged `(1|batch)` is still scored and, being the global minimum,
    # is selected.
    @test extract(noskip.model) == batchspec
    @test noskip.selected.caic ≈ BATCH_CAIC atol = 1e-3
    @test any(s -> isapprox(s.caic, BATCH_CAIC; atol=1e-3), noskip.saved)

    # With the flag the non-converged `(1|batch)` is excluded from the comparison: it never wins, so
    # the global-minimum drop is passed over. The next-best drop `(1|cask)` does not beat the crossed
    # incumbent, so the search keeps the incumbent — and `(1|batch)` is absent from the saved k-best.
    @test extract(skip.model) == extract(m)
    @test skip.selected.caic == caic(m).caic
    @test !any(s -> isapprox(s.caic, BATCH_CAIC; atol=1e-3), skip.saved)

    # In the first step its effective cAIC is +Inf (the `NA` analogue the greedy rule never picks).
    batchcand = only(filter(c -> c.spec == batchspec, skip.path[1].candidates))
    @test batchcand.caic == typemax(Float64)
end
