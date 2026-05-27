@testitem "caic scores a Gaussian LMM end-to-end: returns a coherent CAICResult" tags = [
    :level2
] begin
    # Tracer: the first time the whole spine runs end-to-end on a real fit — the bridge
    # (fit → components), `dof_lmm` (ρ), and `condloglik` (ℓ) assembled into a
    # `CAICResult`. The behaviour asserted is the assembly identity cAIC = −2ℓ + 2ρ
    # (doc 0002 §1); a wiring break anywhere along the spine breaks it.
    using MixedModels
    using cAIC: caic, CAICResult

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    r = caic(m)
    @test r isa CAICResult
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof
end

@testitem "caic's effective df ρ is at least the naive plug-in ρ₀ = tr(H₁)" tags = [:level2] begin
    # The Greven–Kneib correction accounts for the estimation of θ; it raises the naive
    # plug-in ρ₀ = tr(H₁) and never lowers it (doc 0002 §5, ρ ≥ ρ₀). `MixedModels`' own
    # `leverage(m)` gives ρ₀ = Σᵢ Hᵢᵢ via triangular solves against the fit's Cholesky —
    # an independent path from `Components`' Woodbury `V₀⁻¹`/`A` build, so this also
    # cross-checks that the bridge reproduces the hat-matrix trace.
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    r = caic(m)
    ρ0 = sum(MixedModels.leverage(m))
    @test r.dof ≥ ρ0
end

@testitem "caic records the method and B-source it actually ran (provenance)" tags = [
    :level2
] begin
    # The result must say what was *resolved and run*, not what was asked: for the Gaussian
    # family `method = :auto` resolves to the analytic Greven–Kneib correction (`:steinian`)
    # and the default B-source is the closed-form `:analytic`. This provenance lets candidate
    # models be checked for consistent scoring downstream.
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    r = caic(m)                 # method defaults to :auto
    @test r.method === :steinian
    @test r.bsource === :analytic
    # explicitly asking for :steinian records the same thing
    @test caic(m; method=:steinian).method === :steinian
end

@testitem "caic carries sigmapenalty through to ρ (one estimated σ²)" tags = [:level2] begin
    # `sigmapenalty` is the count of estimated residual-variance parameters added to ρ
    # (doc 0002 §4, the additive `+ sigmapenalty`). cAIC4's default is 1 (one estimated σ²);
    # dropping it to 0 (known error variance) must lower ρ — and hence the cAIC penalty 2ρ —
    # by exactly 1.
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    r1 = caic(m)                       # sigmapenalty = 1 (default)
    r0 = caic(m; sigmapenalty=0)
    @test r1.dof - r0.dof ≈ 1
    @test r1.caic - r0.caic ≈ 2        # cAIC = −2ℓ + 2ρ, ℓ unchanged
end

@testitem "a CAICResult prints a readable human summary" tags = [:level2] begin
    # The result is a user-facing return value; its text/plain rendering must surface the
    # headline cAIC and the scoring provenance rather than the raw struct dump.
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    r = caic(m)
    s = sprint(show, MIME("text/plain"), r)
    @test occursin("cAIC", s)
    @test occursin(string(r.method), s)       # provenance is visible
    @test occursin(string(r.bsource), s)
end

@testitem "caic rejects invalid options with ArgumentError" tags = [:level2] begin
    # Invalid input fails loudly with the documented exception type (CLAUDE §4) — never a
    # silently-wrong number. Covers the unknown enum values, a negative penalty count, and
    # the two `nboot` misuses (supplied off the bootstrap path; non-positive on it).
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    @test_throws ArgumentError caic(m; method=:nonsense)
    @test_throws ArgumentError caic(m; hessian=:nonsense)
    @test_throws ArgumentError caic(m; sigmapenalty=-1)
    @test_throws ArgumentError caic(m; nboot=10)                      # nboot off :bootstrap
    @test_throws ArgumentError caic(m; method=:bootstrap, nboot=0)    # non-positive nboot
end

@testitem "caic accepts but defers the unimplemented method/B-source paths" tags = [:level2] begin
    # `:bootstrap`, `:forwarddiff`, `:finitediff` are valid (they parse and validate) but
    # land on estimators not delivered in #8. They must fail with a clear "not yet
    # implemented" message — never silently fall back to the analytic path.
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    @test_throws "not yet implemented" caic(m; method=:bootstrap)
    @test_throws "not yet implemented" caic(m; hessian=:forwarddiff)
    @test_throws "not yet implemented" caic(m; hessian=:finitediff)
end

@testitem "caic rejects a non-Gaussian (GLMM) fit — M3 scope" tags = [:level2] begin
    # The Gaussian bias correction (M2) does not apply to a generalised mixed model; until
    # the GLMM path (M3) lands, `caic` on a `GeneralizedLinearMixedModel` must raise a
    # typed error rather than mis-score it through the Gaussian spine.
    using MixedModels
    using cAIC: caic

    gm = fit(
        MixedModel,
        @formula(r2 ~ 1 + anger + gender + (1 | subj) + (1 | item)),
        MixedModels.dataset(:verbagg),
        Bernoulli();
        progress=false,
    )
    @test gm isa GeneralizedLinearMixedModel
    @test_throws ArgumentError caic(gm)
end

@testitem "caic is type-stable on the Gaussian path" tags = [:level2] begin
    # Type instability in the numerical spine is a defect (CLAUDE §4); `@inferred` asserts
    # the compiler resolves `caic` to a concrete `CAICResult{Float64,…}` with no Any/Union
    # fallback. A helper with no kwargs gives `@inferred` a clean call to infer.
    using MixedModels
    using cAIC: caic, CAICResult

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    scoreit(model) = caic(model)
    r = @inferred scoreit(m)
    @test r isa CAICResult{Float64}
end

@testitem "caic matches cAIC4 end-to-end on Gaussian LMMs (Level-2)" tags = [:level2] begin
    # The correctness gate (CLAUDE §6 Level-2): fit the same models `MixedModels.jl` and
    # `lme4` agree on, and reproduce the conditional AIC that `cAIC4`'s public `cAIC()`
    # returns. The reference caic/df/cll are read straight from the committed fixture
    # (written by `generate_fixtures_level2.R`); no R runs here.
    #
    # `lme4` and `MixedModels.jl` do not produce bit-identical fits — the two optimizers
    # settle at slightly different θ̂ on a near-flat marginal objective — so the comparison
    # uses the fit-discrepancy-derived tolerance recorded in DECISIONS.md (2026-05-27):
    # atol = 1e-3, ≈3× the worst observed |Δcaic| (2.96e-4 on slope_ml, driven by a θ̂
    # discrepancy of 4e-5; the intercept cases, where θ̂ matches to ~1e-9, agree to ~1e-8).
    # A genuine machinery error shifts caic by ≥ O(0.1) — well outside this band.
    using HDF5
    using MixedModels
    using cAIC: caic

    # rhdf5 stores an R length-1 numeric as a 1-element array; coerce before comparing.
    asscalar(x) = x isa AbstractArray ? only(x) : x

    # Mirror of test/generate_fixtures_level2.R's `cases` (the sleepstudy column rename
    # subj/days/reaction ↔ Subject/Days/Reaction is the only cross-package difference).
    specs = Dict(
        "slope_ml" => (@formula(reaction ~ 1 + days + (1 + days | subj)), false),
        "slope_reml" => (@formula(reaction ~ 1 + days + (1 + days | subj)), true),
        "int_ml" => (@formula(reaction ~ 1 + days + (1 | subj)), false),
        "int_reml" => (@formula(reaction ~ 1 + days + (1 | subj)), true),
    )
    L2_ATOL = 1e-3   # derived Level-2 tolerance; see DECISIONS.md (2026-05-27)

    fixture = joinpath(@__DIR__, "fixtures", "caic_level2.h5")
    @test isfile(fixture)

    data = MixedModels.dataset(:sleepstudy)
    h5open(fixture, "r") do f
        cases = filter(!=("meta"), keys(f))
        @test !isempty(cases)
        for name in cases
            haskey(specs, name) || error("fixture case `$name` has no Julia spec")
            form, reml = specs[name]
            g = f[name]
            @test Bool(asscalar(read(g["reml"]))) == reml   # spec/fixture agree on objective

            m = fit(MixedModel, form, data; REML=reml, progress=false)
            r = caic(m)

            @test r.caic ≈ asscalar(read(g["caic"])) atol = L2_ATOL
            @test r.dof ≈ asscalar(read(g["df"])) atol = L2_ATOL
            @test r.condloglik ≈ asscalar(read(g["cll"])) atol = L2_ATOL
        end
    end
end

@testitem "live R re-validation of the Level-2 fixture (gated by CAIC_LIVE_RCALL)" tags = [
    :live_rcall
] begin
    # Fixture-rot guard (CLAUDE §6) for Level-2: regenerate the references with *live* lme4 +
    # cAIC4 and check (a) `caic` still matches the freshly-computed cAIC4 value and (b) the
    # committed reference has not drifted from a fresh regeneration. Skipped in the default
    # (no-R) CI job; enabled by `CAIC_LIVE_RCALL=1` locally and in the scheduled job. Uses
    # the same Rscript + HDF5 hand-off as the generator (no `RCall.jl`).
    using HDF5
    using MixedModels
    using cAIC: caic

    if get(ENV, "CAIC_LIVE_RCALL", "0") == "1"
        asscalar(x) = x isa AbstractArray ? only(x) : x
        here = @__DIR__
        specs = Dict(
            "slope_ml" => (@formula(reaction ~ 1 + days + (1 + days | subj)), false),
            "slope_reml" => (@formula(reaction ~ 1 + days + (1 + days | subj)), true),
            "int_ml" => (@formula(reaction ~ 1 + days + (1 | subj)), false),
            "int_reml" => (@formula(reaction ~ 1 + days + (1 | subj)), true),
        )
        L2_ATOL = 1e-3   # derived Level-2 tolerance; see DECISIONS.md (2026-05-27)

        # Committed reference caic, read before regenerating, for the no-rot check.
        committed = joinpath(here, "fixtures", "caic_level2.h5")
        committed_caic = Dict{String,Float64}()
        h5open(committed, "r") do f
            for name in filter(!=("meta"), keys(f))
                committed_caic[name] = asscalar(read(f[name]["caic"]))
            end
        end

        # Regenerate the references with live R into a temp fixture.
        tmp = joinpath(mktempdir(), "caic_level2.h5")
        run(
            addenv(
                `Rscript $(joinpath(here, "generate_fixtures_level2.R"))`, "FIXTURE" => tmp
            ),
        )

        data = MixedModels.dataset(:sleepstudy)
        h5open(tmp, "r") do f
            for name in filter(!=("meta"), keys(f))
                form, reml = specs[name]
                live_caic = asscalar(read(f[name]["caic"]))
                m = fit(MixedModel, form, data; REML=reml, progress=false)
                @test caic(m).caic ≈ live_caic atol = L2_ATOL              # vs live cAIC4
                # The R reference is a deterministic lmer+cAIC4 recompute, so committed and
                # fresh must agree to machine precision; drift means the fixture rotted.
                @test live_caic ≈ committed_caic[name] rtol = 1e-8 atol = 1e-8  # no rot
            end
        end
    else
        @info "Skipping Level-2 live-RCall re-validation (set CAIC_LIVE_RCALL=1 to enable)"
    end
end
