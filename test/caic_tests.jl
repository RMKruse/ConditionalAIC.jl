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

@testitem "caic bootstrap: returns a coherent CAICResult (tracer)" tags = [
    :level2, :bootstrap
] begin
    # Tracer bullet for the :bootstrap path (issue #12). Confirms the full spine runs
    # end-to-end: a seeded, low-B run must return a CAICResult satisfying the cAIC identity
    # cAIC = −2ℓ + 2ρ. No convergence assertion here — that is Cycle 8.
    using MixedModels, Random
    using cAIC: caic, CAICResult

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    r = caic(m; method=:bootstrap, nboot=50, rng=Xoshiro(42))
    @test r isa CAICResult
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof
end

@testitem "caic bootstrap: records method=:bootstrap and bsource=:na in result" tags = [
    :level2, :bootstrap
] begin
    # Provenance check (issue #12): the bootstrap path uses no Hessian B, so bsource must
    # be :na (not applicable) and method must be :bootstrap.
    using MixedModels, Random
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    r = caic(m; method=:bootstrap, nboot=50, rng=Xoshiro(42))
    @test r.method === :bootstrap
    @test r.bsource === :na
end

@testitem "caic bootstrap: seeded RNG is reproducible and unseeded varies" tags = [
    :level2, :bootstrap
] begin
    # Reproducibility contract: two calls with the same seed must produce bit-identical dof;
    # two calls on the default (global) RNG must produce different dof with overwhelming
    # probability (probability 1 − 2^{−52} for any nboot ≥ 1).
    using MixedModels, Random
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    r1 = caic(m; method=:bootstrap, nboot=50, rng=Xoshiro(42))
    r2 = caic(m; method=:bootstrap, nboot=50, rng=Xoshiro(42))
    @test r1.dof == r2.dof

    r3 = caic(m; method=:bootstrap, nboot=50)
    r4 = caic(m; method=:bootstrap, nboot=50)
    @test r3.dof != r4.dof
end

@testitem "caic bootstrap: type-stable via @inferred" tags = [:level2, :bootstrap] begin
    using MixedModels, Random
    using cAIC: caic, CAICResult

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    @test (@inferred caic(m; method=:bootstrap, nboot=20, rng=Xoshiro(1))) isa
        CAICResult{Float64,LinearMixedModel{Float64}}
end

@testitem "caic bootstrap: does not mutate the original fit" tags = [:level2, :bootstrap] begin
    # Mutation contract: bootstrap refits are on fresh models; the original θ̂ must be
    # untouched after caic returns (same contract as the :finitediff B-source).
    using MixedModels, Random
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    theta_before = copy(m.theta)
    mu_before = copy(fitted(m))
    caic(m; method=:bootstrap, nboot=50, rng=Xoshiro(7))
    @test m.theta == theta_before
    @test fitted(m) == mu_before
end

@testitem "caic bootstrap: default nboot=500 is used when nboot is not supplied" tags = [
    :level2, :bootstrap
] begin
    # When method=:bootstrap is used without nboot, the default (500 draws, matching
    # cAIC4) must be used rather than erroring. The result must satisfy the cAIC identity.
    using MixedModels, Random
    using cAIC: caic, CAICResult

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    r = caic(m; method=:bootstrap, rng=Xoshiro(3))
    @test r isa CAICResult
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof
end

@testitem "caic bootstrap: converges to analytic df with large nboot" tags = [
    :level2, :bootstrap
] begin
    # Convergence gate (issue #12, DECISIONS.md): with nboot=2000 the bootstrap df must
    # agree with the analytic Greven-Kneib df to within atol=2.0. The tolerance is derived
    # from the MC standard error: for the sleepstudy random-intercept model ρ_analytic ≈ 19,
    # and Monte Carlo noise at B=2000 is ~O(0.5), so atol=2.0 is a 4σ band (see DECISIONS.md).
    # Memory: do NOT tighten this tolerance; the bootstrap does not converge to analytic df
    # (see bootstrap-not-equal-analytic.md).
    using MixedModels, Random
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    r_analytic = caic(m)
    r_boot = caic(m; method=:bootstrap, nboot=2000, rng=Xoshiro(99))
    @test isapprox(r_boot.dof, r_analytic.dof; atol=2.0)
end

@testitem "caic scores the numeric B-sources coherently and records their provenance" tags = [
    :level2
] begin
    # The numeric B-sources run end-to-end through the same spine as `:analytic`, differing
    # only in how B is obtained (`bhessian` → `dof_lmm_numeric`). Each must return a coherent
    # `CAICResult` (cAIC = −2ℓ + 2ρ), record the B-source it actually ran, share the
    # conditional log-likelihood with the analytic path (ℓ does not depend on B), and — the
    # mutation contract — leave the fit at its fitted θ̂ (the self-driven FD path perturbs and
    # restores; a leftover perturbation would poison the score).
    using MixedModels
    using cAIC: caic, CAICResult

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    θ̂ = copy(m.θ)
    r_an = caic(m; hessian=:analytic)

    for src in (:finitediff, :forwarddiff)
        r = caic(m; hessian=src)
        @test r isa CAICResult
        @test r.bsource === src                       # provenance: what actually ran
        @test r.method === :steinian
        @test r.caic ≈ -2 * r.condloglik + 2 * r.dof   # assembly identity
        @test r.condloglik ≈ r_an.condloglik           # ℓ is B-source-independent
        @test m.θ == θ̂                                  # fit left untouched (restore contract)
    end
end

@testitem "the three B-sources are estimators of one ρ: the documented cross-source landscape" tags = [
    :level2
] begin
    # `:analytic`, `:finitediff`, and `:forwarddiff` are three estimators of the *same*
    # Greven–Kneib ρ, not three computations of one number (docs/math/0004 §4). Their pairwise
    # gaps are genuine and recorded, never tolerance-papered (the bootstrap-vs-analytic
    # precedent). This test pins the *structure* of that landscape — the numbers (the measured
    # sleepstudy spread that sets the bounds) live in DECISIONS.md (2026-05-28):
    #   • the two numeric sources cluster — the σ-freezing gap |ρ_ford − ρ_fd| (0004 §3a) is
    #     strictly *smaller* than the closed-form-vs-numeric-Hessian gap |ρ_an − ρ_fd|;
    #   • every gap is a genuine divergence (well above FD/AD noise), yet all three remain
    #     estimators of one ρ — bounded within the recorded same-ρ band;
    #   • the spread grows with the random-effects dimension s (s = 1 tight, s = 3 widest).
    # (`:finitediff ≡ cAIC4 analytic = FALSE` — the *correctness*-tight pair — is the Level-2
    # gate above; here the comparison is purely among the three Julia sources, no R.)
    using MixedModels
    using cAIC: caic

    data = MixedModels.dataset(:sleepstudy)
    ρ(m, src) = caic(m; hessian=src).dof

    # s = 3 correlated slope (the widest spread) and s = 1 random intercept (the tightest).
    m3 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        data;
        REML=false,
        progress=false,
    )
    m1 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj)),
        data;
        REML=false,
        progress=false,
    )
    gaps(m) = (
        an_fd=abs(ρ(m, :analytic) - ρ(m, :finitediff)),
        ford_fd=abs(ρ(m, :forwarddiff) - ρ(m, :finitediff)),
        an_ford=abs(ρ(m, :analytic) - ρ(m, :forwarddiff)),
    )
    g3, g1 = gaps(m3), gaps(m1)

    # Structure: the numeric pair clusters tighter than the analytic closed form, on both s.
    @test g3.ford_fd < g3.an_fd
    @test g1.ford_fd < g1.an_fd

    # Genuine divergences — each gap is a real inter-estimator gap, not numerical noise
    # (the symmetric-Hessian checks put FD/AD noise at ~1e-6); the floor is well below the
    # smallest measured genuine gap (the σ-frozen intercept gap, ≈2.5e-3). See DECISIONS.md.
    GENUINE_FLOOR = 1e-3
    @test g3.an_fd > GENUINE_FLOOR
    @test g3.ford_fd > GENUINE_FLOOR
    @test g1.an_fd > GENUINE_FLOOR
    @test g1.ford_fd > GENUINE_FLOOR

    # …yet all three remain estimators of one ρ — every gap is inside the recorded same-ρ
    # band (the ceiling is > the worst measured |Δ|, ≈1.2 on slope_ml). See DECISIONS.md.
    SAME_RHO_CEIL = 1.5
    @test g3.an_fd < SAME_RHO_CEIL
    @test g3.an_ford < SAME_RHO_CEIL
    @test g3.ford_fd < SAME_RHO_CEIL

    # s-dependence: the s = 3 spread strictly exceeds the s = 1 spread, both pairs (0004 §4).
    @test g3.an_fd > g1.an_fd
    @test g3.ford_fd > g1.ford_fd
end

@testitem "caic on Bernoulli GLMM returns CAICResult — M3 general path (issue #31)" tags = [
    :level2, :glmm
] begin
    # M3 general path (issue #31): caic on a non-singular Bernoulli GLMM must return a
    # CAICResult rather than throw. Uses a small synthetic dataset (32 obs) for speed;
    # comprehensive wiring tests are in caic_glmm_tests.jl.
    using MixedModels
    using cAIC: caic, CAICResult

    y = Float64[
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
    ]
    g = repeat(1:4, inner=8)
    gm = fit(MixedModel, @formula(y ~ 1 + (1 | g)), (; y, g), Bernoulli(); progress=false)
    @test gm isa GeneralizedLinearMixedModel
    @test !issingular(gm)

    r = caic(gm)
    @test r isa CAICResult{Float64,<:GeneralizedLinearMixedModel}
    @test r.method == :auto
    @test r.bsource == :na
    @test r.caic ≈ -2 * r.condloglik + 2 * r.dof rtol = 1e-12
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

@testitem "caic is type-stable on the numeric B-source paths" tags = [:level2] begin
    # Type stability is a defect gate (CLAUDE §4) and must hold on the numeric paths too, not
    # only the default `:analytic` one. The `:finitediff`/`:forwarddiff` branch routes through
    # `bhessian` (→ `Matrix{T}`) and `dof_lmm_numeric` (→ `T`); `@inferred` asserts `caic`
    # still resolves to a concrete `CAICResult{Float64,…}` with no Any/Union fallback. A
    # per-source helper with no kwargs gives `@inferred` a clean call to infer.
    using MixedModels
    using cAIC: caic, CAICResult

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )
    scorefd(model) = caic(model; hessian=:finitediff)
    scoreford(model) = caic(model; hessian=:forwarddiff)
    @test (@inferred scorefd(m)) isa CAICResult{Float64}
    @test (@inferred scoreford(m)) isa CAICResult{Float64}
end

@testitem "caic dispatches on the fit's REML flag (objective dispatch, no force-refit)" tags = [
    :level2
] begin
    # The defining behaviour of #9: `caic` scores the fit under the objective it was
    # estimated with, read from `m.optsum.REML`, and never force-refits to ML. Comparing two
    # *separate* ML and REML fits would conflate the flag with their differing θ̂; flipping
    # only the flag on a *single* fit isolates the dispatch — every component (e, A, V₀⁻¹,
    # Wⱼ, …) is built from the frozen θ̂, so the score may move only through the objective
    # branch (ρ's nθ = n vs n−p, and σ̂'s denominator in the conditional log-lik). Both must
    # respond; a regression that ignored the flag (e.g. always ML) would leave them identical.
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=false,
        progress=false,
    )

    θ̂ = copy(m.θ)
    r_ml = caic(m)                 # scored as the ML fit it is
    m.optsum.REML = true           # flip ONLY the objective flag — do not refit
    r_reml = caic(m)
    m.optsum.REML = false          # restore before asserting (a failed @test won't skip this)

    @test m.θ == θ̂                 # θ̂ frozen across the flip: the flag alone moved
    @test r_reml.dof != r_ml.dof   # the Greven–Kneib branch (nθ) responds to the flag
    @test r_reml.caic != r_ml.caic # end-to-end the conditional AIC responds to the flag
end

@testitem "caic computes on the fit as-is: scoring leaves the fit unmutated" tags = [
    :level2
] begin
    # The "no force-refit" guarantee (#9, DECISIONS 2026-05-27): `caic` reads the fitted
    # quantities and must not refit or otherwise mutate the model — in particular it must not
    # flip a REML fit to ML to score it. Observable contract: the REML flag and θ̂ are
    # byte-identical before and after scoring.
    using MixedModels
    using cAIC: caic

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=true,
        progress=false,
    )

    flag_before, θ̂_before = m.optsum.REML, copy(m.θ)
    caic(m)
    @test m.optsum.REML == flag_before    # objective not forced to ML
    @test m.θ == θ̂_before                 # fit not re-optimised
end

@testitem "caic is type-stable on the REML path" tags = [:level2] begin
    # Type stability is a defect gate (CLAUDE §4) and must hold on both objective branches,
    # not only the ML one exercised above. `@inferred` asserts `caic` on a REML fit still
    # resolves to a concrete `CAICResult{Float64,…}` with no Any/Union fallback from the
    # `isREML` dispatch.
    using MixedModels
    using cAIC: caic, CAICResult

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        REML=true,
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

@testitem "caic(:finitediff) matches cAIC4 analytic=FALSE end-to-end (Level-2)" tags = [
    :level2
] begin
    # The correctness gate for the self-driven finite-difference B-source (#11). `:finitediff`
    # differentiates the *profiled* deviance — the same object `lme4`'s optimiser differentiates
    # for `m@optinfo$derivs$Hessian` — so `caic(m; hessian=:finitediff)` must reproduce the ρ
    # and cAIC that `cAIC4::cAIC(fit, analytic = FALSE)` returns (fixture keys `df_numeric`,
    # `caic_numeric`, written by `generate_fixtures_level2.R`).
    #
    # The agreement band is the *same* fit-discrepancy-derived `L2_ATOL = 1e-3` the analytic
    # Level-2 gate uses (DECISIONS.md 2026-05-27): the measured worst |Δρ| is 1.37e-4 (slope_ml,
    # s = 3; FD accuracy + the lme4↔MixedModels θ̂ discrepancy), with the intercept cases agreeing
    # to ~1e-7. This is the *correctness*-tight pair of docs/math/0004 §4; the σ-frozen
    # `:forwarddiff` source is deliberately **not** compared here (it diverges — see DECISIONS).
    using HDF5
    using MixedModels
    using cAIC: caic

    asscalar(x) = x isa AbstractArray ? only(x) : x

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
            haskey(g, "df_numeric") || error(
                "fixture case `$name` has no df_numeric — run generate_fixtures_level2.R",
            )

            m = fit(MixedModel, form, data; REML=reml, progress=false)
            r = caic(m; hessian=:finitediff)

            @test r.dof ≈ asscalar(read(g["df_numeric"])) atol = L2_ATOL
            @test r.caic ≈ asscalar(read(g["caic_numeric"])) atol = L2_ATOL
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
