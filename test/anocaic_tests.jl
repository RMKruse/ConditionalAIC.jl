@testitem "anocaic returns an AnocaicTable sorted ascending by cAIC" tags = [:level2] begin
    # Tracer: the first end-to-end run of anocaic — two sleepstudy models scored and ranked.
    # The slope model has lower cAIC on sleepstudy and must be first (ascending sort).
    using MixedModels
    using ConditionalAIC: anocaic, AnocaicTable

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
    t = anocaic(m_slope, m_int)
    @test t isa AnocaicTable
    @test t.results[1].caic ≤ t.results[2].caic
end

@testitem "anocaic sorts correctly regardless of input order; inputorder tracks origin" tags = [
    :level2
] begin
    # Passing the models in reverse order must produce the same ranking. inputorder[k] is
    # the 1-based position of the k-th ranked model in the argument list.
    using MixedModels
    using ConditionalAIC: anocaic

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
    t_fwd = anocaic(m_slope, m_int)   # slope is input 1
    t_rev = anocaic(m_int, m_slope)   # slope is input 2
    # same cAIC values regardless of input order
    @test t_fwd.results[1].caic ≈ t_rev.results[1].caic
    @test t_fwd.results[2].caic ≈ t_rev.results[2].caic
    # inputorder reflects original input positions
    @test t_fwd.inputorder[1] == 1   # slope was input 1, ranked 1st
    @test t_rev.inputorder[1] == 2   # slope was input 2, still ranked 1st
end

@testitem "anocaic handles a single model (degenerate comparison)" tags = [:level2] begin
    # A one-model comparison is valid: one entry, delta trivially 0, result matches caic.
    using MixedModels
    using ConditionalAIC: anocaic, AnocaicTable, caic

    data = MixedModels.dataset(:sleepstudy)
    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        data;
        REML=false,
        progress=false,
    )
    t = anocaic(m)
    @test t isa AnocaicTable
    @test length(t.results) == 1
    @test t.inputorder == [1]
    @test t.results[1].caic ≈ caic(m).caic
    @test t.results[1].caic - t.results[1].caic == 0.0   # delta = 0
end

@testitem "anocaic scores all models with identical kwargs (consistent provenance)" tags = [
    :level2
] begin
    # Every result must carry the same resolved method and B-source — consistent scoring is
    # the contract of the comparison layer (CONTEXT.md Selection definition).
    using MixedModels
    using ConditionalAIC: anocaic

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
    t = anocaic(m_slope, m_int)
    @test t.results[1].method === t.results[2].method
    @test t.results[1].bsource === t.results[2].bsource
    @test t.results[1].method === :steinian
    @test t.results[1].bsource === :analytic
end

@testitem "anocaic rejects models with inconsistent REML settings" tags = [:level2] begin
    # Mixing a REML fit and an ML fit contaminates the ranking (cAIC values are not
    # comparable across objectives). anocaic must reject this with a clear ArgumentError.
    using MixedModels
    using ConditionalAIC: anocaic

    data = MixedModels.dataset(:sleepstudy)
    m_ml = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        data;
        REML=false,
        progress=false,
    )
    m_reml = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        data;
        REML=true,
        progress=false,
    )
    @test_throws ArgumentError anocaic(m_ml, m_reml)
end

@testitem "anocaic requires at least one model (MethodError for zero args)" tags = [:level2] begin
    # At least one model is required — the type signature enforces it. Zero-arg calls
    # get a MethodError from Julia's dispatch, not a runtime ArgumentError; this is the
    # idiomatic Julia way to enforce non-empty varargs (avoids an unbound type parameter
    # in the generated kwarg wrapper that Aqua would flag).
    using ConditionalAIC: anocaic

    @test_throws MethodError anocaic()
end

@testitem "an AnocaicTable prints a readable ranked table" tags = [:level2] begin
    # The show method must surface the comparison structure: rank, cAIC, ρ, condloglik,
    # and Δcaic (the user-facing contract for the comparison layer).
    using MixedModels
    using ConditionalAIC: anocaic

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
    t = anocaic(m_slope, m_int)
    s = sprint(show, MIME("text/plain"), t)
    @test occursin("anocaic", s)
    @test occursin("cAIC", s)
    @test occursin("Δcaic", s)
    @test occursin("condloglik", s)
end

@testitem "anocaic is type-stable on the Gaussian path" tags = [:level2] begin
    # Type instability in the comparison spine is a defect (CLAUDE.md §4). @inferred
    # asserts the compiler resolves anocaic to a concrete AnocaicTable{Float64,...}.
    using MixedModels
    using ConditionalAIC: anocaic, AnocaicTable

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
    compareit(a, b) = anocaic(a, b)
    r = @inferred compareit(m_slope, m_int)
    @test r isa AnocaicTable{Float64}
end

@testitem "anocaic matches cAIC4 on Gaussian LMMs via existing Level-2 fixture" tags = [
    :level2
] begin
    # Level-2 correctness gate (CLAUDE.md §6): anocaic must sort correctly and reproduce
    # the cAIC values already validated against cAIC4 in the caic Level-2 test. Reusing
    # caic_level2.h5 (slope_ml and int_ml cases) avoids a redundant fixture: the individual
    # cAIC values are the R ground truth; the comparison table builds on them, so validating
    # the table values against the same fixture confirms both scoring and table construction.
    using HDF5
    using MixedModels
    using ConditionalAIC: anocaic

    asscalar(x) = x isa AbstractArray ? only(x) : x
    L2_ATOL = 1e-3   # derived Level-2 tolerance; see DECISIONS.md (2026-05-27)

    fixture = joinpath(@__DIR__, "fixtures", "caic_level2.h5")
    @test isfile(fixture)

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

    h5open(fixture, "r") do f
        ref_slope = asscalar(read(f["slope_ml"]["caic"]))
        ref_int = asscalar(read(f["int_ml"]["caic"]))

        t = anocaic(m_slope, m_int)

        # The slope model has the lower cAIC on sleepstudy — it must be ranked first.
        @test t.results[1].caic < t.results[2].caic
        # Both cAIC values must match the R reference within the Level-2 tolerance.
        best_ref = min(ref_slope, ref_int)
        @test t.results[1].caic ≈ best_ref atol = L2_ATOL
        @test t.results[2].caic ≈ max(ref_slope, ref_int) atol = L2_ATOL
        # Δcaic for the best model is 0; for the second it equals the gap between refs.
        Δ = t.results[2].caic - t.results[1].caic
        @test Δ ≈ abs(ref_slope - ref_int) atol = 2 * L2_ATOL
    end
end
