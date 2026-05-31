@testitem "extract reads a correlated slope+intercept RE term into a RESpec" begin
    using cAIC
    using MixedModels

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        progress=false,
    )

    spec = cAIC.extract(m)

    @test spec isa cAIC.RESpec
    @test length(spec.groups) == 1
    g = only(spec.groups)
    @test g.grouping === :subj
    @test g.directions == ["(Intercept)", "days"]
    @test g.correlated === true
end

@testitem "render round-trips a correlated slope+intercept fit (sleepstudy)" begin
    using cAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)
    m = fit(
        MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), data; progress=false
    )
    spec = cAIC.extract(m)

    f = cAIC.render(spec, cAIC.MMInternals.fixedterm(m), cAIC.MMInternals.responseterm(m))
    refit = fit(MixedModel, f, data; progress=false)

    @test cAIC.extract(refit) == spec
end

@testitem "extract/render round-trips crossed intercept-only terms (Pastes)" begin
    using cAIC
    using MixedModels

    data = MixedModels.dataset(:pastes)
    m = fit(
        MixedModel, @formula(strength ~ 1 + (1 | batch) + (1 | cask)), data; progress=false
    )
    spec = cAIC.extract(m)

    @test length(spec.groups) == 2
    @test spec.groups[1] == cAIC.REGroup(:batch, ["(Intercept)"], true)
    @test spec.groups[2] == cAIC.REGroup(:cask, ["(Intercept)"], true)

    f = cAIC.render(spec, cAIC.MMInternals.fixedterm(m), cAIC.MMInternals.responseterm(m))
    refit = fit(MixedModel, f, data; progress=false)
    @test cAIC.extract(refit) == spec
end

@testitem "extract/render preserves an uncorrelated (zerocorr) term" begin
    using cAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)
    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
        data;
        progress=false,
    )
    spec = cAIC.extract(m)

    @test spec.groups == [cAIC.REGroup(:subj, ["(Intercept)", "days"], false)]

    f = cAIC.render(spec, cAIC.MMInternals.fixedterm(m), cAIC.MMInternals.responseterm(m))
    refit = fit(MixedModel, f, data; progress=false)
    @test cAIC.extract(refit) == spec
    @test only(cAIC.extract(refit).groups).correlated === false
end

@testitem "extract/render handles a suppressed intercept (0 + slope | g)" begin
    using cAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)
    m = fit(
        MixedModel, @formula(reaction ~ 1 + days + (0 + days | subj)), data; progress=false
    )
    spec = cAIC.extract(m)

    @test spec.groups == [cAIC.REGroup(:subj, ["days"], true)]
    @test !("(Intercept)" in only(spec.groups).directions)

    f = cAIC.render(spec, cAIC.MMInternals.fixedterm(m), cAIC.MMInternals.responseterm(m))
    refit = fit(MixedModel, f, data; progress=false)
    @test cAIC.extract(refit) == spec
end

@testitem "extract is type-stable; render rejects an empty spec" begin
    using cAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)
    m = fit(
        MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), data; progress=false
    )

    @inferred cAIC.RESpec cAIC.extract(m)

    empty = cAIC.RESpec(cAIC.REGroup[])
    @test_throws ArgumentError cAIC.render(
        empty, cAIC.MMInternals.fixedterm(m), cAIC.MMInternals.responseterm(m)
    )
end

@testitem "extractkeep parses a keep formula fragment into a RESpec floor" begin
    using cAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)

    # Correlated slope+intercept; the response and any fixed terms are ignored.
    spec = cAIC.extractkeep(@formula(reaction ~ 1 + days + (1 + days | subj)), data)
    @test spec isa cAIC.RESpec
    g = only(spec.groups)
    @test g.grouping === :subj
    @test g.directions == ["(Intercept)", "days"]
    @test g.correlated === true

    # An uncorrelated (`zerocorr`) bar parses to a `correlated = false` group.
    specz = cAIC.extractkeep(@formula(reaction ~ zerocorr(1 + days | subj)), data)
    @test only(specz.groups).correlated === false
end

@testitem "extractkeep parses a crossed keep fragment, pinning only the named groupings" begin
    using cAIC
    using MixedModels

    data = MixedModels.dataset(:pastes)
    spec = cAIC.extractkeep(@formula(strength ~ (1 | batch)), data)
    @test length(spec.groups) == 1
    @test only(spec.groups).grouping === :batch
    @test only(spec.groups).directions == ["(Intercept)"]
end

@testitem "extractkeep rejects a keep formula with no random-effects term" begin
    using cAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)
    @test_throws ArgumentError cAIC.extractkeep(@formula(reaction ~ 1 + days), data)
end
