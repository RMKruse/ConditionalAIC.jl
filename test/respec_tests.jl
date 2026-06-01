@testitem "extract reads a correlated slope+intercept RE term into a RESpec" begin
    using ConditionalAIC
    using MixedModels

    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        MixedModels.dataset(:sleepstudy);
        progress=false,
    )

    spec = ConditionalAIC.extract(m)

    @test spec isa ConditionalAIC.RESpec
    @test length(spec.groups) == 1
    g = only(spec.groups)
    @test g.grouping === :subj
    @test g.directions == ["(Intercept)", "days"]
    @test g.correlated === true
end

@testitem "render round-trips a correlated slope+intercept fit (sleepstudy)" begin
    using ConditionalAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)
    m = fit(
        MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), data; progress=false
    )
    spec = ConditionalAIC.extract(m)

    f = ConditionalAIC.render(
        spec,
        ConditionalAIC.MMInternals.fixedterm(m),
        ConditionalAIC.MMInternals.responseterm(m),
    )
    refit = fit(MixedModel, f, data; progress=false)

    @test ConditionalAIC.extract(refit) == spec
end

@testitem "extract/render round-trips crossed intercept-only terms (Pastes)" begin
    using ConditionalAIC
    using MixedModels

    data = MixedModels.dataset(:pastes)
    m = fit(
        MixedModel, @formula(strength ~ 1 + (1 | batch) + (1 | cask)), data; progress=false
    )
    spec = ConditionalAIC.extract(m)

    @test length(spec.groups) == 2
    @test spec.groups[1] == ConditionalAIC.REGroup(:batch, ["(Intercept)"], true)
    @test spec.groups[2] == ConditionalAIC.REGroup(:cask, ["(Intercept)"], true)

    f = ConditionalAIC.render(
        spec,
        ConditionalAIC.MMInternals.fixedterm(m),
        ConditionalAIC.MMInternals.responseterm(m),
    )
    refit = fit(MixedModel, f, data; progress=false)
    @test ConditionalAIC.extract(refit) == spec
end

@testitem "extract/render preserves an uncorrelated (zerocorr) term" begin
    using ConditionalAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)
    m = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
        data;
        progress=false,
    )
    spec = ConditionalAIC.extract(m)

    @test spec.groups == [ConditionalAIC.REGroup(:subj, ["(Intercept)", "days"], false)]

    f = ConditionalAIC.render(
        spec,
        ConditionalAIC.MMInternals.fixedterm(m),
        ConditionalAIC.MMInternals.responseterm(m),
    )
    refit = fit(MixedModel, f, data; progress=false)
    @test ConditionalAIC.extract(refit) == spec
    @test only(ConditionalAIC.extract(refit).groups).correlated === false
end

@testitem "extract/render handles a suppressed intercept (0 + slope | g)" begin
    using ConditionalAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)
    m = fit(
        MixedModel, @formula(reaction ~ 1 + days + (0 + days | subj)), data; progress=false
    )
    spec = ConditionalAIC.extract(m)

    @test spec.groups == [ConditionalAIC.REGroup(:subj, ["days"], true)]
    @test !("(Intercept)" in only(spec.groups).directions)

    f = ConditionalAIC.render(
        spec,
        ConditionalAIC.MMInternals.fixedterm(m),
        ConditionalAIC.MMInternals.responseterm(m),
    )
    refit = fit(MixedModel, f, data; progress=false)
    @test ConditionalAIC.extract(refit) == spec
end

@testitem "extract is type-stable; render rejects an empty spec" begin
    using ConditionalAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)
    m = fit(
        MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), data; progress=false
    )

    @inferred ConditionalAIC.RESpec ConditionalAIC.extract(m)

    empty = ConditionalAIC.RESpec(ConditionalAIC.REGroup[])
    @test_throws ArgumentError ConditionalAIC.render(
        empty,
        ConditionalAIC.MMInternals.fixedterm(m),
        ConditionalAIC.MMInternals.responseterm(m),
    )
end

@testitem "extractkeep parses a keep formula fragment into a RESpec floor" begin
    using ConditionalAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)

    # Correlated slope+intercept; the response and any fixed terms are ignored.
    spec = ConditionalAIC.extractkeep(
        @formula(reaction ~ 1 + days + (1 + days | subj)), data
    )
    @test spec isa ConditionalAIC.RESpec
    g = only(spec.groups)
    @test g.grouping === :subj
    @test g.directions == ["(Intercept)", "days"]
    @test g.correlated === true

    # An uncorrelated (`zerocorr`) bar parses to a `correlated = false` group.
    specz = ConditionalAIC.extractkeep(@formula(reaction ~ zerocorr(1 + days | subj)), data)
    @test only(specz.groups).correlated === false
end

@testitem "extractkeep parses a crossed keep fragment, pinning only the named groupings" begin
    using ConditionalAIC
    using MixedModels

    data = MixedModels.dataset(:pastes)
    spec = ConditionalAIC.extractkeep(@formula(strength ~ (1 | batch)), data)
    @test length(spec.groups) == 1
    @test only(spec.groups).grouping === :batch
    @test only(spec.groups).directions == ["(Intercept)"]
end

@testitem "extractkeep rejects a keep formula with no random-effects term" begin
    using ConditionalAIC
    using MixedModels

    data = MixedModels.dataset(:sleepstudy)
    @test_throws ArgumentError ConditionalAIC.extractkeep(
        @formula(reaction ~ 1 + days), data
    )
end
