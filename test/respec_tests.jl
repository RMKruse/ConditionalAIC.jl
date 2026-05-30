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
