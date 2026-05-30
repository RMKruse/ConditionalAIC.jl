# Level-1 set-equality tests for the backward candidate enumerator (`backwardcandidates`, M4 /
# #38). Each `@testitem` reads one scenario from the committed fixture
# `fixtures/stepcaic_backward_level1.h5` (written by `generate_fixtures_stepcaic.R` driving
# cAIC4's real `backwardStep`) and asserts the Julia enumerator is **set-equal** to the cAIC4
# ground truth under the canonical encoding of docs/math/0008 §2.5. No model is fitted — this is
# pure combinatorial structure (§6 Level 1).

# Shared helpers + fixture reader. `parsespec` turns a canonical input string
# `"g/cor=1:(Intercept),days;…"` into the `RESpec` whose neighbours we enumerate — the *same*
# string the R side parses to a `cnms`. `canon` renders a candidate `RESpec` to its canonical
# term-string (an uncorrelated group expands to one single-label term per direction, mirroring
# lme4's repeated-name `cnms`), so the Julia and cAIC4 candidate sets are compared like-for-like.
@testsnippet BackwardFixtures begin
    using cAIC
    using HDF5

    const FIXTURE = joinpath(@__DIR__, "fixtures", "stepcaic_backward_level1.h5")

    function parsespec(s)
        groups = cAIC.REGroup[]
        for grp in split(s, ";")
            meta, dirs = split(grp, ":")
            nm = Symbol(replace(meta, r"/cor=.*" => ""))
            cor = parse(Int, replace(meta, r".*cor=" => "")) == 1
            push!(groups, cAIC.REGroup(nm, String.(split(dirs, ",")), cor))
        end
        return cAIC.RESpec(groups)
    end

    function canon(spec)
        terms = String[]
        for g in spec.groups
            if g.correlated
                push!(terms, string(g.grouping, ":", join(sort(g.directions), ",")))
            else
                for d in g.directions
                    push!(terms, string(g.grouping, ":", d))
                end
            end
        end
        return join(sort(terms), ";")
    end

    expectedset(s) = Set(strip(c) for c in split(s, "\n") if !isempty(strip(c)))

    # Read one scenario group into (spec, keep|nothing, selectcorrelation, allownointercept,
    # expected-set), so a testitem just calls `backwardcandidates` and compares.
    function scenario(name)
        h5open(FIXTURE, "r") do f
            g = f[name]
            spec = parsespec(only(read(g["input"])))
            keep_s = only(read(g["keep"]))
            keep = isempty(keep_s) ? nothing : parsespec(keep_s)
            selcor = only(read(g["selectcorrelation"])) == 1
            noint = only(read(g["allownointercept"])) == 1
            expected = expectedset(only(read(g["expected"])))
            return (; spec, keep, selcor, noint, expected)
        end
    end

    # The enumerator's candidate set under one scenario's flags, in canonical encoding.
    function candidateset(sc)
        cands = cAIC.backwardcandidates(
            sc.spec; keep=sc.keep, selectcorrelation=sc.selcor, allownointercept=sc.noint
        )
        return Set(canon(c) for c in cands)
    end
end

@testitem "drops a direction from a correlated term (sleepstudy default)" setup = [
    BackwardFixtures
] begin
    sc = scenario("sleepstudy_default")
    @test candidateset(sc) == sc.expected            # {(1|subj)}: (days|subj) is intercept-stripped
    @test sc.expected == Set(["subj:(Intercept)"])
end

@testitem "allownointercept keeps the intercept-less neighbour (sleepstudy)" setup = [
    BackwardFixtures
] begin
    sc = scenario("sleepstudy_noint")
    @test candidateset(sc) == sc.expected            # {(1|subj), (days|subj)}
    @test sc.expected == Set(["subj:(Intercept)", "subj:days"])
end

@testitem "selectcorrelation leaves the intercept-only drop unchanged (sleepstudy)" setup = [
    BackwardFixtures
] begin
    sc = scenario("sleepstudy_selcor")
    @test candidateset(sc) == sc.expected            # removeUncor is inert here → still {(1|subj)}
end

@testitem "drops a whole term across crossed intercept-only groups (Pastes)" setup = [
    BackwardFixtures
] begin
    sc = scenario("pastes_default")
    @test candidateset(sc) == sc.expected            # {(1|batch), (1|cask)}
    @test sc.expected == Set(["batch:(Intercept)", "cask:(Intercept)"])
end

@testitem "a single random intercept is the terminal (empty candidate set)" setup = [
    BackwardFixtures
] begin
    sc = scenario("single_default")
    cands = cAIC.backwardcandidates(sc.spec)
    @test isempty(cands)                             # cAIC4's NA terminal → empty Vector{RESpec}
    @test sc.expected == Set{String}()
end

@testitem "keep at the terminal returns the model as its own sole neighbour" setup = [
    BackwardFixtures
] begin
    sc = scenario("single_keep")
    @test candidateset(sc) == sc.expected            # keep ~(1|g) → {(1|g)}
    @test sc.expected == Set(["g:(Intercept)"])
end

@testitem "enumerates drops of a three-direction correlated term" setup = [BackwardFixtures] begin
    sc = scenario("three_default")
    @test candidateset(sc) == sc.expected            # {(1+x|g), (1+y|g)}; (x+y|g) intercept-stripped
    @test sc.expected == Set(["g:(Intercept),x", "g:(Intercept),y"])
end

@testitem "allownointercept retains the intercept-less three-direction drop" setup = [
    BackwardFixtures
] begin
    sc = scenario("three_noint")
    @test candidateset(sc) == sc.expected            # adds (x+y|g)
    @test "g:x,y" in sc.expected
end

@testitem "an uncorrelated split degenerates to the terminal by default" setup = [
    BackwardFixtures
] begin
    # (1|g)+(0+x|g): the faithful [-i] degeneracy drops the intercept copy to empty (notempty)
    # and removeNoInt strips the surviving (x|g) → empty set.
    sc = scenario("uncor_default")
    @test candidateset(sc) == sc.expected
    @test sc.expected == Set{String}()
end

@testitem "the uncorrelated split survives under selectcorrelation + allownointercept" setup = [
    BackwardFixtures
] begin
    sc = scenario("uncor_selcor_noint")
    @test candidateset(sc) == sc.expected            # {(x|g)}
    @test sc.expected == Set(["g:x"])
end

@testitem "mixed correlated + crossed term enumerates per-group drops" setup = [
    BackwardFixtures
] begin
    sc = scenario("mixed_default")
    @test candidateset(sc) == sc.expected
    @test sc.expected == Set([
        "item:(Intercept)", "item:(Intercept);subj:(Intercept)", "subj:(Intercept),days"
    ])
end

@testitem "allownointercept changes the mixed-term per-group survivors" setup = [
    BackwardFixtures
] begin
    sc = scenario("mixed_noint")
    @test candidateset(sc) == sc.expected            # the (days|subj)+(1|item) neighbour survives
    @test "item:(Intercept);subj:days" in sc.expected
end

@testitem "keep pins one of two crossed terms (Pastes keep batch)" setup = [
    BackwardFixtures
] begin
    sc = scenario("pastes_keep_batch")
    @test candidateset(sc) == sc.expected            # {(1|batch), (1|batch)+(1|cask)}
    @test sc.expected == Set(["batch:(Intercept)", "batch:(Intercept);cask:(Intercept)"])
end

@testitem "backwardcandidates is set-equal to cAIC4 across every backward fixture" setup = [
    BackwardFixtures
] begin
    names = h5open(FIXTURE, "r") do f
        [k for k in keys(f) if k != "meta"]
    end
    @test !isempty(names)
    for name in names
        sc = scenario(name)
        @test candidateset(sc) == sc.expected
    end
end

@testitem "backwardcandidates is type-stable" setup = [BackwardFixtures] begin
    sc = scenario("sleepstudy_default")
    @test (@inferred Vector{cAIC.RESpec} cAIC.backwardcandidates(sc.spec)) isa
        Vector{cAIC.RESpec}
end
