# Level-1 set-equality tests for the forward candidate enumerator (`forwardcandidates`, M4 / #39)
# and the pure nesting ingredients (`_allnestsubs`, `isnested`). Each `@testitem` reads one scenario
# from the committed fixture `fixtures/stepcaic_forward_level1.h5` (written by
# `generate_fixtures_stepcaic_forward.R` driving cAIC4's real `forwardStep` / `allNestSubs` /
# `reformulas::isNested`) and asserts the Julia side is **set-equal** to the cAIC4 ground truth under
# the canonical encoding of docs/math/0008 §2.5/§3.6. No model is fitted — pure combinatorial
# structure (§6 Level 1).

# Shared helpers + fixture reader. `parsespec`/`canon` are identical to the backward harness: a
# canonical input string `"g/cor=1:(Intercept),days;…"` ↔ a `RESpec`, and a candidate `RESpec` ↔ its
# canonical term-string (an uncorrelated group expands to one single-label term per direction). The
# slope/group candidates are stored as comma strings ("" = none) and parsed to `Vector{Symbol}`.
@testsnippet ForwardFixtures begin
    using ConditionalAIC
    using HDF5

    const FIXTURE = joinpath(@__DIR__, "fixtures", "stepcaic_forward_level1.h5")

    function parsespec(s)
        groups = ConditionalAIC.REGroup[]
        for grp in split(s, ";")
            meta, dirs = split(grp, ":")
            nm = Symbol(replace(meta, r"/cor=.*" => ""))
            cor = parse(Int, replace(meta, r".*cor=" => "")) == 1
            push!(groups, ConditionalAIC.REGroup(nm, String.(split(dirs, ",")), cor))
        end
        return ConditionalAIC.RESpec(groups)
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
    parsecands(s) = isempty(s) ? Symbol[] : Symbol.(split(s, ","))

    # Read one scenario group into (spec, slope/group candidates, flags, expected-set).
    function scenario(name)
        h5open(FIXTURE, "r") do f
            g = f[name]
            spec = parsespec(only(read(g["input"])))
            slopes = parsecands(only(read(g["slopecandidates"])))
            groups = parsecands(only(read(g["groupcandidates"])))
            maxslopes = Int(only(read(g["maxslopes"])))
            useacross = only(read(g["useacross"])) == 1
            selcor = only(read(g["selectcorrelation"])) == 1
            expected = expectedset(only(read(g["expected"])))
            return (; spec, slopes, groups, maxslopes, useacross, selcor, expected)
        end
    end

    function candidateset(sc)
        cands = ConditionalAIC.forwardcandidates(
            sc.spec;
            slopecandidates=sc.slopes,
            groupcandidates=sc.groups,
            maxslopes=sc.maxslopes,
            useacross=sc.useacross,
            selectcorrelation=sc.selcor,
        )
        return Set(canon(c) for c in cands)
    end

    # Scenario names are every group except the non-scenario fixtures.
    scenarionames() =
        h5open(FIXTURE, "r") do f
            [k for k in keys(f) if !(k in ("meta", "nest", "isnested"))]
        end
end

@testitem "adds a new grouping factor" setup = [ForwardFixtures] begin
    sc = scenario("add_group")
    @test candidateset(sc) == sc.expected
    @test sc.expected == Set(["item:(Intercept);subj:(Intercept)"])
end

@testitem "adds a slope to an existing term" setup = [ForwardFixtures] begin
    sc = scenario("add_slope")
    @test candidateset(sc) == sc.expected            # (1|subj) → (1+days|subj)
    @test sc.expected == Set(["subj:(Intercept),days"])
end

@testitem "slope and group candidates combine (incl. intercept-less new term)" setup = [
    ForwardFixtures
] begin
    sc = scenario("add_slope_group")
    @test candidateset(sc) == sc.expected            # adds (1|item), (days|item), (1+days|subj)
    @test "item:days;subj:(Intercept)" in sc.expected   # no allownointercept: (days|item) is admissible
end

@testitem "selectcorrelation admits the uncorrelated split" setup = [ForwardFixtures] begin
    sc = scenario("add_slope_selcor")
    @test candidateset(sc) == sc.expected            # {(1+days|subj), (1|subj)+(0+days|subj)}
    @test "subj:(Intercept);subj:days" in sc.expected
end

@testitem "selectcorrelation with a group candidate keeps both the split and the new group" setup = [
    ForwardFixtures
] begin
    sc = scenario("slope_group_selcor")
    @test candidateset(sc) == sc.expected
    @test "subj:(Intercept);subj:days" in sc.expected
    @test "item:(Intercept);subj:(Intercept)" in sc.expected
end

@testitem "useacross migrates an existing slope to the new grouping" setup = [
    ForwardFixtures
] begin
    sc = scenario("useacross")
    @test candidateset(sc) == sc.expected            # (days|item) appears though days ∉ slopecandidates
    @test sc.expected ==
        Set(["item:(Intercept);subj:(Intercept),days", "item:days;subj:(Intercept),days"])
end

@testitem "a slope grows each crossed term in turn" setup = [ForwardFixtures] begin
    sc = scenario("two_groups_slope")
    @test candidateset(sc) == sc.expected
    @test sc.expected == Set([
        "item:(Intercept),days;subj:(Intercept)", "item:(Intercept);subj:(Intercept),days"
    ])
end

@testitem "maxslopes caps the combination size" setup = [ForwardFixtures] begin
    sc = scenario("maxslopes_cap")               # maxslopes=1 → only single-slope+intercept growth
    @test candidateset(sc) == sc.expected
    @test sc.expected ==
        Set(["subj:(Intercept),x", "subj:(Intercept),y", "subj:(Intercept),z"])
end

@testitem "the one-direction-larger restriction drops size-3 collapses" setup = [
    ForwardFixtures
] begin
    # slopes x,y, maxslopes=2: the {x,y,(Intercept)} combo collapses (hierarchical order) to a
    # length-3 term, which exceeds length(cnms[subj])+1 = 2 and is dropped.
    sc = scenario("onelarger_cap")
    @test candidateset(sc) == sc.expected
    @test sc.expected == Set(["subj:(Intercept),x", "subj:(Intercept),y"])
end

@testitem "an existing two-direction term cannot grow without useacross" setup = [
    ForwardFixtures
] begin
    sc = scenario("existing2_noacross")
    @test isempty(candidateset(sc))                  # cAIC4 NULL → empty Vector{RESpec}
    @test sc.expected == Set{String}()
end

@testitem "a single random intercept with no candidates is the terminal" setup = [
    ForwardFixtures
] begin
    sc = scenario("no_candidates")
    @test isempty(ConditionalAIC.forwardcandidates(sc.spec))   # NULL terminal
    @test sc.expected == Set{String}()
end

@testitem "forwardcandidates is set-equal to cAIC4 across every forward fixture" setup = [
    ForwardFixtures
] begin
    names = scenarionames()
    @test !isempty(names)
    for name in names
        sc = scenario(name)
        @test candidateset(sc) == sc.expected
    end
end

@testitem "forwardcandidates is type-stable" setup = [ForwardFixtures] begin
    sc = scenario("add_slope")
    @test (@inferred Vector{ConditionalAIC.RESpec} ConditionalAIC.forwardcandidates(
        sc.spec; slopecandidates=[:days]
    )) isa Vector{ConditionalAIC.RESpec}
end

@testitem "forwardcandidates rejects maxslopes < 1" setup = [ForwardFixtures] begin
    spec = ConditionalAIC.RESpec([ConditionalAIC.REGroup(:subj, ["(Intercept)"], true)])
    @test_throws ArgumentError ConditionalAIC.forwardcandidates(spec; maxslopes=0)
end

@testitem "_allnestsubs expands nesting expressions like allNestSubs" setup = [
    ForwardFixtures
] begin
    exprs, expected = h5open(FIXTURE, "r") do f
        (read(f["nest/expr"]), read(f["nest/expected"]))
    end
    for (e, exp) in zip(exprs, expected)
        @test ConditionalAIC._allnestsubs(e) == String.(split(exp, ","))
    end
    @test ConditionalAIC._allnestsubs("a/b") == ["b:a", "a"]
end

@testitem "isnested matches reformulas::isNested" setup = [ForwardFixtures] begin
    names = h5open(FIXTURE, "r") do f
        keys(f["isnested"])
    end
    @test !isempty(names)
    for name in names
        f1, f2, expected = h5open(FIXTURE, "r") do f
            g = f["isnested"][name]
            (read(g["f1"]), read(g["f2"]), only(read(g["expected"])))
        end
        @test ConditionalAIC.isnested(f1, f2) == (expected == 1)
    end
    @test_throws ArgumentError ConditionalAIC.isnested([1, 2, 3], [1, 2])
end
