# Conditional stepwise random-effects selection (M4). This file holds the candidate
# *enumeration* — the pure combinatorial neighbourhood of a fitted model's RE structure that
# the greedy `stepcaic` driver scores. The scoring/driver lives elsewhere; here we only build
# the neighbours.
#
# `backwardcandidates` is the faithful port of `cAIC4`'s internal `backwardStep`
# (`R/helperfuns_stepcAIC.R:93`). It enumerates the RE structures one direction **smaller** than
# the input. The port runs on the `cnms` bridge of docs/math/0008 §2.1: a `RESpec` is lowered to
# the repeated-name `cnms` form lme4 uses (a *correlated* `(1+x|g)` is one two-label entry; an
# *uncorrelated* `(1+x||g)` is two single-label entries), the `backwardStep` transform runs there,
# and each surviving candidate is lifted back to a `RESpec`. The `lm`/`glm` terminal (§0.1) is not
# a `RESpec` ([`render`](@ref cAIC.render) rejects an empty spec), so it is represented by an
# **empty** returned vector — the image of `cAIC4`'s `NA` return.
#
# See docs/math/0008-stepcaic-search.md §2 for the spec and the Level-1 set-equality oracle.

# One `cnms` entry: a grouping name and its direction labels. A correlated term is one entry; an
# uncorrelated term is several single-direction entries sharing a name (mirrors lme4's repeated
# `cnms` names). `_RawTerm` additionally admits the `nothing` "NA" marker `backwardStep` uses for
# a wholly-dropped term before `checkREs` prunes it.
const _Term = Tuple{Symbol,Vector{String}}
const _RawTerm = Tuple{Symbol,Union{Nothing,Vector{String}}}

# Lower a `RESpec` to the repeated-name `cnms` form (docs/math/0008 §2.1 `cnmsform`).
function _cnmsform(spec::RESpec)
    out = _Term[]
    for g in spec.groups
        if g.correlated
            push!(out, (g.grouping, copy(g.directions)))
        else
            for d in g.directions
                push!(out, (g.grouping, [d]))
            end
        end
    end
    return out
end

# Lift a `cnms`-form candidate back to a `RESpec` (the inverse `respec`): regroup by grouping in
# first-appearance order; a grouping seen **once** is a correlated term, a grouping seen in
# several single-direction entries is one uncorrelated term over the concatenated directions.
function _respec(cnms::Vector{_Term})
    order = Symbol[]
    bygrp = Dict{Symbol,Vector{Vector{String}}}()
    for (nm, d) in cnms
        if !haskey(bygrp, nm)
            push!(order, nm)
            bygrp[nm] = Vector{Vector{String}}()
        end
        push!(bygrp[nm], d)
    end
    groups = REGroup[]
    for nm in order
        vs = bygrp[nm]
        if length(vs) == 1
            push!(groups, REGroup(nm, vs[1], true))
        else
            push!(groups, REGroup(nm, reduce(vcat, vs), false))
        end
    end
    return RESpec(groups)
end

# Canonical key of a `cnms`-form candidate (docs/math/0008 §2.1): sorted multiset of
# `"grouping:sorted(directions)"` term-strings. Two candidates are equal iff their keys are.
function _canonkey(cand::Vector{_Term})
    terms = [string(nm, ":", join(sort(d), ",")) for (nm, d) in cand]
    return join(sort(terms), ";")
end

# Lexicographic `k`-combinations of `v` (R's `combn`), order-preserving. Dependency-free (no
# `Combinatorics.jl`); used by the forward add-set (§3.1) and `_allcombn`.
function _combinations(v::Vector{T}, k::Int) where {T}
    n = length(v)
    res = Vector{Vector{T}}()
    (k < 0 || k > n) && return res
    k == 0 && (push!(res, T[]); return res)
    idx = collect(1:k)
    while true
        push!(res, v[idx])
        i = k
        while i >= 1 && idx[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        idx[i] += 1
        for j in (i + 1):k
            idx[j] = idx[j - 1] + 1
        end
    end
    return res
end

# `allCombn` (`R/helperfuns_stepcAIC.R:4`): all **proper** sub-combinations of `x` (sizes
# `1:length(x)-1`). Used by `checkHierarchicalOrder` to find the smaller terms a larger term
# subsumes. `x` arrives sorted (from `checkREs`), so the returned sub-combos are sorted too.
function _allcombn(x::Vector{String})
    out = Vector{Vector{String}}()
    for k in 1:(length(x) - 1)
        for c in _combinations(x, k)
            push!(out, c)
        end
    end
    return out
end

# `checkHierarchicalOrder` (`R/helperfuns_stepcAIC.R:317`): given the (sorted, de-duplicated)
# direction-vectors of **one** grouping, drop every term that is a proper sub-combination of a
# longer surviving term — `(1|g)` is redundant once `(1+x|g)` is present. Faithful port of the
# load-bearing quirks: `lenMax`/`lenMin` are fixed before the loop, the index `i` advances over the
# *shrinking* list, and `listIn[i]` (never its own proper sub-combo) is retained.
function _checkhierorder(terms::Vector{Vector{String}})
    isempty(terms) && return terms
    listin = sort(terms; by=length, rev=true)
    lenmax = length(listin[1])
    lenmin = length(listin[end])
    i = 1
    while i < length(listin)
        leni = length(listin[i])
        if lenmax > 1 && leni > lenmin
            notallowed = _allcombn(listin[i])
            listin = Vector{String}[v for v in listin if !(v in notallowed)]
        end
        i += 1
    end
    return listin
end

# `checkREs` (`R/helperfuns_stepcAIC.R:354`): per candidate, drop NA/empty terms, then within each
# grouping sort each direction-vector, drop duplicate vectors, and — when a grouping keeps **>1**
# distinct term — enforce hierarchical order. A backward single-term-per-grouping candidate never
# trips the `>1` branch (it is a no-op there); the forward enumerator (§3.2) reaches it when a
# multi-direction slope combination is added to an existing term, so the step lives in the shared
# `checkREs` rather than a forward-only fork.
function _checkres(cand::Vector{_RawTerm})
    live = _Term[]
    for (nm, d) in cand
        d === nothing && continue          # NA marker
        isempty(d) && continue             # all-NA / empty term (any(!is.na(·)) == false)
        push!(live, (nm, d))               # `d` narrowed to Vector{String} here
    end
    result = _Term[]
    for nm in unique(Symbol[t[1] for t in live])
        uniq = Vector{Vector{String}}()
        for (tn, d) in live
            tn == nm || continue
            sd = sort(d)
            sd in uniq || push!(uniq, sd)
        end
        length(uniq) > 1 && (uniq = _checkhierorder(uniq))
        for v in uniq
            push!(result, (nm, v))
        end
    end
    return result
end

# `removeUncor` (`:596`, skipped under `selectcorrelation`): drop a candidate when some grouping
# keeps >1 term and one of those terms is itself a mixed intercept+slope term. Inert on
# single-term-per-grouping candidates (every backward candidate), where no grouping has >1 term.
function _removeuncor(cands::Vector{Vector{_Term}})
    out = Vector{_Term}[]
    for c in cands
        order = Symbol[]
        bygrp = Dict{Symbol,Vector{Vector{String}}}()
        for (nm, d) in c
            haskey(bygrp, nm) || (push!(order, nm); bygrp[nm]=Vector{Vector{String}}())
            push!(bygrp[nm], d)
        end
        drop = false
        for nm in order
            vs = bygrp[nm]
            length(vs) > 1 || continue
            for v in vs
                if ("(Intercept)" in v) && any(x -> x != "(Intercept)", v)
                    drop = true
                    break
                end
            end
            drop && break
        end
        drop || push!(out, c)
    end
    return out
end

# `removeNoInt` (`:643`, skipped under `allownointercept`): from each candidate drop every term
# whose grouping carries no `"(Intercept)"` anywhere, then drop candidates left empty. Per-term
# removal (not a per-candidate reject) — this is why `(1+days|subj)` yields only `{(1|subj)}`.
function _removenoint(cands::Vector{Vector{_Term}})
    out = Vector{_Term}[]
    for c in cands
        groupint = Dict{Symbol,Bool}()
        for (nm, d) in c
            groupint[nm] = get(groupint, nm, false) || ("(Intercept)" in d)
        end
        kept = _Term[t for t in c if groupint[t[1]]]
        isempty(kept) || push!(out, kept)
    end
    return out
end

# The `backwardStep` transform on `cnms` form (`:93–199`), returning the post-filter clean
# candidates. `keep` is the parsed survive-set as `cnms`-form entries (the `interpret.random`
# analogue); `nothing` means no floor.
function _backwardstep(
    cnms::Vector{_Term};
    keep::Union{Nothing,Vector{_Term}},
    selectcorrelation::Bool,
    allownointercept::Bool,
)
    # terminal guard (:96): a single random intercept overall. With no `keep` this is the `NA`
    # terminal (empty); with `keep` the model is its own sole neighbour.
    total = sum(length(d) for (_, d) in cnms; init=0)
    if total == 1
        return keep === nothing ? Vector{_Term}[] : Vector{_Term}[cnms]
    end

    # `rep(cnms, lengths)` then `split` by name: each grouping → its copies (one per direction of
    # each of its entries). `split` orders groupings alphabetically; mirror that.
    names_order = sort!(unique(Symbol[nm for (nm, _) in cnms]))   # Symbol isless = lexicographic
    copies = Dict{Symbol,Vector{Vector{String}}}()
    for nm in names_order
        copies[nm] = Vector{Vector{String}}()
    end
    for (nm, d) in cnms
        for _ in 1:length(d)
            push!(copies[nm], d)
        end
    end

    # `keep` (:110): remove every kept direction from the droppable copies of its grouping. The
    # `temp[indRem]` indexing is `unlist(temp) != unlist(keep)` recycled to the copy count —
    # faithful to the source for the single-label keeps it is exercised on.
    if keep !== nothing
        keepflat = String[]
        keepnames = Set{Symbol}()
        for (nm, d) in keep
            push!(keepnames, nm)
            append!(keepflat, d)
        end
        nfk = length(keepflat)
        for nm in names_order
            nm in keepnames || continue
            cs = copies[nm]
            flat = reduce(vcat, cs; init=String[])
            isempty(flat) && continue
            indrem = Bool[flat[k] != keepflat[(k - 1) % nfk + 1] for k in 1:length(flat)]
            ni = length(indrem)
            copies[nm] = cs[Bool[indrem[(i - 1) % ni + 1] for i in 1:length(cs)]]
        end
    end

    # per-direction drop (:128): a grouping with ≤1 copy → the `NA` marker (whole term dropped); a
    # grouping with k>1 copies → k reduced copies, the i-th with its i-th label removed
    # (`d[[i]] <- d[[i]][-i]`). The `[-i]` is a no-op when i exceeds the copy length — the
    # uncorrelated-split degeneracy (docs/math/0008 §2.2).
    newgroups = Tuple{Symbol,Vector{Union{Nothing,Vector{String}}}}[]
    for nm in names_order
        cs = copies[nm]
        k = length(cs)
        if k <= 1
            push!(newgroups, (nm, Union{Nothing,Vector{String}}[nothing]))
        else
            red = Union{Nothing,Vector{String}}[]
            for i in 1:k
                d = cs[i]
                if i <= length(d)
                    push!(red, String[d[j] for j in 1:length(d) if j != i])
                else
                    push!(red, copy(d))
                end
            end
            push!(newgroups, (nm, red))
        end
    end

    # `keep` re-add (:153): every kept grouping gets its kept directions back, so every candidate
    # retains them (replacing the `NA` marker, else appended).
    if keep !== nothing
        for (n, kd) in keep
            idx = findfirst(p -> p[1] == n, newgroups)
            if idx === nothing
                push!(newgroups, (n, Union{Nothing,Vector{String}}[copy(kd)]))
            else
                vals = newgroups[idx][2]
                if length(vals) == 1 && vals[1] === nothing
                    newgroups[idx] = (n, Union{Nothing,Vector{String}}[copy(kd)])
                else
                    push!(vals, copy(kd))
                end
            end
        end
    end

    # flatten to the per-reduced-term list (`unlist(recursive=FALSE)` with names restored).
    newterms = _RawTerm[]
    for (nm, vals) in newgroups
        for v in vals
            push!(newterms, (nm, v))
        end
    end

    # candidate assembly (:171): each reduced term replaces *all* same-name entries of `cnms`.
    rawcands = Vector{_RawTerm}[]
    for (nm_i, v_i) in newterms
        cand = _RawTerm[]
        for (nm, d) in cnms
            nm == nm_i && continue
            push!(cand, (nm, d))
        end
        push!(cand, (nm_i, v_i))
        push!(rawcands, cand)
    end

    # `notempty` (:180): drop a candidate whose first term is empty (the empty-reduction images).
    _vlen(v) = v === nothing ? 1 : length(v)
    rawcands = Vector{_RawTerm}[c for c in rawcands if _vlen(c[1][2]) > 0]

    # `checkREs` then the order-load-bearing filters (:185).
    cands = Vector{_Term}[_checkres(c) for c in rawcands]
    cands = Vector{_Term}[c for c in cands if !isempty(c)]
    selectcorrelation || (cands = _removeuncor(cands))
    allownointercept || (cands = _removenoint(cands))
    return cands
end

"""
    backwardcandidates(spec::RESpec; keep=nothing, selectcorrelation=false, allownointercept=false)
        -> Vector{RESpec}

Enumerate the random-effects structures one direction **smaller** than `spec` — the faithful
port of `cAIC4`'s internal `backwardStep`, the backward branch of the `stepcaic` search (M4).

Each returned [`RESpec`](@ref) is a neighbour obtained by dropping a single random-effects
direction (a slope, an intercept, or a whole term) from `spec`, after `cAIC4`'s `checkREs`
de-duplication and its two structural filters. An **empty** result is the terminal/exhausted
signal — the image of `cAIC4`'s `NA` return and of an empty candidate list; the `lm`/`glm`
no-random-effects terminal is not itself a `RESpec` and so is never an element.

The enumeration runs on the `cnms` representation `cAIC4` uses (docs/math/0008 §2.1): `spec` is
lowered to the repeated-name `cnms` form, transformed, and each surviving candidate lifted back
to a `RESpec`; the result is de-duplicated by the canonical term-multiset encoding.

# Arguments
- `spec::RESpec` — the structure to enumerate neighbours of.
- `keep::Union{Nothing,RESpec}` — directions that must survive every candidate (the `cAIC4`
  `keep` floor, parsed to a `RESpec`); `nothing` for no floor.
- `selectcorrelation::Bool` — when `false` (default), `removeUncor` drops candidates that encode
  a correlated→uncorrelated split.
- `allownointercept::Bool` — when `false` (default), `removeNoInt` strips intercept-less terms,
  dropping candidates left empty.

# Returns
- `Vector{RESpec}` — the de-duplicated backward neighbours; empty at the terminal.

# Example
```julia
m = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), sleepstudy)
backwardcandidates(extract(m))            # [RESpec([REGroup(:subj, ["(Intercept)"], true)])]
```
"""
function backwardcandidates(
    spec::RESpec;
    keep::Union{Nothing,RESpec}=nothing,
    selectcorrelation::Bool=false,
    allownointercept::Bool=false,
)
    cnms = _cnmsform(spec)
    keepterms =
        keep === nothing ? nothing : _Term[(g.grouping, g.directions) for g in keep.groups]
    cands = _backwardstep(cnms; keep=keepterms, selectcorrelation, allownointercept)
    seen = Set{String}()
    out = RESpec[]
    for c in cands
        isempty(c) && continue
        k = _canonkey(c)
        k in seen && continue
        push!(seen, k)
        push!(out, _respec(c))
    end
    return out
end

# `unique(c(unlist(cnms), slopeCandidates), incomparables = "(Intercept)")` (`forwardStep:526`): the
# `incomparables = "(Intercept)"` keeps every intercept label (never de-duplicated), de-duplicating
# only the slope variables. The duplicate single-intercept combinations this yields are removed by
# the combo-dedup and the final candidate dedup, so the candidate **set** is unchanged.
function _uniqueslopes(v::Vector{String})
    out = String[]
    for x in v
        if x == "(Intercept)"
            push!(out, x)
        elseif !(x in out)
            push!(out, x)
        end
    end
    return out
end

# length of the **first** `cnms` entry named `g` (R's `cnms[[g]]`), 0 if absent. The one-larger
# restriction (`forwardStep:566`) measures an existing grouping's current size against this.
function _firstlen(cnms::Vector{_Term}, g::Symbol)
    for (nm, d) in cnms
        nm == g && return length(d)
    end
    return 0
end

# The `forwardStep` transform on `cnms` form (`R/helperfuns_stepcAIC.R:516–590`), returning the
# post-filter clean candidates one direction **larger** than `cnms`. `nrofcombs` is the slope-combo
# cap (`maxslopes + 1`, the `+1` the intercept slot, driver redefine `R/stepcAIC.R:302`).
function _forwardstep(
    cnms::Vector{_Term};
    slopecandidates::Vector{String},
    groupcandidates::Vector{Symbol},
    nrofcombs::Int,
    useacross::Bool,
    selectcorrelation::Bool,
)
    # allSlopes (:525): existing directions ∪ candidates under useacross, else candidates ∪ intercept.
    allslopes = if useacross
        _uniqueslopes(vcat([d for (_, ds) in cnms for d in ds], slopecandidates))
    else
        vcat(slopecandidates, ["(Intercept)"])
    end

    # allGroups (:529): existing groupings ∪ candidates, first-appearance order.
    allgroups = Symbol[]
    for (nm, _) in cnms
        nm in allgroups || push!(allgroups, nm)
    end
    for g in groupcandidates
        g in allgroups || push!(allgroups, g)
    end

    # allSlopeCombs (:533–543): all size-i combos (i ≤ min(nrofcombs, #slopes)) with no repeated label.
    combs = Vector{Vector{String}}()
    for i in 1:nrofcombs
        i <= length(allslopes) || continue
        for c in _combinations(allslopes, i)
            length(unique(c)) == length(c) && push!(combs, c)
        end
    end

    # cross product (:545–548): every (group, combo) appended to cnms.
    assembled = Vector{Vector{_Term}}()
    for combo in combs
        for g in allgroups
            cand = copy(cnms)
            push!(cand, (g, combo))
            push!(assembled, cand)
        end
    end

    # checkREs (:549) — drops/sorts/dedups and collapses hierarchical sub-terms (§3.2).
    cands = Vector{_Term}[_checkres(_RawTerm[(nm, d) for (nm, d) in c]) for c in assembled]

    # same-grouping reject (:550–552, skipped under selectcorrelation): no grouping in >1 term.
    if !selectcorrelation
        cands = Vector{_Term}[
            c for c in cands if length(unique(Symbol[t[1] for t in c])) == length(c)
        ]
    end

    # dedup (:553) + drop-original (:554): discard a candidate equal to the (sorted) input cnms.
    cnmskey = _canonkey(cnms)
    seen = Set{String}()
    deduped = Vector{_Term}[]
    for c in cands
        isempty(c) && continue
        k = _canonkey(c)
        (k == cnmskey || k in seen) && continue
        push!(seen, k)
        push!(deduped, c)
    end

    # removeUncor (:563, skipped under selectcorrelation).
    selectcorrelation || (deduped = _removeuncor(deduped))

    # one-direction-larger (:566–582): a new grouping may gain 1 direction; an existing one grows ≤ 1.
    cnmsnames = Set(nm for (nm, _) in cnms)
    out = Vector{_Term}[]
    for c in deduped
        violates = false
        for (g, dirs) in c
            if g in cnmsnames
                length(dirs) > _firstlen(cnms, g) + 1 && (violates=true; break)
            else
                length(dirs) > 1 && (violates=true; break)
            end
        end
        violates || push!(out, c)
    end
    return out
end

"""
    forwardcandidates(spec::RESpec; slopecandidates=Symbol[], groupcandidates=Symbol[],
                      maxslopes=2, useacross=false, selectcorrelation=false) -> Vector{RESpec}

Enumerate the random-effects structures one direction **larger** than `spec` — the faithful port of
`cAIC4`'s internal `forwardStep`, the forward branch of the `stepcaic` search (M4).

Each returned [`RESpec`](@ref) adds a single random-effects direction to `spec`: a new slope on an
existing term, a new term over an existing grouping, or a new grouping factor — after `cAIC4`'s
`checkREs` de-duplication (including its hierarchical-order collapse), the structural filters, and
the *one-direction-larger* restriction. An **empty** result is the terminal/exhausted signal — the
image of `cAIC4`'s `NULL` return when no admissible enlargement exists. Forward has no `keep` and no
`allownointercept`: intercept-less enlargements (e.g. a `(slope | newgroup)` term) are admissible.

The enumeration runs on the `cnms` representation `cAIC4` uses (docs/math/0008 §2.1): `spec` is
lowered to the repeated-name `cnms` form, transformed, and each surviving candidate lifted back to a
`RESpec`, de-duplicated by the canonical term-multiset encoding.

# Arguments
- `spec::RESpec` — the structure to enumerate neighbours of.
- `slopecandidates::Vector{Symbol}` — slope variables eligible to be added.
- `groupcandidates::Vector{Symbol}` — grouping factors eligible to be added.
- `maxslopes::Int` — cap on slopes per grouping (`cAIC4` `numberOfPermissibleSlopes`); the combo size
  is `maxslopes + 1`, the `+1` reserving the intercept slot.
- `useacross::Bool` — when `true`, existing slopes may migrate to other groupings (`allowUseAcross`).
- `selectcorrelation::Bool` — when `false` (default), uncorrelated splits are rejected (the same-name
  filter and `removeUncor`); when `true` they are admissible candidates.

# Returns
- `Vector{RESpec}` — the de-duplicated forward neighbours; empty at the terminal.

# Example
```julia
m = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), sleepstudy)
forwardcandidates(extract(m); slopecandidates=[:days])   # [RESpec([REGroup(:subj, ["(Intercept)", "days"], true)])]
```
"""
function forwardcandidates(
    spec::RESpec;
    slopecandidates::Vector{Symbol}=Symbol[],
    groupcandidates::Vector{Symbol}=Symbol[],
    maxslopes::Int=2,
    useacross::Bool=false,
    selectcorrelation::Bool=false,
)
    maxslopes >= 1 || throw(ArgumentError("maxslopes must be ≥ 1 (got $maxslopes)"))
    cnms = _cnmsform(spec)
    cands = _forwardstep(
        cnms;
        slopecandidates=String[string(s) for s in slopecandidates],
        groupcandidates=groupcandidates,
        nrofcombs=maxslopes + 1,
        useacross=useacross,
        selectcorrelation=selectcorrelation,
    )
    seen = Set{String}()
    out = RESpec[]
    for c in cands
        isempty(c) && continue
        k = _canonkey(c)
        k in seen && continue
        push!(seen, k)
        push!(out, _respec(c))
    end
    return out
end

# ── nesting ingredients (docs/math/0008 §3.5) ─────────────────────────────────
# The data-dependent `expandnesting` glue (column extraction + warn-and-drop) is driver-side and
# deferred to §4; #39 ships the two pure ingredients below.

"""
    _allnestsubs(s::AbstractString) -> Vector{String}

The pure string expansion of a nesting expression into its sub-groupings — the port of `cAIC4`'s
`allNestSubs` (`R/helperfuns_stepcAIC.R:417`). `(1 | a/b)` expands to the grouping factors `b:a`
(the interaction, innermost-first) and `a` (the outer factor):

```
_allnestsubs("a/b")   == ["b:a", "a"]
_allnestsubs("a/b/c") == ["c:b:a", "b:a", "a"]
```
"""
function _allnestsubs(s::AbstractString)
    parts = strip.(split(s, "/"))
    any(isempty, parts) && throw(ArgumentError("malformed nesting expression: $(repr(s))"))
    out = String[]
    # `findbars(~ (1 | a/b/c))` yields the bars (1|c:b:a), (1|b:a), (1|a): nested interaction terms
    # of decreasing depth. Their grouping exprs are the reversed-prefix `:`-joins, innermost first.
    for k in length(parts):-1:1
        push!(out, join(reverse(parts[1:k]), ":"))
    end
    return out
end

"""
    isnested(f1, f2) -> Bool

Whether factor `f1` is nested within factor `f2` — the port of `lme4`/`reformulas::isNested`
(`R/stepcAIC.R:216`). True iff every level of `f1` co-occurs with **at most one** level of `f2`.

# Arguments
- `f1`, `f2` — equal-length factor vectors (any element type compared by `isequal`).

# Returns
- `Bool` — `true` when `f1` is nested within `f2`.
"""
function isnested(f1::AbstractVector, f2::AbstractVector)
    length(f1) == length(f2) || throw(
        ArgumentError(
            "isnested requires equal-length factors ($(length(f1)) ≠ $(length(f2)))"
        ),
    )
    seen = Dict{Any,Any}()
    for (a, b) in zip(f1, f2)
        if haskey(seen, a)
            isequal(seen[a], b) || return false
        else
            seen[a] = b
        end
    end
    return true
end

# ── the greedy controller (docs/math/0008 §4.1) and its result types (§5.1) ───
# The driver scores the input, then walks the candidate neighbourhood greedily, accepting the
# minimum-cAIC neighbour while it does not increase the cAIC (the `≤` rule). This is the faithful
# backward, non-`both` subset of `cAIC4`'s `stepcAIC` (#40 walking skeleton).

"""
    ScoredCandidate{T}

One scored neighbour of a [`stepcaic`](@ref) step — the candidate random-effects structure, its
conditional AIC, and the bias-corrected effective degrees of freedom ρ (`dof`) that AIC was
penalised by (the `cAIC`/`df` pair `cAIC4`'s `stepcAIC` trace/`aicTab` prints per candidate).
`spec === nothing` marks the `lm`/`glm` terminal node (no `RESpec`, §0.1). Part of a
[`StepRecord`](@ref).
"""
struct ScoredCandidate{T<:AbstractFloat}
    spec::Union{RESpec,Nothing}
    caic::T
    dof::T
end

"""
    StepRecord{T}

One greedy step of a [`stepcaic`](@ref) search — its direction, the incumbent cAIC at the start
of the step, **every** candidate scored that step ([`ScoredCandidate`](@ref)), the `argmin`
index into them, and whether the best candidate was accepted (`minCAIC ≤ incumbentcaic`). The
structured analogue of `cAIC4`'s printed `trace` (the *Search path* of `CONTEXT.md`).
"""
struct StepRecord{T<:AbstractFloat}
    direction::Symbol
    incumbentcaic::T
    candidates::Vector{ScoredCandidate{T}}
    bestindex::Int
    accepted::Bool
end

"""
    StepcaicOptions

The resolved [`stepcaic`](@ref) options retained for provenance (`cAIC4`'s call record analogue).
The forward / `both` enumeration options (`groupcandidates`, `slopecandidates`, `maxslopes`,
`useacross`) carry the forward arc's resolved settings (M4 §5.1); they are empty/defaulted for a
pure backward run. `skipnonconverged` records whether non-converged candidates were excluded from
the comparison (the `cAIC4` `calcNonOptimMod` analogue). `keep` is not retained here — it is a
[`RESpec`](@ref) floor threaded into the backward enumeration, not a scalar provenance field.
"""
struct StepcaicOptions
    direction::Symbol
    selectcorrelation::Bool
    allownointercept::Bool
    steps::Int
    savedmodels::Int
    skipnonconverged::Bool
    groupcandidates::Vector{Symbol}
    slopecandidates::Vector{Symbol}
    maxslopes::Int
    useacross::Bool
end

"""
    StepcaicResult{T<:AbstractFloat,M<:RegressionModel}

The result of a conditional stepwise search ([`stepcaic`](@ref)).

- `selected::CAICResult{T,M}` — the conditional-AIC score of the selected model.
- `model::M` — the selected fitted model (a `MixedModel`, or the `lm`/`glm` terminal at the
  bottom of the lattice).
- `path::Vector{StepRecord{T}}` — the per-step search trace, in order (the *Search path* of
  `CONTEXT.md`, replacing `cAIC4`'s printed `trace`).
- `saved::Vector{CAICResult{T}}` — the k-best scores (`savedmodels`): the distinct models scored
  across the whole search, deduplicated by random-effects structure and ranked by cAIC ascending
  (the selected model is the first element). The default `savedmodels = 1` carries the selected
  score only. The element type is the `M`-erased `CAICResult{T}` (not `CAICResult{T,M}`) so a single
  ranked list can hold both the `MixedModel` candidates and the `lm`/`glm` terminal when both are
  among the best — the across-type heterogeneity of the saved set (the driver is not a hot kernel).
- `options::StepcaicOptions` — the resolved options, for provenance.

Parametric on the *selected* model type `M`; the driver is not a hot kernel, so the across-path
return-type variation (a `MixedModel` above the terminal, the `GLM.jl` terminal at it) is
acceptable.
"""
struct StepcaicResult{T<:AbstractFloat,M<:RegressionModel}
    selected::CAICResult{T,M}
    model::M
    path::Vector{StepRecord{T}}
    saved::Vector{CAICResult{T}}
    options::StepcaicOptions
end

# Total random-effects directions across all groups — the cnms `sum(lengths)` of `backwardStep`'s
# terminal guard (`:96`). `== 1` marks the single-direction terminal at which the only smaller
# neighbour is the `lm`/`glm` no-random-effects node (§0.1), distinguishing the terminal descent
# from a filter-stripped empty neighbourhood (the `minCAIC == Inf` exhausted arc).
_totaldirections(spec::RESpec) = sum(length(g.directions) for g in spec.groups; init=0)

# Structure key for the `savedmodels` dedup (`cAIC4`'s `duplicatedMers`): two scored models share
# an entry iff they have the same random-effects structure. A `RESpec` keys on its canonical
# `cnms` term-multiset (the same encoding `backwardcandidates` de-duplicates on); the `lm`/`glm`
# terminal (no random effects) keys on a reserved sentinel distinct from any `RESpec` key.
const _TERMINALKEY = "\0lm/glm-terminal"
_savedkey(spec::RESpec) = _canonkey(_cnmsform(spec))

function Base.show(io::IO, ::MIME"text/plain", r::StepcaicResult)
    println(io, "Conditional stepwise selection (stepcaic)")
    println(io, "  direction   = :", r.options.direction)
    println(io, "  steps taken = ", length(r.path))
    println(io, "  selected cAIC = ", r.selected.caic)
    print(io, "  selected df (ρ) = ", r.selected.dof)
    return nothing
end

# Shared option validation for both `stepcaic` methods: the supported directions and the
# forward/`both` call-consistency check (`R/stepcAIC.R:347–359`) — a forward or `both` run needs
# something to add (a slope candidate, a group candidate, or `useacross`). `fixEfCandidates` is out
# of scope (fixed effects held constant, §0).
function _validatestepcaic(
    direction::Symbol,
    savedmodels::Int,
    slopecandidates::Vector{Symbol},
    groupcandidates::Vector{Symbol},
    maxslopes::Int,
    useacross::Bool,
)
    direction in (:backward, :forward, :both) || throw(
        ArgumentError(
            "stepcaic direction must be :backward, :forward, or :both; got :$(direction)",
        ),
    )
    savedmodels >= 0 ||
        throw(ArgumentError("savedmodels must be ≥ 0 (0 keeps all); got $savedmodels"))
    if direction in (:forward, :both)
        maxslopes >= 1 || throw(ArgumentError("maxslopes must be ≥ 1 (got $maxslopes)"))
        (isempty(slopecandidates) && isempty(groupcandidates) && !useacross) && throw(
            ArgumentError(
                "stepcaic direction = :$(direction) cannot make forward steps without candidate " *
                "random-effect covariates: supply slopecandidates, groupcandidates, or useacross=true",
            ),
        )
    end
    return nothing
end

"""
    stepcaic(m::LinearMixedModel, data; direction=:backward, groupcandidates=Symbol[],
             slopecandidates=Symbol[], maxslopes=2, useacross=false, keep=nothing,
             selectcorrelation=false, allownointercept=false, steps=50, savedmodels=1,
             method=:auto, hessian=:analytic, nboot=nothing, sigmapenalty=1,
             rng=Random.default_rng()) -> StepcaicResult

Conditional stepwise random-effects selection guided by the conditional AIC — the greedy
controller of `cAIC4`'s `stepcAIC`, in the **backward**, **forward**, or **both** direction
(M4 §4.1–§4.2).

Starting from the fitted model `m`, each step enumerates the random-effects neighbours one
direction *smaller* ([`backwardcandidates`](@ref), backward) or *larger* ([`forwardcandidates`](@ref),
forward), rebuilds and refits each over `data`, and scores it with [`caic`](@ref). The minimum-cAIC
neighbour is accepted while it does not increase the cAIC (the `≤` rule); `direction = :both` starts
forward and alternates after each accepted or non-improving turn (the `improvementInBoth` /
`equalToLastStep` cascade). The search stops when no neighbour improves, the neighbourhood is
exhausted, or `steps` is reached. The selected model, its score, and the full search path are
returned in a [`StepcaicResult`](@ref).

Every candidate is scored with the **same** forwarded scoring kwargs (`method`/`hessian`/`nboot`/
`sigmapenalty`/`rng`) as the input — the consistent-scoring requirement of the search.

# Arguments
- `m`: the fitted Gaussian `LinearMixedModel` to search from.
- `data`: the data table the candidates are refit over (required — a rebuilt formula must be
  fitted, mirroring `cAIC4`'s required `data`).
- `direction`: `:backward` (default), `:forward`, or `:both`.
- `groupcandidates`, `slopecandidates`, `maxslopes`, `useacross`: the forward enumeration inputs
  (see [`forwardcandidates`](@ref)); ignored for a backward run. A forward/`both` run requires at
  least one of `slopecandidates`/`groupcandidates` (or `useacross`), else `ArgumentError`.
- `keep`: a `FormulaTerm` RE fragment (e.g. `@formula(y ~ (1 | g))`) parsed against `data` to the
  [`RESpec`](@ref) floor the backward search must not drop below (the `cAIC4` `keep` analogue);
  `nothing` for no floor. The formula's response and fixed-effects terms are ignored.
- `selectcorrelation`, `allownointercept`: the enumeration flags (see `backwardcandidates`).
- `steps`: the maximum number of search iterations (`cAIC4`'s `steps`, default `50`).
- `savedmodels`: how many of the best distinct scored models to retain in `result.saved`
  (`cAIC4`'s `numberOfSavedModels`); `1` (default) keeps only the selected model, `0` keeps all.
- `method`, `hessian`, `nboot`, `sigmapenalty`, `rng`: forwarded unchanged to every `caic` score.

# Returns
- A [`StepcaicResult`](@ref) carrying the selected score/model, the per-step `path`, and the
  resolved options.

# Throws
- `ArgumentError` for an unsupported `direction`, a forward/`both` run with no candidates, or
  `savedmodels < 0`.

# Example
```julia
m = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), sleepstudy; progress=false)
res = stepcaic(m, sleepstudy; direction=:forward, slopecandidates=[:days])  # grows the random slope
res.selected.caic
```
"""
function stepcaic(
    m::LinearMixedModel{T},
    data;
    direction::Symbol=:backward,
    groupcandidates::Vector{Symbol}=Symbol[],
    slopecandidates::Vector{Symbol}=Symbol[],
    maxslopes::Int=2,
    useacross::Bool=false,
    keep::Union{Nothing,_StatsModels.FormulaTerm}=nothing,
    selectcorrelation::Bool=false,
    allownointercept::Bool=false,
    steps::Int=50,
    savedmodels::Int=1,
    skipnonconverged::Bool=false,
    method::Symbol=:auto,
    hessian::Symbol=:analytic,
    nboot::Union{Int,Nothing}=nothing,
    sigmapenalty::Integer=1,
    rng::AbstractRNG=default_rng(),
) where {T}
    _validatestepcaic(
        direction, savedmodels, slopecandidates, groupcandidates, maxslopes, useacross
    )
    # Parse the `keep` formula fragment to the `RESpec` floor threaded into the backward search.
    keepspec = keep === nothing ? nothing : extractkeep(keep, data)

    # Quarantined formula parts + REML flag; candidates are refit with the input's objective.
    fixed = MMInternals.fixedterm(m)
    lhs = MMInternals.responseterm(m)
    reml = MMInternals.reml(m)
    score(model) = caic(
        model; method, hessian, nboot, sigmapenalty, rng
    )::CAICResult{T,LinearMixedModel{T}}
    refitcand(c) = fit(MixedModel, render(c, fixed, lhs), data; REML=reml, progress=false)
    terminalfit() = (tm=lm(_StatsModels.FormulaTerm(lhs, fixed), data); (tm, caic(tm)))
    gencands(spec, dir) =
        if dir === :forward
            forwardcandidates(
                spec;
                slopecandidates,
                groupcandidates,
                maxslopes,
                useacross,
                selectcorrelation,
            )
        else
            backwardcandidates(spec; keep=keepspec, selectcorrelation, allownointercept)
        end

    options = StepcaicOptions(
        direction,
        selectcorrelation,
        allownointercept,
        steps,
        savedmodels,
        skipnonconverged,
        groupcandidates,
        slopecandidates,
        maxslopes,
        useacross,
    )
    return _runstepcaic(
        T, m, score, refitcand, terminalfit, gencands, options; keep=keepspec
    )
end

# Flip the working direction (`R/stepcAIC.R`'s `ifelse(direction=="forward","backward","forward")`).
_flip(d::Symbol) = d === :forward ? :backward : :forward

# The model-family-agnostic driver shared by the `LinearMixedModel` and
# `GeneralizedLinearMixedModel` `stepcaic` methods, covering all three directions (`:backward`,
# `:forward`, `:both`) — the faithful port of `cAIC4`'s `stepCAIC` decision cascade
# (`R/stepcAIC.R:565–657`) plus the forward-terminal arc (`:435`) and the `lm`/`glm` backward
# terminal (§0.1). The greedy walk, the `mergeChanges` drop-original, the terminal arcs, and the
# `savedmodels` k-best accumulation are identical across families; four pieces differ and are
# injected as closures:
#   • `score`       — `model -> CAICResult{T,M}` with the family's forwarded scoring kwargs;
#   • `refitcand`   — `spec -> M`, rebuild+refit a candidate over `data` (REML for LMM, the GLM
#                     distribution family for GLMM);
#   • `terminalfit` — `() -> (termmodel, termresult)`, fit+score the no-random-effects `lm`/`glm`
#                     terminal (called only at the cnms single-direction node with no keep floor);
#   • `gencands`    — `(spec, dir) -> Vector{RESpec}`, the per-direction candidate enumeration
#                     (`forwardcandidates` with the forward kwargs when `dir === :forward`,
#                     `backwardcandidates` with the backward kwargs otherwise).
# `direction = :both` latches `dirWasBoth = true` and starts the working direction **forward**
# (`:389`), flipping it after each accepted or non-improving turn; `improvementinboth` and
# `equaltolaststep` are the cascade's alternation/plateau guards. The return type varies across the
# terminal branch (a `MixedModel` above it, the `GLM.jl` terminal at it); the driver is not a hot
# kernel, so this is accepted (docs/math/0008 §5.1).
function _runstepcaic(
    ::Type{T},
    m::M,
    score::F,
    refitcand::G,
    terminalfit::H,
    gencands::C,
    options::StepcaicOptions;
    keep::Union{Nothing,RESpec},
) where {T,M<:MixedModel,F,G,H,C}
    savedmodels = options.savedmodels
    skipnonconverged = options.skipnonconverged
    dirwasboth = options.direction === :both
    # Working direction: `both`/`forward` start forward (`:389`); `backward` is the §4.1 skeleton.
    workdir = options.direction in (:both, :forward) ? :forward : :backward
    improvementinboth = true
    equaltolaststep = false

    cur_spec = extract(m)
    cur_model = m
    cur_result = score(m)
    cAICofMod = cur_result.caic

    path = StepRecord{T}[]

    # `savedmodels` (`cAIC4` `numberOfSavedModels`): accumulate every distinct scored model across
    # the whole search, dedup by structure (`_savedkey`, the `duplicatedMers` analogue, keep-first),
    # then rank ascending and keep the best `nsave` at return. `0` keeps all; `1` (default) keeps
    # only the selected model, so the accumulation is skipped entirely (`remember!` is inert).
    nsave = savedmodels == 0 ? typemax(Int) : savedmodels
    savedkeys = String[]
    savedpool = CAICResult{T}[]
    remember!(key::String, r::CAICResult) = begin
        if savedmodels != 1 && !(key in savedkeys)
            push!(savedkeys, key)
            push!(savedpool, r)
        end
        return nothing
    end
    finalsaved(selkey::String, selres::CAICResult{T}) = begin
        savedmodels == 1 && return CAICResult{T}[selres]
        selkey in savedkeys || (push!(savedkeys, selkey); push!(savedpool, selres))
        order = sortperm([r.caic for r in savedpool]; alg=MergeSort)
        CAICResult{T}[savedpool[i] for i in order[1:min(nsave, length(order))]]
    end

    result(res, model, selkey) =
        StepcaicResult(res, model, path, finalsaved(selkey, res), options)

    stepsleft = options.steps
    while stepsleft > 0
        cands = gencands(cur_spec, workdir)

        # Forward-terminal arc (`:435`): forward enumeration exhausted → return the incumbent as
        # best, *before* scoring. Forward never descends to the `lm`/`glm` terminal (it only grows).
        # Fires regardless of `dirwasboth`, so a `both` run whose forward turn yields nothing stops.
        if workdir === :forward && isempty(cands)
            return result(cur_result, cur_model, _savedkey(cur_spec))
        end

        # Backward enumeration exhausted: at the cnms single-direction node with NO keep floor the
        # sole neighbour is the `lm`/`glm` terminal (§0.1) — score it as the candidate set (`allNA`,
        # a finite cAIC). Otherwise the neighbourhood is empty (`minCAIC == Inf`): a `both` run flips
        # direction (`:565–571` branch A), a non-`both` run stops keeping the incumbent.
        terminaldescent = false
        if workdir === :backward && isempty(cands)
            if keep === nothing && _totaldirections(cur_spec) == 1
                terminaldescent = true
            elseif dirwasboth
                workdir = _flip(workdir)
                improvementinboth = false
                continue
            else
                break
            end
        end

        stepsleft -= 1   # `R/stepcAIC.R:468` — one decrement per scoring iteration

        if terminaldescent
            termmodel, termresult = terminalfit()
            termcaic = termresult.caic
            remember!(_TERMINALKEY, termresult)
            accepted = termcaic <= cAICofMod
            push!(
                path,
                StepRecord{T}(
                    workdir,
                    cAICofMod,
                    ScoredCandidate{T}[ScoredCandidate{T}(
                        nothing, termcaic, termresult.dof
                    )],
                    1,
                    accepted,
                ),
            )
            # `allNA` + `≤`: accept-and-stop (branch B's `all(is.na(newSetup))` arc). Otherwise the
            # terminal does not improve: a `both` run with a prior successful turn flips once more
            # (branch F), else stop keeping the incumbent (branch G).
            accepted && return result(termresult, termmodel, _TERMINALKEY)
            if dirwasboth && improvementinboth
                workdir = _flip(workdir)
                improvementinboth = false
                continue
            else
                break
            end
        end

        # `mergeChanges` drop-original: discard every candidate equal to the current model before
        # scoring (the keep re-add / a no-op enlargement can reconstitute the unchanged incumbent).
        cands = RESpec[c for c in cands if c != cur_spec]
        if isempty(cands)
            # `minCAIC == Inf` (branch A): flip in `both`, else stop keeping the incumbent.
            if dirwasboth
                workdir = _flip(workdir)
                improvementinboth = false
                continue
            else
                break
            end
        end

        models = Vector{M}(undef, length(cands))
        results = Vector{CAICResult{T,M}}(undef, length(cands))
        caics = Vector{T}(undef, length(cands))
        for (i, c) in enumerate(cands)
            cm = refitcand(c)
            r = score(cm)
            models[i] = cm
            results[i] = r
            # `skipnonconverged` (the `calcNonOptimMod` analogue): a non-converged candidate is
            # excluded from the comparison — its effective cAIC is +Inf (the `NA` cAIC analogue,
            # never the argmin) and it is not retained in the `savedmodels` k-best.
            excluded = skipnonconverged && !MMInternals.converged(cm)
            caics[i] = excluded ? typemax(T) : r.caic
            excluded || remember!(_savedkey(c), r)
        end

        bestidx = argmin(caics)
        minCAIC = caics[bestidx]
        improves = minCAIC <= cAICofMod
        single = stepsleft == 0 || length(cands) == 1

        # `accepted` for the path record: the best candidate becomes (or would become) the incumbent
        # this step (branches B/C/D) vs. rejected (branches E/F/G). The mixed-candidate set never
        # carries the `lm`/`glm` terminal (`allNA`/`bestIsGLM` are false here), so branch B is unreachable
        # on this path — its terminal/keep-minimal arcs are the `terminaldescent`/single-candidate stops.
        accepted = improves && (!equaltolaststep || improvementinboth)
        push!(
            path,
            StepRecord{T}(
                workdir,
                cAICofMod,
                ScoredCandidate{T}[
                    ScoredCandidate{T}(cands[i], caics[i], results[i].dof) for
                    i in eachindex(cands)
                ],
                bestidx,
                accepted,
            ),
        )

        if improves && !equaltolaststep
            # Branch C: accept the improving (or first-tie) move. A tie latches `equaltolaststep`.
            minCAIC == cAICofMod && (equaltolaststep = true)
            single &&
                return result(results[bestidx], models[bestidx], _savedkey(cands[bestidx]))
            cur_spec = cands[bestidx]
            cur_model = models[bestidx]
            cur_result = results[bestidx]
            cAICofMod = minCAIC
            improvementinboth = true
            dirwasboth && (workdir = _flip(workdir))
        elseif improves && equaltolaststep && improvementinboth
            # Branch D: the plateau's second accepted move (consumes `improvementinboth`).
            cur_spec = cands[bestidx]
            cur_model = models[bestidx]
            cur_result = results[bestidx]
            cAICofMod = minCAIC
            improvementinboth = false
            dirwasboth && (workdir = _flip(workdir))
        elseif !improves && single && !dirwasboth
            # Branch E: no improvement and out of moves (non-`both`) — stop, keep incumbent.
            break
        elseif !improves && dirwasboth && improvementinboth
            # Branch F: a `both` non-improving turn after a prior success — flip and try once more.
            workdir = _flip(workdir)
            improvementinboth = false
        else
            # Branch G: stop, keep the incumbent.
            break
        end
    end

    return result(cur_result, cur_model, _savedkey(cur_spec))
end

"""
    stepcaic(m::GeneralizedLinearMixedModel, data; direction=:backward, groupcandidates=Symbol[],
             slopecandidates=Symbol[], maxslopes=2, useacross=false, keep=nothing,
             selectcorrelation=false, allownointercept=false, steps=50, savedmodels=1,
             method=:auto, nboot=nothing, rng=Random.default_rng()) -> StepcaicResult

Conditional stepwise random-effects selection for a non-Gaussian `GeneralizedLinearMixedModel` —
the GLMM branch of `cAIC4`'s `stepcAIC`, in the **backward**, **forward**, or **both** direction
(M4 §4.1–§4.2).

Identical in structure to the [`LinearMixedModel`](@ref stepcaic) method: each step enumerates the
random-effects neighbours one direction *smaller* ([`backwardcandidates`](@ref)) or *larger*
([`forwardcandidates`](@ref)), rebuilds and refits each over `data` with the model's GLM
**distribution family**, and scores it with [`caic`](@ref)'s GLMM path (the M3 bias correction).
The minimum-cAIC neighbour is accepted while it does not increase the cAIC (the `≤` rule);
`direction = :both` starts forward and alternates; a backward search bottoms out at the `glm`
terminal.

The scoring kwargs are the GLMM `caic` set — `method`/`nboot`/`rng` — and are forwarded
**unchanged** to every candidate score (the consistent-scoring requirement). The Gaussian-only
`hessian`/`sigmapenalty` kwargs are not accepted here.

# Arguments
- `m`: the fitted `GeneralizedLinearMixedModel` to search from.
- `data`: the data table candidates are refit over (required).
- `direction`: `:backward` (default), `:forward`, or `:both`.
- `groupcandidates`, `slopecandidates`, `maxslopes`, `useacross`: the forward enumeration inputs
  (see [`forwardcandidates`](@ref)); ignored for a backward run. A forward/`both` run requires at
  least one of `slopecandidates`/`groupcandidates` (or `useacross`), else `ArgumentError`.
- `keep`, `selectcorrelation`, `allownointercept`, `steps`, `savedmodels`: as in the
  `LinearMixedModel` method.
- `method`, `nboot`, `rng`: forwarded unchanged to every GLMM `caic` score.

# Returns
- A [`StepcaicResult`](@ref) carrying the selected score/model, the per-step `path`, and the
  resolved options.

# Throws
- `ArgumentError` for an unsupported `direction`, a forward/`both` run with no candidates, or a
  negative `savedmodels`.

# Example
```julia
m = fit(MixedModel, @formula(y ~ 1 + x + (1 | sub)), data, Poisson(); progress=false)
res = stepcaic(m, data; direction=:forward, groupcandidates=[:it])   # grows the crossed intercept
res.selected.caic
```
"""
function stepcaic(
    m::GeneralizedLinearMixedModel{T,D},
    data;
    direction::Symbol=:backward,
    groupcandidates::Vector{Symbol}=Symbol[],
    slopecandidates::Vector{Symbol}=Symbol[],
    maxslopes::Int=2,
    useacross::Bool=false,
    keep::Union{Nothing,_StatsModels.FormulaTerm}=nothing,
    selectcorrelation::Bool=false,
    allownointercept::Bool=false,
    steps::Int=50,
    savedmodels::Int=1,
    skipnonconverged::Bool=false,
    method::Symbol=:auto,
    nboot::Union{Int,Nothing}=nothing,
    rng::AbstractRNG=default_rng(),
) where {T,D}
    _validatestepcaic(
        direction, savedmodels, slopecandidates, groupcandidates, maxslopes, useacross
    )
    # Parse the `keep` formula fragment to the `RESpec` floor threaded into the backward search.
    keepspec = keep === nothing ? nothing : extractkeep(keep, data)

    # Quarantined formula parts + the GLM distribution family; candidates refit with that family.
    # The prior weights (binomial denominators nᵢ; empty for Poisson/Bernoulli) are reused on every
    # candidate refit and the terminal glm — without them a multi-trial Binomial candidate is a
    # different model (its trial counts are lost). `weights=`/`wts=` empty is the unweighted default,
    # so the Poisson/Bernoulli paths are unchanged.
    fixed = MMInternals.fixedterm(m)
    lhs = MMInternals.responseterm(m)
    dist = MMInternals.glmmdist(m)
    wts = MMInternals.glmmpriorweights(m)
    score(model) =
        caic(model; method, nboot, rng)::CAICResult{T,GeneralizedLinearMixedModel{T,D}}
    refitcand(c) =
        fit(MixedModel, render(c, fixed, lhs), data, dist; weights=wts, progress=false)
    terminalfit() = (
        tm=GLM.glm(_StatsModels.FormulaTerm(lhs, fixed), data, dist; wts=wts);
        (tm, caic(tm))
    )
    gencands(spec, dir) =
        if dir === :forward
            forwardcandidates(
                spec;
                slopecandidates,
                groupcandidates,
                maxslopes,
                useacross,
                selectcorrelation,
            )
        else
            backwardcandidates(spec; keep=keepspec, selectcorrelation, allownointercept)
        end

    options = StepcaicOptions(
        direction,
        selectcorrelation,
        allownointercept,
        steps,
        savedmodels,
        skipnonconverged,
        groupcandidates,
        slopecandidates,
        maxslopes,
        useacross,
    )
    return _runstepcaic(
        T, m, score, refitcand, terminalfit, gencands, options; keep=keepspec
    )
end
