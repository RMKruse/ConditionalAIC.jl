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

# `checkREs` (`R/helperfuns_stepcAIC.R:354`): per candidate, drop NA/empty terms, then within
# each grouping sort each direction-vector and drop duplicate vectors. The source's
# `checkHierarchicalOrder` step only fires when one grouping keeps **>1** distinct term, which a
# backward single-term-per-grouping candidate never produces — it is a forward/nesting concern
# (docs/math/0008 §3) and is unreachable here.
function _checkres(cand::Vector{_RawTerm})
    live = _Term[]
    for (nm, d) in cand
        d === nothing && continue          # NA marker
        isempty(d) && continue             # all-NA / empty term (any(!is.na(·)) == false)
        push!(live, (nm, d))               # `d` narrowed to Vector{String} here
    end
    result = _Term[]
    for nm in unique(Symbol[t[1] for t in live])
        seen = Vector{String}()
        uniq = Vector{Vector{String}}()
        for (tn, d) in live
            tn == nm || continue
            sd = sort(d)
            sd in uniq || push!(uniq, sd)
        end
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
