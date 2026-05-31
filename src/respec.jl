# The RE-structure spec (`cnms` analogue) and its extract/render round-trip — the
# fit-independent representation the `stepcaic` search (M4) enumerates over. Included
# directly into the `cAIC` module. `extract` reads the structural truth `m.formula`
# through the [`cAIC.MMInternals`](@ref) quarantine; `render` rebuilds a formula from the
# **public** StatsModels/MixedModels term API (no internals). See
# `docs/math/0008-stepcaic-search.md` §1.

using MixedModels: zerocorr

# StatsModels' public term constructors (`term`, `FormulaTerm`), reached through the
# already-direct `GLM` dependency's loaded copy — the same module MixedModels builds
# formulas with — so `render` adds no separate StatsModels dependency (mirrors the
# `TableRegressionModel` alias in `cAIC.jl`).
const _StatsModels = GLM.StatsModels

"""
    REGroup(grouping, directions, correlated)

One grouping factor's random-effects structure — the Julia analogue of one
`(name, cnms-entry)` pair of `cAIC4`'s `object@cnms`, plus the correlated flag MixedModels
encodes structurally (`RandomEffectsTerm` vs `zerocorr`).

- `grouping::Symbol` — the grouping-factor name (e.g. `:subj`).
- `directions::Vector{String}` — the `cnms`-style column labels, intercept first:
  `"(Intercept)"` is present iff the term carries a random intercept; the remaining entries
  are slope variable names. A no-intercept term (`0 + x | g`) omits `"(Intercept)"`.
- `correlated::Bool` — `true` for `(… | g)`, `false` for `zerocorr(… | g)`.

Part of the internal `stepcaic` representation ([`RESpec`](@ref)); compared **by value**.
"""
struct REGroup
    grouping::Symbol
    directions::Vector{String}
    correlated::Bool
end

"""
    RESpec(groups::Vector{REGroup})

The fit-independent random-effects structure of a candidate — the `cAIC4` `cnms` analogue
(CONTEXT.md *RE-structure spec*). An ordered list of [`REGroup`](@ref)s, one per
random-effects term of a formula. The `stepcaic` search enumerates neighbours by pure
add/drop transforms on a `RESpec` and renders it back to a formula only at fit time
([`render`](@ref cAIC.render)); [`extract`](@ref cAIC.extract) is the inverse read.

Compared **by value** (`==`, field-wise over `groups`): structural equality, the round-trip
oracle of `docs/math/0008-stepcaic-search.md` §1.4.
"""
struct RESpec
    groups::Vector{REGroup}
end

function Base.:(==)(a::REGroup, b::REGroup)
    a.grouping == b.grouping && a.directions == b.directions && a.correlated == b.correlated
end
Base.:(==)(a::RESpec, b::RESpec) = a.groups == b.groups

"""
    extract(m::MixedModel) -> RESpec

Read a fitted model's random-effects structure into a [`RESpec`](@ref) — the `cAIC4`
`getComponents` analogue. Reads `m.formula` (the structural truth, not the fit-mutated
`m.reterms`) through the [`cAIC.MMInternals`](@ref) quarantine and wraps the per-term
`(grouping, directions, correlated)` tuples into [`REGroup`](@ref)s.

# Example
```julia
m = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), sleepstudy)
spec = extract(m)
# spec.groups == [REGroup(:subj, ["(Intercept)", "days"], true)]
```
"""
function extract(m::MixedModel)
    groups = REGroup[
        REGroup(grouping, directions, correlated) for
        (grouping, directions, correlated) in MMInternals.reterminfo(m)
    ]
    return RESpec(groups)
end

"""
    extractkeep(keep::FormulaTerm, data) -> RESpec

Parse a `keep` formula fragment into the [`RESpec`](@ref) floor the backward [`stepcaic`](@ref)
search must not drop below — the Julia analogue of `cAIC4`'s `keep\$random` (`interpret.random`).
`keep` is a `FormulaTerm` whose right-hand side carries the random-effects bars to pin (e.g.
`@formula(y ~ (1 | batch))`); its response and any fixed-effects terms are ignored. The bars are
schema-applied against `data` (through the [`cAIC.MMInternals`](@ref) quarantine) and wrapped into
[`REGroup`](@ref)s, exactly as [`extract`](@ref cAIC.extract) does for a fitted model.

# Throws
- `ArgumentError` if `keep` carries no random-effects term — a `keep` floor with nothing to pin is
  a usage error (pass `keep=nothing` for no floor).
"""
function extractkeep(keep, data)
    groups = REGroup[
        REGroup(grouping, directions, correlated) for
        (grouping, directions, correlated) in MMInternals.reterminfo(keep, data)
    ]
    isempty(groups) && throw(
        ArgumentError(
            "keep formula carries no random-effects term to pin; pass keep=nothing for no floor",
        ),
    )
    return RESpec(groups)
end

"""
    render(spec::RESpec, fixed, lhs) -> FormulaTerm

Rebuild a model formula from a [`RESpec`](@ref) — the `cAIC4` `cnmsConverter` + `makeFormula`
analogue. The unchanged fixed-effects term `fixed` and response `lhs` (from
[`fixedterm`](@ref cAIC.MMInternals.fixedterm) / [`responseterm`](@ref
cAIC.MMInternals.responseterm)) are reattached, and each group is rendered via the **public**
term API: each `"(Intercept)"` ↦ `term(1)`, each slope `s` ↦ `term(Symbol(s))`, with a
trailing `term(0)` appended when no intercept is present (`cnmsConverter`'s `"0"`); the
directions are grouped `… | term(grouping)` and wrapped in `zerocorr` when uncorrelated.

The result is accepted by `fit(MixedModel, render(…), data)`; the round-trip
`extract(refit(render(extract(m)))) == extract(m)` holds (doc 0008 §1.4).

# Throws
- `ArgumentError` if `spec` is empty — a no-RE formula is the `lm`/`glm` terminal (§0.1), not
  a `MixedModel` formula (`MixedModels` requires ≥ 1 `|` term).
"""
function render(spec::RESpec, fixed, lhs)
    isempty(spec.groups) && throw(
        ArgumentError(
            "cannot render an empty RESpec: a no-random-effects formula is the lm/glm " *
            "terminal, not a MixedModel formula (MixedModels requires ≥ 1 `|` term).",
        ),
    )
    rhs = fixed + _renderre(first(spec.groups))
    for g in Iterators.drop(spec.groups, 1)
        rhs = rhs + _renderre(g)
    end
    return _StatsModels.FormulaTerm(lhs, rhs)
end

# Render one `REGroup` into a RE term via the public term API (`cnmsConverter`): intercept
# ↦ `term(1)`, slopes ↦ `term(sym)`, trailing `term(0)` when no intercept; grouped with `|`
# and wrapped in `zerocorr` when uncorrelated.
function _renderre(g::REGroup)
    hasintercept = "(Intercept)" in g.directions
    dirterm = hasintercept ? _StatsModels.term(1) : _StatsModels.term(0)
    for d in g.directions
        d == "(Intercept)" && continue
        dirterm = dirterm + _StatsModels.term(Symbol(d))
    end
    reterm = dirterm | _StatsModels.term(g.grouping)
    return g.correlated ? reterm : zerocorr(reterm)
end
