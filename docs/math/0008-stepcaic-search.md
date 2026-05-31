# 0008 — `stepcaic`: conditional stepwise random-effects search (M4)

This note is the §7 step-1 "state the math/spec" record for milestone **M4** (`stepcaic`,
the **Search** verb of `CONTEXT.md`). It pins, before the corresponding Julia code is written,
the candidate space, the enumeration rules, and the greedy controller of `cAIC4`'s `stepcAIC` —
**restricted to the random-effects search for `(g)lmer`/`lm`/`glm` objects** (the `gamm4`
smooth-term route is milestone M5 and out of scope here).

Unlike the scoring notes (`0002`–`0007`), this is a **combinatorial/control-flow** spec, not an
estimand: the cAIC of each candidate is computed by the already-validated `caic` of M2/M3 plus the
`lm`/`glm` terminal of ADR-0006. The contribution of M4 is *which* models are generated and *how*
the greedy walk decides among them — both ported faithfully from `cAIC4`.

> **STATUS: skeleton.** Section headers and the source map are pinned; the precise notation
> under each is filled in before implementation (TDD per CLAUDE.md §7). Do not implement ahead of
> a filled section.

**Ground-truth sources** (read from source, not memory — memory record *verify-caic4-against-source*):
- `cAIC4` **v1.1**: `R/stepcAIC.R` (the driver + decision cascade, lines 169–693) and
  `R/helperfuns_stepcAIC.R` (`getComponents`, `backwardStep`, `forwardStep`, `makeBackward`,
  `makeForward`, `makeFormula`, `cnmsConverter`, `mergeChanges`, `calculateAllCAICs`,
  `removeUncor`, `removeNoInt`, `checkREs`, `interpret.random`, `duplicatedMers`,
  `isNested`/`allNestSubs`).
- `R/cAIC.R:201–240` — the `(g)lm` terminal scoring reproduced here (see ADR-0006).
- Säfken, Rügamer, Kneib & Greven (2021), *JSS* 99(8), §on stepwise selection.

Where any source disagrees, **`cAIC4` is ground truth** (CLAUDE.md §2).

Companion records: **ADR-0006** (`GLM.jl` dep, `lm`/`glm` terminal, `CAICResult` widening) and the
three **DECISIONS.md** entries dated 2026-05-30 (RE-only scope; faithful-cascade + near-tie path
divergence; `data`-required refit).

---

## 0. Object, scope, and the cAIC assembly per candidate

- Supported inputs: `LinearMixedModel`, `GeneralizedLinearMixedModel` (Poisson / Bernoulli /
  multi-trial Binomial, per M3), and the `lm`/`glm` terminal as a search node.
- The **fixed-effects part is held constant** on every candidate (DECISIONS 2026-05-30, RE-only).
- Each candidate model `m'` is scored as `cAIC(m') = −2·ℓ_cond(m') + 2·ρ(m')` with `ℓ_cond`/`ρ`
  from the M2/M3 `caic` (singular candidates internally reduce-and-refit, carrying `refit`), or the
  ADR-0006 closed form at the `lm`/`glm` terminal.
- All candidates in a run are scored with **identical** forwarded kwargs
  (`method`/`hessian`/`nboot`/`rng`/`sigmapenalty`) — the consistent-scoring requirement of
  `CONTEXT.md` (*Selection*). Serial execution; a single forwarded `rng` makes bootstrap-GLMM
  candidates reproducible.

### 0.1 The `lm`/`glm` terminal node

A backward step removes random-effects terms one at a time; removing the *last* `|` term leaves a
fixed-effects-only model, which `MixedModels.jl` (v5.5.1) cannot represent — `fit(MixedModel, …)`
requires at least one RE term. The search therefore bottoms out at a plain `GLM.jl` `lm`/`glm`
fit, scored as a search node by `caic(::RegressionModel)` (`src/scoring.jl`, ADR-0006). This
reproduces `cAIC4`'s own terminal: in `R/cAIC.R:201–240` the `c("glm","lm")` branch scores the
fixed-effects fit directly rather than via the `merMod` bias-correction path.

A terminal candidate is the pair `(m, r)` where `m` is the fitted `GLM` object and
`r = caic(m)` is its `CAICResult`, with provenance `method = :terminal`, `bsource = :na`,
`reducedmodel = nothing`, `refit = false` (a fixed-effects fit is never singular in the RE sense
and is never reduced-and-refit). The conditional AIC at the terminal is the *unconditional* AIC of
the `(g)lm` — there are no random effects to condition on, so `ℓ_cond` collapses to the ordinary
log-likelihood at the fitted mean `μ̂ = Xβ̂`:

```
ρ_terminal   = rank(X) + 1                       # cAIC4's (g)lm df: # estimated params (β̂ and, for
                                                 #   Gaussian, σ̂); rank(X) = length(coef(m))
ℓ_cond       = Σᵢ log f(yᵢ; μ̂ᵢ, φ)               # family log-density at μ̂, summed over observations
cAIC_terminal = −2·ℓ_cond + 2·ρ_terminal
```

The `+1` in `ρ` counts the scale/extra parameter that the `(g)lm` estimates beyond `β`: for the
Gaussian `lm`, `σ̂`; for the one-parameter Poisson/Bernoulli/Binomial families, the dispersion
slot `cAIC4` still adds (matching its `rank + 1`). The family densities reuse the M3 `Loglik`
kernels at `μ̂`:

| Terminal family            | `μ̂`                       | `ℓ_cond` kernel                          | Scale `φ`                    |
|----------------------------|---------------------------|------------------------------------------|------------------------------|
| Gaussian `lm`              | `Xβ̂` (`predict`)          | `condloglik(y, μ̂, σ̂)` (∑ `dnorm`)        | `σ̂ = √(deviance(lm)/n)` (MLE)|
| Poisson `glm` (log)        | `exp(Xβ̂)`                 | `condloglik_poisson(y, μ̂)` (∑ `dpois`)   | —                            |
| Bernoulli `glm` (logit)    | `logistic(Xβ̂)`           | `condloglik_bernoulli(y, μ̂)` (∑ `dbinom`, n≡1) | —                      |
| multi-trial Binomial `glm` | `logistic(Xβ̂)`           | `condloglik_binomial(y, μ̂, n)` (corrected) | trial counts `nᵢ` = prior wts|

The Gaussian `σ̂` is the **MLE** rescaling `cAIC4` applies, `summary$sigma·√((n−p)/n) = √(RSS/n) =
√(deviance(lm)/n)` (deviance of an `lm` is the residual sum of squares). The multi-trial Binomial
row is the documented **deviation**: `cAIC4`'s binomial `getcondLL` returns `−∞` for `nᵢ > 1`
(it evaluates `dbinom` on the success proportion with `size = |unique(y)|−1`), so the terminal
reuses the corrected `condloglik_binomial` at the true trial counts, exactly as the M3 GLMM
binomial path does (DECISIONS 2026-05-29 / 2026-05-30). Bernoulli (`nᵢ ≡ 1`) does not deviate.

An *unsupported* terminal family (anything other than the four rows above — e.g. a Gamma `glm`)
fails loudly with `ArgumentError` rather than returning a silently-wrong number.

## 1. The RE-structure spec (`cnms` analogue) and `getComponents`

The fit-independent representation enumeration operates on. The Julia analogue of `cAIC4`'s `cnms`
(a named list `grouping → c(term-labels)`): each grouping factor mapped to its ordered list of RE
directions (`"(Intercept)"` + slope variables) plus a per-group correlated/uncorrelated flag.

### 1.1 The `RESpec` struct

A `RESpec` is an ordered list of `REGroup`s, one per random-effects term in the formula (matching
the order of the `|` terms in `m.formula.rhs`). Each `REGroup` is the Julia analogue of one
`(grouping, cnms-entry)` pair of `cAIC4`'s `object@cnms`, plus the correlated flag MixedModels
encodes structurally (`RandomEffectsTerm` vs `ZeroCorr`) that `lme4`'s `cnms` does not carry:

```
REGroup
  grouping   :: Symbol           # the grouping-factor name (cnms list name; e.g. :subj)
  directions :: Vector{String}   # cnms-style column labels, intercept first:
                                 #   "(Intercept)" present ⇔ the term carries a random intercept;
                                 #   the remaining entries are slope variable names ("days", …).
                                 #   A no-intercept term (0 + x | g) omits "(Intercept)".
  correlated :: Bool             # true for (… | g) (RandomEffectsTerm), false for zerocorr(… | g)

RESpec
  groups :: Vector{REGroup}      # ordered, one per RE term
```

`directions` is the faithful `cnms` analogue: `cAIC4`'s `object@cnms[[g]]` is exactly this list of
column labels (with `"(Intercept)"` for the random intercept). Both fields are concrete
(`Vector{String}`/`Symbol`/`Bool`), so `REGroup`/`RESpec` are type-stable (CLAUDE.md §4). Two specs
compare **by value** (`==` defined field-wise over `groups`), which is the round-trip oracle below.

### 1.2 Extraction — `extract(::MixedModel) -> RESpec` (`getComponents` analogue)

From `m.formula` (the structural truth, not the fit-mutated `m.reterms` whose `λ` a singular fit has
zeroed). For each RE term in `m.formula.rhs` (every tuple element after the leading fixed-effects
`MatrixTerm`):

- unwrap `ZeroCorr` to its inner `RandomEffectsTerm`, recording `correlated = false` (else `true`);
- `grouping` ← the inner term's `rhs` `CategoricalTerm` symbol (`ret.rhs.sym`);
- `directions` ← map over the lhs `MatrixTerm` directions (`ret.lhs.terms`): `InterceptTerm{true}`
  ↦ `"(Intercept)"`; `InterceptTerm{false}` (a suppressed intercept, `0 + …`) contributes nothing;
  any other term ↦ its variable name (`string(only(termvars(t)))`).

Interpreting these MixedModels/StatsModels term types touches the upstream term representation and
is therefore a **`mm_internals.jl` quarantine** concern (CLAUDE.md §3): the accessor `reterminfo(m)`
returns the raw `Vector{(grouping, directions, correlated)}`, shape-asserted against the pinned
version; `extract` only wraps those tuples into `REGroup`/`RESpec` (no internal access). The response
term and fixed-effects `MatrixTerm` are read by the companion quarantine accessors `responseterm(m)`
(`m.formula.lhs`) and `fixedterm(m)` (`m.formula.rhs[1]`), supplied to `render` as `lhs`/`fixed`.

### 1.3 Rendering — `render(::RESpec, fixed, lhs) -> FormulaTerm` (`cnmsConverter` + `makeFormula`)

Spec → `FormulaTerm`, with the fixed-effects part reattached unchanged, handed to the public `fit`.
Built on the **public** StatsModels/MixedModels formula API only (no internals): `term`, the term
algebra `+`/`|`, the exported `zerocorr`, and the `FormulaTerm` constructor. Per `REGroup`, the
`cnmsConverter` rule reconstructs the directions term:

- each `"(Intercept)"` ↦ `term(1)`; each slope label `s` ↦ `term(Symbol(s))`;
- if no `"(Intercept)"` is present, append `term(0)` (the suppressed-intercept marker `0`), exactly
  as `cnmsConverter` appends `"0"`;
- sum the direction terms and group: `lhsexpr | term(grouping)`, wrapping in `zerocorr(…)` when
  `correlated == false`.

The RE terms are summed and appended to the unchanged `fixed` term; `FormulaTerm(lhs, fixed + ΣRE)`
is the rendered formula. An empty spec (`isempty(groups)`) cannot be rendered to a `MixedModel`
formula (MixedModels requires ≥ 1 `|` term — the no-RE case is the `lm`/`glm` terminal of §0.1, not
a `RESpec`) and raises `ArgumentError`.

### 1.4 The round-trip invariant

`extract` and `render` are mutually inverse **on structure** (not on `λ`/`θ`, which are re-estimated
by the refit). The oracle, on a model `m` with `s = extract(m)`:

```
fit(MixedModel, render(s, fixedterm(m), responseterm(m)), data)   # refit the rendered formula
extract(refit) == s                                               # structurally identical
```

asserted (`==`, by value) on `sleepstudy` (`1 + days + (1 + days | subj)`, correlated slope +
intercept), `Pastes` (`1 + (1 | batch) + (1 | cask)`, crossed intercept-only), the `zerocorr`
variant (`correlated = false` preserved), and the no-intercept variant (`0 + days | subj`,
`directions == ["days"]`). This is the §6 Level-1-style structure equality for the representation
layer — no cAIC value is computed; the candidate enumeration of §2–§3 builds on it.

## 2. Backward enumeration — `backwardStep` / `makeBackward`

`backwardcandidates(spec; keep, selectcorrelation, allownointercept) -> Vector{RESpec}` is the
faithful port of `cAIC4`'s `backwardStep` (`R/helperfuns_stepcAIC.R:93–201`), the only branch of
`makeBackward` (`:805–822`) in RE-only scope. It returns the random-effects neighbours of `spec` that
are **one direction smaller**. The `lm`/`glm` terminal (§0.1) is *not* a `RESpec` (it has no `|`
term, and [`render`](@ref cAIC.render) rejects an empty spec), so it is **not** an element of the
returned vector: an **empty** `Vector{RESpec}` is the terminal/exhausted signal — the faithful image
of `cAIC4`'s `NA` return (`:104`) and of an empty `listOfAllCombs` (the degenerate cases below).

### 2.1 The `cnms` bridge (representation)

`cAIC4` enumerates over `object@cnms`, a *named list* `grouping → c(labels)` whose names may
**repeat**: `lme4` represents an *uncorrelated* term `(1 + x || g)` as the two entries
`g = "(Intercept)"`, `g = "x"`, carrying no correlated flag (doc §1.1). A `RESpec` instead carries
that flag on one `REGroup` per term. The two representations are bridged by a single map, applied
identically on the R (fixture) and Julia sides so set-equality is well posed:

```
cnmsform(spec) :: Vector{(grouping::Symbol, directions::Vector{String})}   # the cAIC4 `cnms`
  for g in spec.groups:
    g.correlated   → one entry  (g.grouping, g.directions)
    ¬g.correlated  → one single-direction entry (g.grouping, [d]) per d in g.directions
```

The inverse `respec(cnmsform)` regroups by `grouping` preserving first-appearance order: a grouping
occurring **once** → `REGroup(grouping, dirs, correlated = true)`; a grouping occurring in **several**
single-direction entries → one `REGroup(grouping, concat(dirs), correlated = false)`. The algorithm
of §2.2–§2.4 runs entirely on `cnmsform`; `backwardcandidates` is `respec ∘ backwardStep ∘ cnmsform`.

The **canonical encoding** used as the Level-1 oracle (§2.5) is `cnmsform` itself, rendered to a
sorted multiset of term-strings `"grouping:sorted(directions)"` — no explicit correlated flag, since
the term *structure* already distinguishes `(1+x|g)` (one two-label term) from `(1+x||g)` (two
one-label terms). Two candidates are equal **iff** their canonical encodings are equal.

### 2.2 The drop set (`backwardStep:107–189`)

Let `cnms = cnmsform(spec)` and `L = Σₜ length(directionsₜ)` (the total number of RE directions).

- **Terminal guard (`:96–105`).** If `L == 1` (a single random intercept overall): with no `keep`,
  return `[]` (`NA` → terminal); with `keep`, return `[spec]` (the model is its own sole neighbour —
  observed: `(1|g)` keep `~(1|g)` → `{(1|g)}`).
- **Per-direction drop (`:107–151`).** Group the `cnms` entries by name (`split`). For a grouping
  whose entries hold a direction-vector of length `k`:
  - `k == 1` (single-direction term) → dropping it yields the marker `NA` (the whole term is removed);
  - `k > 1` → produce `k` reduced vectors, the `i`-th being the direction-vector **with its `i`-th
    label removed** (`d[[i]] <- d[[i]][-i]`). *Faithful quirk:* `cAIC4` indexes each of the `k`
    same-name copies by its position, so for an *uncorrelated* split (each copy length 1) the `i`-th
    copy is indexed `[-i]`, which for `i` past its length is a no-op — reproducing the observed
    degenerate outputs (uncorrelated `(1|g)+(0+x|g)` → `{}` by default, `{(x|g)}` under
    `selectcorrelation ∧ allownointercept`). The port mirrors `[-i]` exactly.
- **Candidate assembly (`:171–184`).** For each reduced/`NA` term of grouping `gᵢ`, the candidate is
  `cnms` with **all** entries named `gᵢ` replaced by that single reduced term (`append(cnms[names ≠
  gᵢ], reduced)`). An empty (`length 0`) reduced term drops the candidate at the `notempty` filter
  (`:180`) — the source of the empty-set degeneracy above.

### 2.3 `keep` (`backwardStep:110–164`, `interpret.random`)

`keep` is an RE formula fragment (`~(1|subj)`), parsed by `interpret.random` to a named list
`grouping → c("(Intercept)"|"0", slopes…)` — the directions that must **survive**. Faithful port:
before the per-direction drop, each kept direction is removed from the droppable set for its grouping
(`indRem <- unlist(temp) != unlist(keep)`, `:114–123`); after, the kept directions are re-appended so
every candidate retains them (`:153–164`). Net effect (observed): `keep` pins those directions —
`pastes` keep `~(1|batch)` → `{(1|batch)+(1|cask), (1|batch)}` (batch never dropped, the original is
retained as a neighbour). `keep` is supplied to `backwardcandidates` as a parsed `RESpec` floor
(§5); the intersection is computed on `cnmsform`.

### 2.4 Filters and dedup (order is load-bearing, `:185–198`)

Applied in exactly this order — `cAIC4` does **not** re-dedup after the two filters:

1. **`checkREs` (`:354–384`)** per candidate: drop `NULL`/all-`NA` terms; within each grouping, sort
   each direction-vector, drop duplicate vectors, and enforce hierarchical order
   (`checkHierarchicalOrder`) when a grouping has `>1` surviving term.
2. **ordered-name dedup (`:188–189`)**: sort each candidate's terms by grouping name, then drop a
   candidate iff *both* it and its name-vector duplicate an earlier one.
3. **`removeUncor` (`:596–638`, skipped iff `selectcorrelation`)**: drop a candidate iff some grouping
   has `>1` term among which at least one carries `"(Intercept)"` and at least one does not — i.e. the
   candidate encodes a correlated→uncorrelated split. (No-op for single-term-per-name candidates, so
   inert on the correlated representative structures; it bites only on already-split starts.)
4. **`removeNoInt` (`:643–667`, skipped iff `allownointercept`)**: from each candidate remove every
   **term** lacking `"(Intercept)"`, then drop candidates left empty. (This — not a per-candidate
   reject — is why `(1+days|subj)` default yields only `{(1|subj)}`: the `(days|subj)` candidate's
   sole term is intercept-less and is stripped, emptying it.)

`backwardcandidates` returns `respec` of the surviving candidates, de-duplicated by canonical
encoding (the post-filter duplicates `cAIC4` may retain are irrelevant to the set oracle; candidate
**multiplicity/ordering** for the driver is a §4 concern).

### 2.5 The Level-1 set-equality oracle

For each representative `(spec, selectcorrelation, allownointercept, keep)` scenario, with `C_R` the
candidate set produced by `cAIC4:::backwardStep` and `C_jl = backwardcandidates(spec; …)`:

```
{ canonical(c) : c ∈ C_R }  ==  { canonical(c) : c ∈ C_jl }        # Set equality, no model fit
```

Scenarios (and their observed `cAIC4` outputs), pinned in `test/generate_fixtures_stepcaic.R`:

| spec | flags | `cAIC4` candidate set |
|:-----|:------|:----------------------|
| `(1+days\|subj)` | default | `{(1\|subj)}` |
| `(1+days\|subj)` | `allownointercept` | `{(days\|subj), (1\|subj)}` |
| `(1+days\|subj)` | `selectcorrelation` | `{(1\|subj)}` |
| `(1\|batch)+(1\|cask)` | default | `{(1\|cask), (1\|batch)}` |
| `(1\|g)` | default | `{}` (terminal) |
| `(1\|g)` | `keep ~(1\|g)` | `{(1\|g)}` |
| `(1+x+y\|g)` | default | `{(1+y\|g), (1+x\|g)}` |
| `(1+x+y\|g)` | `allownointercept` | `{(x+y\|g), (1+y\|g), (1+x\|g)}` |
| `(1\|g)+(0+x\|g)` (uncorr.) | default | `{}` |
| `(1\|g)+(0+x\|g)` (uncorr.) | `selectcorrelation, allownointercept` | `{(x\|g)}` |
| `(1+days\|subj)+(1\|item)` | default | `{(1+days\|subj), (1\|subj)+(1\|item), (1\|item)}` |
| `(1+days\|subj)+(1\|item)` | `allownointercept` | `{(1+days\|subj), (days\|subj)+(1\|item), (1\|subj)+(1\|item)}` |
| `(1\|batch)+(1\|cask)` | `keep ~(1\|batch)` | `{(1\|batch)+(1\|cask), (1\|batch)}` |

This is the §6 Level-1 structure-equality for the backward enumerator; no cAIC value is computed.

## 3. Forward enumeration — `forwardStep` / `makeForward`

`forwardcandidates(spec; slopecandidates, groupcandidates, maxslopes, useacross, selectcorrelation)
-> Vector{RESpec}` is the faithful port of `cAIC4`'s `forwardStep` (`R/helperfuns_stepcAIC.R:516–590`),
the RE branch of `makeForward` (`:929–968`). It returns the random-effects neighbours **one direction
larger** than `spec`: each adds a single slope to an existing term, a single new term to an existing
grouping, or a new grouping factor. It runs on the same `cnms` bridge as backward (§2.1):
`respec ∘ forwardStep ∘ cnmsform`, de-duplicated by the canonical encoding (§2.5). An **empty**
`Vector{RESpec}` is the terminal/exhausted signal — the image of `cAIC4`'s `return(NULL)` (`:556`,
`:584`). Forward has **no** `keep` and **no** `allownointercept` (unlike backward): `forwardStep`
takes neither, so intercept-less candidates (e.g. `(days|item)` over a new grouping) are admissible.

`maxslopes` is the user-facing cap (`cAIC4` `numberOfPermissibleSlopes`, default `2`); the slope-combo
size `nrOfCombs = maxslopes + 1` adds the intercept slot, exactly as the driver's redefine
(`R/stepcAIC.R:302`). The port applies the `+1` internally so `forwardcandidates(spec)` and the R
`forwardStep(cnms, nrOfCombs = maxslopes + 1)` are compared like-for-like.

### 3.1 The add set (`forwardStep:525–548`)

Let `cnms = cnmsform(spec)`.

- **`allslopes`** — the slope labels eligible to appear in an added combination:
  - `useacross == false` (default): `[slopecandidates…, "(Intercept)"]` — only the externally supplied
    slope variables plus the intercept;
  - `useacross == true`: `unique([unlist(cnms)…, slopecandidates…], incomparables = "(Intercept)")` —
    every direction already in the model plus the candidates, letting an existing slope migrate to
    another grouping. *Faithful quirk:* `cAIC4`'s `unique(…, "(Intercept)")` passes `"(Intercept)"`
    as `incomparables`, so intercept labels are never de-duplicated and may appear several times in
    `allslopes`; the duplicate single-intercept combos this produces are removed by the combo-dedup
    (`:543`) and the final candidate dedup (`:553`), so the **set** is unchanged. The port may keep a
    single intercept (proof of set-equivalence: duplicate intercepts yield only duplicate candidates,
    all removed downstream).
- **`allgroups`** — `unique([names(cnms)…, groupcandidates…])`: existing groupings plus the candidates.
- **`allslopecombs`** — all size-`i` combinations of `allslopes` for `i ∈ 1:nrOfCombs` with
  `i ≤ length(allslopes)` (`combn`), dropping any combo with a repeated label (`:543`).
- **cross product (`:545–548`)** — every `(group, combo)` pair (`rep(combs, each = #groups)` named by
  cycling `allgroups`); each candidate is `append(cnms, (group ↦ combo))`.

### 3.2 `checkREs` **with** `checkHierarchicalOrder` (`:354–384`, `:317–345`)

Each candidate passes through `checkREs`: drop `NULL`/all-`NA` terms; per grouping, sort each
direction-vector and drop duplicate vectors; **and**, when a grouping keeps `>1` distinct term, apply
`checkHierarchicalOrder`. The hierarchical step (sort terms by length desc; for each non-minimal term
`t` with `length(t) > 1` and `length(t) > min-length`, delete every surviving term equal to a *proper
sub-combination* of `t`, via `allCombn`) collapses a redundant smaller term into a larger one — e.g.
adding the size-2 combo `{(Intercept), days}` to `(1|item)` yields the two terms `(1|item)` and
`(1+days|item)`, and `checkHierarchicalOrder` deletes the sub-term `(1|item)`, leaving `(1+days|item)`.

> This is the step the backward port omits: the comment in `src/stepcaic.jl` notes
> `checkHierarchicalOrder` is unreachable for a backward single-term-per-grouping candidate. Forward
> reaches it (a size-≥2 combo added to an existing term), so the forward port carries the full
> `checkREs`. The shared `_checkres` is extended (not forked) to apply the hierarchical step.

### 3.3 Filters and dedup (order load-bearing, `:550–563`)

In exactly this order:

1. **same-grouping reject (`:550–552`, skipped iff `selectcorrelation`)**: keep a candidate only if
   `length(unique(names)) == length` — i.e. no grouping appears in more than one term. By default a
   forward step never produces an uncorrelated split; under `selectcorrelation` the split survives
   (e.g. adding `days` to `(1|subj)` yields both `(1+days|subj)` and `(1|subj)+(0+days|subj)`).
2. **dedup (`:553`)** and **drop-original (`:554`)**: discard a candidate equal to `lapply(cnms, sort)`
   (the structure the move was a no-op on — e.g. adding an intercept where one already exists).
3. **null-drop, order-by-name, ordered-name dedup (`:558–560`)** — as backward (§2.4 step 2).
4. **`removeUncor` (`:563`, skipped iff `selectcorrelation`)** — as §2.4 step 3.

### 3.4 The one-direction-larger restriction (`:566–582`)

The defining forward constraint. Drop a candidate if **any** of its terms `t` over grouping `g` is
more than one direction larger than the current model at `g`:

- `g ∉ names(cnms)` (a newly added grouping): drop if `length(t) > 1` — a new grouping enters with a
  single direction only;
- `g ∈ names(cnms)` (existing): drop if `length(t) > length(cnms[[g]]) + 1` — an existing term grows
  by at most one direction.

If no candidate survives, return empty (`:584`).

### 3.5 Nesting expansion — `allNestSubs` + `isNested` (`R/stepcAIC.R:210–223`)

A `groupcandidates` entry written as a nesting expression `"a/b"` expands, **before** `forwardStep`,
to its sub-groupings. `allNestSubs("a/b")` (`R/helperfuns_stepcAIC.R:417–423`) is the pure string
expansion via `findbars(~ (1 | a/b))`: `"a/b" ↦ ["b:a", "a"]`, `"a/b/c" ↦ ["c:b:a", "b:a", "a"]` (the
interaction grouping innermost-first, then the outer factor). Ported as `_allnestsubs(s) ->
Vector{String}` (pure; Level-1 fixtured against `allNestSubs`).

The validity gate `isNested(data[,a], data[,b])` (`lme4`/`reformulas::isNested`, `R/stepcAIC.R:216`)
admits the expansion only when the factors are genuinely nested, warn-and-dropping otherwise. Its
predicate — *every level of `f1` co-occurs with at most one level of `f2`* (`f1` nested within `f2`) —
is ported as `isnested(f1, f2) -> Bool` over two raw factor vectors (no data-table dependency;
Level-1 testable against `reformulas::isNested`).

> **Deferred:** the `expandnesting(groupcandidates, data)` glue that pulls `data` columns to run
> `isnested` + the warn-and-drop loop is **driver-side** (`R/stepcAIC.R:210–223`, *not* `forwardStep`)
> and forces the `stepcaic`-receives-`data` interface decision. It is deferred to the §4 driver, where
> that interface is designed; #39 ships the two pure ingredients (`_allnestsubs`, `isnested`).

### 3.6 The Level-1 set-equality oracle

As §2.5: for each `(spec, slopecandidates, groupcandidates, maxslopes, useacross, selectcorrelation)`
scenario, with `C_R = forwardStep(cnms, nrOfCombs = maxslopes + 1, …)` and
`C_jl = forwardcandidates(spec; …)`:

```
{ canonical(c) : c ∈ C_R }  ==  { canonical(c) : c ∈ C_jl }        # Set equality, no model fit
```

Scenarios (observed `cAIC4` outputs), pinned in the forward fixture generator:

| spec | slopes / groups / flags | `cAIC4` candidate set |
|:-----|:------------------------|:----------------------|
| `(1\|subj)` | group `item` | `{(1\|item)+(1\|subj)}` |
| `(1\|subj)` | slope `days` | `{(1+days\|subj)}` |
| `(1\|subj)` | slope `days`, group `item` | `{(1\|item)+(1\|subj), (days\|item)+(1\|subj), (1+days\|subj)}` |
| `(1\|subj)` | slope `days`, `selectcorrelation` | `{(1+days\|subj), (1\|subj)+(0+days\|subj)}` |
| `(1+days\|subj)` | group `item`, `useacross` | `{(1\|item)+(1+days\|subj), (days\|item)+(1+days\|subj)}` |
| `(1\|subj)+(1\|item)` | slope `days` | `{(1+days\|item)+(1\|subj), (1\|item)+(1+days\|subj)}` |
| `(1\|subj)` | slopes `x,y,z`, `maxslopes=1` | `{(1+x\|subj), (1+y\|subj), (1+z\|subj)}` |
| `(1\|subj)` | slopes `x,y`, `maxslopes=2` | `{(1+x\|subj), (1+y\|subj)}` (size-3 combo capped by one-larger) |
| `(1+days\|subj)` | slope `x` | `{}` (no candidate ≤ one larger without `useacross`) |
| `(1\|g)` | none | `{}` (terminal) |

This is the §6 Level-1 structure-equality for the forward enumerator; no cAIC value is computed. The
nesting ingredients carry their own Level-1 fixtures: `_allnestsubs` against `allNestSubs`, `isnested`
against `reformulas::isNested`.

## 4. The greedy controller — `stepcAIC` driver (lines 410–659)

Faithful port (DECISIONS 2026-05-30). Pin precisely:
- **Entry step**: score the input model; backward on an `lm`/`glm` is an error
  (`R/stepcAIC.R:336`); call-consistency checks (forward needs candidates).
- **Per-step**: generate the candidate set for the current `direction`; if empty (forward) →
  return current as best. Score all; pick `minCAIC`.
- **Acceptance / plateau / both-alternation**: the `≤` rule, `equalToLastStep` guard,
  `improvementInBoth` flip; `direction="both"` starts **forward** (`R/stepcAIC.R:389`) and flips
  after each accepted/plateau step.
- **Stop predicates**: `minCAIC==Inf`; reached `lm`/`glm`; reached the `keep`-minimal model; a
  single candidate; `steps` exhausted.
- **Singular carry-forward**: keep the as-fit (possibly singular) candidate as the incumbent, track
  `refit` (`R/stepcAIC.R:323–324`).

### 4.1 The **backward** greedy loop (the skeleton subset, #40)

The skeleton implements the `direction = :backward`, `dirWasBoth = false` branch of the driver —
the only branch where `forwardStep`/`mergeChanges`/`both`-alternation are unreachable. Let
`score(·)` be the consistent per-candidate cAIC (§0): the M2/M3 `caic` on a `MixedModel`, the
ADR-0006 terminal `caic` on the `lm`/`glm`. Let `incumbent` be the current best `(spec, model,
result)` and `cAICofMod = result.caic`. The loop (faithful to the extracted `stepcAIC` body,
backward arcs only):

```
incumbent ← (extract(m), m, score(m))          # entry: score the input
equalToLastStep ← false
repeat (at most `steps` times):
    cands ← backwardcandidates(incumbent.spec; keep, selectcorrelation, allownointercept)
    if isempty(cands):                          # §2 backwardStep NA / terminal signal
        if keep is nothing and incumbent is a single random intercept:
            cands ← {lm/glm terminal node}          # §0.1 terminal descent (only with NO keep floor)
        else:  break (stop; keep incumbent)         # exhausted neighbourhood / keep floor reached
    cands ← [c for c in cands if c ≠ incumbent.spec]   # mergeChanges drop-original: backwardStep's keep
    if isempty(cands):  break                          #   re-add can reconstitute the unchanged model;
                                                       #   an emptied neighbourhood is the minCAIC==Inf arc
    scored   ← [ (c, score(render+fit c)) for c in cands ]
    bestidx  ← argmin caic over scored          # which.min(aicTab$caic)
    minCAIC  ← scored[bestidx].caic             # (Inf if cands empty — handled above)
    record the step (direction, cAICofMod, scored, bestidx, accepted)
    if minCAIC ≤ cAICofMod:                      # the ≤ acceptance rule (:223–244)
        accept: incumbent ← scored[bestidx]; cAICofMod ← minCAIC
        if minCAIC == cAICofMod (a plateau move):  equalToLastStep ← true   # tie guard (:232)
        if the accepted move is the lm/glm terminal,
           or equals the keep-minimal model,
           or steps exhausted, or a single candidate:  break (stop)         # :223–235
        # else continue from the new incumbent
    else:                                        # minCAIC > cAICofMod (:253–258)
        break (stop; keep incumbent)
return StepcaicResult(incumbent, path, options)
```

The mapping to the extracted source decision cascade (`stepcAIC` body, backward `!dirWasBoth`
arcs): `minCAIC == Inf` (empty, non-terminal) → stop keep incumbent (:217–221); `minCAIC ≤
cAICofMod` with the move terminal / keep-minimal → accept-and-stop (:223–229); `minCAIC ≤
cAICofMod` otherwise → accept, advance unless `steps==0 | #cands==1` (:230–244); `minCAIC >
cAICofMod` → stop keep incumbent (:253–258). The `equalToLastStep` guard prevents an infinite
plateau loop when consecutive moves tie. **Singular carry-forward**: a scored candidate's `model`
is the fit *as given* (possibly singular); `caic` reduces-and-refits internally for the *score*,
recording `refit` on the `CAICResult` (the incumbent carries the as-fit model, mirroring
`R/stepcAIC.R:323–324`). `keep` is supplied as a parsed `RESpec` floor (§5), forwarded to
`backwardcandidates` and used for the keep-minimal stop test.

**Family dispatch.** The loop above is model-family-agnostic; only `score(·)`, the candidate
refit, and the terminal fit differ between a Gaussian `LinearMixedModel` and a non-Gaussian
`GeneralizedLinearMixedModel`. The two public `stepcaic` methods resolve those three pieces and
delegate to one shared driver:

| piece          | `LinearMixedModel`                                  | `GeneralizedLinearMixedModel`                          |
|----------------|-----------------------------------------------------|--------------------------------------------------------|
| `score(model)` | `caic(model; method, hessian, nboot, sigmapenalty, rng)` | `caic(model; method, nboot, rng)`                 |
| candidate refit| `fit(MixedModel, render(c), data; REML)`            | `fit(MixedModel, render(c), data, family)`             |
| terminal fit   | `lm(y ~ fixed, data)` → `caic`                      | `glm(y ~ fixed, data, family)` → `caic`                |

The GLMM scoring-kwarg set is **smaller**: `caic(::GeneralizedLinearMixedModel)` has no Gaussian
`hessian`/`sigmapenalty` arguments (the Greven–Kneib Hessian and the σ-penalty are LMM-only), so
the GLMM method neither accepts nor forwards them. Both methods forward their kwarg set
**unchanged** to every candidate (the consistent-scoring requirement). The terminal fit uses the
model's GLM distribution family (`m.resp.d`), so the GLMM terminal is the family `glm`, scored by
the same ADR-0006 `caic(::RegressionModel)` terminal as the Gaussian `lm`.

### 4.2 The **forward** and **`both`** arcs (#41)

The full controller adds the `direction ∈ {:forward, :both}` branches — the `dirWasBoth` /
`improvementInBoth` / `equalToLastStep` alternation cascade of `R/stepcAIC.R:565–657` plus the
forward-terminal arc (`:435`). The loop is identical in skeleton to §4.1; the differences are: the
per-step candidate set is the **forward** enumeration ([`forwardcandidates`](@ref), §3) when the
working direction is forward, and the decision part flips the working direction in `both` mode.

**Initial direction and the `both` flag (`:379–389`).** `direction = :both` sets the latch
`dirWasBoth = true` and the *working* direction starts **forward** (`:389`:
`ifelse(direction ∈ {both,forward}, forward, backward)`); `direction = :forward` runs forward with
`dirWasBoth = false`; `:backward` is §4.1. Two more latches carry across iterations: `improvementInBoth`
(init `true`) and `equalToLastStep` (init `false`). Let `flip(d)` swap forward↔backward.

**Call-consistency (`:347–359`).** A forward or `both` run needs something to add: it errors with
`ArgumentError` unless at least one of `slopecandidates`, `groupcandidates` is non-empty (or
`useacross` is set). A backward run ignores any candidate variables (`:361–364`). `fixEfCandidates`
is out of scope (fixed effects held constant, §0).

**Forward-terminal arc (`:435–453`).** *Before* scoring, if the working direction is forward and the
forward enumeration is empty (`all candidates null`), the search returns the current incumbent as
best immediately — forward never descends to the `lm`/`glm` terminal (it only grows). This fires
regardless of `dirWasBoth`, so a `both` run whose forward sub-step yields nothing also returns here.

**The decision cascade (`:565–657`).** After scoring the merged candidate set, let
`minCAIC = min caic` (`Inf` if the set is empty after the `mergeChanges` drop-original), `ncands` its
size, and the predicates: `bestIsGLM` (the argmin candidate is the `lm`/`glm` terminal — backward
only), `allNA` (the whole candidate set *is* the terminal node — the §0.1 lm-descent, backward only),
`keepMin` (the argmin equals the keep-minimal model). The branches, in order:

| # | Predicate | Action |
|:--|:----------|:-------|
| A | `minCAIC == Inf` | `dirWasBoth`: `flip`, `improvementInBoth←false`, continue. else: **stop**, keep incumbent. |
| B | `minCAIC ≤ cAICofMod` ∧ [ (`¬dirWasBoth ∧ backward ∧ bestIsGLM`) ∨ (`¬dirWasBoth ∧ backward ∧ keepMin`) ∨ `allNA` ] | **stop**, accept the argmin (terminal/keep-minimal reached). |
| C | `minCAIC ≤ cAICofMod` ∧ `¬allNA` ∧ `¬equalToLastStep` | if `minCAIC == cAICofMod`: `equalToLastStep←true`. if `steps==0 ∨ ncands==1`: **stop**, accept argmin. else: accept (advance incumbent), `improvementInBoth←true`, `dirWasBoth`: `flip`. |
| D | `minCAIC ≤ cAICofMod` ∧ `equalToLastStep` ∧ `improvementInBoth` | accept (advance incumbent), `improvementInBoth←false`, `dirWasBoth`: `flip`. (the plateau second move) |
| E | `minCAIC > cAICofMod` ∧ (`steps==0 ∨ ncands==1`) ∧ `¬dirWasBoth` | **stop**, keep incumbent. |
| F | `minCAIC ≥ cAICofMod` ∧ `dirWasBoth` ∧ `improvementInBoth` | `flip`, `improvementInBoth←false`, continue. (the `both` no-improvement turn) |
| G | otherwise | **stop**, keep incumbent. |

`steps` is decremented once per iteration (`:468`); the `steps==0` tests in C/E are the
budget-exhausted stop. The `equalToLastStep` latch (set in C on a tie, consumed in D) caps a plateau
at two consecutive equal-cAIC moves, preventing an infinite `both` oscillation between a forward and a
backward representation of the same structure. `improvementInBoth` records whether the *previous*
`both` turn advanced: branch F lets a `both` run flip once on a non-improving turn (to try the other
direction) but not twice in a row, so two consecutive non-improving turns (one per direction) stop the
search (F fails on the second → G).

> *Worked `both` trace (`pastes_both_cask`):* start forward from `(1|batch)`, group candidate `cask`.
> **Iter 1** (forward): the sole candidate `(1|batch)+(1|cask)` scores worse → branch F flips to
> backward, `improvementInBoth←false`. **Iter 2** (backward): single intercept, no keep → the `lm`
> terminal `strength~1` is scored (worse) → F fails (`improvementInBoth` is false) → branch G stops,
> keeping `(1|batch)`. Two non-improving turns, one per direction, terminate cleanly.

**Family dispatch** is exactly §4.1's: the forward arc reuses the same three injected closures
(`score`, candidate refit, terminal fit). The candidate-generation closure dispatches on the working
direction — [`forwardcandidates`](@ref) with the forward kwargs (`slopecandidates`, `groupcandidates`,
`maxslopes`, `useacross`, `selectcorrelation`) when forward, [`backwardcandidates`](@ref) with
(`keep`, `selectcorrelation`, `allownointercept`) when backward.

## 5. Options, result, and provenance

- Options (idiomatic Julia names; `→` cAIC4 origin): `direction`, `groupcandidates`/`slopecandidates`,
  `maxslopes`→`numberOfPermissibleSlopes`, `useacross`→`allowUseAcross`,
  `selectcorrelation`→`allowCorrelationSel`, `allownointercept`→`allowNoIntercept`, `keep`,
  `steps`, `savedmodels`→`numberOfSavedModels`, `skipnonconverged`→`calcNonOptimMod`
  (default `false`, i.e. include non-converged, matching `calcNonOptimMod=TRUE`), plus forwarded
  scoring kwargs. Parallelism (`numCores`) deferred — serial.
- `keep` is supplied as a `FormulaTerm` RE fragment (mirrors `keep$random`), parsed to a `RESpec`
  floor.
- **Result** `StepcaicResult{T}`: the selected `CAICResult` (`M<:RegressionModel`, ADR-0006), the
  k-best `saved::Vector{CAICResult{T}}`, the structured `path` (per-step direction + scored
  candidates + move; the *Search path* of `CONTEXT.md`, replacing `cAIC4`'s printed `trace`), and
  the resolved options for provenance.

### 5.1 The result types (the skeleton subset, #40)

The search returns a `StepcaicResult{T,M}` carrying the *selected* model, its score, the full
search `path`, and the resolved options. The `path` is the structured analogue of `cAIC4`'s
printed `trace`: one `StepRecord{T}` per loop iteration, each holding **every** candidate scored
that step (the *Search path* of `CONTEXT.md`), not only the accepted one.

```
ScoredCandidate{T}                       # one scored neighbour of a step
  spec      :: Union{RESpec,Nothing}     #   the candidate RE structure; `nothing` = lm/glm terminal node (§0.1)
  caic      :: T                         #   its conditional AIC (the `score` of §4.1)
  dof       :: T                         #   the bias-corrected effective df ρ that AIC was penalised by (the `df` of the trace)

StepRecord{T}                            # one greedy step
  direction      :: Symbol               #   :backward (the skeleton's only value)
  incumbentcaic  :: T                    #   cAICofMod at the start of the step
  candidates     :: Vector{ScoredCandidate{T}}   #   every neighbour scored this step
  bestindex      :: Int                  #   argmin index into `candidates` (0 if the neighbourhood was empty)
  accepted       :: Bool                 #   whether the best candidate was accepted (minCAIC ≤ incumbentcaic)

StepcaicResult{T,M<:RegressionModel}
  selected   :: CAICResult{T,M}          #   the score of the selected model (M2/M3 or terminal)
  model      :: M                        #   the selected fitted model (a `MixedModel` or the `lm`/`glm` terminal)
  path       :: Vector{StepRecord{T}}    #   the per-step search trace, in order
  saved      :: Vector{CAICResult{T}}    #   the ranked k-best scores (`savedmodels`)
  options    :: StepcaicOptions          #   the resolved options for provenance
```

`ScoredCandidate.spec` is a small `Union{RESpec,Nothing}` (type-stable), the `nothing` reserving
the terminal slot. The `caic`/`dof` pair is the `cAIC`/`df` the trace prints per candidate: `dof` is
the candidate's own `CAICResult.dof` (the bias-corrected ρ it was penalised by), recorded as the
driver already holds it — no re-derivation — so `(caic, dof)` together expose the full penalised
score the greedy rule acted on. `StepcaicResult` is parametric on the *selected* model type `M` (a `MixedModel`
when the search stops above the terminal, the `GLM.jl` `TableRegressionModel` at the terminal); the
driver is not a hot kernel, so the across-path return-type variation is acceptable.

### 5.2 The `savedmodels` k-best set (`numberOfSavedModels`)

`savedmodels` (→ `cAIC4` `numberOfSavedModels`) governs `result.saved`. The driver **accumulates
every distinct scored model across the whole search**, deduplicated by random-effects structure
(the `_savedkey` canonical `cnms` term-multiset for a `RESpec`; a reserved sentinel for the
`lm`/`glm` terminal — the `duplicatedMers` analogue, **keep-first**), and at return ranks them by
cAIC **ascending** and keeps the best `savedmodels` (`0` ⇒ all; the default `1` ⇒ the selected
model only, and the accumulation is skipped). The element type is the `M`-erased `CAICResult{T}`,
not `CAICResult{T,M}`, so one ranked list can hold both the `MixedModel` candidates and the terminal
when both rank among the best.

This **unifies** `cAIC4`'s split return — `cAIC4` hands back the selected model as `finalModel`
separately from the runner-ups in `additionalModels` (which it post-trims with `additionalModels[-1]`
to drop the global minimum, since that *is* `finalModel`). The set of models and their cAICs is
identical; the *packaging* differs (one ranked vector with the selected at `saved[1]` vs the
`finalModel` + `additionalModels` pair). Recorded as a deliberate shape choice in DECISIONS.md
(2026-05-31). The numbers are the ground-truth check: `c(bestCAIC, attr(additionalModels,"cAICs"))`
equals `[s.caic for s in result.saved]` within the Level-2 band.

**`StepcaicOptions` field list (#41).** The resolved options retained for provenance:
`direction::Symbol`, `selectcorrelation::Bool`, `allownointercept::Bool`, `steps::Int`,
`savedmodels::Int`, and the forward-enumeration fields `groupcandidates::Vector{Symbol}`,
`slopecandidates::Vector{Symbol}`, `maxslopes::Int`, `useacross::Bool` (all concrete, so the struct
is type-stable). `keep` is *not* retained on the options (it is a `RESpec` floor threaded into the
backward enumeration, not a scalar provenance field).

## 6. Validation plan (two-level)

- **Level 1 (fit-independent)**: fixture `cAIC4`'s `backwardStep`/`forwardStep` candidate `cnms`
  lists for representative structures and every flag setting; assert the `RESpec` enumeration is
  **set-equal** (structure equality, no model fit). Tight; validates §2–§3 directly.
- **Level 2 (end-to-end)**: run `stepcaic` vs `cAIC4` `stepcAIC` on `Pastes` (crossed, backward →
  `lm`), `sleepstudy` (slope+intercept backward), a forward and a `both` example, and a GLMM
  example; assert selected RE structure + `bestCAIC` within the per-method atol band. Full `path`
  asserted only on well-separated scenarios (DECISIONS 2026-05-30, near-tie divergence).

> *(to fill: the fixture generators `generate_fixtures_stepcaic.{R,jl}`, the embedded-sample
> strategy, and the per-scenario atol bands.)*
