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

> *(to fill: the decision cascade as an explicit predicate table mapping `(minCAIC vs cAICofMod,
> dirWasBoth, direction, improvementInBoth, equalToLastStep, terminal?, keep-min?, steps,
> #candidates)` → `(accept?, newdir, stop?)`, transcribed from lines 565–657.)*

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

> *(to fill: the `StepcaicResult{T}` and per-step `StepRecord{T}` field lists; the k-best
> dedup rule, `cAIC4`'s `duplicatedMers`.)*

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
