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

> *(to fill: notation for the candidate node — a fitted model + its `CAICResult` + provenance.)*

## 1. The RE-structure spec (`cnms` analogue) and `getComponents`

The fit-independent representation enumeration operates on. The Julia analogue of `cAIC4`'s `cnms`
(a named list `grouping → c(term-labels)`): each grouping factor mapped to its ordered list of RE
directions (`"(Intercept)"` / `"0"` + slope variables) plus a per-group correlated/uncorrelated
flag.

- **Extraction** (`getComponents` analogue): from `m.formula` (the structural truth), interpreting
  the MixedModels RE-term types. This interpretation touches MixedModels' term representation and
  is therefore a **`mm_internals.jl` quarantine** concern — add the touched types/accessors to the
  internal-access table before coding (CLAUDE.md §3).
- **Rendering** (`cnmsConverter` + `makeFormula` analogue): spec → `FormulaTerm`, with the fixed
  part reattached unchanged, handed to the public `fit`.

> *(to fill: the `RESpec` struct fields and the extraction/render maps; the round-trip invariant
> `render(extract(m)) ≈ m.formula` on structure.)*

## 2. Backward enumeration — `backwardStep` / `makeBackward`

Drop one RE direction at a time per grouping factor; drop a grouping factor **whole** when it has a
single surviving direction (→ the term is removed; all-removed → the `lm`/`glm` terminal). Filters:
`removeUncor` (no correlated→uncorrelated reduction unless `selectcorrelation`), `removeNoInt`
(every RE keeps its intercept unless `allownointercept`), `keep`-floor enforcement, and
de-duplication (`checkREs`, ordered-name dedup).

> *(to fill: the exact drop set as a function of `cnms`, the `keep` intersection, and the
> filter predicates — the Level-1 candidate-set equality target.)*

## 3. Forward enumeration — `forwardStep` / `makeForward`

Add a new grouping factor (`groupcandidates`) or a new slope (`slopecandidates`) to an existing
group; `maxslopes` (`numberOfPermissibleSlopes`, +1 for the intercept) caps slopes per group;
`useacross` (`allowUseAcross`) lets a slope migrate across groups. Candidates are restricted to RE
structures **exactly one direction larger** than the current one (`R/helperfuns_stepcAIC.R:566–582`),
then passed through the same `removeUncor`/dedup filters. Nesting candidates (`a/b`) expand via
`allNestSubs` after an `isNested` check (warn-and-drop if not actually nested).

> *(to fill: the add set, the "one larger" restriction, and the nesting expansion — the Level-1
> candidate-set equality target for forward.)*

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
