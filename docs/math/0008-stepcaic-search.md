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
