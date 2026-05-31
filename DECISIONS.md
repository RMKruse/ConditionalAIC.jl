# DECISIONS.md — Decision Log

Dated entries for every place where `cAIC.jl` legitimately diverges from `cAIC4`,
with the justified tolerance or behaviour. See CLAUDE.md §10. Architectural
decisions (as opposed to `cAIC4`-divergences) live in `docs/adr/`.

---

## 2026-05-31 — `stepcaic` `skipnonconverged`: convergence signal is the optimizer return code, not `lme4`'s gradient/Hessian check; no Level-2 fixture

**Status:** accepted (design — #43). Milestone M4; option `skipnonconverged` (the `cAIC4`
`calcNonOptimMod` analogue, default `false` ⇒ include non-converged, matching
`calcNonOptimMod=TRUE`).

`cAIC4`'s `calculateAllCAICs` excludes a candidate from the comparison (returns an `NA` cAIC)
when `calcNonOptimMod=FALSE` **and** the fit raised a convergence code — `lme4`'s
`m@optinfo$conv$lme4$code`, a *rich* post-hoc check (scaled gradient norm + Hessian
positive-definiteness against tolerances). `MixedModels.jl` exposes no such check; the only
convergence signal it carries is the **optimizer return code** `m.optsum.returnvalue`.
`cAIC.jl` therefore defines `converged(m) := returnvalue ∉ {:FAILURE, :INVALID_ARGS,
:OUT_OF_MEMORY, :FORCED_STOP, :MAXEVAL_REACHED, :MAXTIME_REACHED}` (mirroring `MixedModels`'
own `_NLOPT_FAILURE_MODES`), and `skipnonconverged=true` gives a non-converged candidate an
effective cAIC of `+Inf` (the `NA`-for-comparison analogue) and drops it from the
`savedmodels` k-best.

**The divergence.** The two ecosystems will flag *different* candidates as non-converged:
`lme4` can flag a numerically-optimal fit whose gradient check trips its tolerance, while
`MixedModels` reports success; conversely a fit that exhausts `MixedModels`' evaluation budget
(`:MAXEVAL_REACHED`) need not raise an `lme4` code. The *selection mechanism* is identical
(exclude-from-comparison); the *set of excluded candidates* is not guaranteed to match. A
singular fit is **not** treated as non-converged (it is a first-class supported case, CLAUDE.md
§9; `MixedModels` returns a success code with `λ` on the boundary).

**Validation consequence.** Because deterministic non-convergence is not reproducible *identically*
across optimizers, there is **no Level-2 `cAIC4` fixture** for this flag. It is validated by (1) a
unit test of `converged` (a converged fit ⇒ `true`; an evaluation-budget-truncated fit ⇒ `false`),
(2) an inert-case test (when every candidate converges, `skipnonconverged=true` reproduces the
default run exactly), and (3) a mechanism test driving the real greedy controller with one candidate
whose return code is tainted to a failure mode — asserting it is excluded from both the selection and
the saved set. See `docs/math/0008-stepcaic-search.md` §5.

---

## 2026-05-31 — GLMM `stepcaic` backward-to-`glm`-terminal scenario: per-scenario Level-2 band (`atol = 1e-2`), measured

**What this records.** The `glmm_poisson_terminal` driver scenario (`stepcaic_driver_level2.h5`, #42)
validates a GLMM backward search **descending to and scoring the `glm` terminal**: a single random
intercept Poisson GLMM `y ~ x + (1 | g)` whose only backward neighbour is the no-RE `glm` (§0.1).
`cAIC4`'s `stepCAIC` scores that terminal `glm(y ~ x, poisson)` (≈ 842.97), rejects it, and keeps
`(1 | g)` — the Poisson analogue of the Gaussian `sleepstudy_int` scenario.

**Two anchors, two bands.** The scored **terminal** candidate matches `cAIC4`'s `glm`-terminal cAIC
to ≈1e-2 *and* equals the project's own (Level-2-validated) `caic(::TableRegressionModel{<:GeneralizedLinearModel})`
exactly — a deterministic Poisson IRLS solve with no dispersion σ̂, so no Gaussian σ̂ divergence
(entry 2026-05-31, terminal). The kept **incumbent** GLMM score is the only piece needing a wider band:
`selected.caic` = 725.4593 (MixedModels) vs `bestCAIC` = 725.4668 (lme4), a measured discrepancy of
**7.57e-3** (relative **1.04e-5**). This is a pure lme4↔MixedModels Laplace-fit discrepancy — both fit
the same conditional model but reach slightly different θ̂, and the Chen–Stein df (ρ ≈ 18.04 over 20
groups) reads that θ̂-dependent penalty. The single-grouping 20-level fit legitimately diverges more
than the crossed-2RE `glmm_poisson_keep` (9.6e-4); per CLAUDE §6 / §10 the **measured 7.57e-3 is the
fit-discrepancy bound**, so this scenario's incumbent anchor uses **`atol = 1e-2`** (the terminal
anchor stays tight). The decision is unambiguous regardless: the terminal sits ≈117 cAIC units above
the incumbent, far outside any fit band, so the gate still discriminates the keep-vs-descend decision.

---

## 2026-05-31 — `stepcaic` on a `GeneralizedLinearMixedModel`: reused GLMM Level-2 band and the smaller scoring-kwarg set

**What this records.** The backward `stepcaic` driver now dispatches on model family
(`LinearMixedModel` / `GeneralizedLinearMixedModel`) through one shared core (`_runstepcaic`); only
the score closure, the candidate refit, and the terminal fit differ (docs/math/0008 §4.1). Two
points are worth pinning.

**1 — The GLMM scoring-kwarg set is smaller, by design.** `caic(::GeneralizedLinearMixedModel)`
takes only `method`/`nboot`/`rng` — it has no Gaussian `hessian`/`sigmapenalty` arguments (the
Greven–Kneib Hessian and the σ-penalty are LMM-only). The GLMM `stepcaic` method therefore neither
accepts nor forwards those two kwargs; it forwards its `{method, nboot, rng}` set unchanged to every
candidate. This is not a divergence from `cAIC4` (whose `stepCAIC` has a single interface) but a
faithful consequence of the project's family-split `caic` surface. The threading is gated by a
deterministic test: `stepcaic(m, data; nboot=5)` must raise `ArgumentError` *through* the forwarded
GLMM `caic` (nboot without `method=:bootstrap`), proving the kwarg reaches the score.

**2 — Level-2 band reused (`atol = 1e-3`), measured.** The GLMM keep-incumbent driver scenario
(`glmm_poisson_keep`, a crossed 2-RE Poisson, fixture `stepcaic_driver_level2.h5`) anchors
`selected.caic ≈ cAIC4 bestCAIC` within **`atol = 1e-3`** — the same GLMM end-to-end band as the M3
cases (entries 2026-05-29 / 2026-05-30). The measured lme4↔MixedModels discrepancy on this exact
shared data is **9.6e-4**, inside the band; the nearest rejected drop sits ≈9 cAIC units above the
incumbent, far outside it, so the gate still discriminates the keep decision. Per CLAUDE §6 the band
is the fit-discrepancy bound, not a loosened tolerance. The Chen–Stein df is non-singular here.

---

## 2026-05-31 — `stepcaic` `savedmodels` k-best: one ranked list vs `cAIC4`'s split `finalModel` + `additionalModels` return

**What diverges.** `cAIC4`'s `stepCAIC` returns the `numberOfSavedModels` best models as **two
pieces**: the selected model in `finalModel`, and the runner-ups in `additionalModels` (with their
cAICs in `attr(., "cAICs")`). Internally it accumulates every step's scored candidates, dedups by
structure (`duplicatedMers`), keeps the top-k by cAIC, then **drops the global minimum from
`additionalModels`** (`additionalModels[-1]`) because that minimum *is* `finalModel`. `cAIC.jl`'s
`StepcaicResult.saved` instead returns **one ranked vector** — the same distinct top-k models,
cAIC-ascending, with the selected model at `saved[1]` — i.e. `{finalModel} ∪ additionalModels`
reunified into a single ordered list.

**Why this is not a numerical divergence.** The *set* of saved models and their cAIC values is
identical to `cAIC4`'s; only the packaging differs (a Julia API choice — a self-contained ranked
list is more natural than a selected/runner-up split, and the `M`-erased `Vector{CAICResult{T}}`
element type lets the `lm`/`glm` terminal sit in the same list as the `MixedModel` candidates). The
two conventions `0 ⇒ keep all` and `1 ⇒ selected only` match `cAIC4` (`numberOfSavedModels == 0 →
Inf`; `== 1 → additionalModels NULL`).

**Validation.** Level-2 fixture `pastes_saved2` (`test/generate_fixtures_stepcaic_driver.R`,
`numberOfSavedModels = 2`) stores `savedcaics = c(bestCAIC, attr(additionalModels, "cAICs"))` —
the reunified ranked set `[301.4828311, 314.2642667]`, both `lmerMod`. The driver test asserts
`[s.caic for s in result.saved]` equals it within the Level-2 band (`atol = 1e-3`) and is sorted
ascending with `saved[1] == selected`. The `k = 3` set additionally pulls in the `lm` terminal
(`314.2727` in `cAIC4`), excluded from the anchored test because that terminal carries the
`glm`-dispersion σ̂ divergence of the entry below (and is numerically degenerate with the singular
`(1|cask)` fit under the project's lm/MLE σ̂).

---

## 2026-05-31 — `stepcaic` backward terminal: the Gaussian σ̂ convention diverges from `cAIC4`'s `stepCAIC` `glm`-dispersion terminal

**Status:** accepted (measured). Milestone M4 (#40); math spec
`docs/math/0008-stepcaic-search.md §0.1`; tests `test/stepcaic_driver_tests.jl`.

When a backward search reaches a single random-effects direction, the only smaller neighbour is
the no-random-effects terminal (§0.1). `cAIC.jl` scores this terminal as a `GLM.jl` **`lm`**, with
the Gaussian σ̂ the **MLE** rescaling `√(RSS/n) = √(deviance(lm)/n)` — reproducing `cAIC4`'s *own*
`cAIC.lm` (`R/cAIC.R`, the `c("glm","lm")` branch) to machine precision (DECISIONS 2026-05-30, the
terminal Level-2 band; ADR-0006).

**The divergence.** `cAIC4`'s `stepCAIC` does **not** route its backward terminal through that
`lm` path. Its `makeBackward` constructs the terminal as a **`glm(…, family = gaussian)`** (the
returned `finalModel` carries class `c("glm","lm")`), and `cAIC.glm` evaluates the conditional
log-likelihood at the **dispersion** σ̂ `√(RSS/(n−p)) = √(deviance/df.residual)` rather than the MLE.
With `df = rank + 1` identical on both paths, the two terminals differ only by that σ̂ convention:
on a non-singular synthetic scenario (`y ~ 1 + x + (1|g)`, no true group effect, `n = 120`, `p = 2`)
`cAIC4`'s `stepCAIC` `bestCAIC = 329.1304` (glm/dispersion) versus `cAIC(lm) = 329.1135`
(lm/MLE) — a **0.017** gap, well outside the terminal Level-2 band (`atol = 1e-3`). `cAIC4` is thus
*internally inconsistent*: `cAIC.lm` and `stepCAIC`'s glm-gaussian terminal disagree on the same
fixed-effects fit.

**Resolution.** `cAIC.jl` keeps the `lm`/MLE terminal — it is the documented ADR-0006 choice, it
matches `cAIC4`'s `cAIC.lm`, and it is consistent with how every other `caic` path estimates σ̂
(the MLE). This is **not** a tolerance to loosen. **Validation consequence:** a backward search
that *descends to and accepts* the terminal is anchored at Level-2 on the **structural** decision
(`cAIC4`'s `finalModel` has class `c("glm","lm")` ⇒ `cAIC.jl` returns a `TableRegressionModel`) and
numerically on the project's own `caic(lm)` (internal consistency, itself Level-2-validated against
`cAIC.lm`); it is **not** anchored on `stepCAIC`'s `bestCAIC`, which carries the glm-dispersion σ̂.
A search that *rejects* the terminal (the common case — `sleepstudy_int`, `Pastes`) is unaffected:
the selected model is the incumbent mixed model, whose `bestCAIC` matches within band.

---

## 2026-05-30 — Added `GLM` as a direct runtime dependency (exact-pinned), for the `lm`/`glm` terminal

**Status:** accepted (design — ADR-0006, issue #36). Milestone M4.

**Reason.** A backward `stepcaic` search drops random-effects terms one at a time; dropping the
*last* RE term yields a fixed-effects-only model. `MixedModels.jl` v5.5.1 cannot represent or fit
a no-RE model (`fit(MixedModel, …)` requires at least one `|` term), so this **terminal node**
must be fit and scored as a plain `GLM.jl` `lm`/`glm` — exactly as `cAIC4` does at the same point
(`cAIC4:::cAIC`, the `c("glm","lm")` branch). The terminal scoring (`caic(::RegressionModel)`,
`src/scoring.jl`) is built on `GLM.jl`'s public surface (`lm`/`glm`, `response`, `predict`,
`deviance`, `coef`, the `LinearModel`/`GeneralizedLinearModel` types). The full rationale, the
alternatives weighed, and the coupled `CAICResult` widening are recorded in
[ADR-0006](docs/adr/0006-glm-terminal-and-result-generalization.md).

**Exact pin (CLAUDE.md §3).** `GLM` is pinned to `=1.9.5` in **both** `Project.toml` and
`test/Project.toml`, walked on any version bump exactly like the `MixedModels` pin. `GLM` is
already a *transitive* dependency of `MixedModels` (5.5.1 resolves `GLM` 1.9.5), so promoting it to
an explicit, exact-pinned direct dependency adds **no** resolved-environment drift — only the
direct `[deps]`/`[compat]` entries. `RegressionModel` (the widened `CAICResult` bound, =
`StatsAPI.RegressionModel`) is sourced through `GLM`'s re-export, so no further direct dependency
(e.g. `StatsAPI`) is introduced.

**No quarantine impact.** Fitting and scoring the terminal touches **no** `MixedModels` internals
(public `GLM.jl` + StatsModels formula API), so the `src/mm_internals.jl` internal-access table is
unchanged by this addition (ADR-0006, Consequences).

---

## 2026-05-30 — `lm`/`glm` terminal scoring: Level-2 tolerance (`atol=1e-3`) and the multi-trial-Binomial terminal deviation

**Status:** accepted (validation — issue #36, ADR-0006). Milestone M4; fixture
`test/fixtures/caic_glm_terminal_level2.h5` (generator `test/generate_fixtures_glm_terminal.R`);
tests `test/glm_terminal_tests.jl`.

**The Level-2 band.** The terminal `caic(::RegressionModel)` is validated end-to-end against
`cAIC4`'s public `cAIC()` on the `c("glm","lm")` branch: the Gaussian `lm`, the log-link Poisson
`glm`, and the logit-link Bernoulli `glm`. The shared `(df, condloglik, caic)` triple must agree
within **`atol=1e-3`** — the same Level-2 band carried by the GLMM end-to-end cases (entry
2026-05-29). The terminal sits *far* inside it: an `lm` is a deterministic OLS solve and a `glm` is
IRLS to the same MLE, so with the sample **embedded** in the fixture (R and Julia score identical
data — their RNGs never meet) the discrepancy is ~machine precision, not the iterative-LMM
discrepancy the band was originally sized for. The band is retained (not tightened) for consistency
with the rest of the Level-2 suite. cAIC4's `(g)lm` df is `rank + 1`, and its Gaussian σ̂ is the MLE
rescaling `summary$sigma·√((n−p)/n) = √(RSS/n) = √(deviance(lm)/n)` — reproduced exactly.

**The multi-trial-Binomial terminal deviation.** A multi-trial Binomial `glm` (per-observation
trial counts nᵢ > 1) has **no finite `cAIC4` reference**: `cAIC4`'s binomial `getcondLL` evaluates
`dbinom` on the success *proportion* with `size = |unique(y)|−1`, returning `−∞` (the defect
documented in entry 2026-05-29). The terminal therefore reuses the corrected `condloglik_binomial`
at the true trial counts (recovered from the fit's prior weights, `m.model.rr.wts`) — exactly as
the M3 GLMM binomial path does (entry 2026-05-29). Ground truth is base-R `dbinom(kᵢ, nᵢ, μ̂ᵢ)`
embedded in the fixture (a Level-1-style reference), validated at the same `atol=1e-3`; the test
also asserts the result is finite, unlike cAIC4's `−∞`. Bernoulli (nᵢ ≡ 1) does **not** deviate:
there `cAIC4`'s `size = |unique(y)|−1 = 1` is correct, so `condloglik_bernoulli` matches `cAIC4`
exactly and is cross-checked against the live `cAIC4` reference above.

---

## 2026-05-30 — `stepcaic` (M4) search scope: random-effects only, fixed effects held constant

**Status:** accepted (design — grilled 2026-05-30). Milestone M4; math spec
`docs/math/0008-stepcaic-search.md`; see `CONTEXT.md` (*Search*) and `PARITY.md` (stepcaic row).

`cAIC4`'s `stepcAIC` searches only **random-effects** structure for the `(g)lmer` use case: in
`makeFormula` (`R/helperfuns_stepcAIC.R`) the fixed-effects part (`nobarsF`) is carried through
**unchanged** on every candidate, and `fixEfCandidates` feed only the `gamm4` smooth-term route
(`forwardGam`), which is milestone M5. `cAIC.jl`'s `stepcaic` matches this: every candidate keeps
the original model's fixed-effects part fixed; only RE terms are added/dropped.

**Why this is recorded.** PARITY.md previously described the row as "RE structure primary, FE
optional", which over-claimed: `cAIC4` performs **no** fixed-effects selection for mixed models.
Fixed-effects *selection* would be a deliberate extension **beyond** `cAIC4` with **no R ground
truth** to validate against (Level-2 impossible by construction), so it is deferred and would
carry its own justification if ever added. Not a tolerance — a scope boundary.

---

## 2026-05-30 — `stepcaic` (M4) controller: faithful port of `cAIC4`'s decision cascade; near-tie path divergence is inherent

**Status:** accepted (design — grilled 2026-05-30). Milestone M4; math spec
`docs/math/0008-stepcaic-search.md`.

The greedy controller reproduces `cAIC4`'s decision cascade (`R/stepcAIC.R:565–657`) predicate
for predicate: the `≤` acceptance rule, the `equalToLastStep` plateau guard, the
`improvementInBoth` alternation for `direction="both"` (which starts **forward**,
`R/stepcAIC.R:389`), and the stop predicates (`minCAIC==Inf`, reached `lm`/`glm`, reached the
`keep`-minimal model, a single candidate, `steps` exhausted). Singular candidates are **carried
forward as fit** (not replaced by their reduced model — `R/stepcAIC.R:323–324`, the
`object <- reducedModel` line is commented out upstream), with the reduced-model cAIC driving
selection and the `refit` flag recorded.

**The inherent divergence.** Faithful path replication requires identical cAIC *values* at each
step to make identical greedy choices. `lme4` and `MixedModels.jl` do not produce bit-identical
fits (CLAUDE.md §6), so a candidate whose cAIC sits within the Level-2 fit-discrepancy band of
the incumbent can be accepted by one ecosystem and rejected by the other — flipping the path on a
**near tie**. This is not a bug and not a tolerance to tighten: it is the propagation of the
documented fit discrepancy through a discrete decision. **Validation consequence:** Level-2
asserts the selected RE structure and `bestCAIC` (within the per-method atol band) on every
fixtured scenario, and the full step *path* only on scenarios where successive cAICs are
well-separated relative to that band. The fit-independent search combinatorics are pinned
separately at Level-1 (candidate-set equality vs `backwardStep`/`forwardStep`).

---

## 2026-05-30 — `stepcaic` (M4) refit mechanism requires the source `data` table

**Status:** accepted (design — grilled 2026-05-30). Milestone M4.

Every candidate is represented as a formula and refit via the **public**
`fit(MixedModel, formula, data)` (forward steps add new design columns, which a fitted
`MixedModels` object does not retain — `m.formula` is kept but the source table is not). `stepcaic`
therefore **requires** a `data` argument (a Tables.jl-compatible table containing the response,
the fixed-effects variables, and every `groupcandidates`/`slopecandidates` variable), mirroring
`cAIC4`'s mandatory `data` argument (`R/stepcAIC.R:197–207`, which errors if `data` is absent).

**Why recorded, though not a numerical divergence.** Backward-only search *could* have reused the
internal `ReMat` column-subset machinery (`_subsetreterm`, the reduced-model reconstruction) and
run table-free; the design chose **one uniform formula+fit mechanism** for all directions instead,
mirroring `cAIC4`'s `update()`-based refit so Level-2 validation is apples-to-apples and
`direction=:both` is a pure formula transform. The cost — `data` is mandatory even for pure
backward search — is the recorded consequence. See ADR-0006 (the `lm`/`glm` terminal of this same
mechanism) and `docs/math/0008`.

---

## 2026-05-29 — GLMM partial-singularity reduction: reconstruction-fidelity tolerance (objective `atol=1e-6`, θ/β `atol=1e-5`)

**Status:** accepted (measured). Issue #32 (M3); math spec `docs/math/0007-glmm-partial-singularity-reduction.md`.

`reduceboundary(::GeneralizedLinearMixedModel)` rebuilds a boundary-reduced GLMM from a fitted
object's internals (column-subset `ReMat`s, working-LMM re-wrap) and refits it under Laplace
(`fast=false, nAGQ=1`). The reconstruction is validated bit-for-bit against a **native**
`MixedModels` fit of the reduced model. This is a Level-1 *machinery* check (two Julia fits of
the same reduced model), **not** a `cAIC4` divergence — but the agreement is looser than the
`atol=1e-7` the Gaussian LMM reconstruction achieves, so the tolerance is recorded here.

**Measured (seed-35 `zerocorr(1 + x | g)` Bernoulli, slope variance on the boundary → reduce to
`(1 | g)`):** Δobjective `= 1.2e-10`, Δθ `= 1.2e-6`, Δβ `= 1.9e-6`, Δμ̂ `= 7.1e-7`.

**Why looser than the LMM's 1e-7, and why it is not a defect.** The minimized quantity — the
Laplace deviance `objective` — matches to `~1e-10` (the genuine bit-for-bit signal: the
reconstruction defines the *identical* optimization problem). The *parameters* θ/β differ at
`~1e-6` because the GLMM Laplace objective is flat near the optimum and the reconstruction's
freshly-built working-LMM `optsum` resolves θ slightly differently than a native GLMM's `optsum`
on that flat surface. Two **native** fits of the reduced model are bit-identical (Δθ `= 0`),
confirming the optimizer is deterministic and the gap is config sensitivity on a flat objective,
not nondeterminism or a reconstruction error. The fitted μ̂ — what feeds `ℓ_cond` and ρ — agrees
to `7e-7`, so the assembled cAIC is unaffected at the Level-2 `atol=1e-3` gate (next entry / the
seed-35 fixture). Tolerances chosen with ~8× headroom over the worst observed deviation across
β-initialisations. Per CLAUDE §6 this is a justified bound on understood, deterministic
machinery — never a loosen-to-pass.

---

## 2026-05-29 — GLMM partial-singularity cAIC: Level-2 end-to-end tolerance (`atol=1e-3`) and the singular-agreement regime

**Status:** accepted (measured). Issue #32 (M3). Fixture: `caic_glmm_singular_level2.h5`,
generated by `generate_fixtures_glmm_singular.{jl,R}`; gate in
`glmm_partial_singularity_tests.jl` ("…matches cAIC4 on a partially-singular Bernoulli GLMM").

`caic(::GeneralizedLinearMixedModel)` detects partial singularity (some — not all — variance
directions on the boundary), drops the boundary directions via `reduceboundary` (one level of
`cAIC4`'s `deleteZeroComponents`), refits the reduced GLMM, and cascades until non-singular —
then scores that reduced fit. The end-to-end correctness gate reproduces the conditional AIC
that `cAIC4`'s **public** `cAIC()` returns on a boundary `glmer` fit, on the *identical*
embedded sample.

**The sample and the singular-agreement regime.** The seed-35 design — `zerocorr(1 + x | g)`
Bernoulli, 24 groups × 14 obs, random slope variance unidentified — lands on the **partial**
boundary in MixedModels.jl *and* lme4 alike: both estimate the intercept SD ≈ 0.16 and the
slope SD = 0 exactly (MixedModels λ = `[-0.1598, 0]`; lme4 θ = `[0.1606, 0]`). This agreement is
*not* automatic — for the Gaussian REML analogue the two ecosystems disagree on *whether* the
slope is singular (see the LMM singular-fixture note), which is why that REML case is omitted
rather than fixtured. The seed-35 Bernoulli sample was searched for and confirmed to put *both*
ecosystems on the boundary before being pinned; the sample is embedded in the fixture so the R
and Julia RNGs never need to meet.

**Measured agreement (cAIC.jl − cAIC4):** Δcaic `≈ 7.8e-5`, Δρ `≈ 5.0e-5`, Δℓ_cond `≈ 1.2e-5`.

**Tolerance.** `atol = 1e-3` — the same fit-discrepancy-derived Level-2 tolerance as the
non-singular and Gaussian-singular gates. The worst observed deviation here (Δcaic ≈ 8e-5) sits
~13× inside it: the reduced model is a scalar `(1 | g)` Efron–Steinian score whose θ̂ is nearly
identical across ecosystems. A machinery error in the reduction or scoring shifts the cAIC by
≥ O(0.1), far outside the band. Per CLAUDE §6 the tolerance bounds known lme4↔MixedModels fit
discrepancy — never loosened to pass.

---

## 2026-05-29 — Multi-trial binomial conditional log-likelihood: correct `dbinom` vs `cAIC4`'s defective `getcondLL.merMod`

**Status:** accepted. Applies to `caic(m::GeneralizedLinearMixedModel; method=:bootstrap)` for
a multi-trial Binomial family (`|unique(y)| > 2`, e.g. the CBPP `incid/hsz ~ period + (1|herd)`
fit with `weights = hsz`). Kernel: `Loglik.condloglik_binomial`. Wired through
`_glmm_condloglik_dispatch` (`src/scoring.jl`).

**The `cAIC4` defect.** `getcondLL.merMod` (`cAIC4` 1.1, `R/getcondLL.R`) computes the binomial
conditional log-likelihood as

```r
sum(dbinom(x = getME(object, "y"), size = length(unique(getME(object, "y"))) - 1,
           prob = getME(object, "mu"), log = TRUE))
```

`size = length(unique(y)) - 1` equals the trial count *only* for Bernoulli (`unique(y) = {0,1}`
→ `size = 1`). For a multi-trial binomial the response `y` is a proportion in `[0,1]`, so
`dbinom` receives a **non-integer** `x` and a `size` unrelated to the trials, returns `0` (R
warns "non-integer x = ..."), and `log = -Inf`. `cAIC4` therefore yields a **non-finite**
conditional log-likelihood — and hence a non-finite assembled `cAIC` — for every multi-trial
binomial, even though its `R/cAIC.R:247–253` guard redirects the *df* route to
`conditionalBootstrap`. The defect is in `getcondLL`, which the guard does not touch.

**The deviation (CLAUDE.md §1, §10).** Copying the bug would propagate `-Inf`; CLAUDE.md §1
(mathematical correctness over fidelity to a known-wrong reference) and §10 (a provable `cAIC4`
defect is resolved by a documented deviation, never silently) require the correct density.
`cAIC.jl` evaluates the true binomial log-density at the actual per-observation trial counts
`nᵢ` — the prior weights `m.resp.wts` exposed by `MMInternals.glmmpriorweights` — and success
counts `kᵢ = nᵢ·yᵢ`:

```
ℓ = Σᵢ [ log C(nᵢ, kᵢ) + kᵢ·log μ̂ᵢ + (nᵢ−kᵢ)·log(1−μ̂ᵢ) ],   kᵢ = nᵢ·yᵢ.
```

This is base R's `sum(dbinom(kᵢ, nᵢ, μ̂ᵢ, log = TRUE))` (the *correct* density, not the
`getcondLL` wrapper) and collapses to the Bernoulli `ℓ_cond` when `nᵢ ≡ 1`. The estimand is
pinned in `docs/math/0006-glmm-bias-correction.md §1.1`.

**Validation.** Level-1 only, against the **base-R `dbinom` arithmetic** (not `cAIC4`'s
`getcondLL`) at the Level-1 tolerance `rtol = 1e-6 / atol = 1e-10`. Following the precedent of
`condloglik_poisson`/`condloglik_bernoulli` (and `docs/math/0003-conditional-loglik.md §3`), the
reference is the per-observation density `lchoose(nᵢ,kᵢ) + kᵢ·log μ̂ᵢ + (nᵢ−kᵢ)·log(1−μ̂ᵢ)`
re-stated inline in the test — a different arrangement from the kernel's aggregated `xlogy`/
`loggamma` form, so it cross-checks the aggregation — anchored by hand-computed scalars that
equal base-R `dbinom(k, n, p, log = TRUE)` (e.g. `n=2,k=1,p=0.5 → log 0.5`). No HDF5 fixture is
introduced, matching how the other GLMM log-likelihood kernels are validated. There is **no**
`cAIC4` Level-2 cross-check for this value: `cAIC4`'s own number is `-Inf`, so no finite
reference exists to match. The bootstrap *df* it feeds is unaffected and keeps its existing
Level-2 fixture against `conditionalBootstrap` (`atol = 2.0`, the 2026-05-28 gate). The deviation
is scoped to `method=:bootstrap`; `method=:auto` on a multi-trial binomial still throws
`ArgumentError` (no analytic df), matching `cAIC4`'s family scope.

---

## 2026-05-29 — Added `SpecialFunctions` as a direct runtime dependency

**Reason.** The Poisson conditional log-likelihood (issue #26, M3) requires `loggamma(y + 1)`
to compute `log(y!)` for real-valued (floating-point) count inputs. `LogExpFunctions` imports
`loggamma` from `SpecialFunctions` internally but does not re-export it, so `using
LogExpFunctions: loggamma` fails. The function is not available from Julia Base. Adding
`SpecialFunctions` directly as an explicit dependency is the correct solution.

`SpecialFunctions` is already present as a transitive dependency (through `LogExpFunctions`);
this entry promotes it to an explicit, versioned direct dependency (`SpecialFunctions = "2"`)
with no change to the resolved environment.

---

## 2026-05-27 — Level-2 tolerance for the analytic Gaussian cAIC: `atol = 1e-3`

**Status:** accepted (measured). Applies to `caic` with `method=:steinian`, `hessian=:analytic`
(issue #8). The pending-validation status of the closed-form/analytic path is hereby resolved.

The Level-2 end-to-end comparison fits the same model in `lme4` and `MixedModels.jl` and checks
`cAIC.caic` against `cAIC4`'s public `cAIC()` (reference frozen in `test/fixtures/caic_level2.h5`,
cAIC4 1.1 / lme4 2.0.1). Four cases: `sleepstudy` correlated intercept+slope and random-intercept-
only, each ML and REML, REML pinned on both sides (per the REML/ML entry below).

**Derivation.** The two packages minimise the *same* marginal (restricted) objective and agree on
it to **≤ 2.5e-8** across all four cases — i.e. both reach the same optimum. But the optimisers
settle at slightly different `θ̂`: up to `‖Δθ̂‖∞ ≈ 4e-5` (slope, ML), because the objective is
locally flat there (that 4e-5 shift moves the objective by only ~2.5e-8). The cAIC is evaluated
*at* `θ̂` and is **not** stationary in `θ`, so the same `Δθ̂` maps to a first-order
`Δcaic ≈ ‖∇_θ cAIC‖·‖Δθ̂‖`. Observed worst case (slope, ML): `|Δcaic| = 2.96e-4`, `|Δdf| = 3.1e-4`,
`|Δcll| = 4.6e-4`. The intercept cases, where `θ̂` matches to ~1e-9, agree to ~1e-8 — a near-exact
machinery anchor confirming the discrepancy is fit-induced, not a math error.

**Tolerance.** `atol = 1e-3` on `caic`, `df` (ρ), and `condloglik`, ≈3× the worst observed
fit-induced discrepancy. It is not a loosened tolerance (CLAUDE §10): a genuine machinery error
moves the penalty `2ρ` in sub-degree-of-freedom units, i.e. `Δcaic ≥ O(0.1)`, an order of magnitude
outside this band, so the gate still discriminates correctness from optimiser noise.

---

## 2026-05-28 — `:finitediff` Greven–Kneib Hessian B vs `cAIC4` `analytic=FALSE`: `atol = 1e-3`

**Status:** accepted (measured). Supersedes the 2026-05-27 *pending validation* note of the
same title. Applies to `caic` with `hessian=:finitediff` (issue #11).

`cAIC.jl`'s `:finitediff` B-source self-drives `FiniteDiff.finite_difference_hessian` over
`MixedModels`' *stable* `objective!` at `MixedModels`' θ̂ (ADR-0002, ADR-0001;
`docs/math/0004` §3b) — **not** `cAIC4`'s lifted, Richardson-extrapolated lme4 Hessian. Because
`objective!` **re-profiles** σ²(θ), it differentiates the *same* profiled deviance lme4 stores in
`m@optinfo$derivs$Hessian`, so `:finitediff` reproduces `cAIC4`'s `analytic = FALSE` ρ to
finite-difference accuracy. It cannot bit-match: different optimiser, θ̂, and FD algorithm.

**Derivation.** Same Level-2 fixture and four `sleepstudy` cases as the analytic entry above
(`test/fixtures/caic_level2.h5`, key `df_numeric`/`caic_numeric`; cAIC4 1.1 / lme4 2.0.1). Measured
`|Δρ| = |ρ_finitediff − cAIC4 analytic=FALSE|`:

| case        | s | `|Δρ|`   |
|-------------|---|----------|
| slope_ml    | 3 | 1.37e-4  |
| slope_reml  | 3 | 2.59e-5  |
| int_ml      | 1 | 2.24e-7  |
| int_reml    | 1 | 6.97e-7  |

The worst case (slope_ml, s = 3) combines the central-difference truncation error with the
lme4↔MixedModels θ̂ discrepancy (the same ~4e-5 flat-objective shift as the analytic entry); the
s = 1 cases, where θ̂ matches to ~1e-9, agree to ~1e-7 — confirming the gap is FD-accuracy +
fit-induced, not a math error.

**Tolerance.** `atol = 1e-3` on `caic` and `df` (ρ), the *same* fit-discrepancy band as the
analytic Level-2 gate (≈7× the worst observed 1.37e-4). Not a loosened tolerance (CLAUDE §10): a
genuine assembly error moves `2ρ` by `≥ O(0.1)`, an order of magnitude outside this band.

---

## 2026-05-27 — REML/ML: compute on the fit as-is; defaults differ from lme4

**Status:** accepted.

`cAIC.jl` computes the cAIC on the fit it is given, dispatching on `m.optsum.REML` and
using `MixedModels`' matching objective for θ̂, b̂, and the Hessian B — mirroring
`cAIC4`'s "use the provided fit" behaviour. It does **not** force-refit to ML.
Rationale: `cAIC`/`stepcAIC`'s primary use is selecting random-effects structure with
the fixed-effects design held fixed, where REML is appropriate and comparable across
candidates.

Validation divergence: `lme4` (hence `cAIC4`) defaults to **REML**, whereas
`MixedModels.jl` defaults to **ML** (`REML=false`). Fixtures and Level-2 comparisons
pin the REML flag explicitly on both sides and cover both `REML=true` and
`REML=false`; a naive "fit the same formula in each" comparison is invalid.

---

## 2026-05-27 — Singular fits: match cAIC4's drop-and-refit; detection + reduced refit diverge

**Status:** accepted (behaviour and measured Level-2 tolerance, issue #10).

`cAIC.jl` matches `cAIC4`'s singular-fit handling: detect the boundary (via `MixedModels`'
`issingular`), remove the variance components on the boundary — including a *partial* term (e.g.
a correlated random intercept+slope where only the slope variance is zero) — refit the reduced
model, and compute the cAIC on it; `CAICResult` carries the reduced model and a was-refitted
flag. When *every* random-effect direction is on the boundary, no random-effects model remains,
and the score falls back to the fixed-effects-only one (ρ = rank(FE) + sigma.penalty, the
conditional log-likelihood of the original fit at b̂ = 0), mirroring `cAIC4`'s `lm` branch.

**Level-2 validation (2026-05-28).** Reference frozen in `test/fixtures/caic_singular_level2.h5`
(cAIC4 1.1 / lme4 2.0.1), one case per code path: `reduce_ml` — a `(1 + x | g)` fit with `x`
constant within group, where the slope is unidentifiable and collapses to the boundary in *both*
ecosystems (the synthetic sample is embedded in the fixture so each scores identical data); and
`dyestuff2_{ml,reml}` — the canonical `Dyestuff2` fit whose batch variance is zero (all-boundary
`lm` fallback). Observed worst discrepancy: `reduce_ml` `|Δcaic| ≈ 3.2e-8`, `|Δdf| ≈ 1.0e-9`,
`|Δcll| ≈ 1.7e-8` (a `(1 | g)` refit, near-identical θ̂ across optimisers); `dyestuff2`
`|Δcaic| ≈ 3e-11` (the fixed-effects-only score involves no boundary refit). The same derived
`atol = 1e-3` as the non-singular Level-2 gate applies, with vast margin — a genuine machinery
error moves the penalty `2ρ` by `≥ O(0.1)`.

**Two unavoidable numerical divergences from `cAIC4`.** (1) *Which* fits are flagged singular
differs, because `MixedModels`' `issingular` tolerance and lme4's boundary test are not
identical; (2) the reduced-model refit differs by optimiser (MixedModels vs lme4) — the Level-2
discrepancy quantified above.

**Why the REML analogue of `reduce_ml` is omitted from the fixture.** On the same x-constant-
within-group data fitted by REML, lme4's optimiser settles at a small but *non-zero* slope
variance (sd ≈ 0.03, not flagged singular) where `MixedModels` lands exactly on the boundary, so
the two ecosystems disagree on *whether* the fit is singular at all — divergence (1) above, in
its starkest form. There is therefore no common ground-truth case to compare, and forcing one
would mean comparing `cAIC.jl`'s reduced `(1 | g)` score against `cAIC4`'s full `(1 + x | g)`
score — a category error, not a tolerance question. The ML construction is used precisely because
it forces *both* optimisers onto the boundary, giving a genuine shared reference. This is the
concrete instance of the detection divergence, recorded rather than papered over.

---

## 2026-05-27 — Conditional-bootstrap df: validated by isolation + analytic cross-check, not bit-match

**Status:** accepted. Superseded in part by the 2026-05-28 entry below (which makes the
Level-1 isolation claim concrete).

The conditional-bootstrap df (`method=:bootstrap`) is stochastic and cannot bit-match `cAIC4`
across languages (independent RNGs; per-draw refits also differ by optimiser). Validation
instead: (1) **Level-1 isolation** — the Efron covariance-penalty arithmetic is checked against
`cAIC4`'s internal function on fixed, shared inputs at the tight Level-1 tolerance (closed
2026-05-28; see below); (2) **internal cross-check** — for a Gaussian LMM the bootstrap df must
*lie inside the MC noise band of* the analytic (steinian) df at large `nboot`, **not** converge
to it (the memory note `bootstrap-not-equal-analytic.md` documents the empirically observed
finite gap between cAIC4's own bootstrap and its own analytic df); any end-to-end comparison
against `cAIC4`'s bootstrap is Monte-Carlo-tolerance only and is not a release gate. Bootstrap
draws are reproducible via an `rng::AbstractRNG` argument.

---

## 2026-05-27 — `ForwardDiff` and `FiniteDiff` as core dependencies (B-source packaging)

**Status:** accepted; mechanism refined by [ADR-0002](docs/adr/0002-bsource-ad-strategy.md).

`cAIC.jl` adds **ForwardDiff** and **FiniteDiff** to its *core* dependencies rather than gating
them behind package extensions. Per §3, non-core deps require justification: §3's two mandates
conflict here — *minimal deps* favours extensions, but *single quarantine file* favours keeping all
`MixedModels`-coupled access physically in `src/mm_internals.jl`. Because the relevant `MixedModels`
AD surface is experimental ("subject to change without being considered breaking"), the
**auditability** mandate wins, and the access stays in `mm_internals.jl`.

How each dependency is used differs (see ADR-0002): `:forwarddiff` calls
`ForwardDiff.hessian(::LinearMixedModel)` via the experimental `MixedModelsForwardDiffExt`;
`:finitediff` drives **FiniteDiff** over `MixedModels`' *stable* `objective`/`setθ!` API and does
**not** use `MixedModelsFiniteDiffExt`. So only the ForwardDiff path sits on experimental surface
(shape-asserted, frozen by the `=5.5.1` pin); the FiniteDiff dependency is exercised against stable
API. Cost accepted: a heavier core dependency set. The default `:analytic` B-source uses neither.

---

## 2026-05-28 — Cross-source landscape: `:analytic`, `:finitediff`, `:forwarddiff` are three estimators of one ρ

**Status:** accepted (measured). Applies to the three `hessian` B-sources of `caic` (issue #11);
pins the bounds the cross-source-agreement spec encodes. The mathematics is in `docs/math/0004` §4.

The three B-sources are **three estimators of the same** Greven–Kneib ρ, not three computations of
one number — their pairwise gaps are genuine and recorded, never tolerance-papered (the
bootstrap-vs-analytic precedent applies). Two gaps are *expected-divergent*, and one pair is
*correctness-tight* (the `:finitediff ≡ analytic=FALSE` entry above). The two genuine gaps:

- **σ-freezing gap** `|ρ_forwarddiff − ρ_finitediff|`: `:forwarddiff` rides
  `MixedModelsForwardDiffExt`, whose `fd_deviance` holds σ̂² **fixed** while varying θ, whereas the
  self-driven `:finitediff` differentiates the **re-profiled** σ²(θ) deviance (`docs/math/0004`
  §3a). Re-profiling adds θ-curvature that freezing removes, so the two Hessians — hence the two ρ —
  differ. This is the accepted, documented σ-freezing divergence (the user directed "ride the ext,
  document σ-freezing").
- **closed-form-vs-numeric gap** `|ρ_analytic − ρ_finitediff|`: the closed-form Hessian is not the
  optimiser's numeric Hessian; the gap grows with s and the profile curvature.

**Measured `sleepstudy` spread (ML), the basis of the bounds:**

| case     | s | `ρ_analytic` | `ρ_finitediff` | `ρ_forwarddiff` | `\|an−fd\|` | `\|ford−fd\|` |
|----------|---|--------------|----------------|-----------------|-----------|-------------|
| int_ml   | 1 | 18.97927     | 18.85977       | 18.85725        | 0.120     | 0.00252     |
| slope_ml | 3 | 30.96983     | 32.17335       | 31.96176        | 1.20      | 0.212       |

The structure is robust across all four ML+REML cases: the σ-freezing gap is strictly *smaller*
than the closed-form-vs-numeric gap (`|ford−fd| < |an−fd|`), and both spreads grow with s (s = 1
tight, s = 3 widest). (Note: `docs/math/0004` §4's loose "`:forwarddiff` sits between" holds only
for the s = 3 cases — in the s = 1 cases ρ_forwarddiff falls just below ρ_finitediff rather than
being bracketed — so the spec encodes the robust inequality, not "between".)

**Derived bounds** (the cross-source-agreement spec, not a correctness gate against R):

- **genuine-divergence floor** `1e-3`: every gap exceeds it, proving a real inter-estimator gap;
  it sits well above FD/AD noise (the symmetric-Hessian checks put that at ~1e-6) and below the
  smallest measured genuine gap (the σ-frozen intercept gap, ≈2.5e-3).
- **same-ρ ceiling** `1.5`: every gap is below it, confirming all three remain estimators of one ρ;
  it is > the worst measured `|Δ|` (1.20, slope_ml). A gap above this band would mean a source
  computes a *different* quantity, not a noisier estimate of the same one.

---

## 2026-05-28 — `Random` added as a core dependency for the bootstrap path

**Status:** accepted.

`cAIC.jl` adds `Random` (stdlib) to its core `[deps]` to expose `AbstractRNG` and `default_rng` in
the public `caic()` signature (the `rng` kwarg) and `randn` in the bootstrap spine. `Random` is a
stdlib — no binary or compile-time overhead — and the reproducibility contract (seeded `rng` for
deterministic results) is a first-class user-facing feature, not a test-only concern. Per §3 the
entry here serves as the formal record.

---

## 2026-05-28 — Conditional-bootstrap df: Level-1 shared-input fixture against `cAIC4::conditionalBootstrap`

**Status:** accepted (measured). Closes the §5 dispositions #1, #2, #3 of
`docs/math/0005-conditional-bootstrap.md`.

The Level-1 isolation gate for the bootstrap path is now realised as a **shared-input fixture
against `cAIC4`'s `conditionalBootstrap` arithmetic** at `rtol = 1e-6` / `atol = 1e-10` — the
same tight tolerance as the analytic Level-1 gate. Fixture generator pair
`test/generate_fixtures_bootstrap.{jl,R}` writes seeded synthetic `(yhat, σ, Y*, Ŷ*)` matrices
on the Julia side and runs cAIC4's bias-correction arithmetic (lines 23–25 of cAIC4 v1.1
`R/conditionalBootstrap.R`) on the R side, with a textual self-check on the function body to
pin the formula against silent cAIC4 drift. The Julia test is
`test/dof_lmm_tests.jl`: *"efron_penalty reproduces cAIC4's conditionalBootstrap arithmetic on
shared Y*/Ŷ*"*. Four cases at `B ∈ {2, 20, 100, 500}` and `n ∈ {6, 8, 25, 50}` exercise the
unbiased `(B−1)` divisor (including its minimum), row-mean centring, and the larger Σ-loops.

**What changed in `efron_penalty`.** The Level-1 unit was *previously* the population-mean /
`B`-divisor estimator (centred on the original fit `ŷ`); it is now `cAIC4`'s sample-covariance
formula:

```math
\rho =
  \frac{1}{(B - 1)\,\hat\sigma^{2}}
  \sum_{b = 1}^{B} \sum_{i = 1}^{n}
    \hat y^{*}(b)_{i} \, \bigl(y^{*}(b)_{i} - \bar y^{*}_{i}\bigr)
  + \texttt{sigmapenalty},
```

with `ȳ*ᵢ = (1/B) Σ_b y*(b)ᵢ`, `B ≥ 2` (validated; raises `ArgumentError` for `B = 1` —
no silent division by zero), and `sigmapenalty` default **`0`** (matching cAIC4's bare
arithmetic; the `_bootstrap` spine passes the user-supplied `sigmapenalty` (default `1`)
explicitly for σ²-parameter-count symmetry with the analytic path). The `yhat` argument is
retained in the signature for symmetry with the analytic and numeric Level-1 units but is
unused arithmetically — documented in the function's docstring.

**Why this is not a tolerance loosening.** The previous formula was *asymptotically* equivalent
to cAIC4's but did not bit-match at finite `B`; a tight Level-1 gate was therefore not
realisable. The fix is a code change (CLAUDE §2: cAIC4 is ground truth), not a tolerance
adjustment. The 2026-05-28 bootstrap-vs-analytic convergence gate (`atol = 2.0`, `nboot = 2000`)
continues to hold under the new formula — the unbiased sample-covariance variant has slightly
larger MC variance, but it is absorbed inside the 4–6σ band.

**End-to-end (Level-2) parity is still not a release gate.** Cross-language RNG and per-draw
optimiser differences make bit-match against `cAIC4::cAIC(..., method = "conditionalBootstrap")`
unachievable; the prior 2026-05-27 entry on "validated by isolation + analytic cross-check"
remains in force. What changes is that the Level-1 isolation claim is now operational, not
prospective.

---

## 2026-05-29 — Level-2 tolerance for Bernoulli GLMM df (Efron Steinian): `atol = 1e-3`

**Status:** accepted (measured). Applies to `DofGLMM.dof_glmm_bernoulli` (issue #29). The
estimand is the effective degrees of freedom from `cAIC4::biasCorrectionBernoulli` v1.1.

**Reference fixture.** `test/fixtures/dof_glmm_bernoulli_level2.h5`, generated by
`test/generate_fixtures_bernoulli.R` (seed 42, n = 120, 10 groups of 12, RE σ = 1.0; bobyqa
optimiser on both sides). Ground-truth value: `rho_ref = 7.387431123239024`.

**Julia result.** `DofGLMM.dof_glmm_bernoulli` on a `MixedModels.jl` fit of the same data:
`ρ_julia = 7.388221171827934`, observed `|Δρ| = 0.000790`.

**Derivation.** The Efron estimator is a sum of `n = 120` per-observation contributions, each
involving a full model refit with `yᵢ` flipped. Every contributing term pairs a `MixedModels.jl`
refit (NLopt/bobyqa) with an `lme4` refit (R/bobyqa), so the fit-discrepancy argument from the
LMM Level-2 entry applies here too: slightly different `θ̂` on a flat objective propagates into
a first-order `|Δρᵢ|` per term, and the sum of 120 such terms compounds those discrepancies.
The observed worst-case `|Δρ| = 0.000790` is entirely within the LMM fit-induced band.

**Tolerance.** `atol = 1e-3` on `dof_glmm_bernoulli`, matching the analytic Gaussian Level-2
gate (≈1.3× the observed 0.000790). Not a loosened tolerance (CLAUDE §10): a genuine formula
error in the Efron sum shifts ρ by at least one contribution unit (~0.01–0.1 for typical logit
differences), an order of magnitude above this band.

**NLopt roundoff warnings.** The fixture refits emit several `NLopt was roundoff limited`
warnings from MixedModels.jl during the flip loop. These are NLopt noise on near-converged
solutions and do not affect correctness; the final fits are non-singular and consistent with R.

---

## 2026-05-28 — Bootstrap-vs-analytic convergence gate: `atol=2.0, nboot=2000`

**Status:** accepted.

The Level-2 convergence gate (`caic bootstrap: converges to analytic df with large nboot`) checks
that the Efron bootstrap df converges to the Greven–Kneib analytic df at high draw count. The
tolerance is derived from the MC standard error for the sleepstudy random-intercept model:

- `ρ_analytic ≈ 19` (random-intercept + slope, ML).
- Each draw independently contributes to the covariance sum; the MC standard error of
  `ρ_bootstrap` at B draws is roughly `σ_MC ≈ C/√B` where `C ≈ 10–15` for this model.
- At `B = 2000`: `σ_MC ≈ 0.3–0.5`, so `atol = 2.0` is a 4–6σ band. This is conservative
  enough to survive unlucky seeds yet tight enough to catch a wrong-formula bug (which would
  produce a bias of several units).

Per the memory note (bootstrap-not-equal-analytic.md): the bootstrap df does *not* converge in
probability to analytic df in the strict frequentist sense — the two estimate different quantities
and their means differ by a finite gap. The convergence checked here is that the *empirical* gap
between a large-sample bootstrap estimate and the analytic value is within the MC noise band, not
that the two estimators are asymptotically equivalent. Do NOT tighten this tolerance.
