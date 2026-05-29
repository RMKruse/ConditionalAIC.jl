# DECISIONS.md — Decision Log

Dated entries for every place where `cAIC.jl` legitimately diverges from `cAIC4`,
with the justified tolerance or behaviour. See CLAUDE.md §10. Architectural
decisions (as opposed to `cAIC4`-divergences) live in `docs/adr/`.

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
