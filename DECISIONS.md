# DECISIONS.md — Decision Log

Dated entries for every place where `cAIC.jl` legitimately diverges from `cAIC4`,
with the justified tolerance or behaviour. See CLAUDE.md §10. Architectural
decisions (as opposed to `cAIC4`-divergences) live in `docs/adr/`.

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

**Status:** accepted (behaviour); Level-2 tolerance pending validation.

`cAIC.jl` matches `cAIC4`'s singular-fit handling: detect the boundary (via `MixedModels`'
`issingular`), remove the variance components on the boundary — including a *partial* term (e.g.
a correlated random intercept+slope where only the slope variance is zero) — refit the reduced
model, and compute the cAIC on it; `CAICResult` carries the reduced model and a was-refitted
flag. Two unavoidable numerical divergences from `cAIC4`: (1) *which* fits are flagged singular
differs, because `MixedModels`' `issingular` tolerance and lme4's boundary test are not
identical; (2) the reduced-model refit differs by optimiser (MixedModels vs lme4) — a Level-2
discrepancy. Both are documented here; the tolerance is measured once fixtures exist.

---

## 2026-05-27 — Conditional-bootstrap df: validated by isolation + analytic cross-check, not bit-match

**Status:** accepted.

The conditional-bootstrap df (`method=:bootstrap`) is stochastic and cannot bit-match `cAIC4`
across languages (independent RNGs; per-draw refits also differ by optimiser). Validation
instead: (1) **Level-1 isolation** — the Efron covariance-penalty arithmetic is checked against
`cAIC4`'s internal function on fixed, shared inputs at the tight Level-1 tolerance; (2)
**internal cross-check** — for a Gaussian LMM the bootstrap df must converge to the *exact*
analytic (steinian) df as `nboot → ∞`, which is the primary correctness gate. Any end-to-end
comparison against `cAIC4`'s bootstrap is Monte-Carlo-tolerance only and is not a release gate.
Bootstrap draws are reproducible via an `rng::AbstractRNG` argument.

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
