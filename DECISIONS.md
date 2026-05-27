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

## 2026-05-27 — FD-sourced Greven–Kneib Hessian B does not bit-match `cAIC4` `analytic=FALSE`

**Status:** pending validation — tolerance to be measured and recorded once Level-2
numbers exist.

`cAIC.jl` will expose a finite-difference source for the Greven–Kneib Hessian B (the
analogue of `cAIC4`'s `analytic=FALSE`), permitted under the reframed §9/§12 (see
[ADR-0001](docs/adr/0001-finite-differences-constrained-not-banned.md)). It cannot
reproduce `cAIC4`'s `analytic=FALSE` values: `cAIC4` lifts lme4's
Richardson-extrapolated (`deriv12`) Hessian of lme4's profiled deviance at lme4's θ̂;
`cAIC.jl` applies a FD scheme to `MixedModels`' objective at `MixedModels`' θ̂
(different optimiser, θ̂, FD algorithm, and possibly REML-vs-ML objective). Agreement
is a Level-2 derived tolerance, to be measured and recorded here. The closed-form and
`ForwardDiff` B-sources remain the validated, default path, held to the Level-1
tolerance.

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
