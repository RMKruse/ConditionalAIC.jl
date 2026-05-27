# PARITY.md — `cAIC4` Parity Matrix

The authoritative scope definition (CLAUDE.md §10). Every public function and
supported model family of `cAIC4`, mapped to a `cAIC.jl` milestone and a status.
**"Feature parity" means every in-scope row is 🟢.** Divergences are not tracked
here — they live in `DECISIONS.md`; architectural choices live in `docs/adr/`.

Terms (Scoring / Comparison / Search, steinian / bootstrap, singular / reduced)
are defined in [CONTEXT.md](CONTEXT.md). Milestones M1–M6 are defined in
CLAUDE.md §11.

## Status legend

| Symbol | Meaning |
|--------|---------|
| 🟢 | done — implemented **and** validated against the R reference |
| 🟡 | in progress / partial |
| 🟦 | planned — scope **resolved** (design grilled, not yet built) |
| ⬜ | planned — scope **not yet grilled** |
| 🚫 | deliberately out of scope / deferred (see note) |

## Near-term scope

The committed near-term target is **LMM scoring and its immediate selection
layer** — i.e. the M2 → M2.5 → M4 spine on Gaussian `LinearMixedModel`s. GLMM
(M3), model averaging (M4.5), and additive models (M5) are deliberately deferred
to their own milestones; M5 is decided late by design (CLAUDE.md §11). The rows
below reflect that: the LMM spine is 🟦 (resolved), GLMM/additive rows are ⬜/🚫.

## Public API

Public surface verified against `cAIC4`'s `NAMESPACE` (2026-05-27): exports are `cAIC`,
`stepcAIC`, `anocAIC`, `getcondLL`, `getModelComponents`, `getWeights`,
`deleteZeroComponents`, `modelAvg`, `predictMA`, `summaryMA`, `print.cAIC`.

| `cAIC4` | `cAIC.jl` | Milestone | Status | Notes |
|---------|-----------|-----------|--------|-------|
| `cAIC(object, method, B, sigma.penalty, analytic)` | `caic(m; method, hessian, nboot, sigmapenalty)` | M2 (Gaussian) | 🟦 | Signature locked. Renamed `cAIC`→`caic` (collides with module name). `analytic` → `hessian::Symbol`; `method=NULL` → `method=:auto`. |
| — (same, GLMM dispatch) | `caic(m::GeneralizedLinearMixedModel; …)` | M3 | ⬜ | Family coverage ungrilled. |
| `anocAIC(...)` (compare a set of fits) | `anocaic(ms...)` → table | M2.5 | 🟦 | **Comparison**: rank a user-supplied fixed set. "Early, right after scoring." Spelling: `cAIC4` exports `anocAIC`; our lowercase port is `anocaic`. |
| `stepcAIC(...)` | `stepcaic(...)` | M4 | 🟦 | **Search**: RE structure primary, FE optional. Shape resolved; details ungrilled. |
| `getcondLL` (exported) | `CAICResult.condloglik` + accessor | M2 | 🟦 | Surfaced as a result field, not a free function. |
| `deleteZeroComponents` (exported) | internal reduced-model path | M2 | 🟦 | `cAIC4` exports it; we fold it into the reduced-model path (drives `CAICResult.reducedmodel` + `refit`). A public Julia equivalent is optional — revisit if parity demands it. See DECISIONS (singular fits). |
| `getModelComponents` (exported) | internal (`mm_internals.jl`) | M2 (Level-2) | 🟦 | The fit→components bridge; carries the θ-parametrization, so it is exercised at Level-2, not Level-1 (ADR-0003). |
| `modelAvg` / `predictMA` / `summaryMA` (+ internal `getWeights` / `weightOptim`) | (API TBD at M4.5) | M4.5 | 🟦 | **Averaging**: cAIC-weighted model combination. In the parity goal as its own milestone M4.5 (CLAUDE.md §11, amended 2026-05-27). API ungrilled. |

## Model families

| Family | `cAIC4` df route | `cAIC.jl` milestone | Status | Notes |
|--------|------------------|---------------------|--------|-------|
| Gaussian LMM | steinian (analytic Greven–Kneib) | M2 | 🟦 | The core; full GK bias correction. |
| Poisson GLMM | steinian | M3 | ⬜ | Analytic Stein route; refitting cost ungrilled. |
| Bernoulli / binomial GLMM | steinian | M3 | ⬜ | Analytic Stein route. |
| other GLMM families | conditional bootstrap | M3 | ⬜ | Bootstrap fallback (Efron). |
| additive (`gamm4`) | — | M5 | 🚫 | No direct Julia analogue; deferred by design (CLAUDE.md §11, §10). |

## df estimators & B-source

| `cAIC4` | `cAIC.jl` | Status | Notes |
|---------|-----------|--------|-------|
| `method=NULL` (auto by family) | `method=:auto` | 🟦 | `:steinian` for gaussian/poisson/bernoulli, else `:bootstrap`. |
| steinian | `method=:steinian` | 🟦 (M2 Gaussian) | Analytic GK; Level-1 tolerance. |
| `conditionalBootstrap` | `method=:bootstrap` | 🟦 (design) | Validated by isolation + analytic cross-check, not bit-match (DECISIONS). `rng` arg for reproducibility. |
| `analytic=TRUE` (closed-form B) | `hessian=:analytic` | 🟦 | Default B-source; no derivative dependency. |
| `analytic=FALSE` (lifted lme4 Hessian) | `hessian=:forwarddiff` / `:finitediff` | 🟦 | No lme4 Hessian to lift in MM; B computed at cAIC-time. No bit-match to `analytic=FALSE` (DECISIONS + ADR-0001). |
| `sigma.penalty` | `sigmapenalty::Int` | 🟦 | Carried through unchanged. |

## REML / ML

| Aspect | `cAIC.jl` | Status | Notes |
|--------|-----------|--------|-------|
| objective used for θ̂, b̂, B | compute on the fit as-is; dispatch on `m.optsum.REML` | 🟦 | No force-refit. Defaults differ from lme4 (MM defaults ML); fixtures pin REML on both sides (DECISIONS). |
