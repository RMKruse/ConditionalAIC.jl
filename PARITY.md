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
| `cAIC(object, method, B, sigma.penalty, analytic)` | `caic(m; method, hessian, nboot, sigmapenalty)` | M2 (Gaussian) | 🟢 | Gaussian `method=:auto/:steinian` with all three B-sources implemented and Level-2-validated vs `cAIC4`: `hessian=:analytic` (#8; atol=1e-3) and `:finitediff`/`:forwarddiff` (#11; the numeric B-sources, DECISIONS). Renamed `cAIC`→`caic` (collides with module name). Only `method=:bootstrap` parses-but-errors not-yet-implemented (#12). |
| — (same, GLMM dispatch) | `caic(m::GeneralizedLinearMixedModel; …)` | M3 | ⬜ | Family coverage ungrilled. |
| `anocAIC(...)` (compare a set of fits) | `anocaic(ms...)` → table | M2.5 | 🟢 | **Comparison**: rank a user-supplied fixed set by cAIC ascending. Implemented and Level-2-validated via `caic_level2.h5` (#13). Reuses the existing scoring fixtures; validates sort order + cAIC values against `cAIC4` within atol=1e-3. Zero-model calls are a `MethodError` (type-system enforcement, not runtime check). |
| `stepcAIC(...)` | `stepcaic(...)` | M4 | 🟦 | **Search**: RE structure primary, FE optional. Shape resolved; details ungrilled. |
| `getcondLL` (exported) | `CAICResult.condloglik` + accessor | M2 | 🟢 | Surfaced as a result field, not a free function; the conditional log-lik is Level-2-validated vs `cAIC4`'s `getcondLL` (#8). |
| `deleteZeroComponents` (exported) | internal reduced-model path | M2 | 🟢 | `cAIC4` exports it; we fold it into the reduced-model path (drives `CAICResult.reducedmodel` + `refit`): boundary detection (`issingular`), partial + whole-term drop, cascading reduction, and the all-boundary `lm` fallback are implemented and Level-2-validated vs `cAIC4` (#10; atol=1e-3, DECISIONS). A public Julia equivalent is optional — revisit if parity demands it. See DECISIONS (singular fits). |
| `getModelComponents` (exported) | internal (`Components` + `mm_internals.jl`) | M2 (Level-2) | 🟢 | The fit→components bridge; carries the θ-parametrization, so it is exercised at Level-2, not Level-1 (ADR-0003). Validated end-to-end through `caic` vs `cAIC4` (#8). |
| `modelAvg` / `predictMA` / `summaryMA` (+ internal `getWeights` / `weightOptim`) | (API TBD at M4.5) | M4.5 | 🟦 | **Averaging**: cAIC-weighted model combination. In the parity goal as its own milestone M4.5 (CLAUDE.md §11, amended 2026-05-27). API ungrilled. |

## Model families

| Family | `cAIC4` df route | `cAIC.jl` milestone | Status | Notes |
|--------|------------------|---------------------|--------|-------|
| Gaussian LMM | steinian (analytic Greven–Kneib) | M2 | 🟢 | The core; full GK bias correction. Level-1 df arithmetic (`calculateGaussianBc` → `dof_lmm`) validated against `cAIC4` v1.1 via the HDF5 fixture pipeline (#7); full `caic` assembly Level-2-validated end-to-end vs `cAIC4`/`lme4` on `sleepstudy` (ML+REML, slope+intercept), atol=1e-3 (#8, DECISIONS). |
| Poisson GLMM | steinian | M3 | ⬜ | Analytic Stein route; refitting cost ungrilled. |
| Bernoulli / binomial GLMM | steinian | M3 | ⬜ | Analytic Stein route. |
| other GLMM families | conditional bootstrap | M3 | ⬜ | Bootstrap fallback (Efron). |
| additive (`gamm4`) | — | M5 | 🚫 | No direct Julia analogue; deferred by design (CLAUDE.md §11, §10). |

## df estimators & B-source

| `cAIC4` | `cAIC.jl` | Status | Notes |
|---------|-----------|--------|-------|
| `method=NULL` (auto by family) | `method=:auto` | 🟢 (Gaussian) | Resolves to `:steinian` for Gaussian (validated #8); `:bootstrap` fallback for other families is M3. |
| steinian | `method=:steinian` | 🟢 (M2 Gaussian) | Analytic GK; Level-1 df tolerance + Level-2 end-to-end (#8). |
| `conditionalBootstrap` | `method=:bootstrap` | 🟦 (design) | Validated by isolation + analytic cross-check, not bit-match (DECISIONS). `rng` arg for reproducibility. |
| `analytic=TRUE` (closed-form B) | `hessian=:analytic` | 🟢 | Default B-source; no derivative dependency. Level-2-validated (#8). |
| `analytic=FALSE` (lifted lme4 Hessian) | `hessian=:forwarddiff` / `:finitediff` | 🟢 | No lme4 Hessian to lift in MM; B computed at cAIC-time (ADR-0002, #11). `:finitediff` self-drives FiniteDiff over MM's stable profiled objective and reproduces `analytic=FALSE` ρ to FD accuracy, Level-2-validated (atol=1e-3, DECISIONS); `:forwarddiff` rides the experimental ext, diverging by σ-freezing. No bit-match to `analytic=FALSE` (DECISIONS + ADR-0001). |
| `sigma.penalty` | `sigmapenalty::Int` | 🟢 | Carried through unchanged; verified to shift ρ by one per unit (#8). |

## REML / ML

| Aspect | `cAIC.jl` | Status | Notes |
|--------|-----------|--------|-------|
| objective used for θ̂, b̂, B | compute on the fit as-is; dispatch on `m.optsum.REML` | 🟢 | No force-refit. Defaults differ from lme4 (MM defaults ML); fixtures pin REML on both sides (DECISIONS). Both ML and REML Level-2-validated against `cAIC4` (#8); the objective dispatch, the no-force-refit guarantee, and REML-path type-stability are pinned by focused behavioural specs (#9). |
