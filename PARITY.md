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
| — (same, GLMM dispatch) | `caic(m::GeneralizedLinearMixedModel; …)` | M3 | 🟢 | Poisson (Chen-Stein, `method=:auto`) and Bernoulli (Efron Steinian, `method=:auto`) implemented and assembly-tested (#31). `method=:bootstrap` works end-to-end for every bootstrap-supported family — Poisson, Bernoulli, **and multi-trial Binomial** — each with a finite conditional log-likelihood: the binomial `ℓ_cond` (`condloglik_binomial`) deviates from `cAIC4`'s defective `getcondLL` (which is `−∞` for multi-trial) and is assembly-tested on CBPP (`caic_glmm_tests.jl`; DECISIONS 2026-05-29, `docs/math/0006 §1.1`). Multi-trial Binomial requires `method=:bootstrap` (no analytic df under `:auto` → `ArgumentError`). Full-singularity fallback (ρ=rank(X)) unchanged from #27. **Partial-singularity** reduction implemented (#32): some-but-not-all boundary directions are dropped, the reduced GLMM is refit (`reduceboundary`, Laplace) and the cascade recurses until non-singular, then scores the reduced fit (`refit=true`, carries `reducedmodel`) — the GLMM analogue of `deleteZeroComponents`. Level-2-validated end-to-end vs. cAIC4's public `cAIC()` on a partially-singular Bernoulli fit (`caic_glmm_singular_level2.h5`; atol=1e-3, DECISIONS 2026-05-29; `docs/math/0007`). |
| `anocAIC(...)` (compare a set of fits) | `anocaic(ms...)` → table | M2.5 | 🟢 | **Comparison**: rank a user-supplied fixed set by cAIC ascending. Implemented and Level-2-validated via `caic_level2.h5` (#13). Reuses the existing scoring fixtures; validates sort order + cAIC values against `cAIC4` within atol=1e-3. Zero-model calls are a `MethodError` (type-system enforcement, not runtime check). |
| `stepcAIC(...)` | `stepcaic(m, data; …)` | M4 | 🟦 | **Search**: random-effects structure **only** — the fixed-effects part is held constant on every candidate, matching `cAIC4`'s `(g)lmer` path (its `fixEfCandidates` are `gamm4`-smooth-only → M5). FE *selection* is a deferred non-`cAIC4` extension (no R ground truth). **Resolved design** (grilled 2026-05-30): backward + forward + both; LMM **and** GLMM; candidates represented by an internal RE-structure spec (the `cnms` analogue), enumerated by pure add/drop transforms (Level-1 set-equality vs `cAIC4`'s `backwardStep`/`forwardStep`) and refit via the public `fit(MixedModel, formula, data)` (`data` required). Backward reaches the `lm`/`glm` terminal via **`GLM.jl`** (new direct dep; `caic(::RegressionModel)` with `df=rank+1`) — the terminal scoring itself is **implemented and Level-2-validated** (#36): Gaussian `lm` and Poisson/Bernoulli `glm` vs `cAIC4`'s public `cAIC()` at atol=1e-3, multi-trial Binomial via the corrected `condloglik_binomial` (the same documented deviation as M3) vs base-R `dbinom`, plus `@inferred` type-stability and an `ArgumentError` on unsupported families (`caic_glm_terminal_level2.h5`; DECISIONS 2026-05-30; `docs/math/0008 §0.1`). The greedy controller that walks to it is a faithful port of `cAIC4`'s decision cascade (acceptance/plateau/both-alternation/stop predicates). Full non-`gamm4` option surface, **serial** (parallelism deferred). Returns a `StepcaicResult{T}` carrying the selected + k-best fitted models and a structured search path; `CAICResult`'s bound widens `M<:MixedModel → M<:RegressionModel`. Level-2-validated end-to-end vs `cAIC4` `stepcAIC` (selected RE structure + `bestCAIC` within per-method atol; path on well-separated cases). See `docs/math/0008`. |
| `getcondLL` (exported) | `CAICResult.condloglik` + accessor | M2 | 🟢 | Surfaced as a result field, not a free function; the conditional log-lik is Level-2-validated vs `cAIC4`'s `getcondLL` (#8). |
| `deleteZeroComponents` (exported) | internal reduced-model path | M2 | 🟢 | `cAIC4` exports it; we fold it into the reduced-model path (drives `CAICResult.reducedmodel` + `refit`): boundary detection (`issingular`), partial + whole-term drop, cascading reduction, and the all-boundary `lm` fallback are implemented and Level-2-validated vs `cAIC4` (#10; atol=1e-3, DECISIONS). The **GLMM** analogue is implemented too (#32): `reduceboundary(::GeneralizedLinearMixedModel)` drops boundary directions and refits the reduced GLMM (Laplace), and `caic` cascades it — Level-2-validated on a partially-singular Bernoulli fit (atol=1e-3; `docs/math/0007`). A public Julia equivalent is optional — revisit if parity demands it. See DECISIONS (singular fits). |
| `getModelComponents` (exported) | internal (`Components` + `mm_internals.jl`) | M2 (Level-2) | 🟢 | The fit→components bridge; carries the θ-parametrization, so it is exercised at Level-2, not Level-1 (ADR-0003). Validated end-to-end through `caic` vs `cAIC4` (#8). |
| `modelAvg` / `predictMA` / `summaryMA` (+ internal `getWeights` / `weightOptim`) | (API TBD at M4.5) | M4.5 | 🟦 | **Averaging**: cAIC-weighted model combination. In the parity goal as its own milestone M4.5 (CLAUDE.md §11, amended 2026-05-27). API ungrilled. |

## Model families

| Family | `cAIC4` df route | `cAIC.jl` milestone | Status | Notes |
|--------|------------------|---------------------|--------|-------|
| Gaussian LMM | steinian (analytic Greven–Kneib) | M2 | 🟢 | The core; full GK bias correction. Level-1 df arithmetic (`calculateGaussianBc` → `dof_lmm`) validated against `cAIC4` v1.1 via the HDF5 fixture pipeline (#7); full `caic` assembly Level-2-validated end-to-end vs `cAIC4`/`lme4` on `sleepstudy` (ML+REML, slope+intercept), atol=1e-3 (#8, DECISIONS). |
| Poisson GLMM | Chen-Stein correction (`biasCorrectionPoisson`) | M3 | 🟢 | Influence-based, *not* Greven–Kneib. `dof_glmm_poisson` (#28): Level-1 arithmetic validated vs. R; Level-2 model dispatch validated via fixture (atol=0.5, fit-discrepancy band). Wired into `caic` assembly (#31). |
| Bernoulli / binomial GLMM | Efron's estimator (`biasCorrectionBernoulli`) | M3 | 🟢 | Influence-based, *not* Greven–Kneib. `dof_glmm_bernoulli` (#29): Level-1 kernel validated; Level-2 fixture (atol=1e-3). Wired into `caic` assembly (#31). Multi-trial Binomial (`\|unique(y)\|>2`) routes to `method=:bootstrap`, with its conditional log-likelihood from the correct `dbinom` (`condloglik_binomial`) — a documented deviation from `cAIC4`'s `−∞` binomial `getcondLL` (DECISIONS 2026-05-29). |
| other GLMM families | conditional bootstrap (`conditionalBootstrap`) | M3 | 🟢 | `dof_glmm_bootstrap` (#30): conditional draws `yᵢ ~ f(μ̂ᵢ)` via `glmmconddraw` (Poisson/Bernoulli/Binomial; ADR-0005), n×B refits, Efron penalty with σ=1. Wired into `caic(m; method=:bootstrap)` (#31) and end-to-end assembly-tested for multi-trial Binomial on CBPP (`caic_glmm_tests.jl`) — the conditional log-likelihood completes the path (`condloglik_binomial`). Level-2 df fixture vs. cAIC4 `conditionalBootstrap` at atol=2.0. |
| additive (`gamm4`) | — | M5 | 🚫 | No direct Julia analogue; deferred by design (CLAUDE.md §11, §10). |

## df estimators & B-source

| `cAIC4` | `cAIC.jl` | Status | Notes |
|---------|-----------|--------|-------|
| `method=NULL` (auto by family) | `method=:auto` | 🟢 | Gaussian: resolves to `:steinian` (validated #8). GLMM: dispatches to family-specific estimator (Poisson → Chen-Stein, Bernoulli → Efron; #31). Non-Poisson/Bernoulli families require `method=:bootstrap`. |
| steinian | `method=:steinian` | 🟢 (Gaussian) / not-a-kwarg (GLMM) | For Gaussian: `:steinian` is the explicit kwarg for the analytic GK correction. For GLMM, `:steinian` is not a valid kwarg — family-specific estimators are selected via `:auto` (no `:steinian` override exists in M3 scope; `cAIC4`'s "steinian" for GLMM is the auto-dispatch, not a separate option). |
| `conditionalBootstrap` | `method=:bootstrap` | 🟢 | Gaussian: parametric bootstrap (#12). GLMM: conditional bootstrap (#30, #31); validated vs. cAIC4 `conditionalBootstrap` (atol=2.0). Reachable end-to-end through `caic` for Poisson, Bernoulli, and multi-trial Binomial (binomial `ℓ_cond` via `condloglik_binomial`). `rng` kwarg for reproducibility. |
| `analytic=TRUE` (closed-form B) | `hessian=:analytic` | 🟢 | Default B-source; no derivative dependency. Level-2-validated (#8). |
| `analytic=FALSE` (lifted lme4 Hessian) | `hessian=:forwarddiff` / `:finitediff` | 🟢 | No lme4 Hessian to lift in MM; B computed at cAIC-time (ADR-0002, #11). `:finitediff` self-drives FiniteDiff over MM's stable profiled objective and reproduces `analytic=FALSE` ρ to FD accuracy, Level-2-validated (atol=1e-3, DECISIONS); `:forwarddiff` rides the experimental ext, diverging by σ-freezing. No bit-match to `analytic=FALSE` (DECISIONS + ADR-0001). |
| `sigma.penalty` | `sigmapenalty::Int` | 🟢 | Carried through unchanged; verified to shift ρ by one per unit (#8). |

## REML / ML

| Aspect | `cAIC.jl` | Status | Notes |
|--------|-----------|--------|-------|
| objective used for θ̂, b̂, B | compute on the fit as-is; dispatch on `m.optsum.REML` | 🟢 | No force-refit. Defaults differ from lme4 (MM defaults ML); fixtures pin REML on both sides (DECISIONS). Both ML and REML Level-2-validated against `cAIC4` (#8); the objective dispatch, the no-force-refit guarantee, and REML-path type-stability are pinned by focused behavioural specs (#9). |
