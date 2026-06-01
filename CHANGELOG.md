# Changelog

All notable changes to `ConditionalAIC.jl` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-01

First public release. `ConditionalAIC.jl` is a Julia reimplementation of the R
package `cAIC4`, providing the **conditional Akaike Information Criterion** and
conditional model selection for mixed-effects models fitted with
[`MixedModels.jl`](https://github.com/JuliaStats/MixedModels.jl) (`LinearMixedModel`,
`GeneralizedLinearMixedModel`).

This release completes the **M1 → M2 → M2.5 → M3 → M4 → M4.5** spine. Every
in-scope row of `PARITY.md` is validated against the R reference; additive models
(`gamm4`, M5) remain deliberately deferred.

### Public API

- **`caic(m; method, hessian, nboot, sigmapenalty, rng)`** — conditional AIC for a
  fitted `LinearMixedModel`, `GeneralizedLinearMixedModel`, or (terminal)
  `RegressionModel`. Returns a `CAICResult` carrying the conditional log-likelihood,
  effective degrees of freedom, the cAIC value, and any reduced model.
  (`cAIC4`'s `cAIC`; renamed to avoid colliding with the module name.)
- **`anocaic(ms...)`** — rank a user-supplied set of fits into a table ordered by
  cAIC ascending. (`cAIC4`'s `anocAIC`.)
- **`stepcaic(m, data; direction, keep, …)`** — conditional stepwise search over
  random-effects structure (backward / forward / both), for LMM and GLMM. Returns a
  `StepcaicResult` with the selected model, the k-best fits, and the search path.
  (`cAIC4`'s `stepcAIC`.)
- **`modelavg(models…; weights)`** — cAIC-weighted model averaging (Gaussian LMM),
  returning a `ModelAvgResult` with name-keyed model-averaged fixed and random
  effects. (`cAIC4`'s `modelAvg`.)
- **`predictma(res, newdata; new_re_levels)`** — model-averaged weighted conditional
  prediction. (`cAIC4`'s `predictMA`.)
- **`getweights(res)`** — Zhang-optimal weight vector for a `ModelAvgResult`,
  returning a `WeightResult`. (`cAIC4`'s `getWeights`.)
- The conditional log-likelihood (`cAIC4`'s exported `getcondLL`) is surfaced as the
  `CAICResult.condloglik` field; the `summaryMA` report is the `text/plain` display
  of `ModelAvgResult`.

### Added — M1 Foundation

- Package scaffolding for `ConditionalAIC` (Julia ≥ 1.10 LTS), `Project.toml` with
  exact-pinned `MixedModels = "=5.5.1"` and `GLM = "=1.9.5"`, and the
  `{1.10, 1.11, nightly}` CI matrix.
- **`src/mm_internals.jl`** — the single quarantine file for all `MixedModels.jl`
  internal-field access, with the auditable internal-access table in its module
  docstring (pinned to the exact `MixedModels` version).
- **`src/numerics.jl`** — the pure numerically-stable primitive layer
  (`cAIC.Numerics`), generic over `T<:AbstractFloat`, with no `MixedModels`
  dependency. Each estimand, its stable form, and its naive reference are recorded
  in `docs/math/0001`.
- The two-level R-reference validation harness (Level 1: machinery in isolation;
  Level 2: end-to-end fit) with the HDF5 fixture pipeline and `Aqua.jl` quality
  gate.

### Added — M2 Gaussian LMM

- **Conditional log-likelihood** (`src/loglik.jl`) for the Gaussian conditional
  density given predicted random effects, in log-space.
- **Analytic Greven–Kneib bias correction** (`src/dof_lmm.jl`) — the effective
  degrees of freedom `ρ` accounting for estimation of the variance parameters,
  with all three B-sources:
  - `hessian=:analytic` — closed-form B, the default (no derivative dependency).
  - `hessian=:finitediff` — `FiniteDiff` over `MixedModels`' stable profiled
    objective (ADR-0002).
  - `hessian=:forwarddiff` — `ForwardDiff` path (ADR-0001).
- **`caic` assembly** (`src/scoring.jl`, `src/components.jl`, `src/types.jl`) wiring
  the conditional log-likelihood and df into the full cAIC, dispatching on model
  family. Level-2-validated end-to-end vs `cAIC4`/`lme4` on `sleepstudy` (ML + REML,
  slope + intercept) at `atol=1e-3`.
- **REML / ML objective dispatch** — scoring is computed on the fit as-is (no
  force-refit), dispatching on `m.optsum.REML`; both paths Level-2-validated.
- **`sigmapenalty`** carried through, verified to shift `ρ` by one per unit.
- **Singular-fit handling** (`deleteZeroComponents` analogue) — boundary-component
  detection, partial and whole-term drop, cascading reduced-model refit, and the
  all-boundary `lm` fallback; Level-2-validated at `atol=1e-3`.
- **Parametric bootstrap** (`method=:bootstrap`) for the Gaussian path — Efron
  penalty realigned to `cAIC4`'s exact formula, `nboot` default 500, `rng` kwarg
  for reproducibility.

### Added — M2.5 Comparison

- **`anocaic(ms...)`** — ranks a fixed set of fits by cAIC ascending into a table;
  Level-2-validated against `cAIC4` (sort order + cAIC values, `atol=1e-3`).

### Added — M3 GLMM

- GLMM accessors added to the quarantine file (`GeneralizedLinearMixedModel`,
  `refit!`, `issingular`, prior weights, conditional simulation).
- **Conditional log-likelihood kernels** (`src/loglik.jl`) for Poisson, Bernoulli,
  and multi-trial Binomial. The Binomial `ℓ_cond` (`condloglik_binomial`) uses the
  correct `dbinom` with trial counts — a documented, intentional deviation from
  `cAIC4`'s defective `−∞` binomial `getcondLL` (DECISIONS 2026-05-29).
- **Family-specific df estimators** (`src/dof_glmm.jl`):
  - Poisson — Chen–Stein correction (`dof_glmm_poisson`).
  - Bernoulli — Efron's Steinian estimator (`dof_glmm_bernoulli`).
  - Other families — conditional bootstrap (`dof_glmm_bootstrap`), drawing
    `yᵢ ~ f(μ̂ᵢ)` for Poisson/Bernoulli/Binomial (ADR-0005).
- **GLMM `caic` dispatch** — `method=:auto` routes by family; `method=:bootstrap`
  works end-to-end for Poisson, Bernoulli, and multi-trial Binomial. Multi-trial
  Binomial requires `method=:bootstrap` (no analytic df under `:auto`).
- **Full- and partial-singularity GLMM reduction** — `reduceboundary` drops boundary
  directions, refits the reduced GLMM (Laplace), and the cascade recurses until
  non-singular; the full-singularity fallback is `ρ = rank(X)`. Level-2-validated on
  a partially-singular Bernoulli fit (`atol=1e-3`; `docs/math/0007`).

### Added — M4 stepcAIC

- **`stepcaic(m, data; …)`** — conditional stepwise search over random-effects
  structure only (fixed effects held constant, matching `cAIC4`'s `(g)lmer` path),
  for LMM and GLMM.
- Internal RE-structure spec (`src/respec.jl`, `RESpec`/`REGroup`) — the `cnms`
  analogue — with `extract`/`render` round-tripping on the public `StatsModels` term
  API and the exported `zerocorr` (no new internals).
- **Backward, forward, and both** candidate enumeration
  (`src/stepcaic_candidates.jl`) — faithful ports of `cAIC4`'s `backwardStep` /
  `forwardStep`, including hierarchical-order checks, correlated/uncorrelated splits,
  and nesting ingredients; Level-1 set-equality validated against the live `cAIC4`
  internals.
- **Greedy controller** (`src/stepcaic.jl`) — a faithful port of `cAIC4`'s decision
  cascade (acceptance / plateau / both-alternation / stop predicates, the
  forward-terminal early return), with a non-convergence guard (effective cAIC of
  `+Inf`). Level-2-validated on `sleepstudy` / `Pastes` / crossed-Poisson scenarios.
- **`keep` kwarg** as a `FormulaTerm` RE fragment (the `keep$random` analogue).
- **GLM/GLMM terminal scoring** — `GLM.jl` added as a direct dependency;
  `caic(::RegressionModel)` scores the `lm`/`glm` terminal (Gaussian, Poisson,
  Bernoulli, multi-trial Binomial) with `df = rank + 1`. `CAICResult`'s bound widened
  `M<:MixedModel → M<:RegressionModel` (ADR-0006). Level-2-validated at `atol=1e-3`.

### Added — M4.5 Model averaging

- **`modelavg(models…; weights=:zhang)`** (`src/averaging.jl`) — cAIC-weighted model
  combination for Gaussian LMM, returning a `ModelAvgResult{T}` with name-keyed
  model-averaged `fixeff`/`raneff` (a `NamedEffects{K,T}` over the union of candidate
  terms), per-candidate cAICs, and weights. Fail-loud candidate-set contract
  (mismatched response/`n`, mixed REML, or non-`LinearMixedModel` → `ArgumentError`).
- Two weight schemes:
  - **Zhang-optimal** (`weights=:zhang`, the default; `cAIC4`'s `opt=TRUE`) — via
    `getweights`, a faithful Julia transcription of `cAIC4`'s `solnp`-based
    augmented-Lagrangian SQP weight optimizer (`src/weightoptim.jl`), with the
    documented §9 carve-out for the literal Cholesky-factor inverse (ADR-0007).
  - **Buckland smoothed** (`weights=:smoothed`; `cAIC4`'s `opt=FALSE`) —
    `exp(−Δ/2)` normalised in log-space via `logsumexp`.
- **`getweights(res)`** → `WeightResult{T}` (weights, objective value, runtime),
  with a fast path that caches the result on a `:zhang` `ModelAvgResult`.
- **`predictma(res, newdata; new_re_levels=:error)`** → `Vector{T}`, the weighted
  conditional prediction `ŷ^MA = Σ wᵢ · predict(mᵢ, D*)`. The `:error` default
  mirrors `lme4`'s `allow.new.levels=FALSE` (`:population`/`:missing` opt-in;
  recorded divergence, DECISIONS 2026-05-31).
- **`summaryMA` report** folded into `Base.show` for `ModelAvgResult` — prints the
  candidate formulas, the per-candidate cAIC + weight table, and the model-averaged
  fixed and random effects (no standalone `summaryma` function, a deliberate parity
  divergence).
- Validation: Level-1 shared-input gate vs `getWeights`/`.weightOptim`
  (`rtol=1e-6, atol=1e-10`); Level-2 anchors for weights, objective, and predictions
  vs `cAIC4::modelAvg(opt=TRUE)` + `predictMA`.

### Documentation & process

- Mathematical specifications in `docs/math/0001`–`0009` (written before each
  estimator, per the implementation ritual).
- Architecture Decision Records `docs/adr/0001`–`0007`.
- `PARITY.md` (the `cAIC4` parity matrix), `DECISIONS.md` (every justified
  divergence and derived tolerance), `CONTEXT.md`, `README.md`, and `CITATION.cff`.

### Known limitations

- **Additive models** (`gamm4` equivalent, M5) are deferred by design — no direct
  Julia analogue.
- **Model averaging** is Gaussian-LMM-only (the Zhang/Mallows objective is Gaussian
  by construction; no R ground truth for GLMM averaging).
- **stepcAIC** searches random-effects structure only; fixed-effect selection and
  driver-side nesting expansion / parallelism are deferred.
- `MixedModels.jl` and `lme4` do not produce bit-identical fits; end-to-end (Level-2)
  agreement is within tolerances derived from the fit discrepancy and recorded in
  `DECISIONS.md`.

[0.1.0]: https://github.com/RMKruse/ConditionalAIC.jl/releases/tag/v0.1.0
