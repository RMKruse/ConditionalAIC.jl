# ADR-0006 — `stepcAIC` `lm`/`glm` terminal: `GLM.jl` direct dependency and `CAICResult` widening

**Date:** 2026-05-30
**Status:** Accepted (design — grilled, not yet built)

## Context

`cAIC4`'s `stepcAIC` searches **random-effects structure** greedily. A backward search drops
RE terms one at a time; dropping the *last* RE term yields a fixed-effects-only model, which
`cAIC4` fits and scores as a plain `lm`/`glm` (`R/cAIC.R:201–240`: `df = rank + 1`,
`cll = Σ` family log-density at μ̂, `caic = −2·cll + 2·(rank+1)`). Its `Pastes` backward example
bottoms out at exactly such an `lm`. The terminal is therefore a **first-class endpoint of the
search**, not an error.

`MixedModels.jl` v5.5.1 **cannot represent or fit a model with no random-effects term** —
`fit(MixedModel, @formula(y ~ x), data)` requires at least one `|` term. So reaching the
backward terminal means producing and scoring a *non-mixed* fit. The M4 design (grilled
2026-05-30) commits to full backward parity for **both** LMM and GLMM, so both a Gaussian `lm`
terminal and a `glm` (Poisson/Bernoulli/Binomial) terminal must be fit and scored.

Two coupled questions follow: **how to fit the terminal**, and **what type carries its score**
through the search machinery (`caic`, `CAICResult`, `StepcaicResult`).

## Decision

1. **Add `GLM.jl` as a direct runtime dependency** and fit the no-RE terminal with it
   (`lm`/`glm`). `GLM.jl` is already an *transitive* dependency of `MixedModels.jl`, so this
   adds no new resolver burden under the exact-pin regime (CLAUDE.md §3) — only a direct
   `[deps]`/`[compat]` entry and a `DECISIONS.md` justification (CLAUDE.md §12: "no runtime
   dependency without a `DECISIONS.md` entry").

2. **Add `caic(::RegressionModel)` terminal-scoring methods** mirroring `cAIC4`'s `(g)lm`
   branch: `df = rank + 1`, conditional (= marginal here) log-likelihood reusing the existing
   `condloglik_*` kernels (`loglik.jl`), assembled into the same `CAICResult`. The Gaussian
   terminal uses the MLE σ (`cAIC4`'s `σ·√((n−p)/n)` rescaling); the binomial terminal carries
   `cAIC4`'s `size = |unique(y)| − 1` convention (the same defect family already documented for
   the multi-trial binomial `getcondLL` deviation, DECISIONS 2026-05-29).

3. **Widen `CAICResult`'s model bound** from `M <: MixedModel` to `M <: RegressionModel`
   (`StatsAPI.RegressionModel`, the common supertype of `LinearMixedModel`,
   `GeneralizedLinearMixedModel`, and `GLM.jl`'s `TableRegressionModel`). This lets every
   candidate — mixed or terminal — score into one uniform result type, with
   `reducedmodel::Union{Nothing,M}` simply `nothing` for the (never-singular) terminal. The
   widening is backward-compatible (a strictly looser bound); existing `anocaic`/`caic` call
   sites are unaffected.

## Alternatives considered

1. **Stop backward at the minimal *mixed* model (one single-direction RE); never reach `lm`/`glm`.**
   Avoids the dependency and the type widening entirely, but breaks parity: `stepcaic` would
   select a different model than `cAIC4` on any problem whose optimum is the no-RE model (e.g.
   the `Pastes` example). Rejected — parity is the milestone goal (CLAUDE.md §10), and this is a
   silent, structural divergence rather than a justified tolerance.

2. **Fit the Gaussian terminal in-house (`X\y`, σ̂) and skip GLM.jl; treat the `glm` terminal as a
   documented gap.** Keeps the dependency out for the LMM path, but the M4 slice covers GLMM, so
   the `glm` terminal is in scope and would still need IRLS. A half-supported terminal (Gaussian
   only) is a worse parity story than one uniform mechanism. Rejected.

3. **A separate terminal-result type instead of widening `CAICResult`.** Keeps `CAICResult`
   pinned to `MixedModel`, but then the search controller, k-best collection, and
   `StepcaicResult` must each handle two score types — more branching, no benefit. Rejected in
   favour of the single widened type.

## Consequences

- `GLM.jl` joins `[deps]`/`[compat]` with an exact pin (CLAUDE.md §3) and a `DECISIONS.md` entry;
  the pin is walked on any version bump like the `MixedModels` pin.
- `CAICResult{T,M}` becomes `M <: RegressionModel`. The `TableRegressionModel` returned by
  `fit(LinearModel/GeneralizedLinearModel, formula, data)` is the terminal's stored model.
- `caic(::RegressionModel)` is **not** a Greven–Kneib / Stein / bootstrap path — it is the exact
  `(g)lm` AIC `cAIC4` uses at the terminal. It is validated against `cAIC4`'s `cAIC(lm)` /
  `cAIC(glm)` output directly (Level-2, the terminal is a deterministic closed form so the band
  is the fit-discrepancy band of the underlying `lm`/`glm` fit).
- Fitting the terminal touches **no** `MixedModels` internals (public `GLM.jl` + StatsModels
  formula API), so the `mm_internals.jl` quarantine table is unchanged by this decision. (The
  formula → `RESpec` extraction that *interprets* MixedModels RE-term types is a separate
  quarantine concern, tracked in `docs/math/0008`.)
- `StepcaicResult{T}` returns the actual fitted terminal model (parity with `cAIC4`'s
  `finalModel`); the k-best vector is therefore heterogeneous (`Vector{CAICResult{T}}` with an
  abstract model slot). This is a **deliberate, documented exception** to CLAUDE.md §4's
  "no abstractly-typed struct fields" rule, which targets *numerical-performance hot paths*; a
  top-level search-result container is not one. Recorded here so the exception is auditable.
