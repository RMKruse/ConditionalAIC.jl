## Problem Statement

I fit mixed-effects models in Julia with `MixedModels.jl`, and when I have several candidate models I have no principled, established way to choose between them. The classical (marginal) AIC is known to misbehave in the mixed-model setting — it favours larger models and does not reflect how well a model predicts *given the random effects it estimated*. R users have had the right tool for over a decade: `cAIC4`, which implements the bias-corrected **conditional AIC** of Greven & Kneib and the conditional model selection built on it. In Julia there is no equivalent layer. So when I want to compare a handful of fitted `LinearMixedModel`s on a conditional basis, I am stuck — I either misuse the marginal AIC, hand-roll a bias correction I cannot trust, or drop back to R.

## Solution

A Julia package that, given a fitted **Gaussian** `LinearMixedModel`, computes its **conditional AIC** — the conditional log-likelihood ℓ(y | b̂, β̂, θ̂) penalised by the **bias-corrected effective degrees of freedom** ρ (the Greven–Kneib correction, which exceeds the naive plug-in ρ₀ = tr(H₁) precisely because the variance parameters θ are estimated) — and lets me rank a set of fitted models by it.

Two entry points, in the project's vocabulary:

- **Scoring** — `caic(model)` computes the conditional AIC of *one* model and returns a rich `CAICResult` (the cAIC, its two components, provenance, and — for a singular fit — the reduced model it was scored on).
- **Comparison** — `anocaic(models...)` scores a user-supplied *fixed set* consistently and returns a ranked table (the literal "best from among several").

It is **validated against `cAIC4`** so the numbers are trustworthy; it treats a **singular fit** as a first-class case (drop the boundary variance components, refit the **reduced model**, score that); and it exposes selectable degrees-of-freedom estimators and Hessian-**B** sources for advanced use, with sensible automatic defaults so the common path is a single call.

## User Stories

1. As a data analyst, I want to compute the conditional AIC of a fitted Gaussian `LinearMixedModel`, so that I can judge its predictive fit *conditional on* the random effects it estimated rather than with them integrated out.
2. As a data analyst, I want a single call `caic(model)` with sensible defaults, so that the common case needs no knowledge of df estimators or Hessian sources.
3. As a data analyst, I want the conditional log-likelihood and the effective degrees of freedom returned alongside the cAIC value, so that I can see and report the two terms the criterion is built from.
4. As a methods researcher, I want the penalty to be the bias-corrected Greven–Kneib ρ and not the naive hat-matrix trace ρ₀, so that the criterion accounts for the uncertainty in the estimated variance parameters θ.
5. As a methods researcher, I want ρ to coincide with the expected gap ρ ≥ ρ₀, so that I can sanity-check that the bias correction is doing its job.
6. As a data analyst, I want the cAIC computed on my fit *as it stands*, respecting whether the model was fit by REML or ML, so that the criterion describes the model I actually estimated rather than a silently re-fit one.
7. As an applied statistician comparing random-effects structures with the fixed-effects design held constant, I want REML fits supported directly, so that the comparison is on the basis for which REML is appropriate.
8. As a data analyst, I want to pass several fitted models to `anocaic` and get back a ranked table, so that I can see at a glance which model the conditional AIC prefers.
9. As a data analyst, I want `anocaic` to score every candidate *consistently* (same df method, B-source, σ-penalty, REML setting), so that the ranking is not contaminated by inconsistent scoring choices.
10. As a data analyst, I want the comparison table to surface each model's cAIC, conditional log-likelihood, and effective df, so that I can understand *why* one model ranks above another.
11. As a data analyst, I want the `CAICResult` to print a clear, readable summary, so that I can interpret it directly in the REPL without digging into fields.
12. As a developer building on this package, I want `CAICResult` to expose typed accessors for the cAIC, the conditional log-likelihood, and the effective df, so that I can build further tooling on a stable contract.
13. As a data analyst, I want the result to record its *provenance* — which df method and which B-source produced it — so that I can tell how a number was computed when I revisit it later.
14. As an advanced user, I want to select the degrees-of-freedom estimator (`:auto`, `:steinian`, `:bootstrap`), so that I can match the method to my model and needs.
15. As a data analyst, I want `:auto` to choose the analytic (steinian) estimator for the Gaussian family by default, so that I get the exact, fast correction without having to know it exists.
16. As an advanced user, I want to choose the source of the Greven–Kneib Hessian **B** (`:analytic`, `:forwarddiff`, `:finitediff`), so that I can fall back to a numerical source if the analytic path is unavailable.
17. As an advanced user, I want `:analytic` B to be the default and to require no automatic-differentiation machinery, so that the common path is light and has no experimental dependencies in play.
18. As an advanced user, I want a finite-difference B-source that does *not* depend on experimental upstream internals, so that it remains a reliable fallback when the `ForwardDiff` path drifts or breaks.
19. As a data analyst with a singular fit (a variance component estimated on the boundary), I want the package to handle it as a normal, supported case rather than erroring, so that boundary fits do not block my analysis.
20. As a data analyst, I want a singular fit to be reduced (boundary components removed) and refit, and the cAIC computed on that reduced model, so that the scoring matches `cAIC4`'s behaviour.
21. As a data analyst, I want the result to carry the reduced model and a "was refitted" flag when a singular fit was reduced, so that I know the score pertains to a reduced model and can inspect it.
22. As a data analyst, I want even a *partially* singular term handled (e.g. a correlated intercept+slope where only the slope variance is zero), so that the reduction is correct for realistic boundary cases.
23. As a methods researcher, I want every cAIC number validated against `cAIC4`, so that I can trust the Julia results match the established R reference.
24. As a cautious user, I want invalid input to raise a clear, typed error (e.g. an unsupported model family, or a malformed option), so that I never silently receive a wrong number.
25. As a performance-sensitive user, I want the scoring to be type-stable and free of explicit matrix inverses, so that it is fast and numerically sound on larger models.
26. As a methods researcher, I want all likelihood and determinant computations done in log-space and via Cholesky factors, so that the criterion is stable for ill-conditioned covariance structures.
27. As an advanced user requesting the bootstrap df, I want to pass a random-number generator, so that my results are reproducible.
28. As an advanced user, I want to set the σ-penalty (`sigmapenalty`), so that I retain the same tuning knob `cAIC4` exposes.
29. As the package maintainer, I want *all* access to `MixedModels` internals confined to a single quarantine module, so that an upstream change is auditable and fixable in one place.
30. As the package maintainer, I want `MixedModels` pinned to an exact version with a committed `Manifest.toml`, so that a version bump is a deliberate, reviewed event and not an incidental break.
31. As the package maintainer, I want the quarantine module to shape-assert what it pulls out of `MixedModels` and fail loudly on drift, so that a silent upstream change surfaces as a clear error rather than a wrong cAIC.
32. As the package maintainer, I want the Greven–Kneib df arithmetic isolated as a pure, fit-independent unit, so that I can validate the correction *mathematics* directly against `cAIC4` independent of any model fit.
33. As the package maintainer, I want a Level-1 validation harness that feeds identical synthetic components to both `cAIC4`'s internal correction and ours, so that a numerical mismatch is localised to the mathematics rather than the fit.
34. As the package maintainer, I want a Level-2 end-to-end validation against `lme4`/`cAIC4`, so that the assembled, real-world cAIC is checked within a justified tolerance.
35. As the package maintainer, I want CI to run against committed fixtures with no R required, so that the default test job is fast and portable, with live R re-validation gated separately.
36. As a contributor, I want the three B-sources to agree (within tolerance) on the same model, so that the analytic, `ForwardDiff`, and finite-difference paths are cross-checked against each other.

## Implementation Decisions

**Scope.** This PRD covers the **M1 foundation + M2 Gaussian LMM Scoring (`caic`) + M2.5 Comparison (`anocaic`)** only. GLMM, search (`stepcaic`), model averaging, and additive models are out of scope (see below).

**Module decomposition.** Built around deep, isolatable modules:

- **`mm_internals`** — the *only* module touching `MixedModels` (pinned `=5.5.1`). Typed, shape-asserted accessors: the predicted random effects b̂ (= λu), σ̂, the naive plug-in df ρ₀ via the leverage accessor, the singular-fit flag, the REML flag, the Greven–Kneib component matrices, and the `objective` / `setθ!` handles plus `ForwardDiff.hessian` needed by the B-source provider. Not a translation layer — audited direct access. Carries the internal-access table (CLAUDE.md §3).
- **`numerics`** — shared numerically-stable primitives: traces computed without materialising full matrix products, `logdet` on Cholesky/triangular factors, log-space operations, Cholesky-based solves. Pure, no `MixedModels` dependency.
- **`dof_lmm`** — the port of `cAIC4`'s `calculateGaussianBc`: a pure *component-set → effective degrees of freedom ρ* function. Fit-independent and parametrisation-neutral; this is the Level-1 isolation unit (ADR-0003).
- **`bhessian`** *(confirmed as a standalone deep module)* — the B-source provider, operationalising ADR-0002 behind one interface: `:analytic` (closed-form, no AD), `:forwarddiff` (via `MixedModels`' experimental `ForwardDiff.hessian`, the only path on experimental surface), `:finitediff` (self-driven `FiniteDiff` over the *stable* `objective`/`setθ!` API, **not** the experimental extension). The cross-source-agreement property lives here as a unit test. The `MixedModels`-touching calls are delegated to `mm_internals`.
- **`loglik`** — the conditional log-likelihood ℓ(y | b̂, β̂, θ̂) for the Gaussian family; a pure function of extracted quantities.
- **`types`** — `CAICResult{T,M}` (with provenance, `Base.show`, accessors) and the `anocaic` comparison-table type.
- **`caic`** — assembly/orchestration: extract → ρ → ℓ → assemble `CAICResult`; the singular-fit **drop-and-refit → reduced-model** path; family dispatch (Gaussian now).
- **`anocaic`** — Comparison: score a fixed set consistently and rank into a table.
- **`cAIC` (entry)** — module definition and explicit exports.

**Public API contract** (locked during design; renamed to avoid the module-name collision `cAIC`→`caic`):

```julia
caic(m::MixedModel;
     method::Symbol       = :auto,        # :auto | :steinian | :bootstrap
     hessian::Symbol      = :analytic,    # :analytic | :forwarddiff | :finitediff
     nboot::Union{Int,Nothing} = nothing, # bootstrap draws (method=:bootstrap)
     sigmapenalty::Int    = 1) -> CAICResult

CAICResult{T,M<:MixedModel}:
    caic         ::T               # the conditional AIC value
    dof          ::T               # effective degrees of freedom ρ
    condloglik   ::T               # conditional log-likelihood
    reducedmodel ::Union{Nothing,M}  # set iff a singular fit was reduced
    refit        ::Bool            # whether scoring used a reduced refit
    # + provenance: the method and B-source actually used
```

- `method=:auto` mirrors `cAIC4`'s `method=NULL`: it selects `:steinian` for the analytically-supported families (Gaussian here). `method`/`hessian`/`sigmapenalty`/`nboot` map onto `cAIC4`'s `method` / `analytic` / `sigma.penalty` / bootstrap controls.
- `anocaic(models...)` returns the ranked Comparison table, scoring each model through the same `caic` path with identical options so the ranking is consistent (provenance enforces this).

**Behavioural decisions (recorded divergences from `cAIC4`; see `DECISIONS.md`):**

- **REML/ML on-the-fit.** Dispatch on the model's REML flag and use `MixedModels`' matching objective for θ̂, b̂, and B; never force-refit to ML. Validation pins the REML flag explicitly on both sides because `MixedModels` defaults to ML while `lme4`/`cAIC4` default to REML.
- **Singular fits.** Detect via `MixedModels`' singular-fit test, remove the boundary variance components (including a partial term), refit the reduced model, and score that; the result carries the reduced model and the refit flag.
- **B-source coupling (ADR-0002).** Only `:forwarddiff` sits on `MixedModels`' experimental AD surface; `:finitediff` rides the stable objective so it can serve as the fallback when `ForwardDiff` drifts. The experimental σ-inclusion risk is contained by the exact version pin + shape assertions.
- **Bootstrap df.** Stochastic; reproducible via an RNG argument; not bit-matched cross-language.
- **Dependencies.** `ForwardDiff` and `FiniteDiff` are direct core dependencies (auditability over minimal-deps, since the relevant `MixedModels` AD surface is experimental); `:analytic` uses neither. `RCall` and `HDF5` stay test-only.

**Numerical-stability constraints (CLAUDE.md §9, non-negotiable):** log-space likelihoods/densities; no explicit inverses (factorisation-based solves; reuse `MixedModels`' Cholesky blocks); `logdet` on factors; traces via identities rather than materialised products; singular fits handled, not papered over.

## Testing Decisions

**What makes a good test here:** it pins *external behaviour* — a returned cAIC / df / ranking, an error type, a type-stability guarantee — against a reference (the `cAIC4` value, a hand-computed quantity, or a cross-source agreement), never an internal intermediate that is free to change. The two validation levels are kept strictly separate so a mismatch is debuggable: **Level-1** (machinery isolated; tight `rtol=1e-6`/`atol=1e-10`) and **Level-2** (end-to-end; a tolerance derived from and justified by the fit discrepancy, recorded in `DECISIONS.md`).

**Dedicated/isolated test suites (all four confirmed as priority targets):**

- **`dof_lmm` — Level-1.** Feed synthetic, parametrisation-neutral components to both `cAIC4:::calculateGaussianBc` and our port; compare the effective df at the Level-1 tolerance. The primary correctness gate (ADR-0003). Fixtures travel through the HDF5 neutral-handoff pipeline: `generate_fixtures.jl` builds seeded components → HDF5; `generate_fixtures.R` reads them, calls the `cAIC4` internal, writes the reference df back; CI reads the HDF5 in Julia with no R.
- **`numerics` + `loglik` — pure units.** Stable trace / `logdet` / log-space primitives checked against naive references; the Gaussian conditional log-likelihood checked against hand-computed values on synthetic inputs. No `MixedModels` dependency, fully isolated.
- **`bhessian` — cross-source.** The `:analytic`, `:forwarddiff`, and `:finitediff` sources must agree within tolerance on the same model; plus the invariant that the self-driven FD driver restores the model's fitted θ̂ (a perturbed `setθ!` left behind is a defect), and the σ-inclusion shape assertion.
- **`caic` + `anocaic` — Level-2 / integration.** Full cAIC vs `lme4`/`cAIC4` at the derived Level-2 tolerance; the singular drop-and-refit reduced-model path; consistent ranking from `anocaic`. Plus `@inferred` type-stability assertions and `Aqua` (ambiguities, stale deps, undefined exports).

**Per-estimator coverage (CLAUDE.md §6):** happy path vs the R reference; type stability via `@inferred`; edge cases (singular fit, θ near the boundary, single grouping factor, crossed and nested random effects, unbalanced data); documented error types for invalid input; Level-1 numerical accuracy.

**Prior art:** the Level-1 tolerances (`rtol=1e-6`, `atol=1e-10`) follow `pymlt`, cited in CLAUDE.md §6; `cAIC4`'s own test suite for reference values; `TestItems.jl` for test organisation; `Aqua.jl` and `JET.jl` as the standing quality gates.

## Out of Scope

- **GLMM scoring** (Poisson/Bernoulli analytic, bootstrap fallback) — milestone M3.
- **Search / `stepcaic`** (conditional stepwise selection over a candidate space) — milestone M4.
- **Model averaging** (`modelAvg`/`predictMA`/`summaryMA`) — milestone M4.5 (surfaced from the `cAIC4` `NAMESPACE`, folded into the parity goal during design).
- **Additive models** (`gamm4` equivalent) — milestone M5; decided late by design.
- **`nlme`/`lme` support** — `cAIC.jl` targets `MixedModels.jl` objects only.
- **The numeric Level-2 tolerance value** — measured and recorded once real fixtures exist; this PRD fixes the *method* of deriving it, not the number.
- **Bit-for-bit reproduction of `cAIC4`'s `analytic=FALSE`** — impossible (different optimiser, θ̂, FD scheme); the FD B-source is a `cAIC4`-`analytic=FALSE` *analogue*, not a match.
- **Performance optimisation beyond type stability and the §9 numerical rules** — correctness first.

## Further Notes

- The authoritative design artefacts already exist in the repo and govern this work: `CONTEXT.md` (the glossary — Scoring / Comparison / effective df / singular / reduced), `PARITY.md` (the `cAIC4` parity matrix), `DECISIONS.md` (the five recorded divergences), and `docs/adr/0001–0003` (FD-as-constraint; the B-source AD strategy; the Level-1 isolation boundary + HDF5 pipeline). CLAUDE.md §9/§12 and §11 were amended during design.
- Per the §7 implementation ritual, the **first step is the M2 `docs/math` specification** — the closed-form Greven–Kneib **B**, ρ₀ = tr(H₁) via the leverage accessor, and the exact `calculateGaussianBc` component definitions (`A`, `Wlist`, `eWelist`, `V0inv`, `e`, …) — written *before* any Julia code, then the HDF5 fixture, then a failing test, then implementation.
- Two `MixedModels` semantics to confirm from upstream docs during that spec: the b̂ = λu convention exposed by the random-effects accessor, and exactly what the leverage accessor returns (per-observation leverages vs the trace).
- `cAIC4` is ground truth; verify its function names, signatures, and call chain against the actual source + `NAMESPACE` before encoding any detail (this caught the `anocAIC` spelling and the `getModelComponents`/`calculateGaussianBc` split during design).
