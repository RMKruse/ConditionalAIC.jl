# CLAUDE.md — cAIC.jl

## Project Overview

`cAIC.jl` is a Julia reimplementation of the R package `cAIC4`, providing the
**conditional Akaike Information Criterion** and conditional model selection for
mixed-effects models.

This is **not a from-scratch port** of `cAIC4` — it is a *re-platforming*. `cAIC4`
sits on `lme4`/`merMod`; `cAIC.jl` sits on **`MixedModels.jl`**
(`LinearMixedModel`, `GeneralizedLinearMixedModel`). The work is therefore:
reconstruct the bias-correction and degrees-of-freedom logic of `cAIC4` on top of
the fitted objects produced by `MixedModels.jl`.

The conditional AIC has the schematic form

```
cAIC = -2 · ℓ_cond(y | b̂, β̂, θ̂) + 2 · ρ
```

where `ℓ_cond` is the conditional log-likelihood given the predicted random
effects `b̂`, and `ρ` is the (bias-corrected) effective degrees of freedom. The
naive plug-in df `ρ₀ = tr(H₁)` — the trace of the hat matrix mapping `y → ŷ` at
fixed variance parameters `θ` — *understates* `ρ` because `θ` is estimated. The
correction accounting for that estimation uncertainty is the mathematical core of
this package.

**Primary references** (authoritative; the project's formulas must match these):
- Greven, S. & Kneib, T. (2010). On the behaviour of marginal and conditional AIC
  in linear mixed models. *Biometrika*.
- Säfken, B., Rügamer, D., Kneib, T. & Greven, S. (2021). Conditional Model
  Selection in Mixed-Effects Models with `cAIC4`. *Journal of Statistical Software*.
- The R packages `cAIC4`, `lme4`.

The authoritative mathematical specification for each estimator lives in
`docs/math/` and must be written *before* the corresponding Julia code.

---

## Critical Principles

These are absolute. They override convenience, brevity, and cleverness in every
case. If a change conflicts with one of these, the change is wrong.

1. **Mathematical correctness over cleverness.** Never "simplify", "optimize", or
   rewrite a formula without a proof of equivalence recorded in `docs/math/` or
   the commit message. A faster expression that is not provably equal to the
   reference expression is a bug, not an optimization.

2. **`cAIC4` / R is ground truth.** Every estimator must be validated against R
   output. If `cAIC.jl` and R disagree, `cAIC.jl` is wrong until proven otherwise.
   Divergence is never resolved by adjusting a tolerance — see §6 and §10.

3. **Numerical stability is non-negotiable.** Every computation involving
   likelihoods, densities, determinants, or covariance factors uses the
   numerically stable formulation (log-space, Cholesky-based solves, no explicit
   inverses). See §9 — it is not advisory.

4. **`MixedModels.jl` internals are a pinned, quarantined dependency.** Direct
   field access is permitted, but only from one file and only against one exact
   pinned version. See §3.

---

## §3 — Architecture & the `MixedModels.jl` Coupling

`cAIC.jl` depends on `MixedModels.jl` **internals** by design (this was a
deliberate project decision). To keep that dependency survivable:

- **Exact version pin.** `MixedModels` MUST be pinned to an exact version in
  `[compat]` (e.g. `MixedModels = "=X.Y.Z"`), and `Manifest.toml` MUST be
  committed. A version bump is a deliberate, reviewed event — never incidental.

- **Single quarantine file.** *All* access to `MixedModels.jl` internal fields
  and unexported functions MUST live in `src/mm_internals.jl`. This is **not** an
  abstraction layer and performs **no** translation — it accesses fields directly
  (`m.reterms`, `m.A`, `m.L`, `m.feterm`, `m.optsum`, `m.parmap`, the `λ`/`θ`/`β`
  properties, `ranef`, the hat-matrix / leverage accessor, etc.). Its sole purpose
  is to make every internal touchpoint **auditable in one place**. No other file
  in `src/` may reach into a `MixedModels` object's internals.

- **Internal-access table.** `src/mm_internals.jl` MUST carry, in its module
  docstring, a table of every internal field/function accessed, with the pinned
  `MixedModels` version. On a version bump, walking this table is the required
  checklist. Accessing an internal not listed in the table is forbidden — add it
  to the table first.

- **Fail loud on internal drift.** Where feasible, `mm_internals.jl` should assert
  the shape/type of what it pulls out, so a silent upstream change surfaces as a
  clear error rather than a wrong number downstream.

**Module layout** (indicative):

```
src/
  cAIC.jl              # module entry point, public API + __all__-equivalent (exports)
  mm_internals.jl      # QUARANTINE: the only file touching MixedModels internals
  types.jl             # cAIC result types, model-family dispatch types
  loglik.jl            # conditional log-likelihood
  dof_lmm.jl           # Gaussian LMM degrees-of-freedom / Greven–Kneib correction
  dof_glmm.jl          # GLMM bias correction
  caic.jl              # the cAIC assembly (dispatch over model family)
  stepcaic.jl          # conditional stepwise selection
  numerics.jl          # shared numerically-stable primitives
```

- **No circular dependencies** between source files.
- **Dependencies stay minimal.** Core: `MixedModels`, `LinearAlgebra`,
  `Statistics`, `LogExpFunctions`. Anything else requires an entry in
  `DECISIONS.md` justifying it. `RCall` is a **test-only** dependency (`[extras]` /
  `test/Project.toml`), never a runtime dependency.

---

## §4 — Code Standards

### Julia version & style
- Julia **≥ 1.10 (LTS)**. CI runs the matrix `{1.10, 1.11, nightly}`.
- Formatting is enforced by **`JuliaFormatter.jl`** with the repo's
  `.JuliaFormatter.toml` (Blue style). A formatting check is a CI gate.
- Public functions, types, and modules carry a **docstring** with: signature, a
  one-line summary, mathematical background in precise notation (the estimand and
  the algorithm), `# Arguments`, `# Returns`, and at least one example.
  Docstrings are built into the manual via `Documenter.jl`.

### Naming
- Types and modules: `CamelCase` (`ConditionalAIC`, `CAICResult`).
- Functions and variables: `lowercase`, words run together; an underscore only
  where readability genuinely demands it. Mutating functions end in `!`.
- Internal (non-exported) helpers are simply not exported; the public surface is
  defined explicitly via `export` in `src/cAIC.jl`.

### Type discipline
- **Concrete, parametric struct fields.** No abstractly-typed struct fields
  (`Matrix{Float64}` or `Matrix{T}` — never `AbstractMatrix` as a stored field).
  Type-unstable structs are a numerical-performance bug.
- Functions are written to be **type-stable**. Type stability is verified — see
  §6 (`@inferred`) and §8 (`JET.jl`). Type instability in a hot path is treated
  as a defect, not a style nit.
- Generic over the float type (`T<:AbstractFloat`) where it costs nothing;
  never hardcode `Float64` in numerical kernels.

### Error handling
- Fail **loudly**. Invalid input raises `ArgumentError`; a value outside a
  mathematical domain raises `DomainError`. Never return a silently-wrong number.
- `@assert` is for tests and internal invariants only — **never** for validating
  user-facing input in library code.
- Never `catch` an exception only to swallow it. If you catch, you handle or
  rethrow with added context.
- No `println`/`print` for diagnostics. Use `@info` / `@warn` / `@debug`, or
  return the information.

---

## §6 — Testing & the R Reference Workflow

Validation runs on **two levels** — keep them strictly separate. `lme4` and
`MixedModels.jl` do **not** produce bit-identical fits, so a naive "run cAIC4 in
R, run cAIC.jl, diff" does **not** work end-to-end.

**Level 1 — machinery isolated.** Feed *identical, synthetically constructed*
inputs (random-effects design `Z`, covariance factor `Λ`, conditional
quantities, etc.) into `cAIC4`'s internal df / bias-correction functions **and**
into the corresponding `cAIC.jl` functions. This validates the correction
*mathematics* independently of any model fit. Tolerance is tight here:
`rtol = 1e-6`, `atol = 1e-10` (as in `pymlt`).

**Level 2 — end-to-end.** Fit the same model in `lme4` and `MixedModels.jl`,
compute the full cAIC in each. Fit differences propagate; the cAIC values must
agree within a tolerance **derived from and justified by** the fit discrepancy.
This derived tolerance is recorded in `DECISIONS.md`. Disagreement beyond it is a
bug. **Do not** mix Level 1 and Level 2 — a numerical mismatch is only debuggable
when you know which level it lives on.

### Harness mechanics (fixtures + live RCall)
- `test/fixtures/` holds **serialized R reference outputs**, committed to the
  repo. They are regenerated by `test/generate_fixtures.{R,jl}`, which drives
  `cAIC4` (including its internal functions for Level 1).
- **CI runs against fixtures only** — fast, no R required in the default job.
- **Live `RCall.jl` tests** run locally and in a separate scheduled CI job, gated
  behind an environment variable (e.g. `CAIC_LIVE_RCALL=1`). They re-validate the
  fixtures against a live `cAIC4` so fixture rot is caught.
- Regenerating fixtures is a reviewed change: a fixture diff must be explained in
  the PR.

### Test requirements per function
Every new estimator gets parametrized tests (`@testset`, `TestItems.jl`) covering:
- happy path against the R reference value;
- **type stability** via `@inferred`;
- edge cases — singular fit, variance parameter on the boundary (`θ` near 0),
  single grouping factor, crossed *and* nested random effects, unbalanced data;
- error handling — invalid input raises the documented exception type;
- numerical accuracy at the Level-1 tolerance.

`Aqua.jl` runs as part of the suite (ambiguities, stale deps, undefined exports).

---

## §7 — Implementation Ritual

Every estimator is implemented in this order. No step is skipped.

1. **State the math.** Write the estimand and the algorithm in precise notation
   in `docs/math/`. Vague is forbidden — not "implement the df correction" but,
   e.g., `ρ = tr(∂ŷ/∂yᵀ)` with the estimation-uncertainty term written out, citing
   Greven & Kneib (2010), eq. (n).
2. **Produce the R reference.** Extend `test/generate_fixtures.*` to emit the
   ground-truth value from `cAIC4` (an internal function for Level 1; the public
   path for Level 2). Commit the fixture.
3. **Write a failing test** that encodes the R output as the expected value.
4. **Implement** — correctness first, in the numerically stable formulation (§9).
5. **Run the gates** (§8). All green.
6. Only now is the function "done."

Refactoring: run the existing tests *first*, change second, run again. A
refactor that changes a numerical result is not a refactor — it is a behavior
change and needs its own R-reference justification.

---

## §8 — Quality Gates ("done" means all of these pass)

A change is **not done** until, in this order:

1. **Format** — `JuliaFormatter` check passes (no diff).
2. **Static analysis** — `JET.jl` reports no errors and no unexpected type
   instabilities in numerical paths (`@report_opt` / `@report_call`).
3. **Tests** — full `Pkg.test()` green, including `@inferred` type-stability
   assertions and `Aqua.jl`.

CI enforces all three. Local pre-commit should run at least 1 and 3.

---

## §9 — Numerical Stability Rules (non-negotiable)

- **Log-space** for all likelihoods, densities, and probabilities. Use
  `LogExpFunctions.jl` (`logsumexp`, `logaddexp`, `log1p`, `xlogx`); never sum
  raw products of small numbers.
- **Never form an explicit inverse.** No `inv(A)`. Solve linear systems with
  factorizations (`A \ b`, `cholesky`, `ldiv!`). `MixedModels.jl` already exposes
  the Cholesky blocks `L` — use them; do not refactorize.
- **Determinants via `logdet`** on a Cholesky/triangular factor — never `det` of a
  large or near-singular matrix.
- **Traces** (for degrees of freedom) are computed without materializing full
  matrix products where an identity allows it (`tr(AB) = Σ A .* Bᵀ`), and from
  triangular solves rather than dense inverses.
- **Cholesky parametrization** for covariance structures; respect the
  parametrization `MixedModels.jl` uses for `λ`/`θ`.
- **Singular fits** (covariance factor `λ` with zeros on the diagonal) are a
  first-class case, not an error to paper over. Detect, handle, and document the
  behavior; match what `cAIC4` does.
- **Analytic derivatives** are preferred. Where an analytic gradient is
  impractical, use `ForwardDiff.jl`. **Finite differences are permitted only as a
  cross-check in tests**, never in the shipped computation.
- Guard the obvious edge cases: empty inputs, `NaN`/`Inf`, non-positive values
  where positivity is required, `θ` on the boundary.

---

## §10 — Parity Discipline & Decision Log

The goal is **feature parity with `cAIC4`**. Two documents govern this:

- **`PARITY.md`** — the parity matrix: every public function and supported model
  family of `cAIC4`, each mapped to a milestone and a status. This is the
  authoritative scope definition. "Feature parity" means every row is green.
- **`DECISIONS.md`** — the decision log. Every place where `cAIC.jl` and `cAIC4`
  legitimately diverge gets a dated entry: what diverges, why, and the justified
  tolerance or behavior. Examples that *will* need entries: the Level-2 tolerance
  derivation; residual-degrees-of-freedom conventions (there is no canonical
  definition for mixed models, and `MixedModels.jl` and `lme4` need not agree);
  the additive-model path, where Julia has no direct `gamm4` analogue.

A divergence from R is **never** resolved silently. Investigate first; then
either fix `cAIC.jl` or record a justified deviation in `DECISIONS.md`.

---

## §11 — Milestones

- **M1 — Foundation.** Package scaffolding, `mm_internals.jl` + internal-access
  table, the two-level validation harness, the §0 quantity-mapping complete. No
  cAIC math yet.
- **M2 — Gaussian LMM.** The analytic Greven–Kneib bias correction. Brings the
  full skeleton end-to-end; builds on the `MixedModels.jl` hat-matrix accessor.
- **M3 — GLMM.** Non-Gaussian families; the refitting / numerical-approximation
  path. The expensive milestone — make refitting cheap (reuse factorizations).
- **M4 — `stepcAIC`.** Conditional stepwise selection on top of M2/M3.
- **M5 — Additive models** (`gamm4` equivalent). Highest open risk: no direct
  Julia analogue; scope is decided deliberately and late, recorded in
  `DECISIONS.md`.
- **M6 — v1.0.** Documentation complete, `PARITY.md` fully green, registered in
  the General registry.

---

## §12 — What NOT To Do

- **Do not** form explicit matrix inverses (`inv`). Use factorizations.
- **Do not** rewrite or "simplify" a formula without a recorded proof of
  equivalence.
- **Do not** access `MixedModels.jl` internals from any file other than
  `src/mm_internals.jl`, and not at all unless listed in its internal-access
  table.
- **Do not** bump the `MixedModels` version without walking the internal-access
  table and reviewing the `Manifest.toml` diff.
- **Do not** loosen a tolerance to make a test pass. Investigate the divergence.
- **Do not** use finite differences in shipped code — analytic or `ForwardDiff`
  only; finite differences are a test-time cross-check.
- **Do not** write Python-in-Julia: type-unstable code, abstractly-typed struct
  fields, manual loops where a clear vectorized form exists, unnecessary
  allocations in kernels.
- **Do not** use `@assert` to validate user-facing input — raise `ArgumentError`
  / `DomainError`.
- **Do not** catch an exception only to swallow it. Fail loudly.
- **Do not** add a runtime dependency without a `DECISIONS.md` entry. `RCall`
  stays test-only.
- **Do not** treat a singular fit as an unexpected error — it is a supported case.
