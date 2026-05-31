# ADR-0007 — M4.5 weight optimizer: faithful `solnp` transcription over a direct convex-QP solve

**Date:** 2026-05-31
**Status:** Accepted (design — grilled, not yet built)

## Context

M4.5 (model averaging) ports `cAIC4`'s `modelAvg` / `predictMA` / `summaryMA` / `getWeights`
(CLAUDE.md §11; CONTEXT.md *Averaging*). Its hard core is `getWeights`: the *Zhang-optimal weights*
minimise the Mallows-type criterion

```
J(w) = (y − μw)ᵀ(y − μw) + 2σ²(w·ρ)      subject to   Σ wᵢ = 1,  0 ≤ wᵢ ≤ 1,
```

where `μ` (n×M) stacks the candidates' conditional fitted means, `ρ` the per-candidate effective
df, and `σ²` the residual variance of the largest-df candidate. This is a **convex quadratic
program over the unit simplex** — `J` is convex (Hessian `2μᵀμ ⪰ 0`; the df term is linear), with a
*unique* minimiser whenever `μᵀμ ≻ 0`.

`cAIC4` does **not** solve it as a QP. `getWeights` + `.weightOptim` are a line-for-line
transcription of `Rsolnp::solnp` — an augmented-Lagrangian SQP with parameter/constraint scaling, a
BFGS Hessian update, a bisection line search, and four `try`-error fallbacks. One inner step
(`R/weightOptim.R:154`, `cz <- try(solve(cz))`) **explicitly inverts** the Cholesky factor
`cz = chol(hess + λ·D²)` and reuses it in matmuls.

Two project principles bear directly:

- **CLAUDE.md §9 / §12 (non-negotiable):** never form an explicit inverse; use factorisation-based
  solves. `solnp`'s line 154 violates this.
- **CLAUDE.md §1 / §2:** `cAIC4` is ground truth; never rewrite a formula without a recorded proof
  of equivalence.

The objective is identical in every option below — only *how it is minimised* differs.

## Decision

1. **Faithfully transcribe `cAIC4`'s `solnp`-based optimizer** (`getWeights` + the internal
   `.weightOptim`) rather than solving the convex QP directly or via a solver dependency. Bit-level
   *algorithmic* parity with `cAIC4` is the goal; the chosen basis is **auditability** — a Julia
   port that maps 1:1 onto `weightOptim.R` can be diffed against the R source line-for-line, whereas
   an algebraic substitution (even a provably-equivalent QP solve) breaks that correspondence and
   forces a future maintainer to re-derive the equivalence. No new runtime dependency is incurred
   (`LinearAlgebra` only).

2. **Keep the explicit `inv` of the Cholesky factor (line 154) literally** — a *documented,
   deliberate carve-out from §9/§12 for this one line*. Rationale: it preserves the 1:1
   transcription that is the whole basis of decision (1). The alternative (a provably-equivalent
   triangular solve `R \ v`) was considered and **rejected**: cross-language bit-parity is
   unattainable regardless (different BLAS, float-reduction order, and `solnp`'s iterative path), so
   the triangular solve would match `cAIC4` to *the same* roundoff while honouring §9 — i.e. it
   gives up nothing measurable — but it was judged not worth the break in source correspondence.
   This carve-out is **scoped to this single line** of this single function; the §9 ban on `inv`
   stands everywhere else in the codebase.

3. **Transcribe the algorithm faithfully, but do not transcribe upstream *defects*.** `cAIC4` feeds
   the optimizer `df` rounded to two decimals (an `anocAIC` print-formatting artifact leaking into
   `getWeights`, `R/methods.R:63`). The port uses **full-precision** df — a documented divergence
   (DECISIONS), on the principle "faithful to the algorithm, not to a bug." The literal `inv` (a
   genuine step of the algorithm) and the full-precision df (a fix to an accidental data-pipeline
   defect) are deliberately treated differently.

4. **The `try`-error fallbacks are preserved, but warn.** On an ill-conditioned/collinear candidate
   set the inner `chol`/`solve`/`qr.solve` can fail; `cAIC4` silently returns the current iterate.
   The port reproduces that degradation (it is the algorithm's own fallback, not error-swallowing)
   but emits a `@warn` that the weight problem was ill-conditioned and the optimum may be
   non-unique — satisfying §4 ("if you catch, you handle or rethrow with context"; never a silently
   wrong number).

## Consequences

- **No new dependency**; `src/mm_internals.jl` is untouched (`getWeights` reads `response`,
  `fitted`/conditional μ, `sigma`, and the effective df from `caic`/`anocaic` — public API, not
  internals). The internal-access table is unchanged.
- **The §9 carve-out is real and must stay visible.** The transcription carries a prominent comment
  at the `inv` site pointing here, so a future reader does not "fix" it (or read it as licence to
  use `inv` elsewhere). A `JET`/lint exception, if any is needed, is localised to this function.
- **Validation cannot be bit-match.** The optimizer transcription is gated at **Level-1** on shared
  synthetic inputs `(y, μ, ρ, σ²)` against `cAIC4`'s `getWeights`/`.weightOptim` arithmetic — feeding
  *identical* df both sides isolates the transcription from the full-precision-df divergence (3).
  **Level-2** (end-to-end `modelAvg`) anchors the *stable functionals* — the model-averaged
  prediction (`predictMA`) and the objective value `J(ŵ)` — within a band = max(lme4↔MixedModels
  fit discrepancy, full-precision-df perturbation); the **weight vector itself** is anchored only on
  a deliberately well-conditioned scenario (`μᵀμ ≻ 0`, non-collinear candidates → unique minimiser),
  because on the common nested-candidate case the minimiser is non-unique and `solnp` and the port
  may legitimately reach different optima of equal objective (the M4.5 analogue of `stepcaic`'s
  "path only on well-separated cases", DECISIONS 2026-05-30). Tolerances are *measured* and recorded
  in DECISIONS at implementation time.
- **Reversibility.** If the transcription proves a maintenance burden, the convex-QP solve remains a
  drop-in replacement (same objective, same `WeightResult`), validated against the same fixtures —
  at the cost of the documented divergence widening to whatever gap separates `solnp`'s converged
  weights from the true optimum.

## Alternatives considered

- **Direct convex-QP solve** (dependency-free active-set / KKT). Honours §9 cleanly, more robust on
  ill-conditioned sets, returns the *true* minimiser. Rejected for M4.5 in favour of source
  correspondence (decision 1); retained as the documented escape hatch above.
- **QP via a solver dependency** (OSQP/COSMO/`Optim` Fminbox). Robust and correct, but adds a runtime
  dependency (DECISIONS entry) against the §3 minimal-deps mandate for what is a small dense QP.
- **Triangular solve at line 154** (provably equivalent, §9-compliant). Rejected per decision (2):
  matches `cAIC4` no better than the literal `inv` while breaking the 1:1 transcription.
