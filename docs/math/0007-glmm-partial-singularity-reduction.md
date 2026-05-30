# 0007 — GLMM partial-singularity reduction: `reduceboundary` for `GeneralizedLinearMixedModel`

This note is the §7 step-1 "state the math" record for issue #32 (milestone M3). It pins,
in precise notation, the **partial** boundary reduction for a fitted
`GeneralizedLinearMixedModel` — the GLMM analogue of one level of `cAIC4`'s
`deleteZeroComponents.merMod` — **before the corresponding Julia code is written**. The
full-singularity fallback (every variance component on the boundary → plain GLM, ρ =
rank(X)) is already pinned in [`0006`](0006-glmm-bias-correction.md) §5 and shipped (issue
#27); this note covers the *intermediate* case it left open: **some but not all** variance
components on the boundary.

The conditional-AIC assembly is unchanged ([`0006`](0006-glmm-bias-correction.md) §0):

```
cAIC = −2 · ℓ_cond(y | b̂, β̂, θ̂) + 2 · ρ ,
```

with both `ℓ_cond` and `ρ` evaluated on the **reduced** model this note produces, exactly
as `cAIC4` evaluates `getcondLL`/`biasCorrection*` on the output of `deleteZeroComponents`
(`R/cAIC.R:270–271`; `R/biasCorrectionPoisson.R:13`, `R/biasCorrectionBernoulli.R:10`).

**Ground-truth sources consulted** (read from source, not memory — memory record
*verify-caic4-against-source*)
- `cAIC4` **v1.1** (CRAN, 2025-04-04): `R/deleteZeroComponents.R`
  (`deleteZeroComponents.merMod` — the one method serving **both** `lmerMod` and `glmerMod`),
  `R/biasCorrectionPoisson.R:13–16`, `R/biasCorrectionBernoulli.R:10–13` (both call
  `deleteZeroComponents` first and short-circuit on the `glm` result), `R/cAIC.R:270–271`
  (the reduced model becomes the scored model).
- Greven, S. & Kneib, T. (2010). On the behaviour of marginal and conditional AIC in linear
  mixed models. *Biometrika* 97(4), 773–789. (`deleteZeroComponents`' own cited justification
  for dropping boundary components before bias correction.)
- Säfken, B., Rügamer, D., Kneib, T. & Greven, S. (2021). Conditional Model Selection in
  Mixed-Effects Models with `cAIC4`. *JSS* 99(8).
- [`0002`](0002-gaussian-bias-correction.md) §1 (the LMM "Precondition (singular fits)"
  paragraph — the reduction the Gaussian path already assumes), [`0006`](0006-glmm-bias-correction.md)
  §5 (the full-singularity fallback this note recurses into), and the project memory records
  *reduced-model-reconstruction* (the column-subset ReMat technique, verified bit-for-bit) and
  *bootstrap-not-equal-analytic*.

Where any source disagrees, **`cAIC4` is ground truth** (CLAUDE.md §2).

---

## 0. The fitted GLMM and its boundary structure

We are handed a `GeneralizedLinearMixedModel` `m` already fitted by `MixedModels.jl`. It
wraps a working `LinearMixedModel` `m.LMM` carrying the random-effects terms `reterms`, the
fixed-effects term `feterm`, and the relative-covariance factors `λ` (one per reterm),
together with a `GLM.GlmResp` `m.resp` holding the family/link and the response `y`.

Write the reterms as `t = 1 … R`, each with `Sₜ` random-effect directions and a relative
covariance factor `λₜ` (an `Sₜ × Sₜ` lower-triangular or diagonal matrix in the
`MixedModels` parametrisation). A direction `d` of reterm `t` is **on the boundary** iff its
diagonal entry vanishes:

```math
\lambda_{t}[d,d] = 0 .
```

This is the `MixedModels` analogue of `cAIC4`'s `which(diag(getME(m,"ST")[[t]]) == 0)`
(`R/deleteZeroComponents.R`, the `merMod` method): `ST` is the relative-covariance factor and
its zero diagonal entries are exactly the components estimated on the boundary. We test the
**exact** `λ[d,d] == 0`, matching `cAIC4`'s `which(theta == 0)` / `diag(ST) != 0` — not a
tolerance; `MixedModels`' optimiser snaps sub-`xtol_zero_abs` positive θ to exactly `0` at the
end of `fit!`, so the boundary directions carry a hard zero. The set membership question —
*"is this fit singular at all?"* — is `MixedModels.issingular(m)` (quarantined as
`MMInternals.issingular`), used by the caller to decide whether to enter the reduction at all.

---

## 1. The reduction estimand (one level)

Let `Kₜ = {d ∈ 1…Sₜ : λₜ[d,d] ≠ 0}` be the surviving directions of reterm `t`. The
single-level reduction `reduceboundary` produces a **new** GLMM `m̃` with:

- each reterm `t` for which `Kₜ ≠ ∅` replaced by its restriction to the surviving directions
  `Kₜ` (a *partial* drop when `∅ ⊊ Kₜ ⊊ {1…Sₜ}`, an *unchanged* term when `Kₜ = {1…Sₜ}`);
- each reterm `t` with `Kₜ = ∅` (every direction on the boundary) **removed entirely**;
- the same family, link, fixed-effects term `feterm`, response `y`, and prior weights;
- the reduced GLMM **refitted** to maximise its own conditional/Laplace likelihood.

Formally, if `Z = [Z₁ … Z_R]` is the random-effects design partitioned by reterm and `Zₜ[Kₜ]`
denotes the columns of `Zₜ` for the surviving directions, the reduced model's random-effects
design is

```math
\tilde Z = \bigl[\, Z_t[K_t] \;:\; t = 1\dots R,\ K_t \neq \emptyset \,\bigr],
```

with the relative-covariance factors reset to the identity on each surviving block and
re-estimated from scratch. The fixed-effects design `X` is untouched.

This is exactly one application of `deleteZeroComponents.merMod`: drop the boundary columns
(`cnms[[i]] <- cnms[[i]][which(diag(ST[[i]]) != 0)]`), rebuild the random-effects formula on
the survivors (`cnms2formula`), keep the fixed part (`nobars(formula(m))`), and `update` the
fit. Because `deleteZeroComponents` has a *single* `merMod` method, the Poisson/Bernoulli/
binomial GLMM reduction is the **same operation** as the Gaussian LMM reduction already pinned
in [`0002`](0002-gaussian-bias-correction.md) §1 — the only GLMM-specific work is rebuilding
the *generalized* fit object around the reduced working LMM and refitting it with the GLMM
likelihood rather than the LMM one (§3).

---

## 2. Recursion and termination

`deleteZeroComponents` ends by calling **itself** on the updated model
(`return(deleteZeroComponents(newMod))`): a reduced fit may itself land on the boundary, so the
reduction iterates until a fixed point. `cAIC4` expresses this as direct recursion inside
`deleteZeroComponents`; `cAIC.jl` expresses it as the caller's loop — `caic` calls
`reduceboundary` repeatedly while `issingular` holds, exactly as the LMM spine already does
(the *reduced-model-reconstruction* memory record; [`0002`](0002-gaussian-bias-correction.md)
§1). `reduceboundary` performs **one** level; the cascade lives in `caic`.

Three terminating outcomes, matching `cAIC4`:

| outcome | `cAIC4` (`deleteZeroComponents.merMod`) | `reduceboundary` / `caic` |
|---|---|---|
| no boundary direction (`which(θ==0)` empty) | `return(m)` unchanged | caller's `issingular` is `false`; loop never enters (or stops). Score the current model. |
| **some** boundary directions (`0 < #{θ=0} < length(θ)`) | drop them, `update`, recurse | `reduceboundary` returns the refitted reduced GLMM `m̃`; caller re-checks `issingular(m̃)` and may reduce again. |
| **all** directions on the boundary (`length(θ)==#{θ=0}`) | `return(lm(nobars(formula(m)), …))` — a plain `glm`/`lm` | `reduceboundary` returns `nothing`; caller takes the full-singularity fallback ρ = rank(X), [`0006`](0006-glmm-bias-correction.md) §5. |

The "all on the boundary" case returns `nothing` rather than constructing a degenerate
zero-reterm GLMM: there is no random-effects model left, and the score is the fixed-effects-
only ρ = rank(X) of [`0006`](0006-glmm-bias-correction.md) §5 (no `+1` σ-penalty for
canonical-link families). This mirrors `reduceboundary(::LinearMixedModel)` returning `nothing`
on full collapse.

Termination is guaranteed: each non-terminating level removes at least one random-effect
direction (`#{θ=0} ≥ 1` by the entry condition), so the total direction count `Σₜ Sₜ` strictly
decreases and the recursion reaches either a non-singular fit or the empty model in at most
`Σₜ Sₜ` steps.

---

## 3. The reduced-fit reconstruction (numerical realisation)

`MixedModels.jl` has no `update`-by-formula path that re-derives a GLMM from a reduced
random-effects formula, so the reduced fit is **reconstructed** from the fitted object's
internals (quarantined in `mm_internals.jl`), then refitted. This is the same column-subset
ReMat technique the LMM path uses (memory: *reduced-model-reconstruction*, verified bit-for-bit
against native fits), lifted to the GLMM wrapper:

1. **Reduced reterms.** For each reterm with `Kₜ ≠ ∅`, rebuild a fresh `ReMat` keeping only the
   `Kₜ` design rows, with `λₜ` reset to the identity (preserving its structure — `Diagonal` for
   an uncorrelated/`zerocorr` term, `LowerTriangular` for a correlated one, and
   `LowerTriangular` for any single-direction survivor, since `MixedModels`' `ReMat{T,1}`
   stores a `1×1` `LowerTriangular`). This is the shared `_subsetreterm` helper.
2. **Reduced working LMM.** Build `LinearMixedModel(y, m.LMM.feterm, reducedreterms,
   m.LMM.formula, weights)`, where `weights` is the prior weights `m.wt` when the original fit
   was weighted and a length-`n` vector of ones otherwise — `MixedModels`' own GLMM constructor
   rebuilds the working LMM with unit weights for an unweighted GLMM, and omitting them yields
   an empty `sqrtwts` and a *different* (wrong) working response, so the reconstruction must
   supply them explicitly.
3. **Reduced GLMM wrapper.** Assemble a `GeneralizedLinearMixedModel{T,D}` around the reduced
   working LMM, reusing the family/link from a copy of `m.resp`, with the random-effects scratch
   vectors `u`/`u₀` sized to the reduced reterms.
4. **Refit.** Refit the reduced GLMM with `fit!(m̃; fast=false, nAGQ=1)` — the standard Laplace
   fit, matching `glmer`'s default and the LMM path's "refit with defaults" (project decision,
   issue #32; recorded in `DECISIONS.md`). `fast=false` re-optimises `(β, θ)` jointly under the
   Laplace approximation, the same objective `glmer`/`deleteZeroComponents`'s `update` re-solves.

The reconstruction is validated bit-for-bit against a native `MixedModels` fit of the reduced
model (`θ ≈`, `objective ≈` at `1e-7`) — the established project technique — before any cAIC is
read off it.

### 3.1 Numerical-stability obligations (CLAUDE.md §9)
- **Exact-zero boundary test**, not a tolerance: `λ[d,d] == 0` mirrors `which(theta == 0)`;
  loosening it would drop near-boundary-but-nonzero components `cAIC4` keeps, changing ρ.
- **No state sharing.** Every surviving reterm is rebuilt fresh and `λ` reset to identity, so
  the reduced fit shares no mutable buffer with `m`; refitting cannot corrupt the input model.
- **Fail loud on internal drift.** The reconstruction asserts the shape/type of each internal it
  pulls from `m` (the `reterms` are `ReMat`s, `m.resp` carries the family `D`, the weights are a
  length-`n` vector), per CLAUDE.md §3 — a silent upstream change surfaces as a clear error, not
  a wrong cAIC.
- **Singular fits are first-class.** Entering the reduction is the *expected* path for a singular
  GLMM, not an error; the empty-model outcome routes to the documented rank(X) fallback.

---

## 4. Provenance and what this enables

- **Versions pinned:** `cAIC4` v1.1 (`deleteZeroComponents.merMod`, `biasCorrectionPoisson`,
  `biasCorrectionBernoulli`, `cAIC`); `MixedModels` = 5.5.1 (`GeneralizedLinearMixedModel{T,D}`
  and its low-level constructor, `m.LMM`, `m.resp`, `m.β`, `m.wt`, `reterms`/`feterm`/`λ`,
  `fit!` with `fast`/`nAGQ`, `vsize`, `nlevs`). A bump to either re-opens this note and the
  `mm_internals.jl` internal-access table.
- **Enables `reduceboundary(::GeneralizedLinearMixedModel)`** in `mm_internals.jl` (§1–§3),
  Level-1-testable by bit-for-bit reconstruction against a native reduced fit and by the
  recursion/termination contract of §2.
- **Enables the partial-singularity spine in `caic(::GeneralizedLinearMixedModel)`**: the
  `while issingular → reduceboundary → recurse` loop between the full-singularity check and the
  non-singular scoring of [`0006`](0006-glmm-bias-correction.md) §3–§5, mirroring the LMM spine.
- **Level-2 cross-check:** a partially-singular Bernoulli `zerocorr(1 + x | g)` fit where both
  `cAIC4` and `cAIC.jl` reduce to `(1 | g)` and the assembled cAIC agrees at `atol = 1e-3` (the
  fixture seed and the divergence in *which* fits each ecosystem flags singular are recorded in
  `DECISIONS.md`). No Julia code is written by this note (issue #32 step-1 acceptance:
  documentation only).
