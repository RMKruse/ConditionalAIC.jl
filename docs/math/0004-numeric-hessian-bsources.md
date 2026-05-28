# 0004 — Numeric-Hessian B-sources: `:forwarddiff`, `:finitediff`, and `analytic = FALSE`

This note is the §7 step-1 "state the math" record for issue #11 (milestone M2). It pins
the two **numeric** Hessian **B**-sources of the Greven–Kneib correction — `:forwarddiff`
and `:finitediff` — and the `cAIC4` arithmetic they feed (`calculateGaussianBc` with
`analytic = FALSE`). It builds on, and does not restate, the closed-form `analytic = TRUE`
mathematics of [`0002-gaussian-bias-correction.md`](0002-gaussian-bias-correction.md); read
that first. The B-source *sourcing strategy* (which upstream surface each rides) is decided
in [ADR-0002](../adr/0002-bsource-ad-strategy.md); this note pins the *mathematics* those
choices commit us to and the empirical divergences they produce.

**Ground-truth sources consulted**
- `cAIC4` **v1.1**: `R/calculateGaussianBc.R` (the `analytic = FALSE` branch),
  `R/getModelComponents.R` (`model$B <- m@optinfo$derivs$Hessian`).
- `MixedModels` **= 5.5.1** (the exact pin): `src/linearmixedmodel.jl`
  (`objective`, `objective!`, `setθ!`, `updateL!`); `src/mixedmodel.jl`
  (`objective!(m)` curry); `ext/MixedModelsForwardDiffExt.jl` (`ForwardDiff.hessian`,
  `fd_deviance`); `ext/MixedModelsFiniteDiffExt.jl` (consulted as the reference for the
  self-driven driver — **not** used; ADR-0002).
- Greven, S. & Kneib, T. (2010), **Biometrika** 97(4), 773–789, Theorem 3.

Where `cAIC4` and any other source disagree, **`cAIC4` is ground truth** (CLAUDE.md §2).

---

## 1. What B is, and the one number that changes

The Greven–Kneib correction (`0002` §4–§5) contracts the cross-derivative `C` against
`Λ̂ʸ = B⁻¹C`, where `B` is the **positive-definite negative Hessian of the (restricted)
profile log-likelihood w.r.t. the variance parameters**. `0002` §4 fills `B` in closed form
(`analytic = TRUE`). The numeric B-sources instead obtain `B` by **differentiating
`MixedModels`' objective** w.r.t. `θ`, exactly as `cAIC4`'s `analytic = FALSE` lifts
`lme4`'s stored optimiser Hessian `m@optinfo$derivs$Hessian`.

`MixedModels`' objective (`objective(m)`, `src/linearmixedmodel.jl:836`) is the **deviance**
(`−2·profile log-likelihood`, ML) or the **REML criterion** (REML) — `−2×` the
log-likelihood `0002` §5 differentiates. The two are related by a constant factor of `2`;
`cAIC4`'s `analytic = FALSE` path absorbs that factor (and the unit change from the
analytic `B`) entirely inside its **rescaled `C`** (§2), so the *deviance* Hessian is plugged
in **unchanged**. Concretely: `analytic = TRUE` and `analytic = FALSE` use **different `C`**
and **different `B`** but the **same assembly**, and they are *not* algebraically identical —
they are two estimators of the same ρ that agree only up to the discrepancy between the
closed-form Hessian and the optimiser's numeric Hessian (§4).

---

## 2. `analytic = FALSE`: the rescaled C and the external B

Transcribed from `calculateGaussianBc(model, sigma.penalty, analytic = FALSE)`
(`R/calculateGaussianBc.R`, lines 59–92), with `np = n` (ML) or `np = n − p` (REML),
`p = ncol(X)`, and `tʸᵉ = yᵀe`:

```
C[j, :] = (2·np / tʸᵉ) · ( eᵀ Wⱼ A − (eᵀ Wⱼ e) · eᵀ / tʸᵉ ),        B = m@optinfo$derivs$Hessian,
```

then the **same** solve and assembly as `analytic = TRUE` (`0002` §4):

```
Λ̂ʸ = B⁻¹ C   (factorisation; Cholesky fallback when B is not numerically PD),
ρ  = [ n − tr(A) ]  +  Σⱼ  Λ̂ʸ[j, :] · (A Wⱼ e)  +  sigma.penalty.
```

Contrast with the `analytic = TRUE` `C` of `0002` §4: there the second term carries
`/(2 tʸᵉ)` and there is no outer factor; here the second term carries `/tʸᵉ` and the whole
row is scaled by `2·np/tʸᵉ`. Using `eᵀWⱼA = (A Wⱼ e)ᵀ` (both `A`, `Wⱼ` symmetric), the row
in the package's quantities is

```
C[j, :] = (2·np / tʸᵉ) · ( A Wⱼ e − (eᵀWⱼe / tʸᵉ) · e ).
```

Everything else in the assembly — `A Wⱼ e`, `n − tr(A)`, the contraction, `sigma.penalty`,
the unweighted `R = Iₙ ⟹ RA = A` target — is shared with `dof_lmm` (`0002` §4). The only
inputs that change are `C` (rescaled) and `B` (external, not computed from the components).
This is why `DofLMM.dof_lmm_numeric(c, B)` takes the **same** `GaussianComponents` plus an
externally-supplied `B`.

**Why no `WAlist`/trace term.** The `analytic = FALSE` branch never forms the Fisher trace
`tr(Wⱼ M Wₖ M)` (it lives only in the closed-form `B`); `B` arrives ready-made. So
`dof_lmm_numeric` does **not** touch `V0inv` and is `M`/ML-vs-REML-dependent **only** through
`np` in the `C` rescaling.

---

## 3. The two numeric B-sources (ADR-0002)

Both must hand `dof_lmm_numeric` a `B` on the **deviance scale** (the scale of
`m@optinfo$derivs$Hessian` and of `MixedModels`' `objective`), evaluated at the fitted `θ̂`.
The s×s shape (`s = length(θ̂)`) is asserted in `mm_internals.jl`: the experimental surface
warns that *whether σ and/or the fixed effects are differentiated alongside θ is still being
decided*, and that drift would silently change `B`'s dimension — the assertion turns it into
a loud error against the `=5.5.1` pin.

### 3a. `:forwarddiff` — rides `MixedModelsForwardDiffExt`

`ForwardDiff.hessian(m::LinearMixedModel)` = `ForwardDiff.hessian(fd_deviance(m), θ̂)`. This
is the **only** B-source on experimental surface (ADR-0002: AD through the in-place
objective fails, so we cannot self-drive ForwardDiff without reimplementing the extension).

**The σ-freezing divergence.** `fd_deviance(m, θ)` (`ext/MixedModelsForwardDiffExt.jl`) holds
the residual variance **fixed** at the fitted `σ̂²` (`σ² = model.σ^2`) while varying `θ`:

```
fd_deviance(m, θ) = dof·log(2π σ̂²) + logdet(θ) + pwrss(θ)/σ̂²       (σ̂² frozen).
```

The **stable** objective `objective!(m, θ)` instead **re-profiles** `σ²(θ) = pwrss(θ)/dof`
at every `θ` (`src/linearmixedmodel.jl:836`, the `σ = nothing` branch). At `θ = θ̂` the two
agree in value and (to first order) in gradient — `σ̂²` *is* the profiled `σ²(θ̂)` and the
gradient vanishes at the optimum — but their **Hessians differ**, because re-profiling `σ²`
contributes curvature in `θ` that freezing removes. So `:forwarddiff` returns the Hessian of
a *frozen-σ* deviance, **not** the profiled-deviance Hessian that `lme4`/`cAIC4` use. This is
an accepted, documented divergence (the user directed "ride the ext, document σ-freezing");
it is the source of the `:analytic`–`:forwarddiff`–`:finitediff` spread quantified in §4.

### 3b. `:finitediff` — self-driven over the stable objective

`cAIC.jl` drives `FiniteDiff.finite_difference_hessian` over the curried closure
`objective!(m)` (`= Base.Fix1(objective!, m)`, `src/mixedmodel.jl:145`), evaluated at `θ̂`,
**not** the experimental `MixedModelsFiniteDiffExt` (ADR-0002: FD only *evaluates* the
objective, so it can ride the long-stable `objective!`/`setθ!`/`updateL!` API and thereby
survive churn in the experimental surface — realising ADR-0001's "FD as the fallback when
AD fails"). Because `objective!` **re-profiles** `σ²(θ)`, the resulting Hessian is the
profiled-deviance Hessian — the *same* object `lme4` stores — so `:finitediff` reproduces
`cAIC4`'s `analytic = FALSE` ρ to FD accuracy (§4).

**Mutation contract (ADR-0002, non-negotiable).** `objective!(m, θ)` mutates `m`
(`setθ!` then `updateL!`); `FiniteDiff` evaluates it at perturbed `θ̂ ± h`, leaving `m`
parked at the last probe. The driver therefore **restores** `m` to `θ̂` in a `finally`
(`updateL!(setθ!(m, θ̂))`) and then **asserts** the restoration took (`m.θ == θ̂`) — a
`setθ!` left perturbed is a defect, and we fail loud rather than return a `CAICResult`
computed against a silently-mutated fit.

---

## 4. The cross-source landscape and the derived tolerances

The three B-sources are **three estimators of the same ρ**, not three computations of one
number. Their pairwise gaps are genuine (the bootstrap-vs-analytic precedent applies:
a real inter-estimator gap is recorded, never tolerance-papered). Two facts fix the
expectations the cross-source-agreement test encodes:

1. **`:finitediff` ≡ `cAIC4` `analytic = FALSE`** (both differentiate the profiled
   deviance), to finite-difference accuracy. This is a Level-2 *correctness* gate against
   the R fixture — a tight, fit-discrepancy-derived tolerance.
2. **`:analytic` vs `analytic = FALSE`** genuinely differ — the closed-form Hessian is not
   the optimiser's numeric Hessian — by an amount that grows with `s` and the curvature of
   the profile. **`:forwarddiff`** sits between them, displaced from `:finitediff` by the
   σ-freezing of §3a, again `s`-dependent.

The agreement tolerances are therefore **empirically derived and recorded in
`DECISIONS.md`** (the user directed "derived tolerances, recorded"), not asserted a priori.
The measured `sleepstudy` spread that sets them (random-intercept `s = 1` is tight;
correlated-slope `s = 3` is the widest) is tabulated in that `DECISIONS.md` entry alongside
its derivation; this note pins only the *structure* (which pairs must be tight, which are
expected to diverge and why), so that a future change to the numbers is a `DECISIONS.md`
event, not a silent edit here.

---

## 5. Numerical-stability obligations (CLAUDE.md §9)

- `Λ̂ʸ = B⁻¹C` via the **same** `_lambday` factorisation-with-Cholesky-fallback as
  `dof_lmm` (no explicit inverse); for a numeric `B` near the boundary the symmetric
  (Bunch–Kaufman) fallback is the relevant path.
- No new dense products beyond `A Wⱼ e` (shared with `dof_lmm`); the rescaled `C` is
  elementwise.
- The FD driver restores `θ̂` and fails loud on a left-perturbed fit (§3b) — a stability
  *and* correctness obligation: a mutated fit poisons every downstream quantity.

---

## 6. Provenance and what this enables

- **Versions pinned:** `cAIC4` v1.1 (`calculateGaussianBc` `analytic = FALSE`);
  `MixedModels = 5.5.1` (`objective`/`objective!`/`setθ!`/`updateL!` stable;
  `ForwardDiff.hessian`/`fd_deviance` experimental). A bump to either re-opens this note and
  the `mm_internals.jl` table.
- **Sufficient for the Level-1 numeric port:** §2 is the exact `analytic = FALSE` arithmetic
  `DofLMM.dof_lmm_numeric(c, B)` implements; the Level-1 fixture feeds a synthetic SPD `B`
  (parametrisation-neutral, ADR-0003) and compares against `cAIC4`'s `analytic = FALSE` ρ.
- **Sufficient for the B-sources:** §3 fixes the two upstream paths, the deviance scale, the
  s×s shape assertion, and the FD mutation/restore contract — the `MMInternals.bhessian`
  spec.
- **Sufficient for the cross-source test:** §4 fixes which pairs are correctness-tight
  (`:finitediff` ↔ R `analytic = FALSE`) and which are expected-divergent
  (`:analytic`/`:forwarddiff`), with the numbers deferred to `DECISIONS.md`.
