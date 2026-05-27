# 0003 — Gaussian conditional log-likelihood ℓ(y | b̂, β̂, θ̂)

This note is the §7 step-1 "state the math" record for issue #6 (milestone M2). It pins,
in precise notation, the **conditional log-likelihood** — the first term of the conditional
AIC — *before any Julia code is written*. It is the analogue of `cAIC4`'s `getcondLL`
(exported), surfaced in `cAIC.jl` as the pure kernel `Loglik.condloglik` (later the
`CAICResult.condloglik` field, PARITY.md).

`loglik` is validated here at **Level-1 against hand-computed values on synthetic inputs**
(issue #6 acceptance); the `cAIC4` cross-check happens later at **Level-2** through the
assembled `caic` (issue #8), where the kernel is fed the fit's own conditional mean.

**Sources consulted**
- Greven, S. & Kneib, T. (2010), *On the behaviour of marginal and conditional AIC in
  linear mixed models*, **Biometrika** 97(4), 773–789, eq. (1).
- `cAIC4` **v1.1** (CRAN, 2025-04-04): `R/getcondLL.R` (`getcondLL.lmerMod` — the Gaussian
  conditional log-likelihood as `sum(dnorm(y, mean = fitted, sd = sigma, log = TRUE))`),
  `R/cAIC.R`.
- [`docs/math/0002-gaussian-bias-correction.md`](0002-gaussian-bias-correction.md) §0–§1
  (the fitted LMM, ŷ = X β̂ + Z b̂, the conditional residual e = y − ŷ, the unweighted
  R = Iₙ M2 target).

---

## 1. The model and the conditioning

For the Gaussian `LinearMixedModel` of [0002](0002-gaussian-bias-correction.md) §0,

```
y = X β + Z b + ε,     b ~ N(0, σ² D*),   ε ~ N(0, σ² Iₙ),   b ⫫ ε,
```

the **conditional** distribution of the response given the random effects fixes `b` at its
predictor `b̂` (and `β` at `β̂`):

```
y | b̂, β̂  ~  N( ŷ, σ² Iₙ ),        ŷ = X β̂ + Z b̂   (the conditional fitted mean, `mu`).
```

This is "conditional" in the project's sense (CONTEXT.md): the likelihood of `y` given the
**predicted** (not integrated-out) random effects — contrast the *marginal* likelihood,
which integrates `b` out under `N(0, σ²D*)`. The conditional covariance is the *residual*
covariance `σ² Iₙ` — the unweighted M2 target (all residual weights 1, R = Iₙ;
[0002](0002-gaussian-bias-correction.md) §3/§8; weighted Gaussian is "not yet implemented"
in `cAIC4`).

---

## 2. The estimand

The conditional log-likelihood is the sum of independent univariate Gaussian
log-densities (the conditional covariance `σ² Iₙ` is diagonal, so the joint density
factorises over observations):

```
ℓ(y | b̂, β̂, θ̂)  =  Σᵢ log φ(yᵢ; ŷᵢ, σ̂²)
                =  Σᵢ [ −½ log(2π) − log σ̂ − (yᵢ − ŷᵢ)² / (2 σ̂²) ]
```

where `φ(·; μ, σ²)` is the `N(μ, σ²)` density and `σ̂` is the residual **standard
deviation** (`MixedModels.sigma(m)`, `cAIC4`/`lme4` `sigma(object)`; the `sd` argument of
`dnorm`, not the variance). Writing `eᵢ = yᵢ − ŷᵢ` for the conditional residual and `n` for
the number of observations, the aggregated closed form is

```
ℓ  =  − (n/2) log(2π)  −  n log σ̂  −  (1 / (2 σ̂²)) Σᵢ eᵢ² .          (★)
```

(`n log σ̂ = (n/2) log σ̂²`.) This `(★)` is exactly `cAIC4::getcondLL.lmerMod`:
`sum(dnorm(y, fitted, sigma, log = TRUE))`.

`ℓ` depends on the fit only through `(y, ŷ, σ̂)`. The interface therefore takes the
**conditional fitted mean ŷ directly** — `condloglik(y, ŷ, σ̂)` — rather than reconstructing
`ŷ = X β̂ + Z b̂` from design pieces: this mirrors `getcondLL` (which uses `fitted(object)`)
and avoids a second, divergent computation of a mean `MixedModels` already provides as
`mu`. The scoring spine (#8) extracts `mu` and `sigma` from the fit and passes them.

---

## 3. Stable form (shipped) and reference (test)

- **Stable form (shipped):** `(★)` evaluated directly. It is already numerically benign and
  honours CLAUDE.md §9 with no work:
  - **log-space** — densities enter as `log φ`, never as a product of `exp`'d small
    numbers; the `−n log σ̂` term is `log`-space (and `−n log σ̂`, not `−(n/2) log σ̂²`,
    avoids squaring before the log).
  - **no explicit inverse / no `det`** — the general multivariate-Gaussian log-density
    `−(n/2)log(2π) − ½ logdet(Σ) − ½ eᵀ Σ⁻¹ e` with conditional covariance `Σ = σ̂² Iₙ`
    has `logdet(Σ) = 2n log σ̂` and `eᵀ Σ⁻¹ e = (Σ eᵢ²)/σ̂²`; the diagonal collapses
    `logdetpd`/`invquad` to scalars, so **no Numerics primitive is invoked** and no
    matrix is formed. (Forming `cholesky(σ̂² Iₙ)` to "use a primitive" would be wasteful
    and is *not* what §9 asks for — the scalar form *is* the stable form here. A
    `numerics`-primitive route reappears only for the weighted/general covariance, which
    is out of M2 scope.)
  - `Σᵢ eᵢ²` is accumulated **without materialising** `y − ŷ` (a fused reduction over
    `eachindex`), in the promoted floating element type.

- **Naive reference (test):** the per-observation sum
  `Σᵢ (−½ log(2π) − log σ̂ − (yᵢ − ŷᵢ)²/(2 σ̂²))` — the estimand of §2 written one term per
  observation, a *different arrangement* from the aggregated `(★)`, so the test cross-checks
  the aggregation rather than re-stating it. Pinned hand-computed scalars anchor it
  absolutely (e.g. a perfect fit `ŷ = y` with `n = 1`, `σ̂ = 1` gives `ℓ = −½ log(2π) ≈
  −0.9189385332`).

Level-1 tolerance, as for the numerics primitives: `rtol = 1e-6`, `atol = 1e-10`.

---

## 4. Domain, edge cases, and failure

- **`σ̂` must be a positive real.** `σ̂ ≤ 0` (a variance/standard-deviation outside its
  domain) and `σ̂` non-real-valued are invalid → **`DomainError`** (CLAUDE.md §4: a value
  outside a mathematical domain raises `DomainError`; never a silently-wrong number). `NaN`
  fails the `σ̂ > 0` test and so is rejected as invalid input.
- **`length(y) == length(ŷ)`** is required → otherwise **`DimensionMismatch`**.
- **Empty input (`n = 0`)** is the empty sum: `ℓ = 0` (consistent with the empty-contraction
  convention of `numerics.traceprod`). Not an error.
- **Non-finite data** (`NaN`/`Inf` in `y` or `ŷ`) **propagates** into `ℓ` rather than being
  silently dropped (matching the `numerics` primitives' NaN/Inf contract). A perfect fit
  (`e = 0`) yields the finite maximum `ℓ = −(n/2)log(2π) − n log σ̂`.
- **Generic over `T <: AbstractFloat`.** The result type is the promoted floating type of
  `eltype(y)`, `eltype(ŷ)`, and `typeof(σ̂)`; the `log(2π)` constant is evaluated at that
  precision (`log(2 * T(π))`), so a `Float32` input returns a `Float32` (no silent Float64
  promotion).

---

## 5. Provenance and what this enables

- **Versions pinned:** `cAIC4` v1.1 (`getcondLL.lmerMod`); `MixedModels` = 5.5.1 (`sigma`,
  `fitted`/`mu`). A bump re-opens this note.
- **Enables #6:** the pure kernel `Loglik.condloglik(y, ŷ, σ̂)` and its Level-1
  hand-computed tests.
- **Enables #8 (scoring spine):** `cAIC = −2 ℓ + 2 ρ` ([0002](0002-gaussian-bias-correction.md)
  §1) calls this kernel on the fit's extracted `(y, mu, sigma)`; that is where the Level-2
  `cAIC4` cross-check lands. The `mu`/`sigma` extraction is the `mm_internals` touchpoint,
  exercised at Level-2 (ADR-0003), not here.
