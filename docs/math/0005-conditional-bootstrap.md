# 0005 — Conditional bootstrap df: Efron's covariance penalty (`method = :bootstrap`)

This note is the §7 step-1 "state the math" record for issue #12 (milestone M2). It pins,
in precise notation, the bias-correction mathematics the `method = :bootstrap` path
implements: the parametric-bootstrap draw, the per-draw refit, and Efron's covariance
penalty assembly. It also records — explicitly, not silently — the four places where the
shipped Julia formula departs from `cAIC4`'s `conditionalBootstrap`, with a disposition
for each (code change vs. recorded divergence in `DECISIONS.md`). It is the gate the
estimator should have passed *before* `efron_penalty`, `bootstrapfit`, and the `_bootstrap`
spine were written; written after-the-fact, its job is to surface those decisions.

**This note was written after the estimator shipped (commit `fc8fd02`, issue #12).** That
was a process failure against CLAUDE.md §7 step 1; the formula divergences in §5 were the
direct cost of not having had this gate up front. The §5 dispositions #1, #2, and #3 were
closed on 2026-05-28 with the addition of the Level-1 shared-input fixture against
`cAIC4::conditionalBootstrap` (see DECISIONS.md and `test/generate_fixtures_bootstrap.{jl,R}`).
#4 remains a UX-only open item that does not block correctness.

**Ground-truth sources consulted**
- `cAIC4` **v1.1** (CRAN, 2025-04-04): `R/conditionalBootstrap.R` (the estimator);
  `R/bcMer.R` (dispatch — confirming `sigma.penalty` is *not* passed to the bootstrap path,
  and `B` defaults to `max(n, 100)`); `R/cAIC.R` (the public entry).
- `lme4`: `simulate.merMod(..., use.u = TRUE)` (parametric-bootstrap draw with the fitted
  random effects held in the conditional mean); `refit.merMod(object, newresp)` (the
  per-draw refit).
- `MixedModels` **= 5.5.1** (the exact pin): `LinearMixedModel`, `fit!`, the
  `feterm`/`reterms`/`formula` accessors used by `MMInternals.bootstrapfit` to construct a
  fresh model on `y*`; `response`, the conditional mean accessor `conditionalmean`,
  `sigmahat` from the quarantine.
- Efron, B. (2004). The estimation of prediction error: covariance penalties and
  cross-validation. *JASA* 99(467), 619–632. Equation (2) is the population covariance
  penalty; the bootstrap form here is its parametric-bootstrap estimator.

Where `cAIC4` and any other source disagree, **`cAIC4` is ground truth** (CLAUDE.md §2).
Where the shipped Julia code disagrees with `cAIC4`, §5 records the gap and its
disposition; this note does not paper over the disagreement.

---

## 1. The population quantity

For the Gaussian conditional model `y | b ~ N(Xβ + Zb, σ²Iₙ)` (`0002` §0), the effective
degrees of freedom enter the conditional AIC as `ρ = (1/σ²) Σᵢ cov(yᵢ, ŷᵢ)`, where the
covariance is taken over the joint law of `(y, ŷ)` induced by the model and the estimator
producing `ŷ` (Efron 2004, eq. 2). For a linear estimator `ŷ = H y`, `ρ = tr(H)`; the
mixed-model `ŷ` is *not* linear in `y` once `θ` is estimated, so `ρ` exceeds the naive
plug-in `ρ₀ = tr(H₁)` of `0002` §5. The closed-form correction is the Greven–Kneib
expansion (`0002` §4); the **bootstrap** estimates the same ρ directly by simulating the
joint law and computing a sample covariance.

---

## 2. The parametric bootstrap draw and the per-draw refit

The draws condition on the fitted random effects `b̂` (`lme4`'s `use.u = TRUE`), so the
mean of `y*` is the **conditional** fitted mean `ŷ = Xβ̂ + Zb̂` — the same `ŷ` the
conditional log-likelihood is built around (`0003` §1), and the same `ŷ` `0002` §5 uses as
the centring point of `ρ₀ = n − tr(A)`. With `n` observations, `B` draws, and `ε(b) ~
N(0, Iₙ)` independent,

```
y*(b) = ŷ + σ̂ ε(b),     b = 1, …, B.            (parametric, conditional-on-b̂ bootstrap)
```

For each draw, **refit the model from scratch on `y*(b)`** — same design (`X`, the
`reterms`, the `formula`), same REML/ML setting, fresh `θ` optimisation — and read off the
new conditional fitted mean

```
ŷ*(b) = Xβ̂*(b) + Z b̂*(b).
```

In the package this is `MMInternals.bootstrapfit(m, y*(b))`. The refit is the
expensive step: `B` full LMM fits per `caic(...; method = :bootstrap)` call. Reusing the
original `feterm`/`reterms`/`formula` (rather than re-parsing the formula on the original
table with the response swapped) is a deliberate `MixedModels.jl`-side simplification of
`lme4`'s `refit(object, newresp = x)`; it preserves the design and the parametrisation,
which is all the Efron estimator needs.

---

## 3. `cAIC4`'s estimator — the ground-truth formula

Transcribed from `R/conditionalBootstrap.R` (lines 15–26):

```r
dataMatrix <- simulate(object, nsim = BootStrRep, use.u = TRUE)        # y*(b), n×B
workingEta <- sapply(dataMatrix, predict(refit(object, newresp = x)))  # ŷ*(b), n×B
dataMatrix <- dataMatrix - rowMeans(dataMatrix)                        # centre y* row-wise
bootBC     <- sum(workingEta * dataMatrix) / ((BootStrRep - 1) * sigma(object)^2)
```

In notation, with `ȳ*ᵢ = (1/B) Σ_b y*(b)ᵢ`,

```math
\rho_{\mathrm{cAIC4}} =
  \frac{1}{(B - 1)\,\hat\sigma^{2}}
  \sum_{b = 1}^{B} \sum_{i = 1}^{n}
    \hat y^{*}(b)_{i} \, \bigl(y^{*}(b)_{i} - \bar y^{*}_{i}\bigr).
```

Two algebraic simplifications:

- The cross-term `Σ_b (y*(b)ᵢ − ȳ*ᵢ) · constant_i` vanishes for any `i`-constant, so the
  expression is equivalently a fully-centred sample covariance:
  ```
  ρ_cAIC4 = (1 / ((B − 1) σ̂²)) · Σᵢ Σ_b (y*(b)ᵢ − ȳ*ᵢ) (ŷ*(b)ᵢ − ȳ̂*ᵢ).
  ```
  This is the standard **bootstrap sample covariance** estimator of `cov(y, ŷ)` with the
  unbiasedness divisor `B − 1`.
- No `sigma.penalty` enters; `bcMer.R` calls `conditionalBootstrap(object, B)` with no
  `sigma.penalty` argument, and `conditionalBootstrap` does not add one (only the analytic
  path's `biasCorrectionGaussian` does).

`cAIC4`'s default for `B` is `max(n, 100)` (`bcMer.R` lines 54–56).

---

## 4. The package's estimator — what is actually shipped

`DofLMM.efron_penalty(yhat, sigma, Ystar, Yhatstar, sigmapenalty = 0)` (Level-1, pure)
implements the **same** arithmetic as `cAIC4` in §3:

```math
\rho_{\mathrm{efron\_penalty}} =
  \frac{1}{(B - 1)\,\hat\sigma^{2}}
  \sum_{b = 1}^{B} \sum_{i = 1}^{n}
    \hat y^{*}(b)_{i}\, \bigl(y^{*}(b)_{i} - \bar y^{*}_{i}\bigr)
  + \texttt{sigmapenalty},
```

with `ȳ*ᵢ = (1/B) Σ_b y*(b)ᵢ`, `B ≥ 2`, and the default `sigmapenalty = 0`. The `yhat`
argument is *unused arithmetically* — carried for signature symmetry with the analytic
and numeric Level-1 units. The `_bootstrap` spine in `src/scoring.jl` constructs the
draws `y*(b) = ŷ + σ̂ε(b)` (matching `lme4`'s `simulate(..., use.u = TRUE)`), refits each
through `MMInternals.bootstrapfit` to obtain `ŷ*(b)`, calls `efron_penalty(...,
sigmapenalty)` with the user-supplied `sigmapenalty` (default `1`), and returns ρ. The
spine therefore wraps the cAIC4 bare arithmetic in the package's σ²-parameter-count
convention — see §5 disposition #3.

---

## 5. The four divergences from `cAIC4`, and the disposition for each

Pinning the math forced honesty about where the implementation drifted from ground truth.
Four gaps existed between §3 and the original §4. None has been silently absorbed: each
is either closed in code or recorded in `DECISIONS.md`. Dispositions #1, #2, #3 were
**resolved on 2026-05-28** when the Level-1 shared-input fixture against
`conditionalBootstrap` was added (see the new DECISIONS.md entry of that date); #4
remains an open UX decision that is not a correctness gate.

| # | Quantity         | `cAIC4` (ground truth) | `cAIC.jl` (shipped) | Disposition |
|---|------------------|------------------------|---------------------|-------------|
| 1 | Centring point   | bootstrap row mean `ȳ*ᵢ` | bootstrap row mean `ȳ*ᵢ` | **Closed** (2026-05-28): code-fix to `ȳ*ᵢ`; matches cAIC4. |
| 2 | Denominator      | `B − 1` (sample cov)   | `B − 1` (sample cov) | **Closed** (2026-05-28): code-fix; matches cAIC4. Requires `B ≥ 2` (validated). |
| 3 | `sigmapenalty`   | not added              | not added by default; spine adds it explicitly | **Closed** (2026-05-28): `efron_penalty` defaults to `sigmapenalty = 0` (matches cAIC4's bare arithmetic and the Level-1 fixture); the `_bootstrap` spine passes the user-supplied `sigmapenalty` (default `1`) so the user-facing cAIC keeps σ²-parameter-count symmetry with the analytic path. |
| 4 | Default `nboot`  | `max(n, 100)`          | `500`                | **Open (UX-only).** Not a correctness gate. The 2026-05-28 `DECISIONS.md` entry on the bootstrap-vs-analytic convergence gate already covers why `500` is a sensible Julia-side default; an explicit DECISIONS.md entry pinning the choice is the remaining clean-up. |

**Resolution of #1 + #2.** The shipped `efron_penalty` now implements `cAIC4`'s exact
arithmetic (§3). The two were previously *asymptotically equivalent* but not
algebraically identical, which made a "Level-1 fixture against `conditionalBootstrap` at
`rtol = 1e-6` / `atol = 1e-10`" unrealisable. With #1 + #2 closed, that fixture is now
the bootstrap path's Level-1 correctness gate (see DECISIONS.md 2026-05-28 and
`test/dof_lmm_tests.jl`: "efron_penalty reproduces cAIC4's conditionalBootstrap arithmetic
on shared Y*/Ŷ*"). The fixture is generated by `test/generate_fixtures_bootstrap.{jl,R}`:
Julia writes seeded synthetic `(yhat, sigma, Y*, Ŷ*)` matrices, R applies cAIC4's bias-
correction arithmetic (lines 23–25 of `conditionalBootstrap.R` v1.1) to them, and Julia
re-runs `efron_penalty` and compares.

**Resolution of #3.** `cAIC4`'s `bcMer.R` routes `sigma.penalty` only to
`biasCorrectionGaussian`; the bootstrap path receives no such adjustment. The Level-1
unit `efron_penalty` now matches: its `sigmapenalty` default is `0` (cAIC4's bare
arithmetic), and the shared-input fixture compares at `sigmapenalty = 0`. The user-facing
`caic(...; method = :bootstrap, sigmapenalty = 1)` retains parity with the analytic
path's σ²-parameter-count convention by passing `sigmapenalty` explicitly to
`efron_penalty` from the `_bootstrap` spine in `src/scoring.jl`. This places the
package's σ²-parameter-count convention at the *spine* level (where it belongs as
user-facing semantics), and leaves the *Level-1 unit* a faithful port of cAIC4's
arithmetic (where bit-parity is the correctness gate).

**On #4 (open, UX-only).** `max(n, 100)` is `cAIC4`'s choice; `500` is the package's. A
flat default is simpler in a function signature than a data-dependent one; whether to
match `cAIC4`'s heuristic is a UX decision, not a correctness one. The 2026-05-28
`DECISIONS.md` entry "Bootstrap-vs-analytic convergence gate: `atol = 2.0`,
`nboot = 2000`" already records that the *test* nboot needs to be high (because the MC
standard error is `≈ C / √B` with `C ≈ 10–15`); at `n = 180` (sleepstudy), `cAIC4`'s
`max(n, 100) = 180` would be far below the convergence threshold the package's tests
use. The `500` default lies between `100` and `2000` and was chosen with this MC-noise
envelope in mind. A one-line `DECISIONS.md` entry pinning it is the remaining clean-up.

---

## 6. Internal cross-check: convergence to the analytic ρ

For a Gaussian LMM, both the bootstrap and the analytic (Greven–Kneib) ρ are estimators of
the same Efron ρ. They are therefore expected to lie within Monte-Carlo noise of each
other at large `B`. The convergence gate

```math
\bigl|\rho_{\mathrm{bootstrap}}(B) - \rho_{\mathrm{analytic}}\bigr| \le \texttt{atol}
```

with `atol = 2.0` at `B = 2000` is the package's primary correctness check (the
2026-05-28 `DECISIONS.md` entry derives the tolerance). It is **not** a strict-frequentist
convergence — the per-draw `cAIC4` form, the per-draw cAIC.jl form, and the analytic form
are not asymptotically the *same number* (the memory record
`bootstrap-not-equal-analytic.md` notes the empirically observed finite gap between cAIC4's
own bootstrap and its own analytic df) — but rather a test that the empirical bootstrap
estimate at large `B` lies inside the MC noise band of the analytic value, the band being
the test's `atol`. Tightening this tolerance is forbidden by that DECISIONS entry; widening
it would weaken the gate. The number `2.0` is therefore part of the spec, not free.

With §5 #1 + #2 closed on 2026-05-28, the shipped form is now the centred / `(B−1)`
estimator. The atol = 2.0 / B = 2000 gate has been re-verified to hold under the new
formula (the bootstrap tests in `test/caic_tests.jl` tagged `:bootstrap` continue to
pass); the slightly larger MC variance of the unbiased sample-covariance form is
absorbed within the 4–6σ band of the chosen tolerance.

---

## 7. Numerical-stability obligations (CLAUDE.md §9)

- The sum in §3/§4 is a plain `dot` per draw against the row-mean-centred `y*` column;
  no log-space transforms needed (it is a sample covariance, not a likelihood).
- The refits are full `MixedModels` fits; the package does not reuse factorisations across
  draws (the design is the same but `y*` changes, so the PLS updates differ). This is a
  performance, not correctness, concern; M3's "make refitting cheap (reuse factorisations)"
  applies to the GLMM path and *may* later cascade here.
- `σ̂ > 0` is asserted in `efron_penalty` (`DomainError`); `sigmapenalty ≥ 0` and shape
  consistency of `Ystar`, `Yhatstar` are `ArgumentError`s. The Level-1 unit fails loud on
  any of these (CLAUDE.md §4).
- The draws use the user-supplied `rng::AbstractRNG` (default `Random.default_rng()`),
  surfaced in `caic`'s signature. Seeded `Xoshiro` is the reproducibility contract — see
  the 2026-05-28 `DECISIONS.md` "Random added as a core dependency" entry.

---

## 8. Provenance and what this enables

- **Versions pinned:** `cAIC4` v1.1 (`conditionalBootstrap` and `bcMer`); `MixedModels =
  5.5.1` (`LinearMixedModel(y, feterm, reterms, formula)` + `fit!` + `optsum.REML`
  preservation in `MMInternals.bootstrapfit`). A bump to either re-opens this note and the
  `mm_internals.jl` table.
- **Sufficient for the Level-1 isolation unit:** §3 (cAIC4 ground truth) and §4 (shipped
  formula) now coincide; `efron_penalty` is a faithful port. The Level-1 test
  (`test/dof_lmm_tests.jl`: "efron_penalty reproduces cAIC4's conditionalBootstrap
  arithmetic on shared Y*/Ŷ*") checks parity against `cAIC4` on identical synthetic
  `(yhat, σ, Y*, Ŷ*)` at `rtol = 1e-6` / `atol = 1e-10`, generated by
  `test/generate_fixtures_bootstrap.{jl,R}`. A second arithmetic-and-validation test
  ("efron_penalty: arithmetic, type stability, and validation") covers the formula
  shape, type stability, and the guards in isolation.
- **Sufficient for the spine:** §2 fixes the bootstrap draw (`y* = ŷ + σ̂ε`) and the refit
  contract (same design, same REML/ML, fresh `θ̂`) — the `MMInternals.bootstrapfit` spec
  and the `_bootstrap` loop in `src/scoring.jl`.
- **Status of the §5 dispositions:** #1, #2, #3 closed on 2026-05-28 (see DECISIONS.md);
  #4 (default `nboot`) remains a UX-only open item that does not block correctness.
