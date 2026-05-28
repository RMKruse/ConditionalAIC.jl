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
is a process failure against CLAUDE.md §7 step 1; the formula divergences in §5 are the
direct cost of not having had this gate up front. The note is the catch-up record, not a
post-hoc rationalisation: the §5 dispositions are open items, not closed ones.

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

`DofLMM.efron_penalty(yhat, sigma, Ystar, Yhatstar, sigmapenalty)` (Level-1, pure) and the
`_bootstrap` spine in `src/scoring.jl` together compute, with `ŷ` the original fit's
conditional mean (not the bootstrap row mean),

```math
\rho_{\mathrm{cAIC.jl}} =
  \frac{1}{B\,\hat\sigma^{2}}
  \sum_{b = 1}^{B} \sum_{i = 1}^{n}
    \bigl(y^{*}(b)_{i} - \hat y_{i}\bigr) \bigl(\hat y^{*}(b)_{i} - \hat y_{i}\bigr)
  + \texttt{sigmapenalty}.
```

The draws `y*(b) = ŷ + σ̂ε(b)` are constructed in the spine and have **mean exactly `ŷ`**
under the parametric-bootstrap law, so the centring point matches the *population* mean of
the bootstrap distribution rather than its sample estimate. The default `B` (`nboot`) is
`500`. The same `sigmapenalty` argument as the analytic path is added.

---

## 5. The four divergences from `cAIC4`, and the disposition for each

Pinning the math forces honesty about where the implementation drifts from ground truth.
Four gaps exist between §3 and §4. None is silently absorbed: each is either to be closed
in code or recorded in `DECISIONS.md` with a justification. The disposition column is the
open question this note surfaces.

| # | Quantity         | `cAIC4` (ground truth) | `cAIC.jl` (shipped) | Disposition (open) |
|---|------------------|------------------------|---------------------|--------------------|
| 1 | Centring point   | bootstrap row mean `ȳ*ᵢ` | original fit `ŷᵢ`    | Reconcile: code-fix to `ȳ*ᵢ`, **or** `DECISIONS.md` entry on the "known-mean variance reduction" argument. |
| 2 | Denominator      | `B − 1` (sample cov)   | `B` (known-mean MSE) | Pair with #1; the `B − 1` choice is meaningful only together with the row-mean centring. |
| 3 | `sigmapenalty`   | not added              | added                | Reconcile: code-fix to *not* add, **or** `DECISIONS.md` entry justifying parity with the analytic path's sigmapenalty semantics. |
| 4 | Default `nboot`  | `max(n, 100)`          | `500`                | Either change the default to `max(n, 100)`, **or** record `500` as a deliberate cross-language difference (the `n`-dependent default is poor UX in Julia; the larger floor reduces MC noise at the cost of compute). |

**On #1 + #2.** Both estimators target Efron's population ρ; they differ in finite-`B`
construction. The `cAIC4` form is the unbiased bootstrap sample covariance for `cov(y, ŷ)`
using only the bootstrap draws. The `cAIC.jl` form exploits the fact that the bootstrap
draws have a **known mean exactly equal to `ŷ`** (because they are constructed as `ŷ + σ̂ε`,
not sampled from data), which removes one degree of freedom of centring and replaces the
`B − 1` divisor with `B`. The two are asymptotically equivalent and within `O(1/√B)` of
each other at finite `B`, but they are **not** algebraically identical and do not bit-match
at Level-1 tolerance. The 2026-05-27 `DECISIONS.md` entry's claim that "the Efron
covariance-penalty arithmetic is checked against `cAIC4`'s internal function on fixed,
shared inputs at the tight Level-1 tolerance" is therefore not currently realisable as
stated — there is no fixed-input cAIC4 callable, and even if there were, §3 and §4 would
not bit-match on identical inputs. Resolving #1 + #2 either eliminates this contradiction
(by adopting §3 in code) or requires the DECISIONS entry to be reworded to describe what
the Level-1 test actually checks (the §4 formula against a Julia-side hand-computed value;
see `test/dof_lmm_tests.jl:231`).

**On #3.** `cAIC4`'s `bcMer.R` routes `sigma.penalty` to the analytic
`biasCorrectionGaussian` only; the bootstrap path receives no such adjustment. The shipped
`efron_penalty` adds `sigmapenalty` to the bootstrap ρ, matching the analytic path's
convention. This is an *intentional design choice*, not a transcription error — but
`cAIC4` does not do it. Either it is justified (the user-facing semantics of `sigmapenalty`
across the two methods is more important than `cAIC4` bit-parity) and recorded in
`DECISIONS.md`, or the bootstrap path drops the `+ sigmapenalty` term and the kwarg becomes
a `:steinian`-only option.

**On #4.** `max(n, 100)` is `cAIC4`'s choice; `500` is the package's. A flat default is
simpler in a function signature than a data-dependent one; whether to match `cAIC4`'s
heuristic is a UX decision, not a correctness one. The 2026-05-28 `DECISIONS.md` entry
"Bootstrap-vs-analytic convergence gate: `atol = 2.0`, `nboot = 2000`" already records that
the *test* nboot needs to be high (because the MC standard error is `≈ C / √B` with
`C ≈ 10–15`); at `n = 180` (sleepstudy), `cAIC4`'s `max(n, 100) = 180` would be far below
the convergence threshold the package's tests use. The `500` default lies between `100` and
`2000` and was likely chosen with this MC-noise envelope in mind. Either way: a one-line
`DECISIONS.md` entry pins it.

The §5 dispositions are **open work** — they should be resolved before the bootstrap path
is considered "done" by §7 step 6. This note does not prejudge them.

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

The §5 disposition affects this gate quantitatively: under #1 + #2 the centred / `(B−1)`
form has a slightly larger MC variance than the known-mean / `B` form, so an
implementation flipped to §3 would (marginally) require a wider `atol` or a larger `B`.
This is the kind of dependency the §7 step-1 doc should have surfaced *before* the
tolerance was fixed.

---

## 7. Numerical-stability obligations (CLAUDE.md §9)

- The sum in §4 (or §3 if adopted) is a plain `dot` per draw; no log-space transforms
  needed (it is a sample covariance, not a likelihood).
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
  formula) together are what `efron_penalty` and any future cAIC4-parity fixture must
  encode. The current Level-1 test (`test/dof_lmm_tests.jl:231`) checks §4 against a
  hand-computed value; a §3-parity fixture (computing the cAIC4 expression on identical
  synthetic `Y*`/`Ŷ*`) is the right addition once #1–#3 are dispositioned.
- **Sufficient for the spine:** §2 fixes the bootstrap draw (`y* = ŷ + σ̂ε`) and the refit
  contract (same design, same REML/ML, fresh `θ̂`) — the `MMInternals.bootstrapfit` spec
  and the `_bootstrap` loop in `src/scoring.jl`.
- **Open before "done" by §7 step 6:** the four §5 dispositions. Each is either a code
  change or a `DECISIONS.md` entry; none can stay implicit.
