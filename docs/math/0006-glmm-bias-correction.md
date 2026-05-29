# 0006 — GLMM bias correction: conditional log-likelihood and the three df paths

This note is the §7 step-1 "state the math" record for issue #24 (milestone M3). It pins,
in precise notation, the conditional-AIC mathematics for **generalized** linear mixed
models — the conditional log-likelihood and the three family-dependent degrees-of-freedom
routes (Chen–Stein for Poisson, Efron for Bernoulli, conditional bootstrap for everything
else) — **before any M3 Julia code is written**. Unlike `0005` (written after its
estimator shipped, a §7 process failure recorded there), this note is the gate up front:
the M3 estimators must pass it before `dof_glmm.jl` / the GLMM scoring spine exist.

The conditional AIC assembly is unchanged from the Gaussian path (`cAIC4` `R/cAIC.R:272`):

```
cAIC = −2 · ℓ_cond(y | b̂, β̂, θ̂) + 2 · ρ ,
```

where `ℓ_cond` is the GLMM conditional log-likelihood of §1 (evaluated on the
*possibly reduced* model — see §5) and `ρ` is the family-dependent df of §2–§5.

**Ground-truth sources consulted**
- `cAIC4` **v1.1** (CRAN, 2025-04-04), read directly from source (not asserted from
  memory): `R/getcondLL.R` (`getcondLL.merMod`, the family `switch`), `R/biasCorrectionPoisson.R`,
  `R/biasCorrectionBernoulli.R`, `R/conditionalBootstrap.R`, `R/bcMer.R` (family dispatch +
  the `B = max(n, 100)` default), `R/cAIC.R` (assembly + the binomial-`n>2` fallback to
  bootstrap + the plain-GLM/LM branch), `R/deleteZeroComponents.R` (the
  reduce-to-`glm` boundary path).
- Säfken, B., Rügamer, D., Kneib, T. & Greven, S. (2021). Conditional Model Selection in
  Mixed-Effects Models with `cAIC4`. *JSS* 99(8). doi:10.18637/jss.v099.i08. **The package
  reference for every formula below.**
- Säfken, B., Kneib, T., van Waveren, C.-S. & Greven, S. (2014). A unifying approach to the
  estimation of the conditional Akaike information in generalized linear mixed models.
  *Electronic J. Statist.* 8, 201–225. (The actual derivation of the Poisson and Bernoulli
  estimators; cited in `cAIC4`'s own source comments.)
- Lian, H. (2012). A note on conditional Akaike information for Poisson regression with
  random effects. *Electronic J. Statist.* 6, 1–9. (The Poisson Chen–Stein df.)
- Efron, B. (2004). The estimation of prediction error: covariance penalties and
  cross-validation. *JASA* 99(467), 619–632. (The covariance-penalty df and the Bernoulli
  Steinian estimator.)
- [`docs/math/0003-conditional-loglik.md`](0003-conditional-loglik.md) (the Gaussian
  conditional log-lik this generalises), [`docs/math/0005-conditional-bootstrap.md`](0005-conditional-bootstrap.md)
  (the Gaussian Efron covariance penalty the GLMM bootstrap reuses), and
  [ADR-0005](../adr/0005-glmm-conditional-simulation.md) (the conditional-simulation draw).

Where `cAIC4` and any other source disagree, **`cAIC4` is ground truth** (CLAUDE.md §2).
Two places where this issue's *prose paraphrase* disagrees with the `cAIC4` *source* are
recorded explicitly in §6; the source wins in both.

---

## 0. The fitted GLMM and the conditional mean

We score a `GeneralizedLinearMixedModel` already fitted by `MixedModels.jl`. With link `g`,
response family `f`, fixed-effects design `X`, random-effects design `Z`, and fitted
parameters `(β̂, b̂, θ̂)`, the conditional linear predictor and conditional mean are

```
η̂ = X β̂ + Z b̂ ,        μ̂ = g⁻¹(η̂) = linkinv(η̂) .
```

`μ̂ = m.resp.mu` and `η̂ = m.resp.eta` (both accessed through `mm_internals.jl`; `m.resp.mu`
is already in the quarantine requirement of ADR-0005). Everything below is a function of the
fit only through `(y, μ̂, η̂)` and per-observation refits — never the marginal/PIRLS
likelihood. This is the GLMM sense of *conditional*: the law of `y` given the **predicted**
random effects `b̂`, not integrated over them (CONTEXT.md; `0003` §1).

For the canonical-link families in M3 scope (Poisson log-link; Bernoulli/binomial
logit-link) the dispersion is fixed: `σ = 1`, so `sigma(object) = 1` enters the bootstrap
divisor of §4. Families with a *free* dispersion parameter are out of scope (matches
`cAIC4`'s "not yet supported" warning; ADR-0005 §Consequences).

---

## 1. Conditional log-likelihood ℓ_cond (Poisson, Bernoulli, multi-trial binomial)

Transcribed from `getcondLL.merMod` (`R/getcondLL.R:47–66`), evaluated at the fitted
conditional mean `μ̂` (the `getME(object, "mu")` value), **not** via the PIRLS marginal
deviance. The family `switch`:

**Poisson** (`dpois(y, lambda = μ̂, log = TRUE)`):

```math
\ell_{\mathrm{cond}}^{\mathrm{Pois}}(y \mid \hat\mu)
  = \sum_{i=1}^{n} \bigl[\, y_i \log \hat\mu_i - \hat\mu_i - \log(y_i!) \,\bigr],
```

with `μ̂_i = exp(η̂_i)` the per-observation Poisson mean. The `log(y_i!) = lgamma(y_i + 1)`
term is constant in the parameters but is **kept** (it is in `dpois(..., log=TRUE)`), so the
absolute cAIC value matches `cAIC4`.

**Bernoulli / binomial** (`dbinom(y, size = length(unique(y)) − 1, prob = μ̂, log = TRUE)`):

```math
\ell_{\mathrm{cond}}^{\mathrm{Bin}}(y \mid \hat\mu)
  = \sum_{i=1}^{n} \log \binom{m}{y_i}
    + \sum_{i=1}^{n} \bigl[\, y_i \log \hat\mu_i + (m - y_i)\log(1 - \hat\mu_i) \,\bigr],
  \qquad m = |\{\text{unique } y\}| - 1.
```

For the **Bernoulli** case (the only `steinian`-eligible binomial — see §3 and the
`R/cAIC.R:247–253` guard) `y ∈ {0,1}` so `|unique(y)| = 2`, hence `m = 1`, the binomial
coefficient is `1` (`log = 0`), and this collapses to

```math
\ell_{\mathrm{cond}}^{\mathrm{Bern}}(y \mid \hat\mu)
  = \sum_{i=1}^{n} \bigl[\, y_i \log \hat\mu_i + (1 - y_i)\log(1 - \hat\mu_i) \,\bigr].
```

`ℓ_cond` is evaluated on the model `cAIC` actually penalises: if the boundary reduction of
§5 produced a reduced model, `cAIC4` calls `getcondLL(newModel)` (`R/cAIC.R:270–271`), so
the conditional log-lik is taken on the **reduced** fit, consistent with its df.

### 1.1 Multi-trial binomial — a documented deviation from `cAIC4`

The §1 `getcondLL.merMod` binomial branch is **only correct for Bernoulli**. With `size =
|unique(y)| − 1` and `x = getME(object, "y")`, a *multi-trial* binomial (response stored as
a proportion `yᵢ = kᵢ/nᵢ ∈ [0,1]`, trial counts `nᵢ` in the prior weights) feeds `dbinom` a
**non-integer** `x` and a `size` unrelated to the trials. `dbinom` then returns `0` (R warns
"non-integer x"), so `log = −∞`: `cAIC4` has **no finite conditional log-likelihood for
multi-trial binomial** (and its `R/cAIC.R:247–253` guard only redirects the *df* route to the
bootstrap — it does **not** repair `getcondLL`). The df bootstrap (§5) is thus reachable in
`cAIC4` but the assembled `cAIC` is not finite for this family.

`cAIC.jl` is the bootstrap-df path's *only* consumer of this log-likelihood, and it must
produce a finite cAIC. Per CLAUDE.md §1 (mathematical correctness) and §10 (a provable
`cAIC4` defect is resolved by a **documented deviation**, not by copying the bug), `cAIC.jl`
uses the **correct** binomial density evaluated at the true trial counts `nᵢ` (the prior
weights `m.resp.wts`, `mm_internals.jl::glmmpriorweights`) and the success counts
`kᵢ = nᵢ yᵢ`:

```math
\ell_{\mathrm{cond}}^{\mathrm{Bin}}(y \mid \hat\mu)
  = \sum_{i=1}^{n} \Bigl[\, \log \binom{n_i}{k_i}
      + k_i \log \hat\mu_i + (n_i - k_i)\log(1 - \hat\mu_i) \,\Bigr],
  \qquad k_i = n_i\,y_i ,
```

with `log C(nᵢ, kᵢ) = lgamma(nᵢ+1) − lgamma(kᵢ+1) − lgamma(nᵢ−kᵢ+1)`. This is exactly base
R's `sum(dbinom(kᵢ, nᵢ, μ̂ᵢ, log = TRUE))` — the **correct** binomial density, *not* `cAIC4`'s
`getcondLL` wrapper — and is the Level-1 reference for the kernel
(`Loglik.condloglik_binomial`). For `nᵢ ≡ 1` the coefficient vanishes and it collapses to the
Bernoulli `ℓ_cond` above, so Bernoulli stays on its dedicated kernel and only `|unique(y)| > 2`
binomials take this path. The deviation is logged in `DECISIONS.md` (2026-05-29); there is no
`cAIC4` Level-2 cross-check for this value because `cAIC4`'s own value is `−∞`.

---

## 2. Family dispatch (which df route)

`bcMer` (`R/bcMer.R`) selects the df route by family when `method` is `NULL`/`steinian`:

| family (`MixedModels` family) | df route | `cAIC4` function | §here |
|---|---|---|---|
| Poisson (log link) | Chen–Stein influence | `biasCorrectionPoisson` | §3 |
| Bernoulli (logit, `|unique(y)|=2`) | Efron Steinian | `biasCorrectionBernoulli` | §4 |
| binomial with `|unique(y)| > 2` | **forced** conditional bootstrap | `conditionalBootstrap` | §5 |
| all other families | conditional bootstrap | `conditionalBootstrap` | §5 |

The binomial-`n>2` case is *not* handled by the Bernoulli estimator: `R/cAIC.R:247–253`
detects `length(unique(getME(object,"y"))) > 2`, warns, and overrides `method <-
"conditionalBootstrap"`. The M3 Julia dispatch must replicate this override, not silently
feed multi-category binomial into the Bernoulli path.

---

## 3. Chen–Stein correction (Poisson) — `biasCorrectionPoisson`

The influence-based df for Poisson responses (Lian 2012; Säfken et al. 2014). It is **not**
a Greven–Kneib hat-matrix trace — there is no closed-form hat matrix in the GLMM — but a sum
of per-observation finite influences obtained by refitting under a **unit decrement** of each
nonzero response.

Transcribed from `R/biasCorrectionPoisson.R:13–24`:

```r
zeroLessModel <- deleteZeroComponents(object)        # boundary reduction (§5)
if (inherits(zeroLessModel, "glm")) return(zeroLessModel$rank)   # full-singularity fallback (§5)
y   <- zeroLessModel@resp$y
ind <- which(y != 0)
workingMatrix       <- matrix(rep(y, length(y)), ncol = length(y))
diag(workingMatrix) <- diag(workingMatrix) - 1       # <-- y_i DECREMENTED BY ONE, not zeroed
workingMatrix       <- workingMatrix[, ind]
workingEta          <- diag(apply(workingMatrix, 2, function(x) refit(zeroLessModel, newresp = x)@resp$eta)[ind, ])
bc <- sum(y[ind] * (zeroLessModel@resp$eta[ind] - workingEta))
```

In notation, let `η̂_i` be the fitted linear predictor of the (reduced) model, and let
`η̂_i^{(−i)}` be the *i-th* linear predictor after **refitting the whole model on the response
vector `y` with its `i`-th entry replaced by `y_i − 1`** (every other entry unchanged). Then

```math
\rho_{\mathrm{Pois}}
  = \sum_{i \,:\, y_i \neq 0} y_i \,\bigl(\hat\eta_i - \hat\eta_i^{(-i)}\bigr).
```

- The shift is `y_i → y_i − 1`, the **Chen–Stein/Hudson unit decrement** for the Poisson
  (the discrete analogue `E[\lambda f(Y)] = E[Y f(Y-1)]`), **not** `y_i → 0`. This is the
  single most error-prone point of the M3 spec; see §6 #1.
- Only `y_i ≠ 0` observations contribute (decrementing a zero count is out of domain; their
  influence term is dropped). The number of refits is therefore `n − #{i : y_i = 0}`.
- Cost: one full GLMM refit per nonzero observation. M3's "make refitting cheap (reuse
  factorisations)" mandate (CLAUDE.md §11) targets exactly this loop.

---

## 4. Efron's estimator (Bernoulli) — `biasCorrectionBernoulli`

The asymptotically-unbiased Steinian df for binary responses (Efron 2004; Säfken et al.
2014). One refit per observation, each with that observation's label **flipped**
`y_i → 1 − y_i`. Transcribed from `R/biasCorrectionBernoulli.R:10–23`:

```r
zeroLessModel <- deleteZeroComponents(object)
if (inherits(zeroLessModel, "glm")) return(zeroLessModel$rank)   # full-singularity fallback (§5)
signCor <- -2 * zeroLessModel@resp$y + 1                          # +1 if y_i = 0, −1 if y_i = 1
muHat   <- zeroLessModel@resp$mu
for (i in seq_along(muHat)) {
  workingData    <- zeroLessModel@resp$y ; workingData[i] <- 1 - workingData[i]   # flip y_i
  workingModel   <- refit(zeroLessModel, newresp = workingData)
  workingEta[i]  <- log(workingModel@resp$mu[i] / (1 - workingModel@resp$mu[i]))   # logit(μ̂_i^{flip})
                    - log(muHat[i] / (1 - muHat[i]))                               #   − logit(μ̂_i)
}
bc <- sum(muHat * (1 - muHat) * signCor * workingEta)
```

Writing `μ̂_i^{flip}` for the `i`-th fitted mean after refitting on `y` with `y_i → 1 − y_i`,
and `logit(p) = log(p/(1−p))`,

```math
\rho_{\mathrm{Bern}}
  = \sum_{i=1}^{n} \hat\mu_i(1 - \hat\mu_i)\,(-2 y_i + 1)\,
      \bigl(\operatorname{logit}(\hat\mu_i^{\mathrm{flip}}) - \operatorname{logit}(\hat\mu_i)\bigr).
```

- `signCor_i = −2y_i + 1 ∈ {+1, −1}` orients the difference so a label change in either
  direction adds the same-signed influence.
- `μ̂_i(1−μ̂_i)` is the Bernoulli variance weight at the fitted mean.
- The logit difference is the change in the natural parameter at observation `i` induced by
  flipping its own label; `cAIC4` writes `logit` explicitly as `log(μ/(1−μ))`.
- `n` refits — one per observation, no `y_i = 0` skipping (every binary point is flippable).
- Distinct from the Gaussian Efron *bootstrap* penalty of `0005`: that is a Monte-Carlo
  covariance over simulated responses; this is a deterministic per-observation finite
  difference.

---

## 5. Conditional bootstrap (other families) — `conditionalBootstrap`

For families with no closed/influence estimator (binomial with `>2` categories, and any
other family), `cAIC4` falls back to the Efron covariance-penalty bootstrap of `0005`,
adapted to the GLMM draw. Transcribed from `R/conditionalBootstrap.R:15–26`:

```r
dataMatrix <- simulate(object, nsim = BootStrRep, use.u = TRUE)   # y*(b), conditional on b̂
workingEta <- sapply(dataMatrix, function(x) predict(refit(object, newresp = x)))   # link-scale η̂*(b)
if (is.factor(dataMatrix[[1]])) dataMatrix <- sapply(dataMatrix, as.numeric) - 1
dataMatrix <- dataMatrix - rowMeans(dataMatrix)                   # centre y* row-wise
bootBC     <- sum(workingEta * dataMatrix) / ((BootStrRep - 1) * sigma(object)^2)
```

With `B` draws, `ȳ*_i = (1/B) Σ_b y_i^{(b)}`, and `η̂_i^{(b)}` the link-scale predicted
**natural parameter** from refitting on draw `b` (`predict.merMod` default `type="link"`),

```math
\rho_{\mathrm{boot}}
  = \frac{1}{(B - 1)\,\hat\sigma^{2}}
    \sum_{b=1}^{B} \sum_{i=1}^{n} \hat\eta_i^{(b)}\,\bigl(y_i^{(b)} - \bar y^{*}_i\bigr),
  \qquad \hat\sigma^2 = 1 \text{ for canonical families.}
```

This is the Efron covariance-penalty estimator (`0005` §3) with the natural parameter `η̂`
in place of the Gaussian `ŷ`; for the Gaussian identity link `η̂ = μ̂ = ŷ` and it reduces to
`0005`. The `cAIC.R` doc states the population form as `(1/(B−1)) Σ_i θ_i(z_i)(z_i − z̄)`
with `θ_i` "the i-th element of the estimated natural parameter" — confirming `η̂`, not `μ̂`
(see §6 #2).

**The draw (ADR-0005).** `cAIC4`/`lme4` use `simulate(object, use.u = TRUE)` — random
effects held *fixed* at `b̂`, so draws are conditional. `MixedModels.jl` v5.5.1 has no
`use_u` flag (`simulate!` always redraws `u`, a *marginal* draw). Per ADR-0005 the M3 path
therefore draws **directly from the conditional response law**:

```math
y_i^{(b)} \sim f(\hat\mu_i), \qquad i = 1\dots n,\ b = 1\dots B,
```

with `f = Poisson(μ̂_i)`, `Bernoulli(μ̂_i)`, etc., `μ̂_i = m.resp.mu[i]`. This is equivalent
to `use.u=TRUE` because `μ̂` already encodes `b̂` through `η̂ = Xβ̂ + Zb̂`; no PIRLS re-entry
is needed (ADR-0005 §Decision). `B` default = `max(n, 100)` (`bcMer.R:54–56`); the Julia
default and `rng` contract follow `0005`'s.

### Full-singularity fallback (all variance components θ = 0)

`biasCorrectionPoisson`, `biasCorrectionBernoulli`, and the bootstrap all begin with
`deleteZeroComponents(object)`. When **every** variance component is on the boundary
(`θ = 0`), the reduction collapses the model to a plain GLM (`inherits(zeroLessModel,
"glm")`), and both influence functions `return(zeroLessModel$rank)` (`biasCorrectionPoisson.R:14–16`,
`biasCorrectionBernoulli.R:11–13`):

```math
\rho = \operatorname{rank}(X), \qquad \text{no } \sigma\text{-penalty added.}
```

Note this is `rank(X)` **without** the `+1` that the *plain-GLM-input* branch of `cAIC.R:233–237`
uses (`df = object$rank + 1`): the `+1` there is the Gaussian-LM dispersion parameter, which a
canonical-link Poisson/Bernoulli GLM does not have. The M3 fallback must return bare
`rank(X)`. Partial boundary reduction (some but not all θ = 0) hands a *smaller* GLMM to the
influence/bootstrap routes above — the reduced-model reconstruction is the M2 machinery
(PARITY.md `deleteZeroComponents`; the reduced-model reconstruction memory record).

---

## 6. Two divergences between this issue's prose and the `cAIC4` source

Reading the source up front (memory: *verify cAIC4 against source*) surfaced two points
where issue #24's formula prose does not match `R/`. Per CLAUDE.md §2 the **source is ground
truth**, and the project decision is explicit: **do not diverge from `cAIC4`**. §3–§5 above
encode the source verbatim; both points are **settled in favour of the source** (no
`DECISIONS.md` entry needed — there is no divergence to record).

These two are paraphrase-vs-source, not `cAIC.jl`-vs-`cAIC4`. The *one* place `cAIC.jl`
genuinely diverges from the `cAIC4` source is the **multi-trial binomial conditional
log-likelihood** of §1.1 — where the `cAIC4` source is provably defective (`−∞`) and the
deviation is recorded in `DECISIONS.md` (2026-05-29) per CLAUDE.md §10.

| # | Quantity | Issue #24 prose | `cAIC4` source (ground truth, adopted) | Decision |
|---|---|---|---|---|
| 1 | Poisson refit shift | "`y_i` set to **zero**", `η̂_i^{(0)}` | `diag(workingMatrix) - 1`: `y_i → y_i − 1` (unit decrement) | **Source adopted** (§3): `η̂_i^{(−i)}` is the refit at `y_i − 1`. The issue's "zero" was an inaccurate paraphrase of the Chen–Stein decrement; the implementation decrements by one. |
| 2 | Bootstrap working value | `μ̂_i^{(b)}` (mean scale) | `predict(refit(...))` = link-scale `η̂` (natural parameter) | **Source adopted** (§5): `η̂_i^{(b)}`. Equal to `μ̂` only under the identity link; for log/logit links they differ. `cAIC.R`'s own doc confirms "natural parameter". |

Neither is a numerical-tolerance question and neither is resolved by adjusting a tolerance
(CLAUDE.md §6/§10) — both are exact formula choices, fixed to match the source.

---

## 7. Numerical-stability obligations (CLAUDE.md §9)

- **Conditional log-lik in log-space.** `ℓ_cond` is summed from per-observation log-densities
  (`y log μ̂ − μ̂ − lgamma(y+1)` for Poisson; `y log μ̂ + (1−y) log(1−μ̂)` for Bernoulli),
  never products of raw probabilities. Use `LogExpFunctions` (`xlogy`, `xlog1py`,
  `loggamma`) so `μ̂ → 0` / `μ̂ → 1` give `0` (via `xlogy(0, ·) = 0`) rather than `NaN`, and
  guard `μ̂ ∈ (0,1)` / `μ̂ > 0` per family domain (`DomainError` otherwise).
- **Logit difference (§4)** computed as `logit(μ̂^{flip}) − logit(μ̂)` with a numerically safe
  `logit` (`log(μ̂) − log1p(−μ̂)` or `LogExpFunctions.logit`), not by forming `μ̂/(1−μ̂)` and
  dividing — avoids overflow as `μ̂ → 1`.
- **No marginal/PIRLS likelihood, no explicit inverse, no `det`.** All df routes are sums of
  refit-difference scalars (§3, §4) or a sample covariance (§5); no hat matrix is formed and
  no system is inverted. The bootstrap sum is the `0005` `dot`-per-draw against the
  row-mean-centred draw column.
- **Refits reuse the fitted design.** Each refit keeps `X`, `reterms`, `formula`, and the
  REML/ML setting fixed and swaps only the response (the `0005` `bootstrapfit` contract,
  generalised to GLMM via `MMInternals` and ADR-0005's direct conditional draw). Boundary
  detection (`issingular`) and the reduce-to-GLM fallback are first-class (§5), not errors.
- **Fail loud.** Family outside M3 scope (free dispersion) → documented error matching
  `cAIC4`'s warning; non-binary `y` into the Bernoulli path is prevented by the §2 dispatch
  override, not silently mis-scored.

---

## 8. Provenance and what this enables

- **Versions pinned:** `cAIC4` v1.1 (`getcondLL`, `biasCorrectionPoisson`,
  `biasCorrectionBernoulli`, `conditionalBootstrap`, `bcMer`, `cAIC`, `deleteZeroComponents`);
  `MixedModels` = 5.5.1 (`GeneralizedLinearMixedModel`, `m.resp.mu`/`m.resp.eta`, `fit!`,
  the refit-on-new-response path; `m.resp.mu` per ADR-0005 must enter the `mm_internals.jl`
  table before the bootstrap path is built). A bump to either re-opens this note and the
  internal-access table.
- **Enables the M3 conditional log-lik (#?):** the Poisson and Bernoulli `ℓ_cond` kernels of
  §1, Level-1-testable against hand-computed `dpois`/`dbinom` sums and Level-2 against
  `cAIC4::getcondLL`.
- **Enables the three df estimators:** §3 (`biasCorrectionPoisson`), §4
  (`biasCorrectionBernoulli`), §5 (`conditionalBootstrap` + ADR-0005 draw + full-singularity
  fallback), each traced to its `cAIC4` source function and ready for a failing R-reference
  test (§7-ritual step 3) before implementation.
- **§6 settled (no divergence from `cAIC4`):** #1 (Poisson decrement-by-one) and #2
  (bootstrap natural parameter `η̂`) are fixed to the source; the implementation must match
  these exactly. No Julia code is written by this note (issue #24 acceptance: documentation
  only).
</content>
</invoke>
