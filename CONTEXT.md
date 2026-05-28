# cAIC.jl

Conditional model selection for mixed-effects models fitted with `MixedModels.jl` — a
re-platforming of R's `cAIC4` onto `LinearMixedModel` / `GeneralizedLinearMixedModel`.

## Language

**Conditional AIC** (cAIC):
An information criterion built from the *conditional* log-likelihood of the data given the
predicted random effects, penalised by a bias-corrected effective degrees of freedom. The
central object of this package.
_Avoid_: "AIC" unqualified.

**Marginal AIC** (mAIC):
The classical AIC using the *marginal* likelihood, with random effects integrated out. The
contrast to cAIC; known to favour larger models in the mixed-model setting. Not the package's
primary target.
_Avoid_: calling it just "AIC".

**Conditional log-likelihood**:
ℓ(y | b̂, β̂, θ̂) — the log-likelihood of the response given the *predicted* (not integrated-out)
random effects. The "conditional" in conditional AIC.

**Effective degrees of freedom** (ρ):
The penalty term of the cAIC: a bias-corrected count of effective parameters, larger than the
naive hat-matrix trace ρ₀ = tr(H₁) because the variance parameters θ are estimated.
_Avoid_: "number of parameters" — it is generally non-integer.

**Scoring**:
Computing the cAIC value for *one* fitted model. The M2 deliverable.

**Selection** (umbrella):
Choosing among models by cAIC. Two distinct forms, kept separate:
- **Comparison** — scoring a user-supplied *fixed set* of fitted models and ranking them
  (`cAIC4`'s `anocAIC`; our port `anocaic`). The literal "best from among several".
- **Search** — generating and exploring a *candidate space* (add/drop random- or fixed-effects
  terms) to find a good model (the `stepcAIC` layer, M4).
Both build on Scoring; both require every candidate to be scored *consistently* (same
REML/`method`/B-source/`sigmapenalty`), which the result's provenance enforces.
_Avoid_: using "model selection" to mean scoring one model, or conflating Comparison (a given
set) with Search (a generated space).

**Averaging** (model averaging):
Combining several fitted models into one prediction, *weighted* by their cAIC (lower cAIC → larger
weight), rather than selecting a single model. `cAIC4`'s `modelAvg` / `predictMA` / `summaryMA`.
Distinct from Selection: it keeps all the models and blends them. In the parity goal as its own
milestone **M4.5** (CLAUDE.md §11); outside the near-term scope.
_Avoid_: conflating with Comparison (which ranks and picks one) or Search.

**Singular fit**:
A fit in which one or more random-effects variance components are estimated on the **boundary**
(zero variance) — equivalently, the relative covariance factor λ has a zero on its diagonal. A
first-class, supported case, never an error.
_Avoid_: "degenerate fit", "failed fit".

**Boundary**:
The edge of the variance-parameter space where a variance component is zero. Estimation *on* the
boundary is what makes a fit singular and what the bias correction must handle specially.

**Reduced model**:
The model obtained from a singular fit by removing the variance components estimated at the
boundary and refitting; the cAIC is computed on it, and the scoring result records it alongside a
was-refitted flag (mirroring `cAIC4`).

## Relationships

- **Selection** is built on **Scoring** — score candidates, then select.
- **cAIC** = −2 · **conditional log-likelihood** + 2 · **effective degrees of freedom**.
- **Effective degrees of freedom** (ρ) ≥ naive plug-in df (ρ₀); the gap is the **bias correction**.
- **Conditional AIC** uses the conditional log-likelihood; **Marginal AIC** uses the marginal likelihood.
- A **singular fit** has a variance component on the **boundary**; scoring it yields a **reduced model**, on which the cAIC is computed.

## Example dialogue

> **Dev:** "The aim is to select the best model from several — so the first thing we build is the search?"
> **Statistician:** "No. First we must *score* a single model — compute its conditional AIC. Selection
> (`stepcAIC`) is a search layer on top that scores candidates and compares them. Scoring comes first."

**Chen-Stein correction** (Poisson GLMM df):
The bias-corrected effective df for a Poisson GLMM. For each non-zero observation i, the model
is refit with y_i set to zero; the df contribution is y_i × Δη̂_i. Requires n − #{y=0} refits.
The `method=:steinian` route for Poisson in `cAIC4` (`biasCorrectionPoisson`).
_Avoid_: calling it "analytic Greven-Kneib" — it has no closed-form hat matrix; it is influence-based.

**Efron's estimator** (Bernoulli GLMM df):
The bias-corrected effective df for a Bernoulli GLMM. For each observation i, the model is refit
with y_i flipped (0→1 or 1→0); the df contribution is μ̂_i(1−μ̂_i) × sign_i × Δlogit_i.
Requires n refits.
The `method=:steinian` route for Bernoulli in `cAIC4` (`biasCorrectionBernoulli`).
_Avoid_: conflating with the Gaussian Efron bootstrap penalty (a different formula in M2).

**Conditional simulation** (GLMM bootstrap):
Sampling from the conditional distribution f(y | b̂) by drawing y_i ~ f(μ̂_i) directly, where
μ̂_i = linkinv(η̂_i) is the fitted conditional mean. Used in the GLMM conditional bootstrap.
Does *not* use `MixedModels.simulate!` (which draws new random effects = marginal simulation).
See ADR-0005.
_Avoid_: "simulate from the model" — MixedModels.jl's `simulate!` is marginal, not conditional.

**Full-singularity fallback** (GLMM):
When all GLMM variance components are on the boundary (all θ = 0), the model reduces to a
plain GLM. The effective df is rank(X) with no sigma penalty. Matches `cAIC4`'s
`deleteZeroComponents` → `glm` path which returns `zeroLessModel$rank`.
_Avoid_: confusing with the Gaussian all-singular path, which does add `sigmapenalty`.

**Partial-singularity reduction** (GLMM):
When some but not all GLMM variance components are on the boundary. The zero components are
removed and the model is refit on the reduced parameter space — the GLMM analogue of the LMM
`reduceboundary` function.

## Flagged ambiguities

- "the best model from among several" (aim statement) conflated **Scoring** with **Selection**,
  and within Selection conflated **Comparison** (a given set) with **Search** (a generated space) —
  resolved: Scoring = cAIC for one model (M2); Comparison = rank a user-supplied set
  (`cAIC4`'s `anocAIC`); Search = explore candidates (`stepcAIC`, M4).
- "conditional" is overloaded — it qualifies the AIC, the log-likelihood, *and* (separately)
  "conditional model selection". In this glossary "conditional" always means conditioning on the
  predicted random effects b̂.
- The comparison function is `cAIC4`'s `anocAIC` (verified against the `cAIC4` `NAMESPACE`), **not**
  `anocaic` — the latter is reserved as our lowercase Julia port name. Earlier drafts had the
  spelling wrong.
- `cAIC4` also exports a **model-averaging** suite (`modelAvg` / `predictMA` / `summaryMA`),
  surfaced from its `NAMESPACE`. It is a distinct capability (**Averaging**), out of near-term
  scope, now folded into the parity goal as milestone **M4.5** (CLAUDE.md §11, amended 2026-05-27;
  see PARITY.md).
