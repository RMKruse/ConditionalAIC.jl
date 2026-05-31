# 0009 — Model averaging: cAIC-weighted combination (M4.5)

This note is the §7 step-1 "state the math/spec" record for milestone **M4.5** (model averaging,
the **Averaging** verb of `CONTEXT.md`). It pins, **before** the corresponding Julia code is
written, the weight objective, the optimization algorithm, the auxiliary weight scheme, the
name-keyed effect combination, the prediction rule, and the validation plan of `cAIC4`'s
`modelAvg` / `predictMA` / `summaryMA` / `getWeights` / `weightOptim` — **restricted to Gaussian
`LinearMixedModel` candidates** (the design decision of 2026-05-31; there is no `cAIC4` GLMM
averaging and the weight objective is Gaussian by construction — §1).

Like `0005`/`0008`, part of this is an **algorithm-transcription** spec, not a closed-form
estimand: the per-candidate cAIC and effective df come from the already-validated `caic`/`anocaic`
of M2; the contribution of M4.5 is the *weight criterion*, the *optimizer that minimises it*, and
how the candidates are *combined*. The optimizer is ported faithfully from `cAIC4`'s `solnp`-based
routine (ADR-0007).

> **STATUS: core pinned.** §1 (objective), §3 (Buckland), §4 (effects), §5 (predict), §6
> (divergences) and §7 (validation) are fixed. §2 (the `solnp` transcription) pins the *contract*
> and the deviations; the line-by-line correspondence to `weightOptim.R` is filled as the
> transcription is written under TDD (CLAUDE.md §7). Do not implement ahead of a filled section.

**Ground-truth sources** (read from source, not memory — memory record *verify-caic4-against-source*):
- `cAIC4` **v1.1**: `R/modelAvg.R`, `R/predictMA.R`, `R/summaryMA.R`, `R/getWeights.R`,
  `R/weightOptim.R` (the internal `.weightOptim`), and `R/methods.R:42–76` (`anocAIC` — the df/cll
  source, **input-ordered**, `round(., digits = 2)`).
- Zhang, X., Zou, G. & Liang, H. (2014). Model averaging and weight choice in linear mixed-effects
  models. *Biometrika* 101(1), 205–218. — the optimal weight criterion (§1).
- Buckland, S. T., Burnham, K. P. & Augustin, N. H. (1997). Model selection: an integral part of
  inference. *Biometrics* 53, 603–618. — the smoothed weights (§3).
- Greven, S. & Kneib, T. (2010). *Biometrika* 97(4), 773–789. — the cAIC/effective df entering both
  the criterion and the smoothed weights.
- Nocedal, J. & Wright, S. (2006). *Numerical Optimization.* Springer. — the augmented-Lagrangian /
  SQP basis of `solnp`.
- `Rsolnp::solnp` (Ye, 1989 interior step; Ghalanos & Theussl R port) — the algorithm `.weightOptim`
  transcribes.

Where any source disagrees, **`cAIC4` is ground truth** (CLAUDE.md §2), *except* where `cAIC4`
carries a provable defect (CLAUDE.md §1/§10), which §6 records with its disposition.

Companion records: **ADR-0007** (faithful `solnp` transcription; the "algorithm not bug" principle;
its §9 `inv` carve-out was **withdrawn 2026-05-31** — line 154 is now a §9-compliant triangular
solve, see DECISIONS 2026-05-31) and the DECISIONS.md entries dated 2026-05-31 (full-precision df;
`predictma` `new_re_levels` default; the L1/L2 weight tolerances, measured at implementation).

---

## 0. Objects and scope

- **Input.** A collection of `M ≥ 1` fitted `LinearMixedModel{T}` candidates
  `m₁, …, m_M`. They may differ in **both** fixed- and random-effects structure (the `cAIC4`
  `Orthodont` example mixes `age+Sex+age:Sex`, `age+Sex`, `age`, `Sex`). They **must** share one
  response vector `y` and observation count `n`, and one REML setting — validated, `ArgumentError`
  otherwise (the fail-loud strengthening of `cAIC4`'s unchecked `getME(m[[1]], "y")`; CONTEXT.md
  *Averaging*, CLAUDE.md §4).
- **Per-candidate quantities** (from M2, input-ordered — *not* the sorted `anocaic` table; `cAIC4`'s
  `anocAIC` does not sort, `R/methods.R`):
  - `ρᵢ` — effective df of candidate `i` (`CAICResult.dof`), full precision (§6.1).
  - `cAICᵢ` — conditional AIC of candidate `i`.
  - `μᵢ ∈ ℝⁿ` — the conditional fitted mean `Xᵢβ̂ᵢ + Zᵢb̂ᵢ` (`fitted(mᵢ)`, public StatsAPI; the
    `getME(·,"mu")` analogue).
  - `σ̂² = σ̂²(m_{i*})` with `i* = argmaxᵢ ρᵢ` (first-max tie-break, mirroring `which.max`) — the
    residual variance of the **largest-df** candidate (`getME(tempm,"sigma")²`).
- No `MixedModels` internals are touched: `response`, `fitted`, `sigma` are public; `ρ`/`cAIC`
  come through `caic`. `src/mm_internals.jl` is unchanged (ADR-0007).

`M = (μ₁ ⋯ μ_M) ∈ ℝ^{n×M}` is the stacked conditional-mean matrix; `ρ = (ρ₁,…,ρ_M)ᵀ`.

---

## 1. The Zhang-optimal weight objective (`opt = TRUE`)

The optimal model-averaging weights minimise the Mallows-type criterion of Zhang et al. (2014):

```math
\hat w \;=\; \arg\min_{w \in \mathcal{W}} \; J(w),
\qquad
J(w) \;=\; (y - Mw)^{\!\top}(y - Mw) \;+\; 2\,\hat\sigma^{2}\,(\rho^{\!\top} w),
```

over the **unit simplex**

```math
\mathcal{W} \;=\; \bigl\{\, w \in \mathbb{R}^{M} \;:\; \textstyle\sum_i w_i = 1,\; 0 \le w_i \le 1 \,\bigr\}.
```

This is exactly `getWeights`' `find_weights` (`fun <- function(w){ t(y - Mw) %*% (y - Mw) + 2*varDF*(w %*% df) }`)
with `varDF = σ̂²`, constraint `eqfun(w)=Σw=1`, bounds `[0,1]`.

**Structure.** `J` is a **convex quadratic program**: expanding,
`J(w) = wᵀ(MᵀM)w − 2(Mᵀy)ᵀw + yᵀy + 2σ̂²ρᵀw`, with Hessian `∇²J = 2 MᵀM ⪰ 0` and a *linear* df
term. The minimiser over the convex set `𝒲` is **unique iff `MᵀM ≻ 0`**, i.e. the candidates'
conditional-mean vectors are linearly independent. For *nested/collinear* candidates (the common
case) `MᵀM` is singular and the minimiser is **non-unique** — the basis of the validation rule §7
and the ill-conditioning warning §2/§6.4.

**The criterion is Gaussian.** It uses `σ̂²` and a squared-error fit term; there is no `cAIC4`
non-Gaussian analogue, which is *why* M4.5 is LMM-only (§0; CONTEXT.md *Averaging*).

---

## 2. The optimizer: faithful transcription of `cAIC4`'s `solnp` SQP

`getWeights` minimises `J` over `𝒲` with a hand-transcribed `Rsolnp::solnp` — an
**augmented-Lagrangian sequential quadratic program** with an interior feasibility restoration step.
`cAIC.jl` **transcribes it faithfully** (ADR-0007): the goal is 1:1 auditability against
`R/getWeights.R` + `R/weightOptim.R`, not a from-scratch QP solve.

### 2.1 Algorithm contract (the outer loop, `getWeights`)

Reproduce `R/getWeights.R` lines 89–142:
- initialise `w⁽⁰⁾ = (1/M,…,1/M)`, multiplier `λ=0`, `hess = I_M`, penalty `ρ_aug=0`, scaling
  vector, `maxit=400`, `tol=1e-8`, `δ=1e-7`;
- each outer iterate calls `.weightOptim` (§2.2), updates `(w, λ, hess, μ_aug)`, recomputes
  `targets = (J(w), Σw − 1)`, and applies the three convergence/restart guards
  (constraint-satisfied → reset penalty; objective-decrease test → reset multiplier & diagonalise
  `hess`; `‖(Δrel, eq)‖ ≤ tol` → stop);
- return `(weights = w, objective = J)` (the `functionvalue`), plus a runtime `duration`.

### 2.2 Inner step (`.weightOptim`, augmented Lagrangian)

Reproduce `R/weightOptim.R`:
- **scaling** of cost / constraint / parameters / multipliers / Hessian (lines 32–45);
- **finite-difference gradient** `g` and constraint Jacobian `a` (step `δ`, lines 47–58) — `cAIC4`'s
  own FD over `find_weights`; this is an *internal* FD of the transcribed algorithm, not a cAIC
  derivative, so ADR-0001 / CLAUDE §9's FD constraints do not apply (it is reproducing `solnp`, not
  estimating a statistical quantity);
- **interior feasibility restoration** when the start is infeasible (lines 60–105): the `qr.solve`
  Newton step (line 66, an ordinary linear solve `A \ b`, §9-compliant) with the `dx`-scaled
  ratio test;
- **BFGS Hessian update** (lines 116–124);
- **the constrained search-direction solve** (lines 145–168): Cholesky `chol(hess + λ·D²)`
  (line 146); line 154 `solve(cz)` — `cAIC4` forms a *literal matrix inverse* of the Cholesky
  factor and reuses it in the matmuls at lines 161/166, but the port transcribes this as
  **§9-compliant triangular solves** against the factor (`cz' * v == cz_U' \ v`,
  `cz * v == cz_U \ v` — algebraically exact, ADR-0007 decision 2 *withdrawn 2026-05-31*,
  DECISIONS 2026-05-31); `qr.solve` for the multiplier (line 163); and `λ ← 3λ` Levenberg ramp
  until the trial point is feasible;
- **the three-point bisection line search** on the augmented Lagrangian (lines 178–245), with the
  `con1/con2/con3` interval updates and the reduction stop tests.

Each `try`-error branch (`qr.solve` line 66, `chol` line 146, `solve` line 154, `qr.solve`
line 163) is preserved as a **fallback that returns the current iterate** — the algorithm's own
degradation — but additionally emits a **`@warn`** that the weight problem is ill-conditioned and
the optimum may be non-unique (ADR-0007 decision 4; CLAUDE §4: handle-and-report, never silent).

> The exact variable-by-variable Julia↔R correspondence (with the type-stable, `T<:AbstractFloat`
> generic rewrite of the dynamically-typed R) is filled here as the transcription is written under
> TDD. The contract above is the gate: any deviation from `weightOptim.R` beyond §6 is a bug.

### 2.3 Degenerate inputs

- `M = 1`: `𝒲 = {1}`, so `ŵ = (1)`, `J = (y−μ₁)ᵀ(y−μ₁) + 2σ̂²ρ₁`; the optimizer is short-circuited.
- duplicate/collinear candidates: `MᵀM` singular → a `try`-error fallback fires (warned); the
  returned `ŵ` is a valid but non-unique optimum (§7 anchors stable functionals, not `ŵ`).

---

## 3. The Buckland smoothed weights (`opt = FALSE`)

The simple alternative (`modelAvg(..., opt = FALSE)`): exponential cAIC weights (Buckland 1997),

```math
\Delta_i \;=\; \mathrm{cAIC}_i - \min_j \mathrm{cAIC}_j,
\qquad
w_i \;=\; \frac{\exp(-\Delta_i/2)}{\sum_j \exp(-\Delta_j/2)} .
```

Reproduces `R/modelAvg.R` lines 41–45 (`tempres$delta <- cAIC - min(cAIC); weights <- exp(-delta/2)/sum(exp(-delta/2))`).
Needs only the candidate `cAICᵢ` (from `caic`/`anocaic`), no optimizer. Computed in log-space via
`LogExpFunctions.softmax`-style `−Δ/2` normalisation (CLAUDE §9), exact-equivalent to the R form.

---

## 4. Model-averaged effects (name-keyed combination)

Given weights `w` (optimal or smoothed), the averaged coefficients are **name-keyed weighted sums**
(`R/modelAvg.R` lines 47–69, the `tapply(unlist(...), names(...), FUN = sum)` pattern):

```math
\bar\beta_{[t]} \;=\; \sum_{i\,:\,t \in \beta_i} w_i\,\beta_{i,[t]},
\qquad
\bar b_{[k]} \;=\; \sum_{i\,:\,k \in b_i} w_i\,b_{i,[k]},
```

- **fixed effects:** `t` ranges over the **union** of coefficient names across candidates
  (`coefnames(mᵢ)` / `fixef`); a candidate lacking term `t` contributes 0; reported name-sorted
  (the `tapply` ordering).
- **random effects:** `k` keys on **(grouping factor, level, RE term)**. `cAIC4` flattens
  `ranef(mᵢ)` and `tapply`s by the lme4 string name (`Subject.(Intercept)`, …); the Julia port
  builds the equivalent key from the candidate's reterm group/level/term labels (`raneftables(mᵢ)`
  / `ranef`), unions across candidates, weight-sums, and sorts. Absent (grouping factor, level,
  term) combinations contribute 0.

The result is **not itself a fitted model** — it is a pair of name-indexed averaged-coefficient
vectors plus the weight provenance, carried in `ModelAvgResult{T}` (`fixeff`, `raneff`, `weights`,
candidate `cAICᵢ`, candidate models, opt-mode flag, `WeightResult` when `opt=TRUE`).

---

## 5. Prediction of the averaged model (`predictMA`)

For new data `D*`, each candidate predicts conditionally and the predictions are weight-combined
(`R/predictMA.R`: `MApredict <- w %*% t(sapply(candidates, predict, newdata = D*))`):

```math
\hat y^{\mathrm{MA}}(D^*) \;=\; \sum_{i=1}^{M} w_i\,\hat y_i(D^*),
\qquad
\hat y_i(D^*) \;=\; \texttt{predict}(m_i, D^*).
```

`predict(mᵢ, D*)` is **conditional** for grouping levels seen in training (`Xβ̂ + Zb̂`), matching
`lme4`'s default `re.form = NULL`; this is `MixedModels`' native behaviour. **Unseen** levels: a
`new_re_levels` kwarg is forwarded to `MixedModels.predict`, **default `:error`**, mirroring
`lme4`'s `allow.new.levels = FALSE` (§6.3) — `:population` / `:missing` are opt-in. The averaged
prediction over the union (§4) requires all candidates to predict on the same `D*` schema.

`summaryMA(res; randeff=false)` (`R/summaryMA.R`) prints the call, the averaged fixed effects, the
weights, and — when `randeff=true` — the averaged random effects. The default REPL view of
`ModelAvgResult` is a `Base.show` method.

---

## 6. Divergences from `cAIC4` and their dispositions

1. **Full-precision df (code fix, documented divergence).** `cAIC4` feeds the optimizer `df`
   **rounded to 2 decimals** — `getWeights` reads it from `anocAIC`, whose columns are built with
   `round(unlist(...), digits = 2)` (`R/methods.R:63`); the σ-source `which.max(modelcAIC$df)` is
   also on rounded df. This is a print-formatting artifact leaking into numerics. The port uses
   **full-precision `ρᵢ`** (CLAUDE §1: mathematical correctness; "faithful to the algorithm, not to
   a bug", ADR-0007 decision 3). Consequence: the port's weights deviate from `cAIC4`'s by a
   rounding-induced amount — *not* a bit-match. **Disposition:** recorded divergence (DECISIONS
   2026-05-31); neutralised at Level-1 (§7) by feeding *identical* df both sides; absorbed at
   Level-2 by the measured band.
2. **`solve(cz)` at `weightOptim.R:154` (§9-compliant transcription).** `cAIC4` forms a literal
   matrix inverse of the Cholesky factor; the port transcribes it as the algebraically-exact
   triangular solves `cz' * v == cz_U' \ v` / `cz * v == cz_U \ v`, honouring CLAUDE §9/§12 with no
   exceptions. ADR-0007 decision 2 (which had kept the literal `inv`) was **withdrawn 2026-05-31**.
   **Disposition:** DECISIONS 2026-05-31 + an in-code comment at the site recording the equivalence.
3. **`predictma` `new_re_levels` default.** Default `:error` overrides `MixedModels`' own `:missing`
   default, to mirror `cAIC4`/`lme4`'s `allow.new.levels = FALSE`. **Disposition:** recorded
   divergence (DECISIONS 2026-05-31); kwarg-exposed.
4. **Non-unique minimiser / ill-conditioning.** On collinear candidates `MᵀM` is singular; `cAIC4`
   returns a current iterate silently, the port returns the same but **warns**. **Disposition:**
   CLAUDE §4 strengthening (ADR-0007 decision 4); §7 anchors stable functionals.
5. **No `duration` semantics divergence.** `cAIC4` stores an R `difftime`; the port stores a
   concrete `Float64` seconds (`@elapsed`/`time_ns`), excluded from reproducibility assertions.

---

## 7. Validation plan (two levels, per CLAUDE §6 / ADR-0003)

**Level-1 — optimizer machinery isolated.** A shared-input fixture
(`test/generate_fixtures_modelavg.{jl,R}`) writes synthetic `(y, M, ρ, σ̂²)` on the Julia side and
runs `cAIC4`'s `getWeights`/`.weightOptim` arithmetic on the R side on the **same** inputs — feeding
**identical, full-precision** `ρ` both sides, which **isolates the transcription from the §6.1
df-rounding divergence**. The convex QP is strictly convex on a **well-conditioned** synthetic `M`
(`MᵀM ≻ 0`), so both solvers converge to the *same unique* `ŵ`; the gate is the weight vector and
objective at a **measured** tolerance (target `rtol = 1e-6`; relaxed to the iterative stopping band
if the `tol=1e-8` outer stop forces it — recorded in DECISIONS at implementation, not loosened to
pass). Cases vary `M ∈ {2,3,5}` and conditioning.

**Level-2 — end-to-end `modelAvg`.** Fit a candidate set in `lme4` + `MixedModels` on a common
embedded sample (RNGs never meet), run `cAIC4::modelAvg` vs `modelavg`:
- **Weight vector** anchored *only* on a deliberately **well-conditioned** scenario (distinct,
  non-collinear candidate fits → unique minimiser), within a band = `max(`lme4↔MM fit discrepancy,
  §6.1 df-rounding perturbation`)`, measured.
- **Stable functionals** anchored on **every** scenario (incl. the nested/collinear `Orthodont`-style
  set): the model-averaged **prediction** `ŷ^MA` (`predictma`) and the **objective value** `J(ŵ)`,
  which are stable under a non-unique `ŵ` — the M4.5 analogue of `stepcaic`'s "path only on
  well-separated cases" (DECISIONS 2026-05-30).
- **Buckland weights** (`opt=FALSE`) anchored directly — a deterministic `exp(−Δ/2)` map of the
  Level-2-validated cAICs, so it inherits the M2 `atol = 1e-3` band.

**Type stability / Aqua / JET** per CLAUDE §8: `@inferred` on `getweights`, `modelavg`, `predictma`;
the transcription must be type-stable (`T<:AbstractFloat` generic) despite mirroring dynamically-typed R.
