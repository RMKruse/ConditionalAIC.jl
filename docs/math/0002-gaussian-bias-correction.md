# 0002 — Gaussian LMM bias correction: components, closed-form B, and ρ₀

This note is the §7 step-1 "state the math" record for issue #4 (milestone M2). It pins,
in precise notation, the mathematics the M2 Gaussian path implements **before any Julia
code is written**: the `calculateGaussianBc` component layout, the closed-form
Greven–Kneib Hessian **B** (`analytic = TRUE`), and the naive plug-in effective degrees
of freedom ρ₀ = tr(H₁). It is the single human-reviewed gate of this milestone — a subtly
wrong component definition is *silently* wrong downstream — so every definition is pinned
against the actual `cAIC4` source and the pinned `MixedModels` source, not asserted from
memory.

**Ground-truth sources consulted**
- `cAIC4` **v1.1** (CRAN, 2025-04-04): `R/calculateGaussianBc.R`, `R/getModelComponents.R`
  (`getModelComponents.merMod`), `R/biasCorrectionGaussian.R`, `R/bcMer.R`, `R/cAIC.R`,
  `R/getcondLL.R`. `lme4` ≥ 1.1-6 (the `getME` accessors `X, Z, theta, Lambda, Lambdat,
  L, RX, Lind, y, mu`).
- Greven, S. & Kneib, T. (2010), *On the behaviour of marginal and conditional AIC in
  linear mixed models*, **Biometrika** 97(4), 773–789. Cited below by equation/theorem.
- `MixedModels` **= 5.5.1** (the exact pin; `src/linearmixedmodel.jl`): `leverage`,
  `ranef`.

Where `cAIC4` and the scanned Greven–Kneib (2010) PDF appear to disagree on a sign or a
constant, **`cAIC4` is ground truth** (CLAUDE.md §2); §5 records the one place this
matters and flags it for the reviewer.

---

## 0. The fitted linear mixed model and its scaled marginal variance

We score a Gaussian `LinearMixedModel` already fitted by `MixedModels.jl`. In the notation
of Greven & Kneib (2010), eq. (1),

```
y = X β + Z b + ε,     b ~ N(0, σ² D*),   ε ~ N(0, σ² Iₙ),   b ⫫ ε,
```

with `X` (n×p, rank p) the fixed-effects design, `Z` (n×q) the random-effects design, and
`D*` the **relative** random-effects covariance (covariance of `b` in units of σ²). The
marginal variance is

```
V = cov(y) = σ²(Iₙ + Z D* Zᵀ) = σ² V₀,        V₀ = Iₙ + Z D* Zᵀ
```

— G&K's `V = σ² V*`; we write `V₀` for the **scaled** marginal variance (G&K's `V*`),
matching `cAIC4`'s `V0inv = V₀⁻¹`.

`MixedModels`/`lme4` parametrise `D*` through the **relative covariance factor** `Λ = Λ(θ)`
(`getME(m, "Lambda")`, q×q, lower-triangular), with

```
D* = Λ Λᵀ,        cov(b) = σ² Λ Λᵀ,        b = Λ u,   u ~ N(0, σ² I_q),
```

so `V₀ = Iₙ + Z Λ Λᵀ Zᵀ`. The free covariance parameters are `θ` (`getME(m, "theta")`,
length `s` = number of free lower-triangular entries of the per-group block of `Λ`).

The BLUE/BLUP (G&K eq. (2)) give `β̂ = (Xᵀ V⁻¹ X)⁻¹ Xᵀ V⁻¹ y` and
`b̂ = D Zᵀ V⁻¹(y − X β̂) = D* Zᵀ V₀⁻¹(y − X β̂)`. The conditional fitted mean is
`ŷ = X β̂ + Z b̂` (`= getME(m, "mu")`), and the **conditional residual** is `e = y − ŷ`.

**Residual identity (used throughout).** With the fixed-effects-adjusted projector

```
A = V₀⁻¹ − V₀⁻¹ X (Xᵀ V₀⁻¹ X)⁻¹ Xᵀ V₀⁻¹          (G&K's A*, symmetric, A X = 0)
```

one has `A y = V₀⁻¹(y − X β̂)` and, since `Z D* Zᵀ V₀⁻¹ = (V₀ − Iₙ)V₀⁻¹ = Iₙ − V₀⁻¹`,

```
e = y − ŷ = (Iₙ − Z D* Zᵀ V₀⁻¹)(y − X β̂) = V₀⁻¹(y − X β̂) = A y.
```

So **e = A y** and the scalar `tʸᵉ := yᵀ e = yᵀ A y`. (`cAIC4` computes `e` from the fit
as `y − mu`; the identity `e = A y` is what makes the component formulas below collapse to
quadratic forms in `y`.)

---

## 1. The conditional AIC and the effective degrees of freedom ρ

The conditional AIC is

```
cAIC = −2 · ℓ_cond(y | b̂, β̂, θ̂) + 2 ρ,
```

`ℓ_cond` the Gaussian conditional log-likelihood `Σᵢ log φ(yᵢ; ŷᵢ, σ̂²)` (the `loglik`
module; `cAIC4::getcondLL`), and `ρ` the **bias-corrected effective degrees of freedom**.
`cAIC4` assembles ρ (in `calculateGaussianBc`) as

```
ρ  =  ρ₀  +  Δρ_GK  +  sigma.penalty,
      └─┬─┘   └──┬──┘   └─────┬──────┘
     §2 naive  §4 estimation-  σ²-estimation
     plug-in   uncertainty     penalty (default 1)
               correction
```

- **ρ₀** (§2) is the naive plug-in df `tr(H₁)`, the trace of the hat matrix `y ↦ ŷ` at the
  estimated, *fixed* variance parameters.
- **Δρ_GK** (§4–§5) is the Greven–Kneib correction for the estimation of `θ`. It is the
  term that makes `ρ ≥ ρ₀`; G&K show the naive ρ₀ systematically **understates** the
  effective df because `θ` is estimated from the same `y`.
- **sigma.penalty** is the number of estimated parameters in the residual error
  (co)variance — default `1` (one estimated σ²); `0` if the error variance is known;
  the number of estimated weights for individual variances (`cAIC4::cAIC` docstring,
  `R/cAIC.R`).

**Precondition (singular fits).** `biasCorrectionGaussian` first calls
`deleteZeroComponents(m)`; a variance component estimated on the boundary is dropped and
the **reduced** model is what `getModelComponents` + `calculateGaussianBc` are evaluated
on. If reduction removes all random effects (`inherits(·, "lm")`) the bias correction is
just `rank + sigma.penalty` (`R/biasCorrectionGaussian.R`). The reduction itself is out of
scope for this note (it is the `caic`-assembly / singular-fit path); everything below
assumes the (possibly reduced) model handed to `getModelComponents`.

---

## 2. ρ₀ = tr(H₁): the naive plug-in effective degrees of freedom

`ŷ = X β̂ + Z b̂` is linear in `y` at fixed `θ̂`: `ŷ = H₁ y`. Following Vaida & Blanchard
(2005) as recapitulated by G&K §2.3, the conventional conditional-AIC effective df is the
trace of this hat matrix,

```
ρ₀ = tr(H₁),     H₁ = [Z X] ⎡ZᵀZ + D*⁻¹   ZᵀX⎤⁻¹ ⎡ZᵀZ   ZᵀX⎤   (G&K §2.3 display)
                          ⎣ XᵀZ          XᵀX⎦    ⎣XᵀZ   XᵀX⎦
```

i.e. the trace of the matrix projecting `y` onto `ŷ`. From the residual identity `e = A y`
of §0, `Iₙ − H₁ = A`, hence in the unweighted case (residual weights all 1, `R = Iₙ`)

```
ρ₀ = tr(H₁) = n − tr(A).
```

This is exactly `cAIC4`'s base term `df ← n − tr(R A)` with `R = Iₙ` (see §3 for the
weighted generalisation `R`).

**`MixedModels` source of ρ₀ (confirmed, v5.5.1).** `cAIC.jl` does **not** re-derive ρ₀
from `A`; it reads it from the leverage accessor. `leverage(m::LinearMixedModel)` returns
**the diagonal of the hat matrix** — a length-`n` *per-observation* vector, **not** the
trace (`src/linearmixedmodel.jl`, docstring: "Return the diagonal of the hat matrix … the
sum of the leverage values is the degrees of freedom"). The hat matrix is

```
H₁ = [ZΛ  X] (L Lᵀ)⁻¹ [ZΛ  X]ᵀ
```

(`src/linearmixedmodel.jl:753`), the same `H₁` as above written in `MixedModels`' joint
Cholesky `L`. Therefore

```
ρ₀ = tr(H₁) = sum(leverage(m)).
```

ρ₀ is σ-free (built from the relative `Λ`, `L`); the σ² estimation is accounted for
separately by `sigma.penalty` (§1).

---

## 3. The `getModelComponents.merMod` component layout

`calculateGaussianBc` consumes the list returned by `getModelComponents.merMod`
(`R/getModelComponents.R`). Pinned definitions (unweighted Gaussian `lmer` fit, all
residual weights 1 — the M2 target; weighted Gaussian is "not yet implemented" in `cAIC4`,
`R/cAIC.R`/`R/getcondLL.R`):

| Component | `cAIC4` field | Definition | Shape |
|:----------|:--------------|:-----------|:------|
| `X`         | `model$X`       | fixed-effects design, `getME(m,"X")`                              | n×p |
| `Z`         | `model$Z`       | random-effects design, `getME(m,"Z")`                             | n×q |
| `Λ`, `Λᵀ`   | `model$Lambda`, `model$Lambdat` | relative covariance factor, `getME(m,"Lambda"/"Lambdat")` | q×q |
| `θ`         | `model$theta`   | free covariance parameters, `getME(m,"theta")`                    | s |
| `V₀⁻¹`      | `model$V0inv`   | `Iₙ − (L⁻¹ P Λᵀ Zᵀ)ᵀ(L⁻¹ P Λᵀ Zᵀ) = (Iₙ + Z Λ Λᵀ Zᵀ)⁻¹` (Woodbury, via the sparse `L`; never an explicit inverse) | n×n |
| `A`         | `model$A`       | `V₀⁻¹ − V₀⁻¹ X (Xᵀ V₀⁻¹ X)⁻¹ Xᵀ V₀⁻¹`, formed as `V0inv − crossprod(crossprod(X·RX⁻¹, V0inv))` with `RXᵀRX = Xᵀ V₀⁻¹ X` | n×n |
| `R`         | `model$R`       | residual error-correlation factor; **unset (NULL)** when all weights are 1. Weighted: `R = diag(1/w)` | n×n |
| `y`         | `model$y`       | response, `getME(m,"y")`                                          | n |
| `e`         | `model$e`       | conditional residual `y − getME(m,"mu") = A y` (§0)               | n |
| `tʸᵉ`       | `model$tye`     | `yᵀ e = yᵀ A y`                                                   | scalar |
| `Wₛ`        | `model$Wlist[[s]]` | `Z Dₛ Zᵀ` (the derivative matrices; §6)                        | n×n, s of them |
| `eᵀWₛe`     | `model$eWelist[[s]]` | `eᵀ Wₛ e` (residual quadratic forms)                          | scalar, s of them |
| `B`         | `model$B`       | `analytic=TRUE`: `0_{s×s}`, **filled** in `calculateGaussianBc` (§4); `analytic=FALSE`: `m@optinfo$derivs$Hessian` (lme4's stored numeric Hessian, *lifted* unchanged) | s×s |
| `C`         | `model$C`       | `0_{s×n}`, filled in `calculateGaussianBc` (§4)                   | s×n |
| `isREML`    | `model$isREML`  | `isREML(m)`                                                       | Bool |

`RA` in `calculateGaussianBc` is `R %*% A` if `R` is set, else `A` (unweighted). The M2
spec targets the unweighted path: `R = Iₙ`, `RA = A`.

---

## 4. The closed-form Greven–Kneib correction (`analytic = TRUE`)

This is the exact arithmetic of `calculateGaussianBc(model, sigma.penalty, analytic=TRUE)`
(`R/calculateGaussianBc.R`), the authoritative definition the Level-1 port (#8) implements
and the Level-1 fixture (#7) validates. Index `j, k ∈ {1,…,s}`; `p = ncol(X)`; `n` = #obs.

**Per-θ-component weighted matrices** (the only place ML vs REML changes the matrix used):

```
WAₛ = Wₛ V₀⁻¹      (ML,   isREML = false)
WAₛ = Wₛ A         (REML, isREML = true)
```

**The cross-derivative matrix C (s×n), row j** — same formula for ML and REML:

```
C[j, :] = (eᵀ Wⱼ) A − (eᵀ Wⱼ e) · eᵀ / (2 tʸᵉ)
        = eᵀ Wⱼ A − (eᵀ Wⱼ e) / (2 yᵀe) · eᵀ.
```

**The Hessian-source matrix B (s×s), symmetric, entry (j,k)** for `k ≥ j`, with the scalar
`WkAWjA := Σ (WAⱼ)ᵀ ⊙ WAₖ = tr(Wⱼ A Wₖ A)` (REML) or `tr(Wⱼ V₀⁻¹ Wₖ V₀⁻¹)` (ML) — a trace
formed by elementwise product, **without** materialising the matrix product (CLAUDE.md §9;
`numerics.traceprod`):

```
            tʸᵉ · WkAWjA      (eᵀWⱼe)(eᵀWₖe)
B[j,k] =  − ──────────────  − ───────────────  +  eᵀ Wₖ A Wⱼ e,
              2 · nθ              2 · tʸᵉ
```

```
nθ = n        (ML),        nθ = n − p   (REML).
```

(`B[k,j] = B[j,k]`.) The three terms are, in order: the variance-parameter Fisher-type
trace term (the only ML/REML-dependent term), the squared-residual-quadratic-form term, and
the residual quartic-form term.

**Solve and assemble.** With `Λ̂ʸ := B⁻¹ C` (s×n) — `solve(B) %*% C`, falling back to a
Cholesky solve `backsolve(chol(B), …)` when `B` is (numerically) not invertible by the
direct solve; CLAUDE.md §9 / `numerics.invquad` style, no explicit inverse —

```
ρ  =  [ n − tr(R A) ]  +  Σⱼ  Λ̂ʸ[j, :] · (R A Wⱼ e)  +  sigma.penalty
       └──── ρ₀ ────┘     └──────────── Δρ_GK ────────────┘
```

i.e. `df ← n − tr(RA); for j: df += Λ̂ʸ[j,:] · (RA Wⱼ e); if sigma.penalty: df += sigma.penalty`.
For the unweighted M2 target `R = Iₙ`, so `R A = A` everywhere.

For completeness, the **`analytic = FALSE`** path (the numeric-Hessian source) uses the
same assembly but a different `C` and lifts `B` from `lme4`'s stored Hessian:

```
C[j, :] = (2 nθ / tʸᵉ) · ( eᵀ Wⱼ A − (eᵀ Wⱼ e) · eᵀ / tʸᵉ ),     B = m@optinfo$derivs$Hessian.
```

M2 ships `analytic = TRUE` as the default and primary path (PRD `hessian = :analytic`);
the `:forwarddiff` / `:finitediff` B-sources (ADR-0002) are the `analytic = FALSE`
analogues and are specified separately.

---

## 5. Correspondence to Greven & Kneib (2010), Theorem 3

Theorem 3 gives the analytic corrected effective df `Φ₀` (their notation; our ρ before the
σ-penalty) as

```
Φ₀ = ρ₀ + Σⱼ (correction)ⱼ,
```

built from `A* = V*⁻¹ − V*⁻¹X(XᵀV*⁻¹X)⁻¹XᵀV*⁻¹` (= our `A`), the derivative matrices

```
W*ⱼ = ∂V* / ∂θ*ⱼ,        U*ⱼᵢ = ∂²V* / ∂θ*ᵢ∂θ*ⱼ,
```

the s×n matrix `G*` whose row `j` is `2{(yᵀA*y)(yᵀA*W*ⱼA*) − (yᵀA*W*ⱼA*y)(yᵀA*)}`, and the
**negative-definite** Hessian `B*` with entries combining `(yᵀA*W*ⱼA*y)(yᵀA*W*ₖA*y)`,
`yᵀA*W*ⱼA*W*ₖA*y · yᵀA*y`, and the ML/REML trace term
`tr(U*ⱼₖV*⁻¹ − W*ⱼV*⁻¹W*ₖV*⁻¹)/n` (ML) / `tr(U*ⱼₖA* − W*ⱼA*W*ₖA*)/(n−p)` (REML).

**Why `cAIC4` has no second-derivative (`U`) term.** Theorem 3's canonical case (G&K, the
remark following the theorem) is `D*` block-diagonal with parameters the variance
components, giving `W*ⱼ = Zⱼ Σⱼ Zⱼᵀ` and **`U*ⱼᵢ = 0`**. `cAIC4` realises exactly this by
parametrising `V₀` through the **entries of the relative covariance `D* = Λ Λᵀ`**: the free
entries of (each block of) `D*` share the lower-triangular sparsity pattern of `Λ`, so they
are in one-to-one correspondence with `θ` and are enumerated via `lme4`'s `Lind` (§6). In
this parametrisation

```
V₀ = Iₙ + Z D* Zᵀ   is affine in the entries of D*   ⟹   W*ⱼ = Z Dⱼ Zᵀ,   U*ⱼᵢ = 0,
```

which is why `getModelComponents` builds `Wlist` but no `U`-list, and `calculateGaussianBc`
has no second-derivative term. The Greven–Kneib correction is a sum of chain-rule terms
`Σⱼ (∂ρ₀/∂θ*ⱼ)·(estimation uncertainty in θ̂*ⱼ)` and is **invariant** under smooth
reparametrisation of `θ*` (the Jacobians cancel), so evaluating it in the affine
`D*`-entry parametrisation is exact and yields the same number as `lme4`'s `θ`
parametrisation.

**The estimation-uncertainty term, via implicit differentiation.** With ρ = tr(∂ŷ/∂yᵀ),
the chain rule splits the derivative into the fixed-`θ̂` part (trace ρ₀) and the part
flowing through `θ̂(y)`:

```
ρ = ρ₀ + Σⱼ tr[ (∂ŷ/∂θ*ⱼ) (∂θ̂*ⱼ/∂yᵀ) ],      ∂θ̂*/∂yᵀ = −H⁻¹ J     (implicit function thm),
```

`H = ∂²ℓ/∂θ*∂θ*ᵀ` the (negative-definite) Hessian of the (restricted) profile
log-likelihood and `J = ∂²ℓ/∂θ*∂yᵀ` the cross-derivative. `cAIC4` computes precisely this:
`Λ̂ʸ = B⁻¹ C` plays the role of `∂θ̂*/∂yᵀ`, with `B` the **positive-definite** negative
Hessian (`= −B*`, hence the Cholesky fallback `chol(B)`) and `C` the cross-derivative; the
contraction `Σⱼ Λ̂ʸ[j,:]·(R A Wⱼ e)` is `Σⱼ (∂θ̂*ⱼ/∂yᵀ)(∂ŷ/∂θ*ⱼ)`. This is the term that
makes **`ρ ≥ ρ₀`**, with equality iff the random effects are predicted exactly zero
(G&K: the conventional plug-in ρ₀ "ignores estimation uncertainty in the random-effects
covariance matrix" and so understates the df) — PRD user-story 5, the `ρ ≥ ρ₀` sanity
check.

**Authority and one flagged sign.** `cAIC4`'s `B`/`C` are **not** term-by-term scalar
multiples of Theorem 3's `B*`/`G*`; they are a re-organised but equivalent computation of
the same `∂θ̂*/∂yᵀ`. The §4 transcription of `calculateGaussianBc` is therefore the
**authoritative** definition (CLAUDE.md §2: `cAIC4` is ground truth), and the Level-1
fixture against `cAIC4:::calculateGaussianBc` (#7) is what pins the arithmetic
bit-for-bit. The scanned Greven–Kneib (2010) PDF is legible enough to map every component
above, **except** the interior sign of the `W*ⱼA*W*ₖA*` term inside `B*`, where the OCR is
ambiguous; the §4 code resolves it (the term enters `B[j,k]` with a leading `+`).
*Confirmed (HITL, cAIC4 author, 2026-05-27):* the leading `+` is correct. In any case it
does not affect the port (which transcribes §4) or the fixture (which compares against the
R source), only the narrative cross-reference here.

---

## 6. Construction of the derivative matrices Wₛ

`getModelComponents` builds each `Wₛ = Z Dₛ Zᵀ` (`R/getModelComponents.R`, the loop over
`theta`). For covariance parameter `s`, let `Pₛ` be the set of positions `(k,l)`, `k ≥ l`,
of the lower-triangular factor `Λ` that carry `θₛ` (via `getME(m,"Lind")`, expanded across
all group copies). `Dₛ` is the symmetric q×q matrix

```
(Dₛ)_{kk} = 1   for diagonal positions (k,k) ∈ Pₛ,
(Dₛ)_{kl} = (Dₛ)_{lk} = 1   for off-diagonal positions (k,l) ∈ Pₛ,    0 elsewhere.
```

(In code: zero `Λ`'s nonzeros, set the `Lind == s` entries to 1 to obtain `∂Λ/∂θₛ`, then
symmetrise keeping the original diagonal — `Ds = (LambdaS + LambdaSt)` off-diagonal,
`diag(Ds) = diag(LambdaS)`.) Then `Wₛ = tcrossprod(Z %*% Dₛ, Z) = Z Dₛ Zᵀ`.

Interpretation (§5): `Dₛ = ∂D*/∂(s-th free entry of D* = Λ Λᵀ)`, so
`Wₛ = ∂V₀/∂(D*-entry s)` — the derivative of the scaled marginal variance in the affine
`D*`-entry parametrisation. *Worked check (random intercept):* `Λ = θ I_q`, the single free
entry is the relative variance `λ = θ²`, `Dₛ = ∂(λ I_q)/∂λ = I_q`, `Wₛ = Z Zᵀ`, and `V₀ = Iₙ
+ λ Z Zᵀ` is affine in `λ` (so `U = 0`), matching the construction.

For Level-1, `Wₛ`, `A`, `e`, `V₀⁻¹`, `B`, `C`, … are fed as synthetic *inputs* (ADR-0003,
parametrisation-neutral dense matrices); this §6 construction is what the Level-2 scoring
spine builds from a real `MixedModels` `Λ`.

---

## 7. Confirmed `MixedModels.jl` upstream semantics (pinned v5.5.1)

The two semantics issue #4 requires resolving from upstream, recorded here as ground truth
for the port:

1. **`b̂ = Λ û` (random-effects convention).** `ranef(m::LinearMixedModel; uscale=false)`
   returns the conditional modes "on the original scale" by default; `uscale=true` returns
   the spherical `u` (`src/linearmixedmodel.jl:942`). Original scale = `b̂ = Λ û` (the
   relative covariance factor applied to the spherical modes). The conditional fitted mean
   `ŷ = X β̂ + Z b̂` and hence `e = y − ŷ` use these original-scale modes. (Recorded in the
   `MMInternals.bhat` internal-access table.)

2. **`leverage` returns the per-observation diagonal, not the trace.**
   `leverage(m::LinearMixedModel)` returns the length-`n` diagonal of
   `H₁ = [ZΛ X](LLᵀ)⁻¹[ZΛ X]ᵀ` (§2). The naive plug-in df is its **sum**:
   `ρ₀ = tr(H₁) = sum(leverage(m))`. This is the `MixedModels`-native source of ρ₀; it
   equals `cAIC4`'s `n − tr(RA)` (unweighted) by the identity `Iₙ − H₁ = A` of §0.

---

## 8. Provenance and what this enables

- **Versions pinned:** `cAIC4` v1.1; `lme4` ≥ 1.1-6 (`getME` accessors); `MixedModels`
  = 5.5.1 (`leverage`, `ranef`). A bump to either side re-opens this note.
- **Sufficient for #7 (Level-1 HDF5 fixture layout):** §3 + §4 + §6 fix the component set
  (`X, Z, Λ/Λᵀ, V0inv, A, R, y, e, Wlist, eWelist, B, C, tye, isREML, theta, n`,
  `sigma.penalty`) and the exact `calculateGaussianBc` arithmetic the fixture must
  reproduce.
- **Sufficient for #8 (analytic B + ρ₀ port):** §4 is the closed-form B/C and the ρ
  assembly; §2 + §7 fix ρ₀ via the leverage accessor; §5 is the Greven–Kneib justification
  (`ρ ≥ ρ₀`).
- **Numerical-stability obligations (CLAUDE.md §9):** `V₀⁻¹` via the sparse `L` (no explicit
  inverse); `tr(Wⱼ A Wₖ A)` via elementwise product (`numerics.traceprod`); the `B⁻¹C` solve
  via a factorisation with a Cholesky fallback; `ρ₀` from `leverage` (which `MixedModels`
  computes by triangular solves against `L`).

**Reviewed and approved (HITL, 2026-05-27)** — issue #4 acceptance gate cleared, #7/#8 may
begin: (1) the §5 `B*` cross-term sign confirmed (leading `+`); (2) the unweighted
`R = Iₙ` path confirmed as the intended M2 target (weighted Gaussian deferred, per
`cAIC4`).
