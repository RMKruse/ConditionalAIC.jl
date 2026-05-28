# 0001 — Numerics primitives: stable identities

The `cAIC.Numerics` submodule holds the pure, numerically-stable kernels the cAIC is
assembled from (CLAUDE.md §9). No `MixedModels` dependency; generic over
`T <: AbstractFloat`. Each kernel computes a quantity via its *stable* form and is
validated against the *naive* form on synthetic inputs at the Level-1 tolerance
(`rtol = 1e-6`, `atol = 1e-10`). The naive forms below are the **reference**
expressions used only in tests — never in shipped code.

This note is the §7 step-1 "state the math" record for issue #5.

---

## `traceprod(A, B)` — trace of a product without forming the product

For `A ∈ ℝ^{m×n}` and `B ∈ ℝ^{n×m}`,

```
tr(AB) = Σᵢ (AB)ᵢᵢ = Σᵢ Σₖ Aᵢₖ Bₖᵢ.
```

The right-hand double sum touches each entry of `A` and `B` once and **never
materialises** the `m×m` product `AB` (CLAUDE.md §9: "traces ... without
materialising full matrix products"). Equivalently `tr(AB) = Σ A .* Bᵀ`.

- **Stable form (shipped):** `Σᵢₖ Aᵢₖ Bₖᵢ`.
- **Naive reference (test):** `tr(A * B)`.
- **Domain:** `size(A) == reverse(size(B))`; otherwise `DimensionMismatch`. The empty
  contraction (`n == 0`, or `m == 0`) is `0`.

Used for the naive plug-in df `ρ₀ = tr(H₁)` and the trace terms of the
Greven–Kneib correction.

---

## `logdetpd(A)` — log-determinant of an SPD matrix via its Cholesky factor

For symmetric positive-definite `A = L Lᵀ` (lower Cholesky factor `L`),

```
logdet(A) = log ∏ᵢ Lᵢᵢ² = 2 Σᵢ log Lᵢᵢ.
```

This is computed from the triangular factor's diagonal, in log-space, and **never**
forms `det(A)` (CLAUDE.md §9: "determinants via `logdet` on a Cholesky/triangular
factor — never `det` of a large or near-singular matrix").

- **Stable form (shipped):** `cholesky(Symmetric(A))` then `2 Σ log Lᵢᵢ` (i.e.
  `LinearAlgebra.logdet` on the factor). A `Cholesky` factor may be passed directly to
  avoid refactorising (§9: "use them; do not refactorize").
- **Naive reference (test):** `log(det(A))`.
- **Domain:** `A` square and positive-definite; a non-PD `A` is outside the domain and
  raises `DomainError` (a failed Cholesky), never a silently-wrong number.

Used for the log-determinant terms of the Gaussian (log-)likelihood and bias
correction.

---

## `invquad(A, x)` — quadratic form `xᵀ A⁻¹ x` via a Cholesky solve

For SPD `A = L Lᵀ`,

```
xᵀ A⁻¹ x = xᵀ (L Lᵀ)⁻¹ x = (L⁻¹ x)ᵀ (L⁻¹ x) = ‖L⁻¹ x‖².
```

`L⁻¹ x` is obtained by a triangular **solve** (`L \ x`); no explicit inverse is ever
formed (CLAUDE.md §9/§12: "never form an explicit inverse"). The result is a sum of
squares, hence `≥ 0` for real `x`.

- **Stable form (shipped):** `C = cholesky(Symmetric(A)); y = C.L \ x; ‖y‖²`. A
  `Cholesky` factor may be passed directly.
- **Naive reference (test):** `dot(x, inv(A) * x)`.
- **Domain:** `A` square SPD, `length(x) == size(A, 1)`; otherwise `DimensionMismatch`,
  and a non-PD `A` raises `DomainError`.

Used for the Mahalanobis / weighted-residual terms in the Gaussian likelihood and
covariance penalties.

---

## `logsumexp(x)` — stable log of a sum of exponentials

For `x ∈ ℝⁿ` with `m = maxᵢ xᵢ`,

```
log Σᵢ exp(xᵢ) = m + log Σᵢ exp(xᵢ − m).
```

Factoring out the maximum keeps every exponent `≤ 0`, so the sum neither overflows
nor underflows to `0` (CLAUDE.md §9: "log-space for all likelihoods ... never sum raw
products of small numbers"). Delegates to `LogExpFunctions.logsumexp`; this wrapper
fixes the package's vetted entry point and its generic-`T` / empty-input contract.

- **Stable form (shipped):** `LogExpFunctions.logsumexp(x)`.
- **Naive reference (test):** `log(sum(exp, x))` (valid only in the no-overflow regime).
- **Domain:** empty `x` ↦ `-Inf` (`log` of an empty sum); a `NaN` entry propagates.

Reserved for the GLMM (M3) log-space accumulations; established here as part of the
shared primitive layer.
