# B-source AD strategy: ForwardDiff rides the experimental extension, FiniteDiff is self-driven

**Status:** accepted (2026-05-27). Builds on [ADR-0001](0001-finite-differences-constrained-not-banned.md); refines the `DECISIONS.md` entry "ForwardDiff and FiniteDiff as core dependencies." Pinned against `MixedModels =5.5.1`.

## Context

The Greven–Kneib correction needs **B**, the Hessian of the (restricted) marginal
log-likelihood w.r.t. the variance parameters θ. `cAIC.jl` exposes three B-sources:
`:analytic` (closed form), `:forwarddiff`, `:finitediff` (§9, ADR-0001). The latter
two must obtain a Hessian of `MixedModels`' objective (= −2·profiled log-likelihood)
w.r.t. θ.

`MixedModels` v5.5.1 ships experimental package extensions for exactly this —
`MixedModelsForwardDiffExt` (`ForwardDiff.hessian(m::LinearMixedModel)`) and
`MixedModelsFiniteDiffExt` (`FiniteDiff.finite_difference_hessian(m::LinearMixedModel)`).
The docs mark them "subject to change without being considered breaking," including
whether σ and β are differentiated alongside θ. Such a change would silently alter
**B**'s dimension and meaning, so this dependency is exact-version-pinned (`=5.5.1`)
and quarantined in `mm_internals.jl` with shape assertions (§3).

A hard asymmetry governs the design. `MixedModels` relies on in-place methods that
break naive AD — which is *why* the extension reimplements an **out-of-place**
objective:

- **ForwardDiff has no real choice.** AD through the in-place objective fails; we must
  use `MixedModelsForwardDiffExt`. Reimplementing the out-of-place objective ourselves
  would merely duplicate that extension.
- **FiniteDiff has a choice.** FD only *evaluates* the objective at perturbed θ — no
  AD-compatibility is needed and in-place is fine. We can drive `FiniteDiff` over
  `MixedModels`' **stable** public objective (`objective(m)` + `setθ!`/`updateL!`)
  instead of the experimental extension.

Per ADR-0001, the FD B-source's purpose is to be the **fallback when AD fails on the
upstream object**. Riding the *same* experimental surface as ForwardDiff would make it
break in the same place — disqualifying it as that fallback.

## Decision

- `:analytic` — closed-form B from the model's matrices; no AD, no experimental
  surface (the default).
- `:forwarddiff` — `ForwardDiff.hessian(m::LinearMixedModel)` via the experimental
  `MixedModelsForwardDiffExt`. The **only** path on experimental surface.
- `:finitediff` — `FiniteDiff` driven by `cAIC.jl` over `MixedModels`' **stable**
  `objective`/`setθ!` API; **not** `MixedModelsFiniteDiffExt`.

The `mm_internals.jl` internal-access table therefore lists, for the AD paths:
`ForwardDiff.hessian(::LinearMixedModel)` (experimental — shape-asserted, version-frozen)
and `objective` / `setθ!` / `updateL!` (stable). `FiniteDiff.finite_difference_hessian(::LinearMixedModel)`
is **not** accessed. `ForwardDiff` and `FiniteDiff` remain direct core dependencies.

## Considered alternatives

- **Ride both extensions** (`:finitediff` → `FiniteDiff.finite_difference_hessian(m)`).
  Rejected: the FD path would needlessly inherit the experimental API's instability
  (the σ-inclusion drift) and could not act as the fallback for a broken ForwardDiff
  extension — both paths would fail together.
- **Self-drive both** (drop both extensions). Rejected: AD over an in-place objective
  fails; providing an out-of-place objective for ForwardDiff is exactly what
  `MixedModelsForwardDiffExt` already does — reimplementing it would be large, brittle,
  and duplicate upstream.

## Consequences

- Exactly one B-source (`:forwarddiff`) touches experimental `MixedModels` surface; the
  σ-inclusion drift risk is contained there and caught by shape assertions + the exact pin.
- `cAIC.jl` owns a small FD driver — a θ→objective closure handed to `FiniteDiff`. It
  **must** restore the model to its fitted θ̂ (or operate on a copy) after perturbing θ:
  a `setθ!` left in a perturbed state is a defect — fail loud. This lives in
  `mm_internals.jl` because it touches `setθ!`/`objective`.
- The FD path depends only on the long-stable `objective`/`setθ!` API, so it survives
  churn in the experimental extension — realizing ADR-0001's "FD as the fallback when AD
  fails."
- None of this is the default: `:analytic` touches no AD machinery at all.
