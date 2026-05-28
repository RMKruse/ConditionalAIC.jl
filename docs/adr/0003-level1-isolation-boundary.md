# Level-1 validation isolates `calculateGaussianBc`: synthetic components in, df out

**Status:** accepted (2026-05-27). Elaborates CLAUDE.md §6 (Level-1) for the Gaussian path.

## Context

§6 mandates two-level validation. **Level-1** feeds *identical, synthetically
constructed* inputs into `cAIC4`'s internal df/bias-correction functions **and** into
the corresponding `cAIC.jl` functions, at `rtol=1e-6` / `atol=1e-10`, validating the
correction *mathematics* independently of any model fit.

`cAIC4`'s Gaussian path decomposes (confirmed from the CRAN source) as:

```
bcMer  →  biasCorrectionGaussian(object, sigma.penalty, analytic)
            ├─ getModelComponents(m, analytic)   # fit → components; the `analytic`
            │                                    #   flag selects B's source here,
            │                                    #   and lme4's θ-parametrization enters
            └─ calculateGaussianBc(...)          # components → df  (pure arithmetic)
```

`getModelComponents.merMod` returns `X, Z, Λ/Λᵗ, V0inv` (inverse marginal variance),
`A` (fixed-effects-adjusted projector), `R`, `y`, `e`, `Wlist` (∂V/∂variance-component
derivative matrices), `eWelist` (their residual quadratic forms), `B` (the Hessian —
analytic or numeric per the flag), `isREML`. We must choose the Level-1 isolation
boundary.

## Decision

Isolate at **`calculateGaussianBc`**. Level-1 fixtures hold *synthetic,
parametrization-neutral components* (`X, Z, Λ/Λᵗ, V0inv, A, R, y, e, Wlist, eWelist, B,
isREML, sigma.penalty`), fed to both `cAIC4:::calculateGaussianBc` and `cAIC.jl`'s port;
the returned df are compared at the Level-1 tolerance. `getModelComponents` — the
fit-dependent bridge that reaches into lme4 internals and carries the θ-parametrization
— is **not** a Level-1 unit; it is exercised at **Level-2** (end-to-end) instead.

## Why this boundary

- **Fit-independent.** No lme4 fit in the loop, so the correction arithmetic is tested
  in isolation — exactly what §6's Level-1 is for.
- **Parametrization-neutral.** The components are dense matrices, not θ-vectors, so the
  lme4-vs-`MixedModels` θ-parametrization difference never enters Level-1.
- **B-divergence immunity.** `B` is an *input* here, so the df-arithmetic test is
  independent of how B is sourced (closed-form vs AD) — the B-source divergence
  (`DECISIONS.md`, [ADR-0002](0002-bsource-ad-strategy.md)) cannot contaminate the
  Level-1 df test.

## Considered alternatives

- **Isolate `biasCorrectionGaussian`** (one level up, including `getModelComponents`).
  Its input is a *fitted lme4 model*, so it drags in the fit and lme4's
  θ-parametrization — Level-2 in disguise. Rejected as an L1 boundary.
- **Sub-step isolation inside `calculateGaussianBc`** (trace term, B-contraction, …
  each separately). Maximum mismatch-localization, but these are not separate `cAIC4`
  functions — it needs deep `:::` surgery and couples to `cAIC4`'s exact expression
  order. Reserved for *localizing* a mismatch the chosen boundary surfaces.

## Consequences

- The exact definition of each component (especially `A`, `Wlist`, `eWelist`) must be
  pinned in `docs/math/` **before** the port is written (§7), and `cAIC.jl`'s
  `calculateGaussianBc` port must accept components in `cAIC4`'s layout. Source of truth:
  `calculateGaussianBc.R` + `getModelComponents.R`.
- Reaching the unexported `cAIC4:::calculateGaussianBc` from the R fixture generator
  pins a `cAIC4` version (test-only, alongside the `lme4` version) — recorded when
  fixtures are wired.
- **Fixture pipeline (resolved).** `generate_fixtures.jl` constructs the seeded
  synthetic components and writes them to an **HDF5** fixture; `generate_fixtures.R`
  reads that HDF5, calls `cAIC4:::calculateGaussianBc`, and writes the reference df
  back. CI reads the HDF5 in Julia with **no R**. HDF5 is chosen for binary-exact
  Float64 round-trip, native nested-array storage (the `Wlist`/`eWelist`
  lists-of-matrices map to HDF5 groups), and independent read/write from both
  languages. Test-only deps: `HDF5.jl` (Julia) and `hdf5r` (R) — neither enters the
  package's runtime dependencies.

## Addendum (2026-05-27) — R HDF5 reader is `rhdf5`, not `hdf5r`

When the pipeline was first stood up (#7), `hdf5r` would not build against Homebrew R on
macOS-ARM: its source build links the static `libhdf5.a`, and under R's
`-undefined dynamic_lookup` bundle link the archive member for `H5Dread_chunk` is left
undefined, so the package fails to load; forcing a shared link in turn defeats its
configure version probe, and no CRAN binary exists for the Homebrew R platform string.

We therefore use **`rhdf5`** (Bioconductor) as the R HDF5 reader instead of `hdf5r`.
`rhdf5` bundles its own correctly-linked HDF5 via `Rhdf5lib`, so it has no system-HDF5
dependency and builds reliably. **Nothing else changes**: the neutral hand-off is still
HDF5, the isolation boundary is still `calculateGaussianBc`, and the Julia side is still
`HDF5.jl`. Only the R package differs (`generate_fixtures.R` uses `rhdf5::h5read` /
`h5write`). The computation-bearing matrices (`A`, `V0inv`, `Wⱼ`) are symmetric, so the
Julia↔R column/row-major round-trip leaves them unchanged; the one non-symmetric
component `X` is not stored at all (only `n`/`p` scalars are, since `calculateGaussianBc`
reads `X` solely as `ncol(X)`). `rhdf5` writes an R length-1 numeric as a 1-element
array, so the reference ρ is coerced back to a scalar on the Julia side.

The gated live-R job (`CAIC_LIVE_RCALL=1`) uses the same Rscript + HDF5 hand-off (not
`RCall.jl`); it sources the pinned `cAIC4` v1.1 `calculateGaussianBc.R` directly, since
that function is pure base R and the Level-1 boundary involves no model fit. Provenance
(the `cAIC4`, `rhdf5`, R, Julia, and `HDF5_jll` versions) is recorded in the fixture's
`meta` group at generation time.
