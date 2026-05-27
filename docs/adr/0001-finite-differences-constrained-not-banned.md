# Finite differences: a constrained, non-default technique — not an absolute ban

**Status:** accepted (2026-05-27). Amends CLAUDE.md §9 and §12.

## Context

CLAUDE.md originally banned finite differences (FD) in shipped code outright —
"analytic or `ForwardDiff` only; FD is a test-time cross-check" (§9, §12). The
numerical core of `cAIC.jl` is the Greven–Kneib correction, whose central object is
**B**, the Hessian of the (restricted) marginal log-likelihood w.r.t. the variance
parameters θ. `cAIC4` sources B either in closed form (`analytic=TRUE`) or by lifting
lme4's stored numeric Hessian (`analytic=FALSE`), which lme4 produces by
Richardson-extrapolated FD *during fitting*. `MixedModels.jl` optimises θ with a
derivative-free method (BOBYQA) and stores **no** Hessian, so there is nothing to
lift — any B is computed at cAIC-time. We want three selectable B-sources
(closed-form, `ForwardDiff`, finite differences): the FD one as a `cAIC4`
`analytic=FALSE` analogue and as a fallback should `ForwardDiff` fail on
`MixedModels`' (experimental) objective. A shipped FD path is impossible under the
original absolute ban.

## Decision

Reframe §9/§12 from an absolute ban to a **constraint**. FD is permitted in shipped
code **only** as an explicitly opt-in, documented, tolerance-justified path, and
**never** as the default where an analytic or `ForwardDiff` derivative exists.
Analytic remains preferred; `ForwardDiff` is the standard non-analytic form. The
rest of the numerical-stability principle (log-space, Cholesky solves, no explicit
inverses, `logdet`, stable traces) is unchanged.

## Considered alternatives

- **Test-only FD (no change).** §9 already permits FD as a test cross-check, so the
  three-way comparison is achievable with no amendment. Rejected: we want a shipped,
  user-selectable FD B-source, not merely a test oracle.
- **Single scoped exception** (FD on the one `:finitediff` path only). Rejected as
  too narrow — FD may be a legitimate documented fallback in more than one place.
- **Full removal of the rule.** Rejected — would license FD as a silent default
  anywhere, dissolving the guarantee the rule exists to provide.

## Consequences

- A shipped, opt-in FD B-source is allowed (off by default).
- It will **not** bit-match `cAIC4`'s `analytic=FALSE`: `cAIC4` lifts lme4's
  Richardson-FD Hessian of lme4's profiled deviance at lme4's θ̂, whereas `cAIC.jl`
  applies a different FD scheme to `MixedModels`' objective at `MixedModels`' θ̂
  (different optimiser, θ̂, FD algorithm, possibly REML-vs-ML objective). Agreement
  is a Level-2 derived tolerance recorded in `DECISIONS.md`, not a Level-1 match.
- A shipped FD path pulls in a `FiniteDiff` dependency (`DECISIONS.md` entry per §3)
  unless sourced via `MixedModels`' `FiniteDiff` extension.
- Any future shipped FD use must be opt-in, documented as approximate, and carry a
  justified tolerance. FD as a silent default remains a bug.
