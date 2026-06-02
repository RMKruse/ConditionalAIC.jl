#!/usr/bin/env Rscript
#
# Level-2 fixture generator (R / ground-truth side) — issue #8 / CLAUDE §6.
#
# Fits the reference models in `lme4` and evaluates `cAIC4`'s *public* `cAIC()` path on
# each, writing the ground-truth conditional AIC, effective degrees of freedom ρ, and
# conditional log-likelihood to an HDF5 fixture. The Julia Level-2 test fits the *same*
# models in `MixedModels.jl`, computes `cAIC.caic`, and compares against these references
# within the fit-discrepancy-derived tolerance recorded in DECISIONS.md.
#
# Unlike Level-1 (machinery in isolation, identical synthetic inputs both sides), Level-2
# is end-to-end: `lme4` and `MixedModels.jl` do not produce bit-identical fits, so the two
# optimizers settle at slightly different θ̂ on a near-flat marginal objective. The fixture
# therefore also records θ̂ and the marginal objective so that discrepancy is auditable.
#
# `cAIC4` is not installed as a package here; it is pure R, so the Gaussian merMod path is
# sourced directly from the committed cAIC4 source tree (as the Level-1 generator does for
# `calculateGaussianBc`). The gated live-RCall CI job re-validates against an installed
# `cAIC4`.
#
# HDF5 I/O via `test/fixture_io.R` (hdf5r on Linux CI, rhdf5 on macOS-ARM), matching
# `generate_fixtures.R` (see its header and the ADR-0003 addenda).
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src)
#   FIXTURE    path to the HDF5 fixture (default <script dir>/fixtures/caic_level2.h5)
#
# Usage:  Rscript test/generate_fixtures_level2.R

suppressMessages({
  source(file.path(dirname(normalizePath(sub("^--file=","",commandArgs(FALSE)[grep("^--file=",commandArgs(FALSE))]))),"fixture_io.R"))
  library(lme4)
})

caic4_src <- Sys.getenv("CAIC4_SRC", "/private/tmp/cAIC4_src")
fixture <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture <- file.path(here, "fixtures", "caic_level2.h5")
}

# Source the Gaussian merMod cAIC() path. Order matters only in that every callee must be
# defined before cAIC() is *called*, not when sourced; this list is the transitive closure
# the Gaussian, non-boundary path touches.
caic4_files <- c(
  "getcondLL.R", "calculateGaussianBc.R", "getModelComponents.R",
  "deleteZeroComponents.R", "cnms2formula.R", "helperfuns_lme.R",
  "biasCorrectionGaussian.R", "bcMer.R", "cAIC.R"
)
for (f in caic4_files) {
  path <- file.path(caic4_src, "R", f)
  stopifnot("cAIC4 source file not found; set CAIC4_SRC" = file.exists(path))
  source(path)
}
caic4_version <- tryCatch(
  unname(read.dcf(file.path(caic4_src, "DESCRIPTION"))[, "Version"]),
  error = function(e) "unknown"
)

# The reference cases. Mirrored exactly in test/caic_tests.jl (Level-2 testitem): the same
# response, fixed effects, and random-effects structure, fitted ML and REML. The `subj`/
# `days`/`reaction` ↔ `Subject`/`Days`/`Reaction` rename is the only cross-package
# difference; the `sleepstudy` data itself is identical.
cases <- list(
  slope_ml   = list(form = Reaction ~ 1 + Days + (1 + Days | Subject), reml = FALSE),
  slope_reml = list(form = Reaction ~ 1 + Days + (1 + Days | Subject), reml = TRUE),
  int_ml     = list(form = Reaction ~ 1 + Days + (1 | Subject),        reml = FALSE),
  int_reml   = list(form = Reaction ~ 1 + Days + (1 | Subject),        reml = TRUE)
)

if (file.exists(fixture)) file.remove(fixture)
dir.create(dirname(fixture), showWarnings = FALSE, recursive = TRUE)
h5createFile(fixture)
on.exit(h5closeAll())

for (name in names(cases)) {
  cs <- cases[[name]]
  fit <- lmer(cs$form, data = sleepstudy, REML = cs$reml)
  r <- cAIC(fit, sigma.penalty = 1L) # analytic = TRUE (default): the steinian Gaussian Bc
  stopifnot("unexpected boundary refit for a reference case" = isFALSE(r$new))

  # analytic = FALSE: external numeric Hessian B (lme4's optimiser Hessian), rescaled C,
  # shared assembly. This is the cAIC4 ground truth for cAIC.jl's :finitediff B-source,
  # which self-drives FiniteDiff over the same deviance-scale objective lme4 differentiates.
  r_numeric <- cAIC(fit, sigma.penalty = 1L, analytic = FALSE)
  stopifnot("unexpected boundary refit (analytic=FALSE) for a reference case" = isFALSE(r_numeric$new))

  obj <- if (isREML(fit)) REMLcrit(fit) else deviance(fit)
  h5createGroup(fixture, name)
  h5write(as.numeric(r$caic), fixture, paste0(name, "/caic"))
  h5write(as.numeric(r$df), fixture, paste0(name, "/df")) # ρ
  h5write(as.numeric(r$loglikelihood), fixture, paste0(name, "/cll"))
  h5write(as.numeric(r_numeric$caic), fixture, paste0(name, "/caic_numeric"))
  h5write(as.numeric(r_numeric$df), fixture, paste0(name, "/df_numeric")) # ρ, analytic=FALSE
  h5write(as.integer(cs$reml), fixture, paste0(name, "/reml"))
  h5write(as.numeric(getME(fit, "theta")), fixture, paste0(name, "/theta"))
  h5write(as.numeric(obj), fixture, paste0(name, "/objective"))

  cat(sprintf(
    "  %-12s caic=%.10f  df=%.10f  df_num=%.10f  cll=%.10f  (REML=%s)\n",
    name, r$caic, r$df, r_numeric$df, r$loglikelihood, cs$reml
  ))
}

h5createGroup(fixture, "meta")
h5write(caic4_version, fixture, "meta/cAIC4_version")
h5write(as.character(packageVersion("lme4")), fixture, "meta/lme4_version")
h5write(fixture_hdf5_backend(), fixture, "meta/hdf5_backend")
h5write(R.version.string, fixture, "meta/R_version")

cat(sprintf(
  "Wrote %d Level-2 reference case(s) to %s (cAIC4 %s, lme4 %s).\n",
  length(cases), fixture, caic4_version, as.character(packageVersion("lme4"))
))
