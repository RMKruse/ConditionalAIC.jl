#!/usr/bin/env Rscript
#
# Level-2 fixture generator (R / ground-truth side) for model averaging — issue #49 / M4.5.
#
# Fits a candidate set in `lme4` and runs `cAIC4`'s `modelAvg(models, opt = FALSE)` — the
# Buckland (1997) smoothed-weights path — writing the resulting per-candidate weights, the
# averaged fixed effects, the per-candidate conditional AICs, and the REML setting to an
# HDF5 fixture. The Julia Level-2 test fits the *same* candidates in `MixedModels.jl`, runs
# `modelavg(...; weights = :smoothed)`, and compares within the band recorded in DECISIONS.md.
#
# Two cAIC columns are stored per candidate, on purpose:
#   * `caic`        — `cAIC4`'s FULL-PRECISION conditional AIC (`cAIC(fit)$caic`). The Julia
#                     `caic` is validated against this within the M2 Level-2 band (atol=1e-3).
#   * (the weights) — `modelAvg(opt=FALSE)` forms its weights from `anocAIC`'s cAIC column,
#                     which `methods.R:63` ROUNDS to 2 digits. The Buckland weight comparison
#                     therefore absorbs that print-rounding artifact (see DECISIONS.md).
#
# `cAIC4` is sourced directly from its source tree (as the other Level-2 generators do), so
# no installed package is required in the default job; the gated live-RCall job re-validates.
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src)
#   FIXTURE    path to the HDF5 fixture (default <script dir>/fixtures/modelavg_level2.h5)
#
# Usage:  Rscript test/generate_fixtures_modelavg.R

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
  fixture <- file.path(here, "fixtures", "modelavg_level2.h5")
}

# Source the Gaussian merMod cAIC() path + anocAIC (methods.R) + modelAvg (modelAvg.R). Every
# callee must be defined before modelAvg() is *called*; this is the transitive closure the
# Gaussian, non-boundary, opt = FALSE path touches.
caic4_files <- c(
  "getcondLL.R", "calculateGaussianBc.R", "getModelComponents.R",
  "deleteZeroComponents.R", "cnms2formula.R", "helperfuns_lme.R",
  "biasCorrectionGaussian.R", "bcMer.R", "cAIC.R", "methods.R", "modelAvg.R"
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

# The candidate set. Mirrored exactly in test/averaging_tests.jl (Level-2 testitem): the same
# response and random-effects structure on `sleepstudy`, fitted ML. The candidates differ in
# RE structure (correlated slope, uncorrelated slope, intercept-only) so the cAICs of the two
# slope models are comparable — a non-degenerate weight vector that exercises the exp(-Δ/2)
# shape, not a near-{1,0} vector. The `subj`/`days`/`reaction` ↔ `Subject`/`Days`/`Reaction`
# rename is the only cross-package difference; the `sleepstudy` data is identical.
forms <- list(
  corr   = Reaction ~ 1 + Days + (1 + Days | Subject),
  uncorr = Reaction ~ 1 + Days + (1 + Days || Subject),
  int    = Reaction ~ 1 + Days + (1 | Subject)
)

models <- lapply(forms, function(f) lmer(f, data = sleepstudy, REML = FALSE))

# full-precision per-candidate cAIC (not the rounded anocAIC column)
caics_full <- vapply(models, function(m) as.numeric(cAIC(m)$caic), numeric(1))

# the Buckland smoothed-weights path
res <- modelAvg(models, opt = FALSE)
weights <- as.numeric(res$optimresults$weights)
fixeff_names <- names(res$fixeff)
fixeff_vals <- as.numeric(res$fixeff)

if (file.exists(fixture)) file.remove(fixture)
dir.create(dirname(fixture), showWarnings = FALSE, recursive = TRUE)
h5createFile(fixture)
on.exit(h5closeAll())

h5write(as.numeric(caics_full), fixture, "caic")          # input order
h5write(as.numeric(weights), fixture, "weights")          # input order
h5write(fixeff_names, fixture, "fixeff_names")
h5write(as.numeric(fixeff_vals), fixture, "fixeff_vals")
h5write(names(forms), fixture, "candidate_names")

h5createGroup(fixture, "meta")
h5write(caic4_version, fixture, "meta/cAIC4_version")
h5write(as.character(packageVersion("lme4")), fixture, "meta/lme4_version")
h5write(R.version.string, fixture, "meta/R_version")

cat("Candidate cAICs (full precision):\n")
print(setNames(caics_full, names(forms)))
cat("modelAvg(opt = FALSE) weights:\n")
print(setNames(weights, names(forms)))
cat("Averaged fixed effects:\n")
print(setNames(fixeff_vals, fixeff_names))
cat(sprintf(
  "Wrote %d candidate(s) to %s (cAIC4 %s, lme4 %s).\n",
  length(models), fixture, caic4_version, as.character(packageVersion("lme4"))
))
