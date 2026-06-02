#!/usr/bin/env Rscript
#
# Level-2 fixture generator (R / ground-truth side) for Zhang-optimal model-averaging weights
# — issue #51 / M4.5.
#
# Fits a WELL-CONDITIONED candidate set in `lme4` and runs `cAIC4`'s
# `modelAvg(models, opt = TRUE)` (Zhang-optimal path), writing per-candidate weights,
# the minimised Mallows criterion J(ŵ), and the per-candidate full-precision conditional AICs
# to an HDF5 fixture.  The Julia Level-2 test fits the same candidates in `MixedModels.jl`,
# runs `modelavg(...)` (default :zhang), and compares within the band recorded in DECISIONS.md.
#
# Well-conditioned set (docs/math/0009 §7): the two candidates differ in FIXED-EFFECTS
# structure — one includes the `Days` fixed effect, the other does not — so their conditional
# mean vectors μ̂ = Xβ̂ + Zb̂ are genuinely different.  This guarantees MᵀM ≻ 0 and a unique
# QP minimiser, which is the required Level-2 anchor condition.
#
# Candidates (input order, mirrored exactly in test/averaging_tests.jl):
#   m1 : Reaction ~ 1 + Days + (1 + Days | Subject)   [full correlated slope]
#   m2 : Reaction ~ 1 + (1 | Subject)                 [intercept-only RE, no Days FE]
#
# Fields written to the HDF5 fixture (all in candidate INPUT ORDER, i.e. m1, m2):
#   caic       — full-precision cAIC4 cAIC per candidate
#   weights    — modelAvg(opt=TRUE) Zhang-optimal weights
#   objective  — J(ŵ), the minimised Mallows criterion (getWeights$functionvalue)
#   meta/*     — cAIC4/lme4/R version strings
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src)
#   FIXTURE    path to the HDF5 output file  (default <script dir>/fixtures/zhang_modelavg_level2.h5)
#
# Usage:  Rscript test/generate_fixtures_modelavg_zhang.R

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
  fixture <- file.path(here, "fixtures", "zhang_modelavg_level2.h5")
}

# Source the transitive closure of cAIC4 functions needed for the Gaussian, non-boundary,
# opt = TRUE path: cAIC() + getWeights() + .weightOptim() + modelAvg().
caic4_files <- c(
  "getcondLL.R", "calculateGaussianBc.R", "getModelComponents.R",
  "deleteZeroComponents.R", "cnms2formula.R", "helperfuns_lme.R",
  "biasCorrectionGaussian.R", "bcMer.R", "cAIC.R", "methods.R",
  "weightOptim.R", "getWeights.R", "modelAvg.R"
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

# ── Fit the two-candidate well-conditioned set ────────────────────────────────────────
# Input order MUST match the Julia test: full-slope first, intercept-only second.
# `days` / `subj` vs `Days` / `Subject`: sleepstudy uses the capitalised names in both R
# and Julia — MixedModels.dataset(:sleepstudy) uses lowercase column names, so the Julia
# test uses @formula(reaction ~ 1 + days + ...) while lme4 uses Days/Subject.  The fit
# discrepancy absorbed by the Level-2 atol band comes from lme4 ↔ MixedModels.jl, not
# from any name difference.
forms <- list(
  full  = Reaction ~ 1 + Days + (1 + Days | Subject),
  intonly = Reaction ~ 1 + (1 | Subject)
)
models <- lapply(forms, function(f) lmer(f, data = sleepstudy, REML = FALSE))

# Full-precision cAIC per candidate (not the rounded anocAIC column)
caics_full <- vapply(models, function(m) as.numeric(cAIC(m)$caic), numeric(1))

# Zhang-optimal weights + J(ŵ) via the full modelAvg(opt=TRUE) path
res <- modelAvg(models, opt = TRUE)
weights   <- as.numeric(res$optimresults$weights)
objective <- as.numeric(res$optimresults$functionvalue)

# ── Write fixture ─────────────────────────────────────────────────────────────────────
if (file.exists(fixture)) file.remove(fixture)
dir.create(dirname(fixture), showWarnings = FALSE, recursive = TRUE)
h5createFile(fixture)
on.exit(h5closeAll())

h5write(as.numeric(caics_full), fixture, "caic")       # input order: m1, m2
h5write(as.numeric(weights),    fixture, "weights")    # input order: m1, m2
h5write(objective,              fixture, "objective")  # scalar J(ŵ)

h5createGroup(fixture, "meta")
h5write(caic4_version,                           fixture, "meta/cAIC4_version")
h5write(as.character(packageVersion("lme4")),    fixture, "meta/lme4_version")
h5write(R.version.string,                        fixture, "meta/R_version")

cat("Candidate cAICs (full precision, input order):\n")
print(setNames(caics_full, names(forms)))
cat("modelAvg(opt = TRUE) Zhang weights (input order):\n")
print(setNames(weights, names(forms)))
cat(sprintf("J(ŵ) = %.10f\n", objective))
cat(sprintf(
  "Wrote Zhang Level-2 fixture to %s (cAIC4 %s, lme4 %s).\n",
  fixture, caic4_version, as.character(packageVersion("lme4"))
))
