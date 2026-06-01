#!/usr/bin/env Rscript
#
# Level-2 fixture generator (R / ground-truth side) for model-averaged prediction —
# issue #52 / M4.5 (cAIC4's `predictMA`).
#
# Fits two candidate sets in `lme4` and runs `cAIC4`'s `modelAvg(models, opt = TRUE)`
# (Zhang-optimal) followed by `predictMA(res, new.data = sleepstudy)`, writing the
# model-averaged prediction vector ŷ^MA (`MApredict`), the per-candidate Zhang weights,
# and the per-candidate full-precision conditional AICs for each scenario.
#
# The Julia Level-2 test fits the same candidates in `MixedModels.jl`, runs `modelavg(...)`
# (default :zhang) + `predictma(res, data)`, and compares the PREDICTION within the band
# recorded in DECISIONS.md. The prediction is the §7 "stable functional" — it is anchored on
# BOTH scenarios (incl. the nested set, where the weight vector itself is not anchored because
# the QP minimiser need not be unique), the M4.5 analogue of stepcaic's path-only-on-well-
# separated-cases rule.
#
# Scenarios (input order, mirrored exactly in test/averaging_tests.jl):
#   wc      — WELL-CONDITIONED, 2 candidates with DIFFERENT fixed-effects structure so the
#             conditional means μ̂ = Xβ̂ + Zb̂ are genuinely distinct (MᵀM ≻ 0, unique minimiser):
#               m1 : Reaction ~ 1 + Days + (1 + Days | Subject)
#               m2 : Reaction ~ 1 + (1 | Subject)
#   nested  — NESTED, 3 candidates of decreasing structure (the Orthodont-style set of the
#             cAIC4 predictMA example, transposed to sleepstudy):
#               m1 : Reaction ~ 1 + Days + (1 + Days | Subject)
#               m2 : Reaction ~ 1 + Days + (1 | Subject)
#               m3 : Reaction ~ 1 + (1 | Subject)
#
# Prediction is on the TRAINING data (sleepstudy) — every grouping level is seen, so the
# conditional predict path (re.form = NULL, allow.new.levels = FALSE) is exercised end-to-end.
#
# Fields written (HDF5), per scenario group "wc"/"nested" (all in candidate INPUT ORDER):
#   <s>/caic        — full-precision cAIC4 cAIC per candidate
#   <s>/weights     — modelAvg(opt = TRUE) Zhang-optimal weights
#   <s>/prediction  — ŷ^MA = predictMA(...)$prediction, length n = 180
#   meta/*          — cAIC4 / lme4 / R version strings
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src)
#   FIXTURE    path to the HDF5 output file  (default <script dir>/fixtures/predictma_level2.h5)
#
# Usage:  Rscript test/generate_fixtures_predictma.R

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
  fixture <- file.path(here, "fixtures", "predictma_level2.h5")
}

# Source the transitive closure of cAIC4 functions for the Gaussian, non-boundary, opt = TRUE
# path plus predictMA(): cAIC() + getWeights() + .weightOptim() + modelAvg() + predictMA().
caic4_files <- c(
  "getcondLL.R", "calculateGaussianBc.R", "getModelComponents.R",
  "deleteZeroComponents.R", "cnms2formula.R", "helperfuns_lme.R",
  "biasCorrectionGaussian.R", "bcMer.R", "cAIC.R", "methods.R",
  "weightOptim.R", "getWeights.R", "modelAvg.R", "predictMA.R"
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

# ── Scenario definitions (input order mirrored in the Julia test) ──────────────────────
scenarios <- list(
  wc = list(
    Reaction ~ 1 + Days + (1 + Days | Subject),
    Reaction ~ 1 + (1 | Subject)
  ),
  nested = list(
    Reaction ~ 1 + Days + (1 + Days | Subject),
    Reaction ~ 1 + Days + (1 | Subject),
    Reaction ~ 1 + (1 | Subject)
  )
)

if (file.exists(fixture)) file.remove(fixture)
dir.create(dirname(fixture), showWarnings = FALSE, recursive = TRUE)
h5createFile(fixture)
on.exit(h5closeAll())

for (sname in names(scenarios)) {
  forms <- scenarios[[sname]]
  models <- lapply(forms, function(f) lmer(f, data = sleepstudy, REML = FALSE))

  # Full-precision cAIC per candidate (not the rounded anocAIC column)
  caics_full <- vapply(models, function(m) as.numeric(cAIC(m)$caic), numeric(1))

  # Zhang-optimal model averaging + model-averaged prediction on the training data
  res <- modelAvg(models, opt = TRUE)
  weights <- as.numeric(res$optimresults$weights)
  pma <- predictMA(res, new.data = sleepstudy)
  prediction <- as.numeric(pma$prediction)   # MApredict <- w %*% t(sapply(...))

  h5createGroup(fixture, sname)
  h5write(caics_full, fixture, paste0(sname, "/caic"))
  h5write(weights, fixture, paste0(sname, "/weights"))
  h5write(prediction, fixture, paste0(sname, "/prediction"))

  cat(sprintf("[%s] candidate cAICs (full precision, input order):\n", sname))
  print(caics_full)
  cat(sprintf("[%s] Zhang weights (input order):\n", sname))
  print(weights)
  cat(sprintf("[%s] prediction length = %d, range [%.4f, %.4f]\n",
              sname, length(prediction), min(prediction), max(prediction)))
}

h5createGroup(fixture, "meta")
h5write(caic4_version, fixture, "meta/cAIC4_version")
h5write(as.character(packageVersion("lme4")), fixture, "meta/lme4_version")
h5write(R.version.string, fixture, "meta/R_version")

cat(sprintf(
  "Wrote predictMA Level-2 fixture to %s (cAIC4 %s, lme4 %s).\n",
  fixture, caic4_version, as.character(packageVersion("lme4"))
))
