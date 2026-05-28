#!/usr/bin/env Rscript
#
# Singular-fit Level-2 fixture generator (R / ground-truth side) — issue #10 / CLAUDE §6.
#
# Drives `cAIC4`'s *public* `cAIC()` path on boundary (singular) fits and writes the
# ground-truth conditional AIC, effective degrees of freedom ρ, conditional log-likelihood,
# and the boundary-refit flag (`new`) to an HDF5 fixture. The Julia Level-2 test
# (`singular_tests.jl`) fits the *same* data in `MixedModels.jl`, runs `cAIC.caic` — which
# detects the boundary, drops the offending components, and refits the reduced model — and
# compares against these references within the fit-discrepancy-derived tolerance recorded in
# DECISIONS.md.
#
# Two boundary regimes are covered, one per `cAIC4` code path in `biasCorrectionGaussian`:
#
#   reduce_ml      A correlated (1 + x | g) fit whose random *slope* is unidentifiable — the
#                  predictor x is constant within each group, so the slope direction carries
#                  no within-group information and collapses to the boundary in lme4 *and*
#                  MixedModels alike. `deleteZeroComponents` drops it to (1 | g) and refits
#                  (`new = TRUE`). The synthetic data are embedded in the fixture so both
#                  ecosystems score the *identical* sample (R and Julia RNGs never meet).
#
#   dyestuff2_*    The classic `Dyestuff2` fit `Yield ~ 1 + (1 | Batch)`: the batch variance
#                  is estimated at zero, so *every* random-effect component is on the
#                  boundary and no random-effects model remains. `cAIC4` falls back to the
#                  fixed-effects-only (`lm`) score: df = rank + sigma.penalty, `new = FALSE`,
#                  `reducedModel = NULL`. `Dyestuff2` is a shared canonical dataset (identical
#                  in lme4 and MixedModels), so — as with sleepstudy in the non-singular
#                  generator — it is referenced by name rather than embedded.
#
# The REML analogue of `reduce_ml` is *omitted by design*: lme4's REML optimiser settles at a
# small but non-zero slope variance (not flagged singular) where MixedModels' lands exactly on
# the boundary, so the two ecosystems disagree on *whether* the fit is singular. That
# divergence is recorded in DECISIONS.md, not papered over with a fixture.
#
# `cAIC4` is sourced directly from the committed source tree (as the other generators do).
# HDF5 writer: `rhdf5`. Env vars CAIC4_SRC / FIXTURE as in generate_fixtures_level2.R.
#
# Usage:  Rscript test/generate_fixtures_singular.R

suppressMessages({
  library(rhdf5)
  library(lme4)
})

caic4_src <- Sys.getenv("CAIC4_SRC", "/private/tmp/cAIC4_src")
fixture <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture <- file.path(here, "fixtures", "caic_singular_level2.h5")
}

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

# Helper: write the cAIC4 reference quantities for one case into its group.
write_ref <- function(fixture, name, r, reml) {
  h5createGroup(fixture, name)
  h5write(as.numeric(r$caic), fixture, paste0(name, "/caic"))
  h5write(as.numeric(r$df), fixture, paste0(name, "/df")) # ρ
  h5write(as.numeric(r$loglikelihood), fixture, paste0(name, "/cll"))
  h5write(as.integer(reml), fixture, paste0(name, "/reml"))
  h5write(as.integer(isTRUE(r$new)), fixture, paste0(name, "/new")) # boundary-refit flag
}

if (file.exists(fixture)) file.remove(fixture)
dir.create(dirname(fixture), showWarnings = FALSE, recursive = TRUE)
h5createFile(fixture)
on.exit(h5closeAll())

# ── reduce_ml — partial drop (1 + x | g) → (1 | g), refit (cAIC4 new = TRUE) ──────────────
set.seed(42)
ng <- 15
npg <- 6
n <- ng * npg
g <- factor(rep(seq_len(ng), each = npg))
xg <- rnorm(ng) # one x per group …
x <- rep(xg, each = npg) # … held constant within the group → slope unidentifiable
b0 <- rep(rnorm(ng), each = npg) * 2.0 # intercept varies; slope does not
y <- 3.0 + 1.5 * x + b0 + rnorm(n) * 0.5

fit <- suppressWarnings(lmer(y ~ 1 + x + (1 + x | g), REML = FALSE))
stopifnot("reduce_ml: lme4 did not land on the boundary" = isSingular(fit))
r <- suppressWarnings(cAIC(fit, sigma.penalty = 1L))
stopifnot("reduce_ml: expected a boundary refit (new = TRUE)" = isTRUE(r$new))
write_ref(fixture, "reduce_ml", r, reml = FALSE)
h5write(as.numeric(y), fixture, "reduce_ml/y") # embed the shared sample
h5write(as.numeric(x), fixture, "reduce_ml/x")
h5write(as.integer(g), fixture, "reduce_ml/g")
cat(sprintf(
  "  %-14s caic=%.10f  df=%.10f  cll=%.10f  new=%s\n",
  "reduce_ml", r$caic, r$df, r$loglikelihood, r$new
))

# ── dyestuff2_{ml,reml} — all components on the boundary → lm branch (cAIC4 new = FALSE) ──
for (reml in c(FALSE, TRUE)) {
  name <- if (reml) "dyestuff2_reml" else "dyestuff2_ml"
  fit <- suppressWarnings(lmer(Yield ~ 1 + (1 | Batch), data = Dyestuff2, REML = reml))
  stopifnot("dyestuff2: lme4 did not land on the boundary" = isSingular(fit))
  r <- suppressWarnings(cAIC(fit, sigma.penalty = 1L))
  stopifnot("dyestuff2: expected the lm fallback (new = FALSE)" = isFALSE(r$new))
  write_ref(fixture, name, r, reml = reml)
  cat(sprintf(
    "  %-14s caic=%.10f  df=%.10f  cll=%.10f  new=%s\n",
    name, r$caic, r$df, r$loglikelihood, r$new
  ))
}

h5createGroup(fixture, "meta")
h5write(caic4_version, fixture, "meta/cAIC4_version")
h5write(as.character(packageVersion("lme4")), fixture, "meta/lme4_version")
h5write(as.character(packageVersion("rhdf5")), fixture, "meta/rhdf5_version")
h5write(R.version.string, fixture, "meta/R_version")

cat(sprintf(
  "Wrote singular Level-2 reference cases to %s (cAIC4 %s, lme4 %s).\n",
  fixture, caic4_version, as.character(packageVersion("lme4"))
))
