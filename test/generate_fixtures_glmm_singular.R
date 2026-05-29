#!/usr/bin/env Rscript
#
# Partial-singularity GLMM Level-2 fixture — R (ground-truth) side, issue #32 / M3.
#
# Reads the canonical seed-35 Bernoulli sample embedded by the Julia side
# (`generate_fixtures_glmm_singular.jl`) from the HDF5 fixture, fits the equivalent
# `glmer(y ~ 1 + x + (1 + x || g))` (double-bar = uncorrelated intercept+slope, the lme4
# analogue of MixedModels' `zerocorr`), and drives `cAIC4`'s *public* `cAIC()`. On this sample
# the random slope variance is estimated at zero (partial singularity): `deleteZeroComponents`
# drops the slope direction, refits `(1 | g)`, and `cAIC` scores the reduced model
# (`new = TRUE`). The ground-truth caic / df / conditional log-likelihood / refit flag are
# appended to the same fixture group.
#
# The Julia Level-2 test scores the *identical* embedded sample with `cAIC.caic` (which
# cascades the same boundary reduction) and compares within atol = 1e-3 (DECISIONS.md).
#
# `cAIC4` is sourced directly from the committed source tree. HDF5: `rhdf5`.
#
# Run `generate_fixtures_glmm_singular.jl` FIRST (it writes the sample + skeleton).
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src)
#   FIXTURE    path to the HDF5 fixture
#              (default <script dir>/fixtures/caic_glmm_singular_level2.h5)
#
# Usage:  Rscript test/generate_fixtures_glmm_singular.R

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
  fixture <- file.path(here, "fixtures", "caic_glmm_singular_level2.h5")
}
stopifnot(
  "fixture not found — run generate_fixtures_glmm_singular.jl first" = file.exists(fixture)
)

caic4_files <- c(
  "getcondLL.R", "biasCorrectionBernoulli.R", "getModelComponents.R",
  "deleteZeroComponents.R", "cnms2formula.R", "helperfuns_lme.R", "bcMer.R", "cAIC.R"
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

# ── Read the embedded seed-35 sample ──────────────────────────────────────────────────────
grp <- "partial_bernoulli"
y <- as.numeric(h5read(fixture, paste0(grp, "/y")))
x <- as.numeric(h5read(fixture, paste0(grp, "/x")))
g <- as.integer(h5read(fixture, paste0(grp, "/g")))
dat <- data.frame(y = y, x = x, g = factor(g))

# ── Fit with lme4; confirm the partial boundary, then score via the public cAIC() ──────────
fit <- suppressWarnings(glmer(
  y ~ 1 + x + (1 + x || g),
  data = dat, family = binomial,
  control = glmerControl(optimizer = "bobyqa")
))
stopifnot("seed-35: lme4 did not land on the boundary" = isSingular(fit))
theta <- getME(fit, "theta")
stopifnot(
  "seed-35: expected *partial* singularity (intercept survives, slope on boundary)" =
    any(theta != 0) && any(theta == 0)
)

r <- suppressWarnings(cAIC(fit))
stopifnot("seed-35: expected a boundary refit (new = TRUE)" = isTRUE(r$new))

# ── Append the reference quantities to the same group ──────────────────────────────────────
put <- function(path, value) {
  try(h5delete(fixture, path), silent = TRUE)
  h5write(value, fixture, path)
}
put(paste0(grp, "/caic"), as.numeric(r$caic))
put(paste0(grp, "/df"), as.numeric(r$df)) # ρ
put(paste0(grp, "/cll"), as.numeric(r$loglikelihood))
put(paste0(grp, "/new"), as.integer(isTRUE(r$new))) # boundary-refit flag

put("meta/caic4_version", caic4_version)
put("meta/lme4_version", as.character(packageVersion("lme4")))
put("meta/rhdf5_version", as.character(packageVersion("rhdf5")))
put("meta/r_version", R.version.string)

h5closeAll()
cat(sprintf(
  "Wrote partial-singularity Bernoulli reference: caic=%.10f df=%.10f cll=%.10f new=%s (cAIC4 %s, lme4 %s).\n",
  r$caic, r$df, r$loglikelihood, isTRUE(r$new), caic4_version,
  as.character(packageVersion("lme4"))
))
