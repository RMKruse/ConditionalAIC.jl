#!/usr/bin/env Rscript
#
# Level-2 fixture generator (R / ground-truth side) — Bernoulli GLMM df.
#
# Fits a Bernoulli GLMM (`glmer`, family=binomial) on a small synthetic dataset
# with a fixed seed, calls `cAIC4`'s `biasCorrectionBernoulli` on the fitted
# model, and writes the dataset + reference df to an HDF5 fixture.  The Julia
# test reads the same dataset, fits a MixedModels.jl GLMM, calls
# `DofGLMM.dof_glmm_bernoulli`, and compares at atol=1e-3 (the Level-2 tolerance
# from DECISIONS.md — fit discrepancies between lme4 and MixedModels.jl propagate
# into the df estimate).
#
# Ground-truth function: `cAIC4::biasCorrectionBernoulli` (v1.1).
#
# cAIC4 source: sourced directly from CAIC4_SRC (default /private/tmp/cAIC4_src).
# Required packages: lme4, rhdf5.
#
# The formula is pinned textually: if `biasCorrectionBernoulli` drifts from the
# expected source-code pattern, this script stops loud rather than silently writing
# a wrong fixture.
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src)
#   FIXTURE    path to the output HDF5 file
#              (default <script dir>/fixtures/dof_glmm_bernoulli_level2.h5)
#
# Usage:  Rscript test/generate_fixtures_bernoulli.R

suppressMessages(library(lme4))
suppressMessages(library(rhdf5))

caic4_src <- Sys.getenv("CAIC4_SRC", "/private/tmp/cAIC4_src")
fixture <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture <- file.path(here, "fixtures", "dof_glmm_bernoulli_level2.h5")
}

# Source required cAIC4 functions.
for (fname in c("deleteZeroComponents.R", "biasCorrectionBernoulli.R")) {
  src <- file.path(caic4_src, "R", fname)
  if (!file.exists(src)) stop("cAIC4 source not found (", fname, "); set CAIC4_SRC")
  source(src)
}
caic4_version <- tryCatch(
  unname(read.dcf(file.path(caic4_src, "DESCRIPTION"))[, "Version"]),
  error = function(e) "unknown"
)

# Pin the biasCorrectionBernoulli formula textually: the loop structure and the
# accumulation must match the source we read in docs/math/0006.
body_str <- gsub("\\s+", "", paste(deparse(body(biasCorrectionBernoulli)), collapse = ""))
stopifnot(
  "cAIC4 biasCorrectionBernoulli sign-correction formula drifted" =
    grepl("signCor<--2*zeroLessModel@resp$y+1", body_str, fixed = TRUE),
  "cAIC4 biasCorrectionBernoulli logit-diff formula drifted" =
    grepl("log(workingModel@resp$mu[i]/(1-workingModel@resp$mu[i]))-log(muHat[i]/(1-muHat[i]))",
          body_str, fixed = TRUE),
  "cAIC4 biasCorrectionBernoulli accumulation formula drifted" =
    grepl("bc<-sum(muHat*(1-muHat)*signCor*workingEta)", body_str, fixed = TRUE)
)

# ── Synthetic data (fixed seed = 42) ──────────────────────────────────────────
set.seed(42)
n_groups <- 10L
n_per    <- 12L
n        <- n_groups * n_per

group <- rep(seq_len(n_groups), each = n_per)
x     <- rnorm(n, sd = 0.8)
b_re  <- rnorm(n_groups, sd = 1.0)   # larger RE variance ensures non-singular fit
eta   <- 0.3 + 0.6 * x + b_re[group]
y     <- rbinom(n, 1L, plogis(eta))

dat <- data.frame(
  y     = as.numeric(y),
  x     = x,
  group = factor(group)
)

# ── Fit with lme4 ─────────────────────────────────────────────────────────────
m_r <- glmer(y ~ x + (1 | group), data = dat, family = binomial,
             control = glmerControl(optimizer = "bobyqa"))

# ── Reference df from cAIC4 ───────────────────────────────────────────────────
result <- biasCorrectionBernoulli(m_r)

# result is a list with $bc (the effective df), $newModel, $new
rho_ref <- as.numeric(result$bc)
cat(sprintf("rho_ref = %.10g\n", rho_ref))

# ── Write fixture ─────────────────────────────────────────────────────────────
on.exit(h5closeAll())

put <- function(path, value) {
  try(h5delete(fixture, path), silent = TRUE)
  h5write(value, fixture, path)
}

if (file.exists(fixture)) file.remove(fixture)
h5createFile(fixture)
h5createGroup(fixture, "meta")

put("y",       as.numeric(y))
put("x",       as.numeric(x))
put("group",   as.integer(group))
put("rho_ref", rho_ref)

put("meta/cAIC4_version", caic4_version)
put("meta/rhdf5_version", as.character(packageVersion("rhdf5")))
put("meta/R_version",     R.version.string)
put("meta/n",             as.integer(n))
put("meta/n_groups",      as.integer(n_groups))
put("meta/seed",          42L)

cat(sprintf(
  "Wrote fixture to %s (n=%d, %d groups, rho_ref=%.6g, cAIC4 %s).\n",
  fixture, n, n_groups, rho_ref, caic4_version
))
