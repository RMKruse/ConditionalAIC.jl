#!/usr/bin/env Rscript
#
# Level-1 fixture generator (R / ground-truth side) — issue #7 / ADR-0003.
#
# Reads the HDF5 component fixture written by `generate_fixtures.jl`, evaluates
# `cAIC4`'s `calculateGaussianBc(model, sigma.penalty, analytic = TRUE)` on each case,
# and writes the reference effective degrees of freedom ρ back into the same file
# (`<case>/rho_ref`). This is the cAIC4 ground truth the Level-1 test compares against.
#
# `calculateGaussianBc` is pure base R, so we `source()` the committed cAIC4 v1.1 source
# directly rather than installing the full `cAIC4` + `lme4` stack — the Level-1 boundary
# (ADR-0003) is fit-independent, so no model fit and no `getModelComponents` is involved.
# The gated *live-RCall* CI job is what re-validates against the installed `cAIC4` package.
#
# HDF5 I/O goes through `test/fixture_io.R`, which selects the R HDF5 package per platform
# (`hdf5r` on the Linux CI runner, `rhdf5` on macOS-ARM) behind a single function surface.
# The HDF5 file format and the whole pipeline are unchanged — only the backing R package
# differs by platform. See the ADR-0003 addenda (2026-05-27, 2026-06-02).
#
# The computation-bearing matrices A, V0inv, Wⱼ are symmetric, so the Julia↔R
# column/row-major round-trip leaves them unchanged.
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src)
#   FIXTURE    path to the HDF5 fixture (default <script dir>/fixtures/dof_lmm_level1.h5)
#
# Usage:  Rscript test/generate_fixtures.R

suppressMessages(source(file.path(dirname(normalizePath(sub("^--file=","",commandArgs(FALSE)[grep("^--file=",commandArgs(FALSE))]))),"fixture_io.R")))

caic4_src <- Sys.getenv("CAIC4_SRC", "/private/tmp/cAIC4_src")
fixture <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture <- file.path(here, "fixtures", "dof_lmm_level1.h5")
}

bc_src <- file.path(caic4_src, "R", "calculateGaussianBc.R")
stopifnot("cAIC4 source not found; set CAIC4_SRC" = file.exists(bc_src))
source(bc_src)
caic4_version <- tryCatch(
  unname(read.dcf(file.path(caic4_src, "DESCRIPTION"))[, "Version"]),
  error = function(e) "unknown"
)

stopifnot("fixture not found; run generate_fixtures.jl first" = file.exists(fixture))
on.exit(h5closeAll())

put <- function(path, value) { # idempotent dataset write
  try(h5delete(fixture, path), silent = TRUE)
  h5write(value, fixture, path)
}

top <- h5ls(fixture, recursive = FALSE)
case_names <- setdiff(top$name[top$otype == "H5I_GROUP"], "meta")

for (name in case_names) {
  d <- h5read(fixture, name)
  n <- as.integer(d$n)
  p <- as.integer(d$p)
  s <- as.integer(d$s)

  model <- list(
    X = matrix(0, n, p), # only ncol(X) = p is read by calculateGaussianBc
    n = n,
    theta = numeric(s), # only length(theta) = s is read
    e = as.numeric(d$e),
    A = d$A,
    V0inv = d$V0inv,
    R = NULL, # unweighted: RA = A
    Wlist = lapply(seq_len(s), function(j) d$Wlist[[paste0("W", j)]]),
    eWelist = as.list(as.numeric(d$eWelist)),
    tye = as.numeric(d$tye),
    B = matrix(0, s, s),
    C = matrix(0, s, n),
    isREML = as.logical(d$isREML)
  )

  rho <- calculateGaussianBc(model, sigma.penalty = as.integer(d$sigma_penalty), analytic = TRUE)
  put(paste0(name, "/rho_ref"), as.numeric(rho))

  # analytic = FALSE: external B (here the synthetic SPD fixture B), rescaled C, shared
  # assembly. This is the cAIC4 ground truth for cAIC.jl's :forwarddiff / :finitediff
  # B-sources, which supply B numerically rather than from the closed form.
  model_numeric <- model
  model_numeric$B <- matrix(as.numeric(d$B), s, s)
  rho_numeric <- calculateGaussianBc(
    model_numeric, sigma.penalty = as.integer(d$sigma_penalty), analytic = FALSE
  )
  put(paste0(name, "/rho_ref_numeric"), as.numeric(rho_numeric))

  cat(sprintf(
    "  %-22s  rho = %.12g  rho_numeric = %.12g  (n=%d, s=%d, REML=%s)\n",
    name, rho, rho_numeric, n, s, model$isREML
  ))
}

put("meta/cAIC4_version", caic4_version)
put("meta/hdf5_backend", fixture_hdf5_backend())
put("meta/R_version", R.version.string)

cat(sprintf(
  "Wrote reference rho for %d case(s) to %s (cAIC4 %s).\n",
  length(case_names), fixture, caic4_version
))
