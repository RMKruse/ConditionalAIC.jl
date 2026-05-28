#!/usr/bin/env Rscript
#
# Level-1 fixture generator (R / ground-truth side) — bootstrap path.
#
# Reads the HDF5 fixture written by `generate_fixtures_bootstrap.jl`, applies
# `cAIC4`'s `conditionalBootstrap` bias-correction arithmetic to each case, and writes
# the reference `rho_ref` back into the same file. This is the cAIC4 ground truth the
# Level-1 test compares against (rtol = 1e-6, atol = 1e-10) — the shared-input fixture
# the bootstrap path was missing.
#
# Why we isolate the arithmetic from the function. `conditionalBootstrap(object, B)`
# is a closed pipeline: it `simulate()`s, `refit()`s, and only then applies the
# bias-correction formula on lines 23–25 of `R/conditionalBootstrap.R`. The Level-1
# unit needs *just* that arithmetic, on fixed `Y*` / `Ŷ*` matrices, so we hard-code
# the formula here and `source()` the cAIC4 file purely to **pin the formula** via a
# textual self-check (`grepl` on the function body): if a `cAIC4` bump silently moves
# either the row-mean centring or the `(B−1)` divisor, fixture generation stops loud.
# The committed cAIC4 v1.1 lines this matches are reproduced verbatim in the comment
# above `bootBC_arithmetic` below.
#
# HDF5 reader: `rhdf5` — same as `generate_fixtures.R`; the ADR-0003 addendum
# (2026-05-27) covers the choice.
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src)
#   FIXTURE    path to the HDF5 fixture (default <script dir>/fixtures/bootstrap_level1.h5)
#
# Usage:  Rscript test/generate_fixtures_bootstrap.R

suppressMessages(library(rhdf5))

caic4_src <- Sys.getenv("CAIC4_SRC", "/private/tmp/cAIC4_src")
fixture <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture <- file.path(here, "fixtures", "bootstrap_level1.h5")
}

cb_src <- file.path(caic4_src, "R", "conditionalBootstrap.R")
stopifnot("cAIC4 source not found; set CAIC4_SRC" = file.exists(cb_src))
source(cb_src)
caic4_version <- tryCatch(
  unname(read.dcf(file.path(caic4_src, "DESCRIPTION"))[, "Version"]),
  error = function(e) "unknown"
)

# `cAIC4` v1.1 `R/conditionalBootstrap.R` lines 23–25 (bias-correction arithmetic):
#   dataMatrix <- dataMatrix - rowMeans(dataMatrix)
#   bootBC     <- sum(workingEta * dataMatrix) /
#                 ((BootStrRep - 1) * sigma(object)^2)
# Isolated below as a pure function of `(Ystar, Yhatstar, sigma)`; B = ncol(Ystar).
bootBC_arithmetic <- function(Ystar, Yhatstar, sigma) {
  centered <- Ystar - rowMeans(Ystar)
  B <- ncol(Ystar)
  sum(Yhatstar * centered) / ((B - 1) * sigma^2)
}

# Pin the cAIC4 formula textually: any silent drift in either the centring or the
# divisor must fail fixture regeneration loudly (CLAUDE §10 — don't paper over). The
# deparsed body is whitespace-normalised before matching, because R's `deparse` may
# insert a line break inside `(BootStrRep - 1) * sigma(object)^2`.
body_str <- gsub("\\s+", "", paste(deparse(body(conditionalBootstrap)), collapse = ""))
stopifnot(
  "cAIC4 conditionalBootstrap row-mean centring drifted" =
    grepl("dataMatrix-rowMeans(dataMatrix)", body_str, fixed = TRUE),
  "cAIC4 conditionalBootstrap (B-1)*sigma^2 divisor drifted" =
    grepl("sum(workingEta*dataMatrix)/((BootStrRep-1)*sigma(object)^2)",
          body_str, fixed = TRUE)
)

stopifnot("fixture not found; run generate_fixtures_bootstrap.jl first" =
            file.exists(fixture))
on.exit(h5closeAll())

put <- function(path, value) { # idempotent dataset write
  try(h5delete(fixture, path), silent = TRUE)
  h5write(value, fixture, path)
}

top <- h5ls(fixture, recursive = FALSE)
case_names <- setdiff(top$name[top$otype == "H5I_GROUP"], "meta")

for (name in case_names) {
  d <- h5read(fixture, name)
  Ystar <- d$Ystar
  Yhatstar <- d$Yhatstar
  sigma <- as.numeric(d$sigma)
  rho <- bootBC_arithmetic(Ystar, Yhatstar, sigma)
  put(paste0(name, "/rho_ref"), as.numeric(rho))
  cat(sprintf("  %-10s rho_ref = %.12g  (n=%d, B=%d)\n",
              name, rho, nrow(Ystar), ncol(Ystar)))
}

put("meta/cAIC4_version", caic4_version)
put("meta/rhdf5_version", as.character(packageVersion("rhdf5")))
put("meta/R_version", R.version.string)

cat(sprintf("Wrote rho_ref for %d case(s) to %s (cAIC4 %s).\n",
            length(case_names), fixture, caic4_version))
