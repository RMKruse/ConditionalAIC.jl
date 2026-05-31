#!/usr/bin/env Rscript
#
# Level-1 fixture generator (R / ground-truth side) for the Zhang-optimal weight
# optimizer — issue #50 / M4.5.
#
# Level-1 isolation (CLAUDE §6 / ADR-0003): feeds SYNTHETIC (y, mu, rho, sigma_sq) that
# were chosen on the Julia side directly into the getWeights/.weightOptim arithmetic,
# bypassing lme4 model fitting. Both sides start from IDENTICAL inputs, so any deviation
# flags a transcription bug rather than a fit-discrepancy.
#
# Two test cases, each stored in a separate HDF5 group:
#   case1  M=3, n=30, well-conditioned MᵀM (unique minimiser)
#   case2  M=2, n=20, well-conditioned MᵀM
#
# Inputs (identical to what the Julia test uses — fixed seed → reproducible) are stored
# alongside the R outputs so the Julia test only needs the fixture (no R at runtime).
#
# Env vars:
#   CAIC4_SRC   path to the cAIC4 source tree (default /private/tmp/cAIC4_src)
#   FIXTURE     path to the HDF5 output file (default <script dir>/fixtures/zhang_weights_level1.h5)
#
# Usage:  Rscript test/generate_fixtures_zhang_level1.R

suppressMessages(library(rhdf5))

caic4_src <- Sys.getenv("CAIC4_SRC", "/private/tmp/cAIC4_src")
fixture <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture <- file.path(here, "fixtures", "zhang_weights_level1.h5")
}

# Source the inner optimizer from cAIC4 (only .weightOptim is needed — the outer loop is
# reproduced inline below using synthetic inputs, bypassing getWeights' anocAIC call).
wo_path <- file.path(caic4_src, "R", "weightOptim.R")
stopifnot("cAIC4 weightOptim.R not found; set CAIC4_SRC" = file.exists(wo_path))
source(wo_path)

caic4_version <- tryCatch(
  unname(read.dcf(file.path(caic4_src, "DESCRIPTION"))[, "Version"]),
  error = function(e) "unknown"
)

# ── Synthetic-input optimizer (getWeights body, model-fitting bypassed) ───────────────
# Reproduces getWeights.R lines 62-122 with synthetic (y, mu_mat, df_vec, varDF_val).
run_getweights_synthetic <- function(y, mu_mat, df_vec, varDF_val) {
  nw     <- length(df_vec)
  equB   <- 1
  lowb   <- rep(0, nw)
  uppb   <- rep(1, nw)
  m      <- vector("list", nw)   # only length(m) is used inside .weightOptim
  mu     <- mu_mat                # must be named "mu" for .envi lookup
  varDF  <- varDF_val             # must be named "varDF" for .envi lookup
  df     <- df_vec                # must be named "df" for .envi lookup (the ρ vector)

  find_weights <- function(w) {
    as.numeric(t(y - mu %*% w) %*% (y - mu %*% w) + 2 * varDF * (w %*% df))
  }
  .envi  <- environment()

  weights <- rep(1 / nw, nw)
  funv    <- find_weights(weights)
  eqv     <- sum(weights) - equB
  rho     <- 0       # augmented-Lagrangian rho — always 0 in this implementation
  maxit   <- 400
  tol     <- 1e-8
  j       <- funv
  lambda  <- c(0)     # Lagrange multiplier (1-vector)
  p       <- c(weights)
  hess    <- diag(nw)
  mue     <- nw       # penalty parameter
  .iters  <- 0
  targets <- c(funv, eqv)

  tic <- proc.time()["elapsed"]
  while (.iters < maxit) {
    .iters <- .iters + 1
    scaler <- c(targets[1], rep(1, 1) * max(abs(targets[2:(1 + 1)])))
    scaler <- c(scaler, rep(1, length.out = length(p)))
    scaler <- apply(matrix(scaler, ncol = 1), 1,
                    FUN = function(x) min(max(abs(x), tol), 1 / tol))
    res    <- .weightOptim(weights = p, lm = lambda, targets = targets,
                           hess = hess, lambda = mue, scaler = scaler, .envi = .envi)
    p      <- res$p
    lambda <- res$y
    hess   <- res$hess
    mue    <- res$lambda
    funv   <- find_weights(p)
    eqv    <- sum(p) - equB
    targets <- c(funv, eqv)
    tt     <- (j - targets[1]) / max(targets[1], 1)
    j      <- targets[1]
    if (abs(targets[2]) < 10 * tol) {
      rho <- 0
      mue <- min(mue, tol)
    }
    if ((tol + tt) <= 0) {
      lambda <- 0
      hess   <- diag(diag(hess))
    }
    if (sqrt(sum((c(tt, eqv))^2)) <= tol) {
      maxit <- .iters
    }
  }
  toc <- proc.time()["elapsed"] - tic
  list(weights = p, objective = j, duration = toc)
}

# ── Synthetic inputs (fixed seed — must match the constants in the Julia test) ─────────
# Case 1: M=3, n=30  ─────────────────────────────────────────────────────────────────
set.seed(42)
n1 <- 30L; M1 <- 3L
y1 <- rnorm(n1)
# mu columns intentionally not too correlated: three shifted sinusoids, then orthogonalised
t_vals <- seq(0, 2 * pi, length.out = n1)
mu_raw1 <- cbind(sin(t_vals), cos(t_vals), sin(2 * t_vals))
mu_raw1 <- mu_raw1 + 0.3 * matrix(rnorm(n1 * M1), n1, M1)
# Gram-Schmidt for conditioning guarantee
mu1 <- qr.Q(qr(mu_raw1)) * sqrt(n1)   # orthonormal columns, scaled to unit RMS ~1
rho1 <- c(3.5, 4.2, 2.8)
sigma_sq1 <- 1.5

# Case 2: M=2, n=20  ─────────────────────────────────────────────────────────────────
set.seed(123)
n2 <- 20L; M2 <- 2L
y2 <- rnorm(n2)
mu_raw2 <- cbind(1:n2 / n2, sin(seq(0, pi, length.out = n2)))
mu_raw2 <- mu_raw2 + 0.2 * matrix(rnorm(n2 * M2), n2, M2)
mu2 <- qr.Q(qr(mu_raw2)) * sqrt(n2)
rho2 <- c(2.1, 3.7)
sigma_sq2 <- 2.0

# ── Run optimizer ───────────────────────────────────────────────────────────────────
cat("Running case 1 (M=3, n=30)...\n")
res1 <- run_getweights_synthetic(y1, mu1, rho1, sigma_sq1)
cat(sprintf("  weights:   %s\n", paste(round(res1$weights, 8), collapse = ", ")))
cat(sprintf("  objective: %.10f\n", res1$objective))

cat("Running case 2 (M=2, n=20)...\n")
res2 <- run_getweights_synthetic(y2, mu2, rho2, sigma_sq2)
cat(sprintf("  weights:   %s\n", paste(round(res2$weights, 8), collapse = ", ")))
cat(sprintf("  objective: %.10f\n", res2$objective))

# ── Write fixture ──────────────────────────────────────────────────────────────────
if (file.exists(fixture)) file.remove(fixture)
dir.create(dirname(fixture), showWarnings = FALSE, recursive = TRUE)
h5createFile(fixture)
on.exit(h5closeAll())

for (case_id in c("case1", "case2")) {
  h5createGroup(fixture, case_id)
  h5createGroup(fixture, file.path(case_id, "inputs"))
  h5createGroup(fixture, file.path(case_id, "outputs_r"))
}

# Case 1
h5write(as.numeric(y1),         fixture, "case1/inputs/y")
h5write(mu1,                    fixture, "case1/inputs/mu")      # n×M, column-major
h5write(as.numeric(rho1),       fixture, "case1/inputs/rho")
h5write(sigma_sq1,              fixture, "case1/inputs/sigma_sq")
h5write(as.numeric(res1$weights),   fixture, "case1/outputs_r/weights")
h5write(as.numeric(res1$objective), fixture, "case1/outputs_r/objective")

# Case 2
h5write(as.numeric(y2),         fixture, "case2/inputs/y")
h5write(mu2,                    fixture, "case2/inputs/mu")
h5write(as.numeric(rho2),       fixture, "case2/inputs/rho")
h5write(sigma_sq2,              fixture, "case2/inputs/sigma_sq")
h5write(as.numeric(res2$weights),   fixture, "case2/outputs_r/weights")
h5write(as.numeric(res2$objective), fixture, "case2/outputs_r/objective")

# Meta
h5createGroup(fixture, "meta")
h5write(caic4_version,      fixture, "meta/cAIC4_version")
h5write(R.version.string,   fixture, "meta/R_version")

cat(sprintf("Wrote Level-1 Zhang fixture to %s (cAIC4 %s).\n", fixture, caic4_version))
