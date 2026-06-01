#!/usr/bin/env Rscript
#
# Fixture generator for the Poisson GLMM Chen-Stein df — issue #28.
#
# Generates a synthetic Poisson dataset, fits a GLMM with lme4, instruments the
# Chen-Stein refit loop (biasCorrectionPoisson from cAIC4 v1.1), captures all
# intermediate values, and writes them to an HDF5 fixture.
#
# Fixture layout (one case group per dataset):
#   <case>/raw_data/y      — response counts (n-vector, Float64)
#   <case>/raw_data/x      — fixed-effects covariate (n-vector, Float64)
#   <case>/raw_data/group  — grouping variable (n-vector, Int32, 1-based)
#   <case>/y               — response y from the (zero-less) lme4 fit (m@resp$y)
#   <case>/eta0            — fitted η̂ from the lme4 fit (m@resp$eta, n-vector)
#   <case>/ind             — 1-based nonzero indices (Integer vector)
#   <case>/eta_dec         — per-nonzero refitted η̂_i^{(−i)} (length-|ind| vector)
#   <case>/rho_ref         — final Chen-Stein df ρ = Σᵢ yᵢ(η̂ᵢ − η̂_i^{(−i)})
#
# Level-1 test: reads y, eta0, ind, eta_dec, rho_ref; verifies arithmetic kernel
#   with rtol=1e-6, atol=1e-10 (ADR-0003).
# Level-2 test: reads raw_data, fits the same model with MixedModels.jl, calls
#   dof_glmm_poisson(m), compares to rho_ref with atol=0.5 (fit-discrepancy band;
#   to be tightened once the Level-2 tolerance is characterised in DECISIONS.md).
#
# Note: the intermediate capture below reimplements the biasCorrectionPoisson
# arithmetic verbatim from cAIC4 v1.1 R/biasCorrectionPoisson.R:13-24 with added
# instrumentation to extract eta_dec. The rho_ref is cross-checked against the
# full cAIC4::biasCorrectionPoisson output (sourced if CAIC4_SRC is set).
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src)
#   FIXTURE    path to the output HDF5 file
#              (default <script dir>/fixtures/dof_glmm_poisson_level1.h5)
#
# Usage:  Rscript test/generate_fixtures_glmm_poisson.R

suppressMessages(library(lme4))
suppressMessages(source(file.path(dirname(normalizePath(sub("^--file=","",commandArgs(FALSE)[grep("^--file=",commandArgs(FALSE))]))),"fixture_io.R")))

caic4_src <- Sys.getenv("CAIC4_SRC", "/private/tmp/cAIC4_src")
fixture_path <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture_path)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture_path <- file.path(here, "fixtures", "dof_glmm_poisson_level1.h5")
}

# ── Instrumented Chen-Stein refit loop ────────────────────────────────────────
# Reimplements biasCorrectionPoisson from cAIC4 v1.1 R/biasCorrectionPoisson.R:13-24
# with intermediate value capture. deleteZeroComponents is the caller's responsibility.

chen_stein_instrumented <- function(zeroLessModel) {
  y    <- zeroLessModel@resp$y
  eta0 <- zeroLessModel@resp$eta
  ind  <- which(y != 0)               # 1-based; matches Julia's findall(!=(0), y)
  if (length(ind) == 0) {
    return(list(y = y, eta0 = eta0, ind = integer(0),
                eta_dec = numeric(0), rho_ref = 0))
  }
  # One refit per nonzero observation: decrement y_i by 1 (unit decrement, not
  # zeroing — cAIC4 source verbatim; see docs/math/0006 §6 #1).
  eta_dec <- vapply(ind, function(i) {
    y_dec    <- y
    y_dec[i] <- y_dec[i] - 1
    refit(zeroLessModel, newresp = y_dec)@resp$eta[i]
  }, numeric(1))
  rho_ref <- sum(y[ind] * (eta0[ind] - eta_dec))
  list(y = y, eta0 = eta0, ind = ind, eta_dec = eta_dec, rho_ref = rho_ref)
}

# ── Optional cross-check against installed cAIC4 ─────────────────────────────
bc_src <- file.path(caic4_src, "R", "biasCorrectionPoisson.R")
del_src <- file.path(caic4_src, "R", "deleteZeroComponents.R")
have_caic4 <- file.exists(bc_src) && file.exists(del_src)
if (have_caic4) {
  source(del_src)
  source(bc_src)
  caic4_version <- tryCatch(
    unname(read.dcf(file.path(caic4_src, "DESCRIPTION"))[, "Version"]),
    error = function(e) "unknown"
  )
  message("cAIC4 v", caic4_version, " sourced for cross-check")
} else {
  message("CAIC4_SRC not found — skipping biasCorrectionPoisson cross-check")
}

# ── Dataset builder ───────────────────────────────────────────────────────────

build_case <- function(name, seed, n_groups, n_per_group, beta0, beta1, sigma_u) {
  set.seed(seed)
  n     <- n_groups * n_per_group
  group <- rep(seq_len(n_groups), each = n_per_group)
  x     <- rnorm(n)
  u     <- rnorm(n_groups, mean = 0, sd = sigma_u)
  eta   <- beta0 + beta1 * x + u[group]
  y     <- rpois(n, exp(eta))

  dat   <- data.frame(y = y, x = x, group = factor(group))
  m     <- glmer(y ~ x + (1 | group), data = dat, family = poisson)

  # deleteZeroComponents: for a non-singular fit the model is returned unchanged.
  zeroLessModel <- if (have_caic4) deleteZeroComponents(m) else m

  # Instrumented Chen-Stein loop
  res <- chen_stein_instrumented(zeroLessModel)

  # Cross-check against full biasCorrectionPoisson if cAIC4 is available.
  # biasCorrectionPoisson returns a list: $bc = the df scalar, $newModel, $new.
  if (have_caic4) {
    bc_result <- biasCorrectionPoisson(m)
    bc_full   <- bc_result$bc
    tol <- 1e-10
    if (abs(bc_full - res$rho_ref) > tol) {
      stop(sprintf(
        "Case %s: instrumented rho_ref %.6f != biasCorrectionPoisson $bc %.6f (diff %.2e > %.2e)",
        name, res$rho_ref, bc_full, abs(bc_full - res$rho_ref), tol
      ))
    }
    message(sprintf("  %s: rho_ref = %.6f  (cross-check passed, diff %.2e)", name, res$rho_ref,
                    abs(bc_full - res$rho_ref)))
  } else {
    message(sprintf("  %s: rho_ref = %.6f", name, res$rho_ref))
  }

  list(
    raw_data = list(y = as.numeric(y), x = as.numeric(x), group = as.integer(group)),
    y       = as.numeric(res$y),
    eta0    = as.numeric(res$eta0),
    ind     = as.integer(res$ind),    # 1-based; Julia indices are also 1-based
    eta_dec = as.numeric(res$eta_dec),
    rho_ref = as.numeric(res$rho_ref)
  )
}

# ── Case definitions ──────────────────────────────────────────────────────────

cases <- list(
  # Tracer case: moderate signal, balanced, 6 groups × 4 obs = n = 24
  poisson_v0 = build_case("poisson_v0", seed = 28L,
                           n_groups = 6, n_per_group = 4,
                           beta0 = 1.0, beta1 = 0.3, sigma_u = 0.5),
  # Weak random effect (σ_u near boundary): tests near-singular path
  poisson_weak_re = build_case("poisson_weak_re", seed = 281L,
                               n_groups = 8, n_per_group = 3,
                               beta0 = 1.5, beta1 = -0.2, sigma_u = 0.05),
  # Higher mean (fewer zeros): larger n, more nonzero obs → more refits
  poisson_high_mu = build_case("poisson_high_mu", seed = 282L,
                               n_groups = 5, n_per_group = 5,
                               beta0 = 2.0, beta1 = 0.5, sigma_u = 0.4)
)

# ── Write fixture ─────────────────────────────────────────────────────────────

dir.create(dirname(fixture_path), recursive = TRUE, showWarnings = FALSE)
if (file.exists(fixture_path)) file.remove(fixture_path)

on.exit(h5closeAll())
h5createFile(fixture_path)

put <- function(path, value) {
  h5write(value, fixture_path, path)
}

for (cname in names(cases)) {
  h5createGroup(fixture_path, cname)
  h5createGroup(fixture_path, file.path(cname, "raw_data"))
  c <- cases[[cname]]

  put(file.path(cname, "raw_data", "y"),     c$raw_data$y)
  put(file.path(cname, "raw_data", "x"),     c$raw_data$x)
  put(file.path(cname, "raw_data", "group"), c$raw_data$group)

  put(file.path(cname, "y"),       c$y)
  put(file.path(cname, "eta0"),    c$eta0)
  put(file.path(cname, "ind"),     c$ind)
  put(file.path(cname, "eta_dec"), c$eta_dec)
  put(file.path(cname, "rho_ref"), c$rho_ref)
}

# Metadata
h5createGroup(fixture_path, "meta")
put("meta/generator", "cAIC.jl test/generate_fixtures_glmm_poisson.R")
put("meta/r_version",  as.character(getRversion()))
put("meta/lme4_version",
    as.character(packageVersion("lme4")))
put("meta/caic4_version", if (have_caic4) caic4_version else "not-available")

message("Wrote fixture: ", fixture_path)
message("Cases: ", paste(names(cases), collapse = ", "))
