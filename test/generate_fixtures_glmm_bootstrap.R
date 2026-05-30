#!/usr/bin/env Rscript
#
# Level-2 fixture generator — conditional bootstrap df for Binomial GLMM (issue #30).
#
# Fits the CBPP Binomial GLMM with lme4 (identical model to the Julia Level-2 test:
#   incid/hsz ~ period + (1|herd), family = binomial, weights = hsz)
# and calls cAIC4::conditionalBootstrap(m, BootStrRep = 500) to produce the reference
# effective df rho_ref.
#
# The Julia test (dof_glmm_tests.jl: "dof_glmm_bootstrap Level-2 fixture") reads this
# fixture and checks:
#   dof_glmm_bootstrap(m; nboot=500, rng=Xoshiro(42)) ≈ rho_ref  atol=2.0
#
# Tolerance justification: Monte Carlo variance from B=500 bootstrap draws, plus the
# lme4/MixedModels.jl fit discrepancy for the same model. The same atol=2.0 gate is
# used for the Gaussian Level-2 bootstrap fixture (DECISIONS.md). The R and Julia RNGs
# are independent (MersenneTwister vs. Xoshiro), so agreement within Monte Carlo
# variance is the correct expectation.
#
# Formula pin: the script verifies textually that conditionalBootstrap in the sourced
# cAIC4 still uses the expected row-mean centring and (B-1)*sigma^2 divisor. If a cAIC4
# version bump silently drifts either term, fixture generation stops loud.
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src)
#   FIXTURE    path to the output HDF5 file
#              (default <script dir>/fixtures/dof_glmm_bootstrap_level2.h5)
#
# Usage: Rscript test/generate_fixtures_glmm_bootstrap.R

suppressMessages(library(lme4))
suppressMessages(library(rhdf5))

caic4_src <- Sys.getenv("CAIC4_SRC", "/private/tmp/cAIC4_src")
fixture <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture <- file.path(here, "fixtures", "dof_glmm_bootstrap_level2.h5")
}

# ── Source cAIC4 — conditionalBootstrap + its dependency deleteZeroComponents ──
for (fname in c("deleteZeroComponents.R", "conditionalBootstrap.R")) {
  src <- file.path(caic4_src, "R", fname)
  if (!file.exists(src)) {
    stop("cAIC4 source not found (", fname, "); set CAIC4_SRC to the cAIC4 source tree")
  }
  source(src)
}
caic4_version <- tryCatch(
  unname(read.dcf(file.path(caic4_src, "DESCRIPTION"))[, "Version"]),
  error = function(e) "unknown"
)
message("cAIC4 v", caic4_version, " sourced from ", caic4_src)

# ── Pin the conditionalBootstrap formula textually ────────────────────────────
# Any silent drift in the row-mean centring or (B-1)*sigma^2 divisor must fail
# fixture generation loudly rather than silently writing a wrong rho_ref.
body_str <- gsub("\\s+", "", paste(deparse(body(conditionalBootstrap)), collapse = ""))
stopifnot(
  "cAIC4 conditionalBootstrap row-mean centring drifted" =
    grepl("dataMatrix-rowMeans(dataMatrix)", body_str, fixed = TRUE),
  "cAIC4 conditionalBootstrap (B-1)*sigma^2 divisor drifted" =
    grepl("sum(workingEta*dataMatrix)/((BootStrRep-1)*sigma(object)^2)",
          body_str, fixed = TRUE)
)
message("Formula pin: conditionalBootstrap matches expected centring and divisor.")

# ── Fit the CBPP Binomial GLMM ────────────────────────────────────────────────
# Matches the Julia Level-2 test:
#   fit(MixedModel, @formula(incid/hsz ~ period + (1|herd)), cbpp, Binomial();
#       weights=float.(cbpp.hsz), progress=false)
# lme4's cbpp names the columns `incidence`/`size`; MixedModels.jl's dataset(:cbpp)
# renames them `incid`/`hsz`. Same data — the Julia Level-2 test uses incid/hsz.
data(cbpp, package = "lme4")
m <- glmer(
  incidence / size ~ period + (1 | herd),
  data    = cbpp,
  family  = binomial,
  weights = size
)
message(sprintf("CBPP Binomial GLMM fitted: sigma = %.6g, theta = %.6g",
                sigma(m), getME(m, "theta")))

# ── Run conditionalBootstrap ──────────────────────────────────────────────────
set.seed(42)  # Reproducible R-side result. The Julia side uses Xoshiro(42) — a
              # different RNG, so the draws differ; agreement within Monte Carlo
              # variance (atol=2.0) is the correct check.
B <- 500L
# conditionalBootstrap returns the bias correction (df) directly as an atomic numeric
# (the `bootBC` scalar), not a list — see cAIC4 R/conditionalBootstrap.R.
result  <- conditionalBootstrap(m, BootStrRep = B)
rho_ref <- as.numeric(result)
message(sprintf("rho_ref = %.10g  (B = %d)", rho_ref, B))

# ── Write fixture ─────────────────────────────────────────────────────────────
dir.create(dirname(fixture), recursive = TRUE, showWarnings = FALSE)
if (file.exists(fixture)) file.remove(fixture)
on.exit(h5closeAll())
h5createFile(fixture)

put <- function(path, value) {
  try(h5delete(fixture, path), silent = TRUE)
  h5write(value, fixture, path)
}

put("rho_ref", rho_ref)
put("nboot",   as.integer(B))

h5createGroup(fixture, "meta")
put("meta/generator",     "cAIC.jl test/generate_fixtures_glmm_bootstrap.R")
put("meta/cAIC4_version", caic4_version)
put("meta/rhdf5_version", as.character(packageVersion("rhdf5")))
put("meta/R_version",     R.version.string)
put("meta/lme4_version",  as.character(packageVersion("lme4")))
put("meta/model",         "incid/hsz ~ period + (1|herd), family=binomial, weights=hsz")
put("meta/dataset",       "cbpp (lme4)")
put("meta/seed",          42L)

message(sprintf("Wrote fixture: %s (rho_ref = %.6g, cAIC4 %s)", fixture, rho_ref, caic4_version))
