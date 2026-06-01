#!/usr/bin/env Rscript
#
# `lm`/`glm` terminal Level-2 fixture generator (R / ground-truth side) — issue #36 / M4 /
# ADR-0006 / CLAUDE §6.
#
# A backward `stepcaic` search drops random-effects terms one at a time; dropping the *last*
# RE term yields a fixed-effects-only model that `MixedModels.jl` cannot represent, so the
# **terminal node** of the search is a plain `GLM.jl` `lm`/`glm` fit scored directly. This
# generator drives `cAIC4`'s *public* `cAIC()` on the equivalent R `lm`/`glm` fits and writes
# the ground-truth conditional AIC, df (= rank + 1), and conditional log-likelihood to an HDF5
# fixture. The Julia Level-2 test (`glm_terminal_tests.jl`) fits the *same embedded data* with
# `GLM.jl`, runs `cAIC.caic(::RegressionModel)`, and compares within the fit-discrepancy band
# recorded in DECISIONS.md.
#
# Unlike the LMM generators, the terminal scoring is a deterministic closed form: an `lm` is an
# exact OLS solve (R and Julia land on the same β̂ to ~machine precision) and a `glm` is IRLS to
# the same MLE, so the fit discrepancy here is far tighter than the iterative-LMM band. The data
# are embedded so both ecosystems score the *identical* sample (R and Julia RNGs never meet).
#
# `cAIC4`'s `(g)lm` branch is self-contained (it needs no internal helpers), so the installed
# package is used directly (`library(cAIC4)`) rather than the source tree the LMM generators
# source. The `lm`/`glm` branch re-extracts the response from the model call, so every fit is
# given an explicit `data =` frame.
#
# HDF5 writer: `rhdf5`. Env var FIXTURE overrides the output path.
#
# Usage:  Rscript test/generate_fixtures_glm_terminal.R

suppressMessages({
  library(rhdf5)
  library(cAIC4)
})

fixture <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture <- file.path(here, "fixtures", "caic_glm_terminal_level2.h5")
}

caic4_version <- tryCatch(
  as.character(packageVersion("cAIC4")),
  error = function(e) "unknown"
)

# Write the cAIC4 reference quantities for one terminal case into its group.
write_ref <- function(fixture, name, r) {
  h5createGroup(fixture, name)
  h5write(as.numeric(r$caic), fixture, paste0(name, "/caic"))
  h5write(as.numeric(r$df), fixture, paste0(name, "/df")) # = rank + 1
  h5write(as.numeric(r$loglikelihood), fixture, paste0(name, "/cll"))
}

if (file.exists(fixture)) file.remove(fixture)
dir.create(dirname(fixture), showWarnings = FALSE, recursive = TRUE)
h5createFile(fixture)
on.exit(h5closeAll())

# ── gaussian_lm — the LMM backward terminal: a plain OLS `lm` (cAIC4 gaussian branch) ─────────
# σ̂ is the MLE rescaling cAIC4 applies, summary$sigma · √((n−p)/n); cll = Σ dnorm(y, μ̂, σ̂).
x <- c(
  -1.2, -0.7, -0.4, -0.1, 0.2, 0.5, 0.9, 1.1, 1.4, 1.8,
  -0.9, -0.5, 0.0, 0.3, 0.6, 1.0, 1.3, 1.7, -0.2, 0.8
)
y <- c(
  0.5, 1.1, 1.4, 1.9, 2.2, 2.7, 3.1, 3.4, 3.9, 4.5,
  0.9, 1.3, 2.0, 2.4, 2.8, 3.3, 3.7, 4.4, 1.8, 3.0
)
dat <- data.frame(x = x, y = y)
fit <- lm(y ~ 1 + x, data = dat)
r <- cAIC(fit)
write_ref(fixture, "gaussian_lm", r)
h5write(as.numeric(y), fixture, "gaussian_lm/y") # embed the shared sample
h5write(as.numeric(x), fixture, "gaussian_lm/x")
cat(sprintf(
  "  %-14s caic=%.10f  df=%.10f  cll=%.10f\n",
  "gaussian_lm", r$caic, r$df, r$loglikelihood
))

# ── poisson_glm — the Poisson-GLMM backward terminal: a log-link Poisson `glm` ───────────────
# cAIC4 poisson branch: cll = Σ dpois(y, λ = μ̂), df = rank + 1. Counts drawn once and embedded
# so Julia's IRLS scores the identical sample (both converge to the same Poisson MLE).
set.seed(101)
n_p <- 40
xp <- round(rnorm(n_p), 4)
yp <- rpois(n_p, lambda = exp(0.4 + 0.6 * xp))
datp <- data.frame(x = xp, y = yp)
fitp <- glm(y ~ 1 + x, family = poisson(link = "log"), data = datp)
rp <- cAIC(fitp)
write_ref(fixture, "poisson_glm", rp)
h5write(as.numeric(yp), fixture, "poisson_glm/y")
h5write(as.numeric(xp), fixture, "poisson_glm/x")
cat(sprintf(
  "  %-14s caic=%.10f  df=%.10f  cll=%.10f\n",
  "poisson_glm", rp$caic, rp$df, rp$loglikelihood
))

# ── bernoulli_glm — the Bernoulli-GLMM backward terminal: a logit-link binary `glm` ──────────
# cAIC4 binomial branch with y ∈ {0,1}: size = length(unique(y)) − 1 = 1, so the density reduces
# to Bernoulli and matches `condloglik_bernoulli` exactly. cll = Σ dbinom(y, 1, μ̂), df = rank+1.
set.seed(202)
n_b <- 50
xb <- round(rnorm(n_b), 4)
yb <- rbinom(n_b, size = 1, prob = plogis(0.3 + 1.1 * xb))
datb <- data.frame(x = xb, y = yb)
fitb <- glm(y ~ 1 + x, family = binomial(link = "logit"), data = datb)
rb <- cAIC(fitb)
write_ref(fixture, "bernoulli_glm", rb)
h5write(as.numeric(yb), fixture, "bernoulli_glm/y")
h5write(as.numeric(xb), fixture, "bernoulli_glm/x")
cat(sprintf(
  "  %-14s caic=%.10f  df=%.10f  cll=%.10f\n",
  "bernoulli_glm", rb$caic, rb$df, rb$loglikelihood
))

# ── binomial_glm — the multi-trial-Binomial-GLMM backward terminal: a logit-link binomial `glm`
# with per-observation trial counts nᵢ > 1. This is the documented DEVIATION case (DECISIONS
# 2026-05-29 / 2026-05-30): `cAIC4`'s binomial branch evaluates `dbinom(y, size=|unique(y)|−1, μ̂)`
# on the *proportion* y — a non-integer x and a wrong size — and returns cll = −∞. So there is no
# finite `cAIC4::cAIC` Level-2 reference; the ground truth is the CORRECT binomial density at the
# true trial counts, base-R `dbinom(kᵢ, nᵢ, μ̂ᵢ)` (the same Level-1 reference that validates
# `condloglik_binomial`). df = rank + 1 as for every terminal.
set.seed(303)
n_bin <- 35
xm <- round(rnorm(n_bin), 4)
trials <- sample(2:8, n_bin, replace = TRUE) # multi-trial: every nᵢ > 1
k <- rbinom(n_bin, size = trials, prob = plogis(0.2 + 0.9 * xm))
ym <- k / trials # success proportion
datm <- data.frame(x = xm)
fitm <- glm(cbind(k, trials - k) ~ 1 + x, family = binomial(link = "logit"), data = datm)
mu_m <- as.numeric(predict(fitm, type = "response"))
cll_correct <- sum(dbinom(k, size = trials, prob = mu_m, log = TRUE))
df_m <- fitm$rank + 1
caic_correct <- -2 * cll_correct + 2 * df_m
caic4_cll <- suppressWarnings(cAIC(fitm)$loglikelihood) # the defective cAIC4 value (−∞), for the record

h5createGroup(fixture, "binomial_glm")
h5write(as.numeric(caic_correct), fixture, "binomial_glm/caic")
h5write(as.numeric(df_m), fixture, "binomial_glm/df")
h5write(as.numeric(cll_correct), fixture, "binomial_glm/cll")
h5write(as.numeric(ym), fixture, "binomial_glm/y") # success proportion kᵢ/nᵢ
h5write(as.numeric(xm), fixture, "binomial_glm/x")
h5write(as.numeric(trials), fixture, "binomial_glm/n") # per-observation trial counts
cat(sprintf(
  "  %-14s caic=%.10f  df=%.10f  cll=%.10f  (cAIC4 cll = %s, the −∞ defect)\n",
  "binomial_glm", caic_correct, df_m, cll_correct, format(caic4_cll)
))

h5createGroup(fixture, "meta")
h5write(caic4_version, fixture, "meta/cAIC4_version")
h5write(as.character(packageVersion("rhdf5")), fixture, "meta/rhdf5_version")
h5write(R.version.string, fixture, "meta/R_version")

cat(sprintf(
  "Wrote lm/glm terminal Level-2 reference cases to %s (cAIC4 %s).\n",
  fixture, caic4_version
))
