#!/usr/bin/env Rscript
#
# Level-2 fixture generator (R / ground-truth side) — stepcAIC *driver* (M4, #40).
#
# Drives cAIC4's public `stepcAIC` end-to-end on real `lme4` fits and records, per scenario,
# the selected model's `bestCAIC` and a provenance string for the final RE structure
# (`finalformula`). The Julia driver test (`test/stepcaic_driver_tests.jl`) runs the SAME
# search with `MixedModels.jl`/`cAIC.stepcaic` and asserts: the selected RE structure equals the
# structure authored Julia-side from cAIC4's decision (the `finalformula` here is its provenance,
# since the grouping NAMES differ across packages — R `Subject` vs Julia `subj` — so the canonical
# string is not directly comparable), and `selected.caic ≈ bestCAIC` within the Level-2 band.
#
# This is end-to-end (§6 Level 2): `lme4` and `MixedModels.jl` do not produce bit-identical fits,
# so the selected `bestCAIC` is compared within the fit-discrepancy-derived tolerance recorded in
# DECISIONS.md, exactly as the `caic` Level-2 gate (`generate_fixtures_level2.R`). No machinery is
# isolated here — the candidate ENUMERATION is the separate Level-1 gate (§2.5 / §3.6).
#
# HDF5 writer: `rhdf5`, as in the other generators. cAIC4 is used as the INSTALLED package (the
# public `stepcAIC`), matching the Level-1 stepcaic generator.
#
# Env vars:
#   FIXTURE  path to the HDF5 fixture (default <script dir>/fixtures/stepcaic_driver_level2.h5)
#
# Usage:  Rscript test/generate_fixtures_stepcaic_driver.R

suppressMessages({
  library(cAIC4)
  library(lme4)
  library(rhdf5)
})

fixture <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture <- file.path(here, "fixtures", "stepcaic_driver_level2.h5")
}

caic4_version <- as.character(packageVersion("cAIC4"))

# Pin the stepcAIC decision landmarks textually: a cAIC4 bump that silently changes the `<=`
# acceptance rule or the backward-terminal stop must stop fixture regeneration loud (CLAUDE §10).
body_str <- gsub("\\s+", "", paste(deparse(body(stepcAIC)), collapse = ""))
stopifnot(
  "cAIC4 stepcAIC <= acceptance rule drifted" =
    grepl("minCAIC<=cAICofMod", body_str, fixed = TRUE),
  "cAIC4 stepcAIC backward-terminal stop drifted" =
    grepl('direction=="backward"', body_str, fixed = TRUE)
)

data(Pastes)

# Each scenario: a tag, the lme4 formula, the data frame, the direction, an optional `keep`
# random-effects formula ("" = none), and `nsaved` (`numberOfSavedModels`, default 1). When
# `nsaved > 1` the scenario additionally records the ranked k-best set `savedcaics` =
# c(bestCAIC, attr(additionalModels, "cAICs")) — cAIC4 returns the selected `finalModel`
# separately from the runner-up `additionalModels`, so the full ranked k-best set is their
# concatenation (the `finalModel` is the global minimum, hence first). Mirrored in
# test/stepcaic_driver_tests.jl.
scenarios <- list(
  list(tag = "sleepstudy_slope",
       form = Reaction ~ 1 + Days + (1 + Days | Subject), data = sleepstudy,
       direction = "backward", keep = "", nsaved = 1),
  list(tag = "sleepstudy_int",
       form = Reaction ~ 1 + Days + (1 | Subject), data = sleepstudy,
       direction = "backward", keep = "", nsaved = 1),
  list(tag = "pastes_crossed",
       form = strength ~ 1 + (1 | batch) + (1 | cask), data = Pastes,
       direction = "backward", keep = "", nsaved = 1),
  list(tag = "pastes_keepbatch",
       form = strength ~ 1 + (1 | batch) + (1 | cask), data = Pastes,
       direction = "backward", keep = "~(1 | batch)", nsaved = 1),
  # k-best (`numberOfSavedModels = 2`): the two scored *mixed* candidates of the crossed model's
  # single backward step — `(1|batch)` (selected, 301.4828) and `(1|cask)` (314.2643). Both are
  # `lmerMod`; the `lm` terminal only enters at k = 3 (314.2727) and is excluded here to keep the
  # whole saved set anchorable within the Level-2 band (the terminal's stepCAIC glm-dispersion σ̂
  # diverges from the project's lm/MLE terminal — DECISIONS 2026-05-31).
  list(tag = "pastes_saved2",
       form = strength ~ 1 + (1 | batch) + (1 | cask), data = Pastes,
       direction = "backward", keep = "", nsaved = 2),
  # ── forward / both arcs (#41) ──────────────────────────────────────────────
  # Forward grows the slope: from (1|Subject), slopeCandidate Days adds the random Days slope
  # (the grown model improves), selecting (1 + Days | Subject). bestCAIC 1711.799.
  list(tag = "sleep_fwd_days",
       form = Reaction ~ 1 + Days + (1 | Subject), data = sleepstudy,
       direction = "forward", slope = "Days", keep = "", nsaved = 1),
  # `both` starts forward (R/stepcAIC.R:389) and reaches the same grown model in one accepted step.
  list(tag = "sleep_both_days",
       form = Reaction ~ 1 + Days + (1 | Subject), data = sleepstudy,
       direction = "both", slope = "Days", keep = "", nsaved = 1),
  # Forward-terminal arc (:435): from the full (1 + Days | Subject), slopeCandidate Days has no
  # admissible one-larger enlargement → forwardStep returns NULL → return the input as best. Forward
  # never descends to lm. bestCAIC = the input's own cAIC, 1711.799.
  list(tag = "sleep_fwd_full",
       form = Reaction ~ 1 + Days + (1 + Days | Subject), data = sleepstudy,
       direction = "forward", slope = "Days", keep = "", nsaved = 1),
  # Forward rejects: from (1|batch), groupCandidate cask adds (1|cask), but the crossed model does
  # NOT improve → keep (1|batch). bestCAIC 301.483 (the same incumbent the backward Pastes runs hit).
  list(tag = "pastes_fwd_cask",
       form = strength ~ 1 + (1 | batch), data = Pastes,
       direction = "forward", group = "cask", keep = "", nsaved = 1),
  # `both` from (1|batch): forward turn (add cask) does not improve → flip backward → lm terminal
  # does not improve → stop, keeping (1|batch). Exercises the two-non-improving-turns termination.
  list(tag = "pastes_both_cask",
       form = strength ~ 1 + (1 | batch), data = Pastes,
       direction = "both", group = "cask", keep = "", nsaved = 1)
)

if (file.exists(fixture)) file.remove(fixture)
dir.create(dirname(fixture), showWarnings = FALSE, recursive = TRUE)
h5createFile(fixture)
on.exit(h5closeAll())

put <- function(path, value) {
  try(h5delete(fixture, path), silent = TRUE)
  h5write(value, fixture, path)
}

for (sc in scenarios) {
  # `stepcAIC` resolves `data` via get(deparse(substitute(data))) — it must be a plain named
  # variable in this frame, not `sc$data` (which would deparse to the literal "sc$data").
  dat <- sc$data
  fit <- lmer(sc$form, data = dat, REML = FALSE)
  keep <- if (nzchar(sc$keep)) list(random = as.formula(sc$keep)) else NULL
  nsaved <- if (is.null(sc$nsaved)) 1 else sc$nsaved
  # forward / both arcs supply slope-/groupCandidates ("" or absent = none, the backward default).
  slopeCands <- if (!is.null(sc$slope) && nzchar(sc$slope)) sc$slope else NULL
  groupCands <- if (!is.null(sc$group) && nzchar(sc$group)) sc$group else NULL
  res <- stepcAIC(fit, direction = sc$direction, data = dat, keep = keep,
                  slopeCandidates = slopeCands, groupCandidates = groupCands,
                  numberOfSavedModels = nsaved, returnResult = TRUE, trace = FALSE)
  fm <- res$finalModel
  finalclass <- paste(class(fm), collapse = ",")
  finalformula <- paste(deparse(formula(fm)), collapse = " ")

  h5createGroup(fixture, sc$tag)
  put(paste0(sc$tag, "/bestCAIC"), as.numeric(res$bestCAIC))
  put(paste0(sc$tag, "/direction"), sc$direction)
  put(paste0(sc$tag, "/keep"), sc$keep)
  put(paste0(sc$tag, "/slope"), if (is.null(slopeCands)) "" else slopeCands)
  put(paste0(sc$tag, "/group"), if (is.null(groupCands)) "" else groupCands)
  put(paste0(sc$tag, "/initialformula"), paste(deparse(sc$form), collapse = " "))
  put(paste0(sc$tag, "/finalformula"), finalformula)
  put(paste0(sc$tag, "/finalclass"), finalclass)

  if (nsaved > 1) {
    # The full ranked k-best set: the selected `finalModel` (global minimum) first, then the
    # runner-up `additionalModels` in cAIC4's stored ascending order.
    am <- res$additionalModels
    savedcaics <- c(as.numeric(res$bestCAIC), as.numeric(attr(am, "cAICs")))
    savedformulas <- c(finalformula,
                       vapply(am, function(x) paste(deparse(formula(x)), collapse = " "),
                              character(1)))
    savedclasses <- c(finalclass,
                      vapply(am, function(x) paste(class(x), collapse = ","), character(1)))
    put(paste0(sc$tag, "/nsaved"), as.integer(nsaved))
    put(paste0(sc$tag, "/savedcaics"), savedcaics)
    put(paste0(sc$tag, "/savedformulas"), savedformulas)
    put(paste0(sc$tag, "/savedclasses"), savedclasses)
    cat(sprintf("  %-18s bestCAIC=%.6f  nsaved=%d savedcaics=[%s]\n",
                sc$tag, res$bestCAIC, nsaved,
                paste(format(savedcaics, digits = 10), collapse = ", ")))
  } else {
    cat(sprintf("  %-18s bestCAIC=%.6f  final[%s]: %s\n",
                sc$tag, res$bestCAIC, finalclass, finalformula))
  }
}

# ── GLMM scenario (Poisson, keep-incumbent) ───────────────────────────────────
# A crossed 2-RE Poisson `glmer` fit whose backward stepcAIC keeps the FULL model (both random
# intercepts are supported — dropping either raises the cAIC). Unlike the `lme4`-dataset scenarios
# above, GLMM Level-2 needs bit-identical data on both sides (lme4 and MixedModels.jl do not share
# datasets), so the synthetic data is stored as `raw_data` (the same sharing mechanism as
# generate_fixtures_glmm_poisson.R) and the Julia driver test re-fits the SAME columns. The
# Chen-Stein df is non-singular here; the measured lme4↔MixedModels cAIC discrepancy on this data
# is 9.6e-4, inside the GLMM end-to-end Level-2 band (atol=1e-3; DECISIONS 2026-05-29/30).
set.seed(404)
g_nsub <- 12; g_nit <- 10
g_sub <- gl(g_nsub, g_nit)                          # 12 subjects × 10 obs, crossed with item
g_it  <- factor(rep(1:g_nit, times = g_nsub))
g_x   <- rnorm(g_nsub * g_nit)
g_us  <- rnorm(g_nsub, 0, 0.7)
g_ui  <- rnorm(g_nit,  0, 0.5)
g_eta <- 0.8 + 0.4 * g_x + g_us[as.integer(g_sub)] + g_ui[as.integer(g_it)]
g_y   <- rpois(length(g_eta), exp(g_eta))
gdat  <- data.frame(y = g_y, x = g_x, sub = g_sub, it = g_it)
gfit  <- glmer(y ~ x + (1 | sub) + (1 | it), data = gdat, family = poisson)
gres  <- stepcAIC(gfit, direction = "backward", data = gdat,
                  numberOfSavedModels = 1, returnResult = TRUE, trace = FALSE)
gfm   <- gres$finalModel
gtag  <- "glmm_poisson_keep"
h5createGroup(fixture, gtag)
h5createGroup(fixture, file.path(gtag, "raw_data"))
put(file.path(gtag, "raw_data", "y"),   as.numeric(g_y))     # counts as Float64
put(file.path(gtag, "raw_data", "x"),   as.numeric(g_x))
put(file.path(gtag, "raw_data", "sub"), as.integer(g_sub))   # 1-based grouping codes
put(file.path(gtag, "raw_data", "it"),  as.integer(g_it))
put(paste0(gtag, "/bestCAIC"), as.numeric(gres$bestCAIC))
put(paste0(gtag, "/direction"), "backward")
put(paste0(gtag, "/family"), "poisson")
put(paste0(gtag, "/initialformula"), "y ~ 1 + x + (1 | sub) + (1 | it)")
put(paste0(gtag, "/finalformula"), paste(deparse(formula(gfm)), collapse = " "))
put(paste0(gtag, "/finalclass"), paste(class(gfm), collapse = ","))
cat(sprintf("  %-18s bestCAIC=%.6f  final[%s]: %s\n",
            gtag, gres$bestCAIC, paste(class(gfm), collapse = ","),
            paste(deparse(formula(gfm)), collapse = " ")))

# ── GLMM forward scenario (#41) ───────────────────────────────────────────────
# Same seed-404 crossed-Poisson data, but the search STARTS from a single random intercept
# `y ~ x + (1 | sub)` and grows forward with groupCandidate `it`. cAIC4 adds the second random
# intercept, selecting `y ~ x + (1 | it) + (1 | sub)` — the same full crossed model as above
# (bestCAIC 448.206). Exercises the forward arc on the GLMM family-dispatch path (the candidate
# refit uses the GLM distribution family). raw_data is stored again so the Julia test is
# self-contained (the columns are bit-identical to glmm_poisson_keep's).
gffit <- glmer(y ~ x + (1 | sub), data = gdat, family = poisson)
gfres <- stepcAIC(gffit, direction = "forward", data = gdat, groupCandidates = "it",
                  numberOfSavedModels = 1, returnResult = TRUE, trace = FALSE)
gffm  <- gfres$finalModel
gftag <- "glmm_fwd_it"
h5createGroup(fixture, gftag)
h5createGroup(fixture, file.path(gftag, "raw_data"))
put(file.path(gftag, "raw_data", "y"),   as.numeric(g_y))
put(file.path(gftag, "raw_data", "x"),   as.numeric(g_x))
put(file.path(gftag, "raw_data", "sub"), as.integer(g_sub))
put(file.path(gftag, "raw_data", "it"),  as.integer(g_it))
put(paste0(gftag, "/bestCAIC"), as.numeric(gfres$bestCAIC))
put(paste0(gftag, "/direction"), "forward")
put(paste0(gftag, "/family"), "poisson")
put(paste0(gftag, "/group"), "it")
put(paste0(gftag, "/initialformula"), "y ~ 1 + x + (1 | sub)")
put(paste0(gftag, "/finalformula"), paste(deparse(formula(gffm)), collapse = " "))
put(paste0(gftag, "/finalclass"), paste(class(gffm), collapse = ","))
cat(sprintf("  %-18s bestCAIC=%.6f  final[%s]: %s\n",
            gftag, gfres$bestCAIC, paste(class(gffm), collapse = ","),
            paste(deparse(formula(gffm)), collapse = " ")))

h5createGroup(fixture, "meta")
put("meta/cAIC4_version", caic4_version)
put("meta/lme4_version", as.character(packageVersion("lme4")))
put("meta/rhdf5_version", as.character(packageVersion("rhdf5")))
put("meta/R_version", R.version.string)

cat(sprintf("Wrote %d driver scenario(s) to %s (cAIC4 %s, lme4 %s).\n",
            length(scenarios) + 2L, fixture, caic4_version, as.character(packageVersion("lme4"))))
