#!/usr/bin/env Rscript
#
# Level-1 fixture generator (R / ground-truth side) — stepcAIC backward enumeration (M4, #38).
#
# Drives cAIC4's *real* internal `backwardStep` (`R/helperfuns_stepcAIC.R:93`) across a fixed
# set of random-effects structures and flag settings, canonicalises each resulting candidate
# `cnms` list, and writes the reference candidate sets into
# `test/fixtures/stepcaic_backward_level1.h5`. This is the cAIC4 ground truth the Julia
# `backwardcandidates` enumerator is asserted **set-equal** to (docs/math/0008 §2.5). No model
# is fitted: this is pure combinatorial structure (§6 Level 1).
#
# Representation bridge (docs/math/0008 §2.1). `lme4`'s `cnms` is a named list whose names may
# repeat: a *correlated* term `(1 + x | g)` is ONE entry `g = c("(Intercept)","x")`; an
# *uncorrelated* `(1 + x || g)` is the TWO single-label entries `g="(Intercept)"`, `g="x"`. The
# `RESpec` carries the correlation on a flag instead. Scenarios are authored here as canonical
# input strings `grouping/cor=<0|1>:dir1,dir2[;…]`, parsed to `cnms` by `parse_input` (correlated
# → one multi-label entry; uncorrelated → one entry per label). The Julia side parses the SAME
# input string into a `RESpec`. Candidate sets are compared under the canonical encoding
# `canon_cand` — a sorted multiset of `"grouping:sorted(labels)"` term-strings, carrying no
# correlated flag because the term *structure* already distinguishes the two (one two-label term
# vs two one-label terms).
#
# HDF5 I/O via `test/fixture_io.R` (hdf5r/rhdf5; ADR-0003 addenda), as in the other generators.
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src) — for the version stamp
#   FIXTURE    path to the HDF5 fixture (default <script dir>/fixtures/stepcaic_backward_level1.h5)
#
# Usage:  Rscript test/generate_fixtures_stepcaic.R

suppressMessages(library(cAIC4))
suppressMessages(source(file.path(dirname(normalizePath(sub("^--file=","",commandArgs(FALSE)[grep("^--file=",commandArgs(FALSE))]))),"fixture_io.R")))

caic4_src <- Sys.getenv("CAIC4_SRC", "/private/tmp/cAIC4_src")
fixture <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture <- file.path(here, "fixtures", "stepcaic_backward_level1.h5")
}

backwardStep <- cAIC4:::backwardStep
caic4_version <- as.character(packageVersion("cAIC4"))

# Pin the cAIC4 backwardStep landmarks textually: if a cAIC4 bump silently drops the `NA`
# terminal return, the post-dedup `removeUncor`/`removeNoInt` filter calls, or their order,
# fixture regeneration must stop loud (CLAUDE §10 — never paper over drift).
body_str <- gsub("\\s+", "", paste(deparse(body(backwardStep)), collapse = ""))
stopifnot(
  "cAIC4 backwardStep NA terminal return drifted" =
    grepl("return(NA)", body_str, fixed = TRUE),
  "cAIC4 backwardStep removeUncor filter drifted" =
    grepl("if(!allowCorrelationSel)listOfAllCombs<-removeUncor(listOfAllCombs)",
          body_str, fixed = TRUE),
  "cAIC4 backwardStep removeNoInt filter drifted" =
    grepl("if(!allowNoIntercept)listOfAllCombs<-removeNoInt(listOfAllCombs)",
          body_str, fixed = TRUE)
)

# ── representation bridge + canonical encoding ────────────────────────────────

# Parse a canonical input string "g/cor=1:(Intercept),days;item/cor=1:(Intercept)" into a
# repeated-name `cnms` list (correlated → one multi-label entry; uncorrelated → one per label).
parse_input <- function(s) {
  cnms <- list()
  for (grp in strsplit(s, ";", fixed = TRUE)[[1]]) {
    halves <- strsplit(grp, ":", fixed = TRUE)[[1]]
    meta <- halves[1]
    nm <- sub("/cor=.*", "", meta)
    correlated <- as.integer(sub(".*cor=", "", meta)) == 1L
    dirs <- strsplit(halves[2], ",", fixed = TRUE)[[1]]
    if (correlated) {
      cnms <- c(cnms, setNames(list(dirs), nm))
    } else {
      for (d in dirs) cnms <- c(cnms, setNames(list(d), nm))
    }
  }
  cnms
}

# Build a `keep` formula (for `interpret.random`) from a canonical keep-spec string, mirroring
# `cnmsConverter`: "(Intercept)" -> "1"; no-intercept term gets a trailing "0"; "(dirs | name)".
keep_formula <- function(s) {
  frags <- vapply(strsplit(s, ";", fixed = TRUE)[[1]], function(grp) {
    halves <- strsplit(grp, ":", fixed = TRUE)[[1]]
    nm <- sub("/cor=.*", "", halves[1])
    dirs <- strsplit(halves[2], ",", fixed = TRUE)[[1]]
    if ("(Intercept)" %in% dirs) dirs[dirs == "(Intercept)"] <- "1" else dirs <- c(dirs, "0")
    paste0("(", paste(dirs, collapse = " + "), " | ", nm, ")")
  }, character(1))
  as.formula(paste0("~ ", paste(frags, collapse = " + ")))
}

# Canonical encoding of one candidate `cnms` list: a sorted multiset of "name:sorted(labels)".
canon_cand <- function(cand) {
  terms <- vapply(seq_along(cand), function(j) {
    paste0(names(cand)[j], ":", paste(sort(cand[[j]]), collapse = ","))
  }, character(1))
  paste(sort(terms), collapse = ";")
}

# Canonical encoding of a whole backwardStep result: newline-joined, set-deduplicated candidate
# strings. The `NA` terminal (`backwardStep` returns a bare `NA`) and an empty `listOfAllCombs`
# both map to "" — the Julia enumerator returns an empty `Vector{RESpec}` for both (§2).
canon_result <- function(res) {
  if (length(res) == 0) return("")
  if (length(res) == 1 && (is.null(res[[1]]) ||
                           (length(res[[1]]) == 1 && is.na(res[[1]][[1]])))) {
    return("")
  }
  cands <- vapply(res, canon_cand, character(1))
  paste(sort(unique(cands)), collapse = "\n")
}

# ── scenarios (docs/math/0008 §2.5) ───────────────────────────────────────────
# Each: name, canonical input spec, selectcorrelation, allownointercept, keep formula ("" = none).
scenarios <- list(
  list("sleepstudy_default",   "subj/cor=1:(Intercept),days",                     0L, 0L, ""),
  list("sleepstudy_noint",     "subj/cor=1:(Intercept),days",                     0L, 1L, ""),
  list("sleepstudy_selcor",    "subj/cor=1:(Intercept),days",                     1L, 0L, ""),
  list("pastes_default",       "batch/cor=1:(Intercept);cask/cor=1:(Intercept)",  0L, 0L, ""),
  list("single_default",       "g/cor=1:(Intercept)",                             0L, 0L, ""),
  list("single_keep",          "g/cor=1:(Intercept)",                             0L, 0L, "g/cor=1:(Intercept)"),
  list("three_default",        "g/cor=1:(Intercept),x,y",                         0L, 0L, ""),
  list("three_noint",          "g/cor=1:(Intercept),x,y",                         0L, 1L, ""),
  list("uncor_default",        "g/cor=0:(Intercept),x",                           0L, 0L, ""),
  list("uncor_selcor_noint",   "g/cor=0:(Intercept),x",                           1L, 1L, ""),
  list("mixed_default",        "subj/cor=1:(Intercept),days;item/cor=1:(Intercept)", 0L, 0L, ""),
  list("mixed_noint",          "subj/cor=1:(Intercept),days;item/cor=1:(Intercept)", 0L, 1L, ""),
  list("pastes_keep_batch",    "batch/cor=1:(Intercept);cask/cor=1:(Intercept)",  0L, 0L, "batch/cor=1:(Intercept)")
)

# ── write ─────────────────────────────────────────────────────────────────────
if (file.exists(fixture)) file.remove(fixture)
h5createFile(fixture)
on.exit(h5closeAll())

put <- function(path, value) {
  try(h5delete(fixture, path), silent = TRUE)
  h5write(value, fixture, path)
}

for (sc in scenarios) {
  name <- sc[[1]]; input <- sc[[2]]; selcor <- sc[[3]]; noint <- sc[[4]]; keep_s <- sc[[5]]
  cnms <- parse_input(input)
  keep <- if (nzchar(keep_s)) keep_formula(keep_s) else NULL
  res <- backwardStep(cnms,
                      keep = keep,
                      allowCorrelationSel = selcor == 1L,
                      allowNoIntercept = noint == 1L)
  expected <- canon_result(res)

  h5createGroup(fixture, name)
  put(paste0(name, "/input"), input)
  put(paste0(name, "/selectcorrelation"), selcor)
  put(paste0(name, "/allownointercept"), noint)
  put(paste0(name, "/keep"), keep_s)
  put(paste0(name, "/expected"), expected)

  shown <- if (nzchar(expected)) gsub("\n", "  |  ", expected) else "<empty / terminal>"
  cat(sprintf("  %-20s -> %s\n", name, shown))
}

h5createGroup(fixture, "meta")
put("meta/cAIC4_version", caic4_version)
put("meta/hdf5_backend", fixture_hdf5_backend())
put("meta/R_version", R.version.string)

cat(sprintf("Wrote %d backward scenario(s) to %s (cAIC4 %s).\n",
            length(scenarios), fixture, caic4_version))
