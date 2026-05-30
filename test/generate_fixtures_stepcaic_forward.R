#!/usr/bin/env Rscript
#
# Level-1 fixture generator (R / ground-truth side) — stepcAIC forward enumeration (M4, #39).
#
# Drives cAIC4's *real* internal `forwardStep` (`R/helperfuns_stepcAIC.R:516`) across a fixed set of
# random-effects structures, slope/group candidates and flag settings, canonicalises each resulting
# candidate `cnms` list, and writes the reference candidate sets into
# `test/fixtures/stepcaic_forward_level1.h5`. This is the cAIC4 ground truth the Julia
# `forwardcandidates` enumerator is asserted **set-equal** to (docs/math/0008 §3.6). Two further
# groups pin the nesting ingredients: `allNestSubs` (pure string expansion) and `isNested`
# (`reformulas::isNested`). No model is fitted: pure combinatorial structure (§6 Level 1).
#
# Representation bridge (docs/math/0008 §2.1) and canonical encoding match the backward generator:
# a *correlated* term `(1 + x | g)` is ONE `cnms` entry `g = c("(Intercept)","x")`; an *uncorrelated*
# `(1 + x || g)` is TWO single-label entries. The Julia side parses the SAME canonical input string
# into a `RESpec`; candidate sets are compared under `canon_cand` (sorted multiset of
# "grouping:sorted(labels)" term-strings). `maxslopes` maps to `nrOfCombs = maxslopes + 1` (the +1 is
# the intercept slot, the driver redefine `R/stepcAIC.R:302`).
#
# HDF5 writer: `rhdf5` (ADR-0003 addendum 2026-05-27), as in the other generators.
#
# Env vars:
#   CAIC4_SRC  path to the cAIC4 source tree (default /private/tmp/cAIC4_src) — for the version stamp
#   FIXTURE    path to the HDF5 fixture (default <script dir>/fixtures/stepcaic_forward_level1.h5)
#
# Usage:  Rscript test/generate_fixtures_stepcaic_forward.R

suppressMessages(library(cAIC4))
suppressMessages(library(rhdf5))

caic4_src <- Sys.getenv("CAIC4_SRC", "/private/tmp/cAIC4_src")
fixture <- Sys.getenv("FIXTURE", "")
if (!nzchar(fixture)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- if (length(file_arg)) dirname(normalizePath(file_arg)) else "test"
  fixture <- file.path(here, "fixtures", "stepcaic_forward_level1.h5")
}

forwardStep <- cAIC4:::forwardStep
allNestSubs <- cAIC4:::allNestSubs
caic4_version <- as.character(packageVersion("cAIC4"))

# Pin the cAIC4 forwardStep landmarks textually: if a cAIC4 bump silently drops the NULL terminals,
# the removeUncor filter, or the one-direction-larger restriction, fixture regeneration must stop
# loud (CLAUDE §10 — never paper over drift).
body_str <- gsub("\\s+", "", paste(deparse(body(forwardStep)), collapse = ""))
stopifnot(
  "cAIC4 forwardStep NULL terminal return drifted" =
    grepl("if(length(allCombs)==0)return(NULL)", body_str, fixed = TRUE),
  "cAIC4 forwardStep removeUncor filter drifted" =
    grepl("if(!allowCorrelationSel)allCombs<-removeUncor(allCombs)", body_str, fixed = TRUE),
  "cAIC4 forwardStep one-direction-larger restriction drifted" =
    grepl("length(r[[i]])>length(cnms[[names(r)[i]]])+1", body_str, fixed = TRUE)
)

# ── representation bridge + canonical encoding (shared with backward generator) ─

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

canon_cand <- function(cand) {
  terms <- vapply(seq_along(cand), function(j) {
    paste0(names(cand)[j], ":", paste(sort(cand[[j]]), collapse = ","))
  }, character(1))
  paste(sort(terms), collapse = ";")
}

# forwardStep returns NULL (terminal/exhausted) or a list of candidate cnms. Both NULL and an empty
# list map to "" — the Julia enumerator returns an empty Vector{RESpec} for both (§3).
canon_result <- function(res) {
  if (is.null(res) || length(res) == 0) return("")
  cands <- vapply(res, canon_cand, character(1))
  paste(sort(unique(cands)), collapse = "\n")
}

splitcands <- function(s) if (nzchar(s)) strsplit(s, ",", fixed = TRUE)[[1]] else NULL

# ── forwardStep scenarios (docs/math/0008 §3.6) ───────────────────────────────
# Each: name, canonical input spec, slopeCandidates (""=none), groupCandidates (""=none),
#       maxslopes, useacross, selectcorrelation.
scenarios <- list(
  list("add_group",          "subj/cor=1:(Intercept)",        "",       "item", 2L, 0L, 0L),
  list("add_slope",          "subj/cor=1:(Intercept)",        "days",   "",     2L, 0L, 0L),
  list("add_slope_group",    "subj/cor=1:(Intercept)",        "days",   "item", 2L, 0L, 0L),
  list("add_slope_selcor",   "subj/cor=1:(Intercept)",        "days",   "",     2L, 0L, 1L),
  list("slope_group_selcor", "subj/cor=1:(Intercept)",        "days",   "item", 2L, 0L, 1L),
  list("useacross",          "subj/cor=1:(Intercept),days",   "",       "item", 2L, 1L, 0L),
  list("two_groups_slope",   "subj/cor=1:(Intercept);item/cor=1:(Intercept)", "days", "", 2L, 0L, 0L),
  list("maxslopes_cap",      "subj/cor=1:(Intercept)",        "x,y,z",  "",     1L, 0L, 0L),
  list("onelarger_cap",      "subj/cor=1:(Intercept)",        "x,y",    "",     2L, 0L, 0L),
  list("existing2_noacross", "subj/cor=1:(Intercept),days",   "x",      "",     2L, 0L, 0L),
  list("no_candidates",      "g/cor=1:(Intercept)",           "",       "",     2L, 0L, 0L)
)

# ── nesting ingredient fixtures (docs/math/0008 §3.5) ─────────────────────────
# allNestSubs: pure string expansion of a nesting expression -> sub-groupings.
nest_cases <- list(
  list("a/b",   "b:a,a"),
  list("a/b/c", "c:b:a,b:a,a")
)

# isNested: reformulas::isNested(f1, f2) over integer-coded factor vectors. Each case stores the two
# vectors and the expected boolean (1/0). f1 nested in f2 <=> every f1 level meets <=1 f2 level.
isnested <- reformulas::isNested
isnested_cases <- list(
  # f1 = 1:12 finer, f2 = rep(1:3, each=4) coarser -> f1 nested in f2
  list("fine_in_coarse", as.integer(1:12),                as.integer(rep(1:3, each = 4)), 1L),
  # coarse in fine -> not nested
  list("coarse_in_fine", as.integer(rep(1:3, each = 4)),  as.integer(1:12),              0L),
  # crossed -> not nested either way
  list("crossed",        as.integer(rep(1:2, 6)),         as.integer(rep(1:2, each = 6)), 0L),
  # identical factor -> trivially nested
  list("identical",      as.integer(rep(1:4, 3)),         as.integer(rep(1:4, 3)),       1L)
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
  name <- sc[[1]]; input <- sc[[2]]; slopes <- sc[[3]]; groups <- sc[[4]]
  maxslopes <- sc[[5]]; useacross <- sc[[6]]; selcor <- sc[[7]]
  cnms <- parse_input(input)
  res <- forwardStep(cnms,
                     slopeCandidates = splitcands(slopes),
                     groupCandidates = splitcands(groups),
                     nrOfCombs = maxslopes + 1L,
                     allowUseAcross = useacross == 1L,
                     allowCorrelationSel = selcor == 1L)
  expected <- canon_result(res)

  h5createGroup(fixture, name)
  put(paste0(name, "/input"), input)
  put(paste0(name, "/slopecandidates"), slopes)
  put(paste0(name, "/groupcandidates"), groups)
  put(paste0(name, "/maxslopes"), maxslopes)
  put(paste0(name, "/useacross"), useacross)
  put(paste0(name, "/selectcorrelation"), selcor)
  put(paste0(name, "/expected"), expected)

  shown <- if (nzchar(expected)) gsub("\n", "  |  ", expected) else "<empty / terminal>"
  cat(sprintf("  %-22s -> %s\n", name, shown))
}

# allNestSubs group
h5createGroup(fixture, "nest")
nest_names <- vapply(nest_cases, `[[`, character(1), 1)
nest_exp   <- vapply(nest_cases, `[[`, character(1), 2)
put("nest/expr", nest_names)
put("nest/expected", nest_exp)
for (nc in nest_cases) cat(sprintf("  allNestSubs(%-7s) -> %s\n", nc[[1]], nc[[2]]))

# isNested group: one subgroup per case
h5createGroup(fixture, "isnested")
for (ic in isnested_cases) {
  nm <- ic[[1]]
  stopifnot(isnested(ic[[2]], ic[[3]]) == (ic[[4]] == 1L))  # self-check vs reformulas
  h5createGroup(fixture, paste0("isnested/", nm))
  put(paste0("isnested/", nm, "/f1"), ic[[2]])
  put(paste0("isnested/", nm, "/f2"), ic[[3]])
  put(paste0("isnested/", nm, "/expected"), ic[[4]])
  cat(sprintf("  isNested(%-15s) -> %s\n", nm, ic[[4]] == 1L))
}

h5createGroup(fixture, "meta")
put("meta/cAIC4_version", caic4_version)
put("meta/rhdf5_version", as.character(packageVersion("rhdf5")))
put("meta/R_version", R.version.string)

cat(sprintf("Wrote %d forward scenario(s) + nesting ingredients to %s (cAIC4 %s).\n",
            length(scenarios), fixture, caic4_version))
