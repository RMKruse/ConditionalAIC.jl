#!/usr/bin/env Rscript
#
# Unified HDF5 fixture I/O for the R ground-truth generators (issue #7 / ADR-0003).
#
# The generators hand fixtures to/from the Julia side through plain HDF5 files. Two R
# HDF5 packages can do that, and which one is usable depends on the platform:
#
#   * `hdf5r` (CRAN) links the *system* libhdf5. It installs as a fast binary on the
#     Linux CI runners (Posit Package Manager) and is the writer used by the gated
#     live-R re-validation job. It does NOT build against Homebrew R on macOS-ARM (its
#     source build leaves H5* symbols undefined under R's `-undefined dynamic_lookup`
#     bundle link, and there is no CRAN binary for that platform string) — see the
#     ADR-0003 addendum.
#   * `rhdf5` (Bioconductor) bundles its own libhdf5 via `Rhdf5lib`, so it builds on
#     macOS-ARM with no system-HDF5 dependency and is the writer used for *local*
#     fixture regeneration on that platform. Its source build of the bundled HDF5 has,
#     however, broken on the Linux CI image (Rhdf5lib's CMake configure step fails on
#     ubuntu-24.04), which is why CI uses `hdf5r` instead.
#
# This file picks whichever package is installed — `hdf5r` first (the CI / robust path),
# `rhdf5` as the macOS-ARM fallback — and exposes the *same* small function surface the
# generators call: `h5createFile`, `h5createGroup`, `h5write`, `h5read`, `h5delete`,
# `h5closeAll`, `h5ls`. Under the `rhdf5` backend those names are simply rhdf5's own
# exports; under `hdf5r` they are thin wrappers over `H5File` that reproduce rhdf5's
# semantics. Nothing else in the pipeline changes: the on-disk hand-off is still HDF5,
# and `HDF5.jl` reads either writer's output identically (both follow the standard
# reverse-the-dimensions column/row-major convention, verified against HDF5.jl for the
# non-symmetric matrices that cross the boundary).
#
# Sourced (not library()'d) by every `generate_fixtures*.R`; defines its symbols in the
# global environment.

fixture_backend <- if (requireNamespace("hdf5r", quietly = TRUE)) {
  "hdf5r"
} else if (requireNamespace("rhdf5", quietly = TRUE)) {
  "rhdf5"
} else {
  stop("fixture_io.R: neither 'hdf5r' nor 'rhdf5' is installed")
}

# Provenance string for the fixture's `meta` group (replaces the old rhdf5-only field).
fixture_hdf5_backend <- function() {
  paste(fixture_backend, as.character(utils::packageVersion(fixture_backend)))
}

if (fixture_backend == "rhdf5") {
  # rhdf5 already exports h5createFile / h5createGroup / h5write / h5read / h5delete /
  # h5closeAll / h5ls with exactly the semantics the generators rely on. Nothing to wrap.
  suppressMessages(library(rhdf5))
} else {
  suppressMessages(library(hdf5r))

  # One open `H5File` handle per path, opened lazily and closed by `h5closeAll()`. This
  # mirrors rhdf5's path-keyed, stateless-looking API (`h5write(value, file, name)`)
  # while keeping a single underlying handle per file.
  .h5_handles <- new.env(parent = emptyenv())

  .h5_get <- function(file) {
    h <- .h5_handles[[file]]
    if (!is.null(h) && h$is_valid) {
      return(h)
    }
    # Re-open an existing file read/write; the generators always create it first.
    h <- hdf5r::H5File$new(file, mode = "r+")
    .h5_handles[[file]] <- h
    h
  }

  # Create every intermediate group on a dataset path `a/b/c` (hdf5r's create_group is
  # single-level), matching rhdf5's auto-creation under explicit h5createGroup calls.
  .h5_ensure_groups <- function(h, name) {
    parts <- strsplit(name, "/", fixed = TRUE)[[1]]
    if (length(parts) <= 1L) {
      return(invisible())
    }
    acc <- ""
    for (p in parts[-length(parts)]) {
      acc <- if (nzchar(acc)) paste0(acc, "/", p) else p
      if (!h$exists(acc)) h$create_group(acc)
    }
    invisible()
  }

  h5createFile <- function(file) {
    old <- .h5_handles[[file]]
    if (!is.null(old)) try(old$close_all(), silent = TRUE)
    h <- hdf5r::H5File$new(file, mode = "w") # truncates / creates
    .h5_handles[[file]] <- h
    invisible(TRUE)
  }

  h5createGroup <- function(file, group) {
    h <- .h5_get(file)
    .h5_ensure_groups(h, paste0(group, "/_")) # ensure parents of `group`
    if (!h$exists(group)) h$create_group(group)
    invisible(TRUE)
  }

  h5write <- function(obj, file, name, ...) {
    h <- .h5_get(file)
    .h5_ensure_groups(h, name)
    if (h$exists(name)) h$link_delete(name)
    h[[name]] <- obj
    h$flush()
    invisible(TRUE)
  }

  h5delete <- function(file, name) {
    h <- .h5_get(file)
    if (h$exists(name)) h$link_delete(name)
    invisible(TRUE)
  }

  # Recursively materialise a group into a nested named list (rhdf5's whole-group read),
  # or read a single dataset to an R array — matching `h5read(file, "grp")` and
  # `h5read(file, "grp/dset")` respectively.
  .h5_read_obj <- function(obj) {
    if (inherits(obj, "H5Group") || inherits(obj, "H5File")) {
      nm <- obj$ls()$name
      out <- lapply(nm, function(n) .h5_read_obj(obj[[n]]))
      names(out) <- nm
      out
    } else {
      obj$read()
    }
  }

  h5read <- function(file, name, ...) {
    h <- .h5_get(file)
    .h5_read_obj(h[[name]])
  }

  # rhdf5's h5ls returns a data.frame with at least `name` and `otype`
  # ("H5I_GROUP" / "H5I_DATASET"); the generators use only `recursive = FALSE` plus the
  # `otype == "H5I_GROUP"` filter.
  h5ls <- function(file, recursive = TRUE, ...) {
    h <- .h5_get(file)
    info <- h$ls(recursive = isTRUE(recursive))
    data.frame(
      name = as.character(info$name),
      otype = as.character(info$obj_type),
      stringsAsFactors = FALSE
    )
  }

  h5closeAll <- function(...) {
    for (k in ls(.h5_handles)) {
      try(.h5_handles[[k]]$close_all(), silent = TRUE)
      rm(list = k, envir = .h5_handles)
    }
    invisible(TRUE)
  }
}
