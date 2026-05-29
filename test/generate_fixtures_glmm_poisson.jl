#!/usr/bin/env julia
#
# Fixture verifier / metadata printer for the Poisson GLMM Chen-Stein df (issue #28).
#
# The fixture `test/fixtures/dof_glmm_poisson_level1.h5` is generated entirely by the R
# side (`generate_fixtures_glmm_poisson.R`): R fits the lme4 GLMM, instruments the
# Chen-Stein refit loop, captures intermediate values, and writes the HDF5 file. This
# script verifies the fixture is present and prints a human-readable summary; it does not
# modify the file.
#
# Unlike the LMM fixture pipeline (where Julia generates mathematical components and R
# adds reference ρ), the GLMM fixture is entirely R-side because the intermediate values
# (η̂, η̂^{(-i)}) require a fitted model. See `docs/math/0006-glmm-bias-correction.md` §3
# and ADR-0003 for the Level-1 isolation boundary.
#
# Usage:  julia --project=test test/generate_fixtures_glmm_poisson.jl
#         (or run generate_fixtures_glmm_poisson.R first if the fixture is missing)

using HDF5

const FIXTURE = joinpath(@__DIR__, "fixtures", "dof_glmm_poisson_level1.h5")

if !isfile(FIXTURE)
    @error "Fixture not found. Run first:" cmd = "Rscript test/generate_fixtures_glmm_poisson.R"
    exit(1)
end

h5open(FIXTURE, "r") do f
    meta = f["meta"]
    @info "Fixture metadata" generator = read(meta["generator"]) r_version = read(
        meta["r_version"]
    ) lme4_version = read(meta["lme4_version"]) caic4_version = read(meta["caic4_version"])

    cases = filter(!=("meta"), keys(f))
    @info "Cases found: $(length(cases))" cases = cases

    for name in cases
        g = f[name]
        y = read(g["y"])
        eta0 = read(g["eta0"])
        ind = Vector{Int}(read(g["ind"]))
        eta_dec = read(g["eta_dec"])
        rho_ref = only(read(g["rho_ref"]))
        n_nonzero = length(ind)
        n_total = length(y)
        @info "  $name" n = n_total n_nonzero rho_ref
    end
end
