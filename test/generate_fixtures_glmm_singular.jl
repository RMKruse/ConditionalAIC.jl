#!/usr/bin/env julia
#
# Partial-singularity GLMM Level-2 fixture — Julia (sample) side, issue #32 / M3.
#
# Generates the canonical seed-35 Bernoulli sample (a `zerocorr(1 + x | g)` design whose
# random *slope* variance lands exactly on the boundary while the intercept survives — partial
# singularity) and writes it, with metadata, into the HDF5 fixture. The R side
# (`generate_fixtures_glmm_singular.R`) then reads this same sample, fits the equivalent
# `glmer(y ~ 1 + x + (1 + x || g))`, drives `cAIC4`'s public `cAIC()` (which drops the
# boundary slope via `deleteZeroComponents` and refits `(1 | g)`), and appends the
# ground-truth caic / df / conditional log-likelihood / boundary-refit flag.
#
# The sample is embedded so both ecosystems score the *identical* data (the Julia `Xoshiro`
# and R RNGs never meet). The partial-singularity boundary is reached in MixedModels.jl *and*
# lme4 alike on this sample — confirmed before pinning the fixture; the divergence regime where
# the two ecosystems disagree on *whether* a fit is singular is recorded in DECISIONS.md, not
# fixtured.
#
# The Julia Level-2 test (`glmm_partial_singularity_tests.jl`) reads y/x/g + references from
# this fixture, fits MixedModels.jl, runs `cAIC.caic` (which cascades the boundary reduction),
# and compares within atol = 1e-3 (the Level-2 tolerance; fit discrepancies between lme4 and
# MixedModels.jl propagate into the cAIC).
#
# Run this FIRST, then `Rscript test/generate_fixtures_glmm_singular.R`.
#
# Usage:  julia --project=/path/to/cAIC.jl test/generate_fixtures_glmm_singular.jl

using HDF5
using MixedModels
using Random: Xoshiro

const FIXTURE = joinpath(@__DIR__, "fixtures", "caic_glmm_singular_level2.h5")

# ── seed-35 design: 24 groups × 14 obs, random intercept + slope; the slope variance
# collapses to the boundary. `randn`/`rand` draws are pinned to Xoshiro(35) in this order. ──
rng = Xoshiro(35)
ng, npg = 24, 14
g = repeat(1:ng; inner=npg)
xg = randn(rng, ng)
x = repeat(xg; inner=npg)
b0 = repeat(randn(rng, ng); inner=npg) .* 0.7
y = Float64[rand(rng, Bernoulli(1 / (1 + exp(-e)))) for e in (0.3 .+ 0.4 .* x .+ b0)]

# Fail loud if this sample is not actually partially singular in MixedModels.jl — the whole
# point of the fixture is the partial-boundary regime.
m = fit(
    MixedModel,
    @formula(y ~ 1 + x + zerocorr(1 + x | g)),
    (; y, x, g),
    Bernoulli();
    progress=false,
    fast=true,
)
λ = only(m.reterms).λ
issingular(m) || error("seed-35 sample is not singular in MixedModels.jl; fixture invalid")
(λ[1, 1] != 0 && λ[2, 2] == 0) ||
    error("seed-35 sample is not *partially* singular (expected λ11≠0, λ22=0); got λ = $λ")

mkpath(dirname(FIXTURE))
isfile(FIXTURE) && rm(FIXTURE)
h5open(FIXTURE, "w") do f
    f["partial_bernoulli/y"] = y
    f["partial_bernoulli/x"] = x
    f["partial_bernoulli/g"] = collect(g)
    meta = create_group(f, "meta")
    meta["generator"] = "generate_fixtures_glmm_singular.jl"
    meta["seed"] = 35
    meta["n"] = length(y)
    meta["n_groups"] = ng
    meta["mixedmodels_version"] = string(pkgversion(MixedModels))
end

@info "Wrote seed-35 sample" FIXTURE n = length(y) npos = Int(sum(y)) λ11 = λ[1, 1] λ22 = λ[
    2, 2
]
@info "Now run: Rscript test/generate_fixtures_glmm_singular.R"
