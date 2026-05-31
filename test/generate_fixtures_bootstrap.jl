#!/usr/bin/env julia
#
# Level-1 fixture generator (Julia side) — bootstrap path / shared-input parity gate.
#
# Writes seeded, synthetic `yhat`, `sigma`, `Y*`, `Ŷ*` matrices into an HDF5 file. The
# R side (`generate_fixtures_bootstrap.R`) reads it, evaluates `cAIC4`'s
# `conditionalBootstrap` bias-correction arithmetic on each case, and writes the
# reference `rho_ref` back. CI reads the resulting fixture in Julia with **no R** and
# compares against `ConditionalAIC.DofLMM.efron_penalty` at the Level-1 tolerance
# (rtol = 1e-6, atol = 1e-10).
#
# Why the inputs are synthetic, not from a real bootstrap: `conditionalBootstrap`'s
# bias-correction arithmetic is `bootBC = Σ ŷ*·(y*−ȳ*) / ((B−1)σ²)` — pure linear
# algebra in `Y*`, `Ŷ*`, `σ`. The simulate-and-refit machinery only *produces* `Y*`
# and `Ŷ*`; the Level-1 isolation tests the arithmetic itself on **fixed** matrices,
# independent of which optimiser, RNG, or family produced them. This is the same
# isolation principle as `generate_fixtures.jl` for `calculateGaussianBc`.
#
# Usage: `julia --project=test test/generate_fixtures_bootstrap.jl`

using HDF5
using LinearAlgebra
using Random
using Statistics: mean

const FIXTURE = joinpath(@__DIR__, "fixtures", "bootstrap_level1.h5")

"""
    bootstrap_case(rng; n, B, sigma, mean_scale = 1.0, signal = 0.5)

Build one synthetic `(yhat, sigma, Y*, Ŷ*)` case. `Y*` is `ŷ + σε` with
`ε ~ N(0, Iₙ)` (mimicking `lme4`'s `simulate(..., use.u = TRUE)` parametric draw),
and `Ŷ*` is correlated with `Y*` via `signal` to give a non-trivial covariance
penalty — exactly the structure a real `predict(refit(..., newresp = y*))` produces.
The arithmetic is parametrisation-neutral, so these matrices are valid inputs for
`conditionalBootstrap`'s bias-correction formula regardless of how they were generated.
"""
function bootstrap_case(
    rng::AbstractRNG;
    n::Int,
    B::Int,
    sigma::Float64,
    mean_scale::Float64=1.0,
    signal::Float64=0.5,
)
    B >= 2 || throw(ArgumentError("B must be ≥ 2 (cAIC4's (B−1) divisor); got $B"))
    yhat = mean_scale .* randn(rng, n)
    epsY = randn(rng, n, B)
    Ystar = yhat .+ sigma .* epsY
    epsH = randn(rng, n, B)
    # Yhatstar correlated with Y* (a `predict(refit(...))` would couple them similarly).
    Yhatstar = yhat .+ signal .* (Ystar .- yhat) .+ sqrt(1 - signal^2) .* sigma .* epsH
    return (; yhat, sigma, Ystar, Yhatstar)
end

"""
    build_bootstrap_cases() -> Dict{String, NamedTuple}

The seeded Level-1 cases for the bootstrap shared-input fixture. Spans `B = 2`
(the minimum the `(B−1)` divisor admits), small/mid/large `n` and `B`, and a high-
signal case where `ŷ*` is strongly correlated with `y*`.
"""
function build_bootstrap_cases()
    cases = Dict{String,NamedTuple}()

    # tracer: n=8, B=20 — moderate signal, σ=1.5
    cases["tracer"] = bootstrap_case(
        MersenneTwister(0x424f_4f54);            # "BOOT"
        n=8,
        B=20,
        sigma=1.5,
        mean_scale=2.0,
        signal=0.5,
    )

    # mid: n=25, B=100, weaker signal, smaller σ
    cases["mid"] = bootstrap_case(
        MersenneTwister(0x4d_4944_3030);         # "MID00"
        n=25,
        B=100,
        sigma=0.8,
        mean_scale=1.0,
        signal=0.3,
    )

    # large: n=50, B=500 — exercises the larger Σ; strong signal
    cases["large"] = bootstrap_case(
        MersenneTwister(0x4c_4152_4745);         # "LARGE"
        n=50,
        B=500,
        sigma=2.5,
        mean_scale=3.0,
        signal=0.7,
    )

    # min_B: B = 2 (the smallest value admitting cAIC4's (B−1) divisor)
    cases["min_B"] = bootstrap_case(
        MersenneTwister(0x4d_494e_5f42);         # "MIN_B"
        n=6,
        B=2,
        sigma=1.0,
        mean_scale=0.5,
        signal=0.4,
    )

    return cases
end

function write_bootstrap_fixture(path, cases)
    mkpath(dirname(path))
    h5open(path, "w") do f
        for (name, c) in cases
            g = create_group(f, name)
            g["yhat"] = c.yhat
            g["sigma"] = c.sigma
            g["Ystar"] = c.Ystar
            g["Yhatstar"] = c.Yhatstar
            g["n"] = size(c.Ystar, 1)
            g["B"] = size(c.Ystar, 2)
        end
        meta = create_group(f, "meta")
        meta["generator"] = "ConditionalAIC.jl test/generate_fixtures_bootstrap.jl"
        meta["julia_version"] = string(VERSION)
        meta["hdf5_jll_version"] = string(HDF5.API.h5_get_libversion())
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    cases = build_bootstrap_cases()
    write_bootstrap_fixture(FIXTURE, cases)
    @info "Wrote bootstrap Level-1 fixture" path = FIXTURE ncases = length(cases) cases = sort(
        collect(keys(cases))
    )
end
