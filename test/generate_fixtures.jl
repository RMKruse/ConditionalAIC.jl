#!/usr/bin/env julia
#
# Level-1 fixture generator (Julia side) — issue #7 / ADR-0003.
#
# Constructs seeded, synthetic, parametrisation-neutral `calculateGaussianBc` component
# sets and writes them to an HDF5 file. The R side (`generate_fixtures.R`) then reads this
# file, calls `cAIC4`'s `calculateGaussianBc` on each case, and writes the reference ρ
# back into the same file. CI reads the resulting fixture in Julia with **no R** and
# compares against `ConditionalAIC.DofLMM.dof_lmm` at the Level-1 tolerance.
#
# Neutral hand-off (ADR-0003): HDF5 gives a binary-exact Float64 round-trip. The
# computation-bearing matrices `A`, `V0inv`, `Wⱼ = Z Dⱼ Zᵀ` are **symmetric**, so they
# survive the Julia(col-major)↔R round-trip unchanged. `calculateGaussianBc` touches the
# fixed-effects design `X` only through `ncol(X)`, so we store the scalars `n`/`p` rather
# than `X` — avoiding any matrix-orientation ambiguity across the language boundary.
#
# Usage:  julia --project=test test/generate_fixtures.jl
#
# `inv` is used below deliberately: this is synthetic *fixture construction* (test
# infrastructure), not shipped library code, so the §9 "no explicit inverse" rule — which
# governs the package kernels — does not apply here; clarity wins.

using HDF5
using LinearAlgebra
using Random

const FIXTURE = joinpath(@__DIR__, "fixtures", "dof_lmm_level1.h5")

"""
    componentset(; X, Z, λs, Ds, y, isREML, sigma_penalty)

Build a `calculateGaussianBc` component set (`docs/math/0002` §3–§6) from a synthetic
LMM design. `λs[j]` is the relative variance multiplying the `j`-th derivative block
`Ds[j]` (a q×q symmetric 0/1 pattern, `∂D*/∂(component j)`; §6), so the scaled marginal
variance is `V₀ = Iₙ + Z (Σⱼ λs[j] Ds[j]) Zᵀ` and `Wⱼ = Z Ds[j] Zᵀ`.

`B` is a synthetic s×s SPD matrix standing in for the optimiser's numeric Hessian — the
`analytic = FALSE` B-source (`docs/math/0004` §2). It is parametrisation-neutral
(ADR-0003) and independent of the case rng, so adding it leaves every `analytic = TRUE`
component byte-identical; the R side feeds it to `calculateGaussianBc(analytic = FALSE)`.
"""
function componentset(; X, Z, λs, Ds, y, isREML::Bool, sigma_penalty::Int)
    n = size(X, 1)
    p = size(X, 2)
    Dstar = sum(λs[j] .* Ds[j] for j in eachindex(Ds))
    V0 = Matrix{Float64}(I, n, n) .+ Z * Dstar * Z'
    V0inv = inv(V0)
    V0inv = (V0inv + V0inv') / 2                       # exact symmetry for the round-trip
    XtViX = Symmetric(X' * V0inv * X)
    A = V0inv .- V0inv * X * inv(XtViX) * X' * V0inv
    A = (A + A') / 2
    Wlist = [Z * D * Z' for D in Ds]
    e = A * y                                          # residual identity e = A y (§0)
    tye = dot(y, e)
    eWelist = [dot(e, W * e) for W in Wlist]
    B = _syntheticB(length(Ds), hash((n, p, length(Ds), isREML)))
    return (; n, p, A, V0inv, Wlist, eWelist, tye, e, isREML, sigma_penalty, B)
end

# A seeded, well-conditioned symmetric positive-definite s×s matrix, deterministic in the
# case's shape signature and independent of the case rng (so the `analytic = TRUE`
# components stay byte-identical). `M'M + sI` is SPD, so `solve(B)` in `calculateGaussianBc`
# succeeds. Stands in for the optimiser's numeric Hessian (the `analytic = FALSE` B-source).
function _syntheticB(s::Int, signature::UInt)
    rng = MersenneTwister(0x42_0000_0000 ⊻ signature)  # 0x42 = 'B'
    M = randn(rng, s, s)
    B = M' * M + s * Matrix{Float64}(I, s, s)
    return (B + B') / 2                                # exact symmetry for the round-trip
end

# group-indicator design Z (n×ngroups), `nper[g]` observations in group g, with the
# single derivative pattern Dₛ = I_q (one variance component; §6 worked check).
function intercept_design(nper::AbstractVector{<:Integer})
    n = sum(nper)
    q = length(nper)
    Z = zeros(Float64, n, q)
    row = 1
    for (g, m) in enumerate(nper)
        Z[row:(row + m - 1), g] .= 1.0
        row += m
    end
    return Z, [Matrix{Float64}(I, q, q)]
end

# Two grouping factors sharing the `n` observations: A with `gA = maximum(assignA)`
# levels, B with `gB` levels. Crossed (`assignB` independent of `assignA`) or nested
# (B-levels unique to each A-level) is determined entirely by the assignments. The two
# derivative patterns are the per-block identities (one variance component each; §6).
function twofactor_design(
    assignA::AbstractVector{<:Integer}, assignB::AbstractVector{<:Integer}
)
    n = length(assignA)
    gA = maximum(assignA)
    gB = maximum(assignB)
    q = gA + gB
    Z = zeros(Float64, n, q)
    for i in 1:n
        Z[i, assignA[i]] = 1.0
        Z[i, gA + assignB[i]] = 1.0
    end
    DA = zeros(Float64, q, q)
    DB = zeros(Float64, q, q)
    for a in 1:gA
        DA[a, a] = 1.0
    end
    for b in 1:gB
        DB[gA + b, gA + b] = 1.0
    end
    return Z, [DA, DB]
end

# Correlated random intercept + slope (covariate `x`) over the groups in `nper`. The
# per-group 2×2 relative-covariance block has three free entries → three derivative
# patterns `[∂/∂D11, ∂/∂D21, ∂/∂D22]` (§5/§6); the matching `λs` give that block as
# `[D11 D21; D21 D22]`.
function slope_design(nper::AbstractVector{<:Integer}, x::AbstractVector)
    n = sum(nper)
    g = length(nper)
    q = 2g
    Z = zeros(Float64, n, q)
    Dii = zeros(Float64, q, q)
    Dis = zeros(Float64, q, q)
    Dss = zeros(Float64, q, q)
    row = 1
    for (gi, m) in enumerate(nper)
        a, b = 2gi - 1, 2gi
        for i in 0:(m - 1)
            Z[row + i, a] = 1.0
            Z[row + i, b] = x[row + i]
        end
        Dii[a, a] = 1.0
        Dss[b, b] = 1.0
        Dis[a, b] = Dis[b, a] = 1.0
        row += m
    end
    return Z, [Dii, Dis, Dss]
end

resp(rng, n; intercept=2.0) = intercept .+ randn(rng, n)   # synthetic response y

"""
    build_cases() -> Dict{String,Any}

The seeded synthetic Level-1 cases. The set spans the CLAUDE §6 edge cases (issue #7
acceptance): a single grouping factor, crossed and nested factors, correlated
intercept+slope, unbalanced data, θ on the variance boundary, and both ML and REML.
"""
function build_cases()
    cases = Dict{String,Any}()

    # ── intercept_ml: one random intercept, balanced, ML. The tracer case. ──────────
    let
        rng = MersenneTwister(0x6341_4943)               # "cAIC"
        Z, Ds = intercept_design(fill(3, 4))             # 4 groups × 3 obs = 12
        n = size(Z, 1)
        X = ones(Float64, n, 1)
        cases["intercept_ml"] = componentset(;
            X, Z, λs=[0.8], Ds, y=resp(rng, n), isREML=false, sigma_penalty=1
        )
    end

    # ── intercept_reml: single factor, REML. Drives the isREML branch (Wⱼ A, n−p). ──
    let
        rng = MersenneTwister(0x5245_4d4c)               # "REML"
        Z, Ds = intercept_design([2, 4, 3, 3, 5])        # unbalanced, 5 groups, n = 17
        n = size(Z, 1)
        X = ones(Float64, n, 1)
        cases["intercept_reml"] = componentset(;
            X, Z, λs=[1.3], Ds, y=resp(rng, n), isREML=true, sigma_penalty=1
        )
    end

    # ── crossed_ml: two crossed factors, unbalanced, ML, s = 2 (off-diagonal B). ────
    let
        rng = MersenneTwister(0x43_524f_5353)            # "CROSS"
        assignA = [1, 1, 2, 2, 3, 3, 1, 2, 3, 1, 2, 3]   # 3 levels
        assignB = [1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1]   # 2 levels, crossed with A
        Z, Ds = twofactor_design(assignA, assignB)
        n = size(Z, 1)
        X = [ones(Float64, n) range(-1.0, 1.0; length=n)]   # intercept + slope, p = 2
        cases["crossed_ml"] = componentset(;
            X, Z, λs=[0.6, 1.1], Ds, y=resp(rng, n), isREML=false, sigma_penalty=1
        )
    end

    # ── nested_reml: factor B nested in A (B-levels unique per A), REML, s = 2. ──────
    let
        rng = MersenneTwister(0x4e_4553_5444)            # "NESTD"
        assignA = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]         # 2 A-levels, unbalanced
        assignB = [1, 1, 2, 2, 3, 3, 4, 4, 4, 4]         # 4 B-levels, each within one A
        Z, Ds = twofactor_design(assignA, assignB)
        n = size(Z, 1)
        X = ones(Float64, n, 1)
        cases["nested_reml"] = componentset(;
            X, Z, λs=[0.9, 0.4], Ds, y=resp(rng, n), isREML=true, sigma_penalty=1
        )
    end

    # ── corr_slope_reml: correlated intercept+slope, REML, s = 3 (full 3×3 B). ──────
    let
        rng = MersenneTwister(0x53_4c4f_5045)            # "SLOPE"
        nper = [4, 3, 5, 4]
        n = sum(nper)
        x = collect(range(-1.5, 1.5; length=n))
        Z, Ds = slope_design(nper, x)
        X = [ones(Float64, n) x]                         # p = 2
        # D*-block [v_i c; c v_s] = [0.8 0.2; 0.2 0.5] is PD (0.8·0.5 > 0.2²).
        cases["corr_slope_reml"] = componentset(;
            X, Z, λs=[0.8, 0.2, 0.5], Ds, y=resp(rng, n), isREML=true, sigma_penalty=1
        )
    end

    # ── boundary_ml: variance component on the boundary (θ ≈ 0), ML. ────────────────
    let
        rng = MersenneTwister(0x424f_554e)               # "BOUN"
        Z, Ds = intercept_design(fill(4, 3))             # 3 groups × 4 obs = 12
        n = size(Z, 1)
        X = ones(Float64, n, 1)
        cases["boundary_ml"] = componentset(;
            X, Z, λs=[1.0e-7], Ds, y=resp(rng, n), isREML=false, sigma_penalty=1
        )
    end

    return cases
end

function write_fixture(path, cases)
    mkpath(dirname(path))
    h5open(path, "w") do f
        for (name, c) in cases
            g = create_group(f, name)
            g["n"] = c.n
            g["p"] = c.p
            g["s"] = length(c.Wlist)
            g["isREML"] = Int(c.isREML)
            g["sigma_penalty"] = c.sigma_penalty
            g["tye"] = c.tye
            g["e"] = c.e
            g["eWelist"] = c.eWelist
            g["A"] = c.A
            g["V0inv"] = c.V0inv
            g["B"] = c.B                               # synthetic numeric Hessian (analytic=FALSE)
            wg = create_group(g, "Wlist")
            for (j, W) in enumerate(c.Wlist)
                wg["W$j"] = W
            end
        end
        meta = create_group(f, "meta")
        meta["generator"] = "ConditionalAIC.jl test/generate_fixtures.jl"
        meta["julia_version"] = string(VERSION)
        meta["hdf5_jll_version"] = string(HDF5.API.h5_get_libversion())
    end
    return path
end

# Execute only when run as a script (`julia test/generate_fixtures.jl`), so the gated
# live-R re-validation test can `include` this file to reuse `build_cases` /
# `write_fixture` without overwriting the committed fixture.
if abspath(PROGRAM_FILE) == @__FILE__
    cases = build_cases()
    write_fixture(FIXTURE, cases)
    @info "Wrote Level-1 component fixture" path = FIXTURE ncases = length(cases) cases = sort(
        collect(keys(cases))
    )
end
