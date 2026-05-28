@testitem "dof_lmm reproduces cAIC4's calculateGaussianBc on Level-1 components" tags = [
    :level1
] begin
    # Level-1 correctness gate (ADR-0003): for every synthetic component set in the
    # committed HDF5 fixture, the ported `dof_lmm` must reproduce the effective df ρ that
    # `cAIC4::calculateGaussianBc(analytic = TRUE)` computed from the *identical*
    # components (written into the fixture by `generate_fixtures.R`). No R in this job —
    # the reference ρ is read straight from the committed fixture. Tolerance per CLAUDE §6.
    using HDF5
    using cAIC: DofLMM

    # `rhdf5` writes an R length-1 numeric as a 1-element array, so the reference ρ comes
    # back as a `Vector` rather than a scalar; coerce it before comparing.
    asscalar(x) = x isa AbstractArray ? only(x) : x

    fixture = joinpath(@__DIR__, "fixtures", "dof_lmm_level1.h5")
    @test isfile(fixture)

    h5open(fixture, "r") do f
        cases = filter(!=("meta"), keys(f))
        @test !isempty(cases)   # the fixture must actually contain cases
        for name in cases
            g = f[name]
            haskey(g, "rho_ref") ||
                error("fixture case `$name` has no rho_ref — run generate_fixtures.R")

            n = read(g["n"])::Int
            p = read(g["p"])::Int
            s = read(g["s"])::Int
            comps = DofLMM.GaussianComponents(
                zeros(Float64, n, p),                 # X enters only via p = ncol(X)
                read(g["e"]),
                read(g["A"]),
                read(g["V0inv"]),
                [read(g["Wlist"]["W$j"]) for j in 1:s],
                read(g["eWelist"]),
                read(g["tye"]),
                Bool(read(g["isREML"])),
            )
            ρ = DofLMM.dof_lmm(comps; sigmapenalty=asscalar(read(g["sigma_penalty"])))
            ρ_ref = asscalar(read(g["rho_ref"]))

            @test ρ ≈ ρ_ref rtol = 1e-6 atol = 1e-10
        end
    end
end

@testitem "dof_lmm_numeric reproduces calculateGaussianBc(analytic=FALSE) on Level-1 components" tags = [
    :level1
] begin
    # Level-1 correctness gate for the *numeric* B-source assembly (issue #11). The numeric
    # path (`analytic = FALSE`) takes the Hessian **B** externally (here the synthetic SPD
    # fixture B), rebuilds the rescaled cross-product `C`, and runs the *same* ρ assembly.
    # For every synthetic component set, `dof_lmm_numeric` must reproduce the ρ that
    # `cAIC4::calculateGaussianBc(analytic = FALSE)` computed from the *identical*
    # components **and the identical B** (written to the fixture by `generate_fixtures.R`).
    # This isolates the assembly arithmetic from how B is obtained, which differs per source.
    using HDF5
    using cAIC: DofLMM

    asscalar(x) = x isa AbstractArray ? only(x) : x

    fixture = joinpath(@__DIR__, "fixtures", "dof_lmm_level1.h5")
    @test isfile(fixture)

    h5open(fixture, "r") do f
        cases = filter(!=("meta"), keys(f))
        @test !isempty(cases)
        for name in cases
            g = f[name]
            haskey(g, "rho_ref_numeric") || error(
                "fixture case `$name` has no rho_ref_numeric — run generate_fixtures.R"
            )

            n = read(g["n"])::Int
            p = read(g["p"])::Int
            s = read(g["s"])::Int
            comps = DofLMM.GaussianComponents(
                zeros(Float64, n, p),
                read(g["e"]),
                read(g["A"]),
                read(g["V0inv"]),
                [read(g["Wlist"]["W$j"]) for j in 1:s],
                read(g["eWelist"]),
                read(g["tye"]),
                Bool(read(g["isREML"])),
            )
            B = read(g["B"])
            ρ = DofLMM.dof_lmm_numeric(
                comps, B; sigmapenalty=asscalar(read(g["sigma_penalty"]))
            )
            ρ_ref = asscalar(read(g["rho_ref_numeric"]))

            @test ρ ≈ ρ_ref rtol = 1e-6 atol = 1e-10
        end
    end
end

@testitem "live R re-validation against cAIC4 (gated by CAIC_LIVE_RCALL)" tags = [
    :live_rcall
] begin
    # Fixture-rot guard (CLAUDE §6): regenerate the seeded components, recompute the
    # reference ρ with *live* R + `cAIC4`, and check (a) `dof_lmm` still matches live
    # cAIC4 and (b) the committed reference ρ has not drifted. Skipped in the default
    # (no-R) CI job; enabled by `CAIC_LIVE_RCALL=1` locally and in the scheduled job.
    # Uses the same Rscript + HDF5 hand-off as the generator (ADR-0003); no `RCall.jl`.
    using HDF5
    using cAIC: DofLMM

    if get(ENV, "CAIC_LIVE_RCALL", "0") == "1"
        asscalar(x) = x isa AbstractArray ? only(x) : x
        here = @__DIR__
        committed = joinpath(here, "fixtures", "dof_lmm_level1.h5")

        # Read the committed reference ρ (before regenerating) for the rot check.
        committed_rho = Dict{String,Float64}()
        h5open(committed, "r") do f
            for name in filter(!=("meta"), keys(f))
                committed_rho[name] = asscalar(read(f[name]["rho_ref"]))
            end
        end

        # Regenerate components from seed into a temp fixture (reusing the generator's
        # functions), then fill in live-cAIC4 reference ρ via the R generator.
        include(joinpath(here, "generate_fixtures.jl"))
        tmp = joinpath(mktempdir(), "dof_lmm_level1.h5")
        write_fixture(tmp, build_cases())
        run(addenv(`Rscript $(joinpath(here, "generate_fixtures.R"))`, "FIXTURE" => tmp))

        h5open(tmp, "r") do f
            for name in filter(!=("meta"), keys(f))
                g = f[name]
                n = read(g["n"])::Int
                p = read(g["p"])::Int
                s = read(g["s"])::Int
                comps = DofLMM.GaussianComponents(
                    zeros(Float64, n, p),
                    read(g["e"]),
                    read(g["A"]),
                    read(g["V0inv"]),
                    [read(g["Wlist"]["W$j"]) for j in 1:s],
                    read(g["eWelist"]),
                    read(g["tye"]),
                    Bool(read(g["isREML"])),
                )
                ρ = DofLMM.dof_lmm(comps; sigmapenalty=asscalar(read(g["sigma_penalty"])))
                live_ref = asscalar(read(g["rho_ref"]))
                @test ρ ≈ live_ref rtol = 1e-6 atol = 1e-10                  # vs live cAIC4
                @test live_ref ≈ committed_rho[name] rtol = 1e-6 atol = 1e-10  # no rot
            end
        end
    else
        @info "Skipping live-RCall re-validation (set CAIC_LIVE_RCALL=1 to enable)"
    end
end

@testitem "dof_lmm is type-stable, generic over T, and validates component shapes" tags = [
    :level1
] begin
    using cAIC: DofLMM
    using LinearAlgebra

    # A small, self-consistent component set built directly (no fixture / no
    # `MixedModels`). `inv` here is test-only setup, not library code, so the §9
    # no-inverse rule does not apply.
    function tinycomps(::Type{T}; isREML::Bool=false) where {T<:AbstractFloat}
        Z = T[1 0; 1 0; 0 1; 0 1]          # 2 groups × 2 obs
        n = size(Z, 1)
        X = ones(T, n, 1)
        y = T[1.0, -0.5, 0.3, 0.8]
        V0 = Matrix{T}(I, n, n) .+ T(0.5) .* (Z * transpose(Z))
        V0inv = inv(V0)
        V0inv = (V0inv + transpose(V0inv)) / 2
        A =
            V0inv .-
            V0inv * X * inv(Symmetric(transpose(X) * V0inv * X)) * transpose(X) * V0inv
        A = (A + transpose(A)) / 2
        W1 = Z * transpose(Z)
        e = A * y
        return DofLMM.GaussianComponents(
            X, e, A, V0inv, [W1], [dot(e, W1 * e)], dot(y, e), isREML
        )
    end

    # A small SPD Hessian B for the numeric path (s = 1 here): `b > 0` is positive-definite.
    spdB(::Type{T}, c) where {T} =
        Matrix{T}(reshape([T(3)], length(c.Wlist), length(c.Wlist)))

    # Type stability via @inferred, for both objectives and both float widths — both the
    # analytic (`dof_lmm`) and numeric (`dof_lmm_numeric`) entry points.
    for isREML in (false, true)
        c64 = tinycomps(Float64; isREML)
        @test (@inferred DofLMM.dof_lmm(c64)) isa Float64
        @test (@inferred DofLMM.dof_lmm_numeric(c64, spdB(Float64, c64))) isa Float64

        c32 = tinycomps(Float32; isREML)
        ρ32 = @inferred DofLMM.dof_lmm(c32)
        @test ρ32 isa Float32
        @test ρ32 ≈ DofLMM.dof_lmm(c64) rtol = 1e-4   # tracks Float64 to single precision

        ρ32num = @inferred DofLMM.dof_lmm_numeric(c32, spdB(Float32, c32))
        @test ρ32num isa Float32
        @test ρ32num ≈ DofLMM.dof_lmm_numeric(c64, spdB(Float64, c64)) rtol = 1e-4
    end

    # The numeric path validates B's shape: a B that is not s×s raises ArgumentError.
    cnum = tinycomps(Float64)
    @test_throws ArgumentError DofLMM.dof_lmm_numeric(cnum, Matrix{Float64}(I, 2, 2))

    # Shape-inconsistent components must raise ArgumentError, never a silently-wrong ρ.
    c = tinycomps(Float64)
    n = length(c.e)
    @test_throws ArgumentError DofLMM.GaussianComponents(
        c.X,
        c.e,
        Matrix{Float64}(I, n + 1, n + 1),
        c.V0inv,
        c.Wlist,
        c.eWelist,
        c.tye,
        false,
    )
    @test_throws ArgumentError DofLMM.GaussianComponents(
        ones(n + 2, 1), c.e, c.A, c.V0inv, c.Wlist, c.eWelist, c.tye, false
    )
    @test_throws ArgumentError DofLMM.GaussianComponents(
        c.X, c.e, c.A, c.V0inv, c.Wlist, [c.eWelist; 0.0], c.tye, false
    )
end

@testitem "efron_penalty: arithmetic, type stability, and validation" tags = [
    :level1, :bootstrap
] begin
    # Level-1 isolation test for efron_penalty (issue #12, ADR-0003). The Efron formula is
    # tested with a hand-computable synthetic example, type-stabilitychecked at Float32/64,
    # and the validation guards are exercised.
    using cAIC: DofLMM

    # Hand-computed synthetic: n=2, B=1, yhat=[1,2], sigma=1, sigmapenalty=0
    # Ystar = [2, 3]ᵀ (one column), Yhatstar = [1.5, 2.5]ᵀ
    # dot([2-1, 3-2], [1.5-1, 2.5-2]) = dot([1,1],[0.5,0.5]) = 1.0
    # ρ = 1.0 / (1^2 * 1) + 0 = 1.0
    yhat = [1.0, 2.0]
    sigma = 1.0
    Ystar = reshape([2.0, 3.0], 2, 1)
    Yhatstar = reshape([1.5, 2.5], 2, 1)
    @test DofLMM.efron_penalty(yhat, sigma, Ystar, Yhatstar, 0) ≈ 1.0
    @test DofLMM.efron_penalty(yhat, sigma, Ystar, Yhatstar, 1) ≈ 2.0

    # Type stability over Float64 and Float32
    f64 = DofLMM.efron_penalty(
        Float64[1, 2],
        1.0,
        reshape(Float64[2, 3], 2, 1),
        reshape(Float64[1.5, 2.5], 2, 1),
        0,
    )
    @test (@inferred DofLMM.efron_penalty(
        Float64[1, 2],
        1.0,
        reshape(Float64[2, 3], 2, 1),
        reshape(Float64[1.5, 2.5], 2, 1),
        0,
    )) isa Float64
    @test (@inferred DofLMM.efron_penalty(
        Float32[1, 2],
        1.0f0,
        reshape(Float32[2, 3], 2, 1),
        reshape(Float32[1.5, 2.5], 2, 1),
        0,
    )) isa Float32

    # Validation
    @test_throws DomainError DofLMM.efron_penalty(
        [1.0, 2.0], 0.0, reshape([2.0, 3.0], 2, 1), reshape([1.5, 2.5], 2, 1), 0
    )
    @test_throws ArgumentError DofLMM.efron_penalty(
        [1.0, 2.0], 1.0, reshape([2.0, 3.0], 2, 1), reshape([1.5, 2.5], 2, 1), -1
    )
    @test_throws ArgumentError DofLMM.efron_penalty(
        [1.0, 2.0], 1.0, reshape([2.0, 3.0, 4.0], 3, 1), reshape([1.5, 2.5, 3.5], 3, 1), 0
    )
    @test_throws ArgumentError DofLMM.efron_penalty(
        [1.0, 2.0], 1.0, reshape([2.0, 3.0], 2, 1), reshape([1.5, 2.5, 3.5], 3, 1), 0
    )
end
