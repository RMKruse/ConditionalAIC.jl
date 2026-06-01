@testitem "Aqua quality assurance" begin
    using Aqua

    # LinearAlgebra and LogExpFunctions are now exercised by the `Numerics` primitives;
    # ForwardDiff/FiniteDiff back the M2 numeric Hessian B-sources, now wired through
    # `mm_internals.jl`'s `bhessian` (ADR-0002, #11), so they are no longer stale.
    # Statistics is declared ahead of use (later slices).
    Aqua.test_all(ConditionalAIC; stale_deps=(ignore=[:Statistics],))
end

@testitem "JET static analysis" begin
    # JET 0.11 transitively requires Julia >= 1.12 (via PrecompileTools 1.3.x), so it is
    # uninstallable on the 1.10/1.11 LTS matrix; and JET tracks unstable compiler internals
    # with no release for prerelease Julia. On every version except a released >= 1.12 it is
    # dropped from the test env (see CI.yml), so `using JET` must not be reached there. The
    # guard mirrors exactly where JET is installed; `@static` resolves it at lowering so the
    # `using` is only spliced in when JET is present.
    @static if VERSION >= v"1.12" && isempty(VERSION.prerelease)
        using JET

        JET.test_package(
            ConditionalAIC;
            target_modules=(
                ConditionalAIC,
                ConditionalAIC.MMInternals,
                ConditionalAIC.Numerics,
                ConditionalAIC.Loglik,
                ConditionalAIC.DofLMM,
                ConditionalAIC.DofGLMM,
                ConditionalAIC.Components,
            ),
        )
    else
        @info "Skipping JET static analysis on Julia $(VERSION) (JET 0.11 needs released Julia >= 1.12)"
    end
end
