@testitem "Aqua quality assurance" begin
    using Aqua

    # LinearAlgebra and LogExpFunctions are now exercised by the `Numerics` primitives;
    # ForwardDiff/FiniteDiff back the M2 numeric Hessian B-sources, now wired through
    # `mm_internals.jl`'s `bhessian` (ADR-0002, #11), so they are no longer stale.
    # Statistics is declared ahead of use (later slices).
    Aqua.test_all(ConditionalAIC; stale_deps=(ignore=[:Statistics],))
end

@testitem "JET static analysis" begin
    # JET tracks unstable compiler internals and has no release for prerelease
    # Julia; on `nightly` it is also dropped from the test env (see CI.yml), so
    # `using JET` must not be reached there. `@static` resolves this at lowering.
    @static if isempty(VERSION.prerelease)
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
        @info "Skipping JET static analysis on prerelease Julia $(VERSION)"
    end
end
