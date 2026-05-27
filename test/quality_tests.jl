@testitem "Aqua quality assurance" begin
    using Aqua

    # LinearAlgebra and LogExpFunctions are now exercised by the `Numerics` primitives.
    # Statistics is declared ahead of use (later slices); ForwardDiff/FiniteDiff back the
    # M2 Hessian B-sources (ADR-0002) and are intentionally present-but-unused for now.
    Aqua.test_all(cAIC; stale_deps=(ignore=[:Statistics, :ForwardDiff, :FiniteDiff],))
end

@testitem "JET static analysis" begin
    # JET tracks unstable compiler internals and has no release for prerelease
    # Julia; on `nightly` it is also dropped from the test env (see CI.yml), so
    # `using JET` must not be reached there. `@static` resolves this at lowering.
    @static if isempty(VERSION.prerelease)
        using JET

        JET.test_package(cAIC; target_modules=(cAIC, cAIC.MMInternals, cAIC.Numerics))
    else
        @info "Skipping JET static analysis on prerelease Julia $(VERSION)"
    end
end
