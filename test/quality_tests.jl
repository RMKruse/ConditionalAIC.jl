@testitem "Aqua quality assurance" begin
    using Aqua

    # LinearAlgebra and LogExpFunctions are now exercised by the `Numerics` primitives.
    # Statistics is declared ahead of use (later slices); ForwardDiff/FiniteDiff back the
    # M2 Hessian B-sources (ADR-0002) and are intentionally present-but-unused for now.
    Aqua.test_all(cAIC; stale_deps=(ignore=[:Statistics, :ForwardDiff, :FiniteDiff],))
end

@testitem "JET static analysis" begin
    using JET

    JET.test_package(cAIC; target_modules=(cAIC, cAIC.MMInternals, cAIC.Numerics))
end
