@testitem "Aqua quality assurance" begin
    using Aqua

    # These core deps are declared ahead of use per the dependency spec: the numeric
    # primitives that pull in LinearAlgebra/Statistics/LogExpFunctions land in a later
    # slice, and ForwardDiff/FiniteDiff back the M2 Hessian B-sources (ADR-0002). They
    # are intentionally present-but-unused at the walking-skeleton stage.
    Aqua.test_all(
        cAIC;
        stale_deps=(
            ignore=[
                :LinearAlgebra, :Statistics, :LogExpFunctions, :ForwardDiff, :FiniteDiff
            ],
        ),
    )
end

@testitem "JET static analysis" begin
    using JET

    JET.test_package(cAIC; target_modules=(cAIC, cAIC.MMInternals))
end
