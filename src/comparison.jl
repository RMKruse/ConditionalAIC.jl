# The anocaic comparison method. Included directly into the `ConditionalAIC` module.
# Scores a user-supplied set of LinearMixedModel fits with identical kwargs and
# returns them sorted ascending by cAIC (best first) in an AnocaicTable.

"""
    anocaic(m::LinearMixedModel, rest::LinearMixedModel...; method=:auto, hessian=:analytic, nboot=nothing, sigmapenalty=1)
        -> AnocaicTable

Score a user-supplied set of fitted Gaussian linear mixed models by conditional AIC and
return them sorted ascending by `cAIC` (best first). Port of `cAIC4`'s `anocAIC`.

Each model is scored with [`caic`](@ref) using **identical** keyword arguments, so all
entries in the returned [`AnocaicTable`](@ref) are scored consistently (same df method,
B-source, σ-penalty, and REML setting). Mixed REML/ML inputs are rejected before scoring.

Requires at least one model (zero-model calls result in a `MethodError`).

# Arguments
- `m, rest...`: one or more fitted `LinearMixedModel` objects of the same float type.
- All keyword arguments are forwarded to [`caic`](@ref) unchanged for every model.

# Returns
- An [`AnocaicTable`](@ref) with `results` sorted ascending by `cAIC` (best first) and
  `inputorder` recording the original 1-based position of each model in the argument list.

# Throws
- `ArgumentError` if the models have inconsistent REML settings.
- Any error raised by [`caic`](@ref) (e.g. `ArgumentError` for an unknown `method`).

# Example
```jldoctest
julia> using MixedModels, ConditionalAIC

julia> data = MixedModels.dataset(:sleepstudy);

julia> m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), data; REML=false, progress=false);

julia> m2 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), data; REML=false, progress=false);

julia> t = anocaic(m1, m2);

julia> t.results[1].caic ≤ t.results[2].caic
true
```
"""
function anocaic(
    m1::LinearMixedModel{T},
    rest::LinearMixedModel{T}...;
    method::Symbol=:auto,
    hessian::Symbol=:analytic,
    nboot::Union{Int,Nothing}=nothing,
    sigmapenalty::Integer=1,
) where {T}
    ms = (m1, rest...)

    reml0 = MMInternals.reml(m1)
    for (i, m) in enumerate(rest)
        MMInternals.reml(m) == reml0 || throw(
            ArgumentError(
                "all models must share the same REML setting; model $(i + 1) has REML=" *
                "$(MMInternals.reml(m)) but model 1 has REML=$reml0",
            ),
        )
    end

    M = LinearMixedModel{T}
    n = length(ms)
    scored = Vector{CAICResult{T,M}}(undef, n)
    for (i, m) in enumerate(ms)
        scored[i] = caic(
            m; method=method, hessian=hessian, nboot=nboot, sigmapenalty=sigmapenalty
        )
    end

    perm = sortperm(scored; by=r -> r.caic)
    return AnocaicTable{T,M}(scored[perm], perm)
end
