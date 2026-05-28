# Result type for the scoring path. Included directly into the `cAIC` module (not a
# submodule): `CAICResult` is part of the public surface and is returned by `caic`.
# `MixedModel` is imported at the module top.

"""
    CAICResult{T<:AbstractFloat,M<:MixedModel}

The result of scoring a fitted mixed-effects model by its conditional AIC
(`cAIC = −2 ℓ_cond + 2 ρ`). Returned by [`caic`](@ref).

The fields are the public, typed accessors:

- `caic::T` — the conditional AIC value.
- `dof::T` — the bias-corrected effective degrees of freedom ρ (the penalty term).
- `condloglik::T` — the conditional log-likelihood ℓ(y | b̂, β̂, θ̂) (the `cAIC4`
  `getcondLL` quantity).
- `reducedmodel::Union{Nothing,M}` — the reduced model a singular fit was scored on, or
  `nothing` when the fit was non-singular and scored as given.
- `refit::Bool` — whether scoring was performed on a refitted reduced model.
- `method::Symbol` — the degrees-of-freedom **method** actually used (provenance), e.g.
  `:steinian`.
- `bsource::Symbol` — the Hessian **B-source** actually used (provenance), e.g.
  `:analytic`.

`method`/`bsource` record what was *resolved and run* (e.g. `method = :auto` resolves to
`:steinian` for the Gaussian family), so candidates can be checked for consistent scoring.
"""
struct CAICResult{T<:AbstractFloat,M<:MixedModel}
    caic::T
    dof::T
    condloglik::T
    reducedmodel::Union{Nothing,M}
    refit::Bool
    method::Symbol
    bsource::Symbol
end

function Base.show(io::IO, ::MIME"text/plain", r::CAICResult)
    println(io, "Conditional AIC (cAIC)")
    println(io, "  cAIC               = ", r.caic)
    println(io, "  effective df (ρ)   = ", r.dof)
    println(io, "  conditional logLik = ", r.condloglik)
    print(io, "  method = :", r.method, "   B-source = :", r.bsource)
    r.refit && print(io, "   (scored on reduced model)")
    return nothing
end
