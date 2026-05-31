using ConditionalAIC
using Documenter

# Doctests are self-contained (each `using`s what it needs); this just makes the
# package available to every doctest block without an explicit `using`.
DocMeta.setdocmeta!(ConditionalAIC, :DocTestSetup, :(using ConditionalAIC); recursive=true)

# Every module is listed so that (a) every `jldoctest` block in the package is
# executed by `makedocs`, and (b) `checkdocs=:exports` sees the full surface. The
# public API is documented in `api.md`; the non-exported internals (and the
# submodule docstrings) are documented in `internals.md`, so the build resolves the
# package's internal `@ref` cross-links and `checkdocs` passes with nothing hidden.
makedocs(;
    modules=[
        ConditionalAIC,
        ConditionalAIC.Numerics,
        ConditionalAIC.Loglik,
        ConditionalAIC.DofLMM,
        ConditionalAIC.DofGLMM,
        ConditionalAIC.Components,
        ConditionalAIC.MMInternals,
    ],
    authors="RMKruse <rene.marcel.kruse@protonmail.com>",
    sitename="ConditionalAIC.jl",
    format=Documenter.HTML(;
        canonical="https://RMKruse.github.io/ConditionalAIC.jl",
        edit_link="main",
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
        # The Internals page is one dense, auto-generated reference (every internal
        # symbol + the MMInternals access table) — a page developers search, not read
        # straight through. Raise the per-page size thresholds so that single page
        # does not trip the warning/error gate as more internals are documented.
        size_threshold_warn=300 * 1024,
        size_threshold=600 * 1024,
    ),
    pages=[
        "Home" => "index.md",
        "Usage guide" => "guide.md",
        "API reference" => "api.md",
        "Internals" => "internals.md",
    ],
    checkdocs=:exports,
)

deploydocs(; repo="github.com/RMKruse/ConditionalAIC.jl", devbranch="main")
