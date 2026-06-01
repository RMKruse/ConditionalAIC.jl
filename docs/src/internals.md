```@meta
CurrentModule = ConditionalAIC
```

# Internals

!!! warning "Not part of the public API"
    The symbols on this page are internal. They are documented because the
    package's reference graph links to them and because the bias-correction
    mathematics lives here, but they are **not** exported and carry **no**
    stability guarantee — they may change between releases without notice. For
    the supported surface see the [API reference](api.md).

```@docs
ConditionalAIC.ConditionalAIC
```

## Top-level internal helpers

The candidate enumeration, random-effects representation, and scoring helpers
defined directly in the `ConditionalAIC` module (the submodules below are listed
separately).

```@autodocs
Modules = [ConditionalAIC]
Public = false
```

## Numerically-stable primitives

```@autodocs
Modules = [ConditionalAIC.Numerics]
```

## Conditional log-likelihood

```@autodocs
Modules = [ConditionalAIC.Loglik]
```

## Gaussian LMM degrees of freedom

```@autodocs
Modules = [ConditionalAIC.DofLMM]
```

## GLMM degrees of freedom

```@autodocs
Modules = [ConditionalAIC.DofGLMM]
```

## Fit → components bridge

```@autodocs
Modules = [ConditionalAIC.Components]
```

## MixedModels internals quarantine

All access to `MixedModels.jl` internals is confined to this module.

```@autodocs
Modules = [ConditionalAIC.MMInternals]
```
