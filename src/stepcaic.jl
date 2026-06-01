# Conditional stepwise random-effects selection (M4): the greedy controller and its result types.
# The candidate *enumeration* this driver scores — the combinatorial neighbourhood of a fitted
# model's RE structure — lives in `stepcaic_candidates.jl` (`backwardcandidates`/`forwardcandidates`
# and their `cnms`-form helpers). This file holds the result types (`ScoredCandidate`/`StepRecord`/
# `StepcaicOptions`/`StepcaicResult`) and the greedy `stepcaic` driver shared by the LMM and GLMM
# methods (the `_runstepcaic` core plus the two public `stepcaic` entry points).
#
# See docs/math/0008-stepcaic-search.md §4 for the driver spec and §5.1 for the result types.

# ── the greedy controller (docs/math/0008 §4.1) and its result types (§5.1) ───
# The driver scores the input, then walks the candidate neighbourhood greedily, accepting the
# minimum-cAIC neighbour while it does not increase the cAIC (the `≤` rule). This is the faithful
# backward, non-`both` subset of `cAIC4`'s `stepcAIC` (#40 walking skeleton).

"""
    ScoredCandidate{T}

One scored neighbour of a [`stepcaic`](@ref) step — the candidate random-effects structure, its
conditional AIC, and the bias-corrected effective degrees of freedom ρ (`dof`) that AIC was
penalised by (the `cAIC`/`df` pair `cAIC4`'s `stepcAIC` trace/`aicTab` prints per candidate).
`spec === nothing` marks the `lm`/`glm` terminal node (no `RESpec`). Part of a
[`StepRecord`](@ref).
"""
struct ScoredCandidate{T<:AbstractFloat}
    spec::Union{RESpec,Nothing}
    caic::T
    dof::T
end

"""
    StepRecord{T}

One greedy step of a [`stepcaic`](@ref) search — its direction, the incumbent cAIC at the start
of the step, **every** candidate scored that step ([`ScoredCandidate`](@ref)), the `argmin`
index into them, and whether the best candidate was accepted (`minCAIC ≤ incumbentcaic`). The
structured analogue of `cAIC4`'s printed `trace` — the search path of the greedy walk.
"""
struct StepRecord{T<:AbstractFloat}
    direction::Symbol
    incumbentcaic::T
    candidates::Vector{ScoredCandidate{T}}
    bestindex::Int
    accepted::Bool
end

"""
    StepcaicOptions

The resolved [`stepcaic`](@ref) options retained for provenance (`cAIC4`'s call record analogue).
The forward / `both` enumeration options (`groupcandidates`, `slopecandidates`, `maxslopes`,
`useacross`) carry the forward arc's resolved settings; they are empty/defaulted for a
pure backward run. `skipnonconverged` records whether non-converged candidates were excluded from
the comparison (the `cAIC4` `calcNonOptimMod` analogue). `keep` is not retained here — it is a
[`RESpec`](@ref) floor threaded into the backward enumeration, not a scalar provenance field.
"""
struct StepcaicOptions
    direction::Symbol
    selectcorrelation::Bool
    allownointercept::Bool
    steps::Int
    savedmodels::Int
    skipnonconverged::Bool
    groupcandidates::Vector{Symbol}
    slopecandidates::Vector{Symbol}
    maxslopes::Int
    useacross::Bool
end

"""
    StepcaicResult{T<:AbstractFloat,M<:RegressionModel}

The result of a conditional stepwise search ([`stepcaic`](@ref)).

- `selected::CAICResult{T,M}` — the conditional-AIC score of the selected model.
- `model::M` — the selected fitted model (a `MixedModel`, or the `lm`/`glm` terminal at the
  bottom of the lattice).
- `path::Vector{StepRecord{T}}` — the per-step search trace, in order (the search path,
  replacing `cAIC4`'s printed `trace`).
- `saved::Vector{CAICResult{T}}` — the k-best scores (`savedmodels`): the distinct models scored
  across the whole search, deduplicated by random-effects structure and ranked by cAIC ascending
  (the selected model is the first element). The default `savedmodels = 1` carries the selected
  score only. The element type is the `M`-erased `CAICResult{T}` (not `CAICResult{T,M}`) so a single
  ranked list can hold both the `MixedModel` candidates and the `lm`/`glm` terminal when both are
  among the best — the across-type heterogeneity of the saved set (the driver is not a hot kernel).
- `options::StepcaicOptions` — the resolved options, for provenance.

Parametric on the *selected* model type `M`; the driver is not a hot kernel, so the across-path
return-type variation (a `MixedModel` above the terminal, the `GLM.jl` terminal at it) is
acceptable.
"""
struct StepcaicResult{T<:AbstractFloat,M<:RegressionModel}
    selected::CAICResult{T,M}
    model::M
    path::Vector{StepRecord{T}}
    saved::Vector{CAICResult{T}}
    options::StepcaicOptions
end

# Total random-effects directions across all groups — the cnms `sum(lengths)` of `backwardStep`'s
# terminal guard (`:96`). `== 1` marks the single-direction terminal at which the only smaller
# neighbour is the `lm`/`glm` no-random-effects node (§0.1), distinguishing the terminal descent
# from a filter-stripped empty neighbourhood (the `minCAIC == Inf` exhausted arc).
_totaldirections(spec::RESpec) = sum(length(g.directions) for g in spec.groups; init=0)

# Structure key for the `savedmodels` dedup (`cAIC4`'s `duplicatedMers`): two scored models share
# an entry iff they have the same random-effects structure. A `RESpec` keys on its canonical
# `cnms` term-multiset (the same encoding `backwardcandidates` de-duplicates on); the `lm`/`glm`
# terminal (no random effects) keys on a reserved sentinel distinct from any `RESpec` key.
const _TERMINALKEY = "\0lm/glm-terminal"
_savedkey(spec::RESpec) = _canonkey(_cnmsform(spec))

function Base.show(io::IO, ::MIME"text/plain", r::StepcaicResult)
    println(io, "Conditional stepwise selection (stepcaic)")
    println(io, "  direction   = :", r.options.direction)
    println(io, "  steps taken = ", length(r.path))
    println(io, "  selected cAIC = ", r.selected.caic)
    print(io, "  selected df (ρ) = ", r.selected.dof)
    return nothing
end

# Shared option validation for both `stepcaic` methods: the supported directions and the
# forward/`both` call-consistency check (`R/stepcAIC.R:347–359`) — a forward or `both` run needs
# something to add (a slope candidate, a group candidate, or `useacross`). `fixEfCandidates` is out
# of scope (fixed effects held constant, §0).
function _validatestepcaic(
    direction::Symbol,
    savedmodels::Int,
    slopecandidates::Vector{Symbol},
    groupcandidates::Vector{Symbol},
    maxslopes::Int,
    useacross::Bool,
)
    direction in (:backward, :forward, :both) || throw(
        ArgumentError(
            "stepcaic direction must be :backward, :forward, or :both; got :$(direction)",
        ),
    )
    savedmodels >= 0 ||
        throw(ArgumentError("savedmodels must be ≥ 0 (0 keeps all); got $savedmodels"))
    if direction in (:forward, :both)
        maxslopes >= 1 || throw(ArgumentError("maxslopes must be ≥ 1 (got $maxslopes)"))
        (isempty(slopecandidates) && isempty(groupcandidates) && !useacross) && throw(
            ArgumentError(
                "stepcaic direction = :$(direction) cannot make forward steps without candidate " *
                "random-effect covariates: supply slopecandidates, groupcandidates, or useacross=true",
            ),
        )
    end
    return nothing
end

"""
    stepcaic(m::LinearMixedModel, data; direction=:backward, groupcandidates=Symbol[],
             slopecandidates=Symbol[], maxslopes=2, useacross=false, keep=nothing,
             selectcorrelation=false, allownointercept=false, steps=50, savedmodels=1,
             method=:auto, hessian=:analytic, nboot=nothing, sigmapenalty=1,
             rng=Random.default_rng()) -> StepcaicResult

Conditional stepwise random-effects selection guided by the conditional AIC — the greedy
controller of `cAIC4`'s `stepcAIC`, in the **backward**, **forward**, or **both** direction.

Starting from the fitted model `m`, each step enumerates the random-effects neighbours one
direction *smaller* ([`backwardcandidates`](@ref), backward) or *larger* ([`forwardcandidates`](@ref),
forward), rebuilds and refits each over `data`, and scores it with [`caic`](@ref). The minimum-cAIC
neighbour is accepted while it does not increase the cAIC (the `≤` rule); `direction = :both` starts
forward and alternates after each accepted or non-improving turn (the `improvementInBoth` /
`equalToLastStep` cascade). The search stops when no neighbour improves, the neighbourhood is
exhausted, or `steps` is reached. The selected model, its score, and the full search path are
returned in a [`StepcaicResult`](@ref).

Every candidate is scored with the **same** forwarded scoring kwargs (`method`/`hessian`/`nboot`/
`sigmapenalty`/`rng`) as the input — the consistent-scoring requirement of the search.

# Arguments
- `m`: the fitted Gaussian `LinearMixedModel` to search from.
- `data`: the data table the candidates are refit over (required — a rebuilt formula must be
  fitted, mirroring `cAIC4`'s required `data`).
- `direction`: `:backward` (default), `:forward`, or `:both`.
- `groupcandidates`, `slopecandidates`, `maxslopes`, `useacross`: the forward enumeration inputs
  (see [`forwardcandidates`](@ref)); ignored for a backward run. A forward/`both` run requires at
  least one of `slopecandidates`/`groupcandidates` (or `useacross`), else `ArgumentError`.
- `keep`: a `FormulaTerm` RE fragment (e.g. `@formula(y ~ (1 | g))`) parsed against `data` to the
  [`RESpec`](@ref) floor the backward search must not drop below (the `cAIC4` `keep` analogue);
  `nothing` for no floor. The formula's response and fixed-effects terms are ignored.
- `selectcorrelation`, `allownointercept`: the enumeration flags (see `backwardcandidates`).
- `steps`: the maximum number of search iterations (`cAIC4`'s `steps`, default `50`).
- `savedmodels`: how many of the best distinct scored models to retain in `result.saved`
  (`cAIC4`'s `numberOfSavedModels`); `1` (default) keeps only the selected model, `0` keeps all.
- `method`, `hessian`, `nboot`, `sigmapenalty`, `rng`: forwarded unchanged to every `caic` score.

# Returns
- A [`StepcaicResult`](@ref) carrying the selected score/model, the per-step `path`, and the
  resolved options.

# Throws
- `ArgumentError` for an unsupported `direction`, a forward/`both` run with no candidates, or
  `savedmodels < 0`.

# Example
```julia
m = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), sleepstudy; progress=false)
res = stepcaic(m, sleepstudy; direction=:forward, slopecandidates=[:days])  # grows the random slope
res.selected.caic
```
"""
function stepcaic(
    m::LinearMixedModel{T},
    data;
    direction::Symbol=:backward,
    groupcandidates::Vector{Symbol}=Symbol[],
    slopecandidates::Vector{Symbol}=Symbol[],
    maxslopes::Int=2,
    useacross::Bool=false,
    keep::Union{Nothing,_StatsModels.FormulaTerm}=nothing,
    selectcorrelation::Bool=false,
    allownointercept::Bool=false,
    steps::Int=50,
    savedmodels::Int=1,
    skipnonconverged::Bool=false,
    method::Symbol=:auto,
    hessian::Symbol=:analytic,
    nboot::Union{Int,Nothing}=nothing,
    sigmapenalty::Integer=1,
    rng::AbstractRNG=default_rng(),
) where {T}
    _validatestepcaic(
        direction, savedmodels, slopecandidates, groupcandidates, maxslopes, useacross
    )
    # Parse the `keep` formula fragment to the `RESpec` floor threaded into the backward search.
    keepspec = keep === nothing ? nothing : extractkeep(keep, data)

    # Quarantined formula parts + REML flag; candidates are refit with the input's objective.
    fixed = MMInternals.fixedterm(m)
    lhs = MMInternals.responseterm(m)
    reml = MMInternals.reml(m)
    score(model) = caic(
        model; method, hessian, nboot, sigmapenalty, rng
    )::CAICResult{T,LinearMixedModel{T}}
    refitcand(c) = fit(MixedModel, render(c, fixed, lhs), data; REML=reml, progress=false)
    terminalfit() = (tm=lm(_StatsModels.FormulaTerm(lhs, fixed), data); (tm, caic(tm)))
    gencands(spec, dir) =
        if dir === :forward
            forwardcandidates(
                spec;
                slopecandidates,
                groupcandidates,
                maxslopes,
                useacross,
                selectcorrelation,
            )
        else
            backwardcandidates(spec; keep=keepspec, selectcorrelation, allownointercept)
        end

    options = StepcaicOptions(
        direction,
        selectcorrelation,
        allownointercept,
        steps,
        savedmodels,
        skipnonconverged,
        groupcandidates,
        slopecandidates,
        maxslopes,
        useacross,
    )
    return _runstepcaic(
        T, m, score, refitcand, terminalfit, gencands, options; keep=keepspec
    )
end

# Flip the working direction (`R/stepcAIC.R`'s `ifelse(direction=="forward","backward","forward")`).
_flip(d::Symbol) = d === :forward ? :backward : :forward

# The mutable state of the greedy walk in [`_runstepcaic`]: the working direction, the two
# `both`-cascade latches (`improvementinboth`/`equaltolaststep`), the incumbent (its `RESpec`,
# fitted model, score, and cAIC), and the immutable `dirwasboth` flag. Holding the walk variables
# in one struct lets the repeated `both`-turn and accept transitions ([`_flipturn!`], [`_commit!`])
# mutate them in one place rather than reassigning a scatter of locals. `M`/`T` match the driver's
# model and float types; not a hot kernel, but concretely typed so the walk stays type-stable.
mutable struct WalkState{T<:AbstractFloat,M<:MixedModel}
    workdir::Symbol
    improvementinboth::Bool
    equaltolaststep::Bool
    cur_spec::RESpec
    cur_model::M
    cur_result::CAICResult{T,M}
    cAICofMod::T
    dirwasboth::Bool
end

# The `both`-run turn transition (`R/stepcAIC.R`'s direction toggle): flip the working direction
# and clear the improvement latch. This single move is taken at every `both` flip site — the
# exhausted-neighbourhood branches (`:565–571`, the drop-original `minCAIC == Inf` arc) and the
# non-improving branch F — so it lives here once.
function _flipturn!(s::WalkState)
    s.workdir = _flip(s.workdir)
    s.improvementinboth = false
    return nothing
end

# Adopt the best candidate as the new incumbent (the accepting branches C/D of `cAIC4`'s decision
# cascade): take its spec/model/result, set the incumbent cAIC to the accepted score, set the
# both-direction improvement latch, and flip the working direction on a `both` run. The only
# divergence between the two accepting branches is `improvedboth` — `true` for the improving or
# first-tie move (branch C), `false` for the plateau's second accepted move (branch D, which
# consumes the latch).
function _commit!(
    s::WalkState{T,M}, spec::RESpec, model::M, result::CAICResult{T,M}, improvedboth::Bool
) where {T,M}
    s.cur_spec = spec
    s.cur_model = model
    s.cur_result = result
    s.cAICofMod = result.caic
    s.improvementinboth = improvedboth
    s.dirwasboth && (s.workdir = _flip(s.workdir))
    return nothing
end

# The model-family-agnostic driver shared by the `LinearMixedModel` and
# `GeneralizedLinearMixedModel` `stepcaic` methods, covering all three directions (`:backward`,
# `:forward`, `:both`) — the faithful port of `cAIC4`'s `stepCAIC` decision cascade
# (`R/stepcAIC.R:565–657`) plus the forward-terminal arc (`:435`) and the `lm`/`glm` backward
# terminal (§0.1). The greedy walk, the `mergeChanges` drop-original, the terminal arcs, and the
# `savedmodels` k-best accumulation are identical across families; four pieces differ and are
# injected as closures:
#   • `score`       — `model -> CAICResult{T,M}` with the family's forwarded scoring kwargs;
#   • `refitcand`   — `spec -> M`, rebuild+refit a candidate over `data` (REML for LMM, the GLM
#                     distribution family for GLMM);
#   • `terminalfit` — `() -> (termmodel, termresult)`, fit+score the no-random-effects `lm`/`glm`
#                     terminal (called only at the cnms single-direction node with no keep floor);
#   • `gencands`    — `(spec, dir) -> Vector{RESpec}`, the per-direction candidate enumeration
#                     (`forwardcandidates` with the forward kwargs when `dir === :forward`,
#                     `backwardcandidates` with the backward kwargs otherwise).
# `direction = :both` latches `dirWasBoth = true` and starts the working direction **forward**
# (`:389`), flipping it after each accepted or non-improving turn; `improvementinboth` and
# `equaltolaststep` are the cascade's alternation/plateau guards. The return type varies across the
# terminal branch (a `MixedModel` above it, the `GLM.jl` terminal at it); the driver is not a hot
# kernel, so this is accepted (docs/math/0008 §5.1).
function _runstepcaic(
    ::Type{T},
    m::M,
    score::F,
    refitcand::G,
    terminalfit::H,
    gencands::C,
    options::StepcaicOptions;
    keep::Union{Nothing,RESpec},
) where {T,M<:MixedModel,F,G,H,C}
    savedmodels = options.savedmodels
    skipnonconverged = options.skipnonconverged

    # Working direction: `both`/`forward` start forward (`:389`); `backward` is the §4.1 skeleton.
    cur_result = score(m)
    state = WalkState{T,M}(
        options.direction in (:both, :forward) ? :forward : :backward,  # workdir
        true,                                                           # improvementinboth
        false,                                                          # equaltolaststep
        extract(m),                                                     # cur_spec
        m,                                                              # cur_model
        cur_result,                                                     # cur_result
        cur_result.caic,                                                # cAICofMod
        options.direction === :both,                                    # dirwasboth
    )

    path = StepRecord{T}[]

    # `savedmodels` (`cAIC4` `numberOfSavedModels`): accumulate every distinct scored model across
    # the whole search, dedup by structure (`_savedkey`, the `duplicatedMers` analogue, keep-first),
    # then rank ascending and keep the best `nsave` at return. `0` keeps all; `1` (default) keeps
    # only the selected model, so the accumulation is skipped entirely (`remember!` is inert).
    nsave = savedmodels == 0 ? typemax(Int) : savedmodels
    savedkeys = String[]
    savedpool = CAICResult{T}[]
    remember!(key::String, r::CAICResult) = begin
        if savedmodels != 1 && !(key in savedkeys)
            push!(savedkeys, key)
            push!(savedpool, r)
        end
        return nothing
    end
    finalsaved(selkey::String, selres::CAICResult{T}) = begin
        savedmodels == 1 && return CAICResult{T}[selres]
        selkey in savedkeys || (push!(savedkeys, selkey); push!(savedpool, selres))
        order = sortperm([r.caic for r in savedpool]; alg=MergeSort)
        CAICResult{T}[savedpool[i] for i in order[1:min(nsave, length(order))]]
    end

    result(res, model, selkey) =
        StepcaicResult(res, model, path, finalsaved(selkey, res), options)

    stepsleft = options.steps
    while stepsleft > 0
        cands = gencands(state.cur_spec, state.workdir)

        # Forward-terminal arc (`:435`): forward enumeration exhausted → return the incumbent as
        # best, *before* scoring. Forward never descends to the `lm`/`glm` terminal (it only grows).
        # Fires regardless of `dirwasboth`, so a `both` run whose forward turn yields nothing stops.
        if state.workdir === :forward && isempty(cands)
            return result(state.cur_result, state.cur_model, _savedkey(state.cur_spec))
        end

        # Backward enumeration exhausted: at the cnms single-direction node with NO keep floor the
        # sole neighbour is the `lm`/`glm` terminal (§0.1) — score it as the candidate set (`allNA`,
        # a finite cAIC). Otherwise the neighbourhood is empty (`minCAIC == Inf`): a `both` run flips
        # direction (`:565–571` branch A), a non-`both` run stops keeping the incumbent.
        terminaldescent = false
        if state.workdir === :backward && isempty(cands)
            if keep === nothing && _totaldirections(state.cur_spec) == 1
                terminaldescent = true
            elseif state.dirwasboth
                _flipturn!(state)
                continue
            else
                break
            end
        end

        stepsleft -= 1   # `R/stepcAIC.R:468` — one decrement per scoring iteration

        if terminaldescent
            termmodel, termresult = terminalfit()
            termcaic = termresult.caic
            remember!(_TERMINALKEY, termresult)
            accepted = termcaic <= state.cAICofMod
            push!(
                path,
                StepRecord{T}(
                    state.workdir,
                    state.cAICofMod,
                    ScoredCandidate{T}[ScoredCandidate{T}(
                        nothing, termcaic, termresult.dof
                    )],
                    1,
                    accepted,
                ),
            )
            # `allNA` + `≤`: accept-and-stop (branch B's `all(is.na(newSetup))` arc). Otherwise the
            # terminal does not improve: a `both` run with a prior successful turn flips once more
            # (branch F), else stop keeping the incumbent (branch G).
            accepted && return result(termresult, termmodel, _TERMINALKEY)
            if state.dirwasboth && state.improvementinboth
                _flipturn!(state)
                continue
            else
                break
            end
        end

        # `mergeChanges` drop-original: discard every candidate equal to the current model before
        # scoring (the keep re-add / a no-op enlargement can reconstitute the unchanged incumbent).
        cands = RESpec[c for c in cands if c != state.cur_spec]
        if isempty(cands)
            # `minCAIC == Inf` (branch A): flip in `both`, else stop keeping the incumbent.
            if state.dirwasboth
                _flipturn!(state)
                continue
            else
                break
            end
        end

        models = Vector{M}(undef, length(cands))
        results = Vector{CAICResult{T,M}}(undef, length(cands))
        caics = Vector{T}(undef, length(cands))
        for (i, c) in enumerate(cands)
            cm = refitcand(c)
            r = score(cm)
            models[i] = cm
            results[i] = r
            # `skipnonconverged` (the `calcNonOptimMod` analogue): a non-converged candidate is
            # excluded from the comparison — its effective cAIC is +Inf (the `NA` cAIC analogue,
            # never the argmin) and it is not retained in the `savedmodels` k-best.
            excluded = skipnonconverged && !MMInternals.converged(cm)
            caics[i] = excluded ? typemax(T) : r.caic
            excluded || remember!(_savedkey(c), r)
        end

        bestidx = argmin(caics)
        minCAIC = caics[bestidx]
        improves = minCAIC <= state.cAICofMod
        single = stepsleft == 0 || length(cands) == 1

        # `accepted` for the path record: the best candidate becomes (or would become) the incumbent
        # this step (branches B/C/D) vs. rejected (branches E/F/G). The mixed-candidate set never
        # carries the `lm`/`glm` terminal (`allNA`/`bestIsGLM` are false here), so branch B is unreachable
        # on this path — its terminal/keep-minimal arcs are the `terminaldescent`/single-candidate stops.
        accepted = improves && (!state.equaltolaststep || state.improvementinboth)
        push!(
            path,
            StepRecord{T}(
                state.workdir,
                state.cAICofMod,
                ScoredCandidate{T}[
                    ScoredCandidate{T}(cands[i], caics[i], results[i].dof) for
                    i in eachindex(cands)
                ],
                bestidx,
                accepted,
            ),
        )

        if improves && !state.equaltolaststep
            # Branch C: accept the improving (or first-tie) move. A tie latches `equaltolaststep`
            # (read against the pre-commit incumbent cAIC), and `single` returns before committing.
            minCAIC == state.cAICofMod && (state.equaltolaststep = true)
            single &&
                return result(results[bestidx], models[bestidx], _savedkey(cands[bestidx]))
            _commit!(state, cands[bestidx], models[bestidx], results[bestidx], true)
        elseif improves && state.equaltolaststep && state.improvementinboth
            # Branch D: the plateau's second accepted move (consumes `improvementinboth`).
            _commit!(state, cands[bestidx], models[bestidx], results[bestidx], false)
        elseif !improves && single && !state.dirwasboth
            # Branch E: no improvement and out of moves (non-`both`) — stop, keep incumbent.
            break
        elseif !improves && state.dirwasboth && state.improvementinboth
            # Branch F: a `both` non-improving turn after a prior success — flip and try once more.
            _flipturn!(state)
        else
            # Branch G: stop, keep the incumbent.
            break
        end
    end

    return result(state.cur_result, state.cur_model, _savedkey(state.cur_spec))
end

"""
    stepcaic(m::GeneralizedLinearMixedModel, data; direction=:backward, groupcandidates=Symbol[],
             slopecandidates=Symbol[], maxslopes=2, useacross=false, keep=nothing,
             selectcorrelation=false, allownointercept=false, steps=50, savedmodels=1,
             method=:auto, nboot=nothing, rng=Random.default_rng()) -> StepcaicResult

Conditional stepwise random-effects selection for a non-Gaussian `GeneralizedLinearMixedModel` —
the GLMM branch of `cAIC4`'s `stepcAIC`, in the **backward**, **forward**, or **both** direction.

Identical in structure to the [`LinearMixedModel`](@ref stepcaic) method: each step enumerates the
random-effects neighbours one direction *smaller* ([`backwardcandidates`](@ref)) or *larger*
([`forwardcandidates`](@ref)), rebuilds and refits each over `data` with the model's GLM
**distribution family**, and scores it with [`caic`](@ref)'s GLMM path (the GLMM bias correction).
The minimum-cAIC neighbour is accepted while it does not increase the cAIC (the `≤` rule);
`direction = :both` starts forward and alternates; a backward search bottoms out at the `glm`
terminal.

The scoring kwargs are the GLMM `caic` set — `method`/`nboot`/`rng` — and are forwarded
**unchanged** to every candidate score (the consistent-scoring requirement). The Gaussian-only
`hessian`/`sigmapenalty` kwargs are not accepted here.

# Arguments
- `m`: the fitted `GeneralizedLinearMixedModel` to search from.
- `data`: the data table candidates are refit over (required).
- `direction`: `:backward` (default), `:forward`, or `:both`.
- `groupcandidates`, `slopecandidates`, `maxslopes`, `useacross`: the forward enumeration inputs
  (see [`forwardcandidates`](@ref)); ignored for a backward run. A forward/`both` run requires at
  least one of `slopecandidates`/`groupcandidates` (or `useacross`), else `ArgumentError`.
- `keep`, `selectcorrelation`, `allownointercept`, `steps`, `savedmodels`: as in the
  `LinearMixedModel` method.
- `method`, `nboot`, `rng`: forwarded unchanged to every GLMM `caic` score.

# Returns
- A [`StepcaicResult`](@ref) carrying the selected score/model, the per-step `path`, and the
  resolved options.

# Throws
- `ArgumentError` for an unsupported `direction`, a forward/`both` run with no candidates, or a
  negative `savedmodels`.

# Example
```julia
m = fit(MixedModel, @formula(y ~ 1 + x + (1 | sub)), data, Poisson(); progress=false)
res = stepcaic(m, data; direction=:forward, groupcandidates=[:it])   # grows the crossed intercept
res.selected.caic
```
"""
function stepcaic(
    m::GeneralizedLinearMixedModel{T,D},
    data;
    direction::Symbol=:backward,
    groupcandidates::Vector{Symbol}=Symbol[],
    slopecandidates::Vector{Symbol}=Symbol[],
    maxslopes::Int=2,
    useacross::Bool=false,
    keep::Union{Nothing,_StatsModels.FormulaTerm}=nothing,
    selectcorrelation::Bool=false,
    allownointercept::Bool=false,
    steps::Int=50,
    savedmodels::Int=1,
    skipnonconverged::Bool=false,
    method::Symbol=:auto,
    nboot::Union{Int,Nothing}=nothing,
    rng::AbstractRNG=default_rng(),
) where {T,D}
    _validatestepcaic(
        direction, savedmodels, slopecandidates, groupcandidates, maxslopes, useacross
    )
    # Parse the `keep` formula fragment to the `RESpec` floor threaded into the backward search.
    keepspec = keep === nothing ? nothing : extractkeep(keep, data)

    # Quarantined formula parts + the GLM distribution family; candidates refit with that family.
    # The prior weights (binomial denominators nᵢ; empty for Poisson/Bernoulli) are reused on every
    # candidate refit and the terminal glm — without them a multi-trial Binomial candidate is a
    # different model (its trial counts are lost). `weights=`/`wts=` empty is the unweighted default,
    # so the Poisson/Bernoulli paths are unchanged.
    fixed = MMInternals.fixedterm(m)
    lhs = MMInternals.responseterm(m)
    dist = MMInternals.glmmdist(m)
    wts = MMInternals.glmmpriorweights(m)
    score(model) =
        caic(model; method, nboot, rng)::CAICResult{T,GeneralizedLinearMixedModel{T,D}}
    refitcand(c) =
        fit(MixedModel, render(c, fixed, lhs), data, dist; weights=wts, progress=false)
    terminalfit() = (
        tm=GLM.glm(_StatsModels.FormulaTerm(lhs, fixed), data, dist; wts=wts);
        (tm, caic(tm))
    )
    gencands(spec, dir) =
        if dir === :forward
            forwardcandidates(
                spec;
                slopecandidates,
                groupcandidates,
                maxslopes,
                useacross,
                selectcorrelation,
            )
        else
            backwardcandidates(spec; keep=keepspec, selectcorrelation, allownointercept)
        end

    options = StepcaicOptions(
        direction,
        selectcorrelation,
        allownointercept,
        steps,
        savedmodels,
        skipnonconverged,
        groupcandidates,
        slopecandidates,
        maxslopes,
        useacross,
    )
    return _runstepcaic(
        T, m, score, refitcand, terminalfit, gencands, options; keep=keepspec
    )
end
