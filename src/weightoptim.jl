# The Zhang-optimal weight optimizer (docs/math/0009 §1–2, ADR-0007). A faithful,
# line-for-line Julia transcription of `cAIC4`'s `getWeights` / `.weightOptim` `solnp`
# augmented-Lagrangian SQP, isolated here so the public model-averaging surface
# (`src/averaging.jl`) is not buried in `solnp` internals. Variable names shadow the R names
# directly where possible (ADR-0007 decision 1).
#
# Pure / Level-1-testable: these functions take `(y, mu, rho, sigma_sq)` and the feasibility
# data and never touch a `MixedModels` object — the model-fitting bridge (`_zhangweightresult`)
# lives in `src/averaging.jl`. This file touches only `LinearAlgebra`, the `WeightResult` type,
# and `Base` (`time_ns`, `stack` is not used here).

using LinearAlgebra: Diagonal, I, Symmetric, cholesky, diag, dot

# Pure optimizer — the getWeights body with model-fitting bypassed (Level-1 testable).
# Transcribes getWeights.R lines 62-122 (initialization + outer loop) using the supplied
# (y, mu, rho, sigma_sq). Variable names shadow R where possible (ADR-0007 decision 1).
function _getweights_raw(
    y::Vector{T}, mu::Matrix{T}, rho::Vector{T}, sigma_sq::T
) where {T<:AbstractFloat}
    nw = length(rho)
    equB = one(T)
    lowb = zeros(T, nw)
    uppb = ones(T, nw)

    # M=1 degenerate case (docs/math/0009 §2.3): ŵ = (1), short-circuit the optimizer.
    if nw == 1
        resid = y - mu[:, 1]
        J = T(dot(resid, resid)) + 2 * sigma_sq * rho[1]
        return WeightResult{T}(ones(T, 1), J, 0.0)
    end

    # R: find_weights <- function(w){ t(y - mu %*% w) %*% (y - mu %*% w) + 2*varDF*(w %*% df) }
    find_weights = let y = y, mu = mu, sigma_sq = sigma_sq, rho = rho
        function (w::AbstractVector)
            resid = y - mu * w
            return T(dot(resid, resid)) + 2 * sigma_sq * T(dot(rho, w))
        end
    end

    # Initialization (getWeights.R lines 62-85)
    p = fill(one(T) / nw, nw)    # R: weights <- rep(1/M, M); p <- c(weights)
    funv = find_weights(p)
    eqv = sum(p) - equB
    maxit = 400
    tol = T(1e-8)
    j = funv                      # R: j <- jh <- funv
    lambda = zero(T)                   # R: lambda <- c(0) (Lagrange multiplier)
    hess = Matrix{T}(I, nw, nw)      # R: hess <- diag(nw)
    mue = T(nw)                     # R: mue <- nw (augmented-Lagrangian penalty)
    iters = 0
    targets = T[funv, eqv]

    t0 = time_ns()
    while iters < maxit
        iters += 1
        # Build scaler (getWeights.R lines 88-91)
        sc1 = min(max(abs(targets[1]), tol), one(T) / tol)
        sc2 = min(max(abs(targets[2]), tol), one(T) / tol)
        scaler = vcat(T[sc1, sc2], ones(T, nw))

        res = _weightoptim(
            p, lambda, targets, hess, mue, scaler, find_weights, equB, lowb, uppb
        )
        p = res.p
        lambda = res.y
        hess = res.hess
        mue = res.lambda

        funv = find_weights(p)
        eqv = sum(p) - equB
        targets = T[funv, eqv]

        tt = (j - targets[1]) / max(targets[1], one(T))   # R: tt <- (j - targets[1])/max(targets[1],1)
        j = targets[1]
        if abs(targets[2]) < 10 * tol      # R: if abs(constraint) < 10*tol
            mue = min(mue, tol)            #    rho <- 0 (stays 0); mue <- min(mue, tol)
        end
        if (tol + tt) <= zero(T)           # R: if (tol + tt) <= 0
            lambda = zero(T)               #    lambda <- 0
            hess = Matrix(Diagonal(diag(hess)))  # R: hess <- diag(diag(hess))
        end
        if sqrt(tt^2 + eqv^2) <= tol      # R: if sqrt(sum((c(tt,eqv))^2)) <= tol
            maxit = iters                  #    maxit <- .iters  (break)
        end
    end
    duration = (time_ns() - t0) / 1e9

    # Renormalize onto the unit simplex (DECISIONS.md 2026-05-31, deliberate divergence from
    # cAIC4). The transcribed `solnp` SQP enforces Σwᵢ = 1 only to its convergence tolerance
    # (the outer break gates on |Σp − 1| ≤ tol = 1e-8, larger if `maxit` is hit without
    # convergence), and — like `cAIC4`'s `getWeights` — returns that raw iterate. Dividing by
    # the sum makes the public weights sum to 1 to machine precision, so the model-averaged
    # effects are an exact convex combination. The objective is re-evaluated at the projected
    # weights so `WeightResult.objective == find_weights(weights)` stays exactly consistent.
    s = sum(p)
    s > zero(T) || throw(
        DomainError(
            s,
            "getweights: optimized weights sum to $s (≤ 0); cannot project onto the unit " *
            "simplex. The weight optimization did not converge to a feasible point.",
        ),
    )
    p ./= s
    j = find_weights(p)
    return WeightResult{T}(p, j, duration)
end

# Inner step of the SQP — faithful Julia transcription of cAIC4's `.weightOptim`
# (weightOptim.R). Variable names shadow R names directly (ADR-0007 decision 1).
# `find_weights` is the Mallows objective closure; `equB`, `lowb`, `uppb` are the
# feasibility data. Returns (p, y, hess, lambda) unscaled, mirroring R's `ans` list.
#
# NOTE: `rho` (augmented-penalty coefficient inside this function) is always 0 in this
# implementation — it is a dead variable in both getWeights.R and weightOptim.R.
# Kept for auditability (ADR-0007 decision 1).
function _weightoptim(
    weights_in::AbstractVector{T},   # R: weights (= p0 on entry)
    lm_in::T,                # R: lm (Lagrange multiplier)
    targets_in::Vector{T},   # R: targets [funv, eqv] (unscaled)
    hess_in::Matrix{T},      # R: hess
    lambda_in::T,            # R: lambda (augmented-Lagrangian penalty)
    scaler::Vector{T},       # R: scaler (nw+2 vector)
    find_weights,            # R: find_weights closure
    equB::T,                 # R: equB (= 1.0)
    lowb::Vector{T},         # R: lowb
    uppb::Vector{T},         # R: uppb
) where {T<:AbstractFloat}
    # ── Local constants (weightOptim.R lines 11-14) ───────────────────────────────────
    rho_aug = zero(T)       # R: rho <- 0  (augmented penalty; stays 0, see NOTE above)
    inner_maxit = 800        # R: maxit (inner loop limit)
    delta = T(1e-7)       # R: delta
    tol = T(1e-8)       # R: tol
    nw = length(weights_in)   # R: numw = length(m)
    mm = nw            # R: mm = numw

    # ── Mutable working copies (R: p0 <- weights; hess and targets are passed by value) ─
    p0 = Vector{T}(weights_in)
    hess = copy(hess_in)
    lambda = lambda_in
    targets = copy(targets_in)

    l = zeros(T, 3)
    ab = hcat(lowb, uppb)    # M×2: [lowb  uppb], R: ab <- cbind(lowb, uppb)
    st = zeros(T, 3)
    sc = zeros(T, 2)

    # ── Scale (weightOptim.R lines 26-32) ─────────────────────────────────────────────
    targets ./= scaler[1:2]
    p_sc = scaler[3:(nw + 2)]                       # p-scalers (always 1.0)
    p0 ./= p_sc
    ab ./= reshape(p_sc, :, 1)                  # divide each row i by p_sc[i]
    lm = scaler[2] * lm_in / scaler[1]          # R: lm <- scaler[2]*lm/scaler[1]
    hess .*= (p_sc * p_sc') ./ scaler[1]        # R: hess <- hess*(outer(p_sc,p_sc))/scaler[1]

    # Shared ill-conditioning bailout for the four `try`-error sites below (qr.solve / chol /
    # triangular solve / KKT solve). On any of them the current iterate is returned unscaled,
    # exactly as cAIC4's `solnp` does (ADR-0007 decision 4; DECISIONS 2026-05-31, *Ill-
    # conditioned weight fallback*). `p_local` is the iterate to return and `lambda_ret` the
    # penalty to carry out — the only details that legitimately differ across the four sites
    # (the feasibility-restoration site returns `p0_ext[1:nw]`; the three inner-loop sites
    # return `p`). `p_sc`, `hess`, and `scaler` are captured by reference; `hess` is mutated
    # in place after this point, so the closure reports its up-to-date contents.
    bailout = let p_sc = p_sc, hess = hess, scaler = scaler
        function (p_local::AbstractVector{T}, lambda_ret::T, msg::AbstractString)
            @warn msg
            return (
                p=p_local .* p_sc,
                y=zero(T),
                hess=scaler[1] .* hess ./ (p_sc * p_sc'),
                lambda=lambda_ret,
            )
        end
    end

    # ── Gradient and Jacobian via finite differences (lines 34-48) ────────────────────
    j = targets[1]
    a = zeros(T, 1, nw)                 # R: a <- matrix(0, 1, numw)
    g = zeros(T, nw)                    # R: g <- rep(0, numw)
    p = copy(p0)                        # R: p <- p0[1:numw]
    constraint = targets[2]                      # R: constraint <- targets[2]

    for i in 1:nw
        p0[i] += delta
        tmpv = p0 .* p_sc                       # R: p0[1:numw] * scaler[3:(numw+2)]
        funv = find_weights(tmpv)
        eqv = sum(tmpv) - equB
        tv = T[funv, eqv] ./ scaler[1:2]
        g[i] = (tv[1] - j) / delta
        a[1, i] = (tv[2] - constraint) / delta
        p0[i] -= delta
    end

    b = dot(vec(a), p0) - constraint          # R: b <- a %*% p0 - constraint (scalar)
    ind = -1
    l[1] = tol - abs(constraint)                 # R: l[1] <- tol - max(abs(constraint))

    # ── Feasibility restoration (lines 50-100) ────────────────────────────────────────
    if l[1] <= zero(T)
        ind = 1
        # Extend p0 and a by one element/column (slack variable)
        p0_ext = vcat(p0, one(T))               # R: p0[numw+1] <- 1
        a_ext = hcat(a, T(-constraint))        # R: a <- cbind(a, -constraint)
        cx = hcat(zeros(T, 1, nw), ones(T, 1, 1))  # R: cx <- cbind(matrix(0,1,numw), 1)
        dx_ext = ones(T, nw + 1)               # R: dx <- rep(1, numw+1)
        go = one(T)
        minit_f = 0

        while go >= tol
            minit_f += 1
            gap = hcat(p0_ext[1:mm] - ab[:, 1], ab[:, 2] - p0_ext[1:mm])  # M×2
            _sort_rows2!(gap)
            dx_ext[1:mm] = gap[:, 1]
            dx_ext[nw + 1] = p0_ext[nw + 1]

            # R: y <- try(qr.solve(t(a %*% diag(dx)), dx * t(cx)), silent=TRUE)
            A_f = (a_ext * Diagonal(dx_ext))'    # (nw+1)×1 matrix
            rhs_f = dx_ext .* vec(cx')           # nw+1 vector
            y_f = try
                A_f \ rhs_f
            catch
                return bailout(
                    p0_ext[1:nw],
                    lambda,
                    "getweights: feasibility restoration (qr.solve) failed — ill-conditioned weight problem; optimum may be non-unique.",
                )
            end

            # R: v <- dx * (dx * (t(cx) - t(a) %*% y))
            v_f = dx_ext .* (dx_ext .* (vec(cx') .- vec(a_ext' * y_f)))
            if v_f[nw + 1] > zero(T)
                z = p0_ext[nw + 1] / v_f[nw + 1]
                for i in 1:mm
                    if v_f[i] < zero(T)
                        z = min(z, -(ab[i, 2] - p0_ext[i]) / v_f[i])
                    elseif v_f[i] > zero(T)
                        z = min(z, (p0_ext[i] - ab[i, 1]) / v_f[i])
                    end
                end
                if z >= p0_ext[nw + 1] / v_f[nw + 1]
                    p0_ext .-= z .* v_f
                else
                    p0_ext .-= T(0.9) * z .* v_f
                end
                go = p0_ext[nw + 1]
                if minit_f >= 10
                    go = zero(T)
                end
            else
                go = zero(T)
            end
        end

        a = a_ext[:, 1:nw]                       # R: a <- matrix(a[,1:numw], ncol=numw)
        b = dot(vec(a), p0_ext[1:nw])            # R: b <- a %*% p0[1:numw] (scalar)
        p = p0_ext[1:nw]
    else
        p = copy(p0)
    end

    # ── Recompute targets after feasibility (lines 102-111) ───────────────────────────
    y = zero(T)
    if ind > 0
        tmpv = p .* p_sc
        funv = find_weights(tmpv)
        eqv = sum(tmpv) - equB
        targets .= T[funv, eqv] ./ scaler[1:2]
    end

    j = targets[1]
    targets[2] -= dot(vec(a), p) - b             # R: targets[2] <- targets[2] - a%*%p + b
    j = targets[1] - lm * targets[2] + rho_aug * targets[2]^2

    # ── Inner loop (BFGS + bisection line search, lines 113-262) ─────────────────────
    sx = copy(p)
    yg = copy(g)

    y_val = zero(T)   # initialized here so it is in scope after the inner loop
    minit = 0
    while minit < inner_maxit
        minit += 1

        # Gradient of augmented Lagrangian (lines 115-127)
        if ind > 0
            for i in 1:nw
                p[i] += delta
                tmpv = p .* p_sc
                funv = find_weights(tmpv)
                eqv = sum(tmpv) - equB
                tv = T[funv, eqv] ./ scaler[1:2]
                tv[2] -= dot(vec(a), p) - b
                tv_aug = tv[1] - lm * tv[2] + rho_aug * tv[2]^2
                g[i] = (tv_aug - j) / delta
                p[i] -= delta
            end
        end

        # BFGS Hessian update (lines 128-136)
        if minit > 1
            yg_d = g .- yg                       # gradient difference
            sx_d = p .- sx                       # step
            sc[1] = dot(sx_d, hess * sx_d)
            sc[2] = dot(sx_d, yg_d)
            if sc[1] * sc[2] > zero(T)
                Hsx = hess * sx_d
                hess .-= (Hsx * Hsx') ./ sc[1]
                hess .+= (yg_d * yg_d') ./ sc[2]
            end
        end
        sx = copy(p)
        yg = copy(g)

        # Barrier diagonal (lines 138-142)
        dx = fill(T(0.1), nw)
        gap = hcat(p[1:mm] - ab[:, 1], ab[:, 2] - p[1:mm])   # M×2
        _sort_rows2!(gap)
        gap1 = gap[:, 1] .+ sqrt(eps(T))        # R: gap[,1] + sqrt(.Machine$double.eps)
        dx[1:mm] = one(T) ./ gap1

        go_lm = T(-1)
        lambda /= 10                             # R: lambda <- lambda/10

        # Levenberg–Marquardt feasibility ramp (lines 145-175)
        p_trial = similar(p)
        y_val = zero(T)
        while go_lm <= zero(T)
            # R: cz <- try(chol(hess + lambda * diag(dx*dx, nw, nw)), silent=TRUE)
            H_reg = Symmetric(hess .+ lambda .* Diagonal(dx .* dx))
            cz_U = try
                cholesky(H_reg).U
            catch
                return bailout(
                    p,
                    lambda,
                    "getweights: Cholesky decomposition failed (ill-conditioned). Weights may be non-unique.",
                )
            end

            # R: cz <- try(solve(cz), silent=TRUE);  yg <- t(cz) %*% g
            # §9-compliant transcription: the inverse Cholesky factor cz = inv(cz_U) is
            # never materialised. Every downstream use is a triangular solve against the
            # factor cz_U — provably equivalent (cz' * v == cz_U' \ v, cz * v == cz_U \ v)
            # and matching cAIC4 to the same roundoff. Supersedes ADR-0007 decision (2);
            # see DECISIONS 2026-05-31.
            yg_kkt, A_kkt = try
                (cz_U' \ g, cz_U' \ a')          # R: yg <- t(cz)%*%g ;  t(cz)%*%t(a)
            catch
                return bailout(
                    p,
                    lambda,
                    "getweights: triangular solve of the Cholesky factor failed (ill-conditioned). Weights may be non-unique.",
                )
            end
            y_kkt = try
                A_kkt \ yg_kkt
            catch
                return bailout(
                    p,
                    lambda,
                    "getweights: KKT multiplier solve failed (ill-conditioned). Weights may be non-unique.",
                )
            end
            y_val = only(y_kkt)

            # R: u <- -cz %*% (yg - (t(cz) %*% t(a)) %*% y)
            u_step = -(cz_U \ (yg_kkt .- A_kkt .* y_val))   # cz * v == cz_U \ v
            p_trial .= u_step[1:nw] .+ p
            go_lm = minimum(vcat(p_trial[1:mm] - ab[:, 1], ab[:, 2] - p_trial[1:mm]))
            lambda *= 3                         # R: lambda <- 3*lambda
        end

        # ── Three-point bisection line search (lines 176-232) ─────────────────────────
        l[1] = zero(T)
        targets1 = copy(targets)
        targets2 = copy(targets)
        st[1] = j
        st[2] = j
        p1 = copy(p)              # ptt[:,1]
        p2 = copy(p)              # ptt[:,2] (midpoint, updated each bisection step)
        l[3] = one(T)
        p3 = copy(p_trial)        # ptt[:,3] (trial point)

        tmpv = p3 .* p_sc
        funv = find_weights(tmpv)
        eqv = sum(tmpv) - equB
        targets3 = T[funv, eqv] ./ scaler[1:2]
        st[3] = targets3[1]
        targets3[2] -= dot(vec(a), p3) - b
        st[3] = targets3[1] - lm * targets3[2] + rho_aug * targets3[2]^2

        go_bs = one(T)
        while go_bs > tol
            l[2] = (l[1] + l[3]) / 2
            p2 = (one(T) - l[2]) .* p .+ l[2] .* p_trial   # R: ptt[,2] <- (1-l[2])*p + l[2]*p0
            tmpv = p2 .* p_sc
            funv = find_weights(tmpv)
            eqv = sum(tmpv) - equB
            targets2 .= T[funv, eqv] ./ scaler[1:2]
            st[2] = targets2[1]
            targets2[2] -= dot(vec(a), p2) - b
            st[2] = targets2[1] - lm * targets2[2] + rho_aug * targets2[2]^2

            targetsm = maximum(st)
            if targetsm < j
                targetsn = minimum(st)
                go_bs = tol * (targetsm - targetsn) / (j - targetsm)
            end

            con1 = st[2] >= st[1]
            con2 = st[1] <= st[3] && st[2] < st[1]
            con3 = st[2] < st[1] && st[1] > st[3]

            if con1
                st[3] = st[2];
                targets3 = copy(targets2);
                l[3] = l[2];
                p3 = copy(p2)
            end
            if con2
                st[3] = st[2];
                targets3 = copy(targets2);
                l[3] = l[2];
                p3 = copy(p2)
            end
            if con3
                st[1] = st[2];
                targets1 = copy(targets2);
                l[1] = l[2];
                p1 = copy(p2)
            end

            if go_bs >= tol
                go_bs = l[3] - l[1]
            end
        end

        # ── Select best of three-point bracket (lines 233-261) ────────────────────────
        ind = 1
        targetsn = minimum(st)
        if j <= targetsn
            inner_maxit = minit                  # converged: no progress, break
        end
        reduce = (j - targetsn) / (one(T) + abs(j))
        if reduce < tol
            inner_maxit = minit
        end

        con1 = st[1] < st[2]
        con2 = st[3] < st[2] && st[1] >= st[2]
        con3 = st[1] >= st[2] && st[3] >= st[2]

        if con1
            j = st[1];
            p = copy(p1);
            targets = copy(targets1)
        end
        if con2
            j = st[3];
            p = copy(p3);
            targets = copy(targets3)
        end
        if con3
            j = st[2];
            p = copy(p2);
            targets = copy(targets2)
        end
    end  # inner loop

    # ── Unscale and return (weightOptim.R lines 263-267) ─────────────────────────────
    p_out = p .* p_sc                         # = p * 1.0 (p_sc = ones)
    y_out = scaler[1] * y_val / scaler[2]     # R: y <- scaler[1]*y/scaler[2]
    hess_out = scaler[1] .* hess ./ (p_sc * p_sc')  # = scaler[1] * hess
    return (p=p_out, y=y_out, hess=hess_out, lambda=lambda)
end

# Sort the two columns of an M×2 matrix in-place so each row is in ascending order.
# Transcribes R's `t(apply(gap, 1, FUN=function(x) sort(x)))`.
function _sort_rows2!(m::Matrix)
    for i in axes(m, 1)
        if m[i, 1] > m[i, 2]
            m[i, 1], m[i, 2] = m[i, 2], m[i, 1]
        end
    end
    return m
end
