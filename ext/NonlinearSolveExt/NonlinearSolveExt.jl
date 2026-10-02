module NonlinearSolveExt
    # Correctors of continuation by NonlinearSolve.jl, see `NonlinearSolveCorrector`.
    using BifurcationKit
    using NonlinearSolve: NonlinearSolveBase, SciMLBase
    import LinearAlgebra as LA
    const BK = BifurcationKit

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Norm of the residual `[F; N]` of the bordered system, `N` being the last component
    struct BorderedNorm{Tn}
        normN::Tn
    end
    (bn::BorderedNorm)(r) = max(bn.normN(@view(r[begin:end-1])), abs(r[end]))

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # The problem's Jacobian (and `∂ₚF` for PALC) at the last point asked for, built once for the products and the dense matrix of one iterate
    struct Linearization{Tx, TJ, Tdp}
        point::Tx
        J::TJ
        dFdp::Tdp
    end

    function check_matrix(J)
        if !(J isa AbstractMatrix)
            throw(ArgumentError("a dense linear solver needs a Jacobian which is an `AbstractMatrix`, got $(typeof(J)). Use a matrix-free linear solver such as `KrylovJL_GMRES`."))
        end
        return J
    end

    function remembered(slot::Base.RefValue{Any}, point)
        linearization = slot[]
        if !isnothing(linearization) && linearization.point == point
            return linearization
        end
        return nothing
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Solves of `F(x, p0) = 0`. The parameter `p` of the NonlinearProblem is a `FixedContext`.
    struct FixedContext{Tprob, Tpar}
        prob::Tprob
        par::Tpar
        memo::Base.RefValue{Any}
    end

    FixedContext(prob, par) = FixedContext(prob, par, Ref{Any}(nothing))

    fixed_residual(x, ctx::FixedContext) = BK.residual(ctx.prob, x, ctx.par)

    function linearization!(ctx::FixedContext, x)
        known = remembered(ctx.memo, x)
        if !isnothing(known)
            return known
        end
        linearization = Linearization(BK._copy(x), BK.jacobian(ctx.prob, x, ctx.par), nothing)
        ctx.memo[] = linearization
        return linearization
    end

    fixed_jacobian(x, ctx::FixedContext) = check_matrix(linearization!(ctx, x).J)

    fixed_jvp(v, x, ctx::FixedContext) = BK.apply(linearization!(ctx, x).J, v)

    BK.corrector_jacobian(ctx::FixedContext, x) = linearization!(ctx, x).J
    BK.corrector_matrix(::FixedContext, x, state_matrix::AbstractMatrix) = state_matrix

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Solves of the bordered system of PALC in `(x, p)`. The parameter of the NonlinearProblem is a `PALCContext`.
    # `row` and `c` are the last row `[θ A τu' (1-θ) τp]` of the bordered matrix of `MatrixBLS`.
    struct PALCContext{Tprob, Tpar, Tlens, Tdot, Tu, Tp, Tθ, Trow}
        prob::Tprob
        par::Tpar
        lens::Tlens
        dotθ::Tdot
        z0u::Tu
        z0p::Tp
        τu::Tu
        τp::Tp
        θ::Tθ
        ds::Tp
        row::Trow
        c::Tp
        memo::Base.RefValue{Any}
    end

    function PALCContext(prob, par, lens, dotθ, z0u, z0p, τu, τp, θ, ds)
        row = BK._copy(τu) .* θ
        applyξu! = BK._get_apply_dot(dotθ)
        if !isnothing(applyξu!)
            applyξu!(row)
        end
        return PALCContext(prob, par, lens, dotθ, z0u, z0p, τu, τp, θ, ds, row, (one(θ) - θ) * τp, Ref{Any}(nothing))
    end

    palc_state(w) = w[begin:end-1]

    function palc_residual(w, ctx::PALCContext)
        x, p = palc_state(w), w[end]
        Fx = BK.residual(ctx.prob, x, BK.set(ctx.par, ctx.lens, p))
        Nx = BK.arc_length_eq(ctx.dotθ, x, ctx.z0u, p - ctx.z0p, ctx.τu, ctx.τp, ctx.θ, ctx.ds)
        return vcat(Fx, Nx)
    end

    function linearization!(ctx::PALCContext, w)
        known = remembered(ctx.memo, w)
        if !isnothing(known)
            return known
        end
        x, p = palc_state(w), w[end]
        par = BK.set(ctx.par, ctx.lens, p)
        linearization = Linearization(BK._copy(w), BK.jacobian(ctx.prob, x, par), BK.R01(BK.FiniteDifferences(), ctx.prob, x, par))
        ctx.memo[] = linearization
        return linearization
    end

    # `[J ∂ₚF; row c]`, the matrix of `MatrixBLS`
    function palc_jacobian(w, ctx::PALCContext)
        (; J, dFdp) = linearization!(ctx, w)
        return bordered_matrix(ctx, check_matrix(J), dFdp)
    end

    bordered_matrix(ctx::PALCContext, J::AbstractMatrix, dFdp) = vcat(hcat(J, dFdp), hcat(LA.adjoint(ctx.row), ctx.c))

    # products of `[J ∂ₚF; row c]` with `v = (vx, vp)`
    function palc_jvp(v, w, ctx::PALCContext)
        (; J, dFdp) = linearization!(ctx, w)
        vx, vp = palc_state(v), v[end]
        return vcat(BK.apply(J, vx) .+ dFdp .* vp, LA.dot(ctx.row, vx) + ctx.c * vp)
    end

    BK.corrector_jacobian(ctx::PALCContext, w) = linearization!(ctx, w).J
    BK.corrector_matrix(ctx::PALCContext, w, state_matrix::AbstractMatrix) = bordered_matrix(ctx, state_matrix, linearization!(ctx, w).dFdp)

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # One NonlinearSolve cache is kept by the corrector between solves, rebuilt when the tolerance, the number of iterations or the size of the system change.
    struct CacheKey{T, Tb}
        tol::T
        max_iterations::Int
        length::Int
        bounds::Tb
    end

    # `p ∈ [p_min, p_max]` as the box of the last coordinate of PALC's unknown, for an algorithm that handles bounds
    function palc_bounds(corrector::NonlinearSolveCorrector, w0, p_min, p_max)
        if !SciMLBase.allowsbounds(corrector.alg)
            return nothing
        end
        lb, ub = fill(typeof(p_min)(-Inf), length(w0)), fill(typeof(p_max)(Inf), length(w0))
        lb[end], ub[end] = p_min, p_max
        return (; lb, ub)
    end

    function solve_cache!(slot::Base.RefValue{Any}, corrector::NonlinearSolveCorrector, residual, jacobian, jvp, u0, ctx, options::NewtonPar, norm, bounds = nothing)
        key = CacheKey(options.tol, options.max_iterations, length(u0), bounds)
        if !isnothing(slot[]) && first(slot[]) == key
            cache = last(slot[])
            # only passed when asked, so a NonlinearSolve without the keyword works with the default
            if corrector.reuse_jacobian
                SciMLBase.reinit!(cache, u0; p = ctx, reuse_jacobian = true)
            else
                SciMLBase.reinit!(cache, u0; p = ctx)
            end
        else
            f = SciMLBase.NonlinearFunction{false}(residual; jac = jacobian, jvp)
            problem = SciMLBase.NonlinearProblem(f, u0, ctx; something(bounds, (;))...)
            cache = SciMLBase.init(problem, corrector.alg;
                abstol = options.tol,
                reltol = zero(options.tol),
                maxiters = options.max_iterations,
                termination_condition = NonlinearSolveBase.AbsNormTerminationMode(norm),
                corrector.solve_kwargs...)
            slot[] = (key, cache)
        end
        return SciMLBase.solve!(cache)
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    function BK._newton_nonlinearsolve(corrector::NonlinearSolveCorrector,
                                       prob::BK.AbstractBifurcationProblem,
                                       x0, params0, options::NewtonPar;
                                       normN = LA.norm,
                                       callback = BK.cb_default,
                                       kwargs...)
        (; tol) = options
        x0 = BK._copy(x0)
        ctx = FixedContext(prob, params0)
        fx = BK.residual(prob, x0, params0)
        res = normN(fx)
        compute = callback((; x = x0, fx, J = nothing, residual = res, step = 0, options, x0, residuals = [res]); fromNewton = true, kwargs...)
        if !compute || res <= tol
            flag = (res < tol) & callback((; x = x0, fx, residual = res, step = 0, options, x0, residuals = [res]); fromNewton = true, kwargs...)
            return NonLinearSolution(x0, prob, [res], flag, 0, 0)
        end
        sol = solve_cache!(corrector.fixed_cache, corrector, fixed_residual, fixed_jacobian, fixed_jvp, x0, ctx, options, normN)
        resend = normN(sol.resid)
        residuals = [res, resend]
        step = sol.stats.nsteps
        flag = SciMLBase.successful_retcode(sol) & (resend < tol) &
               callback((; x = sol.u, fx = sol.resid, residual = resend, step, options, x0, residuals); fromNewton = true, kwargs...)
        return NonLinearSolution(sol.u, prob, residuals, flag, step, sol.stats.nsolve)
    end

    function BK._newton_palc_nonlinearsolve(corrector::NonlinearSolveCorrector,
                                            iter::BK.AbstractContinuationIterable,
                                            state::BK.AbstractContinuationState,
                                            dotθ = BK.getdot(iter);
                                            normN = LA.norm,
                                            callback = BK.cb_default,
                                            kwargs...)
        prob = iter.prob
        par = BK.getparams(prob)
        lens = BK.getlens(iter)
        contparams = BK.getcontparams(iter)
        z0, τ0 = BK.getsolution(state), state.τ
        (; z_pred, ds) = state
        (; tol, linsolver) = contparams.newton_options
        (; p_min, p_max) = contparams
        x_pred = z_pred.u
        if !(x_pred isa AbstractVector)
            throw(ArgumentError("`NonlinearSolveCorrector` needs the state to be an `AbstractVector`, got $(typeof(x_pred))."))
        end

        ctx = PALCContext(prob, par, lens, dotθ, z0.u, z0.p, τ0.u, τ0.p, BK.getθ(iter), ds)
        w0 = vcat(BK._copy(x_pred), z_pred.p)
        bnorm = BorderedNorm(normN)
        residual0 = palc_residual(w0, ctx)
        res = bnorm(residual0)
        residuals = [res]
        options = (; linsolver)
        compute = callback((; x = x_pred, res_f = residual0[begin:end-1], residual = res, step = 0, contparams, z0, p = z_pred.p, residuals, options); fromNewton = false, kwargs...)
        if !compute || res <= tol
            x, p = x_pred, z_pred.p
            flag = (res < tol) & callback((; x, res_f = residual0[begin:end-1], residual = res, step = 0, contparams, p, residuals, options); fromNewton = false, kwargs...)
            return NonLinearSolution(BorderedArray(BK._copy(x), p), prob, residuals, flag, 0, 0)
        end
        sol = solve_cache!(corrector.palc_cache, corrector, palc_residual, palc_jacobian, palc_jvp, w0, ctx, contparams.newton_options, bnorm, palc_bounds(corrector, w0, p_min, p_max))
        x, p = palc_state(sol.u), sol.u[end]
        resend = bnorm(sol.resid)
        residuals = [res, resend]
        step = sol.stats.nsteps
        # without bounds in `alg`, a corrector that ends outside [p_min, p_max] has failed
        inside = p_min <= p <= p_max
        flag = SciMLBase.successful_retcode(sol) & (resend < tol) & inside &
               callback((; x, res_f = sol.resid[begin:end-1], residual = resend, step, contparams, p, residuals, options); fromNewton = false, kwargs...)
        return NonLinearSolution(BorderedArray(x, p), prob, residuals, flag, step, sol.stats.nsolve)
    end
end
