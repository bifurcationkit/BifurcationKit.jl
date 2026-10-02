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
    # Solves of `F(x, p0) = 0`. The parameter `p` of the NonlinearProblem is a `FixedContext`.
    function check_matrix(J)
        if !(J isa AbstractMatrix)
            throw(ArgumentError("`NonlinearSolveCorrector` needs a Jacobian which is an `AbstractMatrix`, got $(typeof(J))."))
        end
        return J
    end

    struct FixedContext{Tprob, Tpar}
        prob::Tprob
        par::Tpar
    end

    fixed_residual(x, ctx::FixedContext) = BK.residual(ctx.prob, x, ctx.par)

    function fixed_jacobian(x, ctx::FixedContext)
        J = BK.jacobian(ctx.prob, x, ctx.par)
        check_matrix(J)
        return J
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Solves of the bordered system of PALC in `(x, p)`. The parameter of the NonlinearProblem is a `PALCContext`.
    struct PALCContext{Tprob, Tpar, Tlens, Tdot, Tu, Tp, Tθ}
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
    end

    palc_state(w) = w[begin:end-1]

    function palc_residual(w, ctx::PALCContext)
        x, p = palc_state(w), w[end]
        Fx = BK.residual(ctx.prob, x, BK.set(ctx.par, ctx.lens, p))
        Nx = BK.arc_length_eq(ctx.dotθ, x, ctx.z0u, p - ctx.z0p, ctx.τu, ctx.τp, ctx.θ, ctx.ds)
        return vcat(Fx, Nx)
    end

    # `[J ∂ₚF; θ A τu' (1-θ) τp]`, the matrix of `MatrixBLS`
    function palc_jacobian(w, ctx::PALCContext)
        x, p = palc_state(w), w[end]
        par = BK.set(ctx.par, ctx.lens, p)
        J = BK.jacobian(ctx.prob, x, par)
        check_matrix(J)
        dFdp = BK.R01(BK.FiniteDifferences(), ctx.prob, x, par)
        A = vcat(hcat(J, dFdp), hcat(LA.adjoint(ctx.τu .* ctx.θ), (one(ctx.θ) - ctx.θ) * ctx.τp))
        applyξu! = BK._get_apply_dot(ctx.dotθ)
        if !isnothing(applyξu!)
            applyξu!(@view(A[end, begin:end-1]))
        end
        return A
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # One NonlinearSolve cache is kept by the corrector between solves, rebuilt when the tolerance, the number of iterations or the size of the system change.
    struct CacheKey{T}
        tol::T
        max_iterations::Int
        length::Int
    end

    function solve_cache!(slot::Base.RefValue{Any}, corrector::NonlinearSolveCorrector, residual, jacobian, u0, ctx, options::NewtonPar, norm)
        key = CacheKey(options.tol, options.max_iterations, length(u0))
        if !isnothing(slot[]) && first(slot[]) == key
            cache = last(slot[])
            SciMLBase.reinit!(cache, u0; p = ctx)
        else
            f = SciMLBase.NonlinearFunction{false}(residual; jac = jacobian)
            problem = SciMLBase.NonlinearProblem(f, u0, ctx)
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
        sol = solve_cache!(corrector.fixed_cache, corrector, fixed_residual, fixed_jacobian, x0, ctx, options, normN)
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
        sol = solve_cache!(corrector.palc_cache, corrector, palc_residual, palc_jacobian, w0, ctx, contparams.newton_options, bnorm)
        x, p = palc_state(sol.u), sol.u[end]
        resend = bnorm(sol.resid)
        residuals = [res, resend]
        step = sol.stats.nsteps
        # PALC clamps `p` at each iteration, here a corrector ending outside has failed
        inside = p_min <= p <= p_max
        flag = SciMLBase.successful_retcode(sol) & (resend < tol) & inside &
               callback((; x, res_f = sol.resid[begin:end-1], residual = resend, step, contparams, p, residuals, options); fromNewton = false, kwargs...)
        return NonLinearSolution(BorderedArray(x, p), prob, residuals, flag, step, sol.stats.nsolve)
    end
end
