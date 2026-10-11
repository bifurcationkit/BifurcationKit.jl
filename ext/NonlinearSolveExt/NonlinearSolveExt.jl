module NonlinearSolveExt
    # Correctors of continuation by NonlinearSolve.jl, see `NonlinearSolveCorrector`.
    using BifurcationKit
    using NonlinearSolve: NonlinearSolveBase, GeneralizedFirstOrderAlgorithm, LinearSolve, SciMLBase
    using BorderedLinearSolve: BorderedLinearSolve, BorderedLinearProblem
    using RecursiveArrayTools: ArrayPartition
    using Accessors: @set
    import LinearAlgebra as LA
    const BK = BifurcationKit
    const SciMLOperators = SciMLBase.SciMLOperators

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # `dx -> J dx` of a matrix-free problem as an operator, the form in which NonlinearSolve and LinearSolve take it.
    # `A.u` is the state and `A.p` the parameters of the iterate, from which the `precs` of an iterative solver builds its preconditioner.
    struct ApplyJacobian{TJ}
        J::TJ
    end
    (a::ApplyJacobian)(v, u, p, t; kwargs...) = BK.apply(a.J, v)

    jacobian_operator(J::AbstractMatrix, x, par) = J
    jacobian_operator(J::SciMLOperators.AbstractSciMLOperator, x, par) = J
    jacobian_operator(J, x, par) = SciMLOperators.FunctionOperator(ApplyJacobian(J), x, similar(x); u = x, p = par, islinear = true)

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Solves of `F(x, p0) = 0`. The parameter of the NonlinearProblem is a `FixedContext`.
    struct FixedContext{Tprob, Tpar}
        prob::Tprob
        par::Tpar
    end

    fixed_residual(x, ctx::FixedContext) = BK.residual(ctx.prob, x, ctx.par)
    fixed_jacobian(x, ctx::FixedContext) = jacobian_operator(BK.jacobian(ctx.prob, x, ctx.par), x, ctx.par)

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Solves of the bordered system of PALC. The unknown is `ArrayPartition(x, [p])`: the state `x` (any `AbstractArray`) and the parameter `p` stay separate.
    palc_state(w::ArrayPartition) = w.x[1]
    palc_parameter(w::ArrayPartition) = only(w.x[2])

    # Norm of the residual `[F; N]` of the bordered system, `N` being the arclength constraint
    struct BorderedNorm{Tn}
        normN::Tn
    end
    (bn::BorderedNorm)(r::ArrayPartition) = max(bn.normN(palc_state(r)), abs(palc_parameter(r)))

    """
    The Jacobian `[J ∂ₚF; row' c]` of the bordered system of PALC, as an operator on `ArrayPartition(x, [p])`, the way NonlinearSolve.jl takes a Jacobian which is not a matrix. Its linear systems are solved by BorderedLinearSolve.jl (`BorderedSolve`), which only solves with `J` and never forms the bordered matrix.

    - `J`: the Jacobian `∂ₓF` of the problem, an `AbstractMatrix` or an operator,
    - `dFdp`: `∂ₚF`,
    - `row`: the row of the arclength constraint acting on `x`, `θ A τu` for the weighted dot product `A`,
    - `c`: the coefficient of `p` in the arclength constraint, `(1 - θ) τp`.

    It is mutable because NonlinearSolve keeps a copy of the Jacobian (`jac_prototype`), into which it copies (`copyto!`) a new one.
    """
    mutable struct BorderedJacobian{Tt, TJ, Tdp, Trow, Tc} <: SciMLOperators.AbstractSciMLOperator{Tt}
        J::TJ
        dFdp::Tdp
        row::Trow
        c::Tc
    end

    function BorderedJacobian(J, dFdp, row, c)
        Tt = promote_type(eltype(dFdp), eltype(row), typeof(c))
        return BorderedJacobian{Tt, typeof(J), typeof(dFdp), typeof(row), typeof(c)}(J, dFdp, row, c)
    end

    function Base.copyto!(dst::BorderedJacobian, src::BorderedJacobian)
        dst.J, dst.dFdp, dst.row, dst.c = src.J, src.dFdp, src.row, src.c
        return dst
    end
    Base.copy(A::BorderedJacobian) = BorderedJacobian(A.J, A.dFdp, A.row, A.c)
    Base.similar(A::BorderedJacobian) = copy(A)
    Base.size(A::BorderedJacobian) = (length(A.dFdp) + 1, length(A.dFdp) + 1)
    Base.size(A::BorderedJacobian, i::Integer) = size(A)[i]
    Base.eltype(::BorderedJacobian{Tt}) where {Tt} = Tt
    # the products with the transpose of a trust region's gradient
    Base.adjoint(A::BorderedJacobian) = BorderedJacobian(adjoint(A.J), A.row, A.dFdp, conj(A.c))
    Base.transpose(A::BorderedJacobian) = BorderedJacobian(transpose(A.J), conj(A.row), conj(A.dFdp), A.c)
    SciMLOperators.islinear(::BorderedJacobian) = true
    SciMLOperators.has_adjoint(::BorderedJacobian) = true

    function LA.mul!(y::ArrayPartition, A::BorderedJacobian, v::ArrayPartition)
        vx, vp = palc_state(v), palc_parameter(v)
        palc_state(y) .= A.J * vx .+ A.dFdp .* vp
        y.x[2] .= LA.dot(A.row, vx) + A.c * vp
        return y
    end

    function LA.mul!(y::ArrayPartition, A::BorderedJacobian, v::ArrayPartition, α::Number, β::Number)
        y .= α .* (A * v) .+ β .* y
        return y
    end

    Base.:*(A::BorderedJacobian, v::ArrayPartition) = LA.mul!(similar(v), A, v)

    bordered_problem(A::BorderedJacobian, b::ArrayPartition) = BorderedLinearProblem(A.J, A.dFdp, A.row, A.c, palc_state(b), palc_parameter(b))

    """
    The linear solver of NonlinearSolve for a `BorderedJacobian` (`LinearSolve.LinearSolveFunction(BorderedSolve(bls))`): it solves the bordered system with the algorithm `bls` of BorderedLinearSolve.jl.
    """
    struct BorderedSolve{Tbls}
        bls::Tbls
    end

    # The cache of `bls` lives in the LinearSolve cache, so the factorization of `J` (or the preconditioner of an iterative solver) is kept while NonlinearSolve reuses the Jacobian.
    function LinearSolve.init_cacheval(alg::LinearSolve.LinearSolveFunction{<:BorderedSolve}, A::BorderedJacobian, b::ArrayPartition, args...)
        return SciMLBase.init(bordered_problem(A, b), alg.solve_func.bls)
    end

    function (bs::BorderedSolve)(A::BorderedJacobian, b::ArrayPartition, u, p, isfresh, Pl, Pr, cache; kwargs...)
        # the arclength row of the current corrector (`p.p`, its `PALCCache`): a Jacobian reused across correctors carries over `J` and `∂ₚF` only
        ctx = p.p
        problem = BorderedLinearProblem(A.J, A.dFdp, ctx.row, ctx.c, palc_state(b), palc_parameter(b))
        # a new Jacobian `J` is a new object, a reused one is the one of the cache
        if A.J !== cache.prob.J
            cache.inner = SciMLBase.init(problem, bs.bls).inner
        end
        cache.prob = problem
        sol = SciMLBase.solve!(cache)
        palc_state(u) .= sol.v
        u.x[2] .= sol.σ
        return u
    end

    with_linsolve(descent::NonlinearSolveBase.NewtonDescent, linsolve) = @set descent.linsolve = linsolve
    with_linsolve(descent::NonlinearSolveBase.Dogleg, linsolve) = @set descent.newton_descent = with_linsolve(descent.newton_descent, linsolve)

    # `alg` with the linear solver of its Newton step replaced by BorderedLinearSolve's `bls`, which solves with the `linsolve` of `alg` by default
    BK._palc_algorithm(alg::GeneralizedFirstOrderAlgorithm, bls) = palc_algorithm(alg, alg.descent, bls)
    # an algorithm which can still be the corrector of Natural and `solve`
    BK._palc_algorithm(alg, bls) = nothing

    # the descents whose step solves a linear system with the Jacobian, here the bordered one of PALC
    function palc_algorithm(alg, descent::Union{NonlinearSolveBase.NewtonDescent, NonlinearSolveBase.Dogleg}, bls)
        # BEC+1 (one correction sweep) keeps the bordered solve accurate at a fold, where `J` is singular
        bls = something(bls, BorderedLinearSolve.BorderingBLS(; inner = NonlinearSolveBase.get_linear_solver(descent), k = 1))
        return @set alg.descent = with_linsolve(descent, LinearSolve.LinearSolveFunction(BorderedSolve(bls)))
    end
    palc_algorithm(alg, descent, bls) = nothing

    function check_palc_algorithm(corrector::NonlinearSolveCorrector)
        if isnothing(corrector.palc_alg)
            throw(ArgumentError("`NonlinearSolveCorrector` solves the bordered system of PALC with BorderedLinearSolve.jl in the Newton step of a first-order algorithm (`NewtonDescent` or `Dogleg` descent, as in `NewtonRaphson` or `TrustRegion`), got $(nameof(typeof(corrector.alg)))."))
        end
        return corrector.palc_alg
    end

    """
    The part of the corrector's data which does not change during a PALC corrector, the NonlinearSolve parameter of its bordered problem. The solution `z0`, the tangent `τ0` and the step `ds` are read from `state`, the continuation parameter from `prob`.
    """
    struct PALCCache{Tprob, Tpar, Tstate, Tdot, Tθ, Trow, Tc}
        prob::Tprob
        par::Tpar
        state::Tstate
        dotθ::Tdot
        θ::Tθ
        "`θ A τu` for the weighted dot product `A` of `dotθ`, the row of the arclength constraint acting on `x`"
        row::Trow
        "`(1 - θ) τp`, the coefficient of `p` in the arclength constraint"
        c::Tc
    end

    function PALCCache(prob, par, state, dotθ, θ)
        τ0 = state.τ
        row = BK._copy(τ0.u) .* θ
        applyξu! = BK._get_apply_dot(dotθ)
        if !isnothing(applyξu!)
            applyξu!(row)
        end
        return PALCCache(prob, par, state, dotθ, θ, row, (one(θ) - θ) * τ0.p)
    end

    function palc_residual(w::ArrayPartition, ctx::PALCCache)
        x, p = palc_state(w), palc_parameter(w)
        z0, τ0 = BK.getsolution(ctx.state), ctx.state.τ
        Fx = BK.residual(ctx.prob, x, BK.set(ctx.par, BK.getlens(ctx.prob), p))
        Nx = BK.arc_length_eq(ctx.dotθ, x, z0.u, p - z0.p, τ0.u, τ0.p, ctx.θ, ctx.state.ds)
        return ArrayPartition(Fx, [Nx])
    end

    function palc_jacobian(w::ArrayPartition, ctx::PALCCache)
        x, p = palc_state(w), palc_parameter(w)
        par = BK.set(ctx.par, BK.getlens(ctx.prob), p)
        J = jacobian_operator(BK.jacobian(ctx.prob, x, par), x, par)
        return BorderedJacobian(J, BK.R01(BK.FiniteDifferences(), ctx.prob, x, par), ctx.row, ctx.c)
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # One NonlinearSolve cache is kept by the corrector between solves, rebuilt when the tolerance, the number of iterations or the size of the system change.
    struct CacheKey{T, Tb}
        tol::T
        max_iterations::Int
        length::Int
        bounds::Tb
    end

    # `p ∈ [p_min, p_max]` as the box of the parameter of PALC's unknown, for an algorithm that handles bounds
    function palc_bounds(alg, w0::ArrayPartition, p_min, p_max)
        if !SciMLBase.allowsbounds(alg)
            return nothing
        end
        x = palc_state(w0)
        lb = ArrayPartition(fill(oftype(p_min, -Inf), size(x)), [p_min])
        ub = ArrayPartition(fill(oftype(p_max, Inf), size(x)), [p_max])
        return (; lb, ub)
    end

    function solve_cache!(slot::Base.RefValue{Any}, corrector::NonlinearSolveCorrector, alg, residual, jacobian, u0, ctx, options::NewtonPar, norm, bounds = nothing)
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
            f = SciMLBase.NonlinearFunction{false}(residual; jac = jacobian, jac_prototype = jacobian(u0, ctx))
            problem = SciMLBase.NonlinearProblem(f, u0, ctx; something(bounds, (;))...)
            cache = SciMLBase.init(problem, alg;
                abstol = options.tol,
                reltol = zero(options.tol),
                maxiters = options.max_iterations,
                termination_condition = NonlinearSolveBase.AbsNormTerminationMode(norm),
                corrector.solve_kwargs...)
            slot[] = (key, cache)
        end
        return SciMLBase.solve!(cache)
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        sol = solve_cache!(corrector.fixed_cache, corrector, corrector.alg, fixed_residual, fixed_jacobian, x0, ctx, options, normN)
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
        contparams = BK.getcontparams(iter)
        z0, z_pred = BK.getsolution(state), state.z_pred
        (; tol, linsolver) = contparams.newton_options
        (; p_min, p_max) = contparams
        x_pred = z_pred.u
        if !(x_pred isa AbstractArray)
            throw(ArgumentError("`NonlinearSolveCorrector` needs the state to be an `AbstractArray`, got $(typeof(x_pred))."))
        end

        ctx = PALCCache(prob, BK.getparams(prob), state, dotθ, BK.getθ(iter))
        w0 = ArrayPartition(BK._copy(x_pred), [z_pred.p])
        bnorm = BorderedNorm(normN)
        residual0 = palc_residual(w0, ctx)
        res = bnorm(residual0)
        residuals = [res]
        options = (; linsolver)
        compute = callback((; x = x_pred, res_f = palc_state(residual0), residual = res, step = 0, contparams, z0, p = z_pred.p, residuals, options); fromNewton = false, kwargs...)
        if !compute || res <= tol
            x, p = x_pred, z_pred.p
            flag = (res < tol) & callback((; x, res_f = palc_state(residual0), residual = res, step = 0, contparams, p, residuals, options); fromNewton = false, kwargs...)
            return NonLinearSolution(BorderedArray(BK._copy(x), p), prob, residuals, flag, 0, 0)
        end
        alg = check_palc_algorithm(corrector)
        sol = solve_cache!(corrector.palc_cache, corrector, alg, palc_residual, palc_jacobian, w0, ctx, contparams.newton_options, bnorm, palc_bounds(alg, w0, p_min, p_max))
        x, p = BK._copy(palc_state(sol.u)), palc_parameter(sol.u)
        resend = bnorm(sol.resid)
        residuals = [res, resend]
        step = sol.stats.nsteps
        # without bounds in `alg`, a corrector that ends outside [p_min, p_max] has failed
        inside = p_min <= p <= p_max
        flag = SciMLBase.successful_retcode(sol) & (resend < tol) & inside &
               callback((; x, res_f = palc_state(sol.resid), residual = resend, step, contparams, p, residuals, options); fromNewton = false, kwargs...)
        return NonLinearSolution(BorderedArray(x, p), prob, residuals, flag, step, sol.stats.nsolve)
    end
end
