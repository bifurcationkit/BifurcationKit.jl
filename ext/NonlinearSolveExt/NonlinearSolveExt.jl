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
    function check_matrix(J)
        if !(J isa AbstractMatrix)
            throw(ArgumentError("a dense linear solver needs a Jacobian which is an `AbstractMatrix`, got $(typeof(J)). Use a matrix-free linear solver such as `KrylovJL_GMRES`."))
        end
        return J
    end

    # The Jacobian `J` (and `∂ₚF` for PALC) at the iterate `point`, built once per iterate however many products a matrix-free linear solver takes
    struct IterateJacobian{Tu, TJ, Tdp}
        point::Tu
        J::TJ
        dFdp::Tdp
    end

    function remembered(slot::Base.RefValue{Any}, point)
        known = slot[]
        if !isnothing(known) && known.point == point
            return known
        end
        return nothing
    end

    #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Solves of `F(x, p0) = 0`. The parameter of the NonlinearProblem is a `FixedContext`.
    struct FixedContext{Tprob, Tpar}
        prob::Tprob
        par::Tpar
        memo::Base.RefValue{Any}
    end

    FixedContext(prob, par) = FixedContext(prob, par, Ref{Any}(nothing))

    function iterate_jacobian!(ctx::FixedContext, x)
        known = remembered(ctx.memo, x)
        if !isnothing(known)
            return known
        end
        jacobian = IterateJacobian(BK._copy(x), BK.jacobian(ctx.prob, x, ctx.par), nothing)
        ctx.memo[] = jacobian
        return jacobian
    end

    fixed_residual(x, ctx::FixedContext) = BK.residual(ctx.prob, x, ctx.par)
    fixed_jacobian(x, ctx::FixedContext) = check_matrix(iterate_jacobian!(ctx, x).J)
    # the product with the Jacobian at the iterate `x`: an iterative linear solver gets it as NonlinearSolve's operator, bound to the iterate at every step
    fixed_jvp(v, x, ctx::FixedContext) = BK.apply(iterate_jacobian!(ctx, x).J, v)

    BK.corrector_jacobian(ctx::FixedContext, x) = iterate_jacobian!(ctx, x).J
    BK.corrector_matrix(::FixedContext, x, state_matrix::AbstractMatrix) = state_matrix

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
    The Jacobian `[J ∂ₚF; row' c]` of the bordered system of PALC for a Jacobian `J` which is a matrix, as an operator on `ArrayPartition(x, [p])`, the way NonlinearSolve.jl takes a Jacobian which is not a matrix. Its linear systems are solved by BorderedLinearSolve.jl (`BorderedSolve`), which only solves with `J` and never forms the bordered matrix.

    - `J`: the Jacobian `∂ₓF` of the problem, an `AbstractMatrix`,
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
        "the `IterateJacobian` of the last iterate"
        memo::Base.RefValue{Any}
    end

    function PALCCache(prob, par, state, dotθ, θ)
        τ0 = state.τ
        row = BK._copy(τ0.u) .* θ
        applyξu! = BK._get_apply_dot(dotθ)
        if !isnothing(applyξu!)
            applyξu!(row)
        end
        return PALCCache(prob, par, state, dotθ, θ, row, (one(θ) - θ) * τ0.p, Ref{Any}(nothing))
    end

    function palc_residual(w::ArrayPartition, ctx::PALCCache)
        x, p = palc_state(w), palc_parameter(w)
        z0, τ0 = BK.getsolution(ctx.state), ctx.state.τ
        Fx = BK.residual(ctx.prob, x, BK.set(ctx.par, BK.getlens(ctx.prob), p))
        Nx = BK.arc_length_eq(ctx.dotθ, x, z0.u, p - z0.p, τ0.u, τ0.p, ctx.θ, ctx.state.ds)
        return ArrayPartition(Fx, [Nx])
    end

    # `∂ₚF` from the jet of the problem (`R01` of `BifurcationProblem`, automatic differentiation by default); problems without a jet (minimally augmented, periodic orbits) by finite differences
    palc_dFdp(prob::BK.AbstractAllJetBifProblem, x, par) = BK.R01(prob, x, par)
    palc_dFdp(prob, x, par) = BK.R01(BK.FiniteDifferences(), prob, x, par)

    function iterate_jacobian!(ctx::PALCCache, w::ArrayPartition)
        known = remembered(ctx.memo, w)
        if !isnothing(known)
            return known
        end
        x, p = palc_state(w), palc_parameter(w)
        par = BK.set(ctx.par, BK.getlens(ctx.prob), p)
        jacobian = IterateJacobian(BK._copy(w), BK.jacobian(ctx.prob, x, par), palc_dFdp(ctx.prob, x, par))
        ctx.memo[] = jacobian
        return jacobian
    end

    function palc_jacobian(w::ArrayPartition, ctx::PALCCache)
        (; J, dFdp) = iterate_jacobian!(ctx, w)
        return BorderedJacobian(check_matrix(J), dFdp, ctx.row, ctx.c)
    end

    # A matrix-free `J` (a function `dx -> J dx`) gives NonlinearSolve the product with `[J ∂ₚF; row' c]` at the iterate `w`, which NonlinearSolve binds to the iterate at every step:
    # each step is a Newton step, and `jacobian_reuse` keeps only the preconditioner, built by the `precs` of the iterative linear solver for the whole bordered system.
    # The iterative linear solver solves the whole bordered system: BorderedLinearSolve keeps its inner solver's cache for one operator, so each iterate would rebuild the preconditioner.
    function palc_jvp(v::ArrayPartition, w::ArrayPartition, ctx::PALCCache)
        (; J, dFdp) = iterate_jacobian!(ctx, w)
        vx, vp = palc_state(v), palc_parameter(v)
        return ArrayPartition(BK.apply(J, vx) .+ dFdp .* vp, [LA.dot(ctx.row, vx) + ctx.c * vp])
    end

    BK.corrector_jacobian(ctx::PALCCache, w::ArrayPartition) = iterate_jacobian!(ctx, w).J

    function BK.corrector_matrix(ctx::PALCCache, w::ArrayPartition, state_matrix::AbstractMatrix)
        dFdp = iterate_jacobian!(ctx, w).dFdp
        return vcat(hcat(state_matrix, vec(dFdp)), hcat(LA.adjoint(vec(ctx.row)), ctx.c))
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

    # `matrix`: the Jacobian is the operator `jacobian(u0, ctx)`, given as `jac_prototype`; otherwise NonlinearSolve takes `jacobian` for a dense linear solver and builds its own operator from `jvp` for an iterative one
    function solve_cache!(slot::Base.RefValue{Any}, corrector::NonlinearSolveCorrector, alg, residual, jacobian, jvp, u0, ctx, options::NewtonPar, norm, bounds, matrix::Bool)
        key = CacheKey(options.tol, options.max_iterations, length(u0), bounds)
        if !isnothing(slot[]) && slot[].key == key
            cache = slot[].cache
            # only passed when asked, so a NonlinearSolve without the keyword works with the default
            if corrector.reuse_jacobian
                SciMLBase.reinit!(cache, u0; p = ctx, reuse_jacobian = true)
            else
                SciMLBase.reinit!(cache, u0; p = ctx)
            end
        else
            if matrix
                f = SciMLBase.NonlinearFunction{false}(residual; jac = jacobian, jac_prototype = jacobian(u0, ctx))
            else
                f = SciMLBase.NonlinearFunction{false}(residual; jac = jacobian, jvp)
            end
            problem = SciMLBase.NonlinearProblem(f, u0, ctx; something(bounds, (;))...)
            cache = SciMLBase.init(problem, alg;
                abstol = options.tol,
                reltol = zero(options.tol),
                maxiters = options.max_iterations,
                termination_condition = NonlinearSolveBase.AbsNormTerminationMode(norm),
                corrector.solve_kwargs...)
            slot[] = (; key, cache, matrix)
        end
        return SciMLBase.solve!(cache)
    end

    # whether the Jacobian of the problem is a matrix, which PALC borders with BorderedLinearSolve; read once, at the first corrector
    function matrix_jacobian(slot::Base.RefValue{Any}, w0::ArrayPartition, ctx::PALCCache)
        if isnothing(slot[])
            return BK.corrector_jacobian(ctx, w0) isa AbstractMatrix
        end
        return slot[].matrix
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
        sol = solve_cache!(corrector.fixed_cache, corrector, corrector.alg, fixed_residual, fixed_jacobian, fixed_jvp, x0, ctx, options, normN, nothing, false)
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
        matrix = matrix_jacobian(corrector.palc_cache, w0, ctx)
        if matrix
            alg, jvp = check_palc_algorithm(corrector), nothing
        else
            alg, jvp = corrector.alg, palc_jvp
        end
        sol = solve_cache!(corrector.palc_cache, corrector, alg, palc_residual, palc_jacobian, jvp, w0, ctx, contparams.newton_options, bnorm, palc_bounds(alg, w0, p_min, p_max), matrix)
        x, p = BK._copy(palc_state(sol.u)), palc_parameter(sol.u)
        resend = bnorm(sol.resid)
        residuals = [res, resend]
        step = sol.stats.nsteps
        flag = SciMLBase.successful_retcode(sol) & (resend < tol) &
               callback((; x, res_f = palc_state(sol.resid), residual = resend, step, contparams, p, residuals, options); fromNewton = false, kwargs...)
        return NonLinearSolution(BorderedArray(x, p), prob, residuals, flag, step, sol.stats.nsolve)
    end
end
