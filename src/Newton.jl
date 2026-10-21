"""
$(TYPEDEF)

Returns a variable containing parameters to affect the `newton` algorithm when solving `F(x) = 0`.

# Arguments for line search (Armijo)
- `linesearch = false`: use line search algorithm (i.e. Newton with Armijo's rule)
- `α = 1.0`: initial value of α (damping) parameter for line search algorithm
- `αmin  = 0.001`: minimal value of the damping `alpha`

!!! tip "Mutating"
    For performance reasons, we decided to use an immutable structure to hold the parameters. One can use the package `Accessors.jl` to drastically simplify the mutation of different fields. See the tutorials for examples.

# Internal fields
$(TYPEDFIELDS)
"""
@with_kw struct NewtonPar{T, L <: AbstractLinearSolver, E <: AbstractEigenSolver}
    "absolute tolerance for `F(x)`."
    tol::T = 1e-12
    "number of Newton iterations."
    max_iterations::Int64 = 25
    "display Newton iterations?"
    verbose::Bool = false
    "linear solver, must be `<: AbstractLinearSolver`."
    linsolver::L = DefaultLS()
    "eigen solver, must be `<: AbstractEigenSolver`."
    eigsolver::E = DefaultEig()
    "Active linesearch."
    linesearch::Bool = false
    "damping."
    α::T = oftype(tol, 1.0)
    "minimal damping."
    αmin::T = oftype(tol, 0.001)
    @assert 0 <= α <= 1
    @assert 0 <= tol "Tolerance must be non negative."
end

"""
Structure which holds the solution from application of Newton-Krylov algorithm to a nonlinear problem.

For example

    sol = newton(prob, NewtonPar())

# methods

- `converged(sol)` return whether the solution has converged.

# Internal fields
$(TYPEDFIELDS)
"""
struct NonLinearSolution{Tu, Tprob, Tres, Titlin}
    "solution."
    u::Tu
    "nonlinear problem, typically a `BifurcationProblem`."
    prob::Tprob
    "sequence of residuals."
    residuals::Tres
    "has algorithm converged?"
    converged::Bool
    "number of newton steps."
    itnewton::Int
    "total number of linear iterations."
    itlineartot::Titlin
end
@inline converged(sol::NonLinearSolution) = sol.converged

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function _newton(prob::AbstractBifurcationProblem, x0, params0, options::NewtonPar;
                    normN = norm,
                    callback = cb_default,
                    kwargs...)
    # Extract parameters
    (;tol, max_iterations, verbose) = options

    x = _copy(x0)
    fx = residual(prob, x, params0)

    res = normN(fx)
    residuals = [res]

    # newton step
    step = 0

    # total number of linear iterations
    itlineartot = 0

    verbose && print_nonlinear_step(step, res)

    # invoke callback before algo really starts
    compute = callback((; x, fx, J = nothing, residual = res, step, options, x0, residuals); fromNewton = true, kwargs...)

    while (step < max_iterations) && (res > tol) && _continues(compute)
        J = jacobian(prob, x, params0)
        u, cv, itlinear = options.linsolver(J, fx)
        ~cv && @debug "Linear solver for J did not converge."
        itlineartot += sum(itlinear)

        # x = x - J \ fx
        minus!!(x, u) # we use this form instead of just `x .= x .- u` to deal
        # with out-of-place functionals

        fx = residual(prob, x, params0)
        res = normN(fx)

        push!(residuals, res)
        step += 1

        verbose && print_nonlinear_step(step, res, itlinear)

        compute = callback((;x, fx, J, residual = res, step, itlinear, options, x0, residuals); fromNewton = true, kwargs...)
    end
    flag = _converged(residuals[end] < tol, callback((;x, fx, residual = res, step, options, x0, residuals); fromNewton = true, kwargs...))
    (~flag && verbose) && @error("\n──> Newton algorithm failed to converge, residual = $(residuals[end])")
    verbose && print_nonlinear_step(0, res, 0, true) # display last line of the table
    return NonLinearSolution(x, prob, residuals, flag, step, itlineartot)
end

"""
    solve(prob::AbstractBifurcationProblem, ::Newton, options::NewtonPar; normN = norm, callback = (;x, fx, J, residual, step, itlinear, options, x0, residuals; kwargs...) -> true, kwargs...)

This is the Newton-Krylov Solver for `F(x, p0) = 0` with Jacobian w.r.t. `x` written `J(x, p0)` and initial guess `x0`. It is important to set the linear solver `options.linsolver` properly depending on your problem. This linear solver is used to solve ``J(x, p_0)u = F(x, p_0)`` in the Newton step (the update then reads ``x \\leftarrow x - u``). You can for example use `linsolver = DefaultLS()` which is the operator backslash: it works well for Sparse / Dense matrices. See [Linear solvers (LS)](@ref) for more information.

# Arguments
- `prob` a `::AbstractBifurcationProblem`, typically a  [`BifurcationProblem`](@ref) which holds the vector field and its jacobian. We also refer to  [`BifFunction`](@ref) for more details.
- `options::NewtonPar` variable holding the internal parameters used by the `newton` method

# Optional Arguments
- `normN = norm` specifies a norm for the convergence criteria
- `callback` function passed by the user which is called at the end of each iteration. The default one is the following `cb_default((x, fx, J, residual, step, itlinear, options, x0, residuals); k...) = true`. Can be used to update a preconditioner for example. You can use for example `cbMaxNorm` to limit the residuals norms. If yo  want to specify your own, the arguments passed to the callback are as follows
    - `x` current solution
    - `fx` current residual
    - `J` current jacobian
    - `residual` current norm of the residual
    - `step` current newton step
    - `itlinear` number of iterations to solve the linear system
    - `options` a copy of the argument `options` passed to `newton`
    - `residuals` the history of residuals
    - `kwargs` kwargs arguments, contain your initial guess `x0`
    
    They can be used as `callback = (state;k...) -> state.residual<1` for example.

    The callback returns `true` to go on, `false` to stop, not converged, or [`NewtonConverged()`](@ref NewtonConverged) to stop, converged, whatever `options.tol`. The value of the call after the last iteration decides convergence: `true` leaves it to `residual < options.tol`, `false` refuses it and `NewtonConverged()` accepts the last iterate.
- `kwargs` arguments passed to the callback. Useful when `newton` is called from `continuation`

# Output:
- `solution::NonLinearSolution`, we refer to [`NonLinearSolution`](@ref) for more information.

!!! warning "Linear solver"
    Make sure that the linear solver (Matrix-Free...) corresponds to your jacobian (Matrix-Free vs. Matrix based).
"""
solve(prob::AbstractBifurcationProblem, ::Newton, options::NewtonPar; kwargs...) = _newton(prob, getu0(prob), getparams(prob), options::NewtonPar; kwargs...)

# default callback
cb_default(state; k...) = true

"""
    NewtonConverged()

A Newton callback's verdict that the current iterate is a solution: the iterations stop, and the call after the last one returning it marks the solution converged whatever its residual. It hosts convergence tests other than `residual < tol`, such as a tolerance that depends on the iterate. See [`newton`](@ref).
"""
struct NewtonConverged end

# whether the Newton iterations go on after the callback's verdict
_continues(verdict::Bool) = verdict
_continues(::NewtonConverged) = false

# whether a solve whose last residual is under `tol` (`below_tol`) converged, given the last call of the callback
_converged(below_tol::Bool, verdict::Bool) = below_tol & verdict
_converged(::Bool, ::NewtonConverged) = true

# the verdict of a callback whose verdict `verdict` must also pass the check `check`
_verdict_and(verdict::Bool, check::Bool) = verdict & check
_verdict_and(verdict::NewtonConverged, check::Bool) = ifelse(check, verdict, false)

"""
    cb = cbMaxNorm(maxres)
Create a callback used to reject residuals larger than `cb.maxres` in the Newton iterations. See docs for [`newton`](@ref).
"""
struct cbMaxNorm{T}
    maxres::T
end
(cb::cbMaxNorm)(state; k...) = (return state.residual < cb.maxres)

"""
    cb = cbMaxNormAndΔp(maxres, δp)
Create a callback used to reject residuals larger than `cb.maxres` or parameter step larger than `δp` in the Newton iterations. See docs for [`newton`](@ref).
"""
struct cbMaxNormAndΔp{T}
    maxres::T
    δp::T
end

function (cb::cbMaxNormAndΔp)(state; k...)
    fromnewton = get(k, :fromNewton, true)
    z0 = get(state, :z0, nothing)
    p = get(state, :p, 0)
    if fromnewton || isnothing(z0)
        return state.residual < cb.maxres
    else
        return (state.residual < cb.maxres) && (abs(z0.p - p) < cb.δp)
    end
end
