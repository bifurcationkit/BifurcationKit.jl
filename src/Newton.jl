"""
$(TYPEDEF)

Returns a variable containing parameters to affect the `newton` algorithm when solving `F(x) = 0`.

# Arguments (with default values):
$(TYPEDFIELDS)

# Arguments for line search (Armijo)
- `linesearch = false`: use line search algorithm (i.e. Newton with Armijo's rule)
- `α = 1.0`: initial value of α (damping) parameter for line search algorithm
- `αmin  = 0.001 `: minimal value of the damping `alpha`

!!! tip "Mutating"
    For performance reasons, we decided to use an immutable structure to hold the parameters. One can use the package `Accessors.jl` to drastically simplify the mutation of different fields. See the tutorials for examples.
"""
@with_kw struct NewtonPar{T, L <: AbstractLinearSolver, E <: AbstractEigenSolver}
    "absolute tolerance for `F(x)`"
    tol::T = 1e-12
    "number of Newton iterations"
    max_iterations::Int64 = 25
    "display Newton iterations?"
    verbose::Bool = false
    "linear solver, must be `<: AbstractLinearSolver`"
    linsolver::L = DefaultLS()
    "eigen solver, must be `<: AbstractEigenSolver`"
    eigsolver::E = DefaultEig()
    linesearch::Bool = false
    α::T = convert(typeof(tol), 1.0)        # damping
    αmin::T = convert(typeof(tol), 0.001)   # minimal damping
    @assert 0 <= α <= 1
    @assert 0 <= tol "Tolerance must be non negative."
end

"""
Structure which holds the solution from application of Newton-Krylov algorithm to a nonlinear problem.

For example

    sol = newton(prob, NewtonPar())

## Fields

$(TYPEDFIELDS)

## methods

- `converged(sol)` return whether the solution has converged.
"""
struct NonLinearSolution{Tu, Tprob, Tres, Titlin}
    "solution"
    u::Tu
    "nonlinear problem, typically a `BifurcationProblem`"
    prob::Tprob
    "sequence of residuals"
    residuals::Tres
    "has algorithm converged?"
    converged::Bool
    "number of newton steps"
    itnewton::Int
    "total number of linear iterations"
    itlineartot::Titlin
end
@inline converged(sol::NonLinearSolution) = sol.converged

####################################################################################################
function _newton(prob::AbstractBifurcationProblem, x0, p0, options::NewtonPar;
                    normN = norm,
                    callback = cb_default,
                    kwargs...)
    # Extract parameters
    @unpack tol, max_iterations, verbose = options

    x = _copy(x0)
    fx = residual(prob, x, p0)
    u = _copy(fx)

    res = normN(fx)
    residuals = [res]

    # newton step
    step = 0

    # total number of linear iterations
    itlineartot = 0

    verbose && print_nonlinear_step(step, res)

    # invoke callback before algo really starts
    compute = callback((; x, fx, nothing, residual = res, step, options, x0, residuals); fromNewton = true, kwargs...)

    while (step < max_iterations) && (res > tol) && compute
        J = jacobian(prob, x, p0)
        u, cv, itlinear = options.linsolver(J, fx)
        ~cv && @debug "Linear solver for J did not converge."
        itlineartot += sum(itlinear)

        # x = x - J \ fx
        x = minus!(x, u) # we use this form instead of just `minus!(x,u)` to deal
        # with out-of-place functionals

        fx = residual(prob, x, p0)
        res = normN(fx)

        push!(residuals, res)
        step += 1

        verbose && print_nonlinear_step(step, res, itlinear)

        compute = callback((;x, fx, J, residual=res, step, itlinear, options, x0, residuals); fromNewton = true, kwargs...)
    end
    ((residuals[end] > tol) && verbose) && @error("\n──> Newton algorithm failed to converge, residual = $(residuals[end])")
    flag = (residuals[end] < tol) & callback((;x, fx, residual=res, step, options, x0, residuals); fromNewton = true, kwargs...)
    verbose && print_nonlinear_step(0, res, 0, true) # display last line of the table
    return NonLinearSolution(x, prob, residuals, flag, step, itlineartot)
end

"""
        solve(prob::AbstractBifurcationProblem, ::Newton, options::NewtonPar; normN = norm, callback = (;x, fx, J, residual, step, itlinear, options, x0, residuals; kwargs...) -> true, kwargs...)

This is the Newton-Krylov Solver for `F(x, p0) = 0` with Jacobian w.r.t. `x` written `J(x, p0)` and initial guess `x0`. It is important to set the linear solver `options.linsolver` properly depending on your problem. This linear solver is used to solve ``J(x, p_0)u = -F(x, p_0)`` in the Newton step. You can for example use `linsolver = DefaultLS()` which is the operator backslash: it works well for Sparse / Dense matrices. See [Linear solvers (LS)](@ref) for more informations.

# Arguments
- `prob` a `::AbstractBifurcationProblem`, typically a  [`BifurcationProblem`](@ref) which holds the vector field and its jacobian. We also refer to  [`BifFunction`](@ref) for more details.
- `options::NewtonPar` variable holding the internal parameters used by the `newton` method

# Optional Arguments
- `normN = norm` specifies a norm for the convergence criteria
- `callback` function passed by the user which is called at the end of each iteration. The default one is the following `cb_default((x, fx, J, residual, step, itlinear, options, x0, residuals); k...) = true`. Can be used to update a preconditionner for example. You can use for example `cbMaxNorm` to limit the residuals norms. If yo  want to specify your own, the arguments passed to the callback are as follows
    - `x` current solution
    - `fx` current residual
    - `J` current jacobian
    - `residual` current norm of the residual
    - `step` current newton step
    - `itlinear` number of iterations to solve the linear system
    - `options` a copy of the argument `options` passed to `newton`
    - `residuals` the history of residuals
    - `kwargs` kwargs arguments, contain your initial guess `x0`
- `kwargs` arguments passed to the callback. Useful when `newton` is called from `continuation`

# Output:
- `solution::NonLinearSolution`, we refer to [`NonLinearSolution`](@ref) for more information.

!!! warning "Linear solver"
    Make sure that the linear solver (Matrix-Free...) corresponds to your jacobian (Matrix-Free vs. Matrix based).
"""
solve(prob::AbstractBifurcationProblem, ::Newton, options::NewtonPar; kwargs...) = _newton(prob, getu0(prob), getparams(prob), options::NewtonPar; kwargs...)
# newton(F, J, x0, p0, options::NewtonPar; kwargs...) = newton(BifurcationProblem(F, x0, p0; J = J), options; kwargs...)

# default callback
cb_default(state; k...) = true

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
