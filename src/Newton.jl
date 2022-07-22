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
    For performance reasons, we decided to use an immutable structure to hold the parameters. One can use the package `Setfield.jl` to drastically simplify the mutation of different fields. See the tutorials for examples.
"""
@with_kw struct NewtonPar{T, L <: AbstractLinearSolver, E <: AbstractEigenSolver}
	"absolute tolerance for `F(x)`"
	tol::T			= 1e-12
	"number of Newton iterations"
	maxIter::Int64 	 = 25
	"display Newton iterations?"
	verbose::Bool    = false
	"linear solver, must be `<: AbstractLinearSolver`"
	linsolver::L 	 = DefaultLS()
	"eigen solver, must be `<: AbstractEigenSolver`"
	eigsolver::E 	 = DefaultEig()
	linesearch::Bool = false
	α::T             = convert(typeof(tol), 1.0)        # damping
	αmin::T          = convert(typeof(tol), 0.001)      # minimal damping
	@assert 0 <= α <= 1
end

"""
Structure to hold the solution from application of Newton-Krylov algorithm to a nonlinear problem.

$(TYPEDFIELDS)
"""
struct NonLinearSolution{Tu, Tprob, Tres, Titlin}
	"solution"
	u::Tu
	"nonlinear problem"
	prob::Tprob
	"sequence of residuals"
	residuals::Tres
	"has algorithm converged?"
	converged::Bool
	"number of newton iterations"
	itnewton::Int
	"total number of linear iterations"
	itlineartot::Titlin
end
@inline converged(sol::NonLinearSolution) = sol.converged

####################################################################################################
function _newton(prob::AbstractBifurcationProblem, x0, p0, options::NewtonPar; normN = norm, callback = cbDefault, kwargs...)
	# Extract parameters
	@unpack tol, maxIter, verbose, α, αmin, linesearch = options

	x = _copy(x0)
	f = residual(prob, x, p0)
	d = _copy(f)

	res = normN(f)
	resHist = [res]

	# iterations count
	it = 0

	# total number of linear iterations
	itlineartot = 0

	# Displaying results
	verbose && displayIteration(it, res)

	# invoke callback before algo really starts
	compute = callback((;x, f, nothing, res, it, options, x0, resHist); fromNewton = true, kwargs...)
	# Main loop
	while (res > tol) && (it < maxIter) && compute
		J = jacobian(prob, x, p0)
		d, _, itlinear = options.linsolver(J, f)
		itlineartot += sum(itlinear)

		# Update solution: x .= x .- d
		minus!(x, d)

		f = residual(prob, x, p0)
		res = normN(f)

		push!(resHist, res)
		it += 1

		verbose && displayIteration(it, res, itlinear)

		compute = callback((;x, f, J, res, it, itlinear, options, x0, resHist); fromNewton = true, kwargs...)
	end
	((resHist[end] > tol) && verbose) && @error("\n--> Newton algorithm failed to converge, residual = $(res[end])")
	flag = (resHist[end] < tol) & callback((;x, f, res, it, options, x0, resHist); fromNewton = true, kwargs...)
	verbose && displayIteration(0, res, 0, true) # display last line of the table
	return NonLinearSolution(x, prob, resHist, flag, it, itlineartot)
end

"""
		newton(prob::AbstractBifurcationProblem, options::NewtonPar; normN = norm, callback = (;x, f, J, res, iteration, itlinear, optionsN; kwargs...) -> true, kwargs...)

This is the Newton-Krylov Solver for `F(x, p0) = 0` with Jacobian w.r.t. `x` written `J(x, p0)` and initial guess `x0`. The function `normN` allows to specify a norm for the convergence criteria. It is important to set the linear solver `options.linsolver` properly depending on your problem. This linear solver is used to solve ``J(x, p_0)u = -F(x, p_0)`` in the Newton step. You can for example use `linsolver = DefaultLS()` which is the operator backslash: it works well for Sparse / Dense matrices. See [Linear solvers (LS)](@ref) for more informations.

# Arguments:
- `prob` a `::AbstractBifurcationProblem`, typically a  [`BifurcationProblem`](@ref) which holds the vector field and its jacobian. We also refer to  [`BifFunction`](@ref) for more details.
- `options::NewtonPar` variable holding the internal parameters used by the `newton` method
- `callback` function passed by the user which is called at the end of each iteration. The default one is the following `cbDefault(x, f, J, res, it, itlinear, options; k...) = true`. Can be used to update a preconditionner for example. You can use for example `cbMaxNorm` to limit the residuals norms. If yo  want to specify your own, the arguments passed to the callback are as follows
    - `x` current solution
    - `f` current residual
    - `J` current jacobian
    - `res` current norm of the residual
    - `iteration` current newton iteration
    - `itlinear` number of iterations to solve the linear system
    - `optionsN` a copy of the argument `options` passed to `newton`
    - `kwargs` kwargs arguments, contain your initial guess `x0`
- `kwargs` arguments passed to the callback. Useful when `newton` is called from `continuation`

# Output:
- `solution::NonLinearSolution`, we refer to [`NonLinearSolution`](@ref) for more information.

!!! warning "Linear solver"
    Make sure that the linear solver (Matrix-Free...) corresponds to your jacobian (Matrix-Free vs. Matrix based).
"""
newton(prob::AbstractBifurcationProblem, options::NewtonPar; kwargs...) = _newton(prob, getu0(prob), getParams(prob), options::NewtonPar; kwargs...)
# newton(F, J, x0, p0, options::NewtonPar; kwargs...) = newton(BifurcationProblem(F, x0, p0; J = J), options; kwargs...)

# default callback
cbDefault(state; k...) = true

"""
    cb = cbMaxNorm(maxres)
Create a callback used to reject residals larger than `cb.maxres` in the Newton iterations. See docs for [`newton`](@ref).
"""
struct cbMaxNorm{T}
	maxres::T
end
(cb::cbMaxNorm)(state; k...) = (return state.res < cb.maxres)
