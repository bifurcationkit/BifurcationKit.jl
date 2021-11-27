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
	tol::T           = 1e-10
	"number of Newton iterations"
	maxIter::Int64 	 = 50
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

####################################################################################################
"""
		newton(F, J, x0, p0, options::NewtonPar; normN = norm, callback = (x, f, J, res, iteration, itlinear, optionsN; kwargs...) -> true, kwargs...)

This is the Newton-Krylov Solver for `F(x, p0) = 0` with Jacobian w.r.t. `x` written `J(x, p0)` and initial guess `x0`. The function `normN` allows to specify a norm for the convergence criteria. It is important to set the linear solver `options.linsolver` properly depending on your problem. This linear solver is used to solve ``J(x, p_0)u = -F(x, p_0)`` in the Newton step. You can for example use `linsolver = DefaultLS()` which is the operator backslash: it works well for Sparse / Dense matrices. See [Linear solvers (LS)](@ref) for more informations.

# Arguments:
- `F` is a function with input arguments `(x, p)` returning a vector `r` that represents the functional and for type stability, the types of `x` and `r` should match. In particular, it is not **inplace**.
- `J` is the jacobian of `F` at `(x, p)`. It can assume two forms. Either `J` is a function and `J(x, p)` returns a `::AbstractMatrix`. In this case, the default arguments of `NewtonPar` will make `newton` work. Or `J` is a function and `J(x, p)` returns a function taking one argument `dx` and returns `dr` of the same type of `dx`. In our notation, `dr = J * dx`. In this case, the default parameters of `NewtonPar` will not work and you have to use a Matrix Free linear solver, for example `GMRESIterativeSolvers`.
- `x0` initial guess
- `p0` set of parameters to be passed to `F` and `J`
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
- solution
- history of residuals
- flag of convergence
- number of newton iterations
- total number of linear iterations

# Simplified calls
When `J` is not passed, the jacobian **matrix** is then computed with finite differences (beware of large systems of equations!). The call is as follows:

	newton(F, x0, p0, options::NewtonPar; kwargs...)

You can also pass functions which do not have parameters `x -> F(x)`, `x -> J(x)` as follows

	newton(F, J, x0, options::NewtonPar;  kwargs...)

or

	newton(F, x0, options::NewtonPar;  kwargs...)

# Example

```
julia> F(x, p) = x.^3 .- 1
julia> Jac(x, p) = spdiagm(0 => 3 .* x.^2) # sparse jacobian
julia> x0 = rand(1_000)
julia> opts = NewtonPar()
julia> sol, hist, flag, _ = newton(F, Jac, x0, nothing, opts, normN = x -> norm(x, Inf))
```

!!! tip "Other formulation"
    If you don't have parameters, you can still use `newton` as follows `newton((x,p) -> F(x), (x,p)-> J(x), x0, nothing, options)`

!!! warning "Linear solver"
    Make sure that the linear solver (Matrix-Free...) corresponds to your jacobian (Matrix-Free vs. Matrix based).
"""
function newton(Fhandle, Jhandle, x0, p0, options::NewtonPar; normN = norm, callback = cbDefault, kwargs...)
	# Extract parameters
	@unpack tol, maxIter, verbose, α, αmin, linesearch = options

	# Initialize iterations
	x = _copy(x0)
	f = Fhandle(x, p0)
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
	compute = callback(x, f, nothing, res, it, 0, options; x0 = x0, resHist = resHist, fromNewton = true, kwargs...)
	# Main loop
	while (res > tol) && (it < maxIter) && compute
		J = Jhandle(x, p0)
		d, _, itlinear = options.linsolver(J, f)
		itlineartot += sum(itlinear)

		# Update solution: x .= x .- d
		minus!(x, d)

		f = Fhandle(x, p0)
		res = normN(f)

		push!(resHist, res)
		it += 1

		verbose && displayIteration(it, res, itlinear)

		compute = callback(x, f, J, res, it, itlinear, options; x0 = x0, resHist = resHist, fromNewton = true, kwargs...)
	end
	((resHist[end] > tol) && verbose) && @error("\n--> Newton algorithm failed to converge, residual = $(res[end])")
	flag = (resHist[end] < tol) & callback(x, f, nothing, res, it, nothing, options; x0 = x0, resHist = resHist, fromNewton = true, kwargs...)
	verbose && displayIteration(0, res, 0, true) # display last line of the table
	return x, resHist, flag, it, itlineartot
end

# simplified call to newton when no Jacobian is passed in which case we estimate it using finiteDifferences
function newton(Fhandle, x0, p0, options::NewtonPar; kwargs...)
	Jhandle = (u, p) -> finiteDifferences(z -> Fhandle(z, p), u)
	return newton(Fhandle, Jhandle, x0, p0, options; kwargs...)
end


# default callback
cbDefault(x, f, J, res, it, itlinear, options; k...) = true

# newton callback to limit residual
"""
    cb = cbMaxNorm(maxres)

Create a callback used to reject residals larger than `cb.maxres` in the Newton iterations. See docs for [`newton`](@ref).
"""
struct cbMaxNorm{T}
	maxres::T
end
(cb::cbMaxNorm)(x, f, J, res, it, itlinear, options; k...) = (return res < cb.maxres)
