"""
	options = NewtonPar(tol = 1e-4,...)

Returns a variable containing parameters to affect the `newton` algorithm when solving `F(x) = 0`.

# Arguments (with default values):
- `tol = 1e-10`: absolute tolerance for `F(x)`
- `maxIter = 50`: number of Newton iterations
- `verbose = false`: display Newton iterations?
- `linsolver = DefaultLS()`: linear solver, must be `<: AbstractLinearSolver`
- `eigsolver = DefaultEig()`: eigen solver, must be `<: AbstractEigenSolver`

# Arguments only used in `newtonPALC`
- `linesearch = false`: use line search algorithm
- `alpha = 1.0`: alpha (damping) parameter for line search algorithm
- `almin  = 0.001 `: minimal vslue of the damping `alpha`

!!! tip "Mutating"
    For performance reasons, we decided to use an immutable structure to hold the parameters. One can use the package `Setfield.jl` to drastically simplify the mutation of different fields. See the tutorials for examples.
"""
@with_kw struct NewtonPar{T, L <: AbstractLinearSolver, E <: AbstractEigenSolver}
	tol::T			 = 1e-10
	maxIter::Int64 	 = 50
	alpha::T         = convert(typeof(tol), 1.0)        # damping
	almin::T         = convert(typeof(tol), 0.001)      # minimal damping
	verbose::Bool    = false
	linesearch::Bool = false
	linsolver::L 	 = DefaultLS()
	eigsolver::E 	 = DefaultEig()
end

####################################################################################################
"""
		newton(F, J, x0, p0, options::NewtonPar; normN = norm, callback = (x, f, J, res, iteration, itlinear, optionsN; kwargs...) -> true, kwargs...)

This is the Newton-Krylov Solver for `F(x, p0) = 0` with Jacobian w.r.t. `x` written `J(x, p0)` and initial guess `x0`. The function `normN` allows to specify a norm for the convergence criteria. It is important to set the linear solver `options.linsolver` properly depending on your problem. This linear solver is used to solve ``J(x, p0)u = -F(x, p0)`` in the Newton step. You can for example use `linsolver = DefaultLS()` which is the operator backslash: it works well for Sparse / Dense matrices. See [Linear solvers](@ref) for more informations.

# Arguments:
- `(x, p) -> F(x, p)` functional whose zeros are looked for. In particular, it is not **inplace**,
- `dF(x, p) = (x, p) -> J(x, p)` compute the jacobian of `F` at `x`. It is then passed to `options.linsolver`. The Jacobian `J(x, p)` can be a matrix or an out-of-place function.
- `x0` initial guess
- `p0` set of parameters to be passed to `F` and `J`
- `options` variable holding the internal parameters used by the `newton` method
- `callback` function passed by the user which is called at the end of each iteration. Can be used to update a preconditionner for example. The arguments passed to the callback are as follows
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
- solution:
- history of residuals
- flag of convergence
- number of iterations

# Simplified calls
When `J` is not passed, the jacobian **matrix** is then computed with finite differences (beware of large systems of equations!). The call is as follows:

	newton(Fhandle, x0, p0, options::NewtonPar; kwargs...)

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
julia> sol, hist, flag, _ = newton(F, Jac, x0, nothing, opts, normN = x->norm(x, Inf))
```

!!! tip "Other formulation"
    If you don't have parameters, you can still use `newton` as follows `newton((x,p) -> F(x), (x,p)-> J(x), x0, nothing, options)`

!!! warning "Linear solver"
    Make sure that the linear solver (Matrix-Free...) corresponds to you jacobian (Matrix-Free vs. Matrix based).
"""
function newton(Fhandle, Jhandle, x0, p0, options::NewtonPar; normN = norm, callback = (x, f, J, res, iteration, itlinear, optionsN; kwargs...) -> true, kwargs...)
	# Extract parameters
	@unpack tol, maxIter, verbose, linesearch = options

	# Initialize iterations
	x = _copy(x0)
	f = Fhandle(x, p0)
	d = _copy(f)

	neval = 1
	res = normN(f)
	resHist = [res]
	it = 0

	# Displaying results
	verbose && displayIteration(it, neval, res)

	# invoke callback before algo really starts
	compute = callback(x, f, nothing, res, it, 0, options; x0 = x0, kwargs...)

	# Main loop
	while (res > tol) & (it < maxIter) & compute
		J = Jhandle(x, p0)
		d, _, itlinear = options.linsolver(J, f)

		# Update solution: x .= x .- d
		minus!(x, d)

		f = Fhandle(x, p0)
		res = normN(f)

		neval += 1
		push!(resHist, res)
		it += 1

		verbose && displayIteration(it, neval, res, itlinear)

		if callback(x, f, J, res, it, itlinear, options; x0 = x0, kwargs...) == false
			break
		end
	end
	((resHist[end] > tol) && verbose) && @error("\n--> Newton algorithm failed to converge, residual = $(res[end])")
	flag = (resHist[end] < tol) & callback(x, f, nothing, res, it, nothing, options; x0 = x0, kwargs...)
	return x, resHist, flag, it
end

# simplified call to newton when no Jacobian is passed in which case we estimate it using finiteDifferences
function newton(Fhandle, x0, p0, options::NewtonPar; kwargs...)
	Jhandle = (u, p) -> finiteDifferences(z -> Fhandle(z, p), u)
	return newton(Fhandle, Jhandle, x0, p0, options; kwargs...)
end
