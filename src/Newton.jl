@with_kw mutable struct NewtonPar{T, S <: LinearSolver, E <: EigenSolver}
	tol::T			 = 1e-10
	maxIter::Int  	 = 50
	alpha::T         = 1.0        # damping
	almin::T         = 0.001      # minimal damping
	verbose          = false
	linesearch       = false
	linsolve::S 	 = Default()
	eigsolve::E 	 = Default_eig()
end

# this function is used to simplify calls to NewtonPar
function NewtonPar(; kwargs...)
	if haskey(kwargs, :linsolve)
		tls = typeof(kwargs[:linsolve])
	else
		tls = typeof(Default())
	end
	if haskey(kwargs, :eigsolve)
		tes = typeof(kwargs[:eigsolve])
	else
		tes = typeof(Default_eig())
	end
	return NewtonPar{Float64, tls, tes}(;kwargs...)
end

@with_kw mutable struct ContinuationPar{T, S <: LinearSolver, E <: EigenSolver}
	# parameters for arclength continuation
	s0::T		= 0.01
	dsmin::T	= 0.001
	dsmax::T	= 0.02
	ds::T		= 0.001
	dsgrow::T	= 1.1

	# parameters for scaling arclength step size
	theta::T              = 0.5 # parameter in the dot product used for the extended system
	doArcLengthScaling    = false
	gGoal::T              = 0.5
	gMax::T               = 0.8
	thetaMin::T           = 1.0e-3
	isFirstRescale        = true
	a::T                  = 0.5  # aggressiveness factor
	tangentFactorExponent::T = 1.5

	# parameters bound
	pMin::T	= -1.0
	pMax::T	=  1.0

	# maximum number of continuation steps
	maxSteps       = 100

	# Newton solver parameters
	finDiffEps::T  = 1e-9 		#constant for finite differences
	newtonOptions::NewtonPar{T, S, E} = NewtonPar{T, S, E}()
	optNonlinIter  = 5

	save = false 				# save to file?

	# parameters for eigenvalues
 	computeEigenValues = false
	shift = 0.1					# shift used for eigenvalues computation
	nev = 3 					# number of eigenvalues
	save_eig_every_n_steps = 1	# what steps do we keep the eigenvectors
	save_eigenvectors	= true	# useful options because if puts a high memory pressure

	plot_every_n_steps = 3
	@assert dsmin>0
	@assert dsmax>0

	# handling bifucation points
	detect_fold = true
	detect_bifurcation = false
end

# check the logic of the parameters
function check!(contParams::ContinuationPar)
	if contParams.detect_bifurcation
		contParams.computeEigenValues = true
	end
end

# this function is to simplify calls to ContinuationPar
function ContinuationPar(; kwargs...)
	if haskey(kwargs, :newtonOptions)
		on = kwargs[:newtonOptions]
		ContinuationPar{Float64, typeof(on.linsolve), typeof(on.eigsolve)}(;kwargs...)
	else
		ContinuationPar{Float64, typeof(Default()), typeof(Default_eig())}(;kwargs...)
	end
end

"""
		newton(F, J, x0, options, normN = norm)

This is the Newton Solver for `F(x) = 0` with Jacobian `J` and initial guess `x0`. The function `normN` allows to specify a norm for the convergence criteria. It is important to set the linear solver `options.linsolve` properly depending on your problem. This solver is used to solve ``J(x)u = -F(x)`` in the Newton step. You can for example use `linsolve = Default()` which is the operator backslash: it works well for Sparse / Dense matrices. Iterative solver (GMRES) are also provided. You should implement your own solver for maximal efficiency. This is quite easy to do, have a look at `src/LinearSolver.jl`. The functions or callable which need to be passed are as follows:
- `x -> F(x)` functional whose zeros are looked for. In particular, it is not **inplace**,
- `dF(x) = x -> J(x)` compute the jacobian of `F` at `x`. It is then passed to `options.linsolve`. The Jacobian can be a matrix or an out of place function.

Simplified calls are provided, for example when `J` is not passed. It then computed with finite differences.

# Output:
- solution:
- history of residuals
- flag of convergence
- number of iterations
"""
function newton(Fhandle, Jhandle, x0, options:: NewtonPar{T}; normN = norm) where T
	# Extract parameters
	@unpack tol, maxIter, verbose, linesearch = options

	# Initialise iterations
	x = copy(x0)
	f = Fhandle(x)
	d = copy(f)

	neval = 1
	res = normN(f)
	resHist = [res]
	it = 0

	# Displaying results
	verbose && displayIteration(it, neval, res)

	# parameter for linesearch
	step_ok = true

	# Main loop
	while (res > tol) & (it < maxIter)
		J = Jhandle(x)
		d, flag, itlinear = options.linsolve(J, f)

		# Update solution: x .= x .- d
		minus!(x, d)

		copyto!(f, Fhandle(x))
		res = normN(f)

		neval += 1
		push!(resHist, res)
		it += 1

		verbose && displayIteration(it, neval, res, itlinear)
	end
	(resHist[end] > tol) && printstyled("\n--> Newton algorithm failed to converge, residual = ", res[end], color=:red)
	return x, resHist, resHist[end] < tol, it
end

# simplified call to newton when no Jacobian is passed in which case we estimate it using finiteDifferences
function newton(Fhandle, x0, options:: NewtonPar{T};kwargs...) where T
	Jhandle = u -> finiteDifferences(Fhandle, u)
	return newton(Fhandle, Jhandle, x0, options; kwargs...)
end

"""
	newtonDeflated(Fhandle::Function, Jhandle, x0, options:: NewtonPar{T}, defOp::DeflationOperator{T, vectype})

This is the deflated version of the Newton Solver. It penalises the roots saved in `defOp.roots`
"""
function newtonDeflated(Fhandle, Jhandle, x0::vectype, options:: NewtonPar{T}, defOp::DeflationOperator{T, vectype}; kwargs...) where {T, vectype}
	# we create the new functional
	deflatedPb = DeflatedProblem(Fhandle, Jhandle, defOp)

	# and its jacobian
	Jacdf = (u0, pb::DeflatedProblem, ls) -> (return (u0, pb, ls))

	# Rename parameters
	opt_def = @set options.linsolve = DeflatedLinearSolver()
	return newton(u -> deflatedPb(u),
				u-> Jacdf(u, deflatedPb, options.linsolve),
				x0,
				opt_def; kwargs...)
end

# simplified call when no Jacobian is given
function newtonDeflated(Fhandle, x0::vectype, options::NewtonPar{T}, defOp::DeflationOperator{T, vectype};kwargs...) where {T, vectype}
	Jhandle = u -> PseudoArcLengthContinuation.finiteDifferences(Fhandle, u)
	return newtonDeflated(Fhandle,  Jhandle,  x0, options,  defOp;kwargs...)
end

"""
This is the classical matrix-free Newton Solver used to solve `F(x, l) = 0` together
with the scalar condition `n(x, l) = (x - x0) * xp + (l - l0) * lp - n0`
"""
function newtonPseudoArcLength(F, Jh,
						z0::BorderedArray{vectype, T},
						tau0::BorderedArray{vectype, T},
						z_pred::BorderedArray{vectype, T},
						options::ContinuationPar{T};
						linearalgo = :bordering,
						normN = norm) where {T, vectype}
	# Extract parameters
	newtonOpts = options.newtonOptions
	@unpack tol, maxIter, verbose, alpha, almin, linesearch = newtonOpts
	@unpack theta, ds, finDiffEps = options

	N = (x, p) -> arcLengthEq(minus(x, z0.u), p - z0.p, tau0.u, tau0.p, theta, ds)

	# Initialise iterations
	x = copy(z_pred.u)
	l = z_pred.p
	x_pred = copy(x)

	# Initialise residuals
	res_f = F(x, l);  res_n = N(x, l)

	dX   = similar(res_f)
	dl   = T(0)

	# dFdl = (F(x, l + finDiffEps) - res_f) / finDiffEps
	dFdl = copy(F(x, l + finDiffEps))
	minus!(dFdl, res_f); rmul!(dFdl, T(1) / finDiffEps)

	res     = max(normN(res_f), abs(res_n))
	resHist = [res]
	it = 0

	# Displaying results
	verbose && displayIteration(it, 1, res)
	step_ok = true

	# Main loop
	while (res > tol) & (it < maxIter) & step_ok
		# copyto!(dFdl, (F(x, l + epsi) - F(x, l)) / epsi)
		copyto!(dFdl, F(x, l + finDiffEps)); minus!(dFdl, res_f); rmul!(dFdl, T(1) / finDiffEps)

		J = Jh(x, l)
		u, up, liniter = linearBorderedSolver(J, dFdl,
						tau0, res_f, res_n, theta,
						newtonOpts.linsolve,
						algo = linearalgo)

		if linesearch
			step_ok = false
			while !step_ok & (alpha > almin)
				# x_pred = x - alpha * u
				copyto!(x_pred, x)
				axpy!(-alpha, u, x_pred)

				l_pred = l - alpha * up
				copyto!(res_f, F(x_pred, l_pred))

				res_n  = N(x_pred, l_pred)
				res = max(normN(res_f), abs(res_n))

				if res < resHist[end]
					if (res < resHist[end] / 2) & (alpha < 1)
						alpha *=2
					end
					step_ok = true
					copyto!(x, x_pred)
					l  = l_pred
				else
					alpha /= 2
				end
			end
		else
			minus!(x, u) 	# x .= x .- u
			l = l - up

			copyto!(res_f, F(x, l))

			res_n  = N(x, l)
			res = max(normN(res_f), res_n)
		end
		# Book-keeping
		push!(resHist, res)
		it += 1
		verbose && displayIteration(it, 1, res, liniter)

	end
	return BorderedArray(x, l), resHist, resHist[end] < tol, it
end
