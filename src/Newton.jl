
@with_kw mutable struct NewtonPar{T, S <: LinearSolver, E <: EigenSolver}
	tol::T   		 = 1e-10
	maxIter::Int  	 = 50
	alpha::T         = 1.0        # damping
	almin::T         = 0.001      # minimal damping
	verbose          = false
	linesearch       = false
	linsolve::S 	 = Default()
	eigsolve::E 	 = Default_eig()
end

# this function is to simplify calls to NewtonPar
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
	theta::T              = 1.0 # parameter in the dot product used for the extended system
    doArcLengthScaling    = false
    gGoal::T              = 0.5
    gMax::T               = 0.8
    thetaMin::T           = 1.0e-3
    isFirstRescale        = true
	a::T                  = 0.5  # aggressiveness factor
	tangentFactorExponent::T = 1.5

	# predictor based on ... tangent or secant?
	secant	= true
	natural = false

	# parameters bound
	pMin::T	= -1.0
	pMax::T	=  1.0

	# Newton solver parameters
	maxSteps       = 100
	finDiffEps::T  = 1e-9 		#constant for finite differences
	newtonOptions::NewtonPar{T, S, E} = NewtonPar{T, S, E}()
	optNonlinIter  = 5

	save = false 				# save to file?

	# parameters for eigenvalues
 	computeEigenValues = false
	nev = 3 					# number of eigenvalues
	save_eig_every_n_steps = 1	# what steps do we keep the eigenvalues

	plot_every_n_steps = 3
	shift = 0.1
	@assert dsmin>0
	@assert dsmax>0

	# handling bifucation points
	detect_fold = false
	detect_bifurcation = false
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

This is the Newton Solver for `F(x) = 0` with Jacobian `J` and initial guess `x0`. The function `normN` allows to specify a norm for the convergence criteria. It is important to set the linear solver `options.linsolve` properly depending on your problem. This solver is used to solve ``J(x)u = -F(x)`` in the Newton step. You can for example use `Default()` which is the operator backslash which works well for Sparse / Dense matrices. Iterative solver (GMRES) are also implemented. You should implement your own for maximal efficiency. This is quite easy to do, have a look at `src/LinearSolver.jl`

# Output:
- solution
- history of residuals
- flag of convergence
- number of iterations
"""
function newton(Fhandle, Jhandle, x0, options:: NewtonPar{T}; normN = norm) where T
	# Rename parameters
	nltol       = options.tol
	nlmaxit     = options.maxIter
	verbose     = options.verbose

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

	# Main loop
	while (res > nltol) & (it < nlmaxit)
		J = Jhandle(x)
		d, flag, itlinear = options.linsolve(J, f)

		# Update solution
		# x .= x .- d
		minus_!(x,d)

		copyto!(f,Fhandle(x))
		res = normN(f)

		neval += 1
		push!(resHist, res)
		it += 1

		verbose && displayIteration(it, neval, res, itlinear)
	end
	(resHist[end] > nltol) && printstyled("\n--> Newton algorithm failed to converge, res = ", res[end], color=:red)
	return x, resHist, resHist[end] < nltol, it
end

#simplified call to newton when no Jacobian is passed in which case we estimate it using finiteDifferences
function newton(Fhandle, x0, options:: NewtonPar{T};kwargs...) where T
	Jhandle = u-> finiteDifferences(Fhandle, u)
	return newton(Fhandle, Jhandle, x0, options; kwargs...)
end

"""
	newtonDeflated(Fhandle::Function, Jhandle, x0, options:: NewtonPar{T}, defOp::DeflationOperator{T, vectype})

This is the deflated version of the Newton Solver. It penalises the roots saved in `defOp.roots`
"""
function newtonDeflated(Fhandle, Jhandle, x0, options:: NewtonPar{T}, defOp::DeflationOperator{T, vectype};kwargs...) where {T, vectype}
	# we create the new functional
	deflatedPb = DeflatedProblem(Fhandle, Jhandle, defOp)

	# and its jacobian
	Jacdf = (u0, pb::DeflatedProblem, ls) -> (return (u0, pb, ls))

	# Rename parameters
	opt_def = @set options.linsolve = DeflatedLinearSolver()
	return newton(u -> deflatedPb(u),
						u-> Jacdf(u, deflatedPb, options.linsolve),
						x0,
						opt_def;kwargs...)
end

function newtonDeflated(Fhandle, x0, options:: NewtonPar{T}, defOp::DeflationOperator{T, vectype};kwargs...) where {T, vectype}
	Jhandle = u-> PseudoArcLengthContinuation.finiteDifferences(Fhandle, u)
	return newtonDeflated(Fhandle,  Jhandle,  x0, options,  defOp;kwargs...)
end

"""
This is the classical matrix-free Newton Solver used to solve `F(x, l) = 0` together
with the scalar condition `n(x, l) = (x - x0) * xp + (l - l0) * lp - n0`
"""
function newtonPsArcLength(F, Jh,
						z0::M, tau0::M, z_pred::M,
						options::ContinuationPar{T};
						linearalgo = :bordering,
						normN = norm) where {T, vectype, M<:BorderedVector{vectype, T}}
	# Rename parameters
	newtonOpts = options.newtonOptions
	nltol   = newtonOpts.tol
	nlmaxit = newtonOpts.maxIter
	verbose = newtonOpts.verbose
	alpha   = convert(eltype(z0.p), newtonOpts.alpha)
	almin   = convert(eltype(z0.p), newtonOpts.almin)
	theta   = convert(eltype(z0.p), options.theta)
	ds      = convert(eltype(z0.p), options.ds)
	epsi    = convert(eltype(z0.p), options.finDiffEps)

	N = (x, p) -> arcLengthEq(x - z0.u, p - z0.p, tau0.u, tau0.p, theta, ds)

	# Initialise iterations
	x = copy(z_pred.u);  l = z_pred.p

	# Initialise residuals
	res_f = F(x, l);  res_n = N(x, l)
	# println("------> NewtBordered, resn = $res_n, ", arcLengthEq(z_pred-z0, tau0, xi, ds))

	dX   = similar(res_f)
	dl   = T(0)
	dFdl = (F(x, l + epsi) - res_f) / epsi

	res     = max(normN(res_f), abs(res_n))
	resHist = [res]
	it = 0

	# Displaying results
	verbose && displayIteration(it, 1, res)
	step_ok = true

	# Main loop
	while (res > nltol) & (it < nlmaxit) & step_ok
		copyto!(dFdl, (F(x, l+epsi) - F(x, l)) / epsi)

		J = Jh(x, l)
		u, up, liniter = linearBorderedSolver(J, dFdl,
						tau0, res_f, res_n, theta,
						newtonOpts.linsolve,
						algo = linearalgo)

		if newtonOpts.linesearch
			step_ok = false
			while (step_ok == false) & (alpha > almin)
				x_pred = x - alpha * u
				l_pred = l - alpha * up
				res_f .= F(x_pred, l_pred)
				res_n  = N(x_pred, l_pred)
				res = max(normN(res_f), abs(res_n))

				if res < resHist[end]
					if (res < resHist[end]/2) & (alpha < 1)
						alpha *=2
					end
					step_ok = true
					x .= x_pred
					l  = l_pred
				else
					alpha /= 2
				end
			end
		else
			minus_!(x, u) 	# x .= x .- u
			l = l - up

			copyto!(res_f, F(x, l))
			res_n  = N(x, l)
			res = sqrt(normN(res_f)^2 + res_n^2)
		end
		# Book-keeping
		push!(resHist, res)
		it += 1
		verbose && displayIteration(it, 1, res, liniter)

	end
	return BorderedVector(x, l), resHist, resHist[end] < nltol, it
end
