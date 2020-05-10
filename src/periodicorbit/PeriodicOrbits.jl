abstract type AbstractPeriodicOrbitProblem end

# Periodic orbit computations by finite differences
abstract type AbstractPOFDProblem <: AbstractPeriodicOrbitProblem end

abstract type AbstractShootingProblem <: AbstractPeriodicOrbitProblem end

# get the number of time slices / sections
@inline getM(pb::AbstractPeriodicOrbitProblem) = pb.M

# update a problem with arguments
function update(prob::AbstractPeriodicOrbitProblem, F, dF, hopfpt, ζr, M, orbitguess_a, period) end

####################################################################################################
# if we use the same code as for newton (see below) in continuation, it is difficult to tell the eigensolver not to use the jacobian but instead the monodromy matrix. So we have to use a dedicated composite type for the jacobian to handle this case.

struct ShootingJacobian{Tpb <: AbstractShootingProblem, Torbitguess, Tp}
	pb::Tpb
	x::Torbitguess
	par::Tp
end

# evaluation of the jacobian
(shjac::ShootingJacobian)(dx) = shjac.pb(shjac.x, shjac.par, dx)

####################################################################################################
# newton wrapper
"""
	newton(prob::T, orbitguess, options::NewtonPar; kwargs...) where {T <: AbstractShootingProblem}

This is the Newton Solver for computing a periodic orbit using Shooting method.

# Output:
- solution
- history of residuals
- flag of convergence
- number of iterations
"""
function newton(prob::AbstractShootingProblem, orbitguess, par, options::NewtonPar; kwargs...)
	return newton(prob,
			(x, p) -> (dx -> prob(x, p, dx)),
			orbitguess, par,
			options; kwargs...)
end

"""
	newton(prob::T, orbitguess, options::NewtonPar, defOp::DeflationOperator; kwargs...) where {T <: AbstractShootingProblem}

This is the deflated Newton Solver for computing a periodic orbit using Shooting method.

# Output:
- solution
- history of residuals
- flag of convergence
- number of iterations
"""
function newton(prob::AbstractShootingProblem, orbitguess, options::NewtonPar, defOp::DeflationOperator; kwargs...)
	return newton(prob,
			(x, p) -> (dx -> prob(x, p, dx)),
			orbitguess, par,
			options, defOp; kwargs...)
end

####################################################################################################
# Continuation for shooting problems

function continuation(prob::AbstractShootingProblem, orbitguess, par, lens::Lens, _contParams::ContinuationPar, linearAlgo::AbstractBorderedLinearSolver; printPeriod = true, kwargs...)

	contParams = check(_contParams)

	options = contParams.newtonOptions

	if contParams.computeEigenValues
		contParams = @set contParams.newtonOptions.eigsolver = FloquetQaDShooting(contParams.newtonOptions.eigsolver)
	end

	if (prob isa PoincareShootingProblem)

		if printPeriod
			printSolutionPS = (x, p) -> getPeriod(prob, x, set(par, lens, p))
			return continuation(
				prob,
				(x, p) -> ShootingJacobian(prob, x, p),
				orbitguess, par, lens,
				contParams, linearAlgo;
				printSolution = printSolutionPS,
				kwargs...)
		end
	end

	return continuation(
		prob,
		(x, p) -> ShootingJacobian(prob, x, p),
		orbitguess, par, lens,
		contParams, linearAlgo;
		kwargs...)
end

"""
	continuationPOShooting(prob, orbitguess, p0::Real, contParams::ContinuationPar; printPeriod = true, kwargs...)

This is the continuation routine for computing a periodic orbit using a functional G based on a Shooting method.

# Arguments
- `p -> prob(p)` is a function or family such that `prob(p)::AbstractShootingProblem` encodes the functional G
- `orbitguess` a guess for the periodic orbit. For the type of `orbitguess`, please see the information concerning [`ShootingProblem`](@ref) and [`PoincareShootingProblem`](@ref).
- `p0` initial parameter, must be a real number
- `contParams` same as for the regular `continuation` method
- `printPeriod` in the case of Poincaré Shooting, plot the period of the cycle.
"""
function continuation(prob::AbstractShootingProblem, orbitguess, par, lens::Lens, contParams::ContinuationPar; linearAlgo = BorderingBLS(), printPeriod = true, kwargs...)
	_linearAlgo = @set linearAlgo.solver = contParams.newtonOptions.linsolver
	return continuation(prob, orbitguess, par, lens, contParams, _linearAlgo; printPeriod = printPeriod, kwargs...)
end

####################################################################################################
"""
	continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, contParams::ContinuationPar, prob::AbstractPeriodicOrbitProblem ; Jt = nothing, δ = 1e-8, δp = nothing, linearPO = :BorderedLU, M = 21, printSolution = (u,p) -> u[end], linearAlgo = BorderingBLS(), kwargs...)

Perform branch switching from Hopf bifurcation point labelled `ind_bif` in the list of the bifurcated points on a previously computed branch `br`. The periodic orbits are computed using Finite Differences (see [`PeriodicOrbitTrapProblem`](@ref) for more information).

# Arguments

- `F, dF, d2F, d3F`: function `(x,p) -> F(x,p)` and its differencials `(x,p,dx) -> d1F(x,p,dx)`, `(x,p,dx1,dx2) -> d2F(x,p,dx1,dx2)`...
- `br` branch result from a call to `continuation`
- `ind_hopf` index of the bifurcation point in `br`
- `contParams` parameters for the call to `continuation`

# Optional arguments

- `M = 21` number of time discretization to parametrize the periodic orbits
- `linearPO` linear algorithm used for the computation of periodic orbits. (see [`PeriodicOrbitTrapProblem`](@ref))
- `Jt` is the jacobian adjoint, used for computation of the eigen-elements of the jacobian adjoint, needed to compute the spectral projector for the Hopf normal form!!!!!! COMME NEWTON FOLD
- `δ = 1e-8` used for finite differences
- `δp = 0.1` used to specify a particular guess for the parameter in the branch. This allows to use a step larger than `dsmax`.

!!! note "Linear solver"
    You have to be carefull about the options `contParams.newtonOptions.linsolver`. In the case of Matrix-Free solver, you have to pass the right number of unknowns `N * M + 1`. Note that the options for the preconditioner are not accessible yet.
"""
function continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, _contParams::ContinuationPar, prob::AbstractPeriodicOrbitProblem ; Jt = nothing, δ = 1e-8, δp = nothing, ampfactor = 1, kwargs...)
	# compute the normal form of the branch point
	verbose = get(kwargs, :verbosity, 0) > 1 ? true : false

	hopfpt = hopfNormalForm(F, dF, d2F, d3F, br, ind_bif ; Jt = Jt, δ = δ, nev = _contParams.nev, verbose = verbose)

	# compute predictor for point on new branch
	ds = isnothing(δp) ? _contParams.ds : δp
	Ty = typeof(ds)
	pred = predictor(hopfpt, ds; verbose = verbose, ampfactor = Ty(ampfactor))

	verbose && printstyled(color = :green, "#"^51*
			"\n--> Start Hopf branch switching.
			\n--> Bifurcation type = ", hopfpt.type,
			"\n----> newp = ", pred.p, ", δp = ", pred.p - br.bifpoint[ind_bif].param,
			"\n----> amplitude = ", pred.amp,
			"\n----> period = ", abs(2pi/pred.ω), "\n")

	# we compute a phase so that the constraint equation
	# < u(0) − u_hopf, ψ > is satisfied, i.e. equal to zero.
	ζr = real.(hopfpt.ζ); ζi = imag.(hopfpt.ζ)
	# this phase is for POTrap problem constraint to be satisfied
	ϕ = atan(dot(ζr, ζr), dot(ζi, ζr))
	verbose && printstyled(color = :green,"----> phase ϕ = ", ϕ/pi, "⋅π\n")

	M = getM(prob)
	orbitguess_a = [pred.orbit(t - ϕ) for t in LinRange(0, 2pi ,M+1)[1:M]]

	# build the variable to hold the functional for computing PO based on finite differences
	probPO, orbitguess = update(prob, F, dF, hopfpt, ζr, M, orbitguess_a, abs(2pi/pred.ω))

	if _contParams.newtonOptions.linsolver isa GMRESIterativeSolvers
		_contParams = @set _contParams.newtonOptions.linsolver.N = length(orbitguess)
	end

	# perform continuation
	branch, u, tau = continuation(probPO, orbitguess, set(br.params, br.param_lens, pred.p), br.param_lens, _contParams; kwargs...)

	return setproperties(branch; type = :PeriodicOrbit, functional = probPO), u, tau

end
