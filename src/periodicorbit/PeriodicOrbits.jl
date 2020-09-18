abstract type AbstractPeriodicOrbitProblem end

# Periodic orbit computations by finite differences
abstract type AbstractPOFDProblem <: AbstractPeriodicOrbitProblem end

abstract type AbstractShootingProblem <: AbstractPeriodicOrbitProblem end

# get the number of time slices / sections
@inline getM(pb::AbstractPeriodicOrbitProblem) = pb.M

# update a problem with arguments
function update(prob::AbstractPeriodicOrbitProblem, F, dF, hopfpt, ζr, M, orbitguess_a, period) end

####################################################################################################
"""
$(TYPEDEF)

If we use the same code as for newton (see below) in continuation, it is difficult to tell the eigensolver not to use the jacobian but instead the monodromy matrix. So we have to use a dedicated composite type for the jacobian to handle this case.

$(TYPEDFIELDS)
"""
struct ShootingJacobian{Tpb, Tjacpb, Torbitguess, Tp}
	pb::Tpb
	jacpb::Tjacpb
	x::Torbitguess
	par::Tp
end
ShootingJacobian(pb, x, par) = ShootingJacobian(pb, dx -> pb(x, par, dx), x, par)

# evaluation of the jacobian
(shjac::ShootingJacobian)(dx) = shjac.jacpb(dx)

####################################################################################################
# newton wrapper
"""
$(SIGNATURES)

This is the Newton-Krylov Solver for computing a periodic orbit using (Standard / Poincaré) Shooting method.
Note that the linear solver has to be apropriately set up.

# Arguments

Similar as [`newton`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref). These two problems have specific options to be tuned, we refer to their link for more information and to the tutorials.

# Output:
- solution
- history of residuals
- flag of convergence
- number of iterations
"""
function newton(prob::AbstractShootingProblem, orbitguess, par, options::NewtonPar; kwargs...)
	return newton(prob,
			(x, p) -> (dx -> prob(x, p, dx)),
			orbitguess, par, options; kwargs...)
end

"""
$(SIGNATURES)

This is the deflated Newton-Krylov Solver for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar as [`newton`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref).

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

"""
$(SIGNATURES)

This is the continuation method for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar as [`continuation`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref).

- `printPeriod` boolean to print the period of the solution. This is useful for `prob::PoincareShootingProblem` as this information is not easily available.
"""

function continuation(prob::AbstractShootingProblem, orbitguess, par, lens::Lens, contParams::ContinuationPar, linearAlgo::AbstractBorderedLinearSolver; printPeriod = true, kwargs...)

	options = contParams.newtonOptions

	if computeEigenElements(contParams)
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
$(SIGNATURES)

This is the continuation routine for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar as [`continuation`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref).

- `printPeriod` boolean to print the period of the solution. This is useful for `prob::PoincareShootingProblem` as this information is not easily available.
"""
function continuation(prob::AbstractShootingProblem, orbitguess, par, lens::Lens, contParams::ContinuationPar; linearAlgo = BorderingBLS(), printPeriod = true, kwargs...)
	_linearAlgo = @set linearAlgo.solver = contParams.newtonOptions.linsolver
	return continuation(prob, orbitguess, par, lens, contParams, _linearAlgo; printPeriod = printPeriod, kwargs...)
end

####################################################################################################
"""
$(SIGNATURES)

Perform automatic branch switching from a Hopf bifurcation point labelled `ind_bif` in the list of the bifurcated points on a previously computed branch `br::ContResult`. It first computes a Hopf normal form.

# Arguments

- `F, dF, d2F, d3F`: function `(x,p) -> F(x,p)` and its differentials `(x,p,dx) -> d1F(x,p,dx)`, `(x,p,dx1,dx2) -> d2F(x,p,dx1,dx2)`... These are used to compute the Hopf normal form.
- `br` branch result from a call to `continuation`
- `ind_hopf` index of the bifurcation point in `br`
- `contParams` parameters for the call to `continuation`
- `prob` problem used to specify the way the periodic orbit is computed. It can be [`PeriodicOrbitTrapProblem`](@ref), [`ShootingProblem`](@ref) or [`PoincareShootingProblem`](@ref) .

# Optional arguments

- `linearPO` linear algorithm used for the computation of periodic orbits when `prob` is [`PeriodicOrbitTrapProblem`](@ref))
- `Jt` is the jacobian adjoint, used for computation of the eigen-elements of the jacobian adjoint, needed to compute the spectral projector for the Hopf normal form!!!!!! COMME NEWTON FOLD
- `δ = 1e-8` used for finite differences
- `δp` used to specify a particular guess for the parameter on the bifurcated branch which is otherwise determined by `contParams.ds`. This allows to use a step larger than `contParams.dsmax`.
- `ampfactor = 1` factor which alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch

!!! note "Linear solver"
    You have to be carefull about the options `contParams.newtonOptions.linsolver`. In the case of Matrix-Free solver, you have to pass the right number of unknowns `N * M + 1`. Note that the options for the preconditioner are not accessible yet.

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jt = nothing` by using `tranpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jt = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not `nothing`, is computed with Finite differences.
"""
function continuation(F, dF, d2F, d3F, br::ContResult, ind_bif::Int, _contParams::ContinuationPar, prob::AbstractPeriodicOrbitProblem ; Jt = nothing, δ = 1e-8, δp = nothing, ampfactor = 1, usedeflation = false, nev = _contParams.nev, kwargs...)
	# compute the normal form of the branch point
	verbose = get(kwargs, :verbosity, 0) > 1 ? true : false
	cb = get(kwargs, :callbackN, cbDefault)

	hopfpt = hopfNormalForm(F, dF, d2F, d3F, br, ind_bif ; Jt = Jt, δ = δ, nev = nev, verbose = verbose)

	# compute predictor for point on new branch
	ds = isnothing(δp) ? _contParams.ds : δp
	Ty = typeof(ds)
	pred = predictor(hopfpt, ds; verbose = verbose, ampfactor = Ty(ampfactor))

	verbose && printstyled(color = :green, "#"^61*
			"\n--> Start branching from Hopf branch to periodic orbits.
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

	return Branch(setproperties(branch; type = :PeriodicOrbit, functional = probPO), hopfpt), u, tau

end
