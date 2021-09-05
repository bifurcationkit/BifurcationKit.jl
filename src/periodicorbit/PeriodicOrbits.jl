abstract type AbstractPeriodicOrbitProblem end

# Periodic orbit computations by finite differences
abstract type AbstractPOFDProblem <: AbstractPeriodicOrbitProblem end
# Periodic orbit computations by shooting
abstract type AbstractShootingProblem <: AbstractPeriodicOrbitProblem end

# get the number of time slices
@inline getM(pb::AbstractPeriodicOrbitProblem) = pb.M

# update a problem with arguments
function updateForBS(prob::AbstractPeriodicOrbitProblem, F, dF, hopfpt, ζr, M, orbitguess_a, period) end

# update a section with a problem, we do nothing by default
updateSection!(prob::AbstractPeriodicOrbitProblem, x, par) = @warn "Not yet implemented!"

Base.size(pb::AbstractPOFDProblem) = (pb.M, pb.N)
onGpu(pb::AbstractPOFDProblem) = pb.ongpu
hasHessian(pb::AbstractPOFDProblem) = pb.d2F == nothing
isInplace(pb::AbstractPOFDProblem) = pb.isinplace

function applyF(pb::AbstractPOFDProblem, dest, x, p)
	if isInplace(pb)
		pb.F(dest, x, p)
	else
		dest .= pb.F(x, p)
	end
	dest
end

function applyJ(pb::AbstractPOFDProblem, dest, x, p, dx)
	if isInplace(pb)
		pb.J(dest, x, p, dx)
	else
		dest .= apply(pb.J(x, p), dx)
	end
	dest
end

# function to extract trajectories from branch
getTrajectory(br::AbstractBranchResult, ind::Int) = getTrajectory(br.functional, br.sol[ind].x, setParam(br, br.sol[ind].p))
####################################################################################################
"""
$(TYPEDEF)

Define a type to interface the Jacobian of the Shooting Problem with the Floquet computation methods. If we use the same code as for `newton` (see below) in `continuation`, it is difficult to tell the eigensolver not to use the jacobian but instead the monodromy matrix.

$(TYPEDFIELDS)
"""
struct FloquetWrapper{Tpb, Tjacpb, Torbitguess, Tp}
	pb::Tpb
	jacpb::Tjacpb
	x::Torbitguess
	par::Tp
end
FloquetWrapper(pb, x, par) = FloquetWrapper(pb, dx -> pb(x, par, dx), x, par)

# jacobian evaluation
(shjac::FloquetWrapper)(dx) = apply(shjac.jacpb, dx)

# specific linear solver to dispatch
struct FloquetWrapperLS{T} <: AbstractLinearSolver
	solver::T
end
# this constructor prevents from having FloquetWrapperLS(FloquetWrapperLS(ls))
FloquetWrapperLS(ls::FloquetWrapperLS) = ls
(ls::FloquetWrapperLS)(J, rhs; kwargs...) = ls.solver(J, rhs; kwargs...)
(ls::FloquetWrapperLS)(J::FloquetWrapper, rhs; kwargs...) = ls.solver(J.jacpb, rhs; kwargs...)
(ls::FloquetWrapperLS)(J::FloquetWrapper, rhs1, rhs2) = ls.solver(J.jacpb, rhs1, rhs2)

# this is for the use of MatrixBLS
LinearAlgebra.hcat(shjac::FloquetWrapper, dR) = hcat(shjac.jacpb, dR)

####################################################################################################
# newton wrapper
function buildJacobian(prob::AbstractShootingProblem, orbitguess, par, linearPO = :MatrixFree)
	if linearPO == :autodiffDenseAnalytical
		_J = prob(Val(:JacobianMatrix), orbitguess, par)
		jac = (x, p) -> prob(Val(:JacobianMatrixInplace), _J, x, p)
	elseif linearPO == :autodiffDense
		jac = (x, p) -> ForwardDiff.jacobian(z -> prob(z, p), x)
	elseif linearPO == :autodiffMF
		jac = (x, p) -> (dx -> ForwardDiff.derivative(z -> prob((@. x + z * dx), p), 0))
	elseif linearPO == :FiniteDifferencesDense
		jac = (x, p) -> finiteDifferences(z -> prob(z, p), x; δ = 1e-8)
	else
		jac = (x, p) -> (dx -> prob(x, p, dx))
	end
end

"""
$(SIGNATURES)

This is the Newton-Krylov Solver for computing a periodic orbit using (Standard / Poincaré) Shooting method.
Note that the linear solver has to be apropriately set up in `options`.

# Arguments

Similar to [`newton`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref). These two problems have specific options to be tuned, we refer to their link for more information and to the tutorials.

- `prob` a problem of type `<: AbstractShootingProblem` encoding the shooting functional G.
- `orbitguess` a guess for the periodic orbit. See [`ShootingProblem`](@ref) and See [`PoincareShootingProblem`](@ref) for information regarding the shape of `orbitguess`.
- `par` parameters to be passed to the functional
- `options` same as for the regular [`newton`](@ref) method.

# Optional argument
- `linearPO` Specify the choice of the linear algorithm, which must belong to `(:autodiffMF, :MatrixFree, :autodiffDense, :FiniteDifferences)`. This is used to select a way of inverting the jacobian dG
    - For `:MatrixFree`, we use an iterative solver (e.g. GMRES) to solve the linear system. The jacobian was specified by the user in `prob`.
    - For `:autodiffMF`, we use iterative solver (e.g. GMRES) to solve the linear system. We use Automatic Differentiation to compute the (matrix-free) derivative of `x -> prob(x, p)`.
    - For `:autodiffDense`. Same as for `:autodiffMF` but the jacobian is formed as a dense Matrix. You can use a direct solver or an iterative one using `options`.
	- For `:autodiffDenseAnalytical`. Same as for `:autodiffDense` but the jacobian is using a mix of AD and analytical formula.
    - For `:FiniteDifferencesDense`, same as for `:autodiffDense` but we use Finite Differences to compute the jacobian of `x -> prob(x, p)` using the `δ = 1e-8` which can be passed as an argument.

# Output:
- solution
- history of residuals
- flag of convergence
- number of iterations
"""
function newton( prob::AbstractShootingProblem, orbitguess, par, options::NewtonPar;
		linearPO = :MatrixFree, δ = 1e-8, kwargs...)
	jac = buildJacobian(prob, orbitguess, par, linearPO)
	return newton(prob, jac, orbitguess, par, options; kwargs...)
end

"""
$(SIGNATURES)

This is the deflated Newton-Krylov Solver for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar to [`newton`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref).

# Optional argument
- `linearPO` Specify the choice of the linear algorithm, which must belong to `[:autodiffMF, :MatrixFree, :autodiffDense, :FiniteDifferences]`. This is used to select a way of inverting the jacobian dG
    - For `:MatrixFree`, we use an iterative solver (e.g. GMRES) to solve the linear system. The jacobian was specified by the user in `prob`.
    - For `:autodiffMF`, we use iterative solver (e.g. GMRES) to solve the linear system. We use Automatic Differentiation to compute the (matrix-free) derivative of `x -> prob(x, p)`.
    - For `:autodiffDense`. Same as for `:autodiffMF` but the jacobian is formed as a dense Matrix. You can use a direct solver or an iterative one using `options`.
	- For `:autodiffDenseAnalytical`. Same as for `:autodiffDense` but the jacobian is using a mix of AD and analytical formula.
    - For `:FiniteDifferencesDense`, same as for `:autodiffDense` but we use Finite Differences to compute the jacobian of `x -> prob(x, p)` using the `δ = 1e-8` which can be passed as an argument.


# Output:
- solution
- history of residuals
- flag of convergence
- number of iterations
"""
function newton(
    prob::AbstractShootingProblem,
    orbitguess::vectype,
    par,
    options::NewtonPar{T,S,E},
    defOp::DeflationOperator{Tp,Tdot,T,vectype};
    linearPO = :MatrixFree,
    kwargs...,
) where {T,Tp,Tdot,vectype,S,E}
	jac = buildJacobian(prob, orbitguess, par, linearPO)
	return newton(prob, jac, orbitguess, par, options, defOp; kwargs...)
end

####################################################################################################
# Continuation for shooting problems

"""
$(SIGNATURES)

This is the continuation method for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar to [`continuation`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref). By default, it prints the period of the periodic orbit.

# Optional argument
- `δ = 1e-8` used for finite differences
- `linearPO` Specify the choice of the linear algorithm, which must belong to `[:autodiffMF, :MatrixFree, :autodiffDense, :FiniteDifferences]`. This is used to select a way of inverting the jacobian dG
    - For `:MatrixFree`, we use an iterative solver (e.g. GMRES) to solve the linear system. The jacobian was specified by the user in `prob`.
    - For `:autodiffMF`, we use iterative solver (e.g. GMRES) to solve the linear system. We use Automatic Differentiation to compute the (matrix-free) derivative of `x -> prob(x, p)`.
    - For `:autodiffDense`. Same as for `:autodiffMF` but the jacobian is formed as a dense Matrix. You can use a direct solver or an iterative one using `options`.
    - For `:FiniteDifferencesDense`, same as for `:autodiffDense` but we use Finite Differences to compute the jacobian of `x -> prob(x, p)` using the `δ = 1e-8` which can be passed as an argument.
	- For `:autodiffDenseAnalytical`. Same as for `:autodiffDense` but the jacobian is using a mix of AD and analytical formula.
	- For `:FiniteDifferences`, use Finite Differences to compute the jacobian of `x -> prob(x, p)` using the `δ = 1e-8` which can be passed as an argument.
- `updateSectionEveryStep = 0` updates the section every `updateSectionEveryStep` step during continuation
"""
function continuation(
    prob::AbstractShootingProblem,
    orbitguess,
    par,
    lens::Lens,
    contParams::ContinuationPar,
    linearAlgo::AbstractBorderedLinearSolver;
    linearPO = :MatrixFree,
    updateSectionEveryStep = 0,
    δ = 1e-8,
    kwargs...,
)
    @assert linearPO in
            (:autodiffMF, :MatrixFree, :autodiffDense, :autodiffDenseAnalytical, :FiniteDifferencesDense, :FiniteDifferences)

	if computeEigenElements(contParams)
		contParams = @set contParams.newtonOptions.eigsolver = FloquetQaD(contParams.newtonOptions.eigsolver)
	end

	options = contParams.newtonOptions

	_finsol = get(kwargs, :finaliseSolution, nothing)
	_finsol2 = isnothing(_finsol) ? (z, tau, step, contResult; kwargs...) ->
		begin
			modCounter(step, updateSectionEveryStep) == 1 && updateSection!(prob, z.u, setParam(contResult, z.p))
			true
		end :
		(z, tau, step, contResult; prob = prob, kwargs...) ->
			begin
				modCounter(step, updateSectionEveryStep) == 1 && updateSection!(prob, z.u, setParam(contResult, z.p))
				_finsol(z, tau, step, contResult; prob = prob, kwargs...)
			end

    if linearPO == :autodiffDenseAnalytical
		_J = prob(Val(:JacobianMatrix), orbitguess, par)
		jac = (x, p) -> (prob(Val(:JacobianMatrixInplace), _J, x, p); FloquetWrapper(prob, _J, x, p));
	elseif linearPO == :autodiffDense
		jac = (x, p) -> FloquetWrapper(prob, ForwardDiff.jacobian(z -> prob(z, p), x), x, p)
	elseif linearPO == :FiniteDifferencesDense
		jac = (x, p) -> FloquetWrapper(prob, finiteDifferences(z -> prob(z, p), x), x, p)
	elseif linearPO == :autodiffMF
		jac = (x, p) -> FloquetWrapper(prob, (dx -> ForwardDiff.derivative(z -> prob(x .+ z .* dx, p), 0)), x, p)
	elseif linearPO == :FiniteDifferences
		jac = (x, p) -> FloquetWrapper(prob, dx -> (prob(x .+ δ .* dx, p) .- prob(x, p)) ./ δ, x, p)
	else
		jac = (x, p) -> FloquetWrapper(prob, x, p)
	end

	# we have to change the Bordered linearsolver to cope with our type FloquetWrapper
	linearAlgo = @set linearAlgo.solver = FloquetWrapperLS(linearAlgo.solver)

	branch, u, τ = continuation(
		prob, jac,
		orbitguess, par, lens,
		(@set contParams.newtonOptions.linsolver = FloquetWrapperLS(options.linsolver)), linearAlgo;
		recordFromSolution = (x, p) -> (period = getPeriod(prob, x, set(par, lens, p)),),
		finaliseSolution = _finsol2,
		kwargs...,)
	return setproperties(branch; type = :PeriodicOrbit, functional = prob), u, τ
end

"""
$(SIGNATURES)

This is the continuation routine for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar to [`continuation`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref).

- `printPeriod` boolean to print the period of the solution. This is useful for `prob::PoincareShootingProblem` as this information is not easily available.
"""
function continuation(prob::AbstractShootingProblem, orbitguess, par, lens::Lens, _contParams::ContinuationPar; linearAlgo = nothing, kwargs...)
	_linearAlgo = isnothing(linearAlgo) ?  BorderingBLS(_contParams.newtonOptions.linsolver) : linearAlgo
	return continuation(prob, orbitguess, par, lens, _contParams, _linearAlgo; kwargs...)
end

####################################################################################################
"""
$(SIGNATURES)

Perform automatic branch switching from a Hopf bifurcation point labelled `ind_bif` in the list of the bifurcated points of a previously computed branch `br::ContResult`. It first computes a Hopf normal form.

# Arguments

- `F, dF, d2F, d3F`: function `(x,p) -> F(x,p)` and its differentials `(x,p,dx) -> d1F(x,p,dx)`, `(x,p,dx1,dx2) -> d2F(x,p,dx1,dx2)`... These are used to compute the Hopf normal form.
- `br` branch result from a call to `continuation`
- `ind_hopf` index of the bifurcation point in `br`
- `contParams` parameters for the call to `continuation`
- `prob` problem used to specify the way the periodic orbit is computed. It can be [`PeriodicOrbitTrapProblem`](@ref), [`ShootingProblem`](@ref) or [`PoincareShootingProblem`](@ref) .

# Optional arguments

- `linearPO` linear algorithm used for the computation of periodic orbits when `prob` is [`PeriodicOrbitTrapProblem`](@ref))
- `Jᵗ` is the jacobian adjoint, used for computation of the eigen-elements of the jacobian adjoint, needed to compute the spectral projector for the Hopf normal form. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of Matrix / Sparse based jacobian, `Jᵗ` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jᵗ = (x, p) -> transpose(dF(x, p))`
- `δ = 1e-8` used for finite differences
- `δp` used to specify a particular guess for the parameter on the bifurcated branch which is otherwise determined by `contParams.ds`. This allows to use a step larger than `contParams.dsmax`.
- `ampfactor = 1` factor which alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch
- `updateSectionEveryStep = 0` updates the section every `updateSectionEveryStep` step during continuation
- `linearPO` specify the way the jacobian is computed.
- all `kwargs` from [`continuation`](@ref)

A modified version of `prob` is passed to `plotSolution` and `finaliseSolution`.

!!! note "Linear solver"
    You have to be careful about the options `contParams.newtonOptions.linsolver`. In the case of Matrix-Free solver, you have to pass the right number of unknowns `N * M + 1`. Note that the options for the preconditioner are not accessible yet.

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not `nothing`, is computed with Finite differences.
"""
function continuation(F, dF, d2F, d3F, br::AbstractBranchResult, ind_bif::Int, _contParams::ContinuationPar, prob::AbstractPeriodicOrbitProblem ; Jᵗ = nothing, δ = 1e-8, δp = nothing, ampfactor = 1, usedeflation = false, nev = _contParams.nev, updateSectionEveryStep = 0, kwargs...)
	# compute the normal form of the branch point
	verbose = get(kwargs, :verbosity, 0) > 1 ? true : false
	verbose && (println("--> Considering bifurcation point:"); _show(stdout, br.specialpoint[ind_bif], ind_bif))

	cb = get(kwargs, :callbackN, cbDefault)

	hopfpt = hopfNormalForm(F, dF, d2F, d3F, br, ind_bif ; Jᵗ = Jᵗ, δ = δ, nev = nev, verbose = verbose)

	# compute predictor for point on new branch
	ds = isnothing(δp) ? _contParams.ds : δp
	Ty = typeof(ds)
	pred = predictor(hopfpt, ds; verbose = verbose, ampfactor = Ty(ampfactor))

	verbose && printstyled(color = :green, "#"^61*
			"\n--> Start branching from Hopf bif. point to periodic orbits.
			 \n--> Bifurcation type = ", hopfpt.type,
			"\n----> Hopf param = ", br.specialpoint[ind_bif].param,
			"\n----> newp = ", pred.p, ", δp = ", pred.p - br.specialpoint[ind_bif].param,
			"\n----> amplitude = ", pred.amp,
			"\n----> period = ", abs(2pi/pred.ω), "\n")

	# we compute a phase so that the constraint equation
	# < u(0) − u_hopf, ψ > is satisfied, i.e. equal to zero.
	ζr = real.(hopfpt.ζ)
	ζi = imag.(hopfpt.ζ)
	# this phase is for POTrap problem constraint to be satisfied
	ϕ = atan(dot(ζr, ζr), dot(ζi, ζr))
	verbose && printstyled(color = :green,"----> phase ϕ = ", ϕ/pi, "⋅π\n")

	M = getM(prob)
	orbitguess_a = [pred.orbit(t - ϕ) for t in LinRange(0, 2pi, M+1)[1:M]]

	# build the variable to hold the functional for computing PO based on finite differences
	probPO, orbitguess = problemForBS(prob, F, dF, br.params, hopfpt, ζr, orbitguess_a, abs(2pi/pred.ω))

	if _contParams.newtonOptions.linsolver isa GMRESIterativeSolvers
		_contParams = @set _contParams.newtonOptions.linsolver.N = length(orbitguess)
	end

	# pass the problem to the plotting and recordFromSolution functions
	_plotsol = get(kwargs, :plotSolution, nothing)
	_plotsol2 = isnothing(_plotsol) ? (x, p; k...) -> nothing : (x, p; k...) -> _plotsol(x, (prob = probPO, p = p); k...)

	if :recordFromSolution in keys(kwargs)
		_printsol = get(kwargs, :recordFromSolution, nothing)
        @set! kwargs[:recordFromSolution] = (x, p; k...) -> _printsol(x, (prob = probPO, p = p); k...)
	end

	if usedeflation
		verbose &&
			println("\n--> Attempt branch switching\n--> Compute point on the current branch...")
		prob isa PoincareShootingProblem &&
			@warn "Poincaré Shooting does not work very well with stationary states."
		optn = _contParams.newtonOptions

		# we start with the case of zero amplitude
		orbitzeroamp_a = [hopfpt.x0 for _ = 1:M]
		# this factor prevent shooting jacobian from being singular at fixed points
		Tfactor = (prob isa AbstractPOFDProblem) || (prob isa PoincareShootingProblem) ? 0 : 0.001
		_, orbitzeroamp = problemForBS(prob, F, dF, br.params,
			hopfpt, ζr, orbitzeroamp_a,	Tfactor * abs(2pi / pred.ω))
		sol0, _, flag, _ = newton(probPO, orbitzeroamp, setParam(br, pred.p), optn; callback = cb, kwargs...)

		# find the bifurcated branch using deflation
		if ~(probPO isa PoincareShootingProblem)
			deflationOp = DeflationOperator(2, (x, y) -> dot(x[1:end-1], y[1:end-1]), 1.0, [sol0])
		else
			deflationOp = DeflationOperator(2, (x, y) -> dot(x, y) / M, 1.0, [sol0])
		end

		verbose && println("\n--> Compute point on bifurcated branch...")
		# @assert 1==9 "Mauvais dispatch ICI. Va en L199 Deflation.jl"
		solbif, _, flag, _ = newton(
			probPO, orbitguess, setParam(br, pred.p), (@set optn.maxIter = 10 * optn.maxIter), deflationOp; callback = cb, kwargs...)
		@assert flag "Deflated newton did not converge"
		orbitguess .= solbif

		# @assert 1==09

		# having to points, we call the specific method
		# branch, u, tau = continuation(probPO,
		# 	orbitzeroamp, br.params,
		# 	orbitguess, pred.p,
		# 	br.lens, _contParams; kwargs...)


		branch, u, τ = continuation(
			probPO, orbitguess,
			setParam(br, pred.p), br.lens,
			_contParams;
			kwargs...,
			plotSolution = _plotsol2,
			updateSectionEveryStep = updateSectionEveryStep,
		)

	return Branch(branch, hopfpt), u, τ
	end

	# perform continuation
	branch, u, τ = continuation(
		probPO, orbitguess, setParam(br, pred.p),
		br.lens, _contParams;
		kwargs..., # put this first to be overwritten just below!
		plotSolution = _plotsol2,
		updateSectionEveryStep = updateSectionEveryStep,
	)
	return Branch(branch, hopfpt), u, τ
end

####################################################################################################
# Branch switching from Bifs of PO
"""
$(SIGNATURES)

Branch switching at a Bifurcation point of a branch of periodic orbits (PO) specified by a `br::AbstractBranchResult`. The functional used to compute the PO is `br.functional`. A deflated Newton-Krylov solver can be used to improve the branch switching capabilities.

# Arguments
- `br` branch of periodic orbits computed with a [`PeriodicOrbitTrapProblem`](@ref)
- `ind_bif` index of the branch point
- `_contParams` parameters to be used by a regular [`continuation`](@ref)

# Optional arguments
- `δp = 0.1` used to specify a particular guess for the parameter in the branch which is otherwise determined by `contParams.ds`. This allows to use a step larger than `contParams.dsmax`.
- `ampfactor = 1` factor which alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch
- `linearPO = :BorderedLU` linear solver used for the Newton-Krylov solver when applied to [`PeriodicOrbitTrapProblem`](@ref).
- `recordFromSolution = (u, p) -> u[end]`, print method used in the bifurcation diagram, by default this prints the period of the periodic orbit.
- `linearAlgo = BorderingBLS()`, same as for [`continuation`](@ref)
- `kwargs` keywords arguments used for a call to the regular [`continuation`](@ref) and the ones specific to POs.
"""
function continuation(br::AbstractBranchResult, ind_bif::Int, _contParams::ContinuationPar;
			δp = 0.1, ampfactor = 1,
			usedeflation = false,
			linearAlgo = nothing,
			kwargs...)

	bppt = br.specialpoint[ind_bif]
	bptype = bppt.type
	@assert bptype in (:pd, :bp) "Branching from $(bptype) not possible yet."

	# @assert br.functional isa AbstractShootingProblem
	@assert abs(bppt.δ[1]) == 1 "Only simple bifurcation points are handled"

	verbose = get(kwargs, :verbosity, 0) > 0

	verbose && printstyled(color = :green, "#"^61*
			"\n--> Start branching from $(bptype) point to periodic orbits.
			 \n--> Bifurcation type = ", bppt.type,
			"\n----> bif. param     = ", bppt.param,
			"\n----> period at bif. = ", getPeriod(br.functional, bppt.x, setParam(br, bppt.param)),
			"\n----> newp           = ", bppt.param + δp, ", δp = ", δp,
			"\n----> amplitude      = ", ampfactor,
			"\n")

	_linearAlgo = isnothing(linearAlgo) ? BorderingBLS(_contParams.newtonOptions.linsolver) : linearAlgo

	bifpt = br.specialpoint[ind_bif]

	# we copy the problem for not mutating the one passed by the user
	pb = deepcopy(br.functional)

	# let us compute the kernel
	λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
	verbose && print("--> computing nullspace of Periodic orbit problem...")
	ζ = geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[bifpt.idx].eigenvec, bifpt.ind_ev)
	# we normalize it by the sup norm because it could be too small/big in L2 norm
	# TODO: user defined scaleζ
	ζ ./= norm(ζ, Inf)
	verbose && println("Done!")

	# compute the full eigenvector
	ζ_a = MonodromyQaD(Val(:ExtractEigenVector), pb, bifpt.x, setParam(br, bifpt.param), real.(ζ))
	ζs = reduce(vcat, ζ_a)

	## predictor
	pbnew, orbitguess = predictor(pb, bifpt, ampfactor, real.(ζs), bifpt.type)
	newp = bifpt.param + δp

	pbnew(orbitguess, setParam(br, newp))[end] |> abs > 1 && @warn "PO Trap constraint not satisfied"

	# a priori, the following do not overwrite the options in br
	# hence the results / parameters in br are kept intact
	if pb isa AbstractShootingProblem
		if _contParams.newtonOptions.linsolver isa GMRESIterativeSolvers
			@set! _contParams.newtonOptions.linsolver.N = length(orbitguess)
		elseif _contParams.newtonOptions.linsolver isa FloquetWrapperLS
			if _contParams.newtonOptions.linsolver.solver isa GMRESIterativeSolvers
				@set! _contParams.newtonOptions.linsolver.solver.N = length(orbitguess)
			end
		end
	end

	# pass the problem to the plotting and recordFromSolution functions
	_plotsol = get(kwargs, :plotSolution, nothing)
	_plotsol2 = isnothing(_plotsol) ? (x, p; k...) -> nothing :
		(x, p; k...) -> _plotsol(x, (prob = pbnew, p = p); k...)

	if :recordFromSolution in keys(kwargs)
		_printsol = get(kwargs, :recordFromSolution, nothing)
		@set! kwargs[:recordFromSolution] = (x, p; k...) -> _printsol(x, (prob = pbnew, p = p); k...)
	end

	if usedeflation
		verbose && println("\n--> Attempt branch switching\n--> Compute point on the current branch...")
		optn = _contParams.newtonOptions
		# find point on the first branch
		sol0, _, flag, _ = newton(pbnew, bifpt.x, setParam(br, newp), optn; kwargs...)

		# find the bifurcated branch using deflation
		deflationOp = DeflationOperator(2, (x, y) -> dot(x[1:end-1], y[1:end-1]), 1.0, [sol0])
		verbose && println("\n--> Compute point on bifurcated branch...")
		solbif, _, flag, _ = newton(pbnew, orbitguess, setParam(br, newp),
			(@set optn.maxIter = 10 * optn.maxIter),
			deflationOp; kwargs...,)
		@assert flag "Deflated newton did not converge"
		orbitguess .= solbif
	end

	# perform continuation
	branch, u, τ = continuation( pbnew, orbitguess, setParam(br, newp), br.lens, _contParams;
		kwargs..., # put this first to be overwritten just below!
		plotSolution = _plotsol2,
		linearAlgo = _linearAlgo,
	)

	# create a branch
	bppo = Pitchfork(bifpt.x, bifpt.param, setParam(br, bifpt.param), br.lens, ζ, ζ, nothing, :nothing)

	return Branch(setproperties(branch; type = :PeriodicOrbit, functional = br.functional), bppo), u, τ
end
