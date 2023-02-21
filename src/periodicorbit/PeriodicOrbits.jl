abstract type AbstractBoundaryValueProblem <: AbstractBifurcationProblem end
abstract type AbstractPeriodicOrbitProblem <: AbstractBoundaryValueProblem end

# Periodic orbit computations by finite differences
abstract type AbstractPODiffProblem <: AbstractPeriodicOrbitProblem end
abstract type AbstractPOFDProblem <: AbstractPODiffProblem end
# Periodic orbit computations by shooting
abstract type AbstractShootingProblem <: AbstractPeriodicOrbitProblem end
abstract type AbstractPoincareShootingProblem <: AbstractShootingProblem end

# get the number of time slices
@inline getMeshSize(pb::AbstractPeriodicOrbitProblem) = pb.M

"""
$(SIGNATURES)

Compute the period of the periodic orbit associated to `x`.
"""
@inline getPeriod(::AbstractPeriodicOrbitProblem, x, par = nothing) = extractPeriod(x)
@inline extractPeriod(x::AbstractVector) = x[end]
@inline extractPeriod(x::BorderedArray)  = x.p
setParamsPO(pb::AbstractPODiffProblem, pars) = (@set pb.prob_vf = reMake(pb.prob_vf; params = pars))
setParamsPO(pb::AbstractShootingProblem, pars) = (@set pb.par = pars)

getPeriodicOrbit(prob::WrapPOColl, u::AbstractVector, p) = getPeriodicOrbit(prob.prob, u, p)
getPeriodicOrbit(prob::WrapPOSh, u::AbstractVector, p) = getPeriodicOrbit(prob.prob, u, p)

# update a problem with arguments
function updateForBS(prob::AbstractPeriodicOrbitProblem, F, dF, hopfpt, ζr, M, orbitguess_a, period) end

# update a section with a problem, we do nothing by default
updateSection!(prob::AbstractPeriodicOrbitProblem, x, par) = @warn "Not yet implemented!"

Base.size(pb::AbstractPOFDProblem) = (pb.M, pb.N)
onGpu(pb::AbstractPOFDProblem) = pb.ongpu
hasHessian(pb::AbstractPOFDProblem) = pb.d2F == nothing
isInplace(pb::AbstractPOFDProblem) = isInplace(pb.prob_vf)

function applyF(pb, dest, x, p)
	if isInplace(pb)
		pb.prob_vf.VF.F(dest, x, p)
	else
		dest .= residual(pb.prob_vf, x, p)
	end
	dest
end

function applyJ(pb, dest, x, p, dx)
	if isInplace(pb)
		pb.prob_vf.VF.J(dest, x, p, dx)
	else
		dest .= apply(pb.prob_vf.VF.J(x, p), dx)
	end
	dest
end

# function to extract trajectories from branch
getPeriodicOrbit(br::AbstractBranchResult, ind::Int) = getPeriodicOrbit(br.prob, br.sol[ind].x, setParam(br, br.sol[ind].p))

"""
$(SIGNATURES)

This function generates an initial guess for the solution of the problem `pb` based on the orbit `t -> orbit(t)` for t ∈ [0,2π] and the period `period`.
"""
function generateSolution(pb::AbstractPeriodicOrbitProblem, orbit, period)
	M = getMeshSize(pb)
	orbitguess_a = [orbit(t) for t in LinRange(0, 2pi, M + 1)[1:M]]
	# append period at the end of the initial guess
	orbitguess_v = reduce(vcat, orbitguess_a)
	if pb  isa PoincareShootingProblem
		return vec(orbitguess_v)
	else
		return vcat(vec(orbitguess_v), period) |> vec
	end
end

"""
Structure to encode the solution associated to a functional like `::PeriodicOrbitOCollProblem` or `::ShootingProblem`. In the particular case of `::PeriodicOrbitOCollProblem`, this allows to use the collocation polynomials to interpolate the solution. Hence, if `sol::POSolution`, one can call

    sol = BifurcationKit.POSolution(prob_coll, x)
	sol(t)

on any time `t`.
"""
struct POSolution{Tpb, Tx, Tp}
	pb::Tpb
	x::Tx
	pars::Tp
end

# simplified constructor
POSolution(prob::AbstractPeriodicOrbitProblem, x) = POSolution(prob, x, nothing)
####################################################################################################
# this struct allows to have a unified interface with Shooting methods in term of plotting
@with_kw_noshow struct SolPeriodicOrbit{Ts, Tu}
	t::Ts
	u::Tu
end
Base.getindex(sol::SolPeriodicOrbit, i...) = getindex(sol.u, i...)
####################################################################################################
"""
$(TYPEDEF)

Define a structure to interface the Jacobian of the periodic orbits Problem with the Floquet computation methods. If we use the same code as for `newton` (see below) but in `continuation`, it is difficult to tell the eigensolver that it should not use the jacobian but the monodromy matrix instead.

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

# this is for use with BorderingBLS with checkPrecision = true
apply(shjac::FloquetWrapper, dx) = apply(shjac.jacpb, dx)

# specific linear solver to dispatch
struct FloquetWrapperLS{T} <: AbstractLinearSolver
	solver::T # use solver as a field is good for BLS
end
# this constructor prevents from having FloquetWrapperLS(FloquetWrapperLS(ls))
FloquetWrapperLS(ls::FloquetWrapperLS) = ls
(ls::FloquetWrapperLS)(J, rhs; kwargs...) = ls.solver(J, rhs; kwargs...)
(ls::FloquetWrapperLS)(J::FloquetWrapper, rhs; kwargs...) = ls.solver(J.jacpb, rhs; kwargs...)
(ls::FloquetWrapperLS)(J::FloquetWrapper, rhs1, rhs2) = ls.solver(J.jacpb, rhs1, rhs2)

# this is for the use of MatrixBLS
LinearAlgebra.hcat(shjac::FloquetWrapper, dR) = hcat(shjac.jacpb, dR)

####################################################################################################
const DocStrjacobianPOSh = """
- `jacobian` Specify the choice of the linear algorithm, which must belong to `[:autodiffMF, :MatrixFree, :autodiffDense, :autodiffDenseAnalytical, :FiniteDifferences]`. This is used to select a way of inverting the jacobian dG
    - For `:MatrixFree`, we use an iterative solver (e.g. GMRES) to solve the linear system. The jacobian was specified by the user in `prob`.
    - For `:autodiffMF`, we use iterative solver (e.g. GMRES) to solve the linear system. We use Automatic Differentiation to compute the (matrix-free) derivative of `x -> prob(x, p)`.
    - For `:autodiffDense`. Same as for `:autodiffMF` but the jacobian is formed as a dense Matrix. You can use a direct solver or an iterative one using `options`.
    - For `:FiniteDifferencesDense`, same as for `:autodiffDense` but we use Finite Differences to compute the jacobian of `x -> prob(x, p)` using the `δ = 1e-8` which can be passed as an argument.
    - For `:autodiffDenseAnalytical`. Same as for `:autodiffDense` but the jacobian is using a mix of AD and analytical formula.
    - For `:FiniteDifferences`, use Finite Differences to compute the jacobian of `x -> prob(x, p)` using the `δ = 1e-8` which can be passed as an argument.
"""
##########################
residual(prob::WrapPOSh, x, p) = prob.prob(x, p)
jacobian(prob::WrapPOSh, x, p) = prob.jacobian(x, p)
@inline isSymmetric(prob::WrapPOSh) = false

# newton wrapper
function buildJacobian(prob::AbstractShootingProblem, orbitguess, par; δ = convert(eltype(orbitguess), 1e-8))
	jacobianPO = prob.jacobian
	if jacobianPO == :autodiffDenseAnalytical
		_J = prob(Val(:JacobianMatrix), orbitguess, par)
		jac = (x, p) -> prob(Val(:JacobianMatrixInplace), _J, x, p)
	elseif jacobianPO == :autodiffDense
		jac = (x, p) -> ForwardDiff.jacobian(z -> prob(z, p), x)
	elseif jacobianPO == :autodiffMF
		jac = (x, p) -> (dx -> ForwardDiff.derivative(z -> prob((@. x + z * dx), p), 0))
	elseif jacobianPO == :FiniteDifferencesDense
		jac = (x, p) -> finiteDifferences(z -> prob(z, p), x; δ = δ)
	elseif jacobianPO == :FiniteDifferences
		jac = (x, p) -> dx -> (prob(x .+ δ .* dx, p) .- prob(x .- δ .* dx, p)) ./ (2δ)
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
$DocStrjacobianPOSh
"""
function newton(prob::AbstractShootingProblem,
				orbitguess,
				options::NewtonPar;
				lens::Union{Lens, Nothing} = nothing,
				δ = convert(eltype(orbitguess), 1e-8),
				kwargs...)
	jac = buildJacobian(prob, orbitguess, getParams(prob); δ = δ)
	probw = WrapPOSh(prob, jac, orbitguess, getParams(prob), lens, nothing, nothing)
	return newton(probw, options; kwargs...)
end

"""
$(SIGNATURES)

This is the deflated Newton-Krylov Solver for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar to [`newton`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref).

# Optional argument
$DocStrjacobianPOSh

# Output:
- solution::NonLinearSolution, see [`NonLinearSolution`](@ref)
"""
function newton(prob::AbstractShootingProblem,
				orbitguess::vectype,
				defOp::DeflationOperator{Tp, Tdot, T, vectype},
				options::NewtonPar{T, S, E};
				lens::Union{Lens, Nothing} = nothing,
				kwargs...,
			) where {T, Tp, Tdot, vectype, S, E}
	jac = buildJacobian(prob, orbitguess, getParams(prob))
	probw = WrapPOSh(prob, jac, orbitguess, getParams(prob), lens, nothing, nothing)
	return newton(probw, defOp, options; kwargs...)
end

####################################################################################################
# Continuation for shooting problems

"""
$(SIGNATURES)

This is the continuation method for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar to [`continuation`](@ref) except that `probPO` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref). By default, it prints the period of the periodic orbit.

# Optional arguments
- `eigsolver` specify an eigen solver for the computation of the Floquet exponents, defaults to `FloquetQaD`
$DocStrjacobianPOSh
"""
function continuation(probPO::AbstractShootingProblem, orbitguess,
						alg::AbstractContinuationAlgorithm,
						contParams::ContinuationPar,
						linearAlgo::AbstractBorderedLinearSolver;
						δ = convert(eltype(orbitguess), 1e-8),
						eigsolver = FloquetQaD(contParams.newtonOptions.eigsolver),
						kwargs...,
					)
	jacobianPO = probPO.jacobian
	@assert ~isnothing(getLens(probPO)) "You need to provide a lens for your periodic orbit problem."
	@assert jacobianPO in
			(:autodiffMF, :MatrixFree, :autodiffDense, :autodiffDenseAnalytical, :FiniteDifferencesDense, :FiniteDifferences) "This jacobian is not defined. Please chose another one."

	if computeEigenElements(contParams)
		contParams = @set contParams.newtonOptions.eigsolver = eigsolver
	end

	options = contParams.newtonOptions

	# change the user provided finalise function by passing probPO in its parameters
	_finsol = modifyPOFinalise(probPO, kwargs, probPO.updateSectionEveryStep)
	_recordsol = modifyPORecord(probPO, kwargs, getParams(probPO), getLens(probPO))
	_plotsol = modifyPOPlot(probPO, kwargs)

	if jacobianPO == :autodiffDenseAnalytical
		_J = probPO(Val(:JacobianMatrix), orbitguess, getParams(probPO))
		jac = (x, p) -> (probPO(Val(:JacobianMatrixInplace), _J, x, p); FloquetWrapper(probPO, _J, x, p));
	elseif jacobianPO == :autodiffDense
		jac = (x, p) -> FloquetWrapper(probPO, ForwardDiff.jacobian(z -> probPO(z, p), x), x, p)
	elseif jacobianPO == :FiniteDifferencesDense
		jac = (x, p) -> FloquetWrapper(probPO, finiteDifferences(z -> probPO(z, p), x), x, p)
	elseif jacobianPO == :autodiffMF
		jac = (x, p) -> FloquetWrapper(probPO, (dx -> ForwardDiff.derivative(z -> probPO(x .+ z .* dx, p), 0)), x, p)
	elseif jacobianPO == :FiniteDifferences
		jac = (x, p) -> FloquetWrapper(probPO, dx -> (probPO(x .+ δ .* dx, p) .- probPO(x .- δ .* dx, p)) ./ (2δ), x, p)
	else
		jac = (x, p) -> FloquetWrapper(probPO, x, p)
	end

	# VERIFIER LE LINEARALGO VS COLLOCATION PAR EXAMPLE

	# we have to change the Bordered linearsolver to cope with our type FloquetWrapper
	linearAlgo = @set linearAlgo.solver = FloquetWrapperLS(linearAlgo.solver)
	alg = update(alg, contParams, linearAlgo)

	probwp = WrapPOSh(probPO, jac, orbitguess, getParams(probPO), getLens(probPO), _plotsol, _recordsol)

	br = continuation(
		probwp, alg,
		(@set contParams.newtonOptions.linsolver = FloquetWrapperLS(options.linsolver));
		kwargs...,
		kind = PeriodicOrbitCont(),
		finaliseSolution = _finsol)
	return br
end

"""
$(SIGNATURES)

This is the continuation routine for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar to [`continuation`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref). By default, it prints the period of the periodic orbit.

# Optional argument
- `linearAlgo::AbstractBorderedLinearSolver`
$DocStrjacobianPOSh

"""
function continuation(prob::AbstractPeriodicOrbitProblem,
					orbitguess,
					alg::AbstractContinuationAlgorithm,
					_contParams::ContinuationPar;
					linearAlgo = nothing,
					kwargs...)
	_linearAlgo = isnothing(linearAlgo) ?  MatrixBLS() : linearAlgo
	return continuation(prob, orbitguess, alg, _contParams, _linearAlgo; kwargs...)
end

####################################################################################################
"""
$(SIGNATURES)

Perform automatic branch switching from a Hopf bifurcation point labelled `ind_bif` in the list of the bifurcated points of a previously computed branch `br::ContResult`. It first computes a Hopf normal form.

# Arguments

- `br` branch result from a call to `continuation`
- `ind_hopf` index of the bifurcation point in `br`
- `contParams` parameters for the call to `continuation`
- `probPO` problem used to specify the way the periodic orbit is computed. It can be [`PeriodicOrbitTrapProblem`](@ref), [`ShootingProblem`](@ref) or [`PoincareShootingProblem`](@ref) .

# Optional arguments

- `alg = br.alg` continuation algorithm
- `δp` used to specify a particular guess for the parameter on the bifurcated branch which is otherwise determined by `contParams.ds`. This allows to use a step larger than `contParams.dsmax`.
- `ampfactor = 1` factor which alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch
- `nev` number of eigenvalues to be computed to get the right eigenvector
- all `kwargs` from [`continuation`](@ref)

A modified version of `prob` is passed to `plotSolution` and `finaliseSolution`.

!!! note "Linear solver"
	You have to be careful about the options `contParams.newtonOptions.linsolver`. In the case of Matrix-Free solver, you have to pass the right number of unknowns `N * M + 1`. Note that the options for the preconditioner are not accessible yet.
"""
function continuation(br::AbstractBranchResult, ind_bif::Int,
					_contParams::ContinuationPar,
					probPO::AbstractPeriodicOrbitProblem ;
					alg = br.alg,
					δp = nothing,
					ampfactor = 1,
					usedeflation = false,
					nev = length(eigenvalsfrombif(br, ind_bif)),
					kwargs...)
	# compute the normal form of the branch point
	verbose = get(kwargs, :verbosity, 0) > 1 ? true : false
	verbose && (println("--> Considering bifurcation point:"); _show(stdout, br.specialpoint[ind_bif], ind_bif))

	cb = get(kwargs, :callbackN, cbDefault)

	hopfpt = hopfNormalForm(br.prob, br, ind_bif; nev = nev, verbose = verbose)

	# compute predictor for point on new branch
	ds = isnothing(δp) ? _contParams.ds : δp
	Ty = typeof(ds)
	pred = predictor(hopfpt, ds; verbose = verbose, ampfactor = Ty(ampfactor))

	verbose && printstyled(color = :green, "#"^61*
			"\n┌─ Start branching from Hopf bif. point to periodic orbits.",
			"\n├─ Bifurcation type = ", hopfpt.type,
			"\n├─── Hopf param  p0 = ", br.specialpoint[ind_bif].param,
			"\n├─── new param    p = ", pred.p, ", p - p0 = ", pred.p - br.specialpoint[ind_bif].param,
			"\n├─── amplitude p.o. = ", pred.amp,
			"\n├─── period       T = ", pred.period,
			"\n├─ Method = \n", probPO, "\n")

	# we compute a phase so that the constraint equation
	# < u(0) − u_hopf, ψ > is satisfied, i.e. equal to zero.
	ζr = real.(hopfpt.ζ)
	ζi = imag.(hopfpt.ζ)
	# this phase is for POTrap problem constraint to be satisfied
	ϕ = atan(dot(ζr, ζr), dot(ζi, ζr))
	verbose && printstyled(color = :green, "├─── phase ϕ        = ", ϕ / pi, "⋅π\n")

	M = getMeshSize(probPO)
	orbitguess_a = [pred.orbit(t - ϕ) for t in LinRange(0, 2pi, M + 1)[1:M]]

	# extract the vector field and use it possibly to affect the PO functional
	prob_vf = reMake(br.prob, params = setParam(br, pred.p))

	# build the variable to hold the functional for computing PO based on finite differences
	probPO, orbitguess = reMake(probPO, prob_vf, hopfpt, ζr, orbitguess_a, abs(2pi/pred.ω); orbit = pred.orbit)

	if _contParams.newtonOptions.linsolver isa GMRESIterativeSolvers
		_contParams = @set _contParams.newtonOptions.linsolver.N = length(orbitguess)
	end

	if usedeflation
		verbose &&
			println("\n├─ Attempt branch switching\n--> Compute point on the current branch...")
		probPO isa PoincareShootingProblem &&
			@warn "Poincaré Shooting does not work very well with stationary states."
		optn = _contParams.newtonOptions

		# we start with the case of zero amplitude
		orbitzeroamp_a = [hopfpt.x0 for _ = 1:M]
		# this factor prevent shooting jacobian from being singular at fixed points
		if probPO isa PoincareShootingProblem
			Tfactor = 0
		elseif probPO isa AbstractPOFDProblem
			Tfactor = 100 / abs(2pi / pred.ω)
		else
			Tfactor = 0.001
		end

		# TODO should only update guess here, cf Poincaré
		probPO0, orbitzeroamp = reMake(probPO, prob_vf, hopfpt, ζr, orbitzeroamp_a, Tfactor * abs(2pi / pred.ω))
		sol0 = newton(probPO0, orbitzeroamp, optn; callback = cb, kwargs...)

		# find the bifurcated branch using deflation
		if ~(probPO isa PoincareShootingProblem)
			deflationOp = DeflationOperator(2, (x, y) -> dot(x[1:end-1], y[1:end-1]), 1.0, [sol0.u]; autodiff = true)
		else
			deflationOp = DeflationOperator(2, (x, y) -> dot(x, y) / M, 1.0, [sol0.u]; autodiff = true)
		end

		verbose && println("\n--> Compute point on bifurcated branch...")
		solbif = newton(probPO, orbitguess, deflationOp, (@set optn.maxIter = 10 * optn.maxIter); callback = cb, kwargs...)
		@assert converged(solbif) "Deflated newton did not converge"
		orbitguess .= solbif.u

		branch = continuation(
			probPO, orbitguess, alg,
			_contParams;
			kwargs...,
		)

		return Branch(branch, hopfpt)
	end

	# perform continuation
	branch = continuation(
		probPO, orbitguess, alg,
		_contParams;
		kwargs...
	)

	return Branch(branch, hopfpt)
end

####################################################################################################
# Branch switching from bifurcations of periodic orbits
"""
$(SIGNATURES)

Branch switching at a bifurcation point on a branch of periodic orbits (PO) specified by a `br::AbstractBranchResult`. The functional used to compute the PO is `br.prob`. A deflated Newton-Krylov solver can be used to improve the branch switching capabilities.

# Arguments
- `br` branch of periodic orbits computed with a [`PeriodicOrbitTrapProblem`](@ref)
- `ind_bif` index of the branch point
- `_contParams` parameters to be used by a regular [`continuation`](@ref)

# Optional arguments
- `δp = 0.1` used to specify a particular guess for the parameter in the branch which is otherwise determined by `contParams.ds`. This allows to use a step larger than `contParams.dsmax`.
- `ampfactor = 1` factor which alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch
- `recordFromSolution = (u, p) -> u[end]`, record method used in the bifurcation diagram, by default this records the period of the periodic orbit.
- `linearAlgo = BorderingBLS()`, same as for [`continuation`](@ref)
- `kwargs` keywords arguments used for a call to the regular [`continuation`](@ref) and the ones specific to periodic orbits (POs).
"""
function continuation(br::AbstractResult{PeriodicOrbitCont, Tprob},
			ind_bif::Int,
			_contParams::ContinuationPar;
			alg = br.alg,
			δp = 0.1, ampfactor = 1,
			usedeflation = false,
			linearAlgo = nothing,
			kwargs...) where Tprob

	bifpt = br.specialpoint[ind_bif]
	bptype = bifpt.type
	@assert bptype in (:pd, :bp) "Branching from $(bptype) not possible yet."
	@assert abs(bifpt.δ[1]) == 1 "Only simple bifurcation points are handled"

	verbose = get(kwargs, :verbosity, 0) > 0
	verbose && printstyled(color = :green, "#"^61*
			"\n┌─ Start branching from $(bptype) point to periodic orbits.\n├─ Bifurcation type = ", bifpt.type,
			"\n├─── bif. param  p0 = ", bifpt.param,
			"\n├─── period at bif. = ", getPeriod(br.prob.prob, bifpt.x, setParam(br, bifpt.param)),
			"\n├─── new param    p = ", bifpt.param + δp, ", p - p0 = ", δp,
			"\n├─── amplitude p.o. = ", ampfactor,
			"\n")

	_linearAlgo = isnothing(linearAlgo) ? BorderingBLS(_contParams.newtonOptions.linsolver) : linearAlgo

	# we copy the problem for not mutating the one passed by the user. This is a AbstractPeriodicOrbitProblem.
	pb = deepcopy(br.prob.prob)

	nf = getNormalForm(br, ind_bif)
	pred = predictor(nf, δp, ampfactor)
	orbitguess = pred.orbitguess
	newp = pred.pnew  # new parameter value
	pbnew = pred.prob # mofified problem

	# # let us compute the kernel
	# λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
	# verbose && print("├─ computing nullspace of Periodic orbit problem...")
	# ζ = geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
	# # we normalize it by the sup norm because it could be too small/big in L2 norm
	# # TODO: user defined scaleζ
	# ζ ./= norm(ζ, Inf)
	# verbose && println("Done!")
	#
	# # compute the full eigenvector
	# floquetsolver = br.contparams.newtonOptions.eigsolver
	# ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setParam(br, bifpt.param), real.(ζ))
	# ζs = reduce(vcat, ζ_a)
	#
	# ## predictor
	# pbnew, orbitguess = predictor(pb, bifpt, ampfactor, real.(ζs), bifpt.type)
	# newp = bifpt.param + δp

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

	if usedeflation
		verbose && println("\n├─ Attempt branch switching\n--> Compute point on the current branch...")
		optn = _contParams.newtonOptions
		# find point on the first branch
		pbnew = setParamsPO(pbnew, setParam(br, newp))
		sol0 = newton(pbnew, bifpt.x, optn; kwargs...)

		# find the bifurcated branch using deflation
		@assert pbnew isa AbstractPOFDProblem || pbnew isa ShootingProblem "Deflated newton is not available for your problem. Try Trapezoid / collocation method or ShootingProblem"
		deflationOp = DeflationOperator(2, (x, y) -> dot(x[1:end-1], y[1:end-1]), 1.0, [sol0.u]; autodiff = true)
		verbose && println("\n--> Compute point on bifurcated branch...")
		solbif = newton(pbnew, orbitguess, deflationOp,
			(@set optn.maxIter = 10 * optn.maxIter) ; kwargs...,)
		@assert converged(solbif) "Deflated newton did not converge"
		orbitguess .= solbif.u
	end

	# perform continuation
	pbnew = setParamsPO(pbnew, setParam(br, newp))

	pbnew(orbitguess, setParam(br, newp))[end] |> abs > 1 && @warn "PO constraint not satisfied"

	branch = continuation( pbnew, orbitguess, alg, _contParams;
		kwargs..., # put this first to be overwritten just below!
		linearAlgo = _linearAlgo,
		kind = br.kind
	)

	# create a branch
	bppo = Pitchfork(bifpt.x, bifpt.param, setParam(br, bifpt.param), getLens(br), nf.ζ, nf.ζ, nothing, :nothing)

	# return Branch(setproperties(branch; prob = br.prob), bppo)
	return Branch(branch, bppo)
end
