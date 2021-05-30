"""
For an initial guess from the index of a Fold bifurcation point located in ContResult.specialpoint, returns a point which will be refined using `newtonFold`.
"""
function FoldPoint(br::AbstractBranchResult, index::Int64)
	bptype = br.specialpoint[index].type
	@assert bptype == :bp || bptype == :nd || bptype == :fold "This should be a Fold / BP point"
	specialpoint = br.specialpoint[index]
	return BorderedArray(_copy(specialpoint.x), specialpoint.param)
end
####################################################################################################
"""
$(TYPEDEF)

Structure to encode Fold functional based on a Minimally Augmented formulation.

# Fields

$(FIELDS)
"""
struct FoldProblemMinimallyAugmented{TF, TJ, TJa, Td2f, Tl <: Lens, vectype, S <: AbstractLinearSolver, Sa <: AbstractLinearSolver, Sbd <: AbstractBorderedLinearSolver} <: ProblemMinimallyAugmented
	"Function F(x, p) = 0"
	F::TF
	"Jacobian of F w.r.t. x"
	J::TJ
	"Adjoint of the Jacobian of F"
	Jᵗ::TJa
	"Hessian of F"
	d2F::Td2f
	"parameter axis for the Fold point"
	lens::Tl
	"close to null vector of Jᵗ"
	a::vectype
	"close to null vector of J"
	b::vectype
	"vector zero, to avoid allocating it many times"
	zero::vectype
	"linear solver. Used to invert the jacobian of MA functional."
	linsolver::S
	"linear solver for the jacobian adjoint"
	linsolverAdjoint::Sa
	"bordered linear solver"
	linbdsolver::Sbd
	"whether the Jacobian is Symmetric, avoid computing Jᵗ"
	issymmetric::Bool
end

FoldProblemMinimallyAugmented(F, J, Ja, d2F, lens::Lens, a, b, issymmetric::Bool, linsolve::AbstractLinearSolver, linbdsolver = BorderingBLS(linsolve)) = FoldProblemMinimallyAugmented(F, J, Ja, d2F, lens, a, b, 0*a, linsolve, linsolve, linbdsolver, issymmetric)
FoldProblemMinimallyAugmented(F, J, Ja, d2F, lens::Lens, a, b, linsolve::AbstractLinearSolver, linbdsolver = BorderingBLS(linsolve)) = FoldProblemMinimallyAugmented(F, J, Ja, d2F, lens, a, b, false, linsolve, linbdsolver)

FoldProblemMinimallyAugmented(F, J, Ja, lens::Lens, a, b, issymmetric::Bool, linsolve::AbstractLinearSolver, linbdsolver = BorderingBLS(linsolve)) = FoldProblemMinimallyAugmented(F, J, Ja, nothing, lens, a, b, 0*a, linsolve, linbdsolver, issymmetric)
FoldProblemMinimallyAugmented(F, J, Ja, lens::Lens, a, b, linsolve::AbstractLinearSolver, linbdsolver = BorderingBLS(linsolve)) = FoldProblemMinimallyAugmented(F, J, Ja, lens, a, b, false, linsolve, linbdsolver)


@inline issymmetric(pb::FoldProblemMinimallyAugmented) = pb.issymmetric

@inline hasHessian(pb::FoldProblemMinimallyAugmented{TF, TJ, TJa, Td2f, Tl, vectype, S, Sa, Sbd}) where {TF, TJ, TJa, Td2f, Tp, Tl, vectype, S, Sa, Sbd} = Td2f != Nothing

@inline hasAdjoint(pb::FoldProblemMinimallyAugmented{TF, TJ, TJa, Td2f, Tl, vectype, S, Sa, Sbd}) where {TF, TJ, TJa, Td2f, Tl, vectype, S, Sa, Sbd} = TJa != Nothing

function applyJacobian(pb::ProblemMinimallyAugmented, x, par, dx, transposeJac = false)
	if issymmetric(pb)
		return apply(pb.J(x, par), dx)
	else
		if transposeJac == false
			return apply(pb.J(x, par), dx)
		else
			if hasAdjoint(pb)
				return apply(pb.Jᵗ(x, par), dx)
			else
				return apply(transpose(pb.J(x, par)), dx)
			end
		end
	end
end

function (fp::FoldProblemMinimallyAugmented)(x, p::T, _par) where {T}
	# These are the equations of the minimally augmented (MA) formulation of the Fold bifurcation point
	# input:
	# - x guess for the point at which the jacobian is singular
	# - p guess for the parameter value `<: Real` at which the jacobian is singular
	# The jacobian of the MA problem is solved with a bordering method
	a = fp.a
	b = fp.b
	# update parameter
	par = set(_par, fp.lens, p)
	# ┌      ┐┌  ┐ ┌ ┐
	# │ J  a ││v │=│0│
	# │ b  0 ││σ1│ │1│
	# └      ┘└  ┘ └ ┘
	# In the notations of Govaerts 2000, a = w, b = v
	# Thus, b should be a null vector of J
	#       a should be a null vector of J'
	# we solve Jv + a σ1 = 0 with <b, v> = n
	# the solution is v = -σ1 J\a with σ1 = -n/<b, J^{-1}a>
	n = T(1)
	J = fp.J(x, par)
	σ1 = fp.linbdsolver(J, a, b, T(0), fp.zero, n)[2]

	return fp.F(x, par), σ1
end

# this function is for the functional
function (foldpb::FoldProblemMinimallyAugmented)(x::BorderedArray, params)
	res = foldpb(x.u, x.p, params)
	return BorderedArray(res[1], res[2])
end

# Struct to invert the jacobian of the fold MA problem.
struct FoldLinearSolverMinAug <: AbstractLinearSolver; end

function foldMALinearSolver(x, p::T, pb::FoldProblemMinimallyAugmented, par,
							rhsu, rhsp;
							debugArray = nothing) where T
	################################################################################################
	# debugArray is used as a temp to be filled with values used for debugging. If debugArray = nothing, then no debugging mode is entered. If it is AbstractArray, then it is used
	################################################################################################
	# recall that  the functional we want to solve is [F(x,p), σ(x,p)] where  σ(x,p) is computed in the previous function.
	# The jacobian has to be passed as a tuple as Jac_fold_MA(u0, pb::FoldProblemMinimallyAugmented) = (return (u0, pb, d2F::Bool))
	# The Jacobian J of the vector field is expressed at (x, p)
	# We solve here Jfold⋅res = rhs := [rhsu, rhsp]
	# The Jacobian expression of the Fold problem is
	#           ┌         ┐
	#  Jfold =  │  J  dpF │
	#           │ σx   σp │
	#           └         ┘
	# where σx := ∂_xσ and σp := ∂_pσ
	# We recall the expression of σx = -< w, d2F(x,p)[v, x2]> where (w, σ2) is solution of J'w + b σ2 = 0 with <a, w> = n
	########################## Extraction of function names ########################################
	F = pb.F
	J = pb.J

	d2F = pb.d2F
	a = pb.a
	b = pb.b

	# parameter axis
	lens = pb.lens
	# update parameter
	par0 = set(par, lens, p)

	# we define the following jacobian. It is used at least 3 times below. This avoids doing 3 times the (possibly) costly building of J(x, p)
	J_at_xp = J(x, par0)

	# we do the following in order to avoid computing J_at_xp twice in case pb.Jadjoint is not provided
	if issymmetric(pb)
		JAd_at_xp = J_at_xp
	else
	JAd_at_xp = hasAdjoint(pb) ? pb.Jᵗ(x, par0) : transpose(J_at_xp)
	end

	# normalization
	n = T(1)

	# we solve Jv + a σ1 = 0 with <b, v> = n
	# the solution is v = -σ1 J\a with σ1 = -n/<b, J\a>
	v, σ1, _, itv = pb.linbdsolver(J_at_xp, a, b, T(0), pb.zero, n)

	# we solve J'w + b σ2 = 0 with <a, w> = n
	# the solution is w = -σ2 J'\b with σ2 = -n/<a, J'\b>
	w, σ2, _, itw = pb.linbdsolver(JAd_at_xp, b, a, T(0), pb.zero, n)

	δ = T(1e-8)
	ϵ1, ϵ2, ϵ3 = T(δ), T(δ), T(δ)
	###################  computation of σx σp  ####################
	################### and inversion of Jfold ####################
	dpF = minus(F(x, set(par, lens, p + ϵ1)), F(x, set(par, lens, p - ϵ1))); rmul!(dpF, T(1) / T(2ϵ1))
	dJvdp = minus(apply(J(x, set(par, lens, p + ϵ3)), v), apply(J(x, set(par, lens, p - ϵ3)), v)); rmul!(dJvdp, T(1) / T(2ϵ3))
	σp = -dot(w, dJvdp) / n

	if hasHessian(pb) == false
		# We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is not known analytically.
		u1 = applyJacobian(pb, x + ϵ2 * v, par0, w, true)
		u2 = apply(JAd_at_xp, w)
		σx = minus(u2, u1); rmul!(σx, 1 / ϵ2)
		########## Resolution of the bordered linear system ########
		# we invert Jfold
		dX, dsig, flag, it = pb.linbdsolver(J_at_xp, dpF, σx, σp, rhsu, rhsp)
	else
		# We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is known analytically.

		# we solve it here instead of calling linearBorderedSolver because this removes the need to pass the linear form associated to σx
		# !!! Carefull, this method makes the linear system singular
		x1, x2, _, it = pb.linsolver(J_at_xp, rhsu, dpF)

		d2Fv = d2F(x, par0, x1, v)
		σx1 = -dot(w, d2Fv ) / n

		copyto!(d2Fv, d2F(x, par0, x2, v))
		σx2 = -dot(w, d2Fv ) / n

		dsig = (rhsp - σx1) / (σp - σx2)

		# dX = x1 .- dsig .* x2
		dX = _copy(x1); axpy!(-dsig, x2, dX)
	end

	if debugArray isa AbstractArray
		debugArray .= [J(x, par0) dpF ; σx' σp]
	end

	return dX, dsig, true, sum(it) + sum(itv) + sum(itw)
end

function (foldl::FoldLinearSolverMinAug)(Jfold, du::BorderedArray{vectype, T}; debugArray = nothing) where {vectype, T}
	out =  foldMALinearSolver((Jfold.x).u,
				 (Jfold.x).p,
				 Jfold.fldpb,
				 Jfold.params,
				 du.u, du.p;
				 debugArray = debugArray)
	return BorderedArray{vectype, T}(out[1], out[2]), out[3], out[4]
end

################################################################################################### Newton / Continuation functions
"""
($SIGNATURES)

This function turns an initial guess for a Fold point into a solution to the Fold problem based on a Minimally Augmented formulation. The arguments are as follows
- `F   = (x, p) -> F(x, p)` where `p` is a set of parameters.
- `dF  = (x, p) -> d_xF(x, p)` associated jacobian
- `foldpointguess` initial guess (x_0, p_0) for the Fold point. It should be a `BorderedArray` as returned by the function `FoldPoint`
- `par` parameters used for the vector field
- `lens` parameter axis used to locate the Fold point.
- `eigenvec` guess for the 0 eigenvector
- `eigenvec_ad` guess for the 0 adjoint eigenvector
- `options::NewtonPar` options for the Newton-Krylov algorithm, see [`NewtonPar`](@ref).

# Optional arguments:
- `issymmetric` whether the Jacobian is Symmetric
- `Jᵗ = (x, p) -> transpose(d_xF(x, p))` jacobian adjoint, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jᵗ` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jᵗ = (x, p) -> transpose(dF(x, p))`
- `d2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2)` a bilinear operator representing the hessian of `F`. It has to provide an expression for `d2F(x,p)[v1,v2]`.
- `normN = norm`
- `bdlinsolver` bordered linear solver for the constraint equation
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

# Simplified call
Simplified call to refine an initial guess for a Fold point. More precisely, the call is as follows

	newtonFold(F, J, br::AbstractBranchResult, ind_fold::Int64, lens::Lens; options = br.contparams.newtonOptions, kwargs...)

where the optional argument `Jᵗ` is the jacobian transpose and the Hessian is `d2F`. The parameters / options are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine. You can pass newton parameters different from the ones stored in `br` by using the argument `options`.

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences.
"""
function newtonFold(F, J,
				foldpointguess, par,
				lens::Lens,
				eigenvec, eigenvec_ad,
				options::NewtonPar;
				 normN = norm,
				issymmetric = false,
				Jᵗ = nothing,
				d2F = nothing,
				bdlinsolver::AbstractBorderedLinearSolver = BorderingBLS(options.linsolver),
				kwargs...) where {T, vectype}

	foldproblem = FoldProblemMinimallyAugmented(
		F, J, Jᵗ, d2F,
		lens,
		_copy(eigenvec),
		_copy(eigenvec_ad), issymmetric,
		options.linsolver, @set bdlinsolver.solver = options.linsolver)

	# Jacobian for the Fold problem
	Jac_fold_MA = (x, param) -> (x = x, params = param, fldpb = foldproblem)

	opt_fold = @set options.linsolver = FoldLinearSolverMinAug()

	# solve the Fold equations
	return newton(foldproblem, Jac_fold_MA, foldpointguess, par, opt_fold; normN = normN, kwargs...)..., foldproblem
end

function newtonFold(F, J,
				br::AbstractBranchResult, ind_fold::Int64;
				issymmetric = false,
				Jᵗ = nothing,
				d2F = nothing,
				options = br.contparams.newtonOptions,
				nev = br.contparams.nev,
				startWithEigen = false,
				kwargs...)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.specialpoint[ind_fold]
	eigenvec = bifpt.tau.u
	eigenvec_ad = _copy(eigenvec)

	if startWithEigen
		λ = zero(getvectoreltype(br))
		p = bifpt.param
		parbif = setParam(br, p)

		# jacobian at bifurcation point
		L = J(bifpt.x, parbif)

		# computation of zero eigenvector
		ζstar, = getAdjointBasis(L, λ, br.contparams.newtonOptions.eigsolver; nev = nev, verbose = false)
		eigenvec .= real.(ζstar)

		# computation of adjoint eigenvector
		_Jt = isnothing(Jᵗ) ? adjoint(L) : Jᵗ(bifpt.x, parbif)
		ζstar, = getAdjointBasis(_Jt, λ, br.contparams.newtonOptions.eigsolver; nev = nev, verbose = false)
		eigenvec_ad .= real.(ζstar)
		eigenvec_ad ./= norm(eigenvec_ad)
	end

	# solve the Fold equations
	return newtonFold(F, J, foldpointguess, br.params, br.lens, eigenvec, eigenvec_ad, options; issymmetric = issymmetric, Jᵗ = Jᵗ, d2F = d2F, kwargs...)
end

"""
$(SIGNATURES)

Codim 2 continuation of Fold points. This function turns an initial guess for a Fold point into a curve of Fold points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p) ->	F(x, p)` where `p` is a set of parameters
- `J = (x, p) -> d_xF(x, p)` associated jacobian
- `foldpointguess` initial guess (x_0, p1_0) for the Fold point. It should be a `BorderedArray` as returned by the function `FoldPoint`
- `par` set of parameters
- `lens` parameter axis for parameter 2
- `eigenvec` guess for the 0 eigenvector at p1_0
- `eigenvec_ad` guess for the 0 adjoint eigenvector
- `options_cont` arguments to be passed to the regular [`continuation`](@ref)

# Optional arguments:
- `issymmetric` whether the Jacobian is Symmetric
- `Jᵗ = (x, p) -> transpose(d_xF(x, p))` associated jacobian transpose
- `d2F = (x, p, v1, v2) -> d2F(x, p, v1, v2)` this is the hessian of `F` computed at `(x, p)` and evaluated at `(v1, v2)`.
- `bdlinsolver` bordered linear solver for the constraint equation
- `updateMinAugEveryStep` update vectors `a, b` in Minimally Formulation every `updateMinAugEveryStep` steps
- `kwargs` keywords arguments to be passed to the regular [`continuation`](@ref)

# Simplified call
The call is as follows

	continuationFold(F, J, br::AbstractBranchResult, ind_fold::Int64, lens2::Lens, options_cont::ContinuationPar ; Jᵗ = nothing, d2F = nothing, nev = br.contparams.nev, startWithEigen = false, kwargs...)

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Fold point in `br` you want to continue.

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences.
"""
function continuationFold(F, J,
				foldpointguess::BorderedArray{vectype, T}, par,
				lens1::Lens, lens2::Lens,
				eigenvec, eigenvec_ad,
				options_cont::ContinuationPar ;
				updateMinAugEveryStep = 0,
				issymmetric = false,
				Jᵗ = nothing,
				d2F = nothing,
				bdlinsolver::AbstractBorderedLinearSolver = BorderingBLS(options_cont.newtonOptions.linsolver),
				kwargs...) where {T, vectype}
	@assert lens1 != lens2 "Please choose 2 diffferent parameters"

	# options for the Newton Solver inheritated from the ones the user provided
	options_newton = options_cont.newtonOptions

	foldPb = FoldProblemMinimallyAugmented(
			F, J, Jᵗ, d2F,
			lens1,
			_copy(eigenvec),
			_copy(eigenvec_ad), issymmetric,
			options_newton.linsolver, @set bdlinsolver.solver = options_newton.linsolver)

	# Jacobian for the Fold problem
	Jac_fold_MA = (x, param) -> (x = x, params = param, fldpb = foldPb)

	opt_fold_cont = @set options_cont.newtonOptions.linsolver = FoldLinearSolverMinAug()

	# this functions allows to tackle the case where the two parameters have the same name
	lenses = getLensSymbol(lens1, lens2)

	# this function is used as a Finalizer
	function updateMinAugFold(z, tau, step, contResult; kwargs...)
		~modCounter(step, updateMinAugEveryStep) && return true
		x = z.u.u	# fold point
		p1 = z.u.p	# first parameter
		p2 = z.p	# second parameter
		newpar = set(par, lens1, p1)
		newpar = set(newpar, lens2, p2)

		a = foldPb.a
		b = foldPb.b

		# expression of the jacobian
		J_at_xp = foldPb.J(x, newpar)

		# compute new b
		newb = foldPb.linbdsolver(J_at_xp, a, b, T(0), foldPb.zero, T(1))[1]

		# compute new a
		JAd_at_xp = hasAdjoint(foldPb) ? foldPb.Jᵗ(x, newpar) : transpose(J_at_xp)
		newa = foldPb.linbdsolver(JAd_at_xp, b, a, T(0), foldPb.zero, T(1))[1]

		foldPb.a .= newa ./ norm(newa)
		# do not normalise with dot(newb, foldPb.a), it prevents BT  detection
		foldPb.b .= newb ./ norm(newb)

		return true
	end

	# it allows to append information specific to the codim 2 continuation to the user data
	_printsol = get(kwargs, :printSolution, nothing)
	_printsol2 = isnothing(_printsol) ?
		(u, p; kw...) -> (zip(lenses, (u.p, p))..., BT = dot(foldPb.a, foldPb.b)) :
		(u, p; kw...) -> (namedprintsol(_printsol(u, p;kw...))..., zip(lenses, (u.p, p))..., BT = dot(foldPb.a, foldPb.b),)

	# solve the Fold equations
	br, u, tau = continuation(
		foldPb, Jac_fold_MA,
		foldpointguess, par, lens2,
		(@set opt_fold_cont.newtonOptions.eigsolver = FoldEig(opt_fold_cont.newtonOptions.eigsolver));
		kwargs...,
		printSolution = _printsol2,
		finaliseSolution = updateMinAugFold,
		)

	return codim2FoldBifurcationPoints(setproperties(br; type = :FoldCodim2, functional = foldPb)), u, tau
end

function continuationFold(F, J,
				br::AbstractBranchResult, ind_fold::Int64,
				lens2::Lens,
				options_cont::ContinuationPar ;
				issymmetric = false,
				Jᵗ = nothing,
				d2F = nothing,
				nev = br.contparams.nev,
				startWithEigen = false,
				kwargs...)
	foldpointguess = FoldPoint(br, ind_fold)
	bifpt = br.specialpoint[ind_fold]
	eigenvec = bifpt.tau.u
	eigenvec_ad = _copy(eigenvec)

	p = bifpt.param
	parbif = setParam(br, p)
	if startWithEigen
		eigenvec .= real.(geteigenvector(options_cont.newtonOptions.eigsolver ,br.eig[bifpt.idx].eigenvec, bifpt.ind_ev))
		eigenvec ./= norm(eigenvec)

		# jacobian at bifurcation point
		L = J(bifpt.x, parbif)
		_Jt = isnothing(Jᵗ) ? transpose(L) : Jᵗ(bifpt.x, parbif)

		ζstar, λstar = getAdjointBasis(_Jt, 0, br.contparams.newtonOptions.eigsolver; nev = nev, verbose = options_cont.newtonOptions.verbose)
		eigenvec_ad = real.(ζstar)
		eigenvec_ad ./= dot(eigenvec, eigenvec_ad)
	end

	return continuationFold(F, J, foldpointguess, parbif, br.lens, lens2, eigenvec, eigenvec_ad, options_cont ; issymmetric = issymmetric, Jᵗ = Jᵗ, d2F = d2F, kwargs...)
end

struct FoldEig{S} <: AbstractEigenSolver
	eigsolver::S
end

function (eig::FoldEig)(Jma, nev; kwargs...)
	n = min(nev, length(Jma.x.u))
	J = Jma.fldpb.J(Jma.x.u, set(Jma.params, Jma.fldpb.lens, Jma.x.p))
	eigenelts = eig.eigsolver(J, n; kwargs...)
	return eigenelts
end

"""
$(SIGNATURES)

This function uses information in the branch to detect codim 2 bifurcations like BT, ZH and Cusp.
"""
function codim2FoldBifurcationPoints(contres::AbstractBranchResult)
	if contres.functional isa FoldProblemMinimallyAugmented == false
		return contres
	end
	conversion = Dict(:bp => :bt, :hopf => :zh, :fold => :cusp, :nd => :nd)
	for (ind, bp) in pairs(contres.specialpoint)
		if bp.type in keys(conversion)
			@set! contres.specialpoint[ind].type = conversion[bp.type]
		end
	end
	return contres
end
