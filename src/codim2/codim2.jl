abstract type AbstractProblemMinimallyAugmented end
abstract type AbstractCodim2EigenSolver <: AbstractEigenSolver end

@inline getLens(pb::AbstractProblemMinimallyAugmented) = pb.lens
getsolver(eig::AbstractCodim2EigenSolver) = eig.eigsolver

for op in (:FoldProblemMinimallyAugmented, :HopfProblemMinimallyAugmented)
	@eval begin
		"""
		$(TYPEDEF)

		Structure to encode Fold / Hopf functional based on a Minimally Augmented formulation.

		# Fields

		$(FIELDS)
		"""
		struct $op{TF, TJ, TJa, Td2f, Tl <: Lens, vectype, S <: AbstractLinearSolver, Sa <: AbstractLinearSolver, Sbd <: AbstractBorderedLinearSolver, Sbda <: AbstractBorderedLinearSolver} <: AbstractProblemMinimallyAugmented
			"Function F(x, p) = 0"
			F::TF
			"Jacobian of F w.r.t. x"
			J::TJ
			"Adjoint of the Jacobian of F"
			Jᵗ::TJa
			"Hessian of F"
			d2F::Td2f
			"parameter axis for the codim 2 point"
			lens::Tl
			"close to null vector of Jᵗ"
			a::vectype
			"close to null vector of J"
			b::vectype
			"vector zero, to avoid allocating it many times"
			zero::vectype
			"linear solver. Used to invert the jacobian of MA functional"
			linsolver::S
			"linear solver for the jacobian adjoint"
			linsolverAdjoint::Sa
			"bordered linear solver"
			linbdsolver::Sbd
			"linear bordered solver for the jacobian adjoint"
			linbdsolverAdjoint::Sbda
			"whether the Jacobian is Symmetric, avoid computing Jᵗ"
			issymmetric::Bool
		end

		@inline hasHessian(pb::$op{TF, TJ, TJa, Td2f, Tl, vectype, S, Sa, Sbd, Sbda}) where {TF, TJ, TJa, Td2f, Tp, Tl, vectype, S, Sa, Sbd, Sbda} = Td2f != Nothing

		@inline hasAdjoint(pb::$op{TF, TJ, TJa, Td2f, Tl, vectype, S, Sa, Sbd, Sbda}) where {TF, TJ, TJa, Td2f, Tl, vectype, S, Sa, Sbd, Sbda} = TJa != Nothing
	end
end

@inline issymmetric(pb::AbstractProblemMinimallyAugmented) = pb.issymmetric


function applyJacobian(pb::AbstractProblemMinimallyAugmented, x, par, dx, transposeJac = false)
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

function detectCodim2Parameters(detectCodim2Bifurcation, options_cont; kwargs...)
	if detectCodim2Bifurcation > 0
		if get(kwargs, :updateMinAugEveryStep, 0) == 0
			@error "You ask for detection of codim2 bifurcations but passed the option `updateMinAugEveryStep = 0`. The bifurcation detection algorithm may not work faithfully. Please use `updateMinAugEveryStep > 0`."
		end
		return setproperties(options_cont; detectBifurcation = 0, detectEvent = detectCodim2Bifurcation, detectFold = false)
	else
		return options_cont
	end
end

"""
$(SIGNATURES)

This function turns an initial guess for a Fold/Hopf point into a solution to the Fold/Hopf problem based on a Minimally Augmented formulation. The arguments are as follows
- `F  = (x, p) -> F(x, p)` where `p` is a set of parameters.
- `J  = (x, p) -> d_xF(x, p)` associated jacobian
- `br` results returned after a call to [continuation](@ref Library-Continuation)
- `ind_bif` bifurcation index in `br`
- `lens` parameter axis used to locate the Fold/Hopf point.
- `options::NewtonPar`

# Optional arguments:
- `issymmetric` whether the Jacobian is Symmetric (for Fold)
- `Jᵗ = (x, p) -> transpose(d_xF(x, p))` jacobian adjoint, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of Matrix / Sparse based jacobian, `Jᵗ` should not be passed as it is computed internally more efficiently, i.e. it avoids recomputing the jacobian as it would be if you pass `Jᵗ = (x, p) -> transpose(dF(x, p))`
- `d2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2)` a bilinear operator representing the hessian of `F`. It has to provide an expression for `d2F(x,p)[v1,v2]`.
- `normN = norm`
- `options` You can pass newton parameters different from the ones stored in `br` by using this argument `options`.
- `bdlinsolver` bordered linear solver for the constraint equation
- `startWithEigen = false` whether to start the Minimally Augmented problem with information from eigen elements
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

!!! tip "ODE problems"
    For ODE problems, it is more efficient to use the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`
"""
function newton(F, J, br::AbstractBranchResult, ind_bif::Int64; Jᵗ = nothing, d2F = nothing, normN = norm, options = br.contparams.newtonOptions, startWithEigen = false, issymmetric::Bool = false, kwargs...)
	@assert length(br.specialpoint) > 0 "The branch does not contain bifurcation points"
	if br.specialpoint[ind_bif].type == :hopf
		d2Fc = isnothing(d2F) ? nothing : (x,p,dx1,dx2) -> BilinearMap((_dx1, _dx2) -> d2F(x,p,_dx1,_dx2))(dx1,dx2)
		return newtonHopf(F, J, br, ind_bif; Jᵗ = Jᵗ, d2F = d2Fc, normN = normN, options = options, startWithEigen = startWithEigen, kwargs...)
	else
		return newtonFold(F, J, br, ind_bif; issymmetric = issymmetric, Jᵗ = Jᵗ, d2F = d2F, normN = normN, options = options, startWithEigen = startWithEigen, kwargs...)
	end
end

"""
$(SIGNATURES)

codim 2 continuation of Fold / Hopf points. This function turns an initial guess for a Fold/Hopf point into a curve of Fold/Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p) ->	F(x, p)` where `p` is a set of parameters
- `J = (x, p) -> d_xF(x, p)` associated jacobian
- `br` results returned after a call to [continuation](@ref Library-Continuation)
- `ind_bif` bifurcation index in `br`
- `lens2` parameters used for the vector field
- `options_cont = br.contparams` arguments to be passed to the regular [continuation](@ref Library-Continuation)

# Optional arguments:
- `issymmetric` whether the Jacobian is Symmetric (for Fold)
- `Jᵗ = (x, p) -> transpose(d_xF(x, p))` associated jacobian transpose
- `d2F = (x, p, v1, v2) -> d2F(x, p, v1, v2)` this is the hessian of `F` computed at `(x, p)` and evaluated at `(v1, v2)`. This helps solving the linear problem associated to the minimally augmented formulation.
- `d3F = (x, p, v1, v2, v3) -> d3F(x, p, v1, v2, v3)` this is the third derivative of `F` computed at `(x, p)` and evaluated at `(v1, v2, v3)`. This is used to detect **Bautin** bifurcation.
- `bdlinsolver` bordered linear solver for the constraint equation
- `updateMinAugEveryStep` update vectors `a,b` in Minimally Formulation every `updateMinAugEveryStep` steps
- `startWithEigen = false` whether to start the Minimally Augmented problem with information from eigen elements
- `detectCodim2Bifurcation ∈ {0,1,2}` whether to detect Bogdanov-Takens, Bautin and Cusp. If equals `1` non precise detection is used. If equals `2`, a bisection method is used to locate the bifurcations.
- `kwargs` keywords arguments to be passed to the regular [continuation](@ref Library-Continuation)

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Hopf point in `br` you want to refine.

!!! tip "Jacobian transpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`
"""
function continuation(F, J,
				br::AbstractBranchResult, ind_bif::Int64,
				lens2::Lens, options_cont::ContinuationPar = br.contparams ;
				startWithEigen = false,
				issymmetric::Bool = false,
				Jᵗ = nothing,
				d2F = nothing,
				d3F = nothing,
				detectCodim2Bifurcation::Int = 0,
				kwargs...)
	@assert length(br.specialpoint) > 0 "The branch does not contain bifurcation points"
	# options to detect codim2 bifurcations
	computeEigenElements = options_cont.detectBifurcation > 0
	_options_cont = detectCodim2Parameters(detectCodim2Bifurcation, options_cont; kwargs...)

	if br.specialpoint[ind_bif].type == :hopf
		# redefine the multilinear form to accept complex arguments
		d2Fc = isnothing(d2F) ? nothing : (x,p,dx1,dx2) -> BilinearMap((_dx1, _dx2) -> d2F(x,p,_dx1,_dx2))(dx1,dx2)
		d3Fc = isnothing(d3F) ? nothing : (x,p,dx1,dx2,dx3) -> TrilinearMap((_dx1, _dx2, _dx3) -> d3F(x,p,_dx1,_dx2,_dx3))(dx1,dx2,dx3)
		return continuationHopf(F, J, br, ind_bif, lens2, _options_cont; Jᵗ = Jᵗ, d2F = d2Fc, d3F = d3Fc, startWithEigen = startWithEigen, computeEigenElements = computeEigenElements, kwargs...)
	else
		return continuationFold(F, J, br, ind_bif, lens2, _options_cont;
			issymmetric = issymmetric,
			Jᵗ = Jᵗ,
			d2F = d2F,
			startWithEigen = startWithEigen,
			computeEigenElements = computeEigenElements,
			kwargs...)
	end
end
####################################################################################################
# branch switching at bt
function continuation(F, dF, d2F, d3F,
			br::ContResult{Ta, Teigvals, Teigvecbr, Biftype, Ts, Tparc, Tfunc, Tpar, Tl},
			ind_bif::Int, options_cont::ContinuationPar = br.contparams;
			Jᵗ = nothing,
			δ::Real = 1e-8, δp = nothing, ampfactor::Real = 1,
			nev = options_cont.nev,
			issymmetric = false,
			detectCodim2Bifurcation::Int = 0,
			Teigvec = getvectortype(br),
			scaleζ = norm,
			startWithEigen = false,
			autodiff = false,
			kwargs...) where {Ta, Teigvals, Teigvecbr, Biftype, Ts, Tparc, Tfunc <: AbstractProblemMinimallyAugmented, Tpar, Tl <: Lens}

		verbose = get(kwargs, :verbosity, 0) > 0 ? true : false
		verbose && println("--> Considering bifurcation point:"); _show(stdout, br.specialpoint[ind_bif], ind_bif)

		@assert br.specialpoint[ind_bif].type in (:bt,:zh) "Only branching from Bogdanov-Takens and Zero-Hopf (for now)"

		# functional
		prob = br.functional

		# continuation parameters
		computeEigenElements = options_cont.detectBifurcation > 0
		optionsCont = detectCodim2Parameters(detectCodim2Bifurcation, options_cont; kwargs...)

		# higher order differentials
		d2Fc = isnothing(d2F) ? nothing : (x,p,dx1,dx2) -> BilinearMap((_dx1, _dx2) -> d2F(x,p,_dx1,_dx2))(dx1,dx2)
		d3Fc = isnothing(d3F) ? nothing : (x,p,dx1,dx2,dx3) -> TrilinearMap((_dx1, _dx2, _dx3) -> d3F(x,p,_dx1,_dx2,_dx3))(dx1,dx2,dx3)

		# scalar type
		Ty = eltype(Teigvec)

		# compute the normal form of the bifurcation point
		nf = computeNormalForm(F, dF, d2F, d3F, br, ind_bif; Jᵗ = Jᵗ, δ = δ, nev = nev, verbose = verbose, issymmetric = issymmetric, Teigvec = Teigvec, scaleζ = scaleζ, autodiff = autodiff)

		# compute predictor for point on new branch
		ds = isnothing(δp) ? optionsCont.ds : δp

		if prob isa FoldProblemMinimallyAugmented
			# define guess for the first Hopf point on the branch
			pred = predictor(nf, Val(:HopfCurve), ds)

			# new continuation parameters
			parcont = pred.hopf(ds)

			# new full parameters
			params = set(set(nf.params, nf.lens[2], parcont[2]), nf.lens[1], parcont[1])

			# guess for the Hopf point
			hopfpt = BorderedArray(nf.x0 .+ pred.x0(ds), [parcont[1], pred.ω(ds)])

			# estimates for eigenvectors for ±iω
			ζ = pred.EigenVec(ds)
			ζstar = pred.EigenVecAd(ds)

			# put back original options
			@set! optionsCont.newtonOptions.eigsolver =
								getsolver(optionsCont.newtonOptions.eigsolver)
			@set! optionsCont.newtonOptions.linsolver = prob.linsolver

			branch, u, τ = continuationHopf(F, dF, hopfpt, params,
					nf.lens...,
					ζ, ζstar,
					optionsCont;
					d2F = d2Fc, d3F = d3Fc,
					bdlinsolver = prob.linbdsolver,
					startWithEigen = startWithEigen,
					computeEigenElements = computeEigenElements,
					kwargs...
					)
			return Branch(branch, nf), u, τ

		else
			@assert prob isa HopfProblemMinimallyAugmented
			pred = predictor(nf, Val(:FoldCurve), 0.)
			# new continuation parameters
			parcont = pred.fold(ds)

			# new full parameters
			params = set(set(nf.params, nf.lens[2], parcont[2]), nf.lens[1], parcont[1])

			# guess for the fold point
			foldpt = BorderedArray(nf.x0 .+ 0 .* pred.x0(ds), parcont[1])

			# estimates for null eigenvectors
			ζ = pred.EigenVec(ds)
			ζstar = pred.EigenVecAd(ds)

			# put back original options
			@set! optionsCont.newtonOptions.eigsolver =
								getsolver(optionsCont.newtonOptions.eigsolver)
			@set! optionsCont.newtonOptions.linsolver = prob.linsolver
			# @set! optionsCont.detectBifurcation = 0
			# @set! optionsCont.detectEvent = 0

			branch, u, τ = continuationFold(F, dF, foldpt, params,
					nf.lens...,
					ζstar, ζ,
					optionsCont;
					d2F = d2Fc, #d3F = d3Fc,
					bdlinsolver = prob.linbdsolver,
					startWithEigen = startWithEigen,
					computeEigenElements = computeEigenElements,
					kwargs...
					)
			return Branch(branch, nf), u, τ
		end
end

"""
$(SIGNATURES)

This function uses information in the branch to detect codim 2 bifurcations like BT, ZH and Cusp.
"""
function correctBifurcation(contres::ContResult)
	if contres.functional isa AbstractProblemMinimallyAugmented == false
		return contres
	end
	if contres.functional isa FoldProblemMinimallyAugmented
		conversion = Dict(:bp => :bt, :hopf => :zh, :fold => :cusp, :nd => :nd)
	elseif contres.functional isa HopfProblemMinimallyAugmented
		conversion = Dict(:bp => :zh, :hopf => :hh, :fold => :nd, :nd => :nd, :ghbt => :bt)
	else
		throw("Error! this should not occur. Please open an issue on the website of BifurcationKit.jl")
	end
	for (ind, bp) in pairs(contres.specialpoint)
		if bp.type in keys(conversion)
			@set! contres.specialpoint[ind].type = conversion[bp.type]
		end
	end
	return contres
end
