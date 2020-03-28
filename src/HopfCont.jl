"""
For an initial guess from the index of a Hopf bifurcation point located in ContResult.bifpoint, returns a point which will be refined using `newtonHopf`.
"""
function HopfPoint(br::ContResult, index::Int64)
	@assert br.bifpoint[index].type == :hopf "The provided index does not refer to a Hopf point"
	bifpoint = br.bifpoint[index]							# Hopf point
	eigRes   = br.eig										# eigenvector at the Hopf point
	p = bifpoint.param										# parameter value at the Hopf point
	ω = abs(imag(eigRes[bifpoint.idx].eigenvals[bifpoint.ind_bif]))	# frequency at the Hopf point
	return BorderedArray(bifpoint.x, [p, ω] )
end

@with_kw struct HopfProblemMinimallyAugmented{TF, TJ, TJa, Td2f, vectype, S <: AbstractLinearSolver, Sbd <: AbstractBorderedLinearSolver, Sbda <: AbstractBorderedLinearSolver}
	F::TF 								# Function F(x, p) = 0
	J::TJ 								# Jacobian of F wrt x
	Jadjoint::TJa						# Adjoint of the Jacobian of F
	d2F::Td2f = nothing					# Hessian of F
	a::vectype							# close to null vector of (J - iω I)^*
	b::vectype							# close to null vector of J - iω I
	linsolver::S						# linear solver
	linbdsolver::Sbd					# linear bordered solver
	linbdsolverAdjoint::Sbda			# linear bordered solver for the jacobian adjoint
end

hasHessian(pb::HopfProblemMinimallyAugmented{TF, TJ, TJa, Td2f, vectype, S, Sbd, Sbda}) where {TF, TJ, TJa, Td2f, vectype, S, Sbd, Sbda} = Td2f != Nothing

hasAdjoint(pb::HopfProblemMinimallyAugmented{TF, TJ, TJa, Td2f, vectype, S, Sbd, Sbda}) where {TF, TJ, TJa, Td2f, vectype, S, Sbd, Sbda} = TJa != Nothing

HopfProblemMinimallyAugmented(F, J, Ja, a, b, linsolve) = HopfProblemMinimallyAugmented(F, J, Ja, nothing, a, b, linsolve, BorderingBLS(linsolve), BorderingBLS(linsolve))

HopfProblemMinimallyAugmented(F, J, Ja, d2F, a, b, linsolve) = HopfProblemMinimallyAugmented(F, J, Ja, d2F, a, b, linsolve, BorderingBLS(linsolve), BorderingBLS(linsolve))

function (fp::HopfProblemMinimallyAugmented)(x, p::T, ω::T) where {T}
	# These are the equations of the minimally augmented (MA) formulation of the Hopf bifurcation point
	# input:
	# - x guess for the point at which the jacobian has a purely imaginary eigenvalue
	# - p guess for the parameter for which the jacobian has a purely imaginary eigenvalue
	# The jacobian of the MA problem is solved with a bordering method
	a = fp.a
	b = fp.b

	# we solve (J+iω)v + a σ1 = 0 with <b, v> = n
	n = T(1)
	v, σ1, flag, it = fp.linbdsolver(fp.J(x, p),
							a, b,
							T(0), zero(x), n; shift = Complex{T}(0, ω))

	# we solve (J+iω)'w + b σ2 = 0 with <a, w> = n
	# we find sigma2 = conj(sigma1)
	# w, σ2, _ = fp.linbdsolver(fp.Jadjoint(x, p) - Complex(0, ω) * I, b, a, 0., zeros(N), n)

	# the constraint is σ = <w, Jv> / n
	# σ = -dot(w, apply(fp.J(x, p) + Complex(0, ω) * I, v)) / n
	# we should have σ = σ1

	return fp.F(x, p), real(σ1), imag(σ1)
end

function (hopfpb::HopfProblemMinimallyAugmented{TF, TJ, TJa, Td2f, vectypeC, S, Sbd, Sbda})(x::BorderedArray{vectypeR, T}) where {TF, TJ, TJa, Td2f, vectypeC, vectypeR, S, Sbd, Sbda, T}
	res = hopfpb(x.u, x.p[1], x.p[2])
	return BorderedArray(res[1], [res[2], res[3]])
end

# Struct to invert the jacobian of the Hopf MA problem.
struct HopfLinearSolveMinAug <: AbstractLinearSolver; end

"""
The function solve the linear problem associated with a linearization of the minimally augmented formulation of the Hopf bifurcation point. The keyword `debug_` is used to debug the routine by returning several key quantities.
"""
function hopfMALinearSolver(x, p::T, ω::T, pbMA::HopfProblemMinimallyAugmented,
	 						duu, dup, duω;
							debug_ = false) where T
	# N = length(du) - 2
	# The jacobian should be passed as a tuple as Jac_hopf_MA(u0, pb::HopfProblemMinimallyAugmented) = (return (u0, pb, d2F::Bool))
	# The Jacobian J of the vector field is expressed at (x, p)
	# the jacobian expression of the hopf problem Jhopf is
	#					[ J dpF   0
	#					 σx  σp  σω]
	########## Resolution of the bordered linear system ########
	# J * dX	  + dpF * dp		   = du => dX = x1 - dp * x2
	# The second equation
	#	<σx, dX> +  σp * dp + σω * dω = du[end-1:end]
	# thus becomes
	#   (σp - <σx, x2>) * dp + σω * dω = du[end-1:end] - <σx, x1>
	# This 2x2 system is then solved to get (dp, dω)
	############### Extraction of function names #################
	Fhandle = pbMA.F
	J = pbMA.J

	d2F = pbMA.d2F
	a = pbMA.a
	b = pbMA.b

	# we define the following jacobian. It is used at least 3 times below. This avoid doing 3 times the possibly costly building of J(x, p)
	J_at_xp = J(x, p)

	# we do the following to avoid computing J_at_xp twice in case pbMA.Jadjoint is not provided
	if hasAdjoint(pbMA)
		JAd_at_xp = pbMA.Jadjoint(x, p)
	else
		JAd_at_xp = transpose(J_at_xp)
	end

	δ = T(1e-9)
	ϵ1, ϵ2, ϵ3 = T(δ), T(δ), T(δ)

	# we solve Jv + a σ1 = 0 with <b, v> = n
	n = T(1)
	v, σ1, _, _ = pbMA.linbdsolver(J_at_xp, a, b, T(0), zero(x), n; shift = Complex{T}(0, ω))
	w, σ2, _, _ = pbMA.linbdsolverAdjoint(JAd_at_xp, b, a, T(0), zero(x), n; shift = -Complex{T}(0, ω))

	################### computation of σx σp ####################
	dpF   = (Fhandle(x, p + ϵ1)	 - Fhandle(x, p - ϵ1)) / T(2ϵ1)
	dJvdp = (apply(J(x, p + ϵ3), v) - apply(J(x, p - ϵ3), v)) / T(2ϵ3)
	σp = -dot(w, dJvdp) / n

	# case of sigma_omega
	σω = -dot(w, Complex{T}(0, 1) * v) / n

	x1, x2, _, (it1, it2) = pbMA.linsolver(J_at_xp, duu, dpF)

	# the case of ∂_xσ is a bit more involved
	# we first need to compute the value of ∂_xσ written σx
	# σx = zeros(Complex{T}, length(x))
	σx = similar(x, Complex{T})

	if hasHessian(pbMA) == false
		# We invert the jacobian of the Hopf problem when the Hessian of x -> F(x, p) is not known analytically. We thus rely on finite differences which can be slow for large dimensions
		prod(size(x)) > 1e4 && @warn "You might want to pass the Hessian, finite differences with $(prod(size(x))) unknowns"
		e = zero(x)
		for ii in CartesianIndices(x)
			e[ii] = T(1)
			d2Fve = (apply(J(x + ϵ2 * e, p), v) - apply(J(x - ϵ2 * e, p), v)) / T(2ϵ2)
			σx[ii] = -dot(w, d2Fve) / n
			e[ii] = T(0)
		end
		σxx1 = dot(σx, x1)
		σxx2 = dot(σx, x2)
	else
		d2Fv = d2F(x, p, v, x1)
		σxx1 = -dot(w, d2Fv) / n
		d2Fv = d2F(x, p, v, x2)
		σxx2 = -dot(w, d2Fv) / n
	end
	# we need to be carefull here because the dot produce conjugates. Hence the + dot(σx, x2) and + imag(dot(σx, x1) and not the opposite
	dp, dω = [real(σp - σxx2) real(σω);
			  imag(σp + σxx2) imag(σω) ] \
			  [dup - real(σxx1), duω + imag(σxx1)]

	if debug_
		return x1 - dp * x2, dp, dω, true, it1 + it2, (σx, σp, σω, dpF)
	else
		return x1 - dp * x2, dp, dω, true, it1 + it2
	end
end

function (hopfl::HopfLinearSolveMinAug)(Jhopf, du::BorderedArray{vectype, T}; debug_ = false)  where {vectype, T}
	out = hopfMALinearSolver((Jhopf[1]).u,
				(Jhopf[1]).p[1],
				(Jhopf[1]).p[2],
				Jhopf[2],
				du.u, du.p[1], du.p[2];
				debug_ = debug_)
	if debug_ == false
		return BorderedArray(out[1], [out[2], out[3]]), out[4], out[5]
	else
		return BorderedArray(out[1], [out[2], out[3]]), out[4], out[5], out[6]
	end
end

################################################################################################### Newton / Continuation functions
"""
	`newtonHopf(F, J, hopfpointguess::BorderedArray{vectypeR, T}, eigenvec, eigenvec_ad, options::NewtonPar; Jt = nothing, d2F = nothing, normN = norm)`

This function turns an initial guess for a Hopf point into a solution to the Hopf problem based on a Minimally Augmented formulation. The arguments are as follows
- `F  = (x, p) -> F(x, p)` where `p` is the parameter associated to the Hopf point
- `J  = (x, p) -> d_xF(x, p)` associated jacobian
- `hopfpointguess` initial guess (x_0, p_0) for the Hopf point. It should a `BorderedArray` as given by the function HopfPoint.
- `eigenvec` guess for the  iω eigenvector
- `eigenvec_ad` guess for the -iω eigenvector
- `options::NewtonPar`

Optional arguments:
- `Jt = (x, p) -> transpose(d_xF(x, p))` associated jacobian transpose
- `d2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2)` a bilinear operator representing the hessian of `F`. It has to provide an expression for `d2F(x,p)[v1, v2]`.
- `normN = norm`

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jt = nothing` by using `tranpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jt = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""
function newtonHopf(F, J, hopfpointguess::BorderedArray{vectypeR, T}, eigenvec, eigenvec_ad, options::NewtonPar; Jt = nothing, d2F = nothing, normN = norm) where {vectypeR, T}
	hopfvariable = HopfProblemMinimallyAugmented(
		F,
		J,
		Jt,
		d2F,
		copy(eigenvec),
		copy(eigenvec_ad),
		options.linsolver)
	hopfPb = u -> hopfvariable(u)

	# Jacobian for the Hopf problem
	Jac_hopf_MA(u0, pb::HopfProblemMinimallyAugmented) = (return (u0, pb))

	# options for the Newton Solver
	opt_hopf = @set options.linsolver = HopfLinearSolveMinAug()

	# solve the hopf equations
	return newton(x ->  hopfPb(x),
				x -> Jac_hopf_MA(x, hopfvariable),
				hopfpointguess,
				opt_hopf, normN = normN)
end

"""
Simplified call to refine an initial guess for a Hopf point. More precisely, the call is as follows

	`newtonHopf(F, J, br::ContResult, index::Int64, options; Jt = nothing, d2F = nothing, normN = norm)`

where the optional argument `Jt` is the jacobian transpose and the Hessian is `d2F`. The parameters are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.

!!! warning "Eigenvectors"
    This simplified call has been written when the eigenvectors are organised in a 2d Array `evec` where `evec[:,2]` is the second eigenvector in the list.

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jt = nothing` by using `tranpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jt = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""
function newtonHopf(F, J, br::ContResult, ind_hopf::Int64, options::NewtonPar ; Jt = nothing, d2F = nothing, normN = norm)
	hopfpointguess = HopfPoint(br, ind_hopf)
	bifpt = br.bifpoint[ind_hopf]
	options.verbose && println("--> Newton Hopf, the eigenvalue considered here is ", br.eig[bifpt.idx].eigenvals[bifpt.ind_bif])
	@assert bifpt.idx == bifpt.step + 1 "Error, the bifurcation index does not refer to the correct step"
	eigenvec = geteigenvector(options.eigsolver ,br.eig[bifpt.idx].eigenvec, bifpt.ind_bif)
	eigenvec_ad = conj.(eigenvec)

	# solve the hopf equations
	outhopf, hist, flag =  newtonHopf(F, J, hopfpointguess, eigenvec_ad, eigenvec, options; Jt = Jt, d2F = d2F, normN = normN)
	return outhopf, hist, flag
end

"""
codim 2 continuation of Hopf points. This function turns an initial guess for a Hopf point into a curve of Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `(x, p1, p2)-> F(x, p1, p2)` where `p` is the parameter associated to the hopf point
- `J = (x, p1, p2)-> d_xF(x, p1, p2)` associated jacobian
- `hopfpointguess` initial guess (x_0, p1_0) for the Hopf point. It should be a `Vector` or a `BorderedArray`
- `p2` parameter p2 for which hopfpointguess is a good guess
- `eigenvec` guess for the iω eigenvector at p1_0
- `eigenvec_ad` guess for the -iω eigenvector at p1_0
- `options::NewtonPar`

Optional arguments:

- `Jt = (x, p1, p2) -> transpose(d_xF(x, p1, p2))` associated jacobian transpose
- `d2F = p2 -> ((x, p1, v1, v2) -> d2F(x, p1, p2, v1, v2))` this is the hessian of `F` computed at `(x, p1, p2)` and evaluated at `(v1, v2)`.

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jt = nothing` by using `tranpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jt = (x, p1, p2) -> transpose(d_xF(x, p1, p2))` otherwise the jacobian will be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""
function continuationHopf(F, J, hopfpointguess::BorderedArray{vectype, Tb}, p2_0::T, eigenvec, eigenvec_ad, options_cont::ContinuationPar ; Jt = nothing, d2F = p2 -> nothing, kwargs...) where {T,Tb,vectype}
	# Bad way it creates a struct for every p2?
	# Jacobian for the hopf problem
	Jac_hopf_MA(u0, pb) = (return (u0, pb))

	# options for the Newton Solver inheritated from the ones the user provided
	options_newton = options_cont.newtonOptions

	if isnothing(Jt)
		hopfvariable = p2 -> HopfProblemMinimallyAugmented(
			(x, p1) -> F(x, p1, p2),
			(x, p1) -> J(x, p1, p2),
			nothing,
			d2F(p2),
			copy(eigenvec),
			copy(eigenvec_ad),
			options_newton.linsolver)
		hopfPb = (u, p2) -> hopfvariable(p2)(u)
	else
		hopfvariable = p2 -> HopfProblemMinimallyAugmented(
			(x, p1) -> F(x, p1, p2),
			(x, p1) -> J(x, p1, p2),
			(x, p1) -> Jt(x, p1, p2),
			d2F(p2),
			copy(eigenvec),
			copy(eigenvec_ad),
			options_newton.linsolver)
		hopfPb = (u, p2) -> hopfvariable(p2)(u)
	end

	opt_hopf_cont = @set options_cont.newtonOptions.linsolver = HopfLinearSolveMinAug()

	# solve the hopf equations
	return continuation(
		(x, p2) -> hopfPb(x, p2),
		(x, p2) -> Jac_hopf_MA(x, hopfvariable(p2)),
		hopfpointguess, p2_0,
		opt_hopf_cont,
		plot = true,
		printSolution = (u, p) -> u.p[1],
		plotSolution = (x, p;kwargs...) -> (xlabel!("p2", subplot=1); ylabel!("p1", subplot=1)  ) ; kwargs...)
end

"""
Simplified call for continuation of Hopf point. More precisely, the call is as follows `continuationHopf(F, J, br::ContResult, index::Int64, options; Jt = nothing, d2F = p2 -> nothing)` where the parameters are as for `continuationHopf` except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.

Simplified calls are also provided but at the cost of using finite differences.

!!! warning "Eigenvectors"
    This simplified call has been written when the eigenvectors are organised in a 2d Array `evec` where `evec[:,2]` is the second eigenvector in the list.

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jt = nothing` by using `tranpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jt = (x, p1, p2) -> transpose(d_xF(x, p1, p2))` otherwise the jacobian will be computed twice!

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""
function continuationHopf(F, J, br::ContResult, ind_hopf::Int64, p2_0::Real, options_cont::ContinuationPar ;  Jt = nothing, d2F = p2 -> nothing, kwargs...)
	hopfpointguess = HopfPoint(br, ind_hopf)
	bifpt = br.bifpoint[ind_hopf]
	eigenvec = geteigenvector(options_cont.newtonOptions.eigsolver ,br.eig[bifpt.idx].eigenvec, bifpt.ind_bif)
	eigenvec_ad = conj.(eigenvec)
	return continuationHopf(F, J, hopfpointguess, p2_0, eigenvec_ad, eigenvec, options_cont ; Jt = Jt, d2F = d2F, kwargs...)
end
