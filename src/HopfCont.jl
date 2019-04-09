using Parameters
"""
For an initial guess from the index of a bifurcation point located in ContResult.bifpoint
"""
function HopfPoint(br::ContResult, index::Int64)
	@assert br.bifpoint[index][1] == :hopf "The index provided does not refer to a Hopf point"
	bifpoint = br.bifpoint[index]
	eigRes   = br.eig
	p = bifpoint[3]
	ω = abs(imag(eigRes[bifpoint[2]][1][bifpoint[end]]))
	return BorderedVector(bifpoint[5], [p, ω] )
end

struct HopfProblemMinimallyAugmented{vectype, S <: LinearSolver}
    F 					# Function F(x, p) = 0
    J 					# Jacobian of F wrt x
    Jadjoint  			# Adjoint of the Jacobian of F
    a::vectype			# close to null vector of (J - iω I)^*
    b::vectype			# close to null vector of J - iω I
    linsolve::S
end

function (fp::HopfProblemMinimallyAugmented{vectype, S})(x, p::T, ω::T) where {vectype, S <: LinearSolver, T}
    # These are minimally augmented turning point equations
    # The jacobian will be inverted using a bordering method
    a = fp.a
    b = fp.b

    # we solve (J+iω)v + a σ1 = 0 with <b, v> = n
    n = T(1)
    v, σ1, _ = linearBorderedSolver(fp.J(x, p),
	 								a, b,
									T(0), zero(x), n,
									fp.linsolve; shift = Complex{T}(0, ω))

    # we solve (J+iω)'w + b σ2 = 0 with <a, w> = n
	# we find sigma2 = conj(sigma1)
    # w, σ2, _ = linearBorderedSolver(fp.J(x, p) - Complex(0, ω) * I, b, a, 0., zeros(N), n, fp.linsolve)

    # the constraint is σ = <w, Jv> / n
    # σ = -dot(w, apply(fp.J(x, p) + Complex(0, ω) * I, v)) / n
	# we should have σ = σ1

    return fp.F(x, p), real(σ1), imag(σ1)
end

function (hopfpb::HopfProblemMinimallyAugmented{vectypeC, S})(x::BorderedVector{vectypeR, T}) where {vectypeC, vectypeR, S <: LinearSolver, T}
	res = hopfpb(x.u, x.p[1], x.p[2])
	return BorderedVector(res[1], [res[2], res[3]])
end

# Method to solve the associated linear system
@with_kw struct HopfLinearSolveMinAug <: LinearSolver
	# whether the Hessian is known analytically
	d2F_is_known = false
end

function hopfMALinearSolver(x, p::T, ω::T, pbMA::HopfProblemMinimallyAugmented,
	 						duu, dup, duω, d2F;
							debug_ = false,
							d2F_is_known = false) where T
    # N = length(du) - 2
    # the jacobian should just be a tuple
    # the Jacobian J is expressed at (x, p)
    # the jacobian expression of the hopf problem is
	#					[ J dpF   0
	#					 σx  σp  σω]
	########## Resolution of the bordered linear system ########
	# J * dX      + dpF * dp           = du => dX = x1 - dp * x2
	# The second equation
	#    <σx, dX> +  σp * dp + σω * dω = du[end-1:end]
	# thus becomes
	#   (σp - <σx, x2>) * dp + σω * dω = du[end-1:end] - <σx, x1>
	# This 2x2 system is then solved to get (dp, dω)
    ############### Extraction of function names #################
    Fhandle = pbMA.F
    J = pbMA.J
    Jadjoint = pbMA.Jadjoint
    a = pbMA.a
    b = pbMA.b

    δ = T(1e-9)
    ϵ1, ϵ2, ϵ3 = T(δ), T(δ), T(δ)

    # we solve Jv + a σ1 = 0 with <b, v> = n
    n = T(1)
    v, σ1, _ = linearBorderedSolver(J(x, p), a, b, T(0), zero(x), n, pbMA.linsolve; shift = Complex{T}(0, ω))

    w, σ2, _ = linearBorderedSolver(Jadjoint(x, p), b, a, T(0), zero(x), n, pbMA.linsolve; shift = -Complex{T}(0, ω))

    ################### computation of σx σp ####################
    dpF   = (Fhandle(x, p + ϵ1)     - Fhandle(x, p - ϵ1)) / T(2ϵ1)
    dJvdp = (apply(J(x, p + ϵ3), v) - apply(J(x, p - ϵ3), v)) / T(2ϵ3)
    σp = -dot(w, dJvdp) / n

	# case of sigma_omega
	σω = -dot(w, Complex{T}(0, 1) * v) / n

	x1, _, it1 = pbMA.linsolve(J(x, p), duu)
	x2, _, it2 = pbMA.linsolve(J(x, p), dpF)

    # the case of ∂_xσ is a bit more involved
	# we first need to compute the value of ∂_xσ written σx
	# σx = zeros(Complex{T}, length(x))
	σx = similar(x, Complex{T})

	if d2F_is_known == false
		# We invert the jacobian of the Fold problem when the Hessian of x -> F(x, p) is not known analytically. We thus rely on finite differences which can be slow for large dimensions
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

function (hopfl::HopfLinearSolveMinAug)(Jhopf, du::BorderedVector{vectype, T}; debug_ = false)  where {vectype, T}
	out = hopfMALinearSolver((Jhopf[1]).u, (Jhopf[1]).p[1], (Jhopf[1]).p[2], Jhopf[2],
		 						du.u, du.p[1], du.p[2],
								Jhopf[3]; 		# -> this is the hessian d2F;
								debug_ = debug_,
								d2F_is_known = hopfl.d2F_is_known)
	if debug_ == false
		return BorderedVector(out[1], [out[2], out[3]]), out[4], out[5]
	else
		return BorderedVector(out[1], [out[2], out[3]]), out[4], out[5], out[6]
	end
end

"""
This function turns an initial guess for a Hopf point into a solution to the Hopf problem based on a Minimally Augmented formulation. The arguments are as follows
- `F  = (x, p) -> F(x, p)` where `p` is the parameter associated to the Hopf point
- `J  = (x, p) -> d_xF(x, p)` associated jacobian
- `Jt = (x, p) -> transpose(d_xF(x, p))` associated jacobian
- `d2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2)` a bilinear operator representing the hessian of `F`. It has to provide an expression for `d2F(x,p)[v1,v2]`.
- `hopfpointguess` initial guess (x_0, p_0) for the Hopf point. It should be a `AbstractVector` or a `BorderedVector`.
- `eigenvec` guess for the  iω eigenvector
- `eigenvec_ad` guess for the -iω eigenvector
- `options::NewtonPar`
"""
function newtonHopf(F, J, Jt, d2F, hopfpointguess::BorderedVector{vectypeR, T}, eigenvec, eigenvec_ad, options::NewtonPar; normN = norm, d2F_is_known = true) where {vectypeR, T}
	hopfvariable = HopfProblemMinimallyAugmented(
		(x, p) -> F(x, p),
		(x, p) -> J(x, p),
		(x, p) -> Jt(x, p),
		eigenvec,
		eigenvec_ad,
		options.linsolve)
	hopfPb = u -> hopfvariable(u)
	# Jacobian for the Hopf problem
	# Jac_hopf_fdMA(u0) = Cont.finiteDifferences( u-> hopfPb(u), u0)
	Jac_hopf_MA(u0, pb::HopfProblemMinimallyAugmented) = (return (u0, pb, d2F))

	# options for the Newton Solver
	opt_hopf = @set options.linsolve = HopfLinearSolveMinAug(d2F_is_known = d2F_is_known)

	# solve the hopf equations
	return newton(x ->  hopfPb(x),
		x -> Jac_hopf_MA(x, hopfvariable),
		hopfpointguess,
		opt_hopf, normN = normN)
end

"""
call when hessian is unknown, finite differences are then used
"""
newtonHopf(F, J, Jt, hopfpointguess::BorderedVector{vectype, T}, eigenvec, eigenvec_ad, options::NewtonPar; normN = norm) where {T,vectype} = newtonHopf(F, J, Jt, x -> x, hopfpointguess, eigenvec, eigenvec_ad, options; normN = normN, d2F_is_known = false)

"""
Simplified call to refine an initial guess for a Hopf point. More precisely, the call is as follows `newtonHopf(F, J, Jt, d2F, br::ContResult, index::Int64, options)` where the parameters are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.

!!! warning "Eigenvectors`"
    This simplified call has been written when the eigenvectors are organised in a 2d Array `evec` where `evec[:,2]` is the second eigenvector in the list.
"""
function newtonHopf(F, J, Jt, d2F, br::ContResult, ind_hopf::Int64, options::NewtonPar ; d2F_is_known = true)
	hopfpointguess = HopfPoint(br, ind_hopf)
	bifpt = br.bifpoint[ind_hopf]
	options.verbose && println("--> Newton Hopf, the eigenvalue considered here is ", br.eig[bifpt[2]][1][bifpt[end]])
	eigenvec = getEigenVector(options.eigsolve ,br.eig[bifpt[2]][2] ,bifpt[end])
	eigenvec_ad = conj.(eigenvec)

	# solve the hopf equations
	outhopf, hist, flag =  newtonHopf(F, J, Jt, d2F, hopfpointguess, eigenvec, eigenvec_ad, options; d2F_is_known = d2F_is_known)
	return outhopf, hist, flag
end

newtonHopf(F, J, Jt, br::ContResult, ind_hopf::Int64, options::NewtonPar) =  newtonHopf(F, J, Jt, x -> x, br, ind_hopf, options::NewtonPar ; d2F_is_known = false)

newtonHopf(F, J, br::ContResult, ind_hopf::Int64, options::NewtonPar) = newtonHopf(F, J, (x, p)->transpose(J(x, p)), br, ind_hopf, options)

"""
codim 2 continuation of Hopf points. This function turns an initial guess for a Hopf point into a curve of Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `(x, p1, p2)-> F(x, p1, p2)` where `p` is the parameter associated to the hopf point
- `J = (x, p1, p2)-> d_xF(x, p1, p2)` associated jacobian
- `Jt = (x, p1, p2) -> transpose(d_xF(x, p1, p2))` associated jacobian
- `d2F = (x, p1, p2, v1, v2) -> d2F(x, p1, p2, v1, v2)` this is the hessian of `F` computed at `(x, p1, p2)` and evaluated at `(v1, v2)`.
- `hopfpointguess` initial guess (x_0, p1_0) for the Hopf point. It should be a `Vector` or a `BorderedVector`
- `p2` parameter p2 for which hopfpointguess is a good guess
- `eigenvec` guess for the iω eigenvector at p1_0
- `eigenvec_ad` guess for the -iω eigenvector at p1_0
- `options::NewtonPar`
"""
function continuationHopf(F, J, Jt, d2F, hopfpointguess::BorderedVector{vectype, Tb}, p2_0::T, eigenvec, eigenvec_ad, options_cont::ContinuationPar ; d2F_is_known = true, kwargs...) where {T,Tb,vectype}
	@warn "Bad way it creates a struct for every p2"
	# Jacobian for the hopf problem
	Jac_hopf_MA(u0, pb, hess) = (return (u0, pb, hess))

	# options for the Newton Solver inheritated from the ones the user provided
	options_newton = options_cont.newtonOptions

	hopfvariable = p2 -> HopfProblemMinimallyAugmented(
		(x, p1) -> F(x, p1, p2),
		(x, p1) -> J(x, p1, p2),
		(x, p1) -> Jt(x, p1, p2),
		eigenvec,
		eigenvec_ad,
		options_newton.linsolve)

	hopfPb = (u, p2) -> hopfvariable(p2)(u)
	println("--> Start Hopf continuation with Hessian known? = ", d2F_is_known)

	opt_hopf_cont = @set options_cont.newtonOptions.linsolve = HopfLinearSolveMinAug(d2F_is_known = d2F_is_known)

	# solve the hopf equations
	return continuation((x, p2) -> hopfPb(x, p2),
		(x, p2) -> Jac_hopf_MA(x, hopfvariable(p2), d2F(p2)),
		hopfpointguess, p2_0,
		opt_hopf_cont,
		plot = true,
		printsolution = u -> u.p[1],
		plotsolution = (x;kwargs...) -> (xlabel!("p2", subplot=1); ylabel!("p1", subplot=1)  ) ; kwargs...)
end

continuationHopf(F, J, Jt, hopfpointguess::BorderedVector{vectype, Tb}, p2_0::T, eigenvec, eigenvec_ad, options_cont::ContinuationPar ; kwargs...) where {T, Tb, vectype} = continuationHopf(F, J, Jt, x -> x, hopfpointguess, p2_0, eigenvec, eigenvec_ad, options_cont ; d2F_is_known = false, kwargs...)

continuationHopf(F, J, Jt, hopfpointguess::BorderedVector{vectype, Tb}, p2_0::T, eigenvec, eigenvec_ad, options_cont::ContinuationPar ; kwargs...) where {T, Tb, vectype} = continuationHopf(F, J, (x, p1, p2) -> transpose(J(x, p1, p2)), hopfpointguess, p2_0, eigenvec, eigenvec_ad, options_cont ; kwargs...)

"""
Simplified call for continuation of Hopf point. More precisely, the call is as follows `continuationHopf(F, J, Jt, d2F, br::ContResult, index::Int64, options)` where the parameters are as for `continuationHopf` except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.

Simplified calls are also provided but at the cost of using finite differences.

!!! warning "Eigenvectors`"
    This simplified call has been written when the eigenvectors are organised in a 2d Array `evec` where `evec[:,2]` is the second eigenvector in the list.
"""
function continuationHopf(F, J, Jt, d2F, br::ContResult, ind_hopf::Int64, p2_0::Real, options_cont::ContinuationPar ; d2F_is_known = true, kwargs...)
	hopfpointguess = HopfPoint(br, ind_hopf)
	bifpt = br.bifpoint[ind_hopf]
	eigenvec = getEigenVector(options_cont.newtonOptions.eigsolve ,br.eig[bifpt[2]][2] ,bifpt[end])
	eigenvec_ad = conj.(eigenvec)
	return continuationHopf(F, J, Jt, d2F, hopfpointguess, p2_0, eigenvec, eigenvec_ad, options_cont ; d2F_is_known = d2F_is_known, kwargs...)
end

continuationHopf(F, J, Jt, br::ContResult, ind_hopf::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...) = continuationHopf(F, J, Jt, x-> x, br, ind_hopf, p2_0, options_cont ; d2F_is_known = false, kwargs...)

continuationHopf(F, J, br::ContResult, ind_hopf::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...) = continuationHopf(F, J, (x, p1, p2) -> transpose(J(x, p1, p2)), br, ind_hopf, p2_0, options_cont ; d2F_is_known = false, kwargs...)
