"""
For an initial guess from the index of a bifurcation point located in ContResult.bifpoint
"""
function HopfPoint(br::ContResult, index::Int64)
	@assert br.bifpoint[index][1] == :hopf "The index provided does not refer to a Hopf point"
	bifpoint = br.bifpoint[index]
	eigRes   = br.eig
	return vcat(bifpoint[5], bifpoint[3], abs(imag(eigRes[bifpoint[2]][1][bifpoint[end]])))
end

struct HopfProblemMinimallyAugmented{vectype, S <: LinearSolver}
    F 					# Function F(x, p) = 0
    J 					# Jacobian of F wrt x
    Jadjoint  			# Adjoint of the Jacobian of F
    a::vectype			# close to null vector of (J - iω I)^*
    b::vectype			# close to null vector of J - iω I
    linsolve::S
end

function (fp::HopfProblemMinimallyAugmented{vectype, S})(u::Vector) where {vectype, S <: LinearSolver}
    # These are minimally augmented turning point equations
    # The jacobian will be solved using a minimally augmented method

    N = length(u)-2
    x = @view u[1:N]
    p = u[N+1]
    ω = u[N+2]

    a = fp.a
    b = fp.b

    # we solve (J+iω)v + a σ1 = 0 with <b, v> = n
    n = 1.0
    v, σ1, _ = linearBorderedSolver(fp.J(x, p) + Complex(0, ω) * I, a, b, 0., zeros(N), n, fp.linsolve)

    # we solve (J+iω)'w + b σ2 = 0 with <a, w> = n
	# we find sigma2 = conj(sigma1)
    # w, σ2, _ = linearBorderedSolver(fp.J(x, p) - Complex(0, ω) * I, b, a, 0., zeros(N), n, fp.linsolve)

    # the constraint is σ = <w, Jv> / n
    # σ = -dot(w, apply(fp.J(x, p) + Complex(0, ω) * I, v)) / n
	# we should have σ = σ1

    return vcat(fp.F(x, p), real(σ1), imag(σ1))
end


# Method to solve the associated linear system
mutable struct HopfLinearSolveMinAug <: LinearSolver end

function (hopfl::HopfLinearSolveMinAug)(Jhopf, du::Vector{T}, debug_ = false) where {T}
    N = length(du) - 2
    # the jacobian should just be a tuple
    # the Jacobian J is expressed at (x, p)
    # the jacobian expression of the hopf problem is
	#					[ J dpF   0
	#					 σx  σp  σω]
    ############### Extraction of function names #################
    x = @view Jhopf[1][1:N]
    p = Jhopf[1][N+1]
    ω = Jhopf[1][N+2]

    Fhandle = Jhopf[2].F
    J = Jhopf[2].J
    Jadjoint = Jhopf[2].Jadjoint
    a = Jhopf[2].a
    b = Jhopf[2].b

    δ = 1e-9
    ϵ1, ϵ2, ϵ3 = δ, δ, δ

    # we solve Jv + a σ1 = 0 with <b, v> = n
    n = 1.0
    v, σ1, _ = linearBorderedSolver(J(x, p) + Complex(0, ω) * I, a, b, 0., zeros(N), n, Jhopf[2].linsolve)

    w, σ2, _ = linearBorderedSolver(Jadjoint(x, p) - Complex(0, ω) * I, b, a, 0., zeros(N), n, Jhopf[2].linsolve)

    ################### computation of σx σp ####################
    dpF   = (Fhandle(x, p + ϵ1)     - Fhandle(x, p - ϵ1)) / (2ϵ1)
    dJvdp = (apply(J(x, p + ϵ3), v) - apply(J(x, p - ϵ3), v)) / (2ϵ3)
    σp = -dot(w, dJvdp) / n

    # the case of sigma_x is a bit more involved
    σx = zeros(Complex{Float64}, length(x))
    e = zero(x)
    for ii=1:N
        e .= 0 .* e
        e[ii] = 1.0
        d2Fve = (apply(J(x + ϵ2 * e, p), v) - apply(J(x - ϵ2 * e, p), v)) / (2ϵ2)
        σx[ii] = -dot(w, d2Fve) / n
    end

    # case of sigma_omega
    σω = -dot(w, Complex(0, 1.0) * v) / n

    ########## Resolution of the bordered linear system ########
	# J * dX + dF * dp  = du => dX = x1 - dp * x2
	#    <σx, dX> + σp * dp + σω * dω = du[end-2:end] hence
	#  (σp - <σx, x2>) * dp + σω * dω = du[end-2:end] - <σx, x1>

    x1, _, it1 = Jhopf[2].linsolve(J(x, p), du[1:N])
    x2, _, it2 = Jhopf[2].linsolve(J(x, p), dpF)

    # we need to be carefull here because the dot produce conjugates. Hence the + dot(σx, x2) and + imag(dot(σx, x1) and not the opposite
    dp, dω = [real(σp - dot(σx, x2)) real(σω);
              imag(σp + dot(σx, x2)) imag(σω) ] \
              [du[end-1] - real(dot(σx, x1)), du[end] + imag(dot(σx, x1))]


	if debug_
		return vcat(x1 - dp * x2, dp, dω), true, it1 + it2, (σx, σp, σω, dpF)
	else
		return vcat(x1 - dp * x2, dp, dω), true, it1 + it2
	end

end

"""
This function turns an initial guess for a Hopf point into a solution to the Hopf problem based on a Minimally Augmented formulation. The arguments are as follows
- `F  = (x, p) -> F(x, p)` where `p` is the parameter associated to the Hopf point
- `J  = (x, p) -> d_xF(x, p)` associated jacobian
- `Jt = (x, p) -> transpose(d_xF(x, p))` associated jacobian
- `hopfpointguess` initial guess (x_0, p_0) for the Hopf point. It should be a `AbstractVector` or a `BorderedVector`.
- `eigenvec` guess for the  iω eigenvector
- `eigenvec_ad` guess for the -iω eigenvector
- `options::NewtonPar`
"""
function newtonHopf(F, J, Jt, hopfpointguess::AbstractVector, eigenvec, eigenvec_ad, options::NewtonPar)
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
	Jac_hopf_MA(u0, pb::HopfProblemMinimallyAugmented) = (return (u0, pb))

	# options for the Newton Solver
	opt_hopf = @set options.linsolve = HopfLinearSolveMinAug()

	# solve the hopf equations
	return newton(x ->  hopfPb(x),
						x -> Jac_hopf_MA(x, hopfvariable),
						hopfpointguess,
						opt_hopf)
end

"""
Simplified call to refine an initial guess for a Hopf point. More precisely, the call is as follows `newtonHopf(F, J, Jt, br::ContResult, index::Int64, options)` where the parameters are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.

!!! warning "Eigenvectors`"
    This simplified call has been written when the eigenvectors are organised in a 2d Array `evec` where `evec[:,2]` is the second eigenvector in the list.
"""
function newtonHopf(F, J, Jt, br::ContResult, ind_hopf::Int64, options::NewtonPar)
	hopfpointguess = HopfPoint(br, ind_hopf)
	bifpt = br.bifpoint[ind_hopf]
	println("--> Newton Hopf, the eigenvalue considered here is ", br.eig[bifpt[2]][1][bifpt[end]])
	eigenvec = getEigenVector(options.eigsolve ,br.eig[bifpt[2]][2] ,bifpt[end])
	eigenvec_ad = conj.(eigenvec)

	# solve the hopf equations
	outhopf, hist, flag =  newtonHopf(F, J, Jt, hopfpointguess, eigenvec, eigenvec_ad, options)
	return outhopf, hist, flag
end

newtonHopf(F, J, br::ContResult, ind_hopf::Int64, options::NewtonPar) = newtonHopf(F, J, (x, p)->transpose(J(x, p)), br, ind_hopf, options)

"""
codim 2 continuation of Hopf points. This function turns an initial guess for a Hopf point into a curve of Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `(x, p1, p2)-> F(x, p1, p2)` where `p` is the parameter associated to the hopf point
- `J = (x, p1, p2)-> d_xF(x, p1, p2)` associated jacobian
- `hopfpointguess` initial guess (x_0, p1_0) for the Hopf point. It should be a `Vector` or a `BorderedVector`
- `p2` parameter p2 for which hopfpointguess is a good guess
- `eigenvec` guess for the iω eigenvector at p1_0
- `eigenvec_ad` guess for the -iω eigenvector at p1_0
- `options::NewtonPar`
"""
function continuationHopf(F, J, Jt, hopfpointguess::AbstractVector, p2_0, eigenvec, eigenvec_ad, options_cont::ContinuationPar ; kwargs...)
	@warn "Bad way it creates a struct for every p2"
	# Jacobian for the hopf problem
	Jac_hopf_MA(u0::Vector, p2, pb) = (return (u0, pb))

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

	opt_hopf_cont = @set options_cont.newtonOptions.linsolve = HopfLinearSolveMinAug()

	# solve the hopf equations
	return continuation((x, p2) -> hopfPb(x, p2),
						(x, p2) -> Jac_hopf_MA(x, p2, hopfvariable(p2)),
						hopfpointguess, p2_0,
						opt_hopf_cont,
						plot = true,
						printsolution = u -> u[end-1],
						plotsolution = (x;kwargs...) -> (xlabel!("p2", subplot=1); ylabel!("p1", subplot=1)  ) ; kwargs...)
end

continuationHopf(F, J, hopfpointguess::AbstractVector, p2_0, eigenvec, eigenvec_ad, options_cont::ContinuationPar ; kwargs...) = continuationHopf(F, J, (x, p1, p2)->transpose(J(x, p1, p2)), hopfpointguess, p2_0, eigenvec, eigenvec_ad, options_cont ; kwargs...)

"""
Simplified call for continuation of Hopf point. More precisely, the call is as follows `continuationHopf(F, J, Jt, br::ContResult, index::Int64, options)` where the parameters are as for `continuationHopf` except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine.

!!! warning "Eigenvectors`"
    This simplified call has been written when the eigenvectors are organised in a 2d Array `evec` where `evec[:,2]` is the second eigenvector in the list.
"""
function continuationHopf(F, J, Jt, br::ContResult, ind_hopf::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)
	hopfpointguess = HopfPoint(br, ind_hopf)
	bifpt = br.bifpoint[ind_hopf]
	eigenvec = getEigenVector(options_cont.newtonOptions.eigsolve ,br.eig[bifpt[2]][2] ,bifpt[end])
	eigenvec_ad = conj.(eigenvec)
	return continuationHopf(F, J, Jt, hopfpointguess, p2_0, eigenvec, eigenvec_ad, options_cont ; kwargs...)
end

continuationHopf(F, J, br::ContResult, ind_hopf::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...) = continuationHopf(F, J, (x, p1, p2) -> transpose(J(x, p1, p2)), br, ind_hopf, p2_0, options_cont::ContinuationPar ; kwargs...)
