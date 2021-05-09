"""
For an initial guess from the index of a Hopf bifurcation point located in ContResult.bifpoint, returns a point which will be refined using `newtonHopf`.
"""
function HopfPoint(br::AbstractBranchResult, index::Int64)
	@assert br.bifpoint[index].type == :hopf "The provided index does not refer to a Hopf point"
	bifpoint = br.bifpoint[index]								# Hopf point
	eigRes   = br.eig											# eigenvector at the Hopf point
	p = bifpoint.param											# parameter value at the Hopf point
	ω = imag(eigRes[bifpoint.idx].eigenvals[bifpoint.ind_ev])	# frequency at the Hopf point
	return BorderedArray(bifpoint.x, [p, ω] )
end

struct HopfProblemMinimallyAugmented{TF, TJ, TJa, Td2f, Tl <: Lens, vectype, S <: AbstractLinearSolver, Sbd <: AbstractBorderedLinearSolver, Sbda <: AbstractBorderedLinearSolver}
	F::TF 						# Function F(x, p) = 0
	J::TJ 						# Jacobian of F wrt x
	Jᵗ::TJa						# Adjoint of the Jacobian of F
	d2F::Td2f					# Hessian of F
	lens::Tl					# parameter axis for the Hopf point
	a::vectype					# close to null vector of (J - iω I)^*
	b::vectype					# close to null vector of  J - iω I
	zero::vectype				# vector zero, to avoid allocating it
	linsolver::S				# linear solver
	linbdsolver::Sbd			# linear bordered solver
	linbdsolverAdjoint::Sbda	# linear bordered solver for the jacobian adjoint
end

@inline hasHessian(pb::HopfProblemMinimallyAugmented{TF, TJ, TJa, Td2f, Tl, vectype, S, Sbd, Sbda}) where {TF, TJ, TJa, Td2f, Tl, vectype, S, Sbd, Sbda} = Td2f != Nothing

@inline hasAdjoint(pb::HopfProblemMinimallyAugmented{TF, TJ, TJa, Td2f, Tl, vectype, S, Sbd, Sbda}) where {TF, TJ, TJa, Td2f, Tl, vectype, S, Sbd, Sbda} = TJa != Nothing

HopfProblemMinimallyAugmented(F, J, Ja, d2F, lens::Lens, a, b, linsolve::AbstractLinearSolver, linbdsolve = BorderingBLS(linsolve)) = HopfProblemMinimallyAugmented(F, J, Ja, d2F, lens, a, b, 0*a, linsolve, linbdsolve, linbdsolve)

HopfProblemMinimallyAugmented(F, J, Ja, lens::Lens, a, b, linsolve::AbstractLinearSolver,  linbdsolver = BorderingBLS(linsolve)) = HopfProblemMinimallyAugmented(F, J, Ja, nothing, lens, a, b, linsolve)

function (hp::HopfProblemMinimallyAugmented)(x, p::T, ω::T, _par) where {T}
	# These are the equations of the minimally augmented (MA) formulation of the Hopf bifurcation point
	# input:
	# - x guess for the point at which the jacobian has a purely imaginary eigenvalue
	# - p guess for the parameter for which the jacobian has a purely imaginary eigenvalue
	# The jacobian of the MA problem is solved with a bordering method
	a = hp.a
	b = hp.b
	# update parameter
	par = set(_par, hp.lens, p)
	# ┌         ┐┌  ┐ ┌ ┐
	# │ J-iω  a ││v │=│0│
	# │  b    0 ││σ1│ │1│
	# └         ┘└  ┘ └ ┘
	# In the notations of Govaerts 2000, a = w, b = v
	# Thus, b should be a null vector of J - iω
	#       a should be a null vector of J'+ iω
	# we solve (J-iω)v + a σ1 = 0 with <b, v> = n
	n = T(1)
	# note that the shift argument only affect J in this call:
	σ1 = hp.linbdsolver(hp.J(x, par), a, b, T(0), hp.zero, n; shift = Complex{T}(0, -ω))[2]

	# we solve (J+iω)'w + b σ2 = 0 with <a, w> = n
	# we find sigma2 = conj(sigma1)
	# w, σ2, _ = fp.linbdsolver(fp.Jadjoint(x, p) - Complex(0, ω) * I, b, a, 0., zeros(N), n)

	# the constraint is σ = <w, Jv> / n
	# σ = -dot(w, apply(fp.J(x, p) + Complex(0, ω) * I, v)) / n
	# we should have σ = σ1

	return hp.F(x, par), real(σ1), imag(σ1)
end

function (hopfpb::HopfProblemMinimallyAugmented)(x::BorderedArray, params)
	res = hopfpb(x.u, x.p[1], x.p[2], params)
	return BorderedArray(res[1], [res[2], res[3]])
end

# Struct to invert the jacobian of the Hopf MA problem.
struct HopfLinearSolverMinAug <: AbstractLinearSolver; end

"""
The function solve the linear problem associated with a linearization of the minimally augmented formulation of the Hopf bifurcation point. The keyword `debug_` is used to debug the routine by returning several key quantities.
"""
function hopfMALinearSolver(x, p::T, ω::T, pb::HopfProblemMinimallyAugmented, par,
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
	Fhandle = pb.F
	J = pb.J

	d2F = pb.d2F
	a = pb.a
	b = pb.b

	# parameter axis
	lens = pb.lens
	# update parameter
	par0 = set(par, lens, p)

	# we define the following jacobian. It is used at least 3 times below. This avoid doing 3 times the possibly costly building of J(x, p)
	J_at_xp = J(x, par0)

	# we do the following to avoid computing J_at_xp twice in case pb.Jadjoint is not provided
	JAd_at_xp = hasAdjoint(pb) ? pb.Jᵗ(x, par0) : transpose(J_at_xp)

	n = T(1)

	# we solve (J-iω)v + a σ1 = 0 with <b, v> = n
	v, σ1, _, itv = pb.linbdsolver(J_at_xp, a, b, T(0), pb.zero, n; shift = Complex{T}(0, -ω))

	# we solve (J+iω)'w + b σ1 = 0 with <a, w> = n
	w, σ2, _, itw = pb.linbdsolverAdjoint(JAd_at_xp, b, a, T(0), pb.zero, n; shift = Complex{T}(0, ω))

	δ = T(1e-9)
	ϵ1, ϵ2, ϵ3 = T(δ), T(δ), T(δ)
	###################  computation of σx σp  ####################
	################### and inversion of Jhopf ####################
	dpF   = (Fhandle(x, set(par, lens, p + ϵ1))	 - Fhandle(x, set(par, lens, p - ϵ1))) / T(2ϵ1)
	dJvdp = (apply(J(x, set(par, lens, p + ϵ3)), v) - apply(J(x, set(par, lens, p - ϵ3)), v)) / T(2ϵ3)
	σp = -dot(w, dJvdp) / n

	# case of sigma_omega
	# σω = dot(w, Complex{T}(0, 1) * v) / n
	σω = Complex{T}(0, 1) * dot(w, v) / n

	x1, x2, _, (it1, it2) = pb.linsolver(J_at_xp, duu, dpF)

	# the case of ∂_xσ is a bit more involved
	# we first need to compute the value of ∂_xσ written σx
	# σx = zeros(Complex{T}, length(x))
	σx = similar(x, Complex{T})

	if hasHessian(pb) == false
		# We invert the jacobian of the Hopf problem when the Hessian of x -> F(x, p) is not known analytically. We thus rely on finite differences which can be slow for large dimensions
		prod(size(x)) > 1e4 && @warn "You might want to pass the Hessian, finite differences with $(prod(size(x))) unknowns"
		e = zero(x)
		for ii in CartesianIndices(x)
			e[ii] = T(1)
			d2Fve = (apply(J(x + ϵ2 * e, par0), v) - apply(J(x - ϵ2 * e, par0), v)) / T(2ϵ2)
			σx[ii] = -dot(w, d2Fve) / n
			e[ii] = T(0)
		end
		σxx1 = dot(σx, x1)
		σxx2 = dot(σx, x2)
	else
		d2Fv = d2F(x, par0, v, x1)
		# this matches the computations in the case hasHessian(pb) == false
		σxx1 = -conj(dot(w, d2Fv) / n)
		d2Fv = d2F(x, par0, v, x2)
		σxx2 = -conj(dot(w, d2Fv) / n)
	end
	# we need to be carefull here because the dot produce conjugates. Hence the + dot(σx, x2) and + imag(dot(σx, x1) and not the opposite
	dp, dω = [real(σp - σxx2) real(σω);
			  imag(σp + σxx2) imag(σω) ] \
			  [dup - real(σxx1), duω + imag(σxx1)]

	if debug_
		return x1 - dp * x2, dp, dω, true, it1 + it2 + sum(itv) + sum(itw), (σx, σp, σω, dpF)
	else
		return x1 - dp * x2, dp, dω, true, it1 + it2 + sum(itv) + sum(itw)
	end
end

function (hopfl::HopfLinearSolverMinAug)(Jhopf, du::BorderedArray{vectype, T}; debug_ = false)  where {vectype, T}
	out = hopfMALinearSolver((Jhopf.x).u,
				(Jhopf.x).p[1],
				(Jhopf.x).p[2],
				Jhopf.hopfpb,
				Jhopf.params,
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
	newtonHopf(F, J, hopfpointguess::BorderedArray{vectypeR, T}, par, lens::Lens, eigenvec, eigenvec_ad, options::NewtonPar; Jᵗ = nothing, d2F = nothing, normN = norm) where {vectypeR, T}

This function turns an initial guess for a Hopf point into a solution to the Hopf problem based on a Minimally Augmented formulation. The arguments are as follows
- `F   = (x, p) -> F(x, p)` where `p` is a set of parameters.
- `dF  = (x, p) -> d_xF(x, p)` associated jacobian
- `hopfpointguess` initial guess (x_0, p_0) for the Hopf point. It should a `BorderedArray` as returned by the function `HopfPoint`.
- `par` parameters used for the vector field
- `lens` parameter axis used to locate the Hopf point.
- `eigenvec` guess for the  iω eigenvector
- `eigenvec_ad` guess for the -iω eigenvector
- `options::NewtonPar` options for the Newton-Krylov algorithm, see [`NewtonPar`](@ref).

# Optional arguments:
- `Jᵗ = (x, p) -> transpose(d_xF(x, p))` jacobian adjoint, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jᵗ` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jᵗ = (x, p) -> transpose(dF(x, p))`
- `d2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2)` a bilinear operator representing the hessian of `F`. It has to provide an expression for `d2F(x,p)[v1,v2]`.
- `normN = norm`
- `bdlinsolver` bordered linear solver for the constraint equation
- `kwargs` keywords arguments to be passed to the regular Newton-Krylov solver

# Simplified call:
Simplified call to refine an initial guess for a Hopf point. More precisely, the call is as follows

	newtonHopf(F, J, br::AbstractBranchResult, ind_hopf::Int64, lens::Lens; Jᵗ = nothing, d2F = nothing, normN = norm, options = br.contparams.newtonOptions, kwargs...)

where the optional argument `Jᵗ` is the jacobian transpose and the Hessian is `d2F`. The parameters / options are as usual except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of bifurcation point in `br` you want to refine. You can pass newton parameters different from the ones stored in `br` by using the argument `options`.

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian will be computed twice!

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6
"""
function newtonHopf(F, J,
			hopfpointguess::BorderedArray{vectypeR, T},
			par, lens::Lens,
			eigenvec, eigenvec_ad,
			options::NewtonPar;
			Jᵗ = nothing,
			d2F = nothing,
			normN = norm,
			bdlinsolver::AbstractBorderedLinearSolver = BorderingBLS(options.linsolver),
			kwargs...) where {vectypeR, T}
	hopfproblem = HopfProblemMinimallyAugmented(
		F, J, Jᵗ, d2F, lens,
		_copy(eigenvec_ad),	# this is pb.a ≈ null space of (J - iω I)^*
		_copy(eigenvec), 	# this is pb.b ≈ null space of  J - iω I
		options.linsolver, @set bdlinsolver.solver = options.linsolver)

	# Jacobian for the Hopf problem
	Jac_hopf_MA = (x, param) -> (x = x, params = param, hopfpb = hopfproblem)

	# options for the Newton Solver
	opt_hopf = @set options.linsolver = HopfLinearSolverMinAug()

	# solve the hopf equations
	return newton(hopfproblem, Jac_hopf_MA, hopfpointguess, par, opt_hopf, normN = normN, kwargs...)..., hopfproblem
end

function newtonHopf(F, J,
			br::AbstractBranchResult, ind_hopf::Int64;
			Jᵗ = nothing,
			d2F = nothing,
			normN = norm,
			options = br.contparams.newtonOptions,
			verbose = true,
			nev = br.contparams.nev,
			startWithEigen = false,
			kwargs...)
	hopfpointguess = HopfPoint(br, ind_hopf)
	ω = hopfpointguess.p[2]
	bifpt = br.bifpoint[ind_hopf]
	options.verbose && println("--> Newton Hopf, the eigenvalue considered here is ", br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
	@assert bifpt.idx == bifpt.step + 1 "Error, the bifurcation index does not refer to the correct step"
	ζ = geteigenvector(options.eigsolver, br.eig[bifpt.idx].eigenvec, bifpt.ind_ev)
	ζ ./= norm(ζ)
	ζad = LinearAlgebra.conj.(ζ)

	if startWithEigen
		# computation of adjoint eigenvalue. Recall that b should be a null vector of J-iω
		λ = Complex(0, ω)
		p = bifpt.param
		parbif = setParam(br, p)

		# jacobian at bifurcation point
		L = J(bifpt.x, parbif)
		_Jt = isnothing(Jᵗ) ? adjoint(L) : Jᵗ(bifpt.x, parbif)
		ζstar, λstar = getAdjointBasis(_Jt, conj(λ), options.eigsolver; nev = nev, verbose = false)
		ζad .= ζstar ./ dot(ζstar, ζ)
	end

	# solve the hopf equations
	return newtonHopf(F, J, hopfpointguess, br.params, br.lens, ζ, ζad, options; Jᵗ = Jᵗ, d2F = d2F, normN = normN, kwargs...)
end

"""
$(SIGNATURES)

codim 2 continuation of Hopf points. This function turns an initial guess for a Hopf point into a curve of Hopf points based on a Minimally Augmented formulation. The arguments are as follows
- `F = (x, p) ->	F(x, p)` where `p` is a set of parameters
- `J = (x, p) -> d_xF(x, p)` associated jacobian
- `hopfpointguess` initial guess (x_0, p1_0) for the Hopf point. It should be a `Vector` or a `BorderedArray`
- `par` set of parameters
- `lens1` parameter axis for parameter 1
- `lens2` parameter axis for parameter 2
- `eigenvec` guess for the iω eigenvector at p1_0
- `eigenvec_ad` guess for the -iω eigenvector at p1_0
- `options_cont` keywords arguments to be passed to the regular [`continuation`](@ref)

# Optional arguments:

- `Jᵗ = (x, p) -> adjoint(d_xF(x, p))` associated jacobian adjoint
- `d2F = (x, p, v1, v2) -> d2F(x, p, v1, v2)` this is the hessian of `F` computed at `(x, p)` and evaluated at `(v1, v2)`.
- `bdlinsolver` bordered linear solver for the constraint equation
- `updateMinAugEveryStep` update vectors `a,b` in Minimally Formulation every `updateMinAugEveryStep` steps
- `kwargs` keywords arguments to be passed to the regular [`continuation`](@ref)

# Simplified call:
The call is as follows

	continuationHopf(F, J, br::AbstractBranchResult, ind_hopf::Int64, lens2::Lens, options_cont::ContinuationPar ;  Jᵗ = nothing, d2F = nothing, startWithEigen = false, kwargs...)

where the parameters are as above except that you have to pass the branch `br` from the result of a call to `continuation` with detection of bifurcations enabled and `index` is the index of Hopf point in `br` you want to refine.

!!! warning "Hessian"
    The hessian of `F`, when `d2F` is not passed, is computed with Finite differences. This can be slow for many variables, e.g. ~1e6

!!! tip "ODE problems"
    For ODE problems, it is more efficient to pass the Bordered Linear Solver using the option `bdlinsolver = MatrixBLS()`

!!! tip "Jacobian tranpose"
    The adjoint of the jacobian `J` is computed internally when `Jᵗ = nothing` by using `transpose(J)` which works fine when `J` is an `AbstractArray`. In this case, do not pass the jacobian adjoint like `Jᵗ = (x, p) -> transpose(d_xF(x, p))` otherwise the jacobian would be computed twice!
"""
function continuationHopf(F, J,
				hopfpointguess::BorderedArray{vectype, Tb}, par,
				lens1::Lens, lens2::Lens,
				eigenvec, eigenvec_ad,
				options_cont::ContinuationPar ;
				updateMinAugEveryStep = 0,
				Jᵗ = nothing,
				d2F = nothing,
				d3F = nothing,
				bdlinsolver::AbstractBorderedLinearSolver = BorderingBLS(options_cont.newtonOptions.linsolver),
				kwargs...) where {Tb,vectype}
	@assert lens1 != lens2

	# options for the Newton Solver inheritated from the ones the user provided
	options_newton = options_cont.newtonOptions

	hopfPb = HopfProblemMinimallyAugmented(
		F, J, Jᵗ, d2F,
		lens1,
		_copy(eigenvec_ad),	# this is pb.a ≈ null space of (J - iω I)^*
		_copy(eigenvec), 	# this is pb.b ≈ null space of  J - iω I
		options_newton.linsolver, @set bdlinsolver.solver = options_newton.linsolver)

	# Jacobian for the Hopf problem
	Jac_hopf_MA = (x, param) -> (x = x, params = param, hopfpb = hopfPb)

	opt_hopf_cont = @set options_cont.newtonOptions.linsolver = HopfLinearSolverMinAug()

	# this functions allows to tackle the case where the two parameters have the same name
	lenses = getLensParam(lens1, lens2)

	function updateMinAugHopf(z, tau, step, contResult; k...)
		~modCounter(step, updateMinAugEveryStep) && return true
		x = z.u.u		# fold point
		p1 = z.u.p[1]	# first parameter
		ω  = z.u.p[2]	# Hopf frequency
		p2 = z.p		# second parameter
		newpar = set(par, lens1, p1)
		newpar = set(newpar, lens2, p2)

		a = hopfPb.a
		b = hopfPb.b

		# expression of the jacobian
		J_at_xp = hopfPb.J(x, newpar)

		# compute new b
		T = typeof(p1)
		n = T(1)
		newb = hopfPb.linbdsolver(J_at_xp, a, b, T(0), hopfPb.zero, n; shift = Complex(0, -ω))[1]

		# compute new a
		JAd_at_xp = hasAdjoint(hopfPb) ? hopfPb.Jᵗ(x, newpar) : transpose(J_at_xp)
		newa = hopfPb.linbdsolver(JAd_at_xp, b, a, T(0), hopfPb.zero, n; shift = Complex(0, ω))[1]

		hopfPb.a .= newa ./ norm(newa)
		hopfPb.b .= newb ./ dot(newb, hopfPb.a)

		return true
	end

	# it allows to append information specific to the codim 2 continuation to the user data
	_printsol = get(kwargs, :printSolution, nothing)
	_printsol2 = isnothing(_printsol) ?
		(u, p; kw...) -> (zip(lenses, (u.p[1], p))..., BT = dot(hopfPb.a, hopfPb.b)) :
		(u, p; kw...) -> (namedprintsol(_printsol(u, p;kw...))..., zip(lenses, (u.p[1], p))..., ω = u.p[2], l1 = 0., BT = dot(hopfPb.a, hopfPb.b))

	# solve the hopf equations
	branch, u, tau = continuation(
		hopfPb, Jac_hopf_MA,
		hopfpointguess, par, lens2,
		(@set opt_hopf_cont.newtonOptions.eigsolver = HopfEig(opt_hopf_cont.newtonOptions.eigsolver));
		kwargs...,
		printSolution = _printsol2,
		finaliseSolution = updateMinAugHopf,

	return setproperties(branch; type = :HopfCodim2, functional = hopfPb), u, tau
end

function continuationHopf(F, J,
						br::AbstractBranchResult, ind_hopf::Int64,
						lens2::Lens, options_cont::ContinuationPar ;
						startWithEigen = false,
						Jᵗ = nothing,
						d2F = nothing,
						d3F = nothing,
						kwargs...)
	hopfpointguess = HopfPoint(br, ind_hopf)
	ω = hopfpointguess.p[2]
	bifpt = br.bifpoint[ind_hopf]
	ζ = geteigenvector(options_cont.newtonOptions.eigsolver ,br.eig[bifpt.idx].eigenvec, bifpt.ind_ev)
	ζ ./= norm(ζ)
	ζad = conj.(ζ)

	if startWithEigen
		# computation of adjoint eigenvalue
		λ = Complex(0, ω)
		p = bifpt.param
		parbif = setParam(br, p)

		# jacobian at bifurcation point
		L = J(bifpt.x, parbif)
		_Jt = isnothing(Jᵗ) ? adjoint(L) : Jᵗ(bifpt.x, parbif)
		ζstar, λstar = getAdjointBasis(_Jt, conj(λ), br.contparams.newtonOptions.eigsolver; nev = br.contparams.nev, verbose = true)
		ζad .= ζstar ./ dot(ζstar, ζ)
	end

	return continuationHopf(F, J, hopfpointguess, br.params, br.lens, lens2, ζ, ζad, options_cont ; Jᵗ = Jᵗ, d2F = d2F, d3F = d3F, kwargs...)
end

struct HopfEig{S} <: AbstractEigenSolver
	eigsolver::S
end

function (eig::HopfEig)(Jma, nev; kwargs...)
	n = min(nev, length(Jma.x.u))
	J = Jma.hopfpb.J(Jma.x.u, set(Jma.params, Jma.hopfpb.lens, Jma.x.p[1]))
	eigenelts = eig.eigsolver(J, n; kwargs...)
	return eigenelts
end
