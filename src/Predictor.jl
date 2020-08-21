emptypredictor!(::Nothing) = nothing
####################################################################################################
"""
	dt = DotTheta( (x,y) -> dot(x,y) / length(x) )

This parametric type allows to define a new dot product from the one saved in `dt::dot`. More precisely:

	dt(u1, u2, p1::T, p2::T, theta::T) where {T <: Real}

computes, the weigthed dot product ``\\langle (u_1,p_1), (u_2,p_2)\\rangle_\\theta = \\theta \\Re \\langle u_1,u_2\\rangle  +(1-\\theta)p_1p_2`` where ``u_i\\in\\mathbb R^N``. The ``\\Re`` factor is put to ensure a real valued result despite possible complex valued arguments.

	normtheta(u, p::T, theta::T)

Compute, the norm associated to weigthed dot product ``\\langle (u_1,p_1), (u_2,p_2)\\rangle_\\theta``.

!!! info "Info"
    This is used in the pseudo-arclength constraint with the dot product ``\\frac{1}{N} \\langle u_1,u_2\\rangle,\\quad u_i\\in\\mathbb R^N``
"""
struct DotTheta{Tdot}
	dot::Tdot		# defaults to (x,y) -> dot(x,y) / length(x)
end

DotTheta() = DotTheta( (x, y) -> dot(x, y) / length(x))

# Implementation of the dot product associated to DotTheta
function (dt::DotTheta)(u1, u2, p1::T, p2::T, θ::T) where {T <: Real}
	# we restrict the type of the parameters because for complex problems, we still want the parameter to be real
	return real(dt.dot(u1, u2) * θ + p1 * p2 * (one(T) - θ))
end

# Implementation of the norm associated to DotTheta
function (dt::DotTheta)(u, p::T, θ::T) where T
	# we restrict the type of the parameters because for complex problems, we still want the parameter to be real
	return sqrt(dt(u, u, p, p, θ))
end

(dt::DotTheta)(a::BorderedArray{vec, T}, b::BorderedArray{vec, T}, θ::T) where {vec, T} = dt(a.u, b.u, a.p, b.p, θ)

(dt::DotTheta)(a::BorderedArray{vec, T}, θ::T) where {vec, T} = dt(a.u, a.p, θ)

####################################################################################################
# equation of the arc length constraint
function arcLengthEq(dt::DotTheta, u, p, du, dp, θ, ds)
	return dt(u, du, p, dp, θ) - ds
end

####################################################################################################
abstract type AbstractTangentPredictor end
abstract type AbstractSecantPredictor <: AbstractTangentPredictor end

# reset the predictor
emptypredictor!(::AbstractTangentPredictor) = nothing

function getPredictor!(z_pred::M, z_old::M, tau::M, ds, algo::Talgo) where {T, vectype, M <: BorderedArray{vectype, T}, Talgo <: AbstractTangentPredictor}
	# we perform z_pred = z_old + ds * tau
	copyto!(z_pred, z_old) # z_pred <-- z_old
	axpy!(ds, tau, z_pred)
end

# generic corrector based on Bordered formulation
function corrector(it, z_old::M, tau::M, z_pred::M, ds, θ,
			algo::Talgo, linearalgo = MatrixFreeLBS();
			normC = norm, callback = cbDefault, kwargs...) where
			{T, vectype, M <: BorderedArray{vectype, T}, Talgo <: AbstractTangentPredictor}
	return newtonPALC(it, z_old, tau, z_pred, ds, θ; linearbdalgo = linearalgo, normN = normC, callback = callback, kwargs...)
end

####################################################################################################
"""
	Natural predictor / corrector
"""
struct NaturalPred <: AbstractTangentPredictor end

# corrector based on natural formulation
function corrector(it, z_old::M, tau::M, z_pred::M, ds, θ,
			algo::NaturalPred, linearalgo = MatrixFreeLBS();
			normC = norm, callback = cbDefault, kwargs...) where
			{T, vectype, M <: BorderedArray{vectype, T}}
	res = newton(it.F, it.J, z_pred.u, set(it.par,it.param_lens,z_pred.p), it.contParams.newtonOptions;
				 normN = normC, callback = callback, kwargs...)
	return BorderedArray(res[1], z_pred.p), res[2], res[3], res[4]
end

function getTangent!(tau::M, z_new::M, z_old::M, it::PALCIterable, ds, θ, algo::NaturalPred, verbosity) where {T, vectype, M <: BorderedArray{vectype, T}}
	(verbosity > 0) && println("--> predictor = ", algo)
	rmul!(tau.u, 0)
	tau.p = one(tau.p)
end
####################################################################################################
"""
	Secant tangent predictor
"""
struct SecantPred <: AbstractSecantPredictor end

# tangent computation using Secant predictor
# tau is the tangent prediction
function getTangent!(tau::M, z_new::M, z_old::M, it::PALCIterable, ds, θ, algo::SecantPred, verbosity) where {T, vectype, M <: BorderedArray{vectype, T}}
	(verbosity > 0) && println("--> predictor = ", algo)
	# secant predictor: tau = z_new - z_old; tau *= sign(ds) / normtheta(tau)
	copyto!(tau, z_new)
	minus!(tau, z_old)
	if algo isa SecantPred
		α = sign(ds) / it.dottheta(tau, θ)
	else
		α = sign(ds) / abs(tau.p)
	end
	rmul!(tau, α)
end
####################################################################################################
"""
	Bordered Tangent predictor
"""
struct BorderedPred <: AbstractTangentPredictor end

# tangent computation using Bordered system
# tau is the tangent prediction
function getTangent!(tau::M, z_new::M, z_old::M, it::PALCIterable, ds, θ, algo::BorderedPred, verbosity) where {T, vectype, M <: BorderedArray{vectype, T}}
	(verbosity > 0) && println("--> predictor = Bordered")
	# tangent predictor
	ϵ = it.contParams.finDiffEps
	# dFdl = (F(z_new.u, z_new.p + ϵ) - F(z_new.u, z_new.p)) / ϵ
	dFdl = it.F(z_new.u, set(it.par, it.param_lens, z_new.p + ϵ))
	minus!(dFdl, it.F(z_new.u, set(it.par, it.param_lens, z_new.p)))
	rmul!(dFdl, 1/ϵ)

	# tau = getTangent(J(z_new.u, z_new.p), dFdl, tau_old, theta, contparams.newtonOptions.linsolve)
	tau_normed = copy(tau)#copyto!(similar(tau), tau) #copy(tau_old)
	rmul!(tau_normed, θ / length(tau.u), 1 - θ)
	# extract tangent as solution of bordered linear system, using zero(z_new.u)
	tauu, taup, flag, itl = it.linearAlgo( it.J(z_new.u, set(it.par, it.param_lens, z_new.p)), dFdl,
			tau_normed, 0*z_new.u, T(1), θ)

	# the new tangent vector must preserve the direction along the curve
	α = T(1) / it.dottheta(tauu, tau.u, taup, tau.p, θ)

	# tau_new = α * tau
	copyto!(tau.u, tauu)
	tau.p = taup
	rmul!(tau, α)
end
####################################################################################################
"""
	Multiple Tangent predictor
"""
mutable struct MultiplePred{T <: Real, Tvec, Talgo} <: AbstractTangentPredictor
	tangentalgo::Talgo	# tangent algo used
	α::T				# damping factor
	τ::Tvec				# save the current tangent
	nb::Int64			# number of predictors
	indconverged::Int	# index of the largest converged predictor
	imax::Int			# maximum value of imax
	pmimax::Int			# index for lookup in residual history
end
MultiplePred(α::Real,nb::Int,τ,algo::AbstractTangentPredictor) = MultiplePred(algo,α,τ,nb,0,5,1)
MultiplePred(α::Real,nb::Int,τ) = MultiplePred(α,nb,τ,SecantPred())
emptypredictor!(mpd::MultiplePred) = (mpd.indconverged = 0; mpd.pmimax = 1)

# callback for newton
function (mpred::MultiplePred)(x, f, J, res, iteration, itlinear, options; kwargs...)
	resHist = get(kwargs, :resHist, nothing)
	iteration - mpred.pmimax > 0 ? resHist[end] <= mpred.α * resHist[end-mpred.pmimax] : true
end

function getTangent!(tau::M, z_new::M, z_old::M, it::PALCIterable, ds, θ, algo::MultiplePred{T, M, Talgo}, verbosity) where {T, vectype, M <: BorderedArray{vectype, T}, Talgo}
	# compute tangent and store it
	(verbosity > 0) && print("--> predictor = MultiplePred\n--")
	getTangent!(tau, z_new, z_old, it, ds, θ, algo.tangentalgo, verbosity)
	copyto!(algo.τ, tau)
end

function corrector(it, z_old::M, tau::M, z_pred::M, ds, θ,
		algo::MultiplePred, linearalgo = MatrixFreeLBS(); normC = norm,
		callback = cbDefault, kwargs...) where {T, vectype, M <: BorderedArray{vectype, T}, Talgo}
	# we combine the callbacks for the newton iterations
	cb = (x, f, J, res, iteration, itlinear, options; k...) -> callback(x, f, J, res, iteration, itlinear, options; k...) & algo(x, f, J, res, iteration, itlinear, options; k...)
	_range = algo.nb:-1:0
	for ii in _range
		# record the largest converged guess
		algo.indconverged = ii
		zpred = _copy(z_pred)
		axpy!(ii*ds, algo.τ, zpred)
		# we restore the original callback if it reaches the usual case ii=0
		zold, res, flag, itnewton = corrector(it, z_old, tau, zpred, ds, θ,
				algo.tangentalgo, linearalgo; normC = normC, callback = cb, kwargs...)
		if flag || ii == _range[end]
			return zold, res, flag, itnewton
		end
	end
	return zold, res, flag, itnewton
end

function stepSizeControl(ds, θ, contparams::ContinuationPar, converged::Bool, it_newton_number::Int, tau::M, mpd::MultiplePred, verbosity) where {T, vectype, M<:BorderedArray{vectype, T}}
	if converged == false || mpd.indconverged == 0
		dsnew = ds
		if abs(ds) >= (1 + mpd.nb) * contparams.dsmin
			dsnew = ds / (1 + mpd.nb)
		end
		if  abs(ds) < contparams.dsmin * (1 + mpd.nb)
			(verbosity > 0) && printstyled("*"^80*"\nFailure to converge with given tolerances\n"*"*"^80, color=:red)
			# we stop the continuation
			mpd.pmimax = min(mpd.imax, mpd.pmimax+1)
			return ds, θ, true
		end
		# dsnew = sign(ds) * max(abs(ds) / 2, contparams.dsmin);
		(verbosity > 0) && printstyled("Halving continuation step, ds=$(dsnew)\n", color=:red)
	else
		# control to have the same number of Newton iterations
		Nmax = contparams.newtonOptions.maxIter
		factor = (Nmax - it_newton_number) / Nmax
		# dsnew = ds * (1 + contparams.a * factor^2)
		dsfact =  (1 + contparams.a * factor^2)
		dsnew = ds
		if mpd.indconverged == mpd.nb && abs(ds)*dsfact <= contparams.dsmax
			(verbosity > 0) && @show 1 + contparams.a * factor^2
			dsnew = ds * dsfact
		end
	end

	# control step to stay between bounds
	dsnew = clampDs(dsnew, contparams)

	thetanew = contparams.doArcLengthScaling ? arcLengthScaling(θ, contparams, tau, verbosity) : θ

	return dsnew, thetanew, false
end
####################################################################################################
"""
	Polynomial Tangent predictor
"""
mutable struct PolynomialPred{T <: Real, Tvec, Talgo} <: AbstractTangentPredictor
	n::Int64							# order of the polynomial
	k::Int64							# last solutions vector
	A::Matrix{T}						# matrix for the interpolation
	tangentalgo::Talgo					# algo for tangent when polynomial predictor is not possible
	solutions::CircularBuffer{Tvec}		# vector of solutions
	parameters::CircularBuffer{T}		# vector of parameters
	arclengths::CircularBuffer{T}		# vector of arclengths
	coeffsSol::Vector{Tvec}				# coefficients for the polynomials for the solution
	coeffsPar::Vector{T}				# coefficients for the polynomials for the parameter
	update::Bool
end

PolynomialPred(n,k,v0,algo) = (@assert n<k; ;PolynomialPred(n,k,zeros(eltype(v0),k,n+1),algo,
	CircularBuffer{typeof(v0)}(k),CircularBuffer{eltype(v0)}(k),
	CircularBuffer{eltype(v0)}(k),
	Vector{typeof(v0)}(undef, n+1),
	Vector{eltype(v0)}(undef, n+1),true))
PolynomialPred(n,k,v0) = PolynomialPred(n,k,v0, SecantPred())

isready(ppd::PolynomialPred) = length(ppd.solutions) >= ppd.k

function emptypredictor!(ppd::PolynomialPred)
	empty!(ppd.solutions);empty!(ppd.parameters);empty!(ppd.arclengths);
end

function getStats(polypred)
	Sbar = sum(polypred.arclengths) / polypred.n
	σ = sqrt(sum(x->(x-Sbar)^2, polypred.arclengths ) / (polypred.n))
	# return 0,1
	return Sbar, σ
end

function (polypred::PolynomialPred)(ds)
	sbar, σ = getStats(polypred)
	s = polypred.arclengths[end] + ds
	S = [((s-sbar)/σ)^(jj-1) for jj=1:polypred.n+1]
	p = sum(S .* polypred.coeffsPar)
	x = sum(S .* polypred.coeffsSol)
	return x, p
end

function updatePred!(polypred::PolynomialPred)
	Sbar, σ = getStats(polypred)
	Ss = (polypred.arclengths .- Sbar) ./ σ
	# construction of the Vandermond Matrix
	polypred.A[:, 1] .= 1
	for jj in 1:polypred.n
		polypred.A[:, jj+1] .= polypred.A[:, jj] .* Ss
	end
	# invert linear system for least square fitting
	B = (polypred.A' * polypred.A) \ polypred.A'
	mul!(polypred.coeffsSol, B, polypred.solutions)
	mul!(polypred.coeffsPar, B, polypred.parameters)
end

function getTangent!(tau::M, z_new::M, z_old::M, it::PALCIterable, ds, θ, polypred::PolynomialPred, verbosity) where {T, vectype, M <: BorderedArray{vectype, T}, Talgo}
	# compute tangent and store it
	(verbosity > 0) && println("--> predictor = PolynomialPred")

	if polypred.update
		# update the list of solutions
		if length(polypred.arclengths)==0
			push!(polypred.arclengths, ds)
		else
			push!(polypred.arclengths, polypred.arclengths[end]+ds)
		end
		push!(polypred.solutions, z_new.u)
		push!(polypred.parameters, z_new.p)
	end

	if ~isready(polypred) || ~polypred.update
		return getTangent!(tau, z_new, z_old, it, ds, θ, polypred.tangentalgo, verbosity)
	else
		return polypred.update ? updatePred!(polypred) : true
	end
end

function getPredictor!(z_pred::M, z_old::M, tau::M, ds, polypred::PolynomialPred) where {T, vectype, M <: BorderedArray{vectype, T}}
	if ~isready(polypred)
		return getPredictor!(z_pred, z_old, tau, ds, polypred.tangentalgo)
	else
		x, p = polypred(ds)
		copyto!(z_pred.u, x)
		z_pred.p = p
		return true
	end
end
####################################################################################################
function arcLengthScaling(θ, contparams, tau::M, verbosity) where {M <: BorderedArray}
	# the arclength scaling algorithm is based on Salinger, Andrew G, Nawaf M Bou-Rabee, Elizabeth A Burroughs, Roger P Pawlowski, Richard B Lehoucq, Louis Romero, and Edward D Wilkes. “LOCA 1.0 Library of Continuation Algorithms: Theory and Implementation Manual,” March 1, 2002. https://doi.org/10.2172/800778.
	thetanew = θ
	g = abs(tau.p * θ)
	(verbosity > 0) && print("Theta changes from $(θ) to ")
	if (g > contparams.gMax)
		thetanew = contparams.gGoal / tau.p * sqrt( abs(1.0 - g^2) / abs(1.0 - tau.p^2) )
		if (thetanew < contparams.thetaMin)
		  thetanew = contparams.thetaMin;
		end
	end
	(verbosity > 0) && print("$(thetanew)\n")
	return thetanew
end
####################################################################################################
function clampDs(dsnew, contparams::ContinuationPar)
	if abs(dsnew) < contparams.dsmin
		dsnew = sign(dsnew) * contparams.dsmin
	end

	if abs(dsnew) > contparams.dsmax
		dsnew = sign(dsnew) * contparams.dsmax
	end
	return dsnew
end

function stepSizeControl(ds, θ, contparams::ContinuationPar, converged::Bool, it_newton_number::Int, tau::M, algo::AbstractTangentPredictor, verbosity) where {T, vectype, M<:BorderedArray{vectype, T}}
	if converged == false
		if  abs(ds) <= contparams.dsmin
			(verbosity > 0) && printstyled("*"^80*"\nFailure to converge with given tolerances\n"*"*"^80, color=:red)
			# we stop the continuation
			return ds, θ, true
		end
		dsnew = sign(ds) * max(abs(ds) / 2, contparams.dsmin);
		(verbosity > 0) && printstyled("Halving continuation step, ds=$(dsnew)\n", color=:red)
	else
		# control to have the same number of Newton iterations
		Nmax = contparams.newtonOptions.maxIter
		factor = (Nmax - it_newton_number) / Nmax
		dsnew = ds * (1 + contparams.a * factor^2)
		(verbosity > 0) && @show 1 + contparams.a * factor^2
	end

	# control step to stay between bounds
	dsnew = clampDs(dsnew, contparams)

	thetanew = contparams.doArcLengthScaling ? arcLengthScaling(θ, contparams, tau, verbosity) : θ

	return dsnew, thetanew, false
end
####################################################################################################
"""
This is the classical Newton-Krylov solver used to solve `F(x, p) = 0` together
with the scalar condition `n(x, p) ≡ θ ⋅ <x - x0, τx> + (1-θ) ⋅ (p - p0) * τp - n0 = 0`. This makes a problem of dimension N + 1.

Here, we specify the p as a subfield of `par` with the `paramLens::Lens`

# Arguments
- `(x, par) -> F(x, par)` where `par` is a set of parameters like `(a=1.0, b=1)`
- `(x, par) -> Jh(x, par)` the jacobian Jh = ∂xF
"""
function newtonPALC(F, Jh, par, paramlens::Lens,
					z0::BorderedArray{vectype, T},
					τ0::BorderedArray{vectype, T},
					z_pred::BorderedArray{vectype, T},
					ds::T, θ::T,
					contparams::ContinuationPar{T, S, E},
					dottheta::DotTheta;
					linearbdalgo = BorderingBLS(),
					normN = norm,
					callback = cbDefault, kwargs...) where {T, S, E, vectype}
	# Extract parameters
	newtonOpts = contparams.newtonOptions
	@unpack tol, maxIter, verbose, alpha, almin, linesearch = newtonOpts
	@unpack finDiffEps = contparams

	N = (x, p) -> arcLengthEq(dottheta, minus(x, z0.u), p - z0.p, τ0.u, τ0.p, θ, ds)
	normAC = (resf, resn) -> max(normN(resf), abs(resn))

	# Initialise iterations
	x = _copy(z_pred.u) # copy(z_pred.u)
	p = z_pred.p
	x_pred = _copy(x) # copy(x)

	# Initialise residuals
	res_f = F(x, set(par, paramlens, p));  res_n = N(x, p)

	dX = _copy(res_f) # copy(res_f)
	dp = T(0)
	up = T(0)
	# dFdp = (F(x, p + finDiffEps) - res_f) / finDiffEps
	dFdp = _copy(F(x, set(par, paramlens,p + finDiffEps)))
	minus!(dFdp, res_f)	# dFdp = dFdp - res_f
	rmul!(dFdp, T(1) / finDiffEps)

	res     = normAC(res_f, res_n)
	resHist = [res]
	it = 0

	# Displaying results
	verbose && displayIteration(it, 1, res)
	step_ok = true

	# invoke callback before algo really starts
	compute = callback(x, res_f, nothing, res, 0, 0, contparams; p = p, resHist = resHist, fromNewton = false, kwargs...)

	# Main loop
	while (res > tol) & (it < maxIter) & step_ok & compute
		# dFdp = (F(x, p + epsi) - F(x, p)) / epsi)
		copyto!(dFdp, F(x, set(par, paramlens, p + finDiffEps)))
			minus!(dFdp, res_f); rmul!(dFdp, T(1) / finDiffEps)

		J = Jh(x, set(par, paramlens, p))
		u, up, flag, liniter = linearbdalgo(J, dFdp, τ0, res_f, res_n, θ)

		if linesearch
			step_ok = false
			while !step_ok & (alpha > almin)
				# x_pred = x - alpha * u
				copyto!(x_pred, x); axpy!(-alpha, u, x_pred)

				p_pred = p - alpha * up
				copyto!(res_f, F(x_pred, set(par, paramlens, p)))

				res_n  = N(x_pred, p_pred)
				res = normAC(res_f, res_n)

				if res < resHist[end]
					if (res < resHist[end] / 2) & (alpha < 1)
						alpha *= 2
					end
					step_ok = true
					copyto!(x, x_pred)
					p  = p_pred
				else
					alpha /= 2
				end
			end
		else
			minus!(x, u) 	# x .= x .- u
			p = p - up

			copyto!(res_f, F(x, set(par, paramlens, p)))

			res_n  = N(x, p); res = normAC(res_f, res_n)
		end

		push!(resHist, res)
		it += 1

		verbose && displayIteration(it, 1, res, liniter)

		# shall we break the loop?
		compute = callback(x, res_f, J, res, it, liniter, contparams; p = p, resHist = resHist, fromNewton = false, kwargs...)
	end
	flag = (resHist[end] < tol) & callback(x, res_f, nothing, res, it, -1, contparams; p = p, resHist = resHist, fromNewton = false, kwargs...)
	return BorderedArray(x, p), resHist, flag, it
end

# conveniency for use in continuation
newtonPALC(it::PALCIterable, z0::M, τ0::M, z_pred::M, ds::T, θ::T; kwargs...) where {T, vectype, M <: BorderedArray{vectype, T}} = newtonPALC(it.F, it.J, it.par, it.param_lens, z0, τ0, z_pred, ds, θ, it.contParams, it.dottheta; kwargs...)
####################################################################################################
