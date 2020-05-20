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

DotTheta() = DotTheta( (x,y) -> dot(x,y) / length(x))

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

"""
	Natural predictor / corrector
"""
struct NaturalPred <: AbstractSecantPredictor end

"""
	Secant tangent predictor
"""
struct SecantPred <: AbstractSecantPredictor end

"""
	Bordered Tangent predictor
"""
struct BorderedPred <: AbstractTangentPredictor end

function getPredictor!(z_pred::M, z_old::M, tau::M, ds, algo::Talgo) where {T, vectype, M <: BorderedArray{vectype, T}, Talgo <: AbstractTangentPredictor}
	# we perform z_pred = z_old + ds * tau
	z_pred = copyto!(z_pred, z_old)
	axpy!(ds, tau, z_pred)
end

# generic corrector based on Bordered formulation
function corrector(it, z_old::M, tau::M, z_pred::M,
			ds, θ,
			algo::Talgo, linearalgo = MatrixFreeLBS();
			normC = norm,
			callback = (x, f, J, res, iteration, itlinear, options; kwargs...) -> true, kwargs...) where {T, vectype, M <: BorderedArray{vectype, T}, Talgo <: AbstractTangentPredictor}
	return newtonPALC(it, z_old, tau, z_pred, ds, θ, linearalgo, normC, callback; kwargs...)
end

# corrector based on natural formulation
function corrector(it, z_old::M, tau::M, z_pred::M,
			ds, θ, contparams, dottheta::DotTheta,
			algo::NaturalPred, linearalgo = MatrixFreeLBS();
			normC = norm,
			callback = (x, f, J, res, iteration, itlinear, options; kwargs...) -> true, kwargs...) where {T, vectype, M <: BorderedArray{vectype, T}}
	res = newton(u -> Fhandle(u, z_pred.p),
				 u -> Jhandle(u, z_pred.p),
				 z_pred.u, contparams.newtonOptions;
				 normN = normC, callback = callback, kwargs...)
	return BorderedArray(res[1], z_pred.p), res[2], res[3], res[4]
end

# tangent computation using Natural / Secant predictor
# tau is the tangent prediction
function getTangent!(tau::M, z_new::M, z_old::M, it::PALCIterable, ds, θ, algo::Talgo, verbosity) where {T, vectype, M <: BorderedArray{vectype, T}, Talgo <: AbstractSecantPredictor}
	(verbosity > 0) && println("--> predictor = ", algo)
	# secant predictor: tau = z_new - z_old; tau *= sign(ds) / normtheta(tau)
	tau = copyto!(tau, z_new)
	minus!(tau, z_old)
	if algo isa SecantPred
		α = sign(ds) / it.dottheta(tau, θ)
	else
		α = sign(ds) / abs(tau.p)
	end
	rmul!(tau, α)
end

# tangent computation using Bordered system
# tau is the tangent prediction
function getTangent!(tau::M, z_new::M, z_old::M, it::PALCIterable, ds, θ, algo::BorderedPred, verbosity) where {T, vectype, M <: BorderedArray{vectype, T}}
	(verbosity > 0) && println("--> predictor = Bordered")
	# tangent predictor
	ϵ = it.contParams.finDiffEps
	# dFdl = (F(z_old.u, z_old.p + ϵ) - F(z_old.u, z_old.p)) / ϵ
	dFdl = it.F(z_old.u, set(it.par, it.param_lens, z_old.p + ϵ))
	minus!(dFdl, it.F(z_old.u, set(it.par, it.param_lens, z_old.p)))
	rmul!(dFdl, 1/ϵ)

	# tau = getTangent(J(z_old.u, z_old.p), dFdl, tau_old, theta, contparams.newtonOptions.linsolve)
	tau_normed = copy(tau)#copyto!(similar(tau), tau) #copy(tau_old)
	rmul!(tau_normed, θ / length(tau.u), 1 - θ)
	# extract tangent as solution of bordered linear system, using zero(z_old.u)
	tauu, taup, flag, itl = it.linearAlgo( it.J(z_old.u, set(it.par, it.param_lens, z_old.p)), dFdl,
			tau_normed, rmul!(similar(z_old.u), false), T(1), θ)

	# the new tangent vector must preserve the direction along the curve
	α = T(1) / it.dottheta(tauu, tau.u, taup, tau.p, θ)

	# tau_new = α * tau
	copyto!(tau.u, tauu)
	tau.p = taup
	rmul!(tau, α)
end
################################################################################################
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
################################################################################################
function stepSizeControl(ds, θ, contparams, converged::Bool, it_newton_number::Int64, tau::M, verbosity) where {T, vectype, M<:BorderedArray{vectype, T}}
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
	if abs(dsnew) < contparams.dsmin
		dsnew = sign(dsnew) * contparams.dsmin
	end

	if abs(dsnew) > contparams.dsmax
		dsnew = sign(dsnew) * contparams.dsmax
	end

	if contparams.doArcLengthScaling
		thetanew = arcLengthScaling(θ, contparams, tau, verbosity)
	else
		thetanew = θ
	end
	@assert abs(dsnew) >= contparams.dsmin "Error with ds value"
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
					contparams::ContinuationPar{T},
					dottheta::DotTheta;
					linearbdalgo = BorderingBLS(),
					normN = norm,
					callback = (x, f, J, res, iteration, itlinear, optionsN; kwargs...) ->  true, kwargs...) where {T, vectype}
	# Extract parameters
	newtonOpts = contparams.newtonOptions
	@unpack tol, maxIter, verbose, alpha, almin, linesearch, saveIterations = newtonOpts
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
	dFdp = rmul!( minus!( F(x,set(par,paramlens,p+finDiffEps)), res_f ), 1/finDiffEps)

	res     = normAC(res_f, res_n)
	resHist = [res]
	it = 0

	# Displaying results
	verbose && displayIteration(it, 1, res)
	step_ok = true

	# invoke callback before algo really starts
	compute = callback(x, res_f, nothing, res, 0, 0, contparams; kwargs...)

	# Main loop
	while (res > tol) & (it < maxIter) & step_ok & compute
		dFdp = rmul!( minus!( F(x,set(par,paramlens,p+finDiffEps)), res_f ), 1/finDiffEps)

		J = Jh(x, set(par, paramlens, p))
		u, up, flag, liniter = linearbdalgo(J, dFdp, τ0, res_f, res_n, θ)

		if linesearch & saveIterations
			step_ok = false
			while !step_ok & (alpha > almin)

				x_pred = axpy!(-alpha, u, copyto!(x_pred, x))
				p_pred = p - alpha * up
				res_f = copyto!(res_f, F(x_pred, p_pred))

				res_n  = N(x_pred, p_pred)
				res = normAC(res_f, res_n)

				if res < resHist[end]
					if (res < resHist[end] / 2) & (alpha < 1)
						alpha *= 2
					end
					step_ok = true
					x  = copyto!(x, x_pred)
					p  = p_pred
				else
					alpha /= 2
				end
			end
		else
			x = minus!(x,u) 	# x .= x .- u
			p = p - up

			res_f = copyto!(res_f, F(x, set(par, paramlens, p)))

			res_n  = N(x, p)
			res = normAC(res_f, res_n)
		end

		saveIterations && push!(resHist, res)
		it += 1

		verbose && displayIteration(it, 1, res, liniter)
		if callback(x, res_f, J, res, it, liniter, contparams; kwargs...) == false
			break
		end

	end
	flag = (res < tol) & callback(x, res_f, nothing, res, it, nothing, contparams; kwargs...)
	return BorderedArray(x, p), resHist, flag, it
end

# conveniency for use in continuation
newtonPALC(it::PALCIterable, z0::M, τ0::M, z_pred::M, ds::T, θ::T, linearbdalgo, norm, callback; kwargs...) where {T, vectype, M <: BorderedArray{vectype, T}} = newtonPALC(it.F, it.J, it.par, it.param_lens, z0, τ0, z_pred, ds, θ, it.contParams, it.dottheta; linearbdalgo=linearbdalgo, normN=norm, callback=callback, kwargs...)
####################################################################################################
