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
function (dt::DotTheta)(u1, u2, p1::T, p2::T, theta::T) where {T <: Real}
	# we restrict the type of the parameters because for complex problems, we still want the parameter to be real
	return real(dt.dot(u1, u2) * theta + p1 * p2 * (one(T) - theta))
end

# Implementation of the norm associated to DotTheta
function (dt::DotTheta)(u, p::T, theta::T) where T
	# we restrict the type of the parameters because for complex problems, we still want the parameter to be real
	return sqrt(dt(u, u, p, p, theta))
end

(dt::DotTheta)(a::BorderedArray{vec, T}, b::BorderedArray{vec, T}, theta::T) where {vec, T} = dt(a.u, b.u, a.p, b.p, theta)

(dt::DotTheta)(a::BorderedArray{vec, T}, theta::T) where {vec, T} = dt(a.u, a.p, theta)

####################################################################################################
# equation of the arc length constraint
function arcLengthEq(dt::DotTheta, u, p, du, dp, theta, ds)
	return dt(u, du, p, dp, theta) - ds
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
	copyto!(z_pred, z_old)
	axpy!(ds, tau, z_pred)
end

# generic corrector based on Bordered formulation
function corrector(Fhandle, Jhandle, z_old::M, tau_old::M, z_pred::M,
			ds, theta, contparams, dottheta::DotTheta,
			algo::Talgo, linearalgo = MatrixFreeLBS();
			normC = norm,
			callback = (x, f, J, res, iteration; kwargs...) -> true, kwargs...) where {T, vectype, M<:BorderedArray{vectype, T}, Talgo <: AbstractTangentPredictor}
	return newtonPALC(Fhandle, Jhandle,
			z_old, tau_old, z_pred,
			ds, theta,
			contparams, dottheta; linearbdalgo = linearalgo, normN = normC, callback = callback, kwargs...)
end

# corrector based on natural formulation
function corrector(Fhandle, Jhandle, z_old::M, tau_old::M, z_pred::M,
			ds, theta, contparams, dottheta::DotTheta,
			algo::NaturalPred, linearalgo = MatrixFreeLBS();
			normC = norm,
			callback = (x, f, J, res, iteration; kwargs...) -> true, kwargs...) where {T, vectype, M <: BorderedArray{vectype, T}}
	res = newton(u -> Fhandle(u, z_pred.p),
				 u -> Jhandle(u, z_pred.p),
				 z_pred.u, contparams.newtonOptions;
				 normN = normC, callback = callback, kwargs...)
	return BorderedArray(res[1], z_pred.p), res[2], res[3], res[4]
end

# tangent computation using Natural / Secant predictor
# tau_new is the tangent prediction
function getTangent!(tau_new::M, z_new::M, z_old::M, tau_old::M, F, J,
	ds, theta, contparams, normtheta::DotTheta,
	algo::Talgo, verbosity, linearalgo) where {T, vectype, M <: BorderedArray{vectype, T}, Talgo <: AbstractSecantPredictor}
	(verbosity > 0) && println("--> predictor = ", algo)
	# secant predictor: tau = z_new - z_old; tau *= sign(ds) / normtheta(tau)
	copyto!(tau_new, z_new)
	minus!(tau_new, z_old)
	if algo isa SecantPred
		α = sign(ds) / normtheta(tau_new, theta)
	else
		α = sign(ds) / abs(tau_new.p)
	end
	rmul!(tau_new, α)
end

# tangent computation using Bordered system
# tau_new is the tangent prediction
function getTangent!(tau_new::M, z_new::M, z_old::M, tau_old::M, F, J,
	ds, theta, contparams, dottheta::DotTheta,
	algo::BorderedPred, verbosity, linearbdalgo) where {T, vectype, M <: BorderedArray{vectype, T}}
	(verbosity > 0) && println("--> predictor = Bordered")
	# tangent predictor
	ϵ = contparams.finDiffEps
	# dFdl = (F(z_old.u, z_old.p + ϵ) - F(z_old.u, z_old.p)) / ϵ
	dFdl = F(z_old.u, z_old.p + ϵ)
	minus!(dFdl, F(z_old.u, z_old.p))
	rmul!(dFdl, 1/ϵ)

	# tau = getTangent(J(z_old.u, z_old.p), dFdl, tau_old, theta, contparams.newtonOptions.linsolve)
	tau_normed = copyto!(similar(tau_old), tau_old) #copy(tau_old)
	rmul!(tau_normed, theta / length(tau_old.u), 1 - theta)
	# extract tangent as solution of bordered linear system, using zero(z_old.u)
	tauu, taup, flag, it = linearbdalgo( J(z_old.u, z_old.p), dFdl,
			tau_normed, rmul!(similar(z_old.u), false), T(1), theta)

	# the new tangent vector must preserve the direction along the curve
	α = T(1) / dottheta(tauu, tau_old.u, taup, tau_old.p, theta)

	# tau_new = α * tau
	copyto!(tau_new.u, tauu)
	tau_new.p = taup
	rmul!(tau_new, α)
end
################################################################################################
function arcLengthScaling(theta, contparams, tau::M, verbosity) where {M <: BorderedArray}
	# the arclength scaling algorithm is based on Salinger, Andrew G, Nawaf M Bou-Rabee, Elizabeth A Burroughs, Roger P Pawlowski, Richard B Lehoucq, Louis Romero, and Edward D Wilkes. “LOCA 1.0 Library of Continuation Algorithms: Theory and Implementation Manual,” March 1, 2002. https://doi.org/10.2172/800778.

	thetanew = theta
	g = abs(tau.p * theta)
	(verbosity > 0) && print("Theta changes from $(theta) to ")
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
function stepSizeControl(ds, theta, contparams, converged::Bool, it_newton_number::Int64, tau::M, verbosity) where {T, vectype, M<:BorderedArray{vectype, T}}
	if converged == false
		if  abs(ds) <= contparams.dsmin
			(verbosity > 0) && printstyled("*"^80*"\nFailure to converge with given tolerances\n"*"*"^80, color=:red)
			# we stop the continuation
			return ds, theta, true
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
		thetanew = arcLengthScaling(theta, contparams, tau, verbosity)
	else
		thetanew = theta
	end
	@assert abs(dsnew) >= contparams.dsmin "Error with ds value"
	return dsnew, thetanew, false
end
####################################################################################################
"""
This is the classical matrix-free Newton Solver used to solve `F(x, p) = 0` together
with the scalar condition `n(x, p) = (x - x0) * xp + (p - p0) * lp - n0`
"""
function newtonPALC(F, Jh,
						z0::BorderedArray{vectype, T},
						tau0::BorderedArray{vectype, T},
						z_pred::BorderedArray{vectype, T},
						ds::T, theta::T,
						contparams::ContinuationPar{T},
						dottheta::DotTheta;
						linearbdalgo = BorderingBLS(),
						normN = norm,
						callback = (x, f, J, res, iteration, optionsN; kwargs...) ->  true, kwargs...) where {T, vectype}
	# Extract parameters
	newtonOpts = contparams.newtonOptions
	@unpack tol, maxIter, verbose, alpha, almin, linesearch = newtonOpts
	@unpack finDiffEps = contparams

	N = (x, p) -> arcLengthEq(dottheta, minus(x, z0.u), p - z0.p, tau0.u, tau0.p, theta, ds)
	normAC = (resf, resn) -> max(normN(resf), abs(resn))

	# Initialise iterations
	x = copyto!(similar(z_pred.u), z_pred.u) # copy(z_pred.u)
	p = z_pred.p
	x_pred = copyto!(similar(x), x) # copy(x)

	# Initialise residuals
	res_f = F(x, p);  res_n = N(x, p)

	dX   = copyto!(similar(res_f), res_f) # copy(res_f)
	dp   = T(0)
	# dFdp = (F(x, p + finDiffEps) - res_f) / finDiffEps
	dFdp = copyto!(similar(res_f), F(x, p + finDiffEps)) # copy(F(x, p + finDiffEps))
	minus!(dFdp, res_f)	# dFdp = dFdp - res_f
	rmul!(dFdp, T(1) / finDiffEps)

	res     = normAC(res_f, res_n)
	resHist = [res]
	it = 0

	# Displaying results
	verbose && displayIteration(it, 1, res)
	step_ok = true

	# Main loop
	while (res > tol) & (it < maxIter) & step_ok
		# copyto!(dFdp, (F(x, p + epsi) - F(x, p)) / epsi)
		copyto!(dFdp, F(x, p + finDiffEps)); minus!(dFdp, res_f); rmul!(dFdp, T(1) / finDiffEps)

		J = Jh(x, p)
		u, up, flag, liniter = linearbdalgo(J, dFdp, tau0, res_f, res_n, theta)

		if linesearch
			step_ok = false
			while !step_ok & (alpha > almin)
				# x_pred = x - alpha * u
				copyto!(x_pred, x)
				axpy!(-alpha, u, x_pred)

				p_pred = p - alpha * up
				copyto!(res_f, F(x_pred, p_pred))

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

			copyto!(res_f, F(x, p))

			res_n  = N(x, p)
			res = normAC(res_f, res_n)
		end

		push!(resHist, res)
		it += 1

		callback(x, res_f, J, res, it, contparams; kwargs...) == false && (it = maxIter)
		verbose && displayIteration(it, 1, res, liniter)

	end
	return BorderedArray(x, p), resHist, resHist[end] < tol, it
end
####################################################################################################
