include("BorderedArrays.jl")

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

function getPredictor!(z_pred::M, z_old::M, tau::M, contparams, algo::Talgo) where {T, vectype, M <: BorderedArray{vectype, T}, Talgo <: AbstractTangentPredictor}
	# we perform z_pred = z_old + contparams.ds * tau
	copyto!(z_pred, z_old)
	axpy!(contparams.ds, tau, z_pred)
end

# generic corrector based on Bordered formulation
function corrector(Fhandle, Jhandle, z_old::M, tau_old::M, z_pred::M, contparams, algo, linearalgo = MatrixFreeLBS(); normC::Function = norm) where {T, vectype, M<:BorderedArray{vectype, T}, Talgo <: AbstractTangentPredictor}
	return newtonPseudoArcLength(Fhandle, Jhandle,
			z_old, tau_old, z_pred,
			contparams; linearbdalgo = linearalgo, normN = normC)
end

# corrector based on natural formulation
function corrector(Fhandle, Jhandle, z_old::M, tau_old::M, z_pred::M, contparams, algo::NaturalPred, linearalgo = :bordered; normC::Function = norm) where {T, vectype, M <: BorderedArray{vectype, T}}
	res = newton(u -> Fhandle(u, z_pred.p), u -> Jhandle(u, z_pred.p), z_pred.u, contparams.newtonOptions, normN = normC)
	return BorderedArray(res[1], z_pred.p), res[2], res[3], res[4]
end

# tangent computation using Secant predictor
function getTangent!(tau_new::M, z_new::M, z_old::M, tau_old::M, F, J, contparams, algo::Talgo, verbosity, linearalgo) where {T, vectype, M <: BorderedArray{vectype, T}, Talgo <: AbstractSecantPredictor}
	(verbosity > 0) && println("--> predictor = Secant")
	ds = contparams.ds
	# secant predictor: tau = z_new - z_old; tau *= sign(ds) / normtheta(tau)
	copyto!(tau_new, z_new)
	minus!(tau_new, z_old)
	α = sign(ds) / normtheta(tau_new, contparams.theta)
	rmul!(tau_new, α)
end

# tangent computation using Bordered system
function getTangent!(tau_new::M, z_new::M, z_old::M, tau_old::M, F, J, contparams, algo::BorderedPred, verbosity, linearbdalgo) where {T, vectype, M <: BorderedArray{vectype, T}}
	(verbosity > 0) && println("--> predictor = Bordered")
	# tangent predictor
	epsi = contparams.finDiffEps
	# dFdl = (F(z_old.u, z_old.p + epsi) - F(z_old.u, z_old.p)) / epsi
	dFdl = F(z_old.u, z_old.p + epsi)
	minus!(dFdl, F(z_old.u, z_old.p))
	rmul!(dFdl, 1/epsi)

	# tau = getTangent(J(z_old.u, z_old.p), dFdl, tau_old, contparams.theta, contparams.newtonOptions.linsolve)
	new_tau = copy(tau_old)
	@warn "This can be improved a lot"
	rmul!(new_tau, contparams.theta / length(tau_old.u), 1 - contparams.theta)
	tauu, taup, flag, it = linearbdalgo( J(z_old.u, z_old.p), dFdl,
			new_tau, zero(z_old.u), T(1), contparams.theta)
	tau = BorderedArray(tauu, taup)
	α = T(1) / normtheta(tau, contparams.theta)
	# tau_new = α * tau
	copyto!(tau_new, tau)
	rmul!(tau_new, α)
end
################################################################################################
function arcLengthScaling(contparams, tau::M, verbosity) where {T, vectype, M<:BorderedArray{vectype, T}}
	g = abs(tau.p * contparams.theta)
	(verbosity > 0) && print("Theta changes from $(contparams.theta) to ")
	if (g > contparams.gMax)
		contparams.theta = contparams.gGoal / tau.p * sqrt( abs(1.0 - g*g) / abs(1.0 - tau.p^2) )
		if (contparams.theta < contparams.thetaMin)
		  contparams.theta = contparams.thetaMin;
	  end
	end
	(verbosity > 0) && print("$(contparams.theta)\n")
end
################################################################################################
function stepSizeControl(contparams, converged::Bool, it_number::Int64, tau::M, branch, verbosity) where {T, vectype, M<:BorderedArray{vectype, T}}
	if converged == false
		(verbosity > 0) && abs(contparams.ds) <= contparams.dsmin && (printstyled("*"^80*"\nFailure to converge with given tolerances\n"*"*"^80, color=:red);return true)
		contparams.ds = sign(contparams.ds) * max(abs(contparams.ds) / 2, contparams.dsmin);
		(verbosity > 0) && printstyled("Halving continuation step, ds=$(contparams.ds)\n", color=:red)
	else
		if (length(branch)>1)
			# control to have the same number of Newton iterations
			Nmax = contparams.newtonOptions.maxIter
			factor = (Nmax - it_number) / Nmax
			contparams.ds *= 1 + contparams.a * factor^2
			(verbosity > 0) && @show 1 + contparams.a * factor^2
		end

	end

	# control step to stay between bounds
	if abs(contparams.ds) < contparams.dsmin
		contparams.ds = sign(contparams.ds) * contparams.dsmin
	end

	if abs(contparams.ds) > contparams.dsmax
		contparams.ds = sign(contparams.ds) * contparams.dsmax
	end

	contparams.doArcLengthScaling && arcLengthScaling(contparams, tau, verbosity)
	@assert abs(contparams.ds) >= contparams.dsmin
	return false
end
################################################################################################
# equation of the arc length constraint
function arcLengthEq(u, p, du, dp, xi, ds)
	return dottheta(u, du, p, dp, xi) - ds
end
################################################################################################
function dottheta(u1, u2, p1::T, p2::T, theta::T) where T
	return dot(u1, u2) * theta / length(u1) + p1 * p2 * (one(T) - theta)
end
################################################################################################
function normtheta(u, p::T, theta::T) where T
	return sqrt(dottheta(u, u, p, p, theta))
end
################################################################################################
