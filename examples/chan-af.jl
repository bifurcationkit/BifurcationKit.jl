using Revise
using ApproxFun, LinearAlgebra, Parameters

using PseudoArcLengthContinuation, Plots
const Cont = PseudoArcLengthContinuation

####################################################################################################
# specific methods for ApproxFun
import Base: length, eltype, copyto!
import LinearAlgebra: norm, dot, axpy!, rmul!

eltype(x::ApproxFun.Fun) = eltype(x.coefficients)
length(x::ApproxFun.Fun) = length(x.coefficients)

norm(x::ApproxFun.Fun, p::Real) = (@show p;norm(x.coefficients, p))
norm(x::Array{Fun, 1}, p::Real)  = (@show p;norm(x[3].coefficients, p))
norm(x::Array{Fun{Chebyshev{Segment{Float64}, Float64}, Float64, Array{Float64, 1}}, 1}, p::Real) = (@show p;norm(x[3].coefficients, p))

dot(x::ApproxFun.Fun, y::ApproxFun.Fun) = sum(x * y)
dot(x::Array{Fun{Chebyshev{Segment{Float64}, Float64}, Float64, Array{Float64, 1}}, 1}, y::Array{Fun{Chebyshev{Segment{Float64}, Float64}, Float64, Array{Float64, 1}}, 1}) = sum(x[3] * y[3])

axpy!(a::Float64, x::ApproxFun.Fun, y::ApproxFun.Fun) = (y .= a .* x .+ y)
rmul!(y::ApproxFun.Fun, b::Float64) = (y .= b .* y)

copyto!(x::ApproxFun.Fun, y::ApproxFun.Fun) = (x.coefficients = copy(y.coefficients))
####################################################################################################

source_term(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)
dsource_term(x; a = 0.5, b = 0.01) = (1 - b * x^2 + 2 * a * x)/(1 + b * x^2)^2

function F_chan(u, alpha, beta = 0.01)
	return [Fun(u(0.), domain(sol)) - beta,
			Fun(u(1.), domain(sol)) - beta,
			Δ * u + alpha * source_term(u, b = beta)]
end

function dF_chan(u, v, alpha, beta = 0.01)
	return [Fun(v(0.), domain(sol)),
			Fun(v(1.), domain(sol)),
			Δ * v + alpha * dsource_term(u, b = beta) * v]#Multiplication(dsource_term(u), u.space)]
end

function Jac_chan(u, alpha, beta = 0.01)
	return [Evaluation(u.space, 0.),
			Evaluation(u.space, 1.),
			Δ + alpha * dsource_term(u, b = beta)]#Multiplication(dsource_term(u), u.space)]
end

function finalise_solution(z, tau, step, contResult)
	printstyled(color=:red,"--> AF length = ", (z, tau) .|> length ,"\n")
	# chop!(z, 1e-14);chop!(tau, 1e-14)
	true
end

sol = Fun( x-> x * (1-x), Interval(0.0, 1.0))
const Δ = Derivative(sol.space, 2)

opt_new = Cont.NewtonPar(tol = 1e-12, verbose = true)
	out, hist, flag = @time Cont.newton(
				u -> F_chan(u, 3.0, 0.01),
				u -> Jac_chan(u, 3.0, 0.01),
				sol, opt_new, normN = x -> norm(x, Inf64))
	# Plots.plot(out, label="Solution")

opts_br0 = ContinuationPar(dsmin = 0.0001, dsmax = 0.1, ds= 0.005, a = 0.1, pMax = 4.1, theta = 0.7, secant = true, plot_every_n_steps = 3, newtonOptions = NewtonPar(tol = 1e-8, maxIter = 50, verbose = true), doArcLengthScaling = false)
	opts_br0.newtonOptions.linesearch  = false
	opts_br0.detect_fold = true
	opts_br0.maxSteps = 143

	br, u1 = @time Cont.continuation(
					(x, p) -> F_chan(x, p, 0.01),
					(x, p) -> Jac_chan(x, p, 0.01),
					out, 3.0, opts_br0,
					plot = true,
					finaliseSolution = finalise_solution,
					plotsolution = (x; kwargs...) -> plot!(x, subplot = 4, label = "l = $(length(x))"),
					normC = x -> norm(x, Inf64))


# plot(br.branch[2,:])
#
# @with_kw struct AFLinSolve <: Cont.LinearSolver
# 	tol = 1e-5
# 	maxlength = 10
# end
#
# function (ls::AFLinSolve)(J, x)
#     return ApproxFun.\\(J, x; tolerance = ls.tol, maxlength = ls.maxlength ), true, 1
# end

####################################################################################################
# tangent predictor with Bordered system
opts = ContinuationPar(dsmin = 0.0001, dsmax = 0.2, ds= 0.01, a = 1.5, pMax = 3.5, theta = 0.3, secant = false, plot_every_n_steps = 3, newtonOptions = NewtonPar(tol = 4e-9, maxIter = 50, verbose = true), doArcLengthScaling = false)
	opts.newtonOptions.damped  = false
	opts.detect_fold = true
	opts.maxSteps = 43

	br, u1 = @time Cont.continuation(
								(x, p) -> F_chan(x, p),
								(x, p) -> Jac_chan(x, p),
								out, 0.5, opts,
								plot = true,
								finaliseSolution = finalise_solution,
								plotsolution = (x;kwargs...)-> plot!(x, subplot=4, label = "l = $(length(x))"))
####################################################################################################
# tangent predictor with Bordered system
opts_br0.newtonOptions.verbose = true
indfold = 3
outfold, hist, flag = @time Cont.newtonFold((x, α) -> F_chan(x, α, 0.01),
										(x, p) -> Jac_chan(x, p, 0.01),
										br, indfold, #index of the fold point
										opts_br0.newtonOptions)
		flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold[end], ", β = 0.01, from ", br.bifpoint[indfold][3],"\n")
