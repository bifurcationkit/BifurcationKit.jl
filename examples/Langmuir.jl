using Revise
using PseudoArcLengthContinuation, LinearAlgebra, Plots, ApproxFun
const Cont = PseudoArcLengthContinuation

####################################################################################################
# specific methods for ApproxFun
import Base: length, eltype, copyto!
import LinearAlgebra: norm, dot

eltype(x::ApproxFun.Fun) = eltype(x.coefficients)
length(x::ApproxFun.Fun) = length(x.coefficients)

norm(x::ApproxFun.Fun, p::Real) = (@show p;norm(x.coefficients, p))
norm(x::Array{Fun, 1}, p::Real)  = (@show p;norm(x[3].coefficients, p))
norm(x::Array{Fun{Chebyshev{Segment{Float64}, Float64}, Float64, Array{Float64, 1}}, 1}, p::Real) = (@show p;norm(x[3].coefficients, p))

dot(x::ApproxFun.Fun, y::ApproxFun.Fun) = sum(x * y)
dot(x::Array{Fun{Chebyshev{Segment{Float64}, Float64}, Float64, Array{Float64, 1}}, 1}, y::Array{Fun{Chebyshev{Segment{Float64}, Float64}, Float64, Array{Float64, 1}}, 1}) = sum(x[3]*y[3])

copyto!(x::ApproxFun.Fun, y::ApproxFun.Fun) = (x.coefficients = y.coefficients)
####################################################################################################
function F(c, V, μ = 0.5, c0 = -0.9)
	# plot(c) |> display
	return [Fun(c(0), domain(c)) - c0,
			Fun((∂ * c)(L), domain(c)),
			Fun((Δ * c)(0), domain(c)),
			Fun((Δ * c)(L), domain(c)),
			-(c'' - c^3 + c - μ * ζ )'' - V * c']
end

function Jac(c, V, μ = 0.5, c0 = -0.9)
	return [Evaluation(c.space, 0.),
			Evaluation(c.space, L, 1),
			Evaluation(c.space, 0, 2),
			Evaluation(c.space, L, 2),
			-Δ * (Δ - 3c^2 + I) - V * ∂]
end


const L = 100.
sol = Fun(x -> -1.0 , Interval(0.0, L))
xs = 10.
ls = 2.
const ζ = -1/2Fun(x -> 1 + tanh((x-xs)/ls), Interval(0.0, L))
const Δ = Derivative(sol.space, 2)
const ∂ = Derivative(sol.space, 1)

F(sol,0.07) |> norm

opt_new = Cont.NewtonPar(tol = 1e-12, verbose = true)
	out, hist, flag = @time Cont.newton(
				x -> F(x, 0.07, 0., -0.9),
				x -> Jac(x, 0.07, 0., -0.9),
				sol, opt_new, normN = x -> norm(x, Inf64))


plot(sol)


opts_br0 = ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds= 0.001, a = 1.0, pMax = 0.5, theta = 0.5, secant = true, plot_every_n_steps = 3, newtonOptions = NewtonPar(tol = 1e-6, maxIter = 50, verbose = true), doArcLengthScaling = false)
	opts_br0.newtonOptions.damped  = false
	opts_br0.detect_fold = true
	opts_br0.maxSteps = 14300

	br, u1 = @time Cont.continuation(
								(x, p) -> F(x, 0.07, p, -0.9),
								(x, p) -> Jac(x, 0.07, p, -0.9),
								out, 0.0, opts_br0,
								plot = true,
								# finaliseSolution = finalise_solution,
								plotsolution = (x;kwargs...) -> plot!(x, subplot = 4, label = "l = $(length(x))"))
