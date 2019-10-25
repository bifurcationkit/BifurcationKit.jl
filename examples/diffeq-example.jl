using Revise, LinearAlgebra
	using PseudoArcLengthContinuation
	const Cont = PseudoArcLengthContinuation

	using DifferentialEquations, DiffEqOperators
	using Parameters

####################################################################################################
using Setfield, Parameters

f = (u, parm, t) -> (@unpack p, q = parm; p .* u .- u.^3 .+ q)

u0 = [-1.0]
	tspan = (0.0, 1.0)
	parms = (p = 1.0, q = 0.01)
	prob = ODEProblem(f, u0, tspan, parms)
	sol = solve(prob, Tsit5())

opts = NewtonPar()
	optscont = ContinuationPar(pMax = 2., pMin = -2., ds = -0.01)
	sol0, _ = Cont.newton(prob, opts)

l = @lens _.p
	Setfield.get(prob.p, l)
	# prob.u0 .= sol0
	br, u = Cont.continuation(prob, l, optscont)
	plotBranch(br)
####################################################################################################

source_term(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)
	dsource_term(x; a = 0.5, b = 0.01) = (1-b*x^2+2*a*x)/(1+b*x^2)^2
	d2source_term(x; a = 0.5, b = 0.01) = -(2*(-a+3*a*b*x^2+3*b*x-b^2*x^3))/(1+b*x^2)^3

function F_chan!(f, x, p, t)
	@unpack α, β = p
	n = length(x)
	f[1] = x[1] - β
	f[n] = x[n] - β
	for i=2:n-1
		f[i] = (x[i-1] - 2 * x[i] + x[i+1]) * (n-1)^2 + α * source_term(x[i], b = β)
	end
	return f
end

function dF_chan!(out, dx, x, p, t)
	@unpack α, β = p
	n = length(x)
	out[1] = dx[1]
	out[n] = dx[n]
	for i=2:n-1
		out[i] = (dx[i-1] - 2 * dx[i] + dx[i+1]) * (n-1)^2 + α * dsource_term(x[i], b = β) * dx[i]
	end
	return out
end

x0 = rand(100);x0[1] = x0[end] = 0.
p = (α = 3., β = 0.01)

ff = ODEFunction(F_chan!, jac_prototype = JacVecOperator{Float64}(F_chan!, x0, p))
prob = ODEProblem(ff, x0, (0., 2.), p)
sol  = @time solve(prob, ImplicitEuler(), dt = 0.1)

sol  = @time solve(prob, KenCarp4(linsolve=LinSolveGMRES()))
sol  = @time solve(prob, KenCarp4())

using Plots
# plot(sol)
heatmap(sol[:,:])


# version with analytical jacobian
ff2 = ODEFunction(F_chan!, jac_prototype = AnalyticalJacVecOperator{Float64}(dF_chan!, x0, p))
prob2 = ODEProblem(ff, x0, (0., 2.), p)
sol2  = @time solve(prob2, KenCarp4(linsolve=LinSolveGMRES()))


opts = NewtonPar(tol = 1e-9)
	optscont = ContinuationPar(pMax = 5., pMin = -5., ds = 0.01, dsmax = 0.1, dsmin = 0.01, maxSteps = 200, newtonOptions = opts)
	sol0, _ = Cont.newton(prob, opts, normN = norm)

plot(sol0)

l = @lens _.α
	Setfield.get(prob.p, l)
	# prob.u0 .= sol0
	br, u = Cont.continuation(prob, l, optscont, verbosity = 2)
	plotBranch(br) |> display
