using Revise
using ApproxFun, LinearAlgebra, Parameters, Setfield

using PseudoArcLengthContinuation, Plots
const PALC = PseudoArcLengthContinuation

####################################################################################################
# specific methods for ApproxFun
import Base: eltype, similar, copyto!, length
import LinearAlgebra: mul!, rmul!, axpy!, axpby!, dot, norm

similar(x::ApproxFun.Fun, T) = (copy(x))
similar(x::ApproxFun.Fun) = copy(x)
mul!(w::ApproxFun.Fun, v::ApproxFun.Fun, α) = (w .= α * v)

eltype(x::ApproxFun.Fun) = eltype(x.coefficients)
length(x::ApproxFun.Fun) = length(x.coefficients)

dot(x::ApproxFun.Fun, y::ApproxFun.Fun) = sum(x * y)

# do not put y .= a .* x .+ y, this puts a lot of coefficients!
axpy!(a::Float64, x::ApproxFun.Fun, y::ApproxFun.Fun) = (y .= a * x + y)
axpby!(a::Float64, x::ApproxFun.Fun, b::Float64, y::ApproxFun.Fun) = (y .= a * x + b * y)
rmul!(y::ApproxFun.Fun, b::Float64) = (y.coefficients .*= b; y)
rmul!(y::ApproxFun.Fun, b::Bool) = b == true ? y : (y.coefficients .*= 0; y)

# copyto!(x::ApproxFun.Fun, y::ApproxFun.Fun) = ( copyto!(x.coefficients, y.coefficients);x)
copyto!(x::ApproxFun.Fun, y::ApproxFun.Fun) = ( (x.coefficients = copy(y.coefficients);x))

####################################################################################################

N(x; a = 0.5, b = 0.01) = 1 + (x + a * x^2) / (1 + b * x^2)
dN(x; a = 0.5, b = 0.01) = (1 - b * x^2 + 2 * a * x)/(1 + b * x^2)^2

function F_chan(u, p)
	@unpack alpha, beta = p
	return [Fun(u(0.), domain(u)) - beta,
			Fun(u(1.), domain(u)) - beta,
			Δ * u + alpha * N(u, b = beta)]
end

function dF_chan(u, v, p)
	@unpack alpha, beta = p
	return [Fun(v(0.), domain(u)),
			Fun(v(1.), domain(u)),
			Δ * v + alpha * dN(u, b = beta) * v]
end

function Jac_chan(u, p)
	@unpack alpha, beta = p
	return [Evaluation(u.space, 0.),
			Evaluation(u.space, 1.),
			Δ + alpha * dN(u, b = beta)]
end

function finalise_solution(z, tau, step, contResult)
	printstyled(color=:red,"--> AF length = ", (z, tau) .|> length ,"\n")
	chop!(z.u, 1e-14);chop!(tau.u, 1e-14)
	true
end

sol = Fun( x -> x * (1-x), Interval(0.0, 1.0))
const Δ = Derivative(sol.space, 2);
par_af = (alpha = 3., beta = 0.01)

optnew = NewtonPar(tol = 1e-12, verbose = true)
	out, _, flag = @time PALC.newton(
		F_chan, Jac_chan, sol, par_af, optnew, normN = x -> norm(x, Inf64))
	# Plots.plot(out, label="Solution")

optcont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 4.1, plotEveryNsteps = 10, newtonOptions = NewtonPar(tol = 1e-8, maxIter = 20, verbose = true), maxSteps = 300)

	br, _ = @time continuation(
		F_chan, Jac_chan, out, par_af, (@lens _.alpha), optcont;
		plot = true,
		plotSolution = (x, p; kwargs...) -> plot!(x; label = "l = $(length(x))", kwargs...),
		verbosity = 2,
		normC = x -> norm(x, Inf64))
####################################################################################################
# Example with deflation technique
deflationOp = DeflationOperator(2.0, (x, y) -> dot(x, y), 1.0, [out])
par_def = @set par_af.alpha = 3.3

optdef = setproperties(optnew; tol = 1e-9, maxIter = 1000)

solp = copy(out)
	solp.coefficients .*= (1 .+ 0.41*rand(length(solp.coefficients)))

plot(out);plot!(solp)

outdef1, _, flag = @time newton(
	F_chan, Jac_chan,
	solp, par_def,
	optdef, deflationOp)
	flag && push!(deflationOp, outdef1)

plot(deflationOp.roots)
####################################################################################################
# other dot product
# dot(x::ApproxFun.Fun, y::ApproxFun.Fun) = sum(x * y) * length(x) # gives 0.1

optcont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 4.1, plotEveryNsteps = 10, newtonOptions = NewtonPar(tol = 1e-8, maxIter = 20, verbose = true), maxSteps = 300, theta = 0.2)

	br, _ = @time continuation(
		F_chan, Jac_chan, out, par_af, (@lens _.alpha), optcont;
		dotPALC = (x, y) -> dot(x, y),
		plot = true,
		# finaliseSolution = finalise_solution,
		plotSolution = (x, p; kwargs...) -> plot!(x; label = "l = $(length(x))", kwargs...),
		verbosity = 2,
		# printsolution = x -> norm(x, Inf64),
		normC = x -> norm(x, Inf64))
####################################################################################################
# tangent predictor with Bordered system
br, _ = @time continuation(
	F_chan, Jac_chan, out, par_af, (@lens _.alpha), optcont,
	tangentAlgo = BorderedPred(),
	plot = true,
	finaliseSolution = finalise_solution,
	plotSolution = (x, p;kwargs...)-> plot!(x; label = "l = $(length(x))", kwargs...))
####################################################################################################
# tangent predictor with Bordered system
# optcont = @set optcont.newtonOptions.verbose = true
indfold = 2
outfold, _, flag = @time newtonFold(
		F_chan, Jac_chan,
		br, indfold, #index of the fold point
		par_af, (@lens _.alpha),
		optcont.newtonOptions)
	flag && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold[end], ", β = 0.01, from ", br.bifpoint[indfold][3],"\n")
#################################################################################################### Continuation of the Fold Point using minimally augmented
indfold = 2

outfold, _, flag = @time newtonFold(
	(x, p) -> F_chan(x, p),
	(x, p) -> Jac_chan(x, p),
	(x, p) -> Jac_chan(x, p),
	br, indfold, #index of the fold point
	optcont.newtonOptions)
