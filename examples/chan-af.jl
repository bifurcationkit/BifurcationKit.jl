using Revise
using ApproxFun
using BifurcationKit, Plots
const BK = BifurcationKit
####################################################################################################
# specific methods for ApproxFun
import Base: length, copyto!
import LinearAlgebra: norm

length(x::ApproxFun.Fun) = length(x.coefficients)
copyto!(x::ApproxFun.Fun, y::ApproxFun.Fun) = x .= y

BK.VI.inner(x::ApproxFun.Fun, y::ApproxFun.Fun) = sum(x * y)
BK.VI.scale!(y::ApproxFun.Fun, α::Number) = (y.coefficients .*= α; y)
BK.VI.scalartype(x::ApproxFun.Fun) = eltype(x.coefficients)
BK.VI.zerovector(x::ApproxFun.Fun) = (x.coefficients .= 0; x)
####################################################################################################
N(x; a = 0.5, b = 0.01) = 1 + (x + a * x^2) / (1 + b * x^2)
dN(x; a = 0.5, b = 0.01) = (1 - b * x^2 + 2 * a * x)/(1 + b * x^2)^2

function F_chan(u, p)
    (;α, β) = p
    return [Fun(u(0.), domain(u)) - β,
            Fun(u(1.), domain(u)) - β,
            Δ * u + α * N(u, b = β)]
end

function dF_chan(u, v, p)
    (;α, β) = p
    return [Fun(v(0.), domain(u)),
            Fun(v(1.), domain(u)),
            Δ * v + α * dN(u, b = β) * v]
end

function Jac_chan(u, p)
    (;α, β) = p
    return [Evaluation(u.space, 0.),
            Evaluation(u.space, 1.),
            Δ + α * dN(u, b = β)]
end

function finalise_solution(z, tau, step, contResult)
    printstyled(color=:red,"--> AF length = ", (z, tau) .|> length ,"\n")
    chop!(z.u, 1e-14);chop!(tau.u, 1e-14)
    true
end

sol0 = Fun( x -> x * (1-x), Interval(0.0, 1.0))
const Δ = Derivative(sol0.space, 2);
par_af = (α = 3., β = 0.01)
prob = BifurcationProblem(F_chan, sol0, par_af, (@optic _.α); 
            J = Jac_chan, 
            plot_solution = (x, p; kwargs...) -> plot!(x; label = "l = $(length(x))", kwargs...))

optnew = NewtonPar(tol = 1e-12, verbose = true)
sol = @time BK.solve(prob, Newton(), optnew, normN = norm)

plot(sol.u, label="Solution")

optcont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, p_max = 4.1, plot_every_step = 20, newton_options = NewtonPar(tol = 1e-9, max_iterations = 20, verbose = true), max_steps = 300, detect_bifurcation = 0)

alg = PALC(bls = BorderingBLS(solver = optnew.linsolver, check_precision = false))

br = @time continuation(
    prob, 
    alg,
    optcont;
    plot = true, verbosity = 2,
    normC = norm)
plot(br)
####################################################################################################
# Example with deflation technique
deflationOp = DeflationOperator(2.0, (x, y) -> BK.VI.inner(x, y), 1.0, [sol.u])
par_def = @set par_af.α = 3.3

optdef = setproperties(optnew; tol = 1e-9, max_iterations = 1000)

solp = copy(sol.u)
solp.coefficients .*= (1 .+ 0.41*rand(length(solp.coefficients)))

plot(sol.u);plot!(solp)

outdef1 = @time BK.solve(BK.re_make(prob, u0 = solp), deflationOp, optdef)
BK.converged(outdef1) && push!(deflationOp, outdef1.u)

plot(deflationOp.roots)
####################################################################################################
# other dot product
# dot(x::ApproxFun.Fun, y::ApproxFun.Fun) = sum(x * y) * length(x) # gives 0.1

optcont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, p_max = 4.1, plot_every_step = 10, newton_options = NewtonPar(tol = 1e-8, max_iterations = 20, verbose = true), max_steps = 300, detect_bifurcation = 0)

    br = @time continuation(
        prob, PALC(bls=BorderingBLS(solver = optnew.linsolver, check_precision = false), θ = 0.2), optcont;
        plot = true,
        verbosity = 2,
        normC = norminf)
####################################################################################################
# tangent predictor with Bordered system
br = @time continuation(
    prob, PALC(tangent=Bordered(), bls=BorderingBLS(solver = optnew.linsolver, check_precision = false)), optcont,
    plot = true,)
####################################################################################################
# fold continuation
indfold = 2
outfold = @time BK.newton(
        br, indfold, #index of the fold point
        bdlinsolver = BorderingBLS(solver = DefaultLS(false), check_precision = false)
        )
BK.converged(outfold) && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.u[end], ", β = 0.01, from ", br.specialpoint[indfold][3],"\n")
