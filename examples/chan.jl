using Revise
using BifurcationKit, LinearAlgebra, Plots
const BK = BifurcationKit

Nl(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)
dNl(x; a = 0.5, b = 0.01) = (1-b*x^2+2*a*x)/(1+b*x^2)^2
d2Nl(x; a = 0.5, b = 0.01) = -(2*(-a+3*a*b*x^2+3*b*x-b^2*x^3))/(1+b*x^2)^3

function F_chan(x, p)
    (;α, β) = p
    f = similar(x)
    n = length(x)
    f[1] = x[1] - β
    f[n] = x[n] - β
    for i=2:n-1
        f[i] = (x[i-1] - 2 * x[i] + x[i+1]) * (n-1)^2 + α * Nl(x[i], b = β)
    end
    return f
end

n = 101
par = (α = 3.3, β = 0.01)
sol0 = [(i-1)*(n-i)/n^2+0.1 for i=1:n]

prob = BifurcationProblem(F_chan, sol0, par, (@optic _.α); 
    plot_solution = (x, p; kwargs...) -> (plot!(x;ylabel="solution",label="", kwargs...)))

optnewton = NewtonPar(tol = 1e-8, verbose = true)
# ca fait dans les 63.59k Allocations
sol = @time BK.solve(prob, Newton(), optnewton)

optscont = ContinuationPar(dsmin = 0.01, dsmax = 0.5, ds= 0.1, p_max = 4.25, nev = 5, detect_fold = true, plot_every_step = 10, newton_options = NewtonPar(max_iterations = 10, tol = 1e-9, verbose = false), max_steps = 150)

alg = PALC(tangent = Bordered())
br = @time continuation( prob, alg, optscont; plot = true, verbosity = 0)

# try Moore-Penrose
br_mp = @time continuation( prob, MoorePenrose(tangent = alg), optscont; plot = true, verbosity = 0)
###################################################################################################
# Example with deflation technique
deflationOp = DeflationOperator(2, 1.0, [sol.u])

optdef = NewtonPar(optnewton; tol = 1e-10, max_iterations = 500)

outdef1 = BK.solve(re_make(prob; u0 = sol.u .* (1 .+ 0.01*rand(n))), deflationOp, optdef)
outdef1 = BK.solve(re_make(prob; u0 = sol.u .* (1 .+ 0.01*rand(n))), deflationOp, optdef, Val(:autodiff))

plot(sol.u, label="newton")
plot!(sol0, label="init guess")
plot!(outdef1.u, label="deflation-1")

#save newly found point to look for new ones
push!(deflationOp, outdef1.u)
outdef2 = @time BK.solve((@set prob.u0 = sol0), deflationOp, optdef; callback = BK.cbMaxNorm(1e5))
plot!(outdef2.u, label="deflation-2")
#################################################################################################### Continuation of the Fold Point using minimally augmented formulation
optscont = (@set optscont.newton_options = setproperties(optscont.newton_options; verbose = true, tol = 1e-10))

#index of the fold point
indfold = 2

outfold = @time newton(br, indfold)
BK.converged(outfold) && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.u.p, ", β = 0.01, from ", br.specialpoint[indfold].param,"\n")

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.05, p_max = 4.1, p_min = 0., newton_options = NewtonPar(verbose=false, tol = 1e-8), max_steps = 1300, detect_bifurcation = 0)
foldbranch = @time continuation(br, indfold, (@optic _.β),
    plot = false, verbosity = 0,
    jacobian_ma = BK.MinAug(),
    start_with_eigen = true,
    optcontfold)
plot(foldbranch, label = "")
################################################################################################### Fold Newton / Continuation when Hessian is known. Does not require state to be AbstractVector
d2F(x, p, u, v; b = 0.01) = p.α .* d2Nl.(x; b = b) .* u .* v

prob2 = re_make(prob; d2F)

outfold = @time newton((@set br.prob = prob2), indfold)
BK.converged(outfold) && printstyled(color=:red, "--> We found a Fold Point at α = ", outfold.u.p, ", β = 0.01, from ", br.specialpoint[indfold].param,"\n")

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, p_max = 4.1, p_min = 0., newton_options = NewtonPar(verbose=true, tol = 1e-8), max_steps = 1300, detect_bifurcation = 0)

outfoldco = continuation((@set br.prob = prob2), indfold, (@optic _.β), optcontfold)
###################################################################################################
# Matrix Free example
function dF_chan(x, dx, p)
    (;α, β) = p
    out = similar(x)
    n = length(x)
    out[1] = dx[1]
    out[n] = dx[n]
    for i=2:n-1
        out[i] = (dx[i-1] - 2 * dx[i] + dx[i+1]) * (n-1)^2 + α * dNl(x[i], b = β) * dx[i]
    end
    return out
end

ls = GMRESKrylovKit(dim = 100)
optnewton_mf = NewtonPar(tol = 1e-11, verbose = false, linsolver = ls)
prob2 = @set prob.VF.J = (x, p) -> (dx -> dF_chan(x, dx, p))
out_mf = @time BK.solve(prob2, Newton(), @set optnewton_mf.verbose = true)

opts_cont_mf  = ContinuationPar(dsmin = 0.01, dsmax = 0.5, ds= 0.01, p_max = 4.2, nev = 5, plot_every_step = 40, newton_options = NewtonPar(optnewton_mf; max_iterations = 10, tol = 1e-8), max_steps = 150, detect_bifurcation = 0)
brmf = @time continuation(prob2, PALC(bls = MatrixFreeBLS(ls)), opts_cont_mf)

plot(brmf)

using SparseArrays
P = spdiagm(0 => -2 * (n-1)^2 * ones(n), -1 => (n-1)^2 * ones(n-1), 1 => (n-1)^2 * ones(n-1))
P[1,1:2] .= [1, 0.];P[end,end-1:end] .= [0, 1.]

lsp = GMRESIterativeSolvers(reltol = 1e-5, N = length(sol.u), restart = 20, maxiter=10, Pl = lu(P))
optnewton_mf = NewtonPar(tol = 1e-9, verbose = false, linsolver = lsp)
out_mf = @time BK.solve(prob2, Newton(), @set optnewton_mf.verbose = true)

plot(brmf, color=:red)

# matrix free with different tangent predictor
brmf = @time continuation(prob2, PALC(tangent = Bordered(), bls = BorderingBLS(lsp)), (@set opts_cont_mf.newton_options = optnewton_mf))

plot(brmf,color=:blue)

alg = brmf.alg
brmf = @time continuation(prob2, MoorePenrose(tangent = alg, method = BifurcationKit.iterative), (@set opts_cont_mf.newton_options = optnewton_mf))

plot(brmf,color=:blue)

brmf = @time continuation(prob2, PALC(tangent = Secant(), bls = BorderingBLS(lsp)), opts_cont_mf)

plot(brmf,color=:green)

brmf = @time continuation(prob2, PALC(tangent = Bordered(), bls = MatrixFreeBLS(ls)), opts_cont_mf)

plot(brmf,color=:orange)
