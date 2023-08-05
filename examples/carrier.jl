using Revise
using LinearAlgebra, Parameters, SparseArrays, BandedMatrices

using Plots, BifurcationKit
const BK = BifurcationKit
####################################################################################################
function F_carr(x, p)
    @unpack ϵ, X, dx = p
    f = similar(x)
    n = length(x)
    f[1] = x[1]
    f[n] = x[n]
    for i=2:n-1
        f[i] = ϵ^2 * (x[i-1] - 2 * x[i] + x[i+1]) / dx^2 +
            2 * (1 - X[i]^2) * x[i] + x[i]^2-1
    end
    return f
end

function Jac_carr!(J, x, p)
    @unpack ϵ, X, dx = p
    n = length(x)
    J[band(-1)] .= ϵ^2/dx^2                                     # set the diagonal band
    J[band(1)]  .= ϵ^2/dx^2                                     # set the super-diagonal band
    J[band(0)]  .= (-2ϵ^2 /dx^2) .+ 2 * (1 .- X.^2) .+ 2 .* x   # set the second super-diagonal band
    J[1, 1] = 1.0
    J[n, n] = 1.0
    J[1, 2] = 0.0
    J[n, n-1] = 0.0
    J
end
Jac_carr(x, p) = Jac_carr!(BandedMatrix{Float64}(undef, (length(x),length(x)), (1,1)), x, p)

N = 200
X = LinRange(-1,1,N)
dx = X[2] - X[1]
par_car = (ϵ = 0.7, X = X, dx = dx)
sol = -(1 .- par_car.X.^2)
norminf(x) = norm(x,Inf)
recordFromSolution(x, p) = (x[2]-x[1]) * sum(x->x^2, x)

prob = BK.BifurcationProblem(F_carr, zeros(N), par_car, (@lens _.ϵ);
    J = Jac_carr,
    recordFromSolution = recordFromSolution)

optnew = NewtonPar(tol = 1e-8, verbose = true)
    out = @time newton(prob, optnew, normN = x -> norm(x, Inf64))
    plot(out.u, label="Solution")

optcont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= -0.01, pMin = 0.05, plotEveryStep = 10, newtonOptions = NewtonPar(tol = 1e-8, maxIter = 20, verbose = false), maxSteps = 300, detectBifurcation = 3, nev = 40, nInversion = 6, maxBisectionSteps = 25)
    br = @time continuation(
        prob, PALC(bls = BorderingBLS(solver = DefaultLS(), checkPrecision = false)), optcont;
        plot = true, verbosity = 0,
        normC = norminf)

plot(br)

####################################################################################################
# Example with deflation technics
deflationOp = DeflationOperator(2, 1.0, empty([out.u]), copy(out.u))
par_def = @set par_car.ϵ = 0.6

optdef = setproperties(optnew; tol = 1e-7, maxIter = 200)

function perturbsol(sol, p, id)
    sol0 = @. exp(-.01/(1-par_car.X^2)^2)
    # solp = copy(sol)
    solp = 0.02*rand(length(sol))
    # plot([sol, sol + solp * sol0], xlims=(-1,1),title = "Perturb $id") |> display
    return sol .+ solp .* sol0
end

outdef1 = @time newton(
    (@set prob.u0 = perturbsol(-out.u, 0, 0)), deflationOp,
    # perturbsol(deflationOp[1],0,0), par_def,
    optdef;
    # callback = BK.cbMaxNorm(1e8)
    )
    BK.converged(outdef1) && push!(deflationOp, outdef1.u)

plot(); for _s in deflationOp.roots; plot!(_s);end;title!("")
perturbsol(-deflationOp[1],0,0) |> plot
####################################################################################################
# bifurcation diagram with deflated continuation
# empty!(deflationOp)
alg = DefCont(deflationOperator = deflationOp,
                perturbSolution = perturbsol,
                maxBranches = 40,
                )

brdc = @time continuation(
    (@set prob.params.ϵ = 0.6), alg,
    setproperties(optcont; ds = -0.001, dsmin=1e-5, maxSteps = 20000,
        pMax = 0.7, pMin = 0.05, detectBifurcation = 0, plotEveryStep = 40,
        newtonOptions = setproperties(optnew; tol = 1e-9, maxIter = 100, verbose = false)),
    ;verbosity = 1,
    normN = norminf,
    )

plot(brdc, legend=true)#, marker=:d)
scatter!([b.branch[end].param for b in brdc], [b.branch[end][1] for b in brdc], marker = :circle, color=:red, markeralpha=0.5,label = "")
    scatter!([b.branch[1].param for b in brdc], [b.branch[1][1] for b in brdc], marker = :cross, color=:green,  label = "")

br2 = [deepcopy(b) for b in brdc[1:8] if length(b) > 1]
    BifurcationKit.mergeBranches!(br2, it; iterbrsmax = 4)

plot(br2...)
    # scatter!(br2[6])
scatter!([b.branch[end].param for b in brdc], [b.branch[end][1] for b in brdc], marker = :circle, color=:red, markeralpha=0.5,label = "")
    scatter!([b.branch[1].param for b in brdc], [b.branch[1][1] for b in brdc], marker = :cross, color=:green,  label = "")


BifurcationKit.mergeBranches!(brdc, it)

####################################################################################################
# bifurcation diagram
diagram = bifurcationdiagram(prob, PALC(bls = BorderingBLS(solver = DefaultLS(), checkPrecision = false)), 2,
        (arg...) -> @set optcont.newtonOptions.verbose=false;
        plot = true)

plot(diagram, legend=false)
