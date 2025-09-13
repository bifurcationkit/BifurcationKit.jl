using Revise
using LinearAlgebra, SparseArrays, BandedMatrices

using Plots, BifurcationKit
const BK = BifurcationKit
####################################################################################################
function F_carr(x, p)
    (;ϵ, X, dx) = p
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
    (;ϵ, X, dx) = p
    n = length(x)
    J[band(-1)] .= ϵ^2/dx^2                                     # set the diagonal band
    J[band(1)]  .= ϵ^2/dx^2                                     # set the super-diagonal band
    J[band(0)]  .= @. (-2ϵ^2 /dx^2) + 2 * (1 - X^2) + 2 * x   # set the second super-diagonal band
    J[1, 1] = 1
    J[n, n] = 1
    J[1, 2] = 0
    J[n, n-1] = 0
    J
end
Jac_carr(x, p) = Jac_carr!(BandedMatrix{Float64}(undef, (length(x),length(x)), (1,1)), x, p)

N = 200
X = LinRange(-1,1,N)
dx = X[2] - X[1]
par_car = (ϵ = 0.7, X = X, dx = dx)
sol = -(1 .- par_car.X.^2)
record_from_solution(x, p; k...) = (x[2]-x[1]) * sum(x->x^2, x)

prob = BK.BifurcationProblem(F_carr, zeros(N), par_car, (@optic _.ϵ);
    J = Jac_carr,
    record_from_solution)

optnew = NewtonPar(tol = 1e-8, verbose = true)
out = @time BK.solve(prob, Newton(), optnew, normN = norminf)
plot(out.u, label = "Solution")

optcont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= -0.01, p_min = 0.05, newton_options = NewtonPar(tol = 1e-8, max_iterations = 20, verbose = false), nev = 40, n_inversion = 6)
br = @time continuation(
    prob, PALC(bls = BorderingBLS(solver = DefaultLS())), optcont;
    plot = true, verbosity = 0,
    normC = norminf)

plot(br)

####################################################################################################
# Example with deflation technics
deflationOp = DeflationOperator(2, 1.0, empty([out.u]), copy(out.u))
par_def = @set par_car.ϵ = 0.6

optdef = setproperties(optnew; tol = 1e-7, max_iterations = 200)

function perturbsol(sol, p, id)
    sol0 = @. exp(-.01/(1-par_car.X^2)^2)
    solp = 0.02*rand(length(sol))
    return sol .+ solp .* sol0
end

outdef1 = @time BK.solve(
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
alg = DefCont(deflation_operator = deflationOp,
                perturb_solution = perturbsol,
                max_branches = 40,
                )

brdc = @time continuation(
    (@set prob.params.ϵ = 0.6), alg,
    setproperties(optcont; ds = -0.0001, dsmin = 1e-5, max_steps = 20000,
        p_max = 0.7, p_min = 0.05, detect_bifurcation = 0, plot_every_step = 40,
        newton_options = setproperties(optnew; tol = 1e-9, max_iterations = 100, verbose = false)),
    ;verbosity = 1,
    normC = norminf,
    )

plot(brdc, legend=true)#, marker=:d)
####################################################################################################
# bifurcation diagram
diagram = bifurcationdiagram(prob, PALC(bls = BorderingBLS(solver = DefaultLS())), 2,
        (@set optcont.newton_options.verbose = false);
        autodiff = false,
        plot = true)

plot(diagram, legend = false)
