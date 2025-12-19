using Revise
using LinearAlgebra, SparseArrays, BandedMatrices

using Plots, BifurcationKit
const BK = BifurcationKit

# ==============================================================================
# Carrier BVP Problem
# ==============================================================================

# 1. Define the vector field (first-order form)
function F_bvp(y, p)
	return [y[2], 
			4/p.ϵ^2 * (1 - y[1]^2 - 2 * ((2*y[3]-1)^2)*y[1]), 
			1,]
end

# 2. Define boundary conditions: y₁(0) = 0, y₁(1) = 0
function g_bvp(u0, uT, p)
    return [u0[1],  # y₁(0)
			uT[1],  # t0=0
			u0[3],] # y₁(1) = 0
end

# 3. Create BVP Model
# State dimension is 3 (u, u', Tau)
# Fixed interval [0, 1] => phase condition fixes T=1.0
nf = 3
ng = 3
model = BVPModel(F_bvp, g_bvp; n=nf, phase = (u, p, T) -> T - 1)
# 4. Discretize using Collocation method
disc = Collocation(Ntst=30, m=4)
bvp = discretize(model, disc)

# 5. Set up parameters and initial guess
params = (ϵ = 0.7,)
y0 = zeros(nf * (disc.m * disc.Ntst) + ng + 0)
yslice = reduce(hcat, [[0,0,t] for t in LinRange(0,1,disc.m * disc.Ntst,)])
# yslice = generate_solution(PeriodicOrbitOCollProblem(disc.Ntst, disc.m; N=3), t->[t*(1-t),-2t,t], 1)[1:nf * (disc.m * disc.Ntst)]
#y0 = vcat(0vec(yslice), zeros(ng))


#y0[end] = 1.0 # Interval length T = 1.0
record_from_solution(y, p; k...) = (y[2]-y[1]) * sum(y->y^2, y)

# 6. Create BVPBifProblem
prob = BVPBifProblem(bvp, y0, params, (@optic _.ϵ);
    record_from_solution,
	plot_solution = (x, p; kwargs...) -> begin
        u = BifurcationKit.get_time_slices(x[1:end], 3, disc.m, disc.Ntst)
        plot!(u[3, :],u[1, :]; ylabel="u(t)", title="Bratu Solution (p₁=)", kwargs...)
    end)

# 7. Setup Continuation Parameters
optnew = NewtonPar(tol = 1e-11, verbose = true)
out = @time BK.solve(prob, Newton(), optnew, normN = norminf)
plot(out.u, label = "Solution")

plot();prob.plotSolution(out.u, params; subplot=1)

optcont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= -0.001, p_min = 0.4, newton_options = NewtonPar(tol = 1e-8, max_iterations = 20, verbose = false), nev = 40, n_inversion = 6, detect_bifurcation = 3)
br = @time continuation(
    prob, PALC(bls = BorderingBLS(solver = DefaultLS())), optcont;
    plot = true,
    normC = norminf)

plot(br)

####################################################################################################
# Example with deflation technics
deflationOp = DeflationOperator(2, 1.0, empty([out.u]), copy(out.u))
par_def = @set params.ϵ = 0.6

optdef = setproperties(optnew; tol = 1e-7, max_iterations = 200)

function perturbsol(sol, p, id)
    sol0 = @. exp(-.01/(1-(2p*sol[3]-1)^2)^2)
    solp = 0.5*rand(length(sol))
    return sol .+ solp .* sol0
end

outdef1 = @time BK.solve(
    (@set prob.u0 = perturbsol(-out.u, 0, 0)), 
    deflationOp,
    optdef;
    )
BK.converged(outdef1) && push!(deflationOp, outdef1.u)

plot(); for _s in deflationOp.roots; plot!(_s);end;title!("")
perturbsol(-deflationOp[1],0,0) |> plot
####################################################################################################
# bifurcation diagram
diagram = bifurcationdiagram(prob, PALC(bls = BorderingBLS(solver = DefaultLS())), 2,
        (@set optcont.newton_options.verbose = false);
        autodiff = false,
        plot = true)

plot(diagram, legend = false)

####################################################################################################
# bifurcation diagram with deflated continuation
# empty!(deflationOp)
alg = DefCont(deflation_operator = deflationOp,
                perturb_solution = perturbsol,
                max_branches = 40,
                )

brdc = @time continuation(
    (@set prob.params.ϵ = 0.6), alg,
    setproperties(optcont; ds = -0.0005, dsmin = 1e-5, max_steps = 20000,
        p_max = 0.7, p_min = 0.05, detect_bifurcation = 0, plot_every_step = 40,
        newton_options = setproperties(optnew; tol = 1e-9, max_iterations = 100, verbose = false)),
    ;verbosity = 1,
    normC = norminf,
    )

plot(brdc, legend=true)#, marker=:d)
