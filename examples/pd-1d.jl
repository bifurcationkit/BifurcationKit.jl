# example taken from Aragón, J. L., R. A. Barrio, T. E. Woolley, R. E. Baker, and P. K. Maini. “Nonlinear Effects on Turing Patterns: Time Oscillations and Chaos.” Physical Review E 86, no. 2 (August 8, 2012): 026201. https://doi.org/10.1103/PhysRevE.86.026201.
using Revise
using DiffEqOperators, ForwardDiff, DifferentialEquations
using BifurcationKit, LinearAlgebra, Plots, SparseArrays, Parameters
const BK = BifurcationKit

f(u, v, p) = p.η * (      u + p.a * v - p.C * u * v - u * v^2)
g(u, v, p) = p.η * (p.H * u + p.b * v + p.C * u * v + u * v^2)

function Laplacian(N, lx, bc = :Dirichlet)
    hx = 2lx/N
    D2x = CenteredDifference(2, 2, hx, N)
    if bc == :Neumann
        Qx = Neumann0BC(hx)
    elseif bc == :Dirichlet
        Qx = Dirichlet0BC(typeof(hx))
    elseif bc == :Periodic
        Qx = PeriodicBC(typeof(hx))
    end
    D2xsp = sparse(D2x * Qx)[1] |> sparse
end

function NL!(dest, u, p, t = 0.)
    N = div(length(u), 2)
    # u1 = @view u[1:N]
    # u2 = @view u[N+1:end]

    # @tturbo dest[1:N]     .= f.(u1, u2, Ref(p))
    # @tturbo dest[N+1:end] .= g.(u1, u2, Ref(p))
    for ii = 1:N
        u1 = u[ii]
        u2 = u[ii+N]
        dest[ii] = f(u1, u2, p)
        dest[ii+N] = g(u1, u2, p)
    end

    return dest
end

function Fbr!(f, u, p, t = 0.)
     NL!(f, u, p)
    mul!(f, p.Δ, u,1,1)
    f
end

NL(u, p) = NL!(similar(u), u, p)
Fbr(x, p, t = 0.) = Fbr!(similar(x), x, p)
# dNL!(o, x, p, dx) = ForwardDiff.derivative!(o, (u,t) -> NL!(u, x .+ t .* dx, p), zero(eltype(dx)))
dNL!(o, x, p, dx) = SparseDiffTools.auto_jacvec(o, (O,U)->NL!(O,U,p,0), x, dx)

function dFbr!(f, x, p, dx)
    # f .= dNL(x, p, dx)
    dNL!(f, x, p, dx)
    mul!(f, p.Δ, dx, 1, 1)
    f
end
dFbr(x, p, dx) = dFbr!(similar(dx), x, p, dx)

Jbr(x, p) = sparse(ForwardDiff.jacobian(x -> Fbr(x, p), x))

using SparseDiffTools

####################################################################################################
N = 100
    n = 2N
    lx = 3pi /2
    X = LinRange(-lx,lx, N)
    Δ = Laplacian(N, lx, :Neumann)
    D = 0.08
    par_br = (η = 1.0, a = -1., b = -3/2., H = 3.0, D = D, C = -0.6, Δ = blockdiag(D*Δ, Δ))

    u0 = cos.(2X)
    solc0 = vcat(u0, u0)

probBif = BifurcationProblem(Fbr, solc0, par_br, (@lens _.C) ;J = Jbr,
        record_from_solution = (x, p) -> norm(x, Inf),
        plot_solution = (x, p; kwargs...) -> plot!(x[1:end÷2];label="",ylabel ="u", kwargs...))
####################################################################################################
# eigls = DefaultEig()
eigls = EigArpack(0.5, :LM)
optnewton = NewtonPar(eigsolver = eigls, verbose=true, max_iterations = 3200, tol=1e-9)

out = @time newton(probBif, optnewton, normN = norminf)
plot();plot!(X,out.u[1:N]);plot!(X,solc0[1:N], label = "sol0",line=:dash)

optcont = ContinuationPar(dsmax = 0.051, ds = -0.001, p_min = -1.8, detect_bifurcation = 3, nev = 21, plot_every_step = 50, newton_options = optnewton, max_steps = 370, n_inversion = 10, max_bisection_steps = 25)

br = @time continuation(re_make(probBif, params = (@set par_br.C = -0.2)), PALC(), (@set optcont.newton_options.verbose = false);
    plot = true, verbosity = 0,)

plot(br)
get_normal_form(br, 1)
####################################################################################################
# branching from Hopf bp using aBS-Trapezoid
opt_po = NewtonPar(tol = 1e-9, verbose = true, max_iterations = 20)

eig = EigKrylovKit(tol= 1e-10, x₀ = rand(2N), verbose = 2, dim = 40)
eig = DefaultEig()
optcontpo = ContinuationPar(dsmin = 0.001, dsmax = 0.015, ds= 0.01, p_min = -1.8, max_steps = 60, newton_options = (@set opt_po.eigsolver = eig), nev = 25, tol_stability = 1e-4, detect_bifurcation = 3, dsmin_bisection = 1e-6)

M = 100
br_po = @time continuation(
    # arguments for branch switching from the first
    # Hopf bifurcation point
    br, 1,
    # arguments for continuation
    optcontpo,
    PeriodicOrbitTrapProblem(M = M, jacobian = :FullSparseInplace, update_section_every_step = 0);
    # OPTIONAL parameters
    # we want to jump on the new branch at phopf + δp
    # ampfactor is a factor to increase the amplitude of the guess
    verbosity = 2,
    plot = true,
    normN = norminf,
    callbackN = BK.cbMaxNorm(1e2),
    plotSolution = (x, p;kwargs...) ->  (heatmap!(reshape(x[1:end-1], 2*N, M)'; ylabel="time", color=:viridis, kwargs...);plot!(br, subplot=1)),
    recordFromSolution = (u, p) -> (max = maximum(u[1:end-1]), period = u[end]),#BK.maximumPOTrap(u, N, M; ratio = 2),
    normC = norminf)

plot(br, br_po, label = "")
####################################################################################################
# branching from PD using aBS
br_po_pd = @time continuation(
    # arguments for branch switching from the first
    # Hopf bifurcation point
    br_po, 1, setproperties(br_po.contparams; detect_bifurcation = 3, plot_every_step = 1, ds = 0.01);
    # OPTIONAL parameters
    # we want to jump on the new branch at phopf + δp
    # ampfactor is a factor to increase the amplitude of the guess
    ampfactor = 0.9, δp = -0.01,
    verbosity = 3,
    plot = true,
    normN = norminf,
    callbackN = BK.cbMaxNorm(1e2),
    # jacobianPO = :FullSparseInplace,
    # jacobianPO = :BorderedSparseInplace,
    plotSolution = (x, p;kwargs...) ->  (heatmap!(reshape(x[1:end-1], 2*N, M)'; ylabel="time", color=:viridis, kwargs...);plot!(br_po, subplot=1)),
    recordFromSolution = (u, p) -> (max = maximum(u[1:end-1]), period = u[end]),#BK.maximumPOTrap(u, N, M; ratio = 2),
    normC = norminf)

plot(br, br_po, br_po_pd, label = "")
####################################################################################################
# shooting
par_br_hopf = @set par_br.C = -0.86
f1 = DiffEqArrayOperator(par_br.Δ)
f2 = NL!
prob_sp = SplitODEProblem(f1, f2, solc0, (0.0, 280.0), par_br_hopf)

sol = @time solve(prob_sp, ETDRK2(krylov=true); abstol=1e-14, reltol=1e-14, dt = 0.1, progress = true)

prob_ode = ODEProblem(Fbr, solc0, (0.0, 280.0), par_br_hopf)
sol = @time solve(prob_ode, Rodas4P(); abstol=1e-14, reltol=1e-7, dt = 0.1, progress = true)
orbitsection = Array(sol[:,[end]])
# orbitsection = orbitguess[:, 1]

initpo = vcat(vec(orbitsection), 3.)

BK.plot_periodic_shooting(initpo[1:end-1], 1);title!("")

probSh = ShootingProblem(prob_sp, ETDRK2(krylov=true), [sol(280.0)]; abstol=1e-14, reltol=1e-14, dt = 0.1, parallel = true,
    lens = (@lens _.C), par = par_br_hopf, jacobian = BK.FiniteDifferencesMF())
# probSh = ShootingProblem(prob_ode, Rodas4P(), [sol(280.0)]; abstol=1e-10, reltol=1e-4, parallel = true)

plot(probSh(initpo, par_br_hopf))

ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(initpo), maxiter = 50, verbose = false)
# ls = GMRESKrylovKit(verbose = 0, dim = 200, atol = 1e-9, rtol = 1e-5)
optn = NewtonPar(verbose = true, tol = 1e-9,  max_iterations = 20, linsolver = ls)
# deflationOp = BK.DeflationOperator(2 (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
outposh = @time newton(probSh, initpo, optn; normN = norminf)
BK.converged(outposh) && printstyled(color=:red, "--> T = ", outposh.u[end], ", amplitude = ", BK.getamplitude(probSh, outposh.u, par_br_hopf; ratio = 2),"\n")

plot(initpo[1:end-1], label = "Init guess"); plot!(outposh.u[1:end-1], label = "sol")

eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2N), verbose = 2, dim = 40)
# eig = DefaultEig()
optcontpo = ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds= -0.005, p_min = -1.8, max_steps = 50, newton_options = (@set optn.eigsolver = eig), nev = 10, tol_stability = 1e-2, detect_bifurcation = 3)
br_po_sh = @time continuation(probSh, outposh.u, PALC(), optcontpo;
    verbosity = 3,    plot = true,
    linear_algo = MatrixFreeBLS(@set ls.N = probSh.M*n+2),
    finaliseSolution = (z, tau, step, contResult; kw...) ->
        (BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals) ;true),
    plotSolution = (x, p; kwargs...) -> BK.plot_periodic_shooting!(x[1:end-1], 1; kwargs...),
    recordFromSolution = (u, p) -> BK.getmaximum(probSh, u, (@set par_br_hopf.C = p.p); ratio = 2), normC = norminf)

# branches = [br_po_sh]
# push!(branches, br_po_sh)
# plot(branches...)

plot(br_po_sh, br, label = "")

####################################################################################################
# shooting Period Doubling
par_br_pd = @set par_br.C = -1.32
f1 = DiffEqArrayOperator(par_br.Δ)
f2 = NL!
prob_sp = SplitODEProblem(f1, f2, solc0, (0.0, 300.0), par_br_pd; abstol=1e-14, reltol=1e-14, dt = 0.01)
# solution close to the PD point.

solpd = @time solve(prob_sp, ETDRK2(krylov=true), progress = true)
    # heatmap(sol.t, X, sol[1:N,:], color=:viridis, xlim=(20,280.0))

orbitsectionpd = Array(solpd[:,end-100])
initpo_pd = vcat(vec(orbitsectionpd), 6.2)
BK.plot_periodic_shooting(initpo_pd[1:end-1], 1);title!("")

# update the section in probSh
probSh.section.center .= initpo_pd[1:2N]
probSh.section.normal .= Fbr(initpo_pd[1:2N], par_br_pd)
probSh.section.normal ./= norm(probSh.section.normal)

plot(probSh(initpo_pd, par_br_pd))

ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(initpo_pd), maxiter = 50, verbose = false)
# ls = GMRESKrylovKit(verbose = 0, dim = 200, atol = 1e-9, rtol = 1e-5)
optn = NewtonPar(verbose = true, tol = 1e-9,  max_iterations = 12, linsolver = ls)
# deflationOp = BK.DeflationOperator(2 (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
outposh_pd = @time newton(BK.set_params_po(probSh,par_br_pd), initpo_pd, optn;
        # callback = (state; kwargs...) -> (@show state.x[end];true),
        normN = norminf)
BK.converged(outposh_pd) && printstyled(color=:red, "--> T = ", outposh_pd.u[end], ", amplitude = ", BK.getamplitude(probSh, outposh_pd.u, (@set par_br.C = -0.86); ratio = 2),"\n")

plot(initpo[1:end-1], label = "Init guess"); plot!(outposh_pd.u[1:end-1], label = "sol")

optcontpo = ContinuationPar(dsmin = 0.0001, dsmax = 0.005, ds= -0.001, p_min = -1.8, max_steps = 500, newton_options = (@set optn.eigsolver = eig), nev = 10, tol_stability = 1e-2, detect_bifurcation = 0)
br_po_sh_pd = @time continuation(outposh_pd.prob.prob, outposh_pd.u, PALC(), optcontpo;
    verbosity = 3, plot = true,
    linear_algo = MatrixFreeBLS(@set ls.N = probSh.M*n+2),
    # finalise_solution = (z, tau, step, contResult; k...) ->
        # (Base.display(contResult.eig[end].eigenvals) ;println("--> T = ", z.u[end]);true),
    plot_solution = (x, p; kwargs...) -> (BK.plot_periodic_shooting!(x[1:end-1], 1; kwargs...); plot!(br_po_sh; subplot=1, legend=false)),
    record_from_solution = (u, p; k...) -> BK.getmaximum(probSh, u, (@set par_br_pd.C = p.p); ratio = 2),
    normC = norminf)

plot(br_po_sh_pd, br, label = "");title!("")
####################################################################################################
# branching from Hopf bp using aBS - Shooting
ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(initpo_pd), maxiter = 50, verbose = false)
eig = EigKrylovKit(tol= 1e-10, x₀ = rand(2N), verbose = 2, dim = 40)
eig = DefaultEig()

opt_po = NewtonPar(tol = 1e-9, verbose = true, max_iterations = 12, linsolver  = ls)
optcontpo = ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds= -0.005, p_min = -1.8, max_steps = 50, newton_options = (@set opt_po.eigsolver = eig), nev = 20, tol_stability = 1e-2, detect_bifurcation = 3, n_inversion = 8)

probPO = ShootingProblem(1, prob_sp,
                        ETDRK2(krylov=true); 
                        abstol=1e-14, reltol=1e-14,
                        jacobian = BK.FiniteDifferencesMF(),
                        # jacobian = BK.AutoDiffMF(),
                        )

br_po = @time continuation(
    # arguments for branch switching from the first
    # Hopf bifurcation point
    br, 1,
    # arguments for continuation
    optcontpo, probPO;
    # OPTIONAL parameters
    # we want to jump on the new branch at phopf + δp
    # ampfactor is a factor to increase the amplitude of the guess
    δp = 0.005,
    verbosity = 3,
    plot = true,
    linear_algo = MatrixFreeBLS(@set ls.N = probPO.M*n+2),
    finalise_solution = (z, tau, step, contResult; kw...) ->
        (BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals) ;true),
    plot_solution = (x, p; kwargs...) -> (BK.plot_periodic_shooting!(x[1:end-1], 1; kwargs...);plot!(br, subplot=1)),
    record_from_solution = (u, p) -> BK.getmaximum(probPO, u, (@set par_br.C = p.p); ratio = 2),
    normC = norminf)

plot(br, br_po, label = "")

# period-doubling normal form
get_normal_form(br_po, 1, detailed = false)

# aBS from PD
# CQ NE CONVERGE PAS VERS PD CF REMAKE?
@set! br_po.γ.contparams.newton_options.tol = 1e-7
br_po_pd = BK.continuation(br_po, 1, setproperties(br_po.contparams, detect_bifurcation = 0, max_steps = 5, ds = -0.01, plot_every_step = 1);
    verbosity = 3, plot = true,
    ampfactor = .2, δp = -0.01,
    usedeflation = false,
    # for aBS from period doubling, we double the sections
    linear_algo = MatrixFreeBLS(@set ls.N = 2probPO.M*n+2),
    plot_solution = (x, p; kwargs...) -> begin
        outt = BK.get_periodic_orbit(p.prob, x, p.p)
        heatmap!(outt[:,:]'; color = :viridis, subplot = 3)
        plot!(br_po; legend=false, subplot=1)
    end,
    record_from_solution = (u, p) -> (BK.getmaximum(p.prob, u, (@set par_br_hopf.C = p.p); ratio = 2)), normC = norminf
    )

plot(br_po, br_po_pd, legend=false)


# codim 2
br_po_pdcodim2 = @time continuation(
    # arguments for branch switching from the first
    # Hopf bifurcation point
    br_po, 1, (@lens _.a),
    # arguments for continuation
    optcontpo;
    # OPTIONAL parameters
    # we want to jump on the new branch at phopf + δp
    # ampfactor is a factor to increase the amplitude of the guess
    δp = 0.005,
    verbosity = 3,
    plot = true,
    linear_algo = MatrixFreeBLS(@set ls.N = probPO.M*n+2),
    finalise_solution = (z, tau, step, contResult; kw...) ->
        (BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals) ;true),
    plot_solution = (x, p; kwargs...) -> (BK.plot_periodic_shooting!(x[1:end-1], 1; kwargs...);plot!(br, subplot=1)),
    record_from_solution = (u, p) -> BK.getmaximum(probPO, u, (@set par_br.C = p.p); ratio = 2),
    normC = norminf)
####################################################################################################
# aBS Poincare Shooting

br_po.contparams.newton_options.linsolver.solver.N
br_po_pd.contparams.newton_options.linsolver.solver.N

####################################################################################################
