# example taken from Aragón, J. L., R. A. Barrio, T. E. Woolley, R. E. Baker, and P. K. Maini. “Nonlinear Effects on Turing Patterns: Time Oscillations and Chaos.” Physical Review E 86, no. 2 (August 8, 2012): 026201. https://doi.org/10.1103/PhysRevE.86.026201.
using Revise
using ForwardDiff
import OrdinaryDiffEq as ODE
using BifurcationKit, LinearAlgebra, Plots, SparseArrays
const BK = BifurcationKit

f(u, v, p) = p.η * (      u + p.a * v - p.C * u * v - u * v^2)
g(u, v, p) = p.η * (p.H * u + p.b * v + p.C * u * v + u * v^2)

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
    mul!(f, p.Δ, u, 1, 1)
    f
end

NL(u, p) = NL!(similar(u), u, p)
Fbr(x, p, t = 0.) = Fbr!(similar(x), x, p)
# dNL!(o, x, p, dx) = ForwardDiff.derivative!(o, (u,t) -> NL!(u, x .+ t .* dx, p), zero(eltype(dx)))
dNL!(o, x, p, dx) = SparseDiffTools.auto_jacvec(o, (O,U)->NL!(O,U,p,0), x, dx)

function dFbr!(f, x, p, dx)
    dNL!(f, x, p, dx)
    mul!(f, p.Δ, dx, 1, 1)
    f
end
dFbr(x, p, dx) = dFbr!(similar(dx), x, p, dx)

Jbr(x, p) = sparse(ForwardDiff.jacobian(x -> Fbr(x, p), x))
####################################################################################################
N = 100
n = 2N
lx = 3pi /2
h = 2lx/N
X = LinRange(-lx, lx, N)
Δ = spdiagm(0 => -2ones(N), 1 => ones(N-1), -1 => ones(N-1) ) / h^2; Δ[1,1]=Δ[end,end]=-1/h^2
D = 0.08
par_br = (η = 1.0, a = -1., b = -3/2., H = 3.0, D = D, C = -0.6, Δ = blockdiag(D*Δ, Δ))

u0 = cos.(2X)
solc0 = vcat(u0, u0)

probBif = BifurcationProblem(Fbr!, solc0, par_br, (@optic _.C) ;
    J = Jbr,
    record_from_solution = (x, p; k...) -> norminf(x),
    plot_solution = (x, p; iter, state, k...) -> plot!(x[1:end÷2];label="",ylabel ="u", k...))
####################################################################################################
# eigls = DefaultEig()
eigls = EigArpack(0.5, :LM)
optnewton = NewtonPar(eigsolver = eigls, verbose=true, max_iterations = 3200, tol=1e-9)

out = @time BK.solve(probBif, Newton(), optnewton, normN = norminf)
plot();plot!(X,out.u[1:N]);plot!(X,solc0[1:N], label = "sol0",line=:dash)

optcont = ContinuationPar(dsmax = 0.051, ds = -0.001, p_min = -1.8, detect_bifurcation = 3, nev = 21, plot_every_step = 50, newton_options = optnewton, max_steps = 370, n_inversion = 10, max_bisection_steps = 25)

br = @time continuation(re_make(probBif, params = (@set par_br.C = -0.2)), PALC(), (@set optcont.newton_options.verbose = false);
    plot = true, verbosity = 0,)

plot(br)
get_normal_form(br, 1)
####################################################################################################
# branching from Hopf bp using aBS-Trapeze
eig = EigKrylovKit(tol= 1e-10, x₀ = rand(2N), verbose = 2, dim = 40)
eig = DefaultEig()
opt_po = NewtonPar(tol = 1e-9, verbose = true, max_iterations = 20, eigsolver = eig)
optcontpo = ContinuationPar(dsmin = 0.001, dsmax = 0.015, ds= 0.01, p_min = -1.8, max_steps = 60, newton_options = opt_po, nev = 25, tol_stability = 1e-4, detect_bifurcation = 3)

br_po = @time continuation(
    # arguments for branch switching from the first
    # Hopf bifurcation point
    br, 1,
    # arguments for continuation
    optcontpo,
    Trapeze(M = 120, jacobian = BK.FullSparseInplace(), N = 2n);
    # Collocation(30, 4, jacobian = BK.FullSparseInplace(), N = 2n);
    # OPTIONAL parameters
    # we want to jump on the new branch at phopf + δp
    # ampfactor is a factor to increase the amplitude of the guess
    verbosity = 2,
    plot = true,
    callback_newton = BK.cbMaxNorm(1e2),
    plot_solution = (x, p; iter, state, k...) -> begin
        xtt = get_periodic_orbit(p.prob, x, getparams(iter, state))
        heatmap!(1:n, xtt.t, xtt.u'; ylabel="time", color=:viridis, k...)
        plot!(br, subplot=1)
    end,
    record_from_solution = (x, p; iter, state, k...) -> (max = maximum(x[1:end-1]), period = getperiod(p.prob, x, getparams(iter, state))),
    normC = norminf)

plot(br, br_po, label = "")
get_normal_form(br_po, 2, verbose = true)
####################################################################################################
# branching from PD using aBS
br_po_pd = @time continuation(
    # arguments for branch switching from the first
    # PD bifurcation point
    deepcopy(br_po), 1, setproperties(br_po.contparams; detect_bifurcation = 3, plot_every_step = 1, ds = 0.01);
    # OPTIONAL parameters
    # we want to jump on the new branch at p_perioddoubling + δp
    # ampfactor is a factor to increase the amplitude of the guess
    δp = -0.0051,
    # ampfactor = 0.1, # for trapeze!
    verbosity = 3,
    plot = true,
    callback_newton = BK.cbMaxNorm(1e2),
    plot_solution = (x, p; iter, state, k...) -> begin
        xtt = get_periodic_orbit(p.prob, x, getparams(iter, state))
        heatmap!(1:n, xtt.t, xtt.u'; ylabel="time", color=:viridis, k...)
        plot!(br, br_po, subplot=1)
    end,
    record_from_solution = (x, p; iter, state, k...) -> (max = maximum(x[1:end-1]), period = getperiod(p.prob, x, getparams(iter, state))),
    normC = norminf)

plot(br, br_po, br_po_pd, label = "")

get_normal_form(br_po_pd, 5, verbose = true)
####################################################################################################
# Manual branching at the Hopf point using shooting
import SciMLBase
using OrdinaryDiffEqExponentialRK
using OrdinaryDiffEqRosenbrock
par_br_hopf = @set par_br.C = -0.86
f1 = SciMLBase.MatrixOperator(par_br.Δ)
f2 = NL!
prob_ode = ODE.SplitODEProblem(f1, f2, solc0, (0.0, 280.0), par_br_hopf)

sol = @time ODE.solve(prob_ode, ETDRK2(krylov=true); abstol=1e-14, reltol=1e-14, dt = 0.1, progress = true)

prob_ode = ODE.ODEProblem(Fbr, solc0, (0.0, 280.0), par_br_hopf)
sol = @time ODE.solve(prob_ode,  ETDRK2(krylov=true); abstol=1e-14, reltol=1e-7, dt = 0.1, progress = true)
orbitsection = Array(sol[:,[end]])
initpo = vcat(vec(orbitsection), 3.)

BK.plot_periodic_shooting(initpo[1:end-1], 1);title!("")

probSh = Shooting(prob_ode, 
    ETDRK2(krylov=true), 
    # Rodas4(),
    [sol(280.0)]; abstol=1e-14, reltol=1e-14, parallel = true,
    lens = (@optic _.C), par = par_br_hopf, 
    # jacobian = BK.AutoDiffMF(),
    jacobian = BK.FiniteDifferencesMF(),
    dt = 0.1
    )

ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(initpo), maxiter = 50, verbose = false)
optn = NewtonPar(verbose = true, tol = 1e-9,  max_iterations = 20, linsolver = ls)
# deflationOp = BK.DeflationOperator(2 (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
outposh = @time BK.newton(probSh, initpo, optn; normN = norminf)
BK.converged(outposh) && printstyled(color=:red, "--> T = ", outposh.u[end])

plot(initpo[1:end-1], label = "Init guess"); plot!(outposh.u[1:end-1], label = "sol")

eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2N), verbose = 2, dim = 40)
optcontpo = ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds= -0.005, p_min = -1.8, max_steps = 50, newton_options = (@set optn.eigsolver = eig), nev = 10, tol_stability = 1e-2, detect_bifurcation = 3)
br_po_sh = @time continuation(probSh, outposh.u, PALC(), optcontpo;
    verbosity = 3,    plot = true,
    linear_algo = MatrixFreeBLS(@set ls.N = probSh.M*n+2),
    finalise_solution = (z, tau, step, contResult; kw...) ->
        (BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals) ;true),
    plot_solution = (x, p; iter, state, k...) -> BK.plot_periodic_shooting!(x[1:end-1], 1; k...),
    record_from_solution = (x, p; iter, state, k...) -> begin
            xtt = BK.get_periodic_orbit(p.prob, x, getparams(iter, state))
            maximum(xtt[1,:])
    end,
)

# branches = [br_po_sh]
# push!(branches, br_po_sh)
# plot(branches...)

plot(br_po_sh, br, label = "")
pd = get_normal_form(br_po_sh, 1)
BK.predictor(pd, -0.0125, 1)
####################################################################################################
# Manual branching at Period Doubling point based on shooting discretizaton
# solution close to the PD point
par_br_pd = @set par_br.C = -1.32
f1 = SciMLBase.MatrixOperator(par_br.Δ)
f2 = NL!
prob_ode_pd = ODE.SplitODEProblem(f1, f2, solc0, (0.0, 1.0), par_br_pd; abstol=1e-14, reltol=1e-14, dt = 0.01)
solpd = @time ODE.solve(prob_ode_pd, ETDRK2(krylov=true), progress = true)

orbitsectionpd = Array(solpd[:,end-100])
initpo_pd = vcat(vec(orbitsectionpd), 6.19)
BK.plot_periodic_shooting(initpo_pd[1:end-1], 1);title!("")

probSh_pd = deepcopy(probSh)
@reset probSh_pd.M = 2probSh.M
# version based on predictor
if false
    M = probSh_pd.M
    pd = get_normal_form(br_po_sh, 1; detailed = Val(false))
    pred = BK.predictor(pd, -0.0125, 0.1)
    probSh_pd = pred.prob
    initpo_pd = pred.orbitguess
    par_br_pd = @set par_br.C = pred.pnew
else
    # update the section in probSh
    probSh_pd.section.center .= initpo_pd[1:2N]
    probSh_pd.section.normal .= Fbr(initpo_pd[1:2N], par_br_pd)
    probSh_pd.section.normal ./= norm(probSh_pd.section.normal)
end

plot(BK.po_residual(probSh_pd, initpo_pd, par_br_pd))

ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(initpo_pd), maxiter = 50, verbose = false)
optn = NewtonPar(verbose = true, tol = 1e-9,  max_iterations = 12, linsolver = ls)
# deflationOp = BK.DeflationOperator(2 (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
outposh_pd = @time BK.newton((@set probSh_pd.par = par_br_pd), initpo_pd, optn; normN = norminf)
BK.converged(outposh_pd) && printstyled(color=:red, "--> T = ", outposh_pd.u[end])

plot(initpo[1:end-1], label = "Init guess"); plot!(outposh_pd.u[1:end-1], label = "sol")

optcontpo = ContinuationPar(dsmin = 0.0001, dsmax = 0.005, ds= -0.001, p_min = -1.8, max_steps = 500, newton_options = (@set optn.eigsolver = eig), nev = 10, tol_stability = 1e-2, detect_bifurcation = 0)
br_po_sh_pd = @time continuation(deepcopy(outposh_pd.prob.disc), outposh_pd.u, PALC(), optcontpo;
    verbosity = 3, plot = true,
    linear_algo = MatrixFreeBLS(@set ls.N = probSh_pd.M*n+2),
    # finalise_solution = (z, tau, step, contResult; k...) ->
        # (Base.display(contResult.eig[end].eigenvals) ;println("--> T = ", z.u[end]);true),
    plot_solution = (x, p; iter, state, k...) -> (BK.plot_periodic_shooting!(x[1:end-1], 1; k...); plot!(br_po_sh; subplot=1, legend=false)),
    record_from_solution = (x, p; iter, state, k...) -> begin
            xtt = BK.get_periodic_orbit(p.prob, x, getparams(iter, state))
            maximum(xtt[1,:])
    end,
    normC = norminf)

plot(br_po_sh_pd, br, label = "");title!("")
####################################################################################################
# branching from Hopf bp using aBS - Shooting
ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(initpo_pd), maxiter = 50, verbose = false)
eig = EigKrylovKit(tol= 1e-10, x₀ = rand(2N), verbose = 2, dim = 40)
opt_po = NewtonPar(tol = 1e-9, verbose = true, max_iterations = 12, linsolver  = ls)
optcontpo = ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds= -0.005, p_min = -1.8, max_steps = 50, newton_options = (@set opt_po.eigsolver = eig), nev = 20, tol_stability = 1e-2, detect_bifurcation = 3, n_inversion = 8)

probPO = Shooting(1, prob_ode,
                        ETDRK2(krylov=true); 
                        abstol=1e-14, reltol=1e-14,
                        # jacobian = BK.FiniteDifferencesMF(),
                        jacobian = BK.AutoDiffMF(),
                        dt = 0.1,
                        )

br_po = @time continuation(
    # arguments for branch switching from the first Hopf bifurcation point
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
    plot_solution = (x, p; iter, state, k...) -> (BK.plot_periodic_shooting!(x[1:end-1], 1; k...);plot!(br, subplot=1)),
    record_from_solution = (x, p; iter, state, k...) -> begin
            xtt = BK.get_periodic_orbit(p.prob, x, getparams(iter, state))
            maximum(xtt[1,:])
    end,
    normC = norminf)

plot(br, br_po, label = "")

# period-doubling normal form
get_normal_form(br_po, 1; detailed = Val(false))

# aBS from PD
# CQ NE CONVERGE PAS VERS PD CF REMAKE?
# @reset br_po.γ.contparams.newton_options.tol = 1e-7
br_po_pd = BK.continuation(deepcopy(br_po), 1, setproperties(br_po.contparams, detect_bifurcation = 0, max_steps = 5, ds = -0.01, plot_every_step = 1);
    verbosity = 3, plot = true,
    ampfactor = 1., δp = -0.01,
    usedeflation = false,
    # for aBS from period doubling, we double the sections
    linear_algo = MatrixFreeBLS(@set ls.N = 2probPO.M*n+2),
    plot_solution = (x, p; iter, state, k...) -> begin
        outt = BK.get_periodic_orbit(p.prob, x, getparams(iter, state))
        heatmap!(outt[:,:]'; color = :viridis, subplot = 3)
        plot!(br_po; legend=false, subplot=1)
    end,
    record_from_solution = (x, p; iter, state, k...) -> begin
            xtt = BK.get_periodic_orbit(p.prob, x, getparams(iter, state))
            maximum(xtt[1,:])
    end,
    normC = norminf
    )

plot(br_po, br_po_pd, legend=false)


# codim 2
br_po_pdcodim2 = @time continuation(
    # arguments for branch switching from the first
    # Hopf bifurcation point
    br_po, 1, (@optic _.a),
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
    plot_solution = (x, p; iter, state, k...) -> (BK.plot_periodic_shooting!(x[1:end-1], 1; k...);plot!(br, subplot=1)),
    record_from_solution = (x, p; iter, state, k...) -> begin
            xtt = BK.get_periodic_orbit(p.prob, x, getparams(iter, state))
            maximum(xtt[1,:])
    end,
    normC = norminf)
####################################################################################################
# aBS Poincare Shooting
