using Revise
using BifurcationKit, LinearAlgebra, Plots, SparseArrays, Parameters
const BK = BifurcationKit

f1(u, v) = u * u * v

# for Plots.jl
function plotsol(x; kwargs...)
    @show typeof(x)
    N = div(length(x), 2)
    plot!(x[1:N], label="u"; kwargs...)
    plot!(x[N+1:2N], label="v"; kwargs...)
end

function Fbru!(f, x, p, t = 0)
    @unpack α, β, D1, D2, l = p
    n = div(length(x), 2)
    h = 1.0 / n; h2 = h*h
    c1 = D1 / l^2 / h2
    c2 = D2 / l^2 / h2

    u = @view x[1:n]
    v = @view x[n+1:2n]

    # Dirichlet boundary conditions
    f[1]   = c1 * (α      - 2u[1] + u[2] ) + α - (β + 1) * u[1] + f1(u[1], v[1])
    f[end] = c2 * (v[n-1] - 2v[n] + β / α)           + β * u[n] - f1(u[n], v[n])

    f[n]   = c1 * (u[n-1] - 2u[n] +  α   ) + α - (β + 1) * u[n] + f1(u[n], v[n])
    f[n+1] = c2 * (β / α  - 2v[1] + v[2])            + β * u[1] - f1(u[1], v[1])

    for i=2:n-1
          f[i] = c1 * (u[i-1] - 2u[i] + u[i+1]) + α - (β + 1) * u[i] + f1(u[i], v[i])
        f[n+i] = c2 * (v[i-1] - 2v[i] + v[i+1])           + β * u[i] - f1(u[i], v[i])
    end
    return f
end

function Jbru_sp(x, p)
    @unpack α, β, D1, D2, l = p
    # compute the Jacobian using a sparse representation
    n = div(length(x), 2)
    h = 1.0 / n; h2 = h*h

    c1 = D1 / p.l^2 / h2
    c2 = D2 / p.l^2 / h2

    u = @view x[1:n]
    v = @view x[n+1:2n]

    diag   = zeros(eltype(x), 2n)
    diagp1 = zeros(eltype(x), 2n-1)
    diagm1 = zeros(eltype(x), 2n-1)

    diagpn = zeros(eltype(x), n)
    diagmn = zeros(eltype(x), n)

    @. diagmn = β - 2 * u * v
    @. diagm1[1:n-1] = c1
    @. diagm1[n+1:end] = c2

    @. diag[1:n]    = -2c1 - (β + 1) + 2 * u * v
    @. diag[n+1:2n] = -2c2 - u * u

    @. diagp1[1:n-1]   = c1
    @. diagp1[n+1:end] = c2

    @. diagpn = u * u
    J = spdiagm(0 => diag, 1 => diagp1, -1 => diagm1, n => diagpn, -n => diagmn)
    return J
end

function finalise_solution(z, tau, step, contResult)
    n = div(length(z.u), 2)
    printstyled(color=:red, "--> Solution constant = ", norm(diff(z.u[1:n])), " - ", norm(diff(z.u[n+1:2n])), "\n")
    return true
end

n = 500
####################################################################################################
# test for the Jacobian expression
# using ForwardDiff
# sol0 = rand(2n)
# J0 = ForwardDiff.jacobian(x-> Fbru(x, par_bru), sol0) |> sparse
# J1 = Jbru_sp(sol0, par_bru)
# J0 - J1
####################################################################################################
# different parameters to define the Brusselator model and guess for the stationary solution
par_bru = (α = 2., β = 5.45, D1 = 0.008, D2 = 0.004, l = 0.3)
sol0 = vcat(par_bru.α * ones(n), par_bru.β/par_bru.α * ones(n))
prob = BifurcationProblem(Fbru!, sol0, par_bru, (@lens _.l); 
        J = Jbru_sp, 
        plotSolution = (x, p; kwargs...) -> plotsol(x; label="", kwargs... ), 
        # plotSolution = (ax, x, p) -> plotsol(ax, x), 
        recordFromSolution = (x, p) -> x[div(n,2)])

# # parameters for an isola of stationary solutions
# par_bru = (α = 2., β = 4.6, D1 = 0.0016, D2 = 0.008, l = 0.061)
# xspace = LinRange(0, par_bru.l, n)
# sol0 = vcat(        par_bru.α .+ 2 .* sin.(pi*xspace/par_bru.l),
#             par_bru.β/par_bru.α .- 0.5 .* sin.(pi*xspace/par_bru.l))

# eigls = EigArpack(1.1, :LM)
# opt_newton = BK.NewtonPar(eigsolver = eigls)
# out = @time newton(re_make(prob, u0 = sol0), opt_newton, normN = norminf)
# plot();plotsol(out.u);plotsol(sol0, label = "sol0",line=:dash)
####################################################################################################
eigls = EigArpack(1.1, :LM)
opts_br_eq = ContinuationPar(dsmin = 0.03, dsmax = 0.05, ds = 0.03, pMax = 1.9, detectBifurcation = 3, nev = 21, plotEveryStep = 50, newtonOptions = NewtonPar(eigsolver = eigls, tol = 1e-9), maxSteps = 1060, nInversion = 6, tolBisectionEigenvalue = 1e-20, maxBisectionSteps = 30)

br = @time continuation(
    prob, PALC(),
    opts_br_eq, verbosity = 0,
    plot = true,
    normC = norminf)
####################################################################################################
hopfpt = getNormalForm(br, 1; verbose = true)
#################################################################################################### Continuation of the Hopf Point using Jacobian expression
ind_hopf = 1
# hopfpt = BK.HopfPoint(br, ind_hopf)
optnew = opts_br_eq.newtonOptions
hopfpoint = @time newton(br, ind_hopf;
                options = (@set optnew.verbose=true), 
                normN = norminf);
BK.converged(hopfpoint) && printstyled(color=:red, "--> We found a Hopf Point at l = ", hopfpoint.u.p[1], ", ω = ", hopfpoint.u.p[2], ", from l = ", br.specialpoint[ind_hopf].param, "\n")

if 1==0
    br_hopf = @time continuation(
        br, ind_hopf, (@lens _.β),
        ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, newtonOptions = optnew);
        jacobian_ma = :minaug,
        verbosity = 2, normC = norminf)

    plot(br_hopf, label="")
end

if 1==1
    br_hopf = @time continuation(
        br, ind_hopf, (@lens _.β),
        ContinuationPar(opts_br_eq; dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 10.5, pMin = 5.1, detectBifurcation = 0, newtonOptions = optnew);
        updateMinAugEveryStep = 1,
        startWithEigen = true,
        detectCodim2Bifurcation = 2,
        jacobian_ma = :minaug,
        plot = true,
        verbosity = 2, normC = norminf, bothside = true)
end

plot(br_hopf)
####################################################################################################
# automatic branch switching from Hopf point
opt_po = NewtonPar(tol = 1e-10, verbose = true, maxIter = 15)
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.04, ds = 0.01, pMax = 2.2, maxSteps = 200, newtonOptions = opt_po, saveSolEveryStep = 2,
    plotEveryStep = 1, nev = 11, tolStability = 1e-6,
    detectBifurcation = 3, dsminBisection = 1e-6, maxBisectionSteps = 15, nInversion = 4)

br_po = continuation(
    # arguments for branch switching
    br, 1,
    # arguments for continuation
    opts_po_cont, probFD;
    ########
    linearAlgo = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
    ########
    δp = 0.01,
    verbosity = 3,
    plot = true,
    finaliseSolution = (z, tau, step, contResult; k...) -> begin
        @info "Floquet exponents:"
        (Base.display(contResult.eig[end].eigenvals) ;true)
        end,
    # plotSolution = (x, p; kwargs...) -> heatmap!(get_periodic_orbit(p.prob, x, par_bru).u'; ylabel="time", color=:viridis, kwargs...),
    normC = norminf)

####################################################################################################
# semi-automatic branch switching from bifurcation BP-PO
br_po2 = BK.continuation(
    # arguments for branch switching
    br_po, 1,
    # arguments for continuation
    opts_po_cont;
    ampfactor = 1., δp = 0.01,
    usedeflation = true,
    verbosity = 3,
    plot = true,
    ########
    linearAlgo = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
    ########
    finaliseSolution = (z, tau, step, contResult; k...) ->
        (Base.display(contResult.eig[end].eigenvals) ;true),
    # plotSolution = (x, p; kwargs...) -> begin
    #             heatmap!(get_periodic_orbit(p.prob, x, par_bru).u'; ylabel="time", color=:viridis, kwargs...)
    #             plot!(br_po,legend = :bottomright, subplot=1)
    #         end,
    normC = norminf)

branches = Any[br_po]

push!(branches, br_po2)

plot(branches...)
####################################################################################################
using DifferentialEquations
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

vf = ODEFunction(Fbru!, jac = (J,u,p,t)-> J .= Jbru_sp(u,p))
prob_ode = ODEProblem(vf, br_po2.specialpoint[1].x[1:2n], (0,150.), @set par_bru.l = 1.37)
sol_ode = @time solve(prob_ode, Rodas4P(linsolve = KrylovJL_GMRES()), progress = true, progress_steps = 1); println("--> #steps = ", length(sol_ode))
heatmap(sol_ode.t, 1:n, sol_ode[1:n,:], color = :viridis)
plot(sol_ode.t, sol_ode[n÷2,:])