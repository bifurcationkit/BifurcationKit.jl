using Revise
using BifurcationKit, LinearAlgebra, Plots, SparseArrays, Parameters
const BK = BifurcationKit
using LoopVectorization

f1(u, v) = u^2 * v
norminf(x) = norm(x, Inf)

function plotsol(x; kwargs...)
    N = div(length(x), 2)
    plot!(x[1:N], label="u"; kwargs...)
    plot!(x[N+1:2N], label="v"; kwargs...)
end

function Fbru!(f, x, p, t = 0)
    @unpack α, β, D1, D2, l = p
    n = div(length(x), 2)
    h2 = 1.0 / n^2
    c1 = D1 / l^2 / h2
    c2 = D2 / l^2 / h2

    u = @view x[1:n]
    v = @view x[n+1:2n]

    # Dirichlet boundary conditions
    f[1]   = c1 * (α      - 2u[1] + u[2] ) + α - (β + 1) * u[1] + f1(u[1], v[1])
    f[end] = c2 * (v[n-1] - 2v[n] + β / α)           + β * u[n] - f1(u[n], v[n])

    f[n]   = c1 * (u[n-1] - 2u[n] +  α   ) + α - (β + 1) * u[n] + f1(u[n], v[n])
    f[n+1] = c2 * (β / α  - 2v[1] + v[2])            + β * u[1] - f1(u[1], v[1])

    @turbo for i=2:n-1
          f[i] = c1 * (u[i-1] - 2u[i] + u[i+1]) + α - (β + 1) * u[i] + f1(u[i], v[i])
        f[n+i] = c2 * (v[i-1] - 2v[i] + v[i+1])           + β * u[i] - f1(u[i], v[i])
    end
    return f
end

Fbru(x, p, t = 0) = Fbru!(similar(x), x, p, t)

function Jbru_sp(x, p)
    @unpack α, β, D1, D2, l = p
    # compute the Jacobian using a sparse representation
    n = div(length(x), 2)
    h2 = 1.0 / n^2

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

n = 100
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
probBif = BK.BifurcationProblem(Fbru, sol0, par_bru, (@lens _.l);
        J = Jbru_sp,
        plotSolution = (x, p; kwargs...) -> (plotsol(x; label="", kwargs... )),
        recordFromSolution = (x, p) -> x[div(n,2)])
# par_bru = (α = 2., β = 4.6, D1 = 0.0016, D2 = 0.008, l = 0.061)
#     xspace = LinRange(0, par_bru.l, n)
#     sol0 = vcat(        par_bru.α .+ 2 .* sin.(pi*xspace/par_bru.l),
#             par_bru.β/par_bru.α .- 0.5 .* sin.(pi*xspace/par_bru.l))

# eigls = EigArpack(1.1, :LM)
#     opt_newton = BK.NewtonPar(eigsolver = eigls)
#     out, hist, flag = @time BK.newton(
#         x ->  Fbru(x, par_bru),
#         x -> Jbru_sp(x, par_bru),
#         sol0, opt_newton, normN = norminf)
#
#         plot();plotsol(out);plotsol(sol0, label = "sol0",line=:dash)
####################################################################################################
eigls = EigArpack(1.1, :LM)
opts_br_eq = ContinuationPar(dsmin = 0.001, dsmax = 0.02, ds = 0.005, pMax = 1.7, detectBifurcation = 3, nev = 21, plotEveryStep = 50, newtonOptions = NewtonPar(eigsolver = eigls, tol = 1e-9), nInversion = 4)

br = @time continuation(
    probBif, PALC(),
    opts_br_eq, verbosity = 0,
    plot = false,
    recordFromSolution = (x, p) -> x[n÷2], normC = norminf)
#################################################################################################### Continuation of Periodic Orbit
M = 10
ind_hopf = 1
l_hopf, Th, orbitguess2, hopfpt, vec_hopf = BK.guessFromHopf(br, ind_hopf, opts_br_eq.newtonOptions.eigsolver, M, 22*0.075)
#
orbitguess_f2 = reduce(hcat, orbitguess2)
orbitguess_f = vcat(vec(orbitguess_f2), Th) |> vec
####################################################################################################
# Standard Shooting
using DifferentialEquations, DiffEqOperators, ForwardDiff

FOde(f, x, p, t) = Fbru!(f, x, p)
Jbru(x, dx, p) = ForwardDiff.derivative(t -> Fbru(x .+ t .* dx, p), 0.)
JacOde(J, x, p, t) = copyto!(J, Jbru_sp(x, p))

u0 = sol0 .+ 0.01 .* rand(2n)
par_hopf = (@set par_bru.l = br.specialpoint[1].param + 0.01)
# prob = ODEProblem(FOde, u0, (0., 520.), par_hopf) # gives 0.68s


#####
jac_prototype = Jbru_sp(ones(2n), @set par_bru.β = 0)
    # jac_prototype.nzval .= ones(length(jac_prototype.nzval))

# vf = ODEFunction(FOde; jac = (J,u,p,t) -> J .= Jbru_sp(u,p), jac_prototype = jac_prototype)
    # prob = ODEProblem(vf,  u0, (0.0, 520.), @set par_bru.l = br.specialpoint[1].param) # gives .37s

using SparseDiffTools, SparseArrays
_colors = matrix_colors(jac_prototype)
# JlgvfColorsAD(J, u, p, colors = _colors) =  SparseDiffTools.forwarddiff_color_jacobian!(J, (out, x) -> Fbru!(out,x,p), u, colorvec = colors)
vf = ODEFunction(FOde; jac_prototype = jac_prototype, colorvec = _colors)
prob = ODEProblem(vf,  sol0, (0.0, 520.), par_bru) # gives 0.22s

sol = @time solve(prob, Rodas4P(); abstol = 1e-10, reltol = 1e-8, progress = true)
####################################################################################################
M = 10
dM = 5
orbitsection = Array(orbitguess_f2[:, 1:dM:M])

initpo = vcat(vec(orbitsection), 3.0)

sol = @time solve(remake(prob, u0=vec(orbitsection), tspan = (0,4.)), QNDF(); abstol = 1e-10, reltol = 1e-8, progress = true)

BK.plotPeriodicShooting(initpo[1:end-1], length(1:dM:M));title!("")

probSh = ShootingProblem(prob,
    QNDF(),
    # QBDF(),#linsolve = KrylovJL_GMRES(verbose=0, rtol = 1e-5), concrete_jac = false),
    # Rodas4P2(),#linsolve = KrylovJL_GMRES(), concrete_jac = false),
    [orbitguess_f2[:,ii] for ii=1:dM:M];
    abstol = 1e-11, reltol = 1e-9,
    parallel = true,
    lens = (@lens _.l),
    par = par_hopf,
    updateSectionEveryStep = 1,
    jacobian = BK.FiniteDifferences(),
    # jacobian = :autodiffMF,
    )

res = @time probSh(initpo, par_hopf)
norminf(res)
res = probSh(initpo, par_hopf, initpo)
norminf(res)

ls = GMRESIterativeSolvers(reltol = 1e-9, N = length(initpo), maxiter = 100, verbose = false)
    optn_po = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 25, linsolver = ls)
    # deflationOp = BK.DeflationOperator(2, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
    outpo = @time newton(probSh,
            initpo, optn_po;
            normN = norminf)
    plot(initpo[1:end-1], label = "Initial guess")
    plot!(outpo.u[1:end-1], label = "solution") |> display
    println("--> amplitude = ", BK.amplitude(outpo.u, n, length(1:dM:M); ratio = 2))
    println("--> period = ", BK.getPeriod(probSh, outpo.u, par_hopf))

opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 1.5, maxSteps = 500, newtonOptions = (@set optn_po.tol = 1e-7), nev = 25, tolStability = 1e-8, detectBifurcation = 0)

# simplified call
eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n), verbose = 0, dim = 40)
# eig = DefaultEig()
opts_po_cont_floquet = @set opts_po_cont.newtonOptions = NewtonPar(linsolver = ls, eigsolver = eig, tol = 1e-7, verbose = true)
opts_po_cont_floquet = setproperties(opts_po_cont_floquet; nev = 10, tolStability = 1e-2, detectBifurcation = 2, maxSteps = 40, ds = 0.03, dsmax = 0.03, pMax = 2.0)

br_po = @time continuation(deepcopy(probSh), outpo.u, PALC(),
        opts_po_cont_floquet;
        verbosity = 3,
        plot = true,
        linearAlgo = MatrixFreeBLS(@set ls.N = 1+length(initpo)),
        # finaliseSolution = (z, tau, step, contResult; k...) ->
            # (Base.display(contResult.eig[end].eigenvals) ;true),
        callbackN = BK.cbMaxNorm(1.),
        plotSolution = (x, p; kwargs...) -> BK.plotPeriodicShooting!(x[1:end-1], length(1:dM:M); kwargs...),
        recordFromSolution = (u, p) -> u[end], normC = norminf)

####################################################################################################
# automatic branch switching with Shooting
# linear solvers
ls = GMRESIterativeSolvers(reltol = 1e-7, maxiter = 100, verbose = false)
eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n), verbose = 0, dim = 40)
# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 25, linsolver = ls, eigsolver = eig)
# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= 0.01, pMax = 1.5, maxSteps = 100, newtonOptions = (@set optn_po.tol = 1e-7), nev = 15, tolStability = 1e-2, detectBifurcation = 3, plotEveryStep = 2, saveSolEveryStep = 2)

Mt = 2
br_po = continuation(
    br, 1,
    # arguments for continuation
    opts_po_cont, ShootingProblem(Mt, prob, QNDF(); abstol = 1e-10, reltol = 1e-8, parallel = false,
                jacobian = BK.FiniteDifferences(),
                # jacobian = BK.AutoDiffMF(),
                updateSectionEveryStep = 1);
    ampfactor = 1., δp = 0.0075,
    verbosity = 3,    plot = true,
    linearAlgo = MatrixFreeBLS(@set ls.N = 2+2n*Mt),
    finaliseSolution = (z, tau, step, contResult; k...) -> begin
        BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
        return true
    end,
    plotSolution = (x, p; kwargs...) -> plot!(x[1:end-1]; label = "", kwargs...),
    normC = norminf)

# ca marche:
br_po2 = continuation(br_po, 2, setproperties(br_po.contparams, detectBifurcation = 0, maxSteps = 300, ds = -0.01);
    verbosity = 3, plot = true,
    ampfactor = .1, δp = 0.01,
    linearAlgo = MatrixFreeBLS(@set ls.N = 2+2n*Mt),
    # usedeflation = false,
    plotSolution = (x, p; kwargs...) -> begin
        BK.plotPeriodicShooting!(x[1:end-1], Mt; kwargs...)
        plot!(br_po; subplot = 1)
    end,
    )

plot(br_po, br_po2, legend=false)
####################################################################################################
# Multiple Poincare Shooting with Hyperplane parametrization
dM = 5
normals = [Fbru(orbitguess_f2[:,ii], par_hopf)/(norm(Fbru(orbitguess_f2[:,ii], par_hopf))) for ii = 1:dM:M]
centers = [orbitguess_f2[:,ii] for ii = 1:dM:M]

probHPsh = PoincareShootingProblem(prob, Rodas4P(), normals, centers;
        abstol = 1e-10, reltol = 1e-8,
        parallel = false,
        δ = 1e-8,
        lens = (@lens _.l),
        par = par_hopf,
        jacobian = :FiniteDifferences)

initpo_bar = reduce(vcat, BK.projection(probHPsh, centers))

# P = @time PrecPartialSchurKrylovKit(dx -> probHPsh(vec(outpo_psh), par_hopf, dx), rand(length(vec(initpo_bar))), 25, :LM; verbosity = 2, krylovdim = 50)
#     scatter(real.(P.eigenvalues), imag.(P.eigenvalues))
#         plot!(1 .+ cos.(LinRange(0,2pi,100)), sin.(LinRange(0,2pi,100)))

ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(vec(initpo_bar)), maxiter = 500, verbose = false)#, Pr = P)
    optn = NewtonPar(verbose = true, tol = 1e-9,  maxIter = 30, linsolver = ls)
    outpo_psh = @time newton(probHPsh, vec(initpo_bar), optn;
        normN = norminf)

plot(outpo_psh.u, label = "Solution")
    plot!(initpo_bar |> vec, label = "Initial guess")

BK.getPeriod(probHPsh, outpo_psh.u, par_hopf)

# simplified call
eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n-1), verbose = 0, dim = 40)
    opts_po_cont_floquet = ContinuationPar(dsmin = 0.0001, dsmax = 0.15, ds= 0.001, pMax = 2.5, maxSteps = 500, nev = 10, tolStability = 1e-5, detectBifurcation = 3, plotEveryStep = 1)
    opts_po_cont_floquet = @set opts_po_cont_floquet.newtonOptions = NewtonPar(linsolver = ls, eigsolver = eig, tol = 1e-7, verbose = true, maxIter = 15)

br_po = @time continuation(probHPsh, outpo_psh.u, PALC(),
    opts_po_cont_floquet;
    linearAlgo = MatrixFreeBLS(@set ls.N = ls.N+1),
    verbosity = 3,
    plot = true,
    plotSolution = (x, p; kwargs...) -> BK.plot!(x; label = "", kwargs...),
    updateSectionEveryStep = 2,
    finaliseSolution = (z, tau, step, contResult; k...) -> begin
        BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
        return true
    end,
    normC = norminf)

####################################################################################################
# automatic branch switching from Hopf point with Poincare Shooting
# linear solver
ls = GMRESIterativeSolvers(reltol = 1e-7, maxiter = 100, verbose = false)
# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-7,  maxIter = 25, linsolver = ls, eigsolver = eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n-1), verbose = 0, dim = 50))
# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= 0.005, pMax = 1.5, maxSteps = 100, newtonOptions = optn_po, nev = 10, tolStability = 1e-5, detectBifurcation = 3, plotEveryStep = 2)

Mt = 2
    br_po = continuation(
    br, 1,
    # arguments for continuation
    opts_po_cont, PoincareShootingProblem(Mt, prob, QNDF(); abstol = 1e-10, reltol = 1e-8, parallel = false, jacobian = :FiniteDifferences);
    linearAlgo = MatrixFreeBLS(@set ls.N = (2n-1)*Mt+1),
    ampfactor = 1.0, δp = 0.005,
    verbosity = 3,    plot = true,
    recordFromSolution = (x, p) -> (period = getPeriod(br_po.prob.prob, x, set(BK.getParams(br_po), BK.getLens(br_po), p.p)),),
    finaliseSolution = (z, tau, step, contResult; k...) -> begin
        BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
        return true
    end,
    plotSolution = (x, p; kwargs...) -> BK.plotPeriodicShooting!(x[1:end-1], Mt; kwargs...),
    normC = norminf)

plot(br_po, legend = :bottomright)

br_po2 = continuation(br_po, 1, setproperties(br_po.contparams, detectBifurcation = 0, maxSteps = 10, saveSolEveryStep = 1);
    verbosity = 3, plot = true,
    ampfactor = .1, δp = 0.01,
    # usedeflation = true,
    linearAlgo = MatrixFreeBLS(@set ls.N = (2n-1)*Mt+1),
    recordFromSolution = (x, p) -> (period = getPeriod(br_po.prob.prob, x, set(BK.getParams(br_po), BK.getLens(br_po), p.p)),),
    plotSolution = (x, p; kwargs...) -> begin
        BK.plotPeriodicShooting!(x[1:end-1], Mt; kwargs...)
        plot!(br_po; subplot = 1)
    end,
    )
