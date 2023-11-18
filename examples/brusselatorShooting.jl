using Revise
using BifurcationKit, LinearAlgebra, Plots, SparseArrays, Parameters
const BK = BifurcationKit
using LoopVectorization

f1(u, v) = u^2 * v

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
probBif = BifurcationProblem(Fbru!, sol0, par_bru, (@lens _.l);
        J = Jbru_sp,
        plot_solution = (x, p; ax1 = 0, kwargs...) -> (plotsol(x; label="", kwargs... )),
        record_from_solution = (x, p) -> x[div(n,2)]
        )
####################################################################################################
eigls = EigArpack(1.1, :LM)
opts_br_eq = ContinuationPar(dsmin = 0.001, dsmax = 0.02, ds = 0.005, p_max = 1.7, detect_bifurcation = 3, nev = 21, plot_every_step = 50, newton_options = NewtonPar(eigsolver = eigls, tol = 1e-9), n_inversion = 4)

br = @time continuation(
    probBif, PALC(),
    opts_br_eq, verbosity = 0,
    plot = true,
    normC = norminf)
#################################################################################################### Continuation of Periodic Orbit
M = 10
ind_hopf = 1
l_hopf, Th, orbitguess2, hopfpt, vec_hopf = BK.guess_from_hopf(br, ind_hopf, opts_br_eq.newton_options.eigsolver, M, 22*0.075)
#
orbitguess_f2 = reduce(hcat, orbitguess2)
orbitguess_f = vcat(vec(orbitguess_f2), Th) |> vec
####################################################################################################
# Standard Shooting
using DifferentialEquations, DiffEqOperators, ForwardDiff

u0 = sol0 .+ 0.01 .* rand(2n)
par_hopf = (@set par_bru.l = br.specialpoint[1].param + 0.01)
prob = ODEProblem(Fbru!, u0, (0., 520.), par_hopf) # gives 0.68s
#####
# this part allows to have AD derive the sparse jacobian for us with very few allocations
jac_prototype = Jbru_sp(ones(2n), @set par_bru.β = 0)
using SparseDiffTools, SparseArrays
_colors = matrix_colors(jac_prototype)
JlgvfColorsAD(J, u, p, colors = _colors) =  SparseDiffTools.forwarddiff_color_jacobian!(J, (out, x) -> Fbru!(out,x,p), u, colorvec = colors)
vf = ODEFunction(Fbru!; jac_prototype = jac_prototype, colorvec = _colors)
prob = ODEProblem(vf,  sol0, (0.0, 520.), par_bru) # gives 0.22s
#####
# solve the Brusselator
sol = @time solve(prob, QNDF(); abstol = 1e-10, reltol = 1e-8, progress = true);
####################################################################################################
M = 10
dM = 5
orbitsection = Array(orbitguess_f2[:, 1:dM:M])

initpo = vcat(vec(orbitsection), 3.0)

sol = @time solve(remake(prob, u0=vec(orbitsection[:, end]), tspan = (0,4.)), QNDF(); abstol = 1e-10, reltol = 1e-8, progress = true)

BK.plot_periodic_shooting(initpo[1:end-1], length(1:dM:M));title!("")

probSh = ShootingProblem(prob,
    Rodas4P(),
    [orbitguess_f2[:,ii] for ii=1:dM:M];
    abstol = 1e-11, reltol = 1e-9,
    parallel = true, #pb with LoopVectorization
    lens = (@lens _.l),
    par = par_hopf,
    update_section_every_step = 1,
    jacobian = BK.FiniteDifferencesMF(),
    # jacobian = BK.AutoDiffMF(),
    )

ls = GMRESIterativeSolvers(reltol = 1e-9, N = length(initpo), maxiter = 100, verbose = false)
optn_po = NewtonPar(verbose = true, tol = 1e-9,  max_iterations = 25, linsolver = ls)
# deflationOp = BK.DeflationOperator(2, (x,y) -> dot(x[1:end-1], y[1:end-1]),1.0, [outpo])
outpo = @time newton(probSh,
                    initpo, optn_po;
                    normN = norminf)
plot(initpo[1:end-1], label = "Initial guess")
plot!(outpo.u[1:end-1], label = "solution") |> display
println("--> amplitude = ", BK.amplitude(outpo.u, n, length(1:dM:M); ratio = 2))
println("--> period = ", BK.getperiod(probSh, outpo.u, par_hopf))

opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, p_max = 1.5, max_steps = 500, newton_options = optn_po, nev = 25, tol_stability = 1e-8, detect_bifurcation = 0)

# simplified call
eig = EigKrylovKit(tol= 1e-7, x₀ = rand(2n), verbose = 0, dim = 40)
# eig = DefaultEig()
opts_po_cont_floquet = @set opts_po_cont.newton_options.eigsolver = eig
opts_po_cont_floquet = setproperties(opts_po_cont_floquet; nev = 10, tol_stability = 1e-2, detect_bifurcation = 2, max_steps = 40, ds = 0.03, dsmax = 0.03, p_max = 2.0)

br_po = @time continuation(deepcopy(probSh), outpo.u, PALC(),
    opts_po_cont_floquet;
    verbosity = 3,
    plot = true,
    linear_algo = MatrixFreeBLS(@set ls.N = 1+length(initpo)),
    # finalise_solution = (z, tau, step, contResult; k...) ->
        # (Base.display(contResult.eig[end].eigenvals) ;true),
    callback_newton = BK.cbMaxNorm(1.),
    plot_solution = (x, p; kwargs...) -> BK.plot_periodic_shooting!(x[1:end-1], length(1:dM:M); kwargs...),
    record_from_solution = (u, p) -> u[end], normC = norminf)

####################################################################################################
# automatic branch switching with Shooting
# linear solvers
ls = GMRESIterativeSolvers(reltol = 1e-9, maxiter = 100, verbose = false)
eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n), verbose = 0, dim = 40)
# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-9,  max_iterations = 25, linsolver = ls, eigsolver = eig)
# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= 0.01, p_max = 1.5, max_steps = 100, newton_options = optn_po, nev = 15, tol_stability = 1e-2, detect_bifurcation = 3, plot_every_step = 2, save_sol_every_step = 2)

Mt = 2
br_po = continuation(
    br, 1,
    # arguments for continuation
    opts_po_cont, 
    ShootingProblem(Mt, prob, Rodas4P(); abstol = 1e-11, reltol = 1e-9, parallel = true,
            jacobian = BK.FiniteDifferencesMF(),
            # jacobian = BK.AutoDiffMF(),
            update_section_every_step = 1);
    ampfactor = 1., δp = 0.005,
    verbosity = 3,
    plot = true,
    linear_algo = MatrixFreeBLS(@set ls.N = 2+2n*Mt),
    finalise_solution = (z, tau, step, contResult; k...) -> begin
        BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
        return true
    end,
    plot_solution = (x, p; kwargs...) -> plot!(x[1:end-1]; label = "", kwargs...),
    normC = norminf)

br_po2 = continuation(deepcopy(br_po), 3, setproperties(br_po.contparams, detect_bifurcation = 0, max_steps = 50, ds = -0.01);
    verbosity = 3, plot = true,
    ampfactor = .2, δp = 0.01,
    linear_algo = MatrixFreeBLS(@set ls.N = 2+2n*Mt),
    # usedeflation = true,
    plot_solution = (x, p; kwargs...) -> begin
        BK.plot_periodic_shooting!(x[1:end-1], Mt; kwargs...)
        plot!(br_po; subplot = 1)
    end,
    callback_newton = BK.cbMaxNorm(20),
    )

plot(br_po, br_po2, legend=false)
####################################################################################################
# Multiple Poincare Shooting with Hyperplane parametrization
dM = 5
Fbru(u,p) = Fbru!(similar(u),u,p)
normals = [Fbru(orbitguess_f2[:,ii], par_hopf)/(norm(Fbru(orbitguess_f2[:,ii], par_hopf))) for ii = 1:dM:M]
centers = [orbitguess_f2[:,ii] for ii = 1:dM:M]

probHPsh = PoincareShootingProblem(prob, QNDF(), normals, centers;
    abstol = 1e-10, reltol = 1e-8,
    parallel = false,
    δ = 1e-8,
    lens = (@lens _.l),
    par = par_hopf,
    jacobian = BK.FiniteDifferencesMF())

initpo_bar = reduce(vcat, BK.projection(probHPsh, centers))

# P = @time PrecPartialSchurKrylovKit(dx -> probHPsh(vec(outpo_psh), par_hopf, dx), rand(length(vec(initpo_bar))), 25, :LM; verbosity = 2, krylovdim = 50)
#     scatter(real.(P.eigenvalues), imag.(P.eigenvalues))
#         plot!(1 .+ cos.(LinRange(0,2pi,100)), sin.(LinRange(0,2pi,100)))

ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(vec(initpo_bar)), maxiter = 500, verbose = false)
optn = NewtonPar(verbose = true, tol = 1e-9,  max_iterations = 30, linsolver = ls)
outpo_psh = @time newton(probHPsh, vec(initpo_bar), optn; normN = norminf)

plot(outpo_psh.u, label = "Solution")
plot!(initpo_bar |> vec, label = "Initial guess")

BK.getperiod(probHPsh, outpo_psh.u, par_hopf)

# eigen solver
eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n-1), verbose = 0, dim = 40)
opts_po_cont_floquet = ContinuationPar(dsmin = 0.0001, dsmax = 0.15, ds= 0.001, p_max = 2.5, max_steps = 500, nev = 10, tol_stability = 1e-5, detect_bifurcation = 3, plot_every_step = 1)
opts_po_cont_floquet = @set opts_po_cont_floquet.newton_options = NewtonPar(linsolver = ls, eigsolver = eig, tol = 1e-7, verbose = true, max_iterations = 15)

br_po = @time continuation(probHPsh, outpo_psh.u, PALC(),
    opts_po_cont_floquet;
    linear_algo = MatrixFreeBLS(@set ls.N = ls.N+1),
    verbosity = 3,
    plot = true,
    plot_solution = (x, p; kwargs...) -> BK.plot!(x; label = "", kwargs...),
    update_section_every_step = 2,
    finalise_solution = (z, tau, step, contResult; k...) -> begin
        BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
        return true
    end,
    normC = norminf)

####################################################################################################
# automatic branch switching from Hopf point with Poincare Shooting
# linear solver
ls = GMRESIterativeSolvers(reltol = 1e-9, maxiter = 100, verbose = false)
# newton parameters
optn_po = NewtonPar(verbose = true, tol = 1e-9,  max_iterations = 25, linsolver = ls, eigsolver = eig = EigKrylovKit(tol= 1e-12, x₀ = rand(2n-1), verbose = 0, dim = 50))
# continuation parameters
opts_po_cont = ContinuationPar(dsmax = 0.03, ds= 0.005, p_max = 1.5, max_steps = 100, newton_options = optn_po, nev = 10, tol_stability = 1e-5, detect_bifurcation = 3, plot_every_step = 2)

br_po = continuation(
    br, 1,
    # arguments for continuation
    opts_po_cont,
    PoincareShootingProblem(1, prob, QNDF(); abstol = 1e-10, reltol = 1e-8, parallel = false, jacobian = BK.FiniteDifferencesMF());
    linear_algo = MatrixFreeBLS(@set ls.N = (2n-1)*Mt+1),
    ampfactor = 1.0, δp = 0.005,
    verbosity = 3,    plot = true,
    record_from_solution = (x, p) -> (period = getperiod(p.prob, x, p.p),),
    finalise_solution = (z, tau, step, contResult; k...) -> begin
        BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
        return true
    end,
    plot_solution = (x, p; kwargs...) -> BK.plot_periodic_shooting!(x[1:end-1], Mt; kwargs...),
    normC = norminf)

plot(br_po, legend = :bottomright)

br_po2 = continuation(deepcopy(br_po), 1, setproperties(br_po.contparams, detect_bifurcation = 0, max_steps = 10, save_sol_every_step = 1);
    verbosity = 3, plot = true,
    ampfactor = .2, δp = 0.01,
    # usedeflation = true,
    linear_algo = MatrixFreeBLS(@set ls.N = (2n-1)*Mt+1),
    record_from_solution = (x, p) -> (period = getperiod(p.prob, x, p.p),),
    plot_solution = (x, p; kwargs...) -> begin
        BK.plot_periodic_shooting!(x[1:end-1], Mt; kwargs...)
        plot!(br_po; subplot = 1)
    end,
    )
