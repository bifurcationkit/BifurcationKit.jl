using Revise
using ForwardDiff, OrdinaryDiffEq
using Plots
# using GLMakie; Makie.inline!(true)
using BifurcationKit, LinearAlgebra, SparseArrays, LoopVectorization
const BK = BifurcationKit

function Laplacian2D(Nx, Ny, lx, ly)
    hx = 2lx/Nx
    hy = 2ly/Ny
    D2x = spdiagm(0 => -2ones(Nx), 1 => ones(Nx-1), -1 => ones(Nx-1) ) / hx^2
    D2y = spdiagm(0 => -2ones(Ny), 1 => ones(Ny-1), -1 => ones(Ny-1) ) / hy^2

    D2x[1,1] = -2/hx^2
    D2x[end,end] = -2/hx^2

    D2y[1,1] = -2/hy^2
    D2y[end,end] = -2/hy^2

    D2xsp = sparse(D2x)
    D2ysp = sparse(D2y)
    A = kron(sparse(I, Ny, Ny), D2xsp) + kron(D2ysp, sparse(I, Nx, Nx))
    return A, D2x
end

function NL!(f, u, p, t = 0.)
    (;r, μ, ν, c3, c5) = p
    n = div(length(u), 2)
    u1 = @view u[1:n]
    u2 = @view u[n+1:2n]

    f1 = @view f[1:n]
    f2 = @view f[n+1:2n]

    @turbo for i=1:n
        ua = u1[i]^2 + u2[i]^2
        f1[i] = r * u1[i] - ν * u2[i] - ua * (c3 * u1[i] - μ * u2[i]) - c5 * ua^2 * u1[i]
        f2[i] = r * u2[i] + ν * u1[i] - ua * (c3 * u2[i] + μ * u1[i]) - c5 * ua^2 * u2[i]
    end

    # ua = u1.^2 .+ u2.^2
    # @. f1 .= r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
    # @. f2 .= r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2

    return f
end

NL(u, p) = NL!(similar(u), u, p)

function Fcgl!(f, u, p, t = 0.)
	mul!(f, p.Δ, u)
	f .= f .+ NL(u, p)
end

function dFcgl!(f, x, p, dx, t = 0)
    dNL!(f, x, p, dx)
    mul!(f, p.Δ, dx, 1,1)
    f
end
dFcgl(x, p, dx) = dFcgl!(similar(dx), x, p, dx)

function Jcgl(u, p, t = 0.)
    (;r, μ, ν, c3, c5, Δ) = p

    n = div(length(u), 2)
    u1 = @view u[1:n]
    u2 = @view u[n+1:2n]

    ua = u1.^2 .+ u2.^2

    f1u = zero(u1)
    f2u = zero(u1)
    f1v = zero(u1)
    f2v = zero(u1)

    @. f1u =  r - 2 * u1 * (c3 * u1 - μ * u2) - c3 * ua - 4 * c5 * ua * u1^2 - c5 * ua^2
    @. f1v = -ν - 2 * u2 * (c3 * u1 - μ * u2)  + μ * ua - 4 * c5 * ua * u1 * u2
    @. f2u =  ν - 2 * u1 * (c3 * u2 + μ * u1)  - μ * ua - 4 * c5 * ua * u1 * u2
    @. f2v =  r - 2 * u2 * (c3 * u2 + μ * u1) - c3 * ua - 4 * c5 * ua * u2 ^2 - c5 * ua^2

    jacdiag = vcat(f1u, f2v)

    Δ + spdiagm(0 => jacdiag, n => f1v, -n => f2u)
end

####################################################################################################
Nx = 41*1
Ny = 21*1
n = Nx*Ny
lx = pi
ly = pi/2

Δ = Laplacian2D(Nx, Ny, lx, ly)[1]
par_cgl = (r = 0.5, μ = 0.1, ν = 1.0, c3 = -1.0, c5 = 1.0, Δ = blockdiag(Δ, Δ))
sol0 = 0.1rand(2Nx, Ny)
sol0_f = vec(sol0)

prob = BK.BifurcationProblem(Fcgl!, sol0_f, par_cgl, (@optic _.r); J = Jcgl)
####################################################################################################
eigls = EigArpack(1.0, :LM)
# eigls = eig_MF_KrylovKit(tol = 1e-8, dim = 60, x₀ = rand(ComplexF64, Nx*Ny), verbose = 1)
opt_newton = NewtonPar(tol = 1e-9, verbose = true, eigsolver = eigls, max_iterations = 20)
opts_br = ContinuationPar(dsmax = 0.02, ds = 0.01, p_max = 2., detect_bifurcation = 3, nev = 15, newton_options = (@set opt_newton.verbose = false), n_inversion = 6)
br = @time continuation(prob, PALC(), opts_br, verbosity = 0)
plot(br)
####################################################################################################
# Look for periodic orbits
f1 = MatrixOperator(par_cgl.Δ);
f2 = NL!
prob_sp = SplitODEProblem(f1, f2, sol0_f, (0.0, 120.0), @set par_cgl.r = 1.2; reltol = 1e-8, dt = 0.1)
prob = ODEProblem(Fcgl!, sol0_f, (0.0, 120.0), (@set par_cgl.r = 1.2))#, jac = Jcgl, jac_prototype = Jcgl(sol0_f, par_cgl))
####################################################################################################
sol = @time OrdinaryDiffEq.solve(prob_sp, ETDRK2(krylov=true); abstol=1e-14, reltol=1e-14, dt = 0.1)
plot(sol.t, [norm(v[1:Nx*Ny], Inf) for v in sol.u], xlims=(105, 120))

# plotting the solution as a movie
for ii = 1:20:length(sol.t)
    # heatmap(reshape(sol[1:Nx*Ny,ii],Nx,Ny),title="$(sol.t[ii])") |> display
end

####################################################################################################
# this encodes the functional for the Shooting problem
probSh = ShootingProblem(
    # we pass the ODEProblem encoding the flow and the time stepper
    prob_sp, ETDRK2(krylov = true),
    [sol[:, end]], abstol = 1e-10, reltol = 1e-8,
    lens = (@optic _.r),
    jacobian = BK.FiniteDifferencesMF())

@assert BK.getparams(probSh) == @set par_cgl.r = 1.2

initpo = vcat(sol[end], 6.3) |> vec
probSh(initpo, @set par_cgl.r = 1.2) |> norminf

ls = GMRESIterativeSolvers(reltol = 1e-4, N = 2n + 1, maxiter = 50, verbose = false)
optn = NewtonPar(verbose = true, tol = 1e-9,  max_iterations = 25, linsolver = ls)
outpo = @time newton(probSh, initpo, optn; normN = norminf);
BK.getperiod(probSh, outpo.u, BK.getparams(probSh))

heatmap(reshape(outpo.u[1:Nx*Ny], Nx, Ny), color = :viridis)

eig = EigKrylovKit(tol = 1e-7, x₀ = rand(2Nx*Ny), verbose = 2, dim = 40)
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.03, ds= -0.01, p_max = 2.5, max_steps = 32, newton_options = (@set optn.eigsolver = eig), nev = 15, tol_stability = 1e-3, detect_bifurcation = 2, plot_every_step = 1)
br_po = @time continuation(probSh, outpo.u, PALC(),
        opts_po_cont;
        verbosity = 3,
        plot = true,
        linear_algo = MatrixFreeBLS(@set ls.N = probSh.M*2n+2),
        plot_solution = (x, p; kwargs...) -> heatmap!(reshape(x[1:Nx*Ny], Nx, Ny); color=:viridis, kwargs...),
        normC = norminf)

####################################################################################################
# automatic branch switching
ls = GMRESIterativeSolvers(reltol = 1e-4, maxiter = 50, verbose = false)
optn = NewtonPar(verbose = true, tol = 1e-9,  max_iterations = 25, linsolver = ls)
eig = EigKrylovKit(tol = 1e-7, x₀ = rand(2Nx*Ny), verbose = 2, dim = 40)
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.02, ds= 0.01, p_max = 2.5, max_steps = 32, newton_options = (@set optn.eigsolver = eig), nev = 15, tol_stability = 1e-3, detect_bifurcation = 3, plot_every_step = 1)

Mt=1
br_po = continuation(
    br, 1,
    # arguments for continuation
    opts_po_cont,
    ShootingProblem(Mt, prob_sp, ETDRK2(krylov = true); abstol = 1e-10, reltol = 1e-8, jacobian = BK.FiniteDifferencesMF(),) ;
    verbosity = 3, plot = true, ampfactor = 1.5, δp = 0.01,
    autodiff_nf = false,
    linear_algo = MatrixFreeBLS(@set ls.N = Mt*2n+2),
    finalise_solution = (z, tau, step, contResult; k...) ->begin
        BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
        return true
    end,
    plot_solution = (x, p; k...) -> heatmap!(reshape(x[1:Nx*Ny], Nx, Ny); color=:viridis, k...),
    # plot_solution = (ax, x, p; kwargs...) -> heatmap!(ax, reshape(x[1:Nx*Ny], Nx, Ny); kwargs...),
    normC = norminf)
