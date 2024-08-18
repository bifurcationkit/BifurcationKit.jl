using Revise
using Plots
# using GLMakie; Makie.inline!(true)
using BifurcationKit, LinearAlgebra, SparseArrays
const BK = BifurcationKit

normbratu(x) = norm(x .* w) / sqrt(length(x))
##########################################################################################
# plotting function
plotsol!(x, nx = Nx, ny = Ny; kwargs...) = heatmap!(LinRange(0,1,nx), LinRange(0,1,ny), reshape(x, nx, ny)'; color = :viridis, xlabel = "x", ylabel = "y", kwargs...)
plotsol(x, nx = Nx, ny = Ny; kwargs...) = (plot();plotsol!(x, nx, ny; kwargs...))
# plotsol!(ax, x, nx = Nx, ny = Ny; ax1=nothing, kwargs...) = heatmap!(ax, LinRange(0,1,nx), LinRange(0,1,ny), reshape(x, nx, ny)')

function Laplacian2D(Nx, Ny, lx, ly)
    hx = 2lx/Nx
    hy = 2ly/Ny
    D2x = spdiagm(0 => -2ones(Nx), 1 => ones(Nx-1), -1 => ones(Nx-1) ) / hx^2
    D2y = spdiagm(0 => -2ones(Ny), 1 => ones(Ny-1), -1 => ones(Ny-1) ) / hy^2

    D2x[1,1] = -1/hx^2
    D2x[end,end] = -1/hx^2

    D2y[1,1] = -1/hy^2
    D2y[end,end] = -1/hy^2

    D2xsp = sparse(D2x)
    D2ysp = sparse(D2y)
    A = kron(sparse(I, Ny, Ny), D2xsp) + kron(D2ysp, sparse(I, Nx, Nx))
    return A, D2x
end

ϕ(u, λ)  = -10(u-λ*exp(u))
dϕ(u, λ) = -10(1-λ*exp(u))

function NL!(dest, u, p)
    (;λ) = p
    dest .= ϕ.(u, λ)
    return dest
end

NL(u, p) = NL!(similar(u), u, p)

function Fmit!(f, u, p)
    mul!(f, p.Δ, u)
    f .= f .+ NL(u, p)
    return f
end

function dFmit(x, p, dx)
    f = similar(dx)
    mul!(f, p.Δ, dx)
    nl = d1NL(x, p, dx)
    f .= f .+ nl
end

function JFmit(x,p)
    J = p.Δ
    dg = dϕ.(x, p.λ)
    return J + spdiagm(0 => dg)
end

# computation of the derivatives
d1NL(x, p, dx) = ForwardDiff.derivative(t -> NL(x .+ t .* dx, p), 0.)
####################################################################################################
Nx = 30
Ny = 30
lx = 0.5
ly = 0.5

Δ, D = Laplacian2D(Nx, Ny, lx, ly)
par_mit = (λ = .01, Δ = Δ)
sol0 = 0*ones(Nx, Ny) |> vec
const w = (lx .+ LinRange(-lx,lx,Nx)) * transpose(LinRange(-ly,ly,Ny)) |> vec
w .-= minimum(w)

prob = BK.BifurcationProblem(Fmit!, sol0, par_mit, (@optic _.λ);
        J = JFmit,
        record_from_solution = (x, p) -> (nw = normbratu(x), n2 = norm(x), n∞ = norminf(x)),
        plot_solution = (x, p; k...) -> plotsol!(x ; k...),
        # plot_solution = (ax, x, p; ax1 = nothing, k...) -> plotsol!(ax, x ; k...),
        issymmetric = true)
####################################################################################################
eigls = EigArpack(20.5, :LM)
# eigls = EigKrylovKit(dim = 70)
# eigls = EigArpack()
opt_newton = NewtonPar(tol = 1e-8, verbose = true, eigsolver = eigls, max_iterations = 20)
sol = newton(prob, opt_newton, normN = norminf)

plotsol(sol.u)
####################################################################################################
function finSol(z, tau, step, br; k...)
    if length(br.specialpoint)>0
        if br.specialpoint[end].step == step
            BK._show(stdout, br.specialpoint[end], step)
        end
    end
    return true
end

function cb(state; kwargs...)
    _x = get(kwargs, :z0, nothing)
    fromNewton = get(kwargs, :fromNewton, false)
    if ~fromNewton && ~isnothing(_x)
        return (norm(_x.u - state.x) < 20.5 && abs(_x.p - state.p) < 0.05)
    end
    true
end

# optional parameters for continuation
kwargsC = (verbosity = 3,
    plot = true,
    callback_newton = cb,
    finalise_solution = finSol,
    normC = norminf
    )

opts_br = ContinuationPar(dsmin = 0.0001, dsmax = 0.04, ds = 0.005, p_max = 3.5, p_min = 0.01, detect_bifurcation = 3, nev = 50, plot_every_step = 10, newton_options = (@set opt_newton.verbose = false), max_steps = 251, tol_stability = 1e-6, n_inversion = 6, max_bisection_steps = 25)

br = @time continuation(prob, PALC(), opts_br; kwargsC...)

BK.plot(br)
####################################################################################################
# automatic branch switching
br1 = continuation(br, 3; kwargsC...)

BK.plot(br, br1, plotfold=false)

br2 = continuation(br1, 1; kwargsC...)

BK.plot(br, br1, br2, plotfold=false)
####################################################################################################
# bifurcation diagram
function optionsCont(x,p,l; opt = opts_br)
    if l <= 1
        return opt
    elseif l==2
        return setproperties(opt ;detect_bifurcation = 3,ds = 0.001, a = 0.75)
    else
        return setproperties(opt ;detect_bifurcation = 3,ds = 0.00051, dsmax = 0.01)
    end
end

diagram = @time bifurcationdiagram(prob, PALC(), 3, optionsCont; kwargsC...,
    usedeflation = true,
    halfbranch = true,
    verbosity = 0
    )

bifurcationdiagram!(prob, get_branch(diagram, (2,1)), 5, optionsCont;
    kwargsC..., usedeflation = true, halfbranch = true,)

code = ()
plot(diagram; code = code,  plotfold = false, putspecialptlegend=false, markersize=2, #vars = (:param, :n2)
vars = (:param, :nw)
)
# plot!(br)
# xlims!(0.01, 0.4)
title!("#branches = $(size(get_branch(diagram, code)))")
# xlims!(0.01, 0.065, ylims=(2.5,6.5))

plot(get_branches_from_BP(diagram, 2); plotfold = false, legend = false, vars = (:param, :n2))

get_branch(diagram, (2,1)) |> plot
####################################################################################################
bp2d = @time get_normal_form(br, 2, nev = 30)

res = BK.continuation(br, 2,
    setproperties(opts_br; detect_bifurcation = 3, ds = 0.001, p_min = 0.01, max_steps = 32 ) ;
    nev = 30, verbosity = 3,
    kwargsC...,
    )

plot(res..., br ;plotfold= false)
####################################################################################################
# deflated continuation
deflationOp = DeflationOperator(2, 1., ([sol0]))
algdc = BK.DefCont(deflation_operator = deflationOp, max_branches = 150, perturb_solution = (x,p,id) -> (x .+ 0.1 .* rand(length(x))))

brdef2 = @time BK.continuation(
    (@set prob.params.λ = 0.367), algdc,
    ContinuationPar(opts_br; ds = -0.0001, max_steps = 800000, plot_every_step = 10, detect_bifurcation = 0);
    plot=true, verbosity = 2,
    normC = norminf)

plot(brdef2, color=:red)
