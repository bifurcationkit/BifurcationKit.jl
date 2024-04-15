using Revise
using DiffEqOperators, ForwardDiff
using Plots
# using GLMakie; Makie.inline!(true)
using BifurcationKit, LinearAlgebra, SparseArrays, Parameters
const BK = BifurcationKit

normbratu(x) = norm(x .* w) / sqrt(length(x))
##########################################################################################
# plotting function
plotsol!(x, nx = Nx, ny = Ny; kwargs...) = heatmap!(LinRange(0,1,nx), LinRange(0,1,ny), reshape(x, nx, ny)'; color = :viridis, xlabel = "x", ylabel = "y", kwargs...)
plotsol(x, nx = Nx, ny = Ny; kwargs...) = (plot();plotsol!(x, nx, ny; kwargs...))
# plotsol!(ax, x, nx = Nx, ny = Ny; ax1=nothing, kwargs...) = heatmap!(ax, LinRange(0,1,nx), LinRange(0,1,ny), reshape(x, nx, ny)')

function Laplacian2D(Nx, Ny, lx, ly, bc = :Neumann)
    hx = 2lx/Nx
    hy = 2ly/Ny
    D2x = CenteredDifference(2, 2, hx, Nx)
    D2y = CenteredDifference(2, 2, hy, Ny)

    if bc == :Neumann
        Qx = Neumann0BC(hx)
        Qy = Neumann0BC(hy)
    elseif bc == :Dirichlet
        Qx = Dirichlet0BC(typeof(hx))
        Qy = Dirichlet0BC(typeof(hy))
    end

    D2xsp = sparse(D2x * Qx)[1]
    D2ysp = sparse(D2y * Qy)[1]
    A = kron(sparse(I, Ny, Ny), D2xsp) + kron(D2ysp, sparse(I, Nx, Nx))
    return A, D2x
end

ϕ(u, λ)  = -10(u-λ*exp(u))
dϕ(u, λ) = -10(1-λ*exp(u))

function NL!(dest, u, p)
    @unpack λ = p
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

Δ, = Laplacian2D(Nx, Ny, lx, ly)
par_mit = (λ = .01, Δ = Δ)
sol0 = 0*ones(Nx, Ny) |> vec
const w = (lx .+ LinRange(-lx,lx,Nx)) * transpose(LinRange(-ly,ly,Ny)) |> vec
w .-= minimum(w)

prob = BK.BifurcationProblem(Fmit!, sol0, par_mit, (@lens _.λ);
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
# analyse 2d bifurcation point
bp2d = @time get_normal_form(br, 4, verbose = true, nev = 30)

BK.nf(bp2d)[2] |> println

using ProgressMeter
Nd = 100
L = 3.9
X = LinRange(-L,L, Nd)
Y = LinRange(-L,L, Nd)
P = LinRange(-0.001,0.001, Nd+1)

V1a = @showprogress [bp2d(Val(:reducedForm),[x1,y1], p1)[1] for p1 in P, x1 in X, y1 in Y]

Ind1 = findall( abs.(V1a) .<= 9e-4 * maximum(abs.(V1a)))

    V2a = @showprogress [bp2d(Val(:reducedForm),[X[ii[2]],Y[ii[3]]], P[ii[1]])[2] for ii in Ind1]

    Ind2 = findall( abs.(V2a) .<= 3e-3 * maximum(abs.(V2a)))
    @show length(Ind2)

resp = Float64[]
    resx = Vector{Float64}[]
    resnrm = Float64[]
    @showprogress for k in Ind2
        ii = Ind1[k]
        push!(resp, P[ii[1]])
        # push!(resx, max(X[ii[2]],Y[ii[3]]))
        push!(resnrm, sqrt(X[ii[2]]^2+Y[ii[3]]^2))
        push!(resx, [X[ii[2]], Y[ii[3]]])
    end

using LaTeXStrings

plot(
    scatter(1e4resp, map(x->x[1], resx), map(x->x[2], resx); label = "", markerstrokewidth=0, xlabel = L"10^4 \cdot \lambda", ylabel = L"x_1", zlabel = L"x_2", zcolor = resnrm, color = :viridis,colorbar=false),
    scatter(1e4resp, resnrm; label = "", markersize =2, markerstrokewidth=0, xlabel = L"10^4 \cdot \lambda", ylabel = L"\|x\|"))

####################################################################################################
bp2d = @time get_normal_form(br, 2, nev = 30)

res = BK.continuation(br, 2,
    setproperties(opts_br; detect_bifurcation = 3, ds = 0.001, p_min = 0.01, max_steps = 32 ) ;
    nev = 30, verbosity = 3,
    kwargsC...,
    )

plot(res..., br ;plotfold= false)

δp = 0.005
    deflationOp = DeflationOperator(2, 1.0, [zeros(2)])
        success = [0]
while sum(success) < 10
    pb = BK.BifurcationProblem((x, p) -> bp2d(Val(:reducedForm), x, p[1]), rand(2), [δp])
    outdef1 = newton(pb, deflationOp, NewtonPar(max_iterations = 50))
    @show BK.converged(outdef1)
    BK.converged(outdef1) && push!(deflationOp, outdef1.u)
    (BK.converged(outdef1)==false) && push!(success, 1)
end
println("--> found $(length(deflationOp)) solutions")

plotsol(bp2d(deflationOp[3], δp))
solbif = newton(prob, bp2d.x0, bp2d(deflationOp[3], δp), (@set par_mit.λ = bp2d.p + δp), opts_br.newton_options)[1]

plotsol(solbif.u-0*bp2d(deflationOp[2], δp))

brnf1 = continuation(re_make(prob, u0 = solbif.u, params = (@set par_mit.λ = bp2d.p + δp)), PALC(), setproperties(opts_br; ds = 0.005);
    plot_solution = (x, p; kwargs...) -> plotsol!(x ; kwargs...),
    plot = true, verbosity = 3, normC = norminf)

branches2 = Any[br,br1,br2,brnf1]
push!(branches2, brnf1)
# plot([br,br1,br2])
# plot!(brnf1)

brnf2 = continuation(re_make(prob, u0 = solbif.u, params = (@set par_mit.λ = bp2d.p + δp)), PALC(), setproperties(opts_br; ds = -0.005);
    plot_solution = (x, p; kwargs...) -> plotsol!(x ; kwargs...),
    plot = true, verbosity = 3, normC = norminf)

# plot([br,br1,br2]);plot!(brnf1);plot!(brnf2)
plot(branches2...)
plot!(brnf2)
####################################################################################################
# find isolated branch, see Farrell et al.
deflationOp = DeflationOperator(2, 1.0, [sol0])
optdef = setproperties(opt_newton; tol = 1e-8, max_iterations = 150)

# eigen-elements close to the second bifurcation point on the branch
# of homogeneous solutions
vp, ve, _, _= eigls(JFmit(sol0, @set par_mit.λ = br.specialpoint[2].param), 5)

for ii=1:size(ve, 1)
        outdef1 = @time newton(
            re_make(prob, u0 = real.(br.specialpoint[2].x .+ 0.01 .* ve[ii] .* (1 .+ 0.01 .* rand(Nx*Ny))), params = (@set par_mit.λ = br.specialpoint[2].param + 0.005)), deflationOp,
            optdef)
            BK.converged(outdef1) && push!(deflationOp, outdef1.u)
    end
    length(deflationOp)


l = @layout grid(3,2)
    plot(layout = l)
    for ii=1:min(6,length(deflationOp))
        plotsol!(deflationOp[ii], title="$ii", subplot = ii, label = "", xlabel="$ii", colorbar=true)
    end
    title!("")

brdef1 = @time BK.continuation(
    re_make(prob, u0 = deflationOp[3], params = (@set par_mit.λ = br.specialpoint[2].param + 0.005)), PALC(),
    setproperties(opts_br;ds = 0.001, detect_bifurcation = 0, dsmax = 0.01, max_steps = 500);
    verbosity = 3, plot = true,
    normC = norminf)

plot(br,br1,br2,brdef1,plotfold=false)


brdef2 = @time BK.continuation(
    re_make(brdef1.prob, u0 = deflationOp[5]), PALC(),
    setproperties(opts_br;ds = -0.001, detect_bifurcation = 0, dsmax = 0.02);
    verbosity = 3, plot = true,
    record_from_solution = (x, p) ->  normbratu(x),
    plot_solution = (x, p; kwargs...) -> plotsol!(x ; kwargs...), normC = norminf)

plot(br,br1,br2, brdef1, brdef2,plotfold=false, putspecialptlegend = false)

plot(brdef1, brdef2,plotfold = false, putspecialptlegend = false)
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
