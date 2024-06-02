using Revise
using BifurcationKit, Plots, SparseArrays, LinearAlgebra
const BK = BifurcationKit

plotsol(x, Nx=Nx, Ny=Ny) = heatmap(reshape(Array(x), Nx, Ny)', color=:viridis)
plotsol!(x, Nx=Nx, Ny=Ny; kwargs...) = heatmap!(reshape(Array(x), Nx, Ny)'; color=:viridis, kwargs...)

Nx = 151
Ny = 100
lx = 8pi
ly = 2*2pi/sqrt(3)

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

function F_sh(u, p)
    (;l, ν, L1) = p
    return -L1 * u .+ (l .* u .+ ν .* u.^2 .- u.^3)
end

function dF_sh(u, p)
    (;l, ν, L1) = p
    return -L1 .+ spdiagm(0 => l .+ 2 .* ν .* u .- 3 .* u.^2)
end

d2F_sh(u, p, dx1, dx2) = (2 .* p.ν .* dx2 .- 6 .* dx2 .* u) .* dx1
d3F_sh(u, p, dx1, dx2, dx3) = (-6 .* dx2 .* dx3) .* dx1

X = -lx .+ 2lx/(Nx) * collect(0:Nx-1)
Y = -ly .+ 2ly/(Ny) * collect(0:Ny-1)

sol0 = [(cos(x) .+ cos(x/2) * cos(sqrt(3) * y/2) ) for x in X, y in Y]
sol0 .= sol0 .- minimum(vec(sol0))
sol0 ./= maximum(vec(sol0))
sol0 = sol0 .- 0.25
sol0 .*= 1.7
heatmap(sol0', color=:viridis)

Δ, D2x = Laplacian2D(Nx, Ny, lx, ly)
const L1 = (I + Δ)^2
par = (l = -0.1, ν = 1.3, L1 = L1);

optnew = NewtonPar(verbose = true, tol = 1e-8, max_iterations = 20)

prob = BifurcationProblem(F_sh, vec(sol0), par, (@lens _.l); J = dF_sh, plot_solution = (x, p; kwargs...) -> (plotsol!((x); label="", kwargs...)),record_from_solution = (x, p) -> (n2 = norm(x), n8 = norm(x, 8)), d2F=d2F_sh, d3F=d3F_sh)
# optnew = NewtonPar(verbose = true, tol = 1e-8, max_iterations = 20, eigsolver = EigArpack(0.5, :LM))
sol_hexa = @time newton(prob, optnew)
println("--> norm(sol) = ", norm(sol_hexa.u, Inf64))
plotsol(sol_hexa.u)

plotsol(0.2vec(sol_hexa.u) .* vec([exp(-(x+0lx)^2/25) for x in X, y in Y]))

plotsol(0.2vec(sol_hexa.u) .* vec([exp(-(x+0lx)^2/25) for x in X, y in Y]))
###################################################################################################
# recherche de solutions
deflationOp = DeflationOperator(2, 1.0, [sol_hexa.u])

optnew = @set optnew.max_iterations = 250
optnewd = @set optnew.max_iterations = 250
outdef = @time newton(
        (@set prob.u0 = 0.4vec(sol_hexa.u) .* vec([exp(-1(x+lx)^2/25) for x in X, y in Y])), deflationOp,
        # 0.4vec(sol_hexa) .* vec([1 .- exp(-1(x+lx)^2/55) for x in X, y in Y]),
        optnewd)
println("--> norm(sol) = ", norm(outdef.u))
plotsol(outdef.u) |> display
BK.converged(outdef) && push!(deflationOp, outdef.u)

plotsol(deflationOp[end])

plotsol(0.4vec(sol_hexa.u) .* vec([1 .- exp(-1(x+0lx)^2/55) for x in X, y in Y]))
###################################################################################################
optcont = ContinuationPar(dsmin = 0.0001, dsmax = 0.005, ds= -0.001, p_max = -0.0, p_min = -1.0, newton_options = setproperties(optnew; tol = 1e-9, max_iterations = 15, verbose = false), max_steps = 146, detect_bifurcation = 3, nev = 40, dsmin_bisection = 1e-9, n_inversion = 6, tol_bisection_eigenvalue= 1e-19)
optcont = @set optcont.newton_options.eigsolver = EigArpack(0.1, :LM)
# prob.u0 .= deflationOp[1]

br = @time continuation(
    prob, PALC(), optcont;
    plot = true, verbosity = 3,
    # finalise_solution = (z, tau, step, contResult; k...) -> (Base.display(contResult.eig[end].eigenvals) ;true),
    # callback_newton = cb,
    normC = norminf)
###################################################################################################
# codim2 Fold continuation
optfold = @set optcont.detect_bifurcation = 0
@set! optfold.newton_options.verbose = true
optfold = setproperties(optfold; p_min = -2., p_max= 2., dsmax = 0.1)

# dispatch plot to fold solution
plotsol!(x::BorderedArray, args...; k...) = plotsol!(x.u, args...; k...)

brfold = continuation(br, 1, (@lens _.ν), optfold;
    verbosity = 3, plot = true,
    bdlinsolver = MatrixBLS(),
    jacobian_ma = :minaug,
    start_with_eigen = true,
    detect_codim2_bifurcation = 0,
    bothside = true,
    # plot_solution = (x, p; kwargs...) -> (plotsol!((x.u); label="", kwargs...)),
    update_minaug_every_step = 1,
    )

plot(brfold)
###################################################################################################
using IncompleteLU
prec = ilu(L1 + I, τ = 0.15);
prec = lu(L1 + I);
ls = GMRESIterativeSolvers(reltol = 1e-5, N = Nx*Ny, Pl = prec)

function dF_sh2(du, u, p)
    (;l, ν, L1) = p
    return -L1 * du .+ (l .+ 2 .* ν .* u .- 3 .* u.^2) .* du
end

prob2 = @set prob.VF.J = (u, p) -> (du -> dF_sh2(du, u, p))
@set! prob2.u0 = vec(sol0)

sol_hexa = @time newton(prob2, @set optnew.linsolver = ls)
println("--> norm(sol) = ", norm(sol_hexa.u, Inf64))
plotsol(sol_hexa.u)
###################################################################################################
# Automatic branch switching
br2 = continuation(br, 2, setproperties(optcont; ds = -0.001, detect_bifurcation = 0, plot_every_step = 5, max_steps = 170);  
    nev = 30,
    plot = true, verbosity = 2,
    normC = norminf)

plot(br, br2)
###################################################################################################
function optionsCont(x,p,l; opt = optcont)
    if l <= 1
        return opt
    elseif l==2
        return setproperties(opt ;detect_bifurcation = 0,ds = 0.001, a = 0.75)
    else
        return setproperties(opt ;detect_bifurcation = 0,ds = 0.00051, dsmax = 0.01)
    end
end

diagram = bifurcationdiagram(br.prob, br, 2, optionsCont;
    plot = true, verbosity = 0,
    # usedeflation = true,
    # δp = 0.005,
    # callback_newton = cb,
    # linear_algo = MatrixBLS(),
    # finalise_solution = (z, tau, step, contResult; k...) ->     (Base.display(contResult.eig[end].eigenvals) ;true),
    normC = norminf,
    verbosediagram = true,
    )

plot(diagram; code = (), legend = false, plotfold = false)
plot!(br)
###################################################################################################
deflationOp = DeflationOperator(2, 1.0, [sol_hexa.u])
optcontdf = @set optcont.newton_options.verbose = false

algdc = BK.DefCont(deflation_operator = deflationOp, perturb_solution = (x,p,id) -> (x  .+ 0.1 .* rand(length(x))), max_iter_defop = 50, max_branches = 40, seek_every_step = 5,)

brdf = continuation(prob, algdc, setproperties(optcontdf; detect_bifurcation = 0, plot_every_step = 1);
    plot = true, verbosity = 2,
    # finalise_solution = (z, tau, step, contResult) ->     (Base.display(contResult.eig[end].eigenvals) ;true),
    # callback_newton = cb,
    normC = x -> norm(x, Inf))

plot(brdf...)
