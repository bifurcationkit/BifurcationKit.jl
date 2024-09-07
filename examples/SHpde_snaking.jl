using Revise
using SparseArrays, LinearAlgebra
using BifurcationKit
using Plots
const BK = BifurcationKit
################################################################################
# case of the SH equation
N = 200
l = 6.
X = -l .+ 2l/N*(0:N-1) |> collect
h = X[2]-X[1]

const _weight = rand(N)
normweighted(x) = norm(_weight .* x)

Δ = spdiagm(0 => -2ones(N), 1 => ones(N-1), -1 => ones(N-1) ) / h^2
L1 = -(I + Δ)^2

function R_SH(u, par)
    (;λ, ν, L1) = par
    out = similar(u)
    out .= L1 * u .+ λ .* u .+ ν .* u.^3 - u.^5
end

Jac_sp(u, par) = par.L1 + spdiagm(0 => par.λ .+ 3 .* par.ν .* u.^2 .- 5 .* u.^4)
d2R(u,p,dx1,dx2) = @. p.ν * 6u * dx1 * dx2 - 5 * 4u^3 * dx1 * dx2
d3R(u,p,dx1,dx2,dx3) = @. p.ν * 6dx3 * dx1 * dx2 - 5 * 4 * 3u^2 * dx1 * dx2 * dx3

parSH = (λ = -0.1, ν = 2., L1 = L1)
sol0 = 1.1cos.(X) .* exp.(-0X.^2/(2*5^2))

prob = BifurcationProblem(R_SH, sol0, parSH, (@optic _.λ); J = Jac_sp,
    record_from_solution = (x, p; k...) -> (n2 = norm(x), nw = normweighted(x), s = sum(x), s2 = x[end ÷ 2], s4 = x[end ÷ 4], s5 = x[end ÷ 5]),
    plot_solution = (x, p;kwargs...)->(plot!(X, x; ylabel="solution", label="", kwargs...))
    )
####################################################################################################
optnew = NewtonPar(tol = 1e-12)
sol1 = newton(prob, optnew)
Plots.plot(X, sol1.u)

opts = ContinuationPar(dsmin = 0.0001, dsmax = 0.01, ds = 0.01,
    newton_options = setproperties(optnew; max_iterations = 30, tol = 1e-8), p_max = 1.,
    max_steps = 300, plot_every_step = 40, detect_bifurcation = 3, n_inversion = 4, tol_bisection_eigenvalue = 1e-17, dsmin_bisection = 1e-7)

function cb(state; kwargs...)
    _x = get(kwargs, :z0, nothing)
    fromNewton = get(kwargs, :fromNewton, false)
    if ~fromNewton
        # if the residual is too large or if the parameter jump
        # is too big, abort continuation step
        return norm(_x.u - state.x) < 20.5 && abs(_x.p - state.p) < 0.05
    end
    true
end

kwargsC = (verbosity = 3,
    plot = true,
    linear_algo  = MatrixBLS(),
    callback_newton = cb
    )

brflat = @time continuation(prob, PALC(#=tangent=Bordered()=#), opts; kwargsC..., verbosity = 0)

plot(brflat, putspecialptlegend = false)
####################################################################################################
# branch switching
function optrec(x, p, l; opt = opts)
    level =  l
    if level <= 2
        return setproperties(opt; max_steps = 300, detect_bifurcation = 3, nev = N, detect_loop = false)
    else
        return setproperties(opt; max_steps = 250, detect_bifurcation = 3, nev = N, detect_loop = true)
    end
end

diagram = @time bifurcationdiagram(
            re_make(prob, params = @set parSH.λ = -0.1), 
            PALC(),
            3, 
            optrec; 
            kwargsC..., 
            halfbranch = true, 
            verbosity = 0, 
            usedeflation = false)

code = ()
vars = (:param, :n2)
plot(diagram; code, plotfold = false,  markersize = 2, putspecialptlegend = false, vars)
title!("# branches = $(size(diagram, code))")

diagram2 = bifurcationdiagram!(diagram.γ.prob, BK.get_branch(diagram, (1,)), 3, optrec; kwargsC..., halfbranch = true)

####################################################################################################
deflationOp = DeflationOperator(2, 1.0, [sol1.u])
algdc = BK.DefCont(deflation_operator = deflationOp, max_branches = 50, perturb_solution = (sol, p, id) -> sol .+ 0.02 .* rand(length(sol)),#= alg = PALC(tangent=Secant())=#)

br = @time continuation(
    re_make(prob, params = @set parSH.λ = -0.1), algdc,
    setproperties(opts; ds = 0.001, max_steps = 20000, p_max = 0.25, p_min = -1., newton_options = setproperties(optnew; tol = 1e-9, max_iterations = 15, verbose = false), save_sol_every_step = 0, detect_bifurcation = 0);
    verbosity = 1,
    normC = norminf,
    # callback_newton = (x, f, J, res, iteration, itlinear, options; kwargs...) ->(true)
    )

plot(br, legend=false, linewidth=1, vars = (:param, :n2))
