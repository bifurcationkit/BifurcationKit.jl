using Revise
using Test, ForwardDiff, Parameters
using Plots
# using GLMakie; Makie.inline!(true)
using BifurcationKit, Test
const BK = BifurcationKit
####################################################################################################
function COm!(du, u, p, t = 0)
    @unpack q1,q2,q3,q4,q5,q6,k = p
    x, y, s = u
    z = 1-x-y-s
    du[1] = 2q1 * z^2 - 2q5 * x^2 - q3 * x * y
    du[2] = q2 * z - q6 * y - q3 * x * y
    du[3] = q4 * (z - k * s)
    du
end

par_com = (q1 = 2.5, q2 = 2.0, q3 = 10., q4 = 0.0675, q5 = 1., q6 = 0.1, k = 0.4)

z0 = [0.001137, 0.891483, 0.062345]

prob = BifurcationProblem(COm!, z0, par_com, (@lens _.q2); record_from_solution = (x, p) -> (x = x[1], y = x[2], s = x[3]))

opts_br = ContinuationPar(dsmax = 0.05, p_min = 0.5, p_max = 2.0, n_inversion = 6, detect_bifurcation = 3, max_bisection_steps = 25, nev = 3)
br = @time continuation(prob, PALC(), opts_br;
    # plot = false, verbosity = 0,
    normC = norminf,
    bothside = true)
show(br)

plot(br, plotfold=false, markersize=4, legend=:topright, ylims=(0,0.16))
####################################################################################################
@set! opts_br.newton_options.verbose = false
@set! opts_br.newton_options.max_iterations = 10
opts_br = @set opts_br.newton_options.tol = 1e-12

sn = newton(br, 3; options = opts_br.newton_options, bdlinsolver = MatrixBLS())

hp = newton(br, 2; options = NewtonPar( opts_br.newton_options; max_iterations = 10),start_with_eigen=true)

hpnf = get_normal_form(br, 2)

sn_codim2 = continuation(br, 3, (@lens _.k), ContinuationPar(opts_br, p_max = 3.2, p_min = 0., detect_bifurcation = 0, dsmin=1e-5, ds = -0.001, dsmax = 0.05, n_inversion = 6, detect_event = 2, detect_fold = false) ; plot = true,
    verbosity = 3,
    normC = norminf,
    update_minaug_every_step = 1,
    start_with_eigen = true,
    # record_from_solution = (u,p; kw...) -> (x = u.u[1] ),
    bothside=true,
    bdlinsolver = MatrixBLS()
    )

using Test
@test sn_codim2.specialpoint[2].printsol.k  ≈ 0.971397 rtol = 1e-4
@test sn_codim2.specialpoint[2].printsol.q2 ≈ 1.417628 rtol = 1e-4
@test sn_codim2.specialpoint[4].printsol.k  ≈ 0.722339 rtol = 1e-4
@test sn_codim2.specialpoint[4].printsol.q2 ≈ 1.161199 rtol = 1e-4

BK.plot(sn_codim2)#, real.(sn_codim2.BT), ylims = (-1,1), xlims=(0,2))

BK.plot(sn_codim2, vars=(:q2, :x), branchlabel = "Fold", plotstability = false);plot!(br,xlims=(0.8,1.8))

hp_codim2 = continuation((@set br.alg.tangent = Bordered()), 2, (@lens _.k), ContinuationPar(opts_br, p_min = 0., p_max = 2.8, detect_bifurcation = 0, ds = -0.0001, dsmax = 0.08, dsmin = 1e-4, n_inversion = 6, detect_event = 2, detect_loop = true, max_steps = 50, detect_fold=false) ; plot = true,
    verbosity = 0,
    normC = norminf,
    detect_codim2_bifurcation = 2,
    update_minaug_every_step = 1,
    start_with_eigen = true,
    # record_from_solution = (u,p; kw...) -> (x = u.u[1] ),
    bothside = true,
    bdlinsolver = MatrixBLS())

@test hp_codim2.branch[6].l1 |> real        ≈ 33.15920 rtol = 1e-1
@test hp_codim2.specialpoint[3].printsol.k  ≈ 0.305879 rtol = 1e-3
@test hp_codim2.specialpoint[3].printsol.q2 ≈ 0.924255 rtol = 1e-3
@test hp_codim2.specialpoint[4].printsol.k  ≈ 0.23248736 rtol = 1e-4
@test hp_codim2.specialpoint[4].printsol.q2 ≈ 0.8913189828755895 rtol = 1e-4

BK.plot(sn_codim2, vars=(:q2, :x), branchlabel = "Fold", plotcirclesbif = true)
plot!(hp_codim2, vars=(:q2, :x), branchlabel = "Hopf",plotcirclesbif = true)
plot!(br,xlims=(0.6,1.5))

plot(sn_codim2, vars=(:k, :q2), branchlabel = "Fold")
plot!(hp_codim2, vars=(:k, :q2), branchlabel = "Hopf",)

plot(hp_codim2, vars=(:q2, :x), branchlabel = "Hopf")
####################################################################################################
