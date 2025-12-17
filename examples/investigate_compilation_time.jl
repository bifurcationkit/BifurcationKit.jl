#!/usr/bin/env julia
#
# issue: https://github.com/bifurcationkit/BifurcationKit.jl/issues/210
#


using BifurcationKit
using Revise

function Fsl(u, p)
    (;r, μ, ν, c3, c5) = p
    u1, u2 = u
    ua = u1^2 + u2^2
    f1 = r * u1 - ν * u2 + ua * (c3 * u1 - μ * u2) + c5 * ua^2 * u1
    f2 = r * u2 + ν * u1 + ua * (c3 * u2 + μ * u1) + c5 * ua^2 * u2
    return [f1, f2]
end

par_sl = (r = -0.5, μ = 0., ν = 1.0, c3 = 0.1, c5 = -0.01)

println("L$(@__LINE__() + 1) BifurcationProblem ")
prob = @time BifurcationProblem(Fsl, [0., 0], par_sl, (@optic _.r))

println("L$(@__LINE__() + 1) ContinuationPar ")
opts = @time ContinuationPar(p_min = -1.)

println("L$(@__LINE__() + 1) continuation ")
br = @time continuation(prob, PALC(), opts)

# branch of periodic orbits
println("L$(@__LINE__() + 1) continuation ")
br_po = @time continuation(br, 1, opts,
        PeriodicOrbitOCollProblem(20, 4)
        )

# computation of folds of periodic orbits
println("L$(@__LINE__() + 1) ContinuationPar ")
opts_pocoll_fold = @time ContinuationPar(br_po.contparams, max_steps = 10, p_max=1.2)

println("L$(@__LINE__() + 1) continuation ")
fold_po_coll = @time continuation(deepcopy(br_po), 1, (@optic _.c5), opts_pocoll_fold;
        detect_codim2_bifurcation = 0,
        jacobian_ma = :minaug,
        bdlinsolver = BorderingBLS(solver = DefaultLS(), check_precision = false),
        )
