# using Revise
using Test, BifurcationKit
const BK = BifurcationKit

let
    prob_ana = BifurcationProblem((x,p)->[x[1]^2*p[1],sum(x)], rand(2), rand(2), (@optic _[1]), 
        R11 = (x,p,dx) -> [dx[1], 0], 
        R01 = (x,p) -> [x[1]^2,0], 
        R02 = (x,p) -> [0,0])
    BK.residual(prob_ana, prob_ana.u0, prob_ana.params)
    BK.jacobian(prob_ana, prob_ana.u0, prob_ana.params)
    @test BK.has_R01(prob_ana) == true
    BK.R01!(prob_ana, prob_ana.u0, prob_ana.u0, prob_ana.params, 0.1)
    BK.R01(prob_ana, prob_ana.u0, prob_ana.params)
    BK.R02(prob_ana, prob_ana.u0, prob_ana.params)
    BK.R11(prob_ana, prob_ana.u0, prob_ana.params, prob_ana.u0)

    for rij in (BK.AutoDiff(), BK.FiniteDifferences())
        prob = BifurcationProblem((x,p)->[x[1]^2*p[1],sum(x)], rand(2), rand(2), (@optic _[1]); R01 = rij, R02 = rij)
        BK.R01!(prob, prob.u0, prob.u0, prob.params, 0.1)
        BK.R01(prob, prob.u0, prob.params)
        BK.R02(prob, prob.u0, prob.params)
        BK.R11(prob, prob.u0, prob.params, prob.u0)
    end
    ######################################################################
    # test show of wrapped problem
    prob = BifurcationProblem((x,p)->[x[1]^2+p[1],sum(x)], rand(2), rand(2), (@optic _[1]))
    BK.PeriodicOrbitFunctionalTrap(prob, prob, prob.u0, BK.plot_default, BK.plot_default) |> show
    BK.PDMAProblem((prob_vf = prob,), prob, prob.u0, prob.lens, BK.plot_default, BK.plot_default) |> show    

    BK._getvectortype(prob)

    BK.d3F(prob, rand(2), rand(2), rand(2), rand(2), rand(2))

    BK.has_adjoint_MF(prob)

    BK.plot_default(0,0)
    BK.plot_default(0,0,0)

    BK.re_make(prob, J = (x,p)->zeros(2,2), Jᵗ = (x,p)->zeros(2,2), d2F=(x,p,dx1,dx2)->x, d3F=(x,p,dx1,dx2,dx3)->x)
    ######################################################################
    # test finite differences
    BK.finite_differences(identity, zeros(2))
    BK.finite_differences!((o, x) -> o .= x, zeros(2, 2), zeros(2))
end