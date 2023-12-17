# using Revise
using Test, BifurcationKit
const BK = BifurcationKit

prob = BifurcationProblem((x,p)->[x[1]^2+p[1],sum(x)], rand(2), rand(2), (@lens _[1]))

BK.residual(prob, prob.u0, prob.params)
BK.jacobian(prob, prob.u0, prob.params)

prob = BifurcationProblem((o,x,p,t=0)-> o.=x, rand(2), rand(2), (@lens _[1]); inplace = true)
######################################################################
# test show of wraped problem
prob = BifurcationProblem((x,p)->[x[1]^2+p[1],sum(x)], rand(2), rand(2), (@lens _[1]))
BK.WrapPOTrap(prob, prob, prob.u0, prob.params, prob.lens, BK.plot_default, BK.plot_default) |> show
BK.PDMAProblem((prob_vf = prob,), prob, prob.u0, prob.params, prob.lens, BK.plot_default, BK.plot_default) |> show    

BK.getvectortype(prob)

BK.d3F(prob, rand(2), rand(2), rand(2), rand(2), rand(2))
BK.d3Fc(prob, rand(2), rand(2), rand(2), rand(2), rand(2))

BK.has_adjoint_MF(prob)

BK.plot_default(0,0)
BK.plot_default(0,0,0)

BK.re_make(prob, J = (x,p)->zeros(2,2), Jᵗ = (x,p)->zeros(2,2),d2F=(x,p,dx1,dx2)->x,d3F=(x,p,dx1,dx2,dx3)->x)
######################################################################
# test finite differences
BK.finite_differences(identity, zeros(2))
BK.finite_differences((x,p)->x, zeros(2), nothing)
