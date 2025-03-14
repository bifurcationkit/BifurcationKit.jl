# using Revise
using Test, BifurcationKit
const BK = BifurcationKit

prob = BifurcationProblem((x,p)->[x[1]^2+p[1],sum(x)], rand(2), rand(2), (@optic _[1]), R11=(x,p,dx,dp) -> dx .* dp)

BK.residual(prob, prob.u0, prob.params)
BK.jacobian(prob, prob.u0, prob.params)
prob.VF.jet.R11(prob.u0, prob.params, 1,1)
BK.R11(prob.VF.jet, prob.u0, prob.params, 1,1)
BK.R11(prob, prob.u0, prob.params, 1,1)
prob = BifurcationProblem((o,x,p)-> o .= x, rand(2), rand(2), (@optic _[1]); inplace = true)
######################################################################
# test show of wraped problem
par = (a=1., b=0.)
prob = BifurcationProblem((x,p)->[x[1]^2+p[1],p.a + sum(x)], rand(2), par, (@optic _.a))
BK.WrapPOTrap(prob, prob, prob.u0, prob.params, prob.lens, BK.plot_default, BK.plot_default) |> show
BK.PDMAProblem((prob_vf = prob,), prob, prob.u0, prob.params, prob.lens, BK.plot_default, BK.plot_default) |> show

BK._getvectortype(prob)

# test type instability
@code_warntype BK.jacobian(prob, prob.u0, prob.params)
@code_warntype BK.jacobian!(prob, zeros(2,2), prob.u0, prob.params)
# @inferred BK.jacobian(prob, prob.u0, prob.params)
# @inferred BK.jacobian!(prob, zeros(2,2), prob.u0, prob.params)

BK.apply(BK.jacobian(prob, rand(2), set(prob.params, prob.lens, 0.)), rand(2))
BK.dF(  prob, prob.u0, prob.params, rand(2))
BK.d2F( prob, prob.u0, prob.params, rand(2), rand(2))
BK.d3F( prob, prob.u0, prob.params, rand(2), rand(2), rand(2))
BK.d3Fc(prob, prob.u0, prob.params, rand(2), rand(2), rand(2))

# compute R11
import DifferentiationInterface as DI
_dx = rand(2)
ForwardDiff.derivative(z-> BK.apply(BK.jacobian(prob, prob.u0, set(prob.params, prob.lens, z)), _dx), 0.)
DI.derivative(z-> BK.apply(BK.jacobian(prob, prob.u0, set(prob.params, prob.lens, z)), _dx), DI.AutoForwardDiff(), 0.)

# strange, does not work
# DI.derivative(z -> BK.dF(prob, prob.u0, set(prob.params, prob.lens, z), _dx), DI.AutoForwardDiff(), 0.)



BK.re_make(prob, J = (x,p)->zeros(2,2), Jᵗ = (x,p)->zeros(2,2), d2F=(x,p,dx1,dx2)->x, d3F=(x,p,dx1,dx2,dx3)->x)
######################################################################
# test finite differences
BK.finite_differences(identity, zeros(2))
BK.finite_differences!((o,x)->o .= x, zeros(2, 2), zeros(2))
