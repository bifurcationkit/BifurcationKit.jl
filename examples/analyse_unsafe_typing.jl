#!/usr/bin/env julia
#
# unsafe typing detection
#
# possible resolution: use DifferentiationInterface (interface to ForwardDiff)
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BifurcationKit
using Revise
using InteractiveUtils

## problem 1

function Fsl(u, p)
    (;r, μ, ν, c3, c5) = p
    u1, u2 = u
    ua = u1^2 + u2^2
    f1 = r * u1 - ν * u2 + ua * (c3 * u1 - μ * u2) + c5 * ua^2 * u1
    f2 = r * u2 + ν * u1 + ua * (c3 * u2 + μ * u1) + c5 * ua^2 * u2
    return [f1, f2]
end

par_sl = (r = -0.5, μ = 0., ν = 1.0, c3 = 0.1, c5 = -0.01)
println("L$(@__LINE__() + 1) BifurcationProblem with FSl")
prob = @time BifurcationProblem(Fsl, [0., 0], par_sl, (@optic _.r))


# everything is blue -> OK
@time BifurcationKit.residual(prob, prob.u0, prob.params)
println("\n=========== should be OK")
@code_warntype BifurcationKit.residual(prob, prob.u0, prob.params)

# Any type detected -> KO
@time BifurcationKit.jacobian(prob, prob.u0, prob.params)
println("\n=========== should detect a problem")
@code_warntype BifurcationKit.jacobian(prob, prob.u0, prob.params)


## problem 2

function TMvf!(dz, z, p, t = 0)
    (;J, α, E0, τ, τD, τF, U0) = p
    E, x, u = z
    SS0 = J * u * x * E + E0
    SS1 = α * log(1 + exp(SS0 / α))
    dz[1] = (-E + SS1) / τ
    dz[2] = (1 - x) / τD - u * x * E
    dz[3] = (U0 - u) / τF +  U0 * (1 - u) * E
    dz
end

par_tm = (α = 1.5, τ = 0.013, J = 3.07, E0 = -2.0, τD = 0.200, U0 = 0.3, τF = 1.5, τS = 0.007)
z0 = [0.238616, 0.982747, 0.367876 ]
println("L$(@__LINE__() + 1) BifurcationProblem with TMvf!")
prob = @time BifurcationProblem(TMvf!, z0, par_tm, (@optic _.E0); record_from_solution = (x, p; k...) -> (E = x[1], x = x[2], u = x[3]))

# Any type in the way -> KO
println("\n=========== should detect a problem")
@code_warntype BifurcationKit.jacobian(prob, z0, par_tm)
