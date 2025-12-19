using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using BifurcationKit, LinearAlgebra, Profile, Random, Test
const BK = BifurcationKit

# --- 1. Define Problem (Stuart-Landau) ---
function Fsl(X, p)
    (;r, μ, ν, c3, c5) = p
    u = X[1]; v = X[2]
    ua = u^2 + v^2
    [r * u - ν * v - ua * (c3 * u - μ * v) - c5 * ua^2 * u,
     r * v + ν * u - ua * (c3 * v + μ * u) - c5 * ua^2 * v]
end

# Hopf is at r=0. We start at -0.5 to cross it.
par_sl = (r = -0.5, μ = 0., ν = 1.0, c3 = 1.0, c5 = 0.0)
u0 = [0.1, 0.1]
prob = BifurcationProblem(Fsl, u0, par_sl, (@optic _.r))

opts = ContinuationPar(ds = 0.05, max_steps = 100, p_min = -1., p_max = 1., detect_bifurcation = 3, n_inversion = 6)

println("\n=== 1. Generic Equilibrium Continuation ===")
# This finds the Hopf point
br = continuation(prob, PALC(), opts; verbosity = 0)

if length(br.specialpoint) == 0
    error("No bifurcation point found! Cannot continue code.")
end

# --- 2. Periodic Orbit Collocation ---
println("\n=== 2. Periodic Orbit Collocation (N=20, m=4) ===")
# Setup Periodic Orbit Collocation
prob_po = PeriodicOrbitOCollProblem(20, 4; prob_vf = prob, N = 2)
opts_po = ContinuationPar(opts, max_steps = 20, ds = 0.01)

println("-> Warmup / First Run...")
# Force compilation if not fully done
try
    br_po = continuation(br, 1, opts_po, prob_po; verbosity = 0)
catch e
    println("Warmup failed (normal if not exactly on branch): ", e)
end

println("-> Profiling Runtime (Second Run)...")
GC.gc()
@time br_po_2 = continuation(br, 1, opts_po, prob_po; verbosity = 0)


# --- 3. Periodic Orbit Trapeze ---
println("\n=== 3. Periodic Orbit Trapeze (M=40) ===")
prob_trap = PeriodicOrbitTrapProblem(prob, 40, 2)
opts_trap = ContinuationPar(opts, max_steps = 20, ds = 0.01)

println("-> Warmup / First Run...")
try
    br_trap = continuation(br, 1, opts_trap, prob_trap; verbosity = 0)
catch e
    println("Warmup failed: ", e)
end

println("-> Profiling Runtime (Second Run)...")
GC.gc()
@time br_trap_2 = continuation(br, 1, opts_trap, prob_trap; verbosity = 0)


# --- 4. Low Level Profiling (Collocation Residual/Jacobian) ---
# println("\n=== 4. Low Level Profiling (Collocation Residual/Jacobian) ===")
# # Create a random guess of correct size
# u_test = rand(length(prob_po))
# res = similar(u_test)
# par_test = par_sl

# println("-> Testing residual! (1000 calls)...")
# # Warmup
# BK.residual!(prob_po, res, u_test, par_test)
# GC.gc()
# @time for _ in 1:1000
#     BK.residual!(prob_po, res, u_test, par_test)
# end

# println("-> Testing jacobian (100 calls)...")
# # Warmup
# J = BK.jacobian(prob_po, u_test, par_test)
# GC.gc()
# @time for _ in 1:100
#     BK.jacobian(prob_po, u_test, par_test)
# end

println("\n=== Benchmarking Complete ===")
