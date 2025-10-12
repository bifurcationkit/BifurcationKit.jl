# Test for higher-order Bautin normal form with jet derivatives from Symbolics.jl
# This test uses the getJet function to compute derivatives symbolically

using Test
using BifurcationKit
using LinearAlgebra
using Symbolics
const BK = BifurcationKit

# Function to compute jet derivatives symbolically
function getJet(system, N)
    @variables x[1:N] α[1:2]
    @variables v₁[1:N] v₂[1:N] v₃[1:N] v₄[1:N] v₅[1:N] v₆[1:N] v₇[1:N] p₁[1:2] p₂[1:2] p₃[1:2]

    x = Symbolics.scalarize(x)
    α = Symbolics.scalarize(α)
    v₁ = Symbolics.scalarize(v₁)
    v₂ = Symbolics.scalarize(v₂)
    v₃ = Symbolics.scalarize(v₃)
    v₄ = Symbolics.scalarize(v₄)
    v₅ = Symbolics.scalarize(v₅)
    v₆ = Symbolics.scalarize(v₆)
    v₇ = Symbolics.scalarize(v₇)
    p₁ = Symbolics.scalarize(p₁)
    p₂ = Symbolics.scalarize(p₂)
    p₃ = Symbolics.scalarize(p₃)

    jac = Symbolics.jacobian

    R10 = eval(build_function(jac(system(x, α), x), x, α, expression=Val{false})[1])
    R20 = eval(build_function(jac(jac(system(x, α), x) * v₁, x) * v₂, x, α, v₁, v₂, expression=Val{false})[1])
    R30 = eval(build_function(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, x) * v₃, x, α, v₁, v₂, v₃, expression=Val{false})[1])
    R40 = eval(build_function(jac(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, x) * v₃, x) * v₄, x, α, v₁, v₂, v₃, v₄, expression=Val{false})[1])
    R50 = eval(build_function(jac(jac(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, x) * v₃, x) * v₄, x) * v₅ , x, α, v₁, v₂, v₃, v₄, v₅, expression=Val{false})[1])
    R60 = eval(build_function(jac(jac(jac(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, x) * v₃, x) * v₄, x) * v₅, x) * v₆ , x, α, v₁, v₂, v₃, v₄, v₅, v₆, expression=Val{false})[1])
    R70 = eval(build_function(jac(jac(jac(jac(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, x) * v₃, x) * v₄, x) * v₅, x) * v₆, x) * v₇ , x, α, v₁, v₂, v₃, v₄, v₅, v₆, v₇, expression=Val{false})[1])

    R01 = eval(build_function(jac(system(x, α), α), x, α, expression=Val{false})[1])
    R11 = eval(build_function(jac(jac(system(x, α), x) * v₁, α) * p₁, x, α, v₁, p₁, expression=Val{false})[1])
    R21 = eval(build_function(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, α) * p₁, x, α, v₁, v₂, p₁, expression=Val{false})[1])
    R31 = eval(build_function(jac(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, x) * v₃, α) * p₁, x, α, v₁, v₂, v₃, p₁, expression=Val{false})[1])
    R41 = eval(build_function(jac(jac(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, x) * v₃, x) * v₄, α) * p₁, x, α, v₁, v₂, v₃, v₄, p₁, expression=Val{false})[1])
    R51 = eval(build_function(jac(jac(jac(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, x) * v₃, x) * v₄, x) * v₅, α) * p₁, x, α, v₁, v₂, v₃, v₄, v₅, p₁, expression=Val{false})[1])

    R02 = eval(build_function(jac(jac(system(x, α), α) * p₁, α) * p₂ , x, α, p₁, p₂, expression=Val{false})[1])
    R12 = eval(build_function(jac(jac(jac(system(x, α), x) * v₁, α) * p₁, α) * p₂ , x, α, v₁, p₁, p₂, expression=Val{false})[1])
    R22 = eval(build_function(jac(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, α) * p₁, α) * p₂ , x, α, v₁, v₂, p₁, p₂, expression=Val{false})[1])
    R32 = eval(build_function(jac(jac(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, x) * v₃, α) * p₁, α) * p₂ , x, α, v₁, v₂, v₃, p₁, p₂, expression=Val{false})[1])

    R03 = eval(build_function(jac(jac(jac(system(x, α), α) * p₁, α) * p₂, α) * p₃ , x, α, p₁, p₂, p₃, expression=Val{false})[1])
    R13 = eval(build_function(jac(jac(jac(jac(system(x, α), x) * v₁, α) * p₁, α) * p₂, α) * p₃ , x, α, v₁, p₁, p₂, p₃, expression=Val{false})[1])
    R23 = eval(build_function(jac(jac(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, α) * p₁, α) * p₂, α) * p₃ , x, α, v₁, v₂, p₁, p₂, p₃, expression=Val{false})[1])
    R33 = eval(build_function(jac(jac(jac(jac(jac(jac(system(x, α), x) * v₁, x) * v₂, x) * v₃, α) * p₁, α) * p₂, α) * p₃ , x, α, v₁, v₂, v₃, p₁, p₂, p₃, expression=Val{false})[1])

    (; R10, R20, R30, R40, R50, R60, R70, R01, R11, R21, R31, R41, R51, R02, R12, R22, R32, R03, R13, R23, R33)
end

# Stuart-Landau equation with additional higher-order terms
# We'll use parameters α = [r, c3] for the jet
function Fsl2_symbolic(x, α)
    u1, u2 = x
    r, c3 = α
    # Fixed parameters (not part of the jet parameter vector)
    μ = 0.0
    ν = 1.0
    c5 = 0.3
    c7 = 1.2

    ua = u1^2 + u2^2

    F1 = r * u1 - ν * u2 + ua * (c3 * u1 - μ * u2) + c5 * ua^2 * u1 + c7 * ua^3 * u1
    F2 = r * u2 + ν * u1 + ua * (c3 * u2 + μ * u1) + c5 * ua^2 * u2 + c7 * ua^3 * u2

    return [F1, F2]
end

# Regular in-place version for BifurcationKit
function Fsl2!(f, u, p, t = 0)
    (;r, μ, ν, c3, c5, c7) = p
    u1, u2 = u
    ua = u1^2 + u2^2
    f[1] = r * u1 - ν * u2 + ua * (c3 * u1 - μ * u2) + c5 * ua^2 * u1 + c7 * ua^3 * u1
    f[2] = r * u2 + ν * u1 + ua * (c3 * u2 + μ * u1) + c5 * ua^2 * u2 + c7 * ua^3 * u2
    return f
end

# Generate jet derivatives (THIS WILL TAKE SEVERAL MINUTES!)
println("="^80)
println("Generating jet derivatives symbolically...")
println("="^80)

jet = getJet(Fsl2_symbolic, 2)

println("\n" * "="^80)
println("Jet generation complete!")
println("="^80)

# Now run the actual test
println("\nRunning Bautin normal form test with higher-order jet...")

par_sl = (r = -0.5, μ = 0., ν = 1.0, c3 = 0.1, c5 = 0.3, c7 = 1.2)

# Create problem with full jet
prob_with_jet = BK.BifurcationProblem(
    Fsl2!, [0.01, 0.01], par_sl, (@optic _.r);
    jet...
)

opt_newton = BK.NewtonPar(tol = 1e-9, max_iterations = 40, verbose = false)
opts_br = BK.ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds = 0.01, p_max = 0.5, p_min = -0.5,
                          detect_bifurcation = 3, nev = 2, newton_options = opt_newton,
                          max_steps = 80, n_inversion = 8, save_sol_every_step = 1)

@reset opts_br.newton_options.verbose = false
@reset opts_br.newton_options.tol = 1e-12
opts_br = BK.setproperties(opts_br; n_inversion = 10, max_bisection_steps = 25)

println("Running continuation...")
br = BK.continuation(prob_with_jet, BK.PALC(), opts_br)

println("Running codim-2 continuation for Bautin point...")
hopf_codim2 = BK.continuation(br, 1, (@optic _.c3),
    BK.ContinuationPar(opts_br, detect_bifurcation = 0, save_sol_every_step = 1,
                    max_steps = 15, p_min = -2., p_max = 2., ds = -0.001);
    detect_codim2_bifurcation = 2,
    start_with_eigen = true,
    update_minaug_every_step = 1,
    bdlinsolver = BK.MatrixBLS(),
)

@test hopf_codim2.specialpoint[1].type == :gh

println("\nComputing Bautin normal form...")
bautin_ho = BK.get_normal_form(hopf_codim2, 1; nev = 2)
show(bautin_ho)

println("\n" * "="^80)
println("TEST RESULTS:")
println("="^80)

# Test that higher-order path was taken
println("Number of normal form coefficients: $(length(bautin_ho.nf))")
@test length(bautin_ho.nf) > 14
println("✓ Higher-order path was used (length > 14)")

# Test l2 coefficient matches expected value
println("\nl2 coefficient: $(bautin_ho.nf.l2)")
println("Expected: $(par_sl.c5 * 4)")
@test bautin_ho.nf.l2 ≈ par_sl.c5 * 4 atol = 1e-6
println("✓ l2 matches expected value")

# Test l3 is computed
@test hasfield(typeof(bautin_ho.nf), :l3)
println("\nl3 coefficient: $(bautin_ho.nf.l3)")
println("✓ l3 was computed (higher-order jet working!)")

# Test l3 coefficient matches expected value
l3_expected = 8 * par_sl.c7
println("\nTheoretical validation:")
println("  Expected l3 = 8·c7 = $(l3_expected)")
println("  Computed l3 = $(bautin_ho.nf.l3)")
println("  Difference: $(abs(bautin_ho.nf.l3 - l3_expected))")
@test bautin_ho.nf.l3 ≈ l3_expected atol = 1e-6
println("✓ l3 matches theoretical formula l3 = 8·c7")

println("\n" * "="^80)
println("Testing higher-order predictor...")
println("="^80)

# Test that the predictor function works with higher-order coefficients
ϵ_test = 0.01
pred_result = BK.predictor(bautin_ho, Val(:FoldPeriodicOrbitCont), ϵ_test)

println("Predictor successfully generated initial guess for fold of periodic orbits")
println("  ε = $ϵ_test")
println("  Predicted ω = $(pred_result.ω)")
println("  Predicted parameters = $(pred_result.params)")

# Verify the predictor returned the expected structure
@test hasfield(typeof(pred_result), :orbit)
@test hasfield(typeof(pred_result), :ω)
@test hasfield(typeof(pred_result), :params)
println("✓ Predictor is functional")

# Now verify that the predictor is actually using the higher-order coefficients
# We do this by creating a modified normal form with c₃ = 0 and comparing predictions
println("\n" * "="^80)
println("Verifying predictor uses higher-order terms...")
println("="^80)

# Create a modified Bautin object with c₃ set to zero (removes l3 contribution)
# We need to modify the nf field while keeping everything else the same
nf_with_c3 = bautin_ho.nf
nf_without_c3 = merge(nf_with_c3, (c₃ = 0.0 + 0.0im, l3 = 0.0))

bautin_without_c3 = BK.Bautin(
    bautin_ho.x0,
    bautin_ho.params,
    bautin_ho.lens,
    bautin_ho.ζ,
    bautin_ho.ζ★,
    nf_without_c3,
    bautin_ho.type
)

# Call the predictor with both versions
println("Calling predictor with full higher-order normal form (with c₃)...")
pred_with_c3 = BK.predictor(bautin_ho, Val(:FoldPeriodicOrbitCont), ϵ_test)

println("Calling predictor with modified normal form (c₃ = 0)...")
pred_without_c3 = BK.predictor(bautin_without_c3, Val(:FoldPeriodicOrbitCont), ϵ_test)

# Compare the predictions
println("\nPredictions WITH c₃:")
println("  ω = $(pred_with_c3.ω)")
println("  params = $(pred_with_c3.params)")

println("\nPredictions WITHOUT c₃:")
println("  ω = $(pred_without_c3.ω)")
println("  params = $(pred_without_c3.params)")

# Compute differences
Δω = abs(pred_with_c3.ω - pred_without_c3.ω)
Δparams = norm(pred_with_c3.params .- pred_without_c3.params)

println("\nDifferences (proof that c₃ is used!):")
println("  Δω = $Δω")
println("  ‖Δparams‖ = $Δparams")

# Verify differences are significant (not just roundoff)
@test Δparams > 1e-10
println("✓ The predictor does use c₃!")
println("✓ Higher-order terms significantly affect parameter predictions!")

println("\n" * "="^80)
println("All tests passed! Higher-order Bautin normal form is working correctly.")
println("="^80)
