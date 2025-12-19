using Revise

using BifurcationKit, LinearAlgebra, Plots


# ==============================================================================
# Bratu Problem BVP Example
# ==============================================================================

# 1. Define the vector field (first-order form)
# u'' + p₁ * exp(u) = 0  =>  u₁' = u₂, u₂' = -p₁ * exp(u₁)
function Fbratu(x, p)
    return [x[2], -p.p₁ * exp(x[1])]
end

# 2. Define boundary conditions: x₁(0) = 0, x₁(1) = 0
function gbratu(u0, uT, p)
    return [u0[1], uT[1]]
end

# 3. Create BVP Model
# State dimension is 2 (u, u')
# Fixed interval [0, 1] => phase condition fixes T=1.0
model = BVPModel(Fbratu, gbratu; n=2, phase = (u, p, T) -> T - 1.0)

# 4. Discretize using Trapezoid method
# Using 201 points for better accuracy
disc = Collocation(Ntst=30, m=4)
bvp = discretize(model, disc)

# 5. Set up parameters and initial guess
# At p₁ = 0, the solution is u(t) = 0, u'(t) = 0
params = (p₁ = 0.0,)
t_vals = LinRange(0, 1, 201)
x0 = zeros(2 * (1 + disc.m * disc.Ntst) + 1) 
x0[end] = 1.0 # Interval length T = 1.0

# 6. Create BVPBifProblem
# We record max(u) to plot the bifurcation diagram
prob = BVPBifProblem(bvp, x0, params, (@optic _.p₁);
    record_from_solution = (x, p; k...) -> begin
        u = BifurcationKit.get_time_slices(x[1:end-1], 2, disc.m, disc.Ntst)
        return (max_u = maximum(u[1, :]),)
    end,
    plot_solution = (x, p; kwargs...) -> begin
        u = BifurcationKit.get_time_slices(x[1:end-1], 2, disc.m, disc.Ntst)
        plot!(u[1, :]; ylabel="u(t)", title="Bratu Solution (p₁=)", kwargs...)
    end
)

# 7. Setup Continuation Parameters
optn = NewtonPar(tol = 1e-10, verbose=false)
optc = ContinuationPar(
    p_min = 0.0, 
    p_max = 5.0, 
    dsmax = 0.1, 
    ds = 0.01, 
    detect_bifurcation = 2, 
    detect_fold = true,
    newton_options = optn,
    max_steps = 200,
    nev = 20,
    n_inversion = 6
)

# 8. Perform initial continuation
println("\nComputing primary branch for Bratu BVP (Collocation)...")
br = continuation(prob, PALC(), optc; 
    plot = true,
    verbosity = 3,
    normC=norminf,
)
show(br)

# 9. Branch switching logic
idx_bp = findfirst(x -> x.type == :bp, br.specialpoint)

if !isnothing(idx_bp)
    println("\n" * "="^60)
    println("BRANCH SWITCHING")
    println("Found Branch Point at step $(br.specialpoint[idx_bp].step)")
    println("Parameter p₁ ≈ $(round(br.specialpoint[idx_bp].param, digits=4))")
    println("="^60)

    # We use a slightly adjusted ContinuationPar for the second branch
    optc_sec = @set optc.ds = 0.005
    optc_sec = @set optc_sec.max_steps = 100
    optc_sec = @set optc_sec.p_min = -1.0 # Allow exploration

    println("\nSwitching to secondary branch...")
    try
        br_sec = continuation(br, idx_bp, optc_sec;
            ampfactor = 0.2,       # Increased perturbation
            verbosity = 1,
            plot = false
        )

        println("\nSecondary branch found with $(length(br_sec)) points.")

        # 10. Plot results
        println("\nPlotting branches...")
        p1 = plot(br, xaxis=:p, yaxis=:max_u, label="Primary", lw=2)
        if length(br_sec) > 1
            plot!(p1, br_sec, xaxis=:p, yaxis=:max_u, label="Secondary", lw=2, linestyle=:dash)
        end
        title!(p1, "Bratu Bifurcation Diagram")
        xlabel!(p1, "p₁")
        ylabel!(p1, "max(u)")
        
        # Add labels for special points if any
        savefig(p1, "bvp_bratu_switching.png")
        println("Plot saved to bvp_bratu_switching.png")
    catch e
        println("\nBranch switching failed or stopped early: ", e)
        Base.display_error(e, catch_backtrace())
        p1 = plot(br, xaxis=:p, yaxis=:max_u, label="Primary")
        savefig(p1, "bvp_bratu_only_primary.png")
    end
else
    println("\nNo branch point (bp) found for switching.")
    p1 = plot(br, xaxis=:p, yaxis=:max_u)
    savefig(p1, "bvp_bratu_primary.png")
end

println("\nExample complete!")
