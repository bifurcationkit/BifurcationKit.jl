# discretize - Create a discretized BVP from model + discretizer
#
# This is the main user-facing function to combine a mathematical
# BVP model with a numerical discretization method.

using DocStringExtensions

"""
$(TYPEDSIGNATURES)

Discretize a BVP model using the specified method.

## Arguments
- `model::BVPModel`: Mathematical BVP formulation
- `disc::AbstractDiscretizer`: Discretization method

## Returns
- `DiscretizedBVP`: A discretized problem ready for Newton/continuation

## Example
```julia
# Define model
F(u, p) = [u[2], -p.ω² * u[1]]
model = PeriodicOrbitModel(F; n=2)

# Choose discretization
disc = Trap(M=100)

# Create discretized problem
bvp = discretize(model, disc)

# Use with continuation
prob = BifurcationProblem(bvp, x0, params, (@optic _.ω))
```
"""
function discretize end

# ============================================================================
# Shooting
# ============================================================================

function discretize(model::BVPModel, disc::Shooting)
    n = state_dimension(model)
    @assert n > 0 "State dimension must be specified in the model"
    
    cache = create_cache(disc, n)
    return DiscretizedBVP(model, disc, cache)
end

function create_cache(disc::Shooting, n::Int)
    M = disc.M
    # Workspace for shooting
    return (
        u_work = zeros(n),
        Φ_work = zeros(n, n),
        F_work = zeros(n),
    )
end

# ============================================================================
# Trapezoid
# ============================================================================

import BifurcationKit: PeriodicOrbitTrapProblem, PeriodicOrbitOCollProblem, BifurcationProblem, TimeMesh
import PreallocationTools: DiffCache, get_tmp

function discretize(model::BVPModel, disc::Trap)
    n = state_dimension(model)
    @assert n > 0 "State dimension must be specified in the model"
    
    M = disc.M
    
    # Create a BifurcationProblem wrapper for the vector field
    # This is needed by PeriodicOrbitTrapProblem
    # Note: We use a dummy params/lens since these are handled by BVPBifProblem
    # record_from_solution is required to avoid errors
    prob_vf = BifurcationProblem(
        (u, p) -> model.F(u, p),    # Vector field
        zeros(n),                     # Dummy initial guess
        (dummy = 0.0,),               # Dummy params
        (@optic _.dummy);             # Dummy lens
        inplace = false,              # Must match the signature above
        record_from_solution = (x, p; k...) -> nothing
    )
    
    # Create PeriodicOrbitTrapProblem for efficient residual computation
    # Phase constraint vectors (will be overwritten later)
    ϕ = zeros(n * M)
    xπ = zeros(n * M)
    
    po_trap = PeriodicOrbitTrapProblem(;
        prob_vf = prob_vf,
        ϕ = ϕ,
        xπ = xπ,
        M = M,
        mesh = TimeMesh(M),
        N = n,
    )
    
    cache = (
        po_trap = po_trap,          # BifurcationKit's trap problem for po_residual_bare!
        F_vals = zeros(n, M),       # Vector field at each slice
        temp = zeros(n),            # Temporary
    )
    
    return DiscretizedBVP(model, disc, cache)
end

# ============================================================================
# Collocation
# ============================================================================

function discretize(model::BVPModel, disc::Collocation)
    n = state_dimension(model)
    @assert n > 0 "State dimension must be specified in the model"
    
    Ntst, m = disc.Ntst, disc.m
    
    # Create a BifurcationProblem wrapper for the vector field
    prob_vf = BifurcationProblem(
        (u, p) -> model.F(u, p),
        zeros(n),
        (dummy = 0.0,),
        (@optic _.dummy);
        inplace = false,
        record_from_solution = (x, p; k...) -> nothing
    )
    
    # Create PeriodicOrbitOCollProblem
    po_coll = PeriodicOrbitOCollProblem(Ntst, m; N = n, prob_vf = prob_vf)
    
    cache = (
        po_coll = po_coll,
        # Lagrange matrices are already inside po_coll.mesh_cache
    )
    return DiscretizedBVP(model, disc, cache)
end

"""
Compute Lagrange interpolation matrices for collocation.
Returns (L, ∂L, gauss_nodes, gauss_weights).
"""
function compute_lagrange_matrices(m::Int)
    # Gauss-Legendre nodes and weights on [-1, 1]
    # For now, use simple approximation; in production use FastGaussQuadrature
    gauss_nodes = cos.(π .* (2 .* (1:m) .- 1) ./ (2m))  # Chebyshev nodes as approximation
    gauss_weights = fill(2.0 / m, m)  # Simplified weights
    
    # Collocation points on [-1, 1]
    σ = collect(LinRange(-1.0, 1.0, m + 1))
    
    # Lagrange basis matrices
    L = zeros(m + 1, m)   # L[j, l] = ℓⱼ(σₗ) where σₗ is Gauss point
    ∂L = zeros(m + 1, m)  # ∂L[j, l] = ℓ'ⱼ(σₗ)
    
    for l in 1:m
        x = gauss_nodes[l]
        for j in 1:(m + 1)
            L[j, l] = lagrange_basis(j, x, σ)
            ∂L[j, l] = lagrange_basis_derivative(j, x, σ)
        end
    end
    
    return L, ∂L, gauss_nodes, gauss_weights
end

"""Lagrange basis polynomial ℓⱼ(x)."""
function lagrange_basis(j::Int, x, nodes)
    n = length(nodes)
    result = one(x)
    for k in 1:n
        if k != j
            result *= (x - nodes[k]) / (nodes[j] - nodes[k])
        end
    end
    return result
end

"""Derivative of Lagrange basis polynomial ℓ'ⱼ(x)."""
function lagrange_basis_derivative(j::Int, x, nodes)
    n = length(nodes)
    result = zero(x)
    for i in 1:n
        if i != j
            term = one(x) / (nodes[j] - nodes[i])
            for k in 1:n
                if k != j && k != i
                    term *= (x - nodes[k]) / (nodes[j] - nodes[k])
                end
            end
            result += term
        end
    end
    return result
end

# ============================================================================
# Utility: Generate initial guess
# ============================================================================

"""
$(TYPEDSIGNATURES)

Generate an initial guess for the discretized BVP from an orbit function.

## Arguments
- `bvp::DiscretizedBVP`: The discretized problem
- `orbit`: Function `t -> u(t)` giving the orbit at time t
- `period`: Estimated period of the orbit

## Returns
- Initial guess vector X

## Example
```julia
x0 = generate_solution(bvp, t -> [cos(t), sin(t)], 2π)
```
"""
function generate_solution(bvp::DiscretizedBVP, orbit, period)
    return generate_solution(bvp.model, bvp.discretizer, orbit, period)
end

function generate_solution(model::BVPModel, disc::Shooting, orbit, period)
    n = state_dimension(model)
    M = disc.M
    
    # Sample at M shooting points
    X = zeros(n * M + 1)
    for i in 1:M
        t = (i - 1) / M * period
        X[(i-1)*n+1 : i*n] .= orbit(t)
    end
    X[end] = period
    
    return X
end

function generate_solution(model::BVPModel, disc::Trap, orbit, period)
    n = state_dimension(model)
    M = disc.M
    
    # Sample at M time slices
    X = zeros(n * M + 1)
    for i in 1:M
        t = (i - 1) / (M - 1) * period
        X[(i-1)*n+1 : i*n] .= orbit(t)
    end
    X[end] = period
    
    return X
end

function generate_solution(model::BVPModel, disc::Collocation, orbit, period)
    n = state_dimension(model)
    Ntst, m = disc.Ntst, disc.m
    N_total = Ntst * m + 1
    
    # Sample at all collocation points
    X = zeros(n * N_total + 1)
    mesh = LinRange(0.0, 1.0, Ntst + 1)
    σ = LinRange(-1.0, 1.0, m + 1)
    
    idx = 1
    for j in 1:Ntst
        for l in 1:(j == Ntst ? m + 1 : m)
            τ = mesh[j] + (1 + σ[l]) / 2 * (mesh[j+1] - mesh[j])
            t = τ * period
            X[(idx-1)*n+1 : idx*n] .= orbit(t)
            idx += 1
        end
    end
    X[end] = period
    
    return X
end
