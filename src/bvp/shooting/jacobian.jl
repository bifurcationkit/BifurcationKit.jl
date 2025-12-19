# Shooting Jacobian Implementation

import BifurcationKit: AbstractJacobianType, AutoDiffDense

"""
$(TYPEDSIGNATURES)

Compute the Jacobian for shooting discretization.

## Jacobian Structure (for M=3 intervals, periodic):
    ┌                                           ┐
    │  Φ₁   -I    0     ∂R₁/∂T                 │
    │   0   Φ₂   -I     ∂R₂/∂T                 │
    │  -I    0   Φ₃     ∂R₃/∂T                 │
    │ ∂g/∂u₁  0    0     ∂g/∂T                 │
    └                                           ┘

Where Φᵢ = ∂φ(uᵢ)/∂uᵢ is the monodromy matrix.
"""

function bvp_jacobian(bvp::DiscretizedBVP{<:BVPModel, <:Shooting}, jac::AutoDiffDense, X, p)
    model = bvp.model
    disc = bvp.discretizer
    
    n = state_dimension(model)
    M = disc.M
    N = n * M + 1
    
    # Extract shooting points and period
    U = reshape(@view(X[1:n*M]), n, M)
    T = X[end]
    dt = T / M
    
    # Allocate Jacobian
    J = zeros(eltype(X), N, N)
    In = Matrix{eltype(X)}(LinearAlgebra.I, n, n)
    
    for i in 1:M
        uᵢ = U[:, i]
        row_start = (i - 1) * n + 1
        row_end = i * n
        
        # Compute flow and monodromy matrix
        u_final, Φᵢ = integrate_with_sensitivity(model.F, uᵢ, p, dt, disc.alg)
        
        # ∂Rᵢ/∂uᵢ = Φᵢ (monodromy matrix)
        col_start = (i - 1) * n + 1
        col_end = i * n
        J[row_start:row_end, col_start:col_end] .= Φᵢ
        
        # ∂Rᵢ/∂uᵢ₊₁ = -I
        i_next = (i == M) ? 1 : i + 1
        col_start_next = (i_next - 1) * n + 1
        col_end_next = i_next * n
        J[row_start:row_end, col_start_next:col_end_next] .= -In
        
        # ∂Rᵢ/∂T = F(u_final) / M
        J[row_start:row_end, end] .= model.F(u_final, p) ./ M
    end
    
    # Last row: phase condition Jacobian
    u0 = U[:, 1]
    uM = U[:, M]
    uT = integrate_shooting(model.F, uM, p, dt, disc.alg)
    
    # ∂g/∂u₁
    J[end, 1:n] .= ForwardDiff.gradient(u -> _shooting_phase(model, u, uT, p), u0)
    
    # ∂g/∂T (finite difference for simplicity)
    δ = 1e-8
    uT_plus = integrate_shooting(model.F, uM, p, dt + δ/M, disc.alg)
    phase_plus = _shooting_phase(model, u0, uT_plus, p)
    phase_current = _shooting_phase(model, u0, uT, p)
    J[end, end] = (phase_plus - phase_current) / δ
    
    return J
end

"""
Integrate ODE and return both final state and monodromy matrix.

This is a dispatch point:
- `alg=nothing`: Uses Euler + ForwardDiff for monodromy
- `alg::OrdinaryDiffEqAlgorithm`: Uses variational equations (requires extension)
"""
function integrate_with_sensitivity end

# Default: nothing algorithm uses Euler + AD
function integrate_with_sensitivity(F, u0, p, dt, ::Nothing)
    # Euler integration + AD for monodromy matrix
    u_final = euler_integrate(F, u0, p, dt)
    Φ = ForwardDiff.jacobian(u -> euler_integrate(F, u, p, dt), u0)
    return u_final, Φ
end

# Fallback for unknown algorithm types
function integrate_with_sensitivity(F, u0, p, dt, alg)
    error("""
    Sensitivity computation with algorithm $(typeof(alg)) requires OrdinaryDiffEq.jl.
    
    Either:
    1. Use `Shooting(alg=nothing)` for built-in Euler integration
    2. Load OrdinaryDiffEq: `using OrdinaryDiffEq`
    """)
end
