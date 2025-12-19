# DiscretizedBVP - Discretized Boundary Value Problem
#
# This module defines the DiscretizedBVP struct which combines a BVPModel
# with a discretization method to create a callable functional.

using DocStringExtensions

"""
$(TYPEDEF)

A discretized BVP ready for Newton iteration and continuation.

Combines a mathematical model (`BVPModel`) with a discretization method
to produce a callable functional F(X, p) = 0.

## Fields
$(TYPEDFIELDS)

## Usage

The DiscretizedBVP is callable:
```julia
model = BVPModel(F, g; n=2)
disc = Trap(M=100)
bvp = discretize(model, disc)

# Evaluate residual
res = bvp_residual(bvp, X, params)

# Use with BVPBifProblem
prob = BVPBifProblem(bvp, X0, params, (@optic _.μ))
```
"""
struct DiscretizedBVP{Tmodel<:BVPModel, Tdisc<:AbstractDiscretizer, Tcache}
    "Mathematical BVP model"
    model::Tmodel
    
    "Discretization method"
    discretizer::Tdisc
    
    "Pre-allocated workspace (for performance)"
    cache::Tcache
end

# ============================================================================
# Interface: bvp_residual and bvp_jacobian
# ============================================================================
# These functions must be implemented for each discretizer type.
# The implementations are in the discretizer-specific files:
#   - trap/residual.jl, trap/jacobian.jl
#   - shooting/residual.jl, shooting/jacobian.jl
#   - collocation/residual.jl, collocation/jacobian.jl

"""
$(TYPEDSIGNATURES)

Compute the residual F(X, p) for the discretized BVP.
Must be implemented for each discretizer type.
"""
function bvp_residual end

"""
$(TYPEDSIGNATURES)

Compute the Jacobian ∂F/∂X with specified jacobian type.
Specialized implementations can be provided for each discretizer type.
"""
function bvp_jacobian end

# Default implementation for AutoDiffDense - uses ForwardDiff
import BifurcationKit: AutoDiffDense
function bvp_jacobian(d_bvp::DiscretizedBVP, ::AutoDiffDense, x, p)
    ForwardDiff.jacobian(z -> bvp_residual(d_bvp, z, p), x)
end

# ============================================================================
# Getters
# ============================================================================

"""State dimension."""
state_dimension(bvp::DiscretizedBVP) = state_dimension(bvp.model)

"""Total dimension of the discretized problem."""
Base.length(bvp::DiscretizedBVP) = total_dim(bvp.discretizer, state_dimension(bvp))

"""Get the underlying model."""
get_model(bvp::DiscretizedBVP) = bvp.model

"""Get the discretizer."""
get_discretizer(bvp::DiscretizedBVP) = bvp.discretizer

"""Get the period/parameter from solution vector X."""
getperiod(bvp::DiscretizedBVP, X, p=nothing) = X[end]

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, bvp::DiscretizedBVP)
    println(io, "┌─ DiscretizedBVP")
    println(io, "├─ State dimension : ", state_dimension(bvp))
    println(io, "├─ Total unknowns  : ", length(bvp))
    println(io, "├─ Model           : BVPModel")
    print(io,   "└─ Discretizer     : ", typeof(bvp.discretizer).name.name)
end
