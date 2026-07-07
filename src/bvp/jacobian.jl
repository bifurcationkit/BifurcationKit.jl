# bvp_jacobian interface and generic implementations
#
# This file defines the generic interface for BVP jacobians.

"""
$(TYPEDSIGNATURES)

Compute the Jacobian ∂F/∂X with specified jacobian type.
Specialized implementations can be provided for each discretizer type.
"""
function bvp_jacobian end

# Default implementation for AutoDiffDense - uses ForwardDiff
function bvp_jacobian(prob::AbstractDiscretizedBVP, ::BK.AutoDiffDense, x, p)
    FD.jacobian(z -> bvp_residual(prob, z, p), x)
end
