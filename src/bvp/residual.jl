# bvp_residual interface
#
# This file defines the generic interface for BVP residuals.

"""
$(TYPEDSIGNATURES)

Compute the residual F(X, p) for the discretized BVP.
Must be implemented for each discretizer type.
"""
function bvp_residual end
