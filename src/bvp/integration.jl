# Utility functions for BVP integration
#
# This file provides utility functions for working with DiscretizedBVP.
# The main integration is done through BVPBifProblem in BVPBifProblem.jl.


# Alternative: make DiscretizedBVP directly usable
# This allows passing bvp directly to newton() if the interface matches



"""
$(TYPEDSIGNATURES)

Extract the periodic orbit from the discretized solution.

Returns a `NamedTuple` with fields:
- `t`: Time points
- `u`: Solution values at each time point (matrix n × M)
- `period`: The period T
"""
function get_periodic_orbit(bvp::DiscretizedBVP, X, p)
    n = state_dimension(bvp)
    disc = get_discretizer(bvp)
    T = X[end]
    
    return _get_periodic_orbit(disc, X, n, T)
end

function _get_periodic_orbit(disc::Shooting, X, n, T)
    M = disc.M
    U = reshape(@view(X[1:n*M]), n, M)
    t = LinRange(0, T, M+1)[1:M]
    return (t = collect(t), u = U, period = T)
end

function _get_periodic_orbit(disc::Trap, X, n, T)
    M = disc.M
    U = reshape(@view(X[1:n*M]), n, M)
    t = LinRange(0, T, M)
    return (t = collect(t), u = U, period = T)
end

function _get_periodic_orbit(disc::Collocation, X, n, T)
    Ntst, m = disc.Ntst, disc.m
    N_total = Ntst * m + 1
    U = reshape(@view(X[1:n*N_total]), n, N_total)
    # Approximate times (actual times depend on mesh)
    t = LinRange(0, T, N_total)
    return (t = collect(t), u = U, period = T)
end

"""
$(TYPEDSIGNATURES)

Update the reference solution for the phase condition.
This is called during continuation to keep the phase constraint valid.
"""
function update_phase_reference!(bvp::DiscretizedBVP, X, p)
    # For now, this is a no-op
    # In a full implementation, this would update the cache with the new reference
    return true
end
