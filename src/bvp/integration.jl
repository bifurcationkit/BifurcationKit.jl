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
    T = X[end]
    return _get_periodic_orbit(bvp, X, T)
end

function _get_periodic_orbit(bvp::DiscretizedBVP{Tmodel, <:Shooting}, X, T) where {Tmodel}
    M = mesh_size(get_discretizer(bvp))
    U = get_time_slices(bvp, X)
    t = LinRange(0, T, M+1)[1:M]
    return (t = collect(t), u = U, period = T)
end

function _get_periodic_orbit(bvp::DiscretizedBVP{Tmodel, <:Trapeze}, X, T) where {Tmodel}
    M = mesh_size(get_discretizer(bvp))
    U = get_time_slices(bvp, X)
    t = LinRange(0, T, M)
    return (t = collect(t), u = U, period = T)
end

function _get_periodic_orbit(bvp::DiscretizedBVP{Tmodel, <:Collocation}, X, T) where {Tmodel}
    disc = get_discretizer(bvp)
    Ntst, m = get_ntst(disc), get_m(disc)
    N_total = Ntst * m + 1
    U = get_time_slices(bvp, X)
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
