"""

$(TYPEDEF)

Structure describing a normalized time mesh through interval sizes `t_{i+1} - t_i`.
If the time steps are constant, we store only the number of intervals, yielding
`TimeMesh{Int}` with `get_time_step(mesh, i) = 1 / n_intervals`.
"""
struct TimeMesh{T}
    ds::T
end

TimeMesh(M::Int) = TimeMesh{Int}(M)

@inline can_adapt(ms::TimeMesh{Ti}) where Ti = !(Ti <: Int)
Base.length(ms::TimeMesh{Ti}) where Ti = length(ms.ds)
Base.length(ms::TimeMesh{Ti}) where {Ti <: Int} = ms.ds

# access the normalized time steps
@inline get_time_step(ms::TimeMesh, i::Int) = ms.ds[i]
@inline get_time_step(ms::TimeMesh{Ti}, i::Int) where {Ti <: Int} = 1.0 / ms.ds

Base.collect(ms::TimeMesh) = ms.ds
Base.collect(ms::TimeMesh{Ti}) where {Ti <: Int} = repeat([get_time_step(ms, 1)], ms.ds)
