import Base: eltype, zero
import LinearAlgebra: norm, length, similar
import KrylovKit: VectorInterface
const VI = VectorInterface

"""
$(TYPEDEF)

This defines an "array" (although not `<: AbstractArray`) to hold two arrays or an array and a scalar. As such, it is a convenient alternative to `cat`, `vcat` and friends. It proves useful for the GPU where the operation `x[end]` can be slow.

This is useful when one wants to add constraints (phase, ...) to a functional for example. 
It is used throughout the package for the Pseudo Arc Length Continuation (PALC), for the continuation of Fold / Hopf points, for periodic orbits...

It is for example used to define periodic orbits as (orbit, period).
We chose not to make it a subtype of `AbstractArray` as we wish to apply the current package to general "arrays", see [Requested methods for Custom State](@ref). 

!!! danger "Required methods"
    The element in each (array, array) or (array, scalar) packed in a `BorderedArray` must comply with the interface of `VectorInterface.jl`.

!!! info "Note"
    In essence, it is very close to the vector `MinimalMVec` from `VectorInterface.jl`.
"""
mutable struct BorderedArray{𝒯1, 𝒯2}
    u::𝒯1
    p::𝒯2
end

# a version of copy which cope with our requirements concerning the methods
# available for
_copy(b) = VI.scale(b, 1)
_copy(::Nothing) = nothing
_copy(b::AbstractArray) = copy(b)

_copyto!(dest::AbstractArray, src::AbstractArray) = copyto!(dest, src)
_copyto!(dest::VI.MinimalVec, src::VI.MinimalVec) = VI.add!(dest, src, VI.One(), VI.Zero())

function _copyto!(dest::BorderedArray{𝒯v, 𝒯1}, src::BorderedArray{𝒯v, 𝒯2}) where {𝒯v, 𝒯1, 𝒯2 }
    _copyto!(dest.u, src.u)
    _copyto!(dest.p, src.p)
    return dest
end

function _copyto!(dest::BorderedArray{𝒯v, T1}, src::BorderedArray{𝒯v, T2}) where {𝒯v, T1 <: Number, T2 <: Number}
    _copyto!(dest.u, src.u)
    dest.p = src.p
    return dest
end

Base.length(b::BorderedArray{𝒯v, T}) where {𝒯v, T} = length(b.u) + length(b.p)
Base.length(b::BorderedArray{𝒯v, T}) where {𝒯v, T <: Number} = length(b.u) + 1
Base.length(u::VI.MinimalVec) = mapreduce(length, +, u.vec)

VI.inner(a::BorderedArray, b::BorderedArray) = VI.inner(a.u, b.u) + VI.inner(a.p, b.p)

function LA.norm(b::BorderedArray{Tv, Tp}, p::Real = 2) where {Tv, Tp}
    if p == 2
        # not using norm(⋅,2) is useful to avoid a fallback for MinimalVec
        return sqrt(norm(b.u)^2 + norm(b.p, 2)^2)
    elseif p == 1
        return LA.norm1(b.u) + LA.norm1(b.p)
    elseif p == Inf
        return max(norm(b.u, Inf), norm(b.p, Inf))
    elseif p == 0
        return norm(b.u, 0) + norm(b.p, 0)
    elseif p == -Inf
        return min(norm(b.u, -Inf), norm(b.p, -Inf))
    else
        return (norm(b.u, p)^p + norm(b.p, p)^p)^(1/p)
    end
end

# getters, useful for dispatch
getvec(x::AbstractVector) = @view x[begin:end-1]
getp(x::AbstractVector) = x[end]
getvec(x::BorderedArray) = x.u
getp(x::BorderedArray{vectype, T}) where {vectype, T <: Number} = x.p
################################################################################
# computes x-y into x and returns x
minus!!(x, y) = VI.add!!(x, y, -VI.One())
################################################################################
# returns x - y
minus(x, y) = VI.add(x, y, -VI.One())
################################################################################
# implements interface from VectorInterface for BorderedArray
# KrylovKit vector interface https://github.com/Jutho/VectorInterface.jl
@inline VI.scalartype(W::BorderedArray) = promote_type(VI.scalartype(W.u), VI.scalartype(W.p))
@inline VI.scalartype(::Type{BorderedArray{Tv, Tp}}) where {Tv, Tp} = promote_type(VI.scalartype(Tv), VI.scalartype(Tp))
########################
function VI.zerovector(b::BorderedArray, S::Type{<:Number} = VI.scalartype(b)) 
    return BorderedArray(VI.zerovector(b.u, S), VI.zerovector(b.p, S))
end

function VI.zerovector!(b::BorderedArray{𝒯v, 𝒯p})  where {𝒯v, 𝒯p}
    VI.zerovector!(b.u)
    VI.zerovector!(b.p)
    return b
end

function VI.zerovector!(b::BorderedArray{𝒯v, 𝒯p})  where {𝒯v, 𝒯p <: Number}
    VI.zerovector!(b.u)
    b.p *= 0
    return b
end
VI.zerovector!!(b::BorderedArray) = VI.zerovector!(b)
########################
VI.scale(x::BorderedArray, α::Number) = BorderedArray(VI.scale(x.u, α), VI.scale(x.p, α))

function VI.scale!(x::BorderedArray, α::Number)
    VI.scale!(x.u, α)
    VI.scale!(x.p, α)
    return x
end

function VI.scale!(x::BorderedArray{𝒯v, 𝒯p}, α::Number) where {𝒯v, 𝒯p <: Number} 
    VI.scale!(x.u, α)
    x.p *= α
    return x
end

function VI.scale!!(x::BorderedArray, α::Number)
    u = VI.scale!!(x.u, α)
    p = VI.scale!!(x.p, α)
    if u === x.u && p === x.p
        return x
    else
        return BorderedArray(u, p)
    end
end
########################
function VI.scale!(y::BorderedArray{𝒯v1, 𝒯p1}, x::BorderedArray{𝒯v2, 𝒯p2}, α::Number) where {𝒯v1, 𝒯p1, 𝒯v2, 𝒯p2}
    VI.scale!(y.u, x.u, α)
    VI.scale!(y.p, x.p, α)
    return y
end

function VI.scale!(y::BorderedArray{𝒯v1, 𝒯p1}, x::BorderedArray{𝒯v2, 𝒯p2}, α::Number) where {𝒯v1, 𝒯v2, 𝒯p1 <: Number, 𝒯p2 <: Number}
    VI.scale!(y.u, x.u, α)
    y.p = x.p * α
    return y
end

function VI.scale!!(y::BorderedArray, x::BorderedArray, α::Number) 
   u = VI.scale!!(y.u, x.u, α)
   if u === y.u
        p = VI.scale!!(y.p, x.p, α)
        return y
    else
        p = VI.scale!!(y.p, x.p, α)
        return BorderedArray(u, p)
    end
end

function VI.scale!!(y::BorderedArray{𝒯v1, 𝒯p1}, x::BorderedArray{𝒯v2, 𝒯p2}, α::Number) where {𝒯v1, 𝒯v2, 𝒯p1 <: Number, 𝒯p2 <: Number}
   u = VI.scale!!(y.u, x.u, α)
   if u === y.u
        y.p = VI.scale!!(y.p, x.p, α)
        return y
    else
        p = VI.scale!!(y.p, x.p, α)
        return BorderedArray(u, p)
    end
end
########################
function VI.add(y::BorderedArray, 
                x::BorderedArray,
                α::Number, 
                β::Number)
    BorderedArray(VI.add(y.u, x.u, α, β), VI.add(y.p, x.p, α, β))
end

function VI.add!(y::BorderedArray, 
                 x::BorderedArray, 
                 α::Number, 
                 β::Number)
    VI.add!(y.u, x.u, α, β)
    VI.add!(y.p, x.p, α, β)
    return y
end

function VI.add!(y::BorderedArray{𝒯v1, 𝒯p1}, 
                 x::BorderedArray, 
                 α::Number, 
                 β::Number) where {𝒯v1, 𝒯p1 <: Number}
    VI.add!(y.u, x.u, α, β)
    y.p = y.p * β + x.p * α
    return y
end

function VI.add!!(y::BorderedArray, 
                  x::BorderedArray, 
                  α::Number, 
                  β::Number) 
    u = VI.add!!(y.u, x.u, α, β)
    p = VI.add!!(y.p, x.p, α, β)
    if u === y.u && p == y.p
        return y
    else
        p = VI.add!!(y.p, x.p, α, β)
        return BorderedArray(u, p)
    end
end

function VI.add!!(y::BorderedArray{𝒯v1, 𝒯p1}, 
                  x::BorderedArray{𝒯v2, 𝒯p2}, 
                  α::Number, 
                  β::Number) where {𝒯v1, 𝒯v2, 𝒯p1 <: Number, 𝒯p2 <: Number}
    u = VI.add!!(y.u, x.u, α, β)
    if u === y.u
        y.p = VI.add!!(y.p, x.p, α, β)
        return y
    else
        p = VI.add!!(y.p, x.p, α, β)
        return BorderedArray(u, p)
    end
end
########################
"""
$(TYPEDSIGNATURES)

Initialize a vector like `randn!`.
"""
_randn(x) = randn!(_copy(x))
_randn(x::VI.MinimalVec) = randn!(_copy(x.vec))
_randn!(x::VI.MinimalVec) = (randn!(x.vec);x)

_randn(y::BorderedArray{T, V}) where {T, V} = (x = _copy(y);_randn!(x);x)

function _randn!(x::BorderedArray{T, V}) where {T, V}
    _randn!(x.u)
    if V <: Number
        x.p = randn(V)
    elseif T <: AbstractArray
        randn!(x.p)
    end
    return x
end