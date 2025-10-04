import Base: eltype, zero, eltype
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
_copy(b) = 1*b
_copy(::Nothing) = nothing
_copy(b::AbstractArray) = copy(b)
Base.copy(b::BorderedArray) = BorderedArray(_copy(b.u), _copy(b.p))

function Base.copyto!(dest::BorderedArray{𝒯v, 𝒯1}, src::BorderedArray{𝒯v, 𝒯2}) where {𝒯v, 𝒯1, 𝒯2 }
    copyto!(dest.u, src.u)
    copyto!(dest.p, src.p)
    return dest
end

function Base.copyto!(dest::BorderedArray{𝒯v, T1}, src::BorderedArray{𝒯v, T2}) where {𝒯v, T1 <: Number, T2 <: Number}
    copyto!(dest.u, src.u)
    dest.p = src.p
    return dest
end

length(b::BorderedArray{𝒯v, T}) where {𝒯v, T} = length(b.u) + length(b.p)
length(b::BorderedArray{𝒯v, T}) where {𝒯v, T <: Number} = length(b.u) + 1

dot(a::BorderedArray, b::BorderedArray) = dot(a.u, b.u) + dot(a.p, b.p)
norm(b::BorderedArray{vectype, T}, p::Real) where {vectype, T} = max(norm(b.u, p), norm(b.p, p))

# getters, useful for dispatch
getvec(x::AbstractVector) = @view x[begin:end-1]
getp(x::AbstractVector) = x[end]
getvec(x::BorderedArray) = x.u
getp(x::BorderedArray{vectype, T}) where {vectype, T <: Number} = x.p
################################################################################
function rmul!(A::BorderedArray{vectype, Tv}, a::T, b::T) where {vectype, T <: Number, Tv}
    # Scale an array A by a scalar b overwriting A in-place
    rmul!(A.u, a)
    rmul!(A.p, b)
    return A
end

function rmul!(A::BorderedArray{vectype, Tv}, a::T, b::T) where {vectype, Tv <: Number, T <: Number }
    # Scale an array A by a scalar b overwriting A in-place
    rmul!(A.u, a)
    A.p = A.p * b
    return A
end

rmul!(A::BorderedArray{vectype, Tv}, a::T) where {vectype, T <: Number, Tv} = rmul!(A, a, a)
################################################################################
function mul!(A::BorderedArray{Tv1, Tp1}, B::BorderedArray{Tv2, Tp2}, α::T) where {Tv1, Tv2, Tp1, Tp2, T <: Number}
    mul!(A.u, B.u, α)
    mul!(A.p, B.p, α)
    return A
end

function mul!(A::BorderedArray{Tv1, Tp1}, B::BorderedArray{Tv2, Tp2}, α::T) where {Tv1, Tv2, Tp1 <: Number, Tp2 <: Number, T <: Number}
    mul!(A.u, B.u, α)
    A.p = B.p * α
    return A
end

mul!(A::BorderedArray{Tv1, Tp1}, α::T, B::BorderedArray{Tv2, Tp2}) where {Tv1, Tv2, Tp1, Tp2, T} = mul!(A, B, α)
################################################################################
function axpy!(a::Number, 
               X::BorderedArray{Tv1, Tp1}, 
               Y::BorderedArray{Tv2, Tp2}) where {Tv1, Tv2, Tp1, Tp2}
    # Overwrite Y with a * X + Y, where a is scalar
    axpy!(a, X.u, Y.u)
    axpy!(a, X.p, Y.p)
    return Y
end

function axpy!(a::Number,
               X::BorderedArray{Tv1, Tp1}, 
               Y::BorderedArray{Tv2, Tp2}) where {Tv1, Tv2, Tp1 <: Number, Tp2 <: Number}
    # Overwrite Y with a * X + Y, where a is scalar
    axpy!(a, X.u, Y.u)
    Y.p = a * X.p + Y.p
    return Y
end
################################################################################
function axpby!(a::Number, X::BorderedArray{vectype, Tv1}, 
                b::Number, Y::BorderedArray{vectype, Tv2}) where {vectype, Tv1, Tv2}
    # Overwrite Y with a * X + b * Y, where a, b are scalar
    axpby!(a, X.u, b, Y.u)
    axpby!(a, X.p, b, Y.p)
    return Y
end

function axpby!(a::Number, X::BorderedArray{vectype, Tv1}, 
                b::Number, Y::BorderedArray{vectype, Tv2}) where {vectype, Tv1 <: Number, Tv2 <: Number}
    # Overwrite Y with a * X + b * Y, where a is a scalar
    axpby!(a, X.u, b, Y.u)
    Y.p = a * X.p + b * Y.p
    return Y
end
################################################################################
# computes x-y into x and returns x
minus!(x, y) = axpy!(-1, y, x)
minus!(x::vec, y::vec) where {vec <: AbstractArray} = (x .= x .- y)
minus!(x::T, y::T) where {T <: Number} = (x = x - y)
function minus!(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T}
    minus!(x.u, y.u)
    minus!(x.p, y.p)
    return x
end
function minus!(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T <: Number}
    minus!(x.u, y.u)
    # Carefull here. If I use the line below, then x.p will be left unaffected
    # minus_!(x.p, y.p)
    x.p = x.p - y.p
    return x
end
################################################################################
#
#    `minus(x, y)`
#
# returns x - y
minus(x, y) = (x1 = 1*x; minus!(x1, y); return x1)
# minus(x, y) = (x - y)
minus(x::vec, y::vec) where {vec <: AbstractArray} = (return x .- y)
minus(x::T, y::T) where {T <:Real} = (return x - y)
minus(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T} = (return BorderedArray(minus(x.u, y.u), minus(x.p, y.p)))
minus(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T <: Number} = (return BorderedArray(minus(x.u, y.u), x.p - y.p))
################################################################################
# implements interface from VectorInterface
# KrylovKit vector interface https://github.com/Jutho/VectorInterface.jl

function VI.add(W::BorderedArray{vectype, Tv1}, 
                V::BorderedArray{vectype, Tv1}, 
                α::Number = 1, 
                β::Number = 1) where {vectype, Tv1 <: Number}
    # compute W * β + V * α
    Z = copy(W)
    axpby!(α, V, β, Z)
    Z
end

VI.add!(y::BorderedArray, x::BorderedArray, α::Number = 1, β::Number = 1) = axpby!(α, x, β, y)
VI.add!!(y::BorderedArray, x::BorderedArray, α::Number = 1, β::Number = 1) = VI.add!(y, x, α, β)
@inline VI.scalartype(W::BorderedArray{vectype, Tv1}) where {vectype, Tv1} = eltype(BorderedArray{vectype, Tv1})
VI.zerovector(b::BorderedArray, S::Type{<:Number} = VI.scalartype(b)) = BorderedArray(VI.zerovector(b.u, S), VI.zerovector(b.p, S))
VI.scale(x::BorderedArray, α::Number) = mul!(similar(x), x, α)
VI.scale!(y::BorderedArray, x::BorderedArray, α::Number) = throw("")#mul!(y, x, α)
VI.scale!!(x::BorderedArray, α::Number) = α * x
VI.scale!!(y::BorderedArray, x::BorderedArray, α::Number) = mul!(y, x, α)
VI.inner(x::BorderedArray, y::BorderedArray) = dot(x, y)

"""
Initialize a vector like `randn!`.
"""
_randn(x) = randn!(_copy(x))
function _randn(y::BorderedArray{T,V}) where {T <: AbstractArray, V}
    x = _copy(y)
    randn!(x.u)
    if V <: Number
        x.p = randn(V)
    elseif T <: AbstractArray
        randn!(x.p)
    end
    return x
end