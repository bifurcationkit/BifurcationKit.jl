# composite type for Bordered arrays
import Base: eltype, zero, eltype
import LinearAlgebra: norm, dot, length, similar, axpy!, axpby!, rmul!, mul!

"""
    x = BorderedArray(vec1, vec2)

This defines an array (although not `<: AbstractArray`) to hold two arrays or an array and a scalar. This is useful when one wants to add constraints (phase, ...) to a functional for example. It is used throughout the package for the Pseudo Arc Length Continuation, for the continuation of Fold / Hopf points, for periodic orbits... It is also used to define periodic orbits as (orbit, period). As such, it is a convenient alternative to `cat`, `vcat` and friends. We chose not to make it a subtype of AbstractArray as we wish to apply the current package to general "arrays", see [Requested methods for Custom State](@ref). Finally, it proves useful for the GPU where the operation `x[end]` can be slow.
"""
mutable struct BorderedArray{vectype1, vectype2}
    u::vectype1
    p::vectype2
end

eltype(::Type{BorderedArray{vectype, T}}) where {vectype, T} = eltype(T)
similar(b::BorderedArray{vectype, T}, ::Type{S} = eltype(b)) where {S, T, vectype} = BorderedArray(similar(b.u, S), similar(b.p, S))
similar(b::BorderedArray{vectype, T}, ::Type{S} = eltype(b)) where {S, T <: Number, vectype} = BorderedArray(similar(b.u, S), S(0))

Base.:*(a::S, b::BorderedArray{vectype, T}) where {vectype, T, S <: Number} = BorderedArray(*(a, b.u),*(a, b.p))

# a version of copy which cope with our requirements concerning the methods
# available for
_copy(b) = 1*b
_copy(b::AbstractArray) = copy(b)
Base.copy(b::BorderedArray) = BorderedArray(_copy(b.u), _copy(b.p))

function Base.copyto!(dest::BorderedArray{vectype, T1}, src::BorderedArray{vectype, T2}) where {vectype, T1, T2 }
    copyto!(dest.u, src.u)
    copyto!(dest.p, src.p)
    return dest
end

function Base.copyto!(dest::BorderedArray{vectype, T1}, src::BorderedArray{vectype, T2}) where {vectype, T1 <: Number, T2 <: Number}
    copyto!(dest.u, src.u)
    dest.p = src.p
    return dest
end

length(b::BorderedArray{vectype, T}) where {vectype, T} = length(b.u) + length(b.p)
length(b::BorderedArray{vectype, T}) where {vectype, T <: Number} = length(b.u) + 1

dot(a::BorderedArray, b::BorderedArray) = dot(a.u, b.u) + dot(a.p, b.p)
norm(b::BorderedArray{vectype, T}, p::Real) where {vectype, T} = max(norm(b.u, p), norm(b.p, p))
zero(b::BorderedArray{vectype, T}) where {vectype, T} = BorderedArray(zero(b.u), zero(b.p))

# getters, useful for dispatch
getVec(x::AbstractVector) = @view x[1:end-1]
getP(x::AbstractVector) = x[end]
getVec(x::BorderedArray) = x.u
getP(x::BorderedArray{vectype, T}) where {vectype, T <: Number} = x.p
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
function axpy!(a::T, X::BorderedArray{Tv1, Tp1}, Y::BorderedArray{Tv2, Tp2}) where {Tv1, Tv2, T <: Number, Tp1, Tp2}
    # Overwrite Y with a*X + Y, where a is scalar
    axpy!(a, X.u, Y.u)
    axpy!(a, X.p, Y.p)
    return Y
end

function axpy!(a::T, X::BorderedArray{Tv1, Tp1}, Y::BorderedArray{Tv2, Tp2}) where {Tv1, Tv2, T <: Number, Tp1 <: Number, Tp2 <: Number}
    # Overwrite Y with a*X + Y, where a is scalar
    axpy!(a, X.u, Y.u)
    Y.p = a * X.p + Y.p
    return Y
end
################################################################################
function axpby!(a::T, X::BorderedArray{vectype, Tv1}, b::T, Y::BorderedArray{vectype, Tv2}) where {vectype, T <: Number, Tv1, Tv2}
    # Overwrite Y with a * X + b * Y, where a, b are scalar
    axpby!(a, X.u, b, Y.u)
    axpby!(a, X.p, b, Y.p)
    return Y
end

function axpby!(a::T, X::BorderedArray{vectype, Tv1}, b::T, Y::BorderedArray{vectype, Tv2}) where {vectype, Tv1 <: Number, Tv2 <: Number, T <: Number}
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
#    `minus(x,y)`
#
# returns x - y
minus(x, y) = (x1 = 1*x; minus!(x1, y); return x1)
# minus(x, y) = (x - y)
minus(x::vec, y::vec) where {vec <: AbstractArray} = (return x .- y)
minus(x::T, y::T) where {T <:Real} = (return x - y)
minus(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T} = (return BorderedArray(minus(x.u, y.u), minus(x.p, y.p)))
minus(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T <: Number} = (return BorderedArray(minus(x.u, y.u), x.p - y.p))
