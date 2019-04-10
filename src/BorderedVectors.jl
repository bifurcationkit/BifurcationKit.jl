# structure for Bordered vectors
import Base: copy, copyto!, eltype, zero
import LinearAlgebra: norm, dot, length, similar, axpy!, axpby!, rmul!

mutable struct BorderedVector{vectype1, vectype2}
	u::vectype1
	p::vectype2
end

eltype(b::BorderedVector{vectype, T}) where {T, vectype} = T
similar(b::BorderedVector{vectype, T}, ::Type{S} = T) where {S, T, vectype} = BorderedVector(similar(b.u, S), similar(b.p, S))
similar(b::BorderedVector{vectype, T}, ::Type{S} = T) where {S, T <: Real, vectype} = BorderedVector(similar(b.u, S), S(0))

copy(b::BorderedVector{vectype, T}) where {vectype, T} =  BorderedVector(copy(b.u), copy(b.p))
copyto!(dest::BorderedVector{vectype, T}, src::BorderedVector{vectype, T}) where {vectype, T } = (copyto!(dest.u, src.u); copyto!(dest.p, src.p);dest)
copyto!(dest::BorderedVector{vectype, T}, src::BorderedVector{vectype, T}) where {vectype, T <: Number} = (copyto!(dest.u, src.u); dest.p = src.p;dest)


length(b::BorderedVector{vectype, T}) where {vectype, T} = length(b.u) + length(b.p)

dot(a::BorderedVector{vectype, T}, b::BorderedVector{vectype, T}) where {vectype, T} = dot(a.u, b.u) + dot(a.p, b.p)

norm(b::BorderedVector{vectype, T}, p::Real) where {vectype, T} = max(norm(b.u, p), norm(b.p, p))

zero(b::BorderedVector{vectype, T}) where {vectype, T } = BorderedVector(zero(b.u), zero(b.p))
################################################################################
function rmul!(A::BorderedVector{vectype, Tv}, b::T) where {vectype, T <:Real, Tv}
	# Scale an array A by a scalar b overwriting A in-place
	rmul!(A.u, b)
	rmul!(A.p, b)
end

function rmul!(A::BorderedVector{vectype, T}, b::T) where {vectype, T <:Real}
	# Scale an array A by a scalar b overwriting A in-place
	rmul!(A.u, b)
	A.p = A.p * b
end
################################################################################
function axpy!(a::T, X::BorderedVector{vectype, Tv}, Y::BorderedVector{vectype, Tv}) where {vectype, T <:Real, Tv}
	# Overwrite Y with a*X + b*Y, where a, b are scalar
	axpy!(a, X.u, Y.u)
	axpy!(a, X.p, Y.p)
	return Y
end

function axpy!(a::T, X::BorderedVector{vectype, T}, Y::BorderedVector{vectype, T}) where {vectype, T <:Real}
	# Overwrite Y with a*X + b*Y, where a, b are scalar
	axpy!(a, X.u, Y.u)
	Y.p = a * X.p + Y.p
	return Y
end
################################################################################
function axpby!(a::T, X::BorderedVector{vectype, Tv}, b::T, Y::BorderedVector{vectype, Tv}) where {vectype, T <: Real, Tv}
	# Overwrite Y with a*X + b*Y, where a, b are scalar
	axpby!(a, X.u, b, Y.u)
	axpby!(a, X.p, b, Y.p)
	return Y
end

function axpby!(a::T, X::BorderedVector{vectype, T}, b::T, Y::BorderedVector{vectype, T}) where {vectype, T <:Real}
	# Overwrite Y with a*X + b*Y, where a is a scalar
	axpby!(a, X.u, b, Y.u)
	Y.p = a * X.p + b * Y.p
	return Y
end
################################################################################
# this function is actually axpy!(-1, y, x)
"""
	minus!(x,y)
computes x-y into x and return x
"""
@inline minus!(x, y) = (x .= x .- y) # necessary to put a dot .= for ApproxFun to work
@inline minus!(x::vec, y::vec) where {vec <: AbstractArray} = (x .= x .- y)
@inline minus!(x::T, y::T) where {T <:Real} = (x = x - y)
minus!(x::BorderedVector{vectype, T},y::BorderedVector{vectype, T}) where {vectype, T}=(minus!(x.u, y.u);minus!(x.p, y.p))
function minus!(x::BorderedVector{vectype, T},y::BorderedVector{vectype, T}) where {vectype, T <: Real}
	minus!(x.u, y.u)
	# Carefull here. If I uncomment the line below, then x.p will be left unaffected
	# minus_!(x.p, y.p)
	x.p = x.p - y.p
	return x
end
################################################################################
# this function is actually axpy!(-1, y, x)
"""
	minus(x,y)
returns x-y
"""
@inline minus(x, y) = (return x .- y) # necessary to put a dot .= for ApproxFun to work
@inline minus(x::vec, y::vec) where {vec <: AbstractArray} = (return x .- y)
@inline minus(x::T, y::T) where {T <:Real} = (return x - y)
@inline minus(x::BorderedVector{vectype, T},y::BorderedVector{vectype, T}) where {vectype, T} = (return BorderedVector(minus(x.u, y.u), minus(x.p, y.p)))
@inline minus(x::BorderedVector{vectype, T},y::BorderedVector{vectype, T}) where {vectype, T <: Real} = (return BorderedVector(minus(x.u, y.u), x.p - y.p))
################################################################################
function normalize(x)
	out = copy(x)
	rmul!(out, norm(x))
	return out
end
################################################################################
function dottheta(u1, u2, p1::T, p2::T, theta::T) where T
	return dot(u1, u2) * theta / length(u1) + p1 * p2 * (one(T) - theta)
end
################################################################################
function normtheta(u, p::T, theta::T) where T
	return sqrt(dottheta(u, u, p, p, theta))
end
################################################################################
function dottheta(a::BorderedVector{vectype, T}, b::BorderedVector{vectype, T}, theta::T) where {vectype, T}
	return dottheta(a.u, b.u, a.p, b.p, theta)
end
################################################################################
function normtheta(a::BorderedVector{vectype, T}, theta::T) where {vectype, T}
	return normtheta(a.u, a.p, theta)
end
