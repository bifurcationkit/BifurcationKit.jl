# structure for Bordered arrays
import Base: copy, copyto!, eltype, zero
import LinearAlgebra: norm, dot, length, similar, axpy!, axpby!, rmul!, mul!

mutable struct BorderedArray{vectype1, vectype2}
	u::vectype1
	p::vectype2
end

eltype(b::BorderedArray{vectype, T}) where {T, vectype} = T
similar(b::BorderedArray{vectype, T}, ::Type{S} = T) where {S, T, vectype} = BorderedArray(similar(b.u, S), similar(b.p, S))
similar(b::BorderedArray{vectype, T}, ::Type{S} = T) where {S, T <: Real, vectype} = BorderedArray(similar(b.u, S), S(0))

copy(b::BorderedArray{vectype, T}) where {vectype, T} =  BorderedArray(copy(b.u), copy(b.p))
copyto!(dest::BorderedArray{vectype, T}, src::BorderedArray{vectype, T}) where {vectype, T } = (copyto!(dest.u, src.u); copyto!(dest.p, src.p);dest)
copyto!(dest::BorderedArray{vectype, T}, src::BorderedArray{vectype, T}) where {vectype, T <: Number} = (copyto!(dest.u, src.u); dest.p = src.p;dest)


length(b::BorderedArray{vectype, T}) where {vectype, T} = length(b.u) + length(b.p)

dot(a::BorderedArray{vectype, T}, b::BorderedArray{vectype, T}) where {vectype, T} = dot(a.u, b.u) + dot(a.p, b.p)

norm(b::BorderedArray{vectype, T}, p::Real) where {vectype, T} = max(norm(b.u, p), norm(b.p, p))

zero(b::BorderedArray{vectype, T}) where {vectype, T } = BorderedArray(zero(b.u), zero(b.p))
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
function mul!(A::BorderedArray{vectype, Tv}, B::BorderedArray{vectype, Tv}, α::T) where {vectype, Tv, T <: Number}
	mul!(A.u, B.u, α)
	mul!(A.p, B.p, α)
	return A
end

function mul!(A::BorderedArray{vectype, Tv}, B::BorderedArray{vectype, Tv}, α::T) where {vectype, Tv <: Number, T <: Number}
	mul!(A.u, B.u, α)
	A.p = B.p * α
	return A
end

mul!(A::BorderedArray{vectype, Tv}, α::T, B::BorderedArray{vectype, Tv}) where {vectype, Tv, T} = mul!(A, B,  α)
################################################################################
function axpy!(a::T, X::BorderedArray{vectype, Tv}, Y::BorderedArray{vectype, Tv}) where {vectype, T <:Real, Tv}
	# Overwrite Y with a*X + b*Y, where a, b are scalar
	axpy!(a, X.u, Y.u)
	axpy!(a, X.p, Y.p)
	return Y
end

function axpy!(a::T, X::BorderedArray{vectype, T}, Y::BorderedArray{vectype, T}) where {vectype, T <:Real}
	# Overwrite Y with a*X + b*Y, where a, b are scalar
	axpy!(a, X.u, Y.u)
	Y.p = a * X.p + Y.p
	return Y
end
################################################################################
function axpby!(a::T, X::BorderedArray{vectype, Tv}, b::T, Y::BorderedArray{vectype, Tv}) where {vectype, T <: Real, Tv}
	# Overwrite Y with a*X + b*Y, where a, b are scalar
	axpby!(a, X.u, b, Y.u)
	axpby!(a, X.p, b, Y.p)
	return Y
end

function axpby!(a::T, X::BorderedArray{vectype, T}, b::T, Y::BorderedArray{vectype, T}) where {vectype, T <:Real}
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
minus!(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T} = (minus!(x.u, y.u);minus!(x.p, y.p))
function minus!(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T <: Real}
	minus!(x.u, y.u)
	# Carefull here. If I use the line below, then x.p will be left unaffected
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
@inline minus(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T} = (return BorderedArray(minus(x.u, y.u), minus(x.p, y.p)))
@inline minus(x::BorderedArray{vectype, T}, y::BorderedArray{vectype, T}) where {vectype, T <: Real} = (return BorderedArray(minus(x.u, y.u), x.p - y.p))
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
function dottheta(a::BorderedArray{vectype, T}, b::BorderedArray{vectype, T}, theta::T) where {vectype, T}
	return dottheta(a.u, b.u, a.p, b.p, theta)
end
################################################################################
function normtheta(a::BorderedArray{vectype, T}, theta::T) where {vectype, T}
	return normtheta(a.u, a.p, theta)
end
