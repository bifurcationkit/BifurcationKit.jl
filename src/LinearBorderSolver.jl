# structure for Bordered vectors
import Base: copy, copyto!
import LinearAlgebra: norm, dot, length, similar, axpy!, rmul!

mutable struct BorderedVector{vectype1, vectype2}
	u::vectype1
	p::vectype2
end

# copy(b::BorderedVector{vectype, T})   where {vectype, T} = BorderedVector(copy(b.u), b.p)

similar(b::BorderedVector{vectype, T}) where {T, vectype} = BorderedVector(similar(b.u), similar(b.p))
similar(b::BorderedVector{vectype, T}) where {T <: Real, vectype} = BorderedVector(similar(b.u), T(0))

copyto!(dest::BorderedVector{vectype, T}, src::BorderedVector{vectype, T}) where {vectype, T } = (copyto!(dest.u, src.u); copyto!(dest.p, src.p))
copyto!(dest::BorderedVector{vectype, T}, src::BorderedVector{vectype, T}) where {vectype, T <: Number} = (copyto!(dest.u, src.u); dest.p = src.p)


length(b::BorderedVector{vectype, T}) where {vectype, T} = length(b.u) + length(b.p)

dot(a::BorderedVector{vectype, T}, b::BorderedVector{vectype, T}) where {vectype, T} = dot(a.u, b.u) + dot(a.p, b.p)

norm(b::BorderedVector{vectype, T}, p::Real) where {vectype, T} = max(norm(b.u, p), norm(b.p, p))
################################################################################
function rmul!(A::BorderedVector{vectype, T}, b::T) where {vectype, T <:Real}
	# Scale an array A by a scalar b overwriting A in-place
	rmul!(A.u, b)
	A.p = A.p * b
end
################################################################################
function axpy!(a::T, X::BorderedVector{vectype, T}, Y::BorderedVector{vectype, T}) where {vectype, T <:Real}
	# Overwrite Y with a*X + Y, where a is a scalar
	axpy!(a, X.u, Y.u)
	Y.p = a * X.p + Y.p
end
################################################################################
# this function is actually axpy!(-1, y, x)
minus!(x, y) = (x .= x .- y) # necessary to put a dot .= for ApproxFun to work
minus!(x::AbstractArray, y::AbstractArray) = (x .= x .- y)
minus!(x::T, y::T) where {T <:Real} = (x = x - y)
function minus!(x::BorderedVector{vectype, T},y::BorderedVector{vectype, T}) where {vectype, T <: Real}
	minus!(x.u, y.u)
	# Carefull here. If I uncomment the line below, then x.p will be left unaffected
	# minus_!(x.p, y.p)
	x.p = x.p - y.p
end
################################################################################
function dottheta(u1, u2, p1::T, p2::T, theta::T) where T
	return dot(u1, u2) * theta / length(u1) + p1 * p2 * (1 - theta)
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
################################################################################
"""
This function extract the jacobian of the bordered system. This is helpful when using Sparse Matrices. Indeed, solving the bordered system requires computing two inverses in the general case. Here by augmenting the sparse Jacobian, there is only one inverse to be computed.
It requires the state space to be Vector like.
"""
function getBorderedLinearSystemFull(J, dR::AbstractVector, tau::BorderedVector{vectype, T}, theta::T) where {vectype, T}
	N = length(tau.u)
	A = spzeros(N+1, N+1)
	A[1:N, 1:N] .= J
	A[1:N, end] .= dR
	A[end, 1:N] .= tau.u .* theta/length(tau.u)
	A[end, end]  = tau.p * (1-theta)
	return A
end
################################################################################
# solve in dX, dl
# J  * dX + a * dl = R
# b' * dX + c * dl = n
function linearBorderedSolver(J, a, b, c::T, R, n::T,
							solver::S)  where {vectype, T, S <: LinearSolver}
		x1, _, it1 = solver(J, R)
		x2, _, it2 = solver(J, a)

		dl = (n - dot(b, x1)) / (c - dot(b, x2))
		dX = x1 .- dl .* x2

		return dX, dl, (it1, it2)
end


# solve in dX, dl
# J  * dX + a * dR = R
# dz.u' * dX + dz.p * dl = n
function linearBorderedSolver(J, dR,
							dz::BorderedVector{vectype, T}, R, n::T, theta::T, solver::S;
							algo=:bordering)  where {T, vectype, S <: LinearSolver}
	# for debugging purposes, we keep a version using finite differences
	if algo == :full
		Aarc = getBorderedLinearSystemFull(J, dR, dz, theta)
		res = Aarc \ vcat(R, n)
		return res[1:end-1], res[end], 1
	end

	if algo == :bordering
		xiu = theta / length(dz.u)
		xip = (1-theta)

		x1, _, it1 = solver(J,  R)
		x2, _, it2 = solver(J, dR)

		dl = (n - dot(dz.u, x1) * xiu) / (dz.p * xip - dot(dz.u, x2) * xiu)
		dX = x1 - dl * x2

		return dX, dl, (it1, it2)
	end
	error("--> Algorithm $algo for Bordered Linear Systems is not implemented")
end
