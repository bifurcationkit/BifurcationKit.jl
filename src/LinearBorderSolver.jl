# structure for Bordered vectors
import Base: +, -, *, copy, copyto!
import LinearAlgebra: norm

mutable struct BorderedVector{vectype, T}
	u::vectype
	p::T
end
copy(b::BorderedVector{vectype, T}) where {vectype, T} = BorderedVector(copy(b.u), b.p)
norm(b::BorderedVector{vectype, T}) where {vectype, T} = sqrt(norm(b.u)^2 + b.p^2)
copyto!(dest::BorderedVector{vectype, T}, src::BorderedVector{vectype, T}) where {vectype, T <: Number} = (copyto!(dest.u, src.u);dest.p = src.p)
################################################################################
minus_!(x,y) = (x .= x .- y)
function minus_!(x::BorderedVector{vectype, T},y::BorderedVector{vectype, T}) where {vectype, T <: Real }
	x.u .= x.u .- y.u
	x.p = x.p - y.p
end
################################################################################
function (*)(a::Real, cp::BorderedVector{vectype, T}) where {vectype, T}
     newx = a .* (cp.u)
     newy = a .* (cp.p)
     result = BorderedVector(newx, newy)
     return result
end
################################################################################
function (-)(a::M, b::M) where {vectype, T, M <: BorderedVector{vectype, T}}
    newx = a.u - b.u
    newy = a.p - b.p
    result  = BorderedVector(newx, newy)
    return result
end
################################################################################
function (+)(a::M, b::M) where {vectype, T, M <: BorderedVector{vectype, T}}
    newx = a.u + b.u
    newy = a.p + b.p
    result  = BorderedVector(newx, newy)
    return result
end
################################################################################
function dottheta(u1, u2, p1::T, p2::T, theta::T) where T
	return dot(u1, u2) * theta/length(u1) + p1*p2*(1-theta)
	# return dot(u1, u2) * xi^2/length(u1) + p1*p2
end
################################################################################
function normtheta(u, p::T, theta::T) where T
	return sqrt(dottheta(u, u, p, p, theta))
end
################################################################################
function dottheta(a::M, b::M, theta::T) where {vectype, T, M <: BorderedVector{vectype, T}}
	return dotxi(a.u, b.u, a.p, b.p, theta)
end
################################################################################
function normtheta(a::M, theta::T) where {vectype, T, M <: BorderedVector{vectype, T}}
	return normtheta(a.u, a.p, theta)
end
################################################################################
"""
This function extract the jacobian of the bordered system. Mainly for debugging purposes.
"""
function getJacArcLength(J, dR, tau::M, theta::T) where {vectype, T, M <: BorderedVector{vectype, T}}
	N = length(tau.u)
	A = spzeros(N+1, N+1)
	A[1:N, 1:N] .= J
	A[1:N, end] .= dR
	A[end, 1:N] .= tau.u .* theta/length(tau.u)
	A[end, end]  = tau.p * (1-theta)
	return A
end
################################################################################
"""
Compute the tangent in the case of continuation with tangent prediction from bordered system
"""
function getTangent(J, dR, tau::M, theta::T, solver::S) where {vectype, T, M <: BorderedVector{vectype, T}, S <: LinearSolver}
	N = length(tau.u)
	A = getJacArcLength(J, dR, tau, theta)
	out = A \ vcat(zeros(N), 1)
	return BorderedVector(out[1:end-1], out[end])
end
################################################################################
function fullBorderedJacobian(out, v, J0, dR::Vector, tau::M, theta::T) where {vectype, T, M <: BorderedVector{vectype, T}}
	out_ = @view out[1:end-1]
	v_   = @view v[1:end-1]
	out_ .= J0 * v_
	out_ .= out_ .+ dR * v[end]
	out[end] = dot(tau.u, v_) * theta/length(tau.u) +
				tau.p * (1-theta) * v[end]
	return
end
################################################################################
# solve in dX, dl
# J  * dX + a * dl = R
# b' * dX + c * dl = n
function linearBorderedSolver(J, a, b, c::T, R, n::T,
							solver::S)  where {vectype, T, M <: BorderedVector{vectype, T}, S <: LinearSolver}
		x1, _, it1 = solver(J, R)
		x2, _, it2 = solver(J, a)

		dl = (n - dot(b, x1)) / (c - dot(b, x2))
		dX = x1 .- dl .* x2

		return dX, dl, (it1, it2)
end

function linearBorderedSolver(J, dR,
							dz::M, R, n::T, theta::T;
							tol=1e-12, maxiter=100,
							solver::S, algo=:bordering)  where {T, vectype, M <: BorderedVector{vectype, T}, S <: LinearSolver}
	# for debugging purposes, we keep a version using finite differences
	if algo == :full
		Aarc = getJacArcLength(J, dR, dz, theta)
		res = Aarc \ vcat(R, n)
		return res[1:end-1], res[end], 1
	end

	if algo == :bordering
		xiu = theta / length(dz.u)
		xip = (1-theta)

		x1, _, it1 = solver(J,  R)
		x2, _, it2 = solver(J, dR)

		dl = (n - dot(dz.u, x1) * xiu) / (dz.p * xip - dot(dz.u, x2) * xiu)
		dX = x1 .- dl .* x2

		return dX, dl, (it1, it2)
	end	
	error("--> Algorithm $algo for Bordered Linear Systems is not implemented")
end
