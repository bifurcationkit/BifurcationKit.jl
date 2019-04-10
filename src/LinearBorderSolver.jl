################################################################################
# struct borderedLinearOperator{Tj, Ta, Tb, Tc}
# 	J::Tj
# 	a::Ta
# 	b::Tb
# 	c::Tc
# end
#
# function (Lb::borderedLinearOperator{Tj, Ta, Tb, Tc})(x::BorderedVector{Ta, Tc}) where {Tj, Ta, Tb, Tc, Tc <: Number}
# 	out = similar(x)
# 	out.u .= apply(Lb.J, x.u) .+ Lb.a .* x.p
# 	out.p = dot(Lb.b, x.u) + Lb.c * x.p
# 	return out
# end
# abstract type LinearBorderSolver <: LinearSolver end
#
# struct BorderingLS <: LinearBorderSolver end
#
# function (lbs::BorderingLS)(J::borderedLinearOperator{Tj, Ta, Tb, Tc},
# 									rhs::BorderedVector{Ta, Tc})
#
# end
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
	A[end, end]  = tau.p * (one(T)-theta)
	return A
end
################################################################################
# solve in dX, dl
# J  * dX + a * dl = R
# b' * dX + c * dl = n
function linearBorderedSolver(J, a, b, c::T, R, n::T, solver::S; shift::Ts = 0.0)  where {vectype, T, S <: LinearSolver, Ts <: Number}
		x1, _, it1 = solver(J, R, shift)
		x2, _, it2 = solver(J, a, shift)

		dl = (n - dot(b, x1)) / (c - dot(b, x2))
		# dX = x1 .- dl .* x2
		dX = copy(x1); axpy!(-dl, x2, dX)

		return dX, dl, (it1, it2)
end
################################################################################
# solve in dX, dl
# J  * dX + a * dR = R
# dz.u' * dX + dz.p * dl = n
# The following function is essentially used by newtonPseudoArcLength
function linearBorderedSolver(J, dR,
							dz::BorderedVector{vectype, T}, R, n::T, theta::T, solver::S;
							algo=:bordering)  where {T, vectype, S <: LinearSolver}
	# for debugging purposes, we keep a version using finite differences
	if algo == :full
		Aarc = getBorderedLinearSystemFull(J, dR, dz, theta)
		res = Aarc \ vcat(R, n)
		return res[1:end-1], res[end], 1

	elseif algo == :fullMatrixFree
		@assert 1==0 "WIP"
		bordedOp = borderedLinearOperator(J, dR, dz.u .* theta/length(dz.u), dz.p * (one(T)-theta))
		reslinear, _, it = solver(bordedOp,  BorderedVector(R, n))
		return reslinear.u, reslinear.p, it

	elseif algo == :bordering
		xiu = theta / length(dz.u)
		xip = one(T) - theta

		x1, _, it1 = solver(J,  R)
		x2, _, it2 = solver(J, dR)

		dl = (n - dot(dz.u, x1) * xiu) / (dz.p * xip - dot(dz.u, x2) * xiu)
		# dX = x1 .- dl .* x2
		dX = copy(x1); axpy!(-dl, x2, dX)

		return dX, dl, (it1, it2)
	end
	error("--> Algorithm $algo for Bordered Linear Systems is not implemented")
end
