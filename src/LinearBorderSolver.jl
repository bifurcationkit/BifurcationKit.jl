include("LinearSolver.jl")

abstract type AbstractBorderedLinearSolver <: AbstractLinearSolver end
####################################################################################################
struct BorderingBLS{S <: AbstractLinearSolver} <: AbstractBorderedLinearSolver
	solver::S
end

# dummy constructor to simplify user passing options to continuation
BorderingBLS() = BorderingBLS(Default())

# solve in dX, dl
#          J  * dX +       dR   * dl = R
# xiu * dz.u' * dX + xip * dz.p * dl = n
function (lbs::BorderingBLS{S})(J, dR, dzu, dzp::T, R, n::T, xiu::T = T(1), xip::T = T(1); shift::Ts = 0)  where {T, S, Ts <: Number}
	# the following parameters are used for the pseudo arc length continuation
	# xiu = theta / length(dz.u)
	# xip = one(T) - theta

	# we make this branching to avoid applying a zero shift
	if shift == 0
		x1, _, it1 = lbs.solver(J,  R)
		x2, _, it2 = lbs.solver(J, dR)
	else
		x1, _, it1 = lbs.solver(J,  R, shift)
		x2, _, it2 = lbs.solver(J, dR, shift)
	end

	dl = (n - dot(dzu, x1) * xiu) / (dzp * xip - dot(dzu, x2) * xiu)
	# dX = x1 .- dl .* x2
	dX = copy(x1); axpy!(-dl, x2, dX)

	return dX, dl, (it1, it2)
end

# call for using BorderedArray input, specific to Arclength Continuation
(lbs::BorderingBLS{S})(J, dR::vec1, dz::BorderedArray{vec2, T}, R, n::T, theta::T; shift::Ts = 0) where {S, T, Ts, vec1, vec2} = (lbs)(J, dR, dz.u, dz.p, R, n, theta / length(dz.u), one(T) - theta; shift = shift)
####################################################################################################
# this interface should work for Sparse Matrices as well as for Matrices
struct MatrixBLS{S <: AbstractLinearSolver} <: AbstractBorderedLinearSolver
	solver::S
end

# dummy constructor to simplify user passing options to continuation
MatrixBLS() = MatrixBLS(Default())

function (lbs::MatrixBLS)(J, dR, dzu, dzp::T, R::vectype, n::T, xiu::T = T(1), xip::T = T(1); shift::Ts = 0)  where {T, vectype <: AbstractVector, S, Ts <: Number}
	N = length(dzu)

	rhs = vcat(R, n)
	A = similar(J, N+1, N+1)

	A[1:N, 1:N] .= J + shift * I
	A[1:N, end] .= dR
	A[end, 1:N] .= dzu .* xiu
	A[end, end]  = dzp * xip
	res = A \ rhs
	return res[1:end-1], res[end], true, 1
end

# call for using BorderedArray input, specific to Arclength Continuation
(lbs::MatrixBLS{S})(J, dR::vec1, dz::BorderedArray{vec2, T}, R, n::T, theta::T; shift::Ts = 0) where {S, T, Ts, vec1, vec2} = (lbs)(J, dR, dz.u, dz.p, R, n, theta / length(dz.u), one(T) - theta; shift = shift)
####################################################################################################
# structure to save the bordered linear system with expression
# [ J	a]
# [b'	c]
# It then solved using Matrix Free algorithm applied to the full operator and not just J as for MatrixFreeBLS
#
struct MatrixFreeBLSmap{Tj, Ta, Tb, Tc}
	J::Tj
	a::Ta
	b::Tb
	c::Tc
end

function (lbmap::MatrixFreeBLSmap{Tj, Ta, Tb, Tc})(x::BorderedArray{Ta, Tc}) where {Tj, Ta, Tb, Tc <: Number}
	out = similar(x)
	copyto!(out.u, apply(lbmap.J, x.u) .+ lbmap.a .* x.p)
	out.p  =   dot(lbmap.b, x.u)  + lbmap.c  * x.p
	return out
end

struct MatrixFreeBLS{S} <: AbstractBorderedLinearSolver
	solver::S
end

MatrixFreeBLS() = MatrixFreeBLS(Default())

# dummy constructor to simplify user passing options to continuation
# We restrict to bordered systems where the added component is scalar
# For now, we restrict to KrylovKit iterative solver
function (lbs::MatrixFreeBLS{S})(J, dR::vectype,
					dzu::vectype, dzp::T,
					R, n::T,
					xiu::T = T(1), xip::T = T(1); shift::Ts = 0) where {T <: Number, vectype, S <: GMRES_KrylovKit, Ts <: Number}
	linearmap = MatrixFreeBLSmap(J, dR, dzu * xiu, dzp * xip)
	rhs = BorderedArray(R, n)
	sol, cv, it = lbs.solver(linearmap, rhs)

end

# call for using BorderedArray input, specific to Arclength Continuation
(lbs::MatrixFreeBLS{S})(J, dR::vec1, dz::BorderedArray{vec2, T}, R, n::T, theta::T; shift::Ts = 0) where {S, T <: Number, Ts, vec1, vec2} = (lbs)(J, dR, dz.u, dz.p, R, n, theta / length(dz.u), one(T) - theta; shift = shift)
####################################################################################################
# Nested algorithm for solving the bordered linear system
# NestedBLS
