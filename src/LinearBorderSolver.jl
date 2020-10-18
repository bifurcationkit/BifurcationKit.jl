abstract type AbstractBorderedLinearSolver <: AbstractLinearSolver end

# call for using BorderedArray input, specific to Arclength Continuation
(lbs::AbstractBorderedLinearSolver)(J, dR::vec1, dz::vec2, R, n::T, theta::T; shift::Ts = nothing) where {vec1, vec2, T, Ts} = (lbs)(J, dR, dz.u, dz.p, R, n, theta / length(dz.u), one(T) - theta; shift = shift)

####################################################################################################
"""
$(TYPEDEF)

This struct is used to  provide the bordered linear solver based on the Bordering Method. It works
$(TYPEDFIELDS)
"""
@with_kw struct BorderingBLS{S <: AbstractLinearSolver, Ttol} <: AbstractBorderedLinearSolver
	"Linear solver used for the Bordering method."
	solver::S = DefaultLS()
	"Tolerance for checking precision"
	tol::Ttol = 1e-12
	"Check precision of the linear solve?"
	checkPrecision::Bool = false
end

# dummy constructor to simplify user passing options to continuation
BorderingBLS(ls::AbstractLinearSolver) = BorderingBLS(solver = ls)

# solve in dX, dl
#          J  * dX +       dR   * dl = R
# xiu * dz.u' * dX + xip * dz.p * dl = n
function (lbs::BorderingBLS{S, Ttol})(  J, dR,
								dzu, dzp::T, R, n::T,
								xiu::T = T(1), xip::T = T(1); shift::Ts = nothing)  where {T, S, Ts, Ttol}
	# the following parameters are used for the pseudo arc length continuation
	# xiu = theta / length(dz.u)
	# xip = one(T) - theta

	# we make this branching to avoid applying a zero shift
	if isnothing(shift)
		x1, x2, _, (it1, it2) = lbs.solver(J, R, dR)
	else
		x1, x2, _, (it1, it2) = lbs.solver(J, R, dR; a₀ = shift)
	end

	dl = (n - dot(dzu, x1) * xiu) / (dzp * xip - dot(dzu, x2) * xiu)

	# dX = x1 .- dl .* x2
	axpy!(-dl, x2, x1)

	# we check the precision of the solution by the bordering algorithm
	# mainly for debugging purposes
	if lbs.checkPrecision
		# at this point, x2 is not used anymore, we can use it for computing the residual
		# hence x2 = J*x1 + dl*dR - R
		x2 = apply(J, x1)
		axpy!(dl, dR, x2)
		axpy!(-1, R, x2)

		printstyled(color=:red,"--> res = ", ( norm(x2), abs(n - xip*dzp*dl -xiu* dot(dzu, x1))), "\n")
	end

	return x1, dl, true, (it1, it2)
end
####################################################################################################
# this interface should work for Sparse Matrices as well as for Matrices
"""
$(TYPEDEF)
This struct is used to  provide the bordered linear solver based on inverting the full matrix.
$(TYPEDFIELDS)
"""
struct MatrixBLS{S <: AbstractLinearSolver} <: AbstractBorderedLinearSolver
	"Linear solver used to invert the full matrix."
	solver::S
end

# dummy constructor to simplify user passing options to continuation
MatrixBLS() = MatrixBLS(DefaultLS())

# case of a scalar additional linear equation
function (lbs::MatrixBLS)(J, dR,
						dzu, dzp::T, R::vectype, n::T,
						xiu::T = T(1), xip::T = T(1); shift::Ts = nothing)  where {T <: Number, vectype <: AbstractVector, S, Ts}

	if isnothing(shift)
		A = J
	else
		A = J + shift * I
	end

	A = hcat(A, dR)
	A = vcat(A, vcat(vec(dzu) .* xiu, dzp * xip)')

	# solve the equations and return the result
	rhs = vcat(R, n)
	res = A \ rhs
	return res[1:end-1], res[end], true, 1
end

####################################################################################################
# composite type to save the bordered linear system with expression
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
	# This implements the case where Tc is a number, ie there is one scalar constraint in the
	# bordered linear system
	out = similar(x)
	copyto!(out.u, apply(lbmap.J, x.u))
	axpy!(x.p, lbmap.a, out.u)
	out.p = dot(lbmap.b, x.u)  + lbmap.c  * x.p
	return out
end

function (lbmap::MatrixFreeBLSmap{Tj, Ta, Tb, Tc})(x::AbstractArray) where {Tj, Ta <: AbstractArray, Tb, Tc <: Number}
	# This implements the case where Tc is a number, ie there is one scalar constraint in the
	# bordered linear system
	out = similar(x)
	xu = @view x[1:end-1]
	xp = x[end]
	# copyto!(out.u, apply(lbmap.J, x.u))
	out[1:end-1] .= @views apply(lbmap.J, xu) .+ xp .* lbmap.a
	out[end] = @views dot(lbmap.b, xu)  + lbmap.c  * xp
	return out
end

"""
$(TYPEDEF)
This struct is used to  provide the bordered linear solver based a matrix free operator for the full system in `(x,p)`.
$(TYPEDFIELDS)
"""
struct MatrixFreeBLS{S} <: AbstractBorderedLinearSolver
	solver::S
end

# dummy constructor to simplify user passing options to continuation
MatrixFreeBLS() = MatrixFreeBLS(DefaultLS())

extractVector(x::AbstractVector) = @view x[1:end-1]
extractVector(x::BorderedArray) = x.u

extractParameter(x::AbstractVector) = x[end]
extractParameter(x::BorderedArray) = x.p

# We restrict to bordered systems where the added component is scalar
function (lbs::MatrixFreeBLS{S})(J, 		dR,
								dzu, 	dzp::T, R, n::T,
								xiu::T = T(1), xip::T = T(1); shift::Ts = nothing) where {T <: Number, S, Ts}
	linearmap = MatrixFreeBLSmap(J, dR, rmul!(copy(dzu), xiu), dzp * xip)
	if S <: GMRESIterativeSolvers
		rhs = vcat(R, n)
	else
		rhs = BorderedArray(copy(R), n)
	end
	sol, cv, it = lbs.solver(linearmap, rhs)
	return extractVector(sol), extractParameter(sol), cv, it
end
