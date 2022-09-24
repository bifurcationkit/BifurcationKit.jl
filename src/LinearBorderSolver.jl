abstract type AbstractBorderedLinearSolver <: AbstractLinearSolver end

# the following stuctures, say `struct BDLS;...;end` rely on the hypotheses:
# - the constructor must provide BDLS() and BDLS(::AbstractLinearSolver)
# - the method (ls::BDLS)(J, dR, dzu, dzp, R, n, ξu, ξp; shift = nothing) must be provided


# Reminder we want to solve the linear system
# Cramer's rule gives σ = det(J) / det(M)
#     ┌        ┐┌   ┐   ┌ ┐
# M = │ J    b ││ v │ = │0│
#     │ c'   d ││ σ │   │1│
#     └        ┘└   ┘   └ ┘
####################################################################################################
"""
$(TYPEDEF)

This struct is used to provide the bordered linear solver based on the Bordering Method. user the options, you can trigger a sequence of Bordering reductions to meet a precision.

# Reference

This is the solver BEC + k in Govaerts, W. “Stable Solvers and Block Elimination for Bordered Systems.” SIAM Journal on Matrix Analysis and Applications 12, no. 3 (July 1, 1991): 469–83. https://doi.org/10.1137/0612034.

$(TYPEDFIELDS)

# Constructors

- there is a  simple constructor `BorderingBLS(ls)` where `ls` is a linear solver, for example `ls = DefaultLS()`
- you can use keyword argument to create such solver, for example `BorderingBLS(solver = DefaultLS(), tol = 1e-4)`

"""
@with_kw struct BorderingBLS{S <: Union{AbstractLinearSolver, Nothing}, Ttol} <: AbstractBorderedLinearSolver
	"Linear solver used for the Bordering method."
	solver::S = nothing

	"Tolerance for checking precision"
	tol::Ttol = 1e-12

	"Check precision of the linear solve?"
	checkPrecision::Bool = true

	"Number of recursions to achieve tolerance"
	k::Int64 = 1

	@assert k > 0 "Number of recursions must be positive"
end

BorderingBLS(ls::AbstractLinearSolver) = BorderingBLS(solver = ls)

# solve in dX, dl
# ┌                           ┐┌  ┐   ┌   ┐
# │ (shift⋅I + J)     dR      ││dX│ = │ R │
# │   ξu * dz.u'   ξp * dz.p  ││dl│   │ n │
# └                           ┘└  ┘   └   ┘
function (lbs::BorderingBLS)(  J, dR,
								dzu, dzp::T,
								R, n::T,
								ξu::Tξ = 1, ξp::Tξ = 1; dotp = dot, shift::Ts = nothing)  where {T, Tξ <: Number, Ts}
	# the following parameters are used for the pseudo arc length continuation
	# ξu = θ / length(dz.u)
	# ξp = 1 - θ

	k = 0 # number of BEC iterations
	BEC0(x, y) = BEC(lbs, J, dR, dzu, dzp, x, y, ξu, ξp; shift = shift, dotp = dotp)
	Residual(x, y) = residualBEC(lbs, J, dR, dzu, dzp, R, n, x, y, ξu, ξp; shift = shift, dotp = dotp)

	dX, dl, cv, itlinear = BEC0(R, n)

	failBLS = true
	while lbs.checkPrecision && k < lbs.k && failBLS
		δX, δl = Residual(dX, dl)
		failBLS = norm(δX) > lbs.tol || abs(δl) > lbs.tol
		@debug k, norm(δX), abs(δl)
		if failBLS
			dX1, dl1, cv, itlinear = BEC0(δX, δl)
			axpy!(1, dX1, dX)
			dl += dl1
			k += 1
		end
	end
	return dX, dl, cv, itlinear
end

function BEC(lbs::BorderingBLS,
							J, dR,
							dzu, dzp,
							R, n,
							ξu::Tξ = 1, ξp::Tξ = 1;
							shift::Ts = nothing, dotp = dot)  where {Tξ, Ts}
	if isnothing(shift)
		x1, δx, success, itlinear = lbs.solver(J, R, dR)
	else
		x1, δx, success, itlinear = lbs.solver(J, R, dR; a₀ = shift)
	end

	~success && @debug "Linear solver failed to converge in BorderingBLS."

	dl = (n - dotp(dzu, x1) * ξu) / (dzp * ξp - dotp(dzu, δx) * ξu)

	# dX = x1 .- dl .* δx
	axpy!(-dl, δx, x1)
	return x1, dl, success, itlinear
end

function residualBEC(lbs::BorderingBLS,
							J, dR,
							dzu, dzp,
							R, n,
							dX, dl,
							ξu::Tξ = 1, ξp::Tξ = 1;
							shift::Ts = nothing, dotp = dot)  where {Tξ, Ts}
	# we check the precision of the solution from the bordering algorithm
	# at this point, δx is not used anymore, we can use it for computing the residual
	# hence δx = R - (shift⋅I + J) * dX	 - dl * dR
	δX = apply(J, dX)
	if ~isnothing(shift)
		axpy!(shift, dX, δX)
	end
	axpy!(dl, dR, δX)
	axpby!(1, R, -1, δX)

	δl = n - ξp * dzp * dl - ξu * dotp(dzu, dX)

	return δX, δl
end

# version used in PALC
(lbs::BorderingBLS)(iter::AbstractContinuationIterable, state::AbstractContinuationState,
				J, dR, R, n::T; shift::Ts = nothing) where {T, Ts} =
	  					  (lbs)(J, dR,
								state.τ.u, state.τ.p,
								R, n,
								state.θ, one(T) - state.θ;
								shift = shift, dotp = iter.dotθ.dot)
####################################################################################################
"""
$(TYPEDEF)

This struct is used to  provide the bordered linear solver based on inverting the full matrix.

$(TYPEDFIELDS)
"""
struct MatrixBLS{S <: Union{AbstractLinearSolver, Nothing}} <: AbstractBorderedLinearSolver
	"Linear solver used to invert the full matrix."
	solver::S
end

# dummy constructor to simplify user passing options to continuation
MatrixBLS() = MatrixBLS(nothing)

# case of a scalar additional linear equation
function (lbs::MatrixBLS)(J, dR,
						dzu, dzp::T, R::AbstractVecOrMat, n::T,
						ξu::T = T(1), ξp::T = T(1);
						shift::Ts = nothing, applyξu! = nothing)  where {T <: Number, S, Ts}

	if isnothing(shift)
		A = J
	else
		A = J + shift * I
	end

	A = hcat(A, dR)
	A = vcat(A, hcat(adjoint(dzu .* ξu), dzp * ξp))

	# apply a linear operator to ξu
	if ~isnothing(applyξu!)
		applyξu!(@view(A[end, begin:end-1]))
	end

	# solve the equations and return the result
	rhs = vcat(R, n)
	res = A \ rhs
	return res[1:end-1], res[end], true, 1
end

# version used in PALC
(lbs::MatrixBLS)(iter::AbstractContinuationIterable, state::AbstractContinuationState,
 					J, dR, R, n::T; shift::Ts = nothing) where {T, Ts} =
	  					  (lbs)(J, dR,
								state.τ.u, state.τ.p,
								R, n,
								state.θ, one(T) - state.θ;
								shift = shift, applyξu! = iter.dotθ.apply!)
####################################################################################################
# composite type to save the bordered linear system with expression
# ┌         ┐
# │  J    a │
# │  b'   c │
# └         ┘
# It then solved using Matrix Free algorithm applied to the full operator and not just J as for MatrixFreeBLS
#
struct MatrixFreeBLSmap{Tj, Ta, Tb, Tc, Ts, Td}
	J::Tj
	a::Ta
	b::Tb
	c::Tc
	shift::Ts
	dot::Td # possibly custom dot product
end

function (lbmap::MatrixFreeBLSmap{Tj, Ta, Tb, Tc, Ts})(x::BorderedArray{Ta, Tc}) where {Tj, Ta, Tb, Tc <: Number, Ts}
	out = similar(x)
	copyto!(out.u, apply(lbmap.J, x.u))
	axpy!(x.p, lbmap.a, out.u)
	if isnothing(lbmap.shift) == false
		axpy!(lbmap.shift, x.u, out.u)
	end
	out.p = lbmap.dot(lbmap.b, x.u)  + lbmap.c  * x.p
	return out
end

function (lbmap::MatrixFreeBLSmap)(x::AbstractArray)
	# This implements the case where Tc is a number, ie there is one scalar constraint in the
	# bordered linear system
	out = similar(x)
	xu = @view x[1:end-1]
	xp = x[end]
	# copyto!(out.u, apply(lbmap.J, x.u))
	if isnothing(lbmap.shift)
		out[1:end-1] .= apply(lbmap.J, xu) .+ xp .* lbmap.a
	else
		out[1:end-1] .= apply(lbmap.J, xu) .+ xp .* lbmap.a .+ xu .* lbmap.shift
	end
	out[end] = lbmap.dot(lbmap.b, xu)  + lbmap.c  * xp
	return out
end

"""
$(TYPEDEF)

This struct is used to  provide the bordered linear solver based a matrix free operator for the full system in `(x, p)`.

$(TYPEDFIELDS)
"""
struct MatrixFreeBLS{S <: Union{AbstractLinearSolver, Nothing}} <: AbstractBorderedLinearSolver
	"Linear solver used to solve the extended linear system"
	solver::S
	"What is the structure used to hold `(x, p)`. If `true`, this is achieved using `BorderedArray`. If `false`, a `Vector` is used."
	useBorderedArray::Bool
end

# dummy constructor to simplify user passing options to continuation
MatrixFreeBLS(useBorderedArray::Bool = true) = MatrixFreeBLS(nothing, useBorderedArray)
MatrixFreeBLS(::Nothing) = MatrixFreeBLS()
MatrixFreeBLS(S::AbstractLinearSolver) = MatrixFreeBLS(S, ~(S isa GMRESIterativeSolvers))

extractVecBLS(x::AbstractVector) = @view x[1:end-1]
extractVecBLS(x::BorderedArray)  = x.u

extractParBLS(x::AbstractVector) = x[end]
extractParBLS(x::BorderedArray)  = x.p

# We restrict to bordered systems where the added component is scalar
function (lbs::MatrixFreeBLS{S})(J, 	dR,
								dzu, 	dzp::T, R, n::T,
								ξu::Tξ = 1, ξp::Tξ = 1; shift = nothing, dotp = dot) where {T <: Number, Tξ, S}
	~isnothing(shift) && @warn "Shift is not implemented for the bordered linear solver MatrixFreeBLS"
	linearmap = MatrixFreeBLSmap(J, dR, rmul!(copy(dzu), ξu), dzp * ξp, shift, dotp)
	rhs = lbs.useBorderedArray ? BorderedArray(copy(R), n) : vcat(R, n)
	sol, cv, it = lbs.solver(linearmap, rhs)
	return extractVecBLS(sol), extractParBLS(sol), cv, it
end

# version used in PALC
(lbs::MatrixFreeBLS)(iter::AbstractContinuationIterable, state::AbstractContinuationState,
				J, dR, R, n::T; shift::Ts = nothing) where {T, Ts} =
	  					  (lbs)(J, dR,
								state.τ.u, state.τ.p,
								R, n,
								state.θ, one(T) - state.θ;
								shift = shift, dotp = iter.dotθ.dot)
####################################################################################################
# Linear Solvers based on a bordered solver
# !!!! This one is used as a linear Solver, not as a Bordered one
####################################################################################################
"""
$(TYPEDEF)

This structure is used to provide the following linear solver. To solve (1) J⋅x = rhs, one decomposes J using Matrix by blocks and then use a bordering strategy to solve (1).

$(TYPEDFIELDS)

!!! warn "Warning"
    The solver only works for `AbstractMatrix`
"""
struct LSFromBLS{Ts} <: AbstractLinearSolver
	"Linear solver used to solve the smaller linear systems."
	solver::Ts
end

LSFromBLS() = LSFromBLS(BorderingBLS(solver = DefaultLS(useFactorization = false), checkPrecision = false))

function (l::LSFromBLS)(J, rhs)
	F = factorize(J[1:end-1, 1:end-1])
	x1, x2, flag, it = l.solver(F, Array(J[1:end-1,end]), Array(J[end,1:end-1]), J[end, end], (@view rhs[1:end-1]), rhs[end])
	return vcat(x1, x2), flag, sum(it)
end

function  (l::LSFromBLS)(J, rhs1, rhs2)
	F = factorize(J[1:end-1,1:end-1])
	x1, x2, flag1, it1 = l.solver(F, Array(J[1:end-1,end]), Array(J[end,1:end-1]), J[end, end], (@view rhs1[1:end-1]), rhs1[end])

	y1, y2, flag2, it2 = l.solver(F, Array(J[1:end-1,end]), Array(J[end,1:end-1]), J[end, end], (@view rhs2[1:end-1]), rhs2[end])

	return vcat(x1, x2), vcat(y1, y2), flag1 & flag2, (1, 1)
end
