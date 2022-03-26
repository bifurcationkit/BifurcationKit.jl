abstract type abstractModulatedWaveFD <: AbstractPOFDProblem end
abstract type abstractModulatedWaveShooting <: AbstractShootingProblem end

"""
	pb = TWProblem(F, J, ∂::Tuple, u₀; DAE = 0)

This composite type implements a functional for freezing symmetries in order, for example, to compute travelling waves (TW). Note that you can freeze many symmetries, not just one, by passing many Lie generators. When you call `pb(x, par)`, it computes:

					┌                  ┐
					│ f(x, par) - s⋅∂x │
					│   <x - u₀, ∂u₀>  │
					└                  ┘

## Arguments
- `(x, p) -> F(x, p)` function with continuous symmetries
- `J` jacobian of `F`. Can be matrix based or matrix-free. The requirements are same as for [`newton`](@ref)
- `∂::Tuple = (T1, T2, ⋯)` tuple of Lie generators. In effect, each of these is an (differential) operator which can be specified as a (sparse) matrix or as an operator implementing `LinearAlgebra.mul!`.
- `u₀` reference solution

## Additional Constructor(s)

	pb = TWProblem(F, J, ∂, u₀; kw...)

This simplified call handles the case where a single symmetry needs to be frozen.

## Useful function

- `updateSection!(pb::TWProblem, u0)` updates the reference solution of the problem using `u0`.
- `nbConstraints(::TWProblem)` number of constraints (or Lie generators)

"""
@with_kw struct TWProblem{Tf, TJf, Tu0, TDu0, TD}
	F::Tf
	J::TJf = nothing
	∂::TD			# differential operator
	u₀::Tu0 		# reference solution, we only need one!
	∂u₀::TDu0 = (D * u₀,)
	DAE::Int = 0
	nc::Int = 1 	# number of constraints
	isinplace::Bool = false
	@assert 0 <= DAE <= 1
	@assert 0 < nc
end

function TWProblem(F, J, ∂::Tuple, u₀; DAE = 0, isinplace = false)
	# ∂u₀ = Tuple( apply(_D, u₀) for _D in ∂)
	∂u₀ = Tuple( mul!(zero(u₀), _D, u₀, 1, 0) for _D in ∂)
	return TWProblem(F = F, J = J, ∂ = ∂,
		u₀ = u₀,
		∂u₀ = ∂u₀,
		# u₀∂u₀ = Tuple( dot(u₀, u) for u in ∂u₀),
		DAE = DAE,
		isinplace = isinplace,
		nc = length(∂) )
end

# constructor
TWProblem(F, J, ∂, u₀; kw...) = TWProblem(F, J, (∂,), u₀; kw...)

@inline nbConstraints(pb::TWProblem) = pb.nc

# we put type information to ensure the user pass a correct u0
function updateSection!(pb::TWProblem{Tf, TJf, Tu0, TDu0, TD}, u₀::Tu0) where {Tf, TJf, Tu0, TDu0, TD}
	copyto!(pb.u₀, u₀)
	for (∂, ∂u₀) in zip(pb.∂, pb.∂u₀)
		# pb.u₀∂u₀ = Tuple( dot(u₀, u) for u in ∂u₀)
		copyto!(∂u₀, ∂ * u₀)
	end
end

"""
- `ss` tuple of speeds
- `D` tuple of Lie generators
"""
function applyD(pb::TWProblem, out, ss, u)
	for (D, s) in zip(pb.∂, ss)
		# out .-=  s .* (D * u)
		mul!(out, D, u, -s, 1)
	end
	out
end
applyD(pb::TWProblem, u) = applyD(pb, zero(u), 1, u)

# s is the speed.
# Return F(u, p) - s * D * u
@views function VFplusD(pb::TWProblem, u::AbstractVector, s::Tuple, pars)
	# apply the vector field
	out = pb.F(u, pars)
	# we add the freezing, it can be done now since out is filled by the previous call!!
	applyD(pb, out, s, u)
	return out
end

# function (u, p) -> F(u, p) - s * D * u to be used with shooting or Trapezoid
VFtw(pb::TWProblem, u::AbstractVector, parsFreez) = VFplusD(pb, u, parsFreez.s, parsFreez.user)

@views function (pb::TWProblem)(x::AbstractVector, pars)
	# number of constraints
	nc = pb.nc
	# number of unknowns
	N = length(x) - nc
	# array containing the result
	out = similar(x)
	u = x[1:N]
	outu = out[1:N]
	# get the speed
	s = Tuple(x[end-nc+1:end])
	# apply the vector field
	outu .= VFplusD(pb, u, s, pars)
	# we put the constraints
	for ii in 0:nc-1
		out[end-ii] = dot(u, pb.∂u₀[ii+1])
		if pb.DAE == 0
			out[end-ii] -= dot(pb.u₀, pb.∂u₀[ii+1])
		end
	end
	return out
end

@views function (pb::TWProblem)(x::AbstractVector, pars, dx::AbstractVector)
	# number of constraints
	nc = pb.nc
	# number of unknowns
	N = length(x) - nc
	# array containing the result
	out = similar(x)
	u = x[1:N]
	du = dx[1:N]
	outu = out[1:N]
	# get the speed
	s = Tuple(x[end-nc+1:end])
	ds = Tuple(dx[end-nc+1:end])
	# get the jacobian
	J = pb.J(u, pars)
	outu .= apply(J, du)
	applyD(pb, outu, s, du)
	applyD(pb, outu, ds, u)
	# we put the constraints
	for ii in 0:nc-1
		out[end-ii] = dot(du, pb.∂u₀[ii+1])
	end
	return out
end

# build the sparse jacobian of the freezed problem
function (pb::TWProblem)(::Val{:JacFullSparse}, ufreez::AbstractVector, par; δ = 1e-9)
	# number of constraints
	nc = nbConstraints(pb)
	# number of unknowns
	N = length(ufreez) - nc
	# get the speed
	s = Tuple(ufreez[end-nc+1:end])
	# get the state space vector
	u = ufreez[1:N]
	# the jacobian of the
	J1 = pb.J(u, par)
	# we add the Lie algebra generators
	rightpart = zeros(N, nc)
	for ii in 1:nc
		J1 = J1 - s[ii] * pb.∂[ii]
		mul!(view(rightpart, :, ii), pb.∂[ii], u, -1, 0)
	end
	J2 = hcat(J1, rightpart)
	for ii in 1:nc
		J2 = vcat(J2, vcat(pb.∂u₀[ii], zeros(nc))')
	end
	return J2
end

function newton(prob::TWProblem, orbitguess, par, optn::NewtonPar;
		jacobian = :MatrixFree, kwargs...)
	@assert jacobian in  (:MatrixFree, :MatrixFreeAD, :AutoDiff, :FullLU)
	if jacobian == :AutoDiff
		jac = (x, p) -> sparse(ForwardDiff.jacobian(z -> prob(z, p), x))
	elseif jacobian == :MatrixFreeAD
		jac = (x, p) -> (dx -> ForwardDiff.derivative(t -> prob(x .+ t .* dx, p), 0))
	elseif jacobian == :FullLU
		jac = (x, p) -> prob(Val(:JacFullSparse), x, p)
	# elseif jacobian == :FullSparseInplace
	elseif jacobian == :MatrixFree
		jac = (x, p) -> (dx ->  prob(x, p, dx))
	end
	return newton(prob, jac, orbitguess, par, optn; kwargs...,)
end

function continuation(prob::TWProblem,
		orbitguess, par, lens::Lens, contParams::ContinuationPar;
		jacobian = :MatrixFree, kwargs...)
	@assert jacobian in (:MatrixFree, :MatrixFreeAD, :AutoDiff, :FullLU)

	if jacobian == :AutoDiff
		jac = (x, p) -> sparse(ForwardDiff.jacobian(z -> prob(z, p), x))
	elseif jacobian == :MatrixFreeAD
		jac = (x, p) -> (dx -> ForwardDiff.derivative(t -> prob(x .+ t .* dx, p), 0))
	elseif jacobian == :FullLU
		jac = (x, p) -> prob(Val(:JacFullSparse), x, p)
	# elseif jacobian == :FullSparseInplace
	elseif jacobian == :MatrixFree
		jac = (x, p) -> (dx ->  prob(x, p, dx))
	end
	# define the mass matrix for the eigensolver
	N = length(orbitguess)
	B = Diagonal(vcat(ones(N-1),0))
	# convert eigsolver to generalised one
	old_eigsolver = contParams.newtonOptions.eigsolver
	contParamsWave = @set contParams.newtonOptions.eigsolver = convertToGEV(old_eigsolver, B)
	# call continuation
	branch, u, τ = continuation(prob, jac, orbitguess, par, lens, contParamsWave; kwargs...,)
	return setproperties(branch; type = :TravellingWave, functional = prob), u, τ
end
