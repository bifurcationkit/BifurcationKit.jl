using BlockArrays, SparseArrays, Setfield

# structure to describe a (Time) mesh using the time steps t_{i+1} - t_{i}. If the time steps are constant, we do not record them but, instead, we save the number of time steps
struct TimeMesh{T}
	ds::T
end

TimeMesh(M::Int64) = TimeMesh{Int64}(M)

@inline canAdapt(ms::TimeMesh{Ti}) where Ti = !(Ti == Int64)
Base.length(ms::TimeMesh{Ti}) where Ti = length(ms.ds)
Base.length(ms::TimeMesh{Ti}) where {Ti <: Int} = ms.ds

# access the time steps
@inline getTimeStep(ms, i::Int) = ms.ds[i]
@inline getTimeStep(ms::TimeMesh{Ti}, i::Int) where {Ti <: Int} = 1.0 / ms.ds

Base.collect(ms::TimeMesh) = ms.ds
Base.collect(ms::TimeMesh{Ti}) where {Ti <: Int} = repeat([getTimeStep(ms, 1)], ms.ds)

####################################################################################################
# method using the Trapezoidal rule (Order 2 in time) and discretisation of the periodic orbit.
"""
	pb = PeriodicOrbitTrapProblem(F, J, ϕ, xπ, M::Int)
This composite type implements Finite Differences based on a Trapezoidal rule to locate periodic orbits. The arguments are as follows
- `F(x,p)` vector field
- `J` jacobian of `F`. Can be matrix based `J(x,p)` or Matrix-Free
- `Jt = nothing` jacobian tranpose of `F` (optional), useful for continuation of Fold of periodic orbits. it should not be passed in case the jacobian is a (sparse) matrix as it is computed internally, and it would be computed twice in that case.
- `d2F = nothing` second derivative of F (optional), useful for continuation of Fold of periodic orbits. It has the definition `d2F(x,p,dx1,dx2)`.`
- `ϕ` used to set a section for the phase constraint equation
- `xπ` used in the section for the phase constraint equation
- `M::Int` number of time slices
- `linsolver: = DefaultLS()` linear solver for each time slice, i.e. to solve `J⋅sol = rhs`. This is only needed for the computation of the Floquet multipliers.
- `isinplace::Bool` whether `F` and `J` are inplace functions (Experimental). In this case, the functions `F` and `J` must have the following definitions `(o, x, p) ->  F(o, x, p)` and `(o, x, p, dx) -> J(o, x, p, dx)`.
- `ongpu::Bool` whether the computation takes place on the gpu (Experimental)

You can then call `pb(orbitguess, p)` to compute the functional on a `orbitguess`. Note that `orbitguess` must be of size M * N + 1 where N is the number of unknowns in the state space and `orbitguess[M*N+1]` is an estimate of the period of the limit cycle.

The scheme is as follows. We first consider a partition of ``[0,1]`` given by ``0<s_0<\\cdots<s_m=1`` and one looks for `T = x[end]` such that

 ``\\left(x_{i} - x_{i-1}\\right) - \\frac{T\\cdot h_i}{2} \\left(F(x_{i}) + F(x_{i-1})\\right) = 0,\\ i=1,\\cdots,m-1``

with ``u_{0} := u_{m-1}`` and the periodicity condition ``u_{m} - u_{1} = 0`` and

where ``h_1 = s_i-s_{i-1}``. Finally, the phase of the periodic orbit is constrained by using a section

 ``\\langle x[1] - x_\\pi, \\phi\\rangle=0.``

 A functional, hereby called `G`, encodes this problem. The following methods are available

- `pb(orbitguess, p)` evaluates the functional G on `orbitguess`
- `pb(orbitguess, p, du)` evaluates the jacobian `dG(orbitguess).du` functional at `orbitguess` on `du`
- `pb(Val(:JacFullSparse), orbitguess, p)` return the sparse matrix of the jacobian `dG(orbitguess)` at `orbitguess` without the constraints. It is called `A_γ` in the docs.
- `pb(Val(:JacFullSparseInplace), J, orbitguess, p)`. Same as `pb(Val(:JacFullSparse), orbitguess, p)` but overwrites `J` inplace. Note that the sparsity pattern must be the same independantly of the values of the parameters or of `orbitguess`. In this case, this is significantly faster than `pb(Val(:JacFullSparse), orbitguess, p)`.
- `pb(Val(:JacCyclicSparse), orbitguess, p)` return the sparse cyclic matrix Jc (see the docs) of the jacobian `dG(orbitguess)` at `orbitguess`
- `pb(Val(:BlockDiagSparse), orbitguess, p)` return the diagonal of the sparse matrix of the jacobian `dG(orbitguess)` at `orbitguess`. This allows to design Jacobi preconditioner. Use `blockdiag`.

!!! note "GPU call"
    For these methods to work on the GPU, for example with `CuArrays` in mode `allowscalar(false)`, we face the issue that the function `extractPeriodFDTrap` won't be well defined because it is a scalar operation. One may have to redefine it like `extractPeriodFDTrap(x::CuArray) = x[end:end]` or something else. Also, note that you must pass the option `ongpu = true` for the functional to be evaluated efficiently on the gpu.
"""
@with_kw struct PeriodicOrbitTrapProblem{TF, TJ, TJt, Td2F, Td3F, vectype, Tls <: AbstractLinearSolver, Tm} <: AbstractPOFDProblem
	# Function F(x, par)
	F::TF = nothing

	# Jacobian of F wrt x
	J::TJ = nothing

	# Jacobian transpose of F wrt x. This is mainly used for matrix-free computation of Folds of limit cycles
	Jt::TJt = nothing

	# Hessian of F wrt x, useful for continuation of Fold of periodic orbits
	d2F::Td2F = nothing

	# 3rd differential of F wrt x, useful for branch switching from branch of periodic orbits
	d3F::Td3F = nothing

	# variables to define a Section for the phase constraint equation
	ϕ::vectype = nothing
	xπ::vectype = nothing

	# discretisation of the time interval
	M::Int = 0
	mesh::Tm = TimeMesh(M)

	# dimension of the problem in case of an AbstractVector
	N::Int = 0

	# linear solver for each slice, i.e. to solve J⋅sol = rhs. This is mainly used for the computation of the Floquet coefficients
	linsolver::Tls = DefaultLS()

	# whether F and J are inplace functions
	isinplace::Bool = false

	# whether the computation takes place on the gpu
	ongpu::Bool = false

	# whether the time discretisation is adaptive
	adaptmesh::Bool = false
end

isInplace(pb::PeriodicOrbitTrapProblem) = pb.isinplace
onGpu(pb::PeriodicOrbitTrapProblem) = pb.ongpu
hasHessian(pb::PeriodicOrbitTrapProblem) = pb.d2F == nothing
@inline getTimeStep(pb::PeriodicOrbitTrapProblem, i::Int) = getTimeStep(pb.mesh, i)

function applyF(pb::PeriodicOrbitTrapProblem, dest, x, p)
	if isInplace(pb)
		pb.F(dest, x, p)
	else
		dest .= pb.F(x, p)
	end
	dest
end

function applyJ(pb::PeriodicOrbitTrapProblem, dest, x, p, dx)
	if isInplace(pb)
		pb.J(dest, x, p, dx)
	else
		dest .= apply(pb.J(x, p), dx)
	end
	dest
end

# dummy constructor, useful for specifying the "algorithm" to look for periodic orbits
# just call PeriodicOrbitTrapProblem()

function PeriodicOrbitTrapProblem(F, J, d2F, ϕ::vectype, xπ::vectype, m::Union{Int, vecmesh}, ls::AbstractLinearSolver = DefaultLS(); isinplace = false, ongpu = false, adaptmesh = false) where {vectype, vecmesh <: AbstractVector}
	_length = ϕ isa AbstractVector ? length(ϕ) : 0
	M = m isa Number ? m : length(m) + 1

	return PeriodicOrbitTrapProblem(F = F, J = J, d2F = d2F, ϕ = ϕ, xπ = xπ, M = M, mesh = TimeMesh(m), N = _length, linsolver = ls, isinplace = isinplace, ongpu = ongpu, adaptmesh = adaptmesh)
end

PeriodicOrbitTrapProblem(F, J, ϕ::vectype, xπ::vectype, m::Union{Int, vecmesh}, ls::AbstractLinearSolver = DefaultLS(); isinplace = false, ongpu = false, adaptmesh = false) where {vectype, vecmesh <: AbstractVector} = PeriodicOrbitTrapProblem(F, J, nothing, ϕ, xπ, m, ls; isinplace = isinplace, ongpu = ongpu, adaptmesh = adaptmesh)

# these functions extract the last component of the periodic orbit guess
@inline extractPeriodFDTrap(x::AbstractVector) = x[end]
@inline extractPeriodFDTrap(x::BorderedArray)  = x.T

# these functions extract the time slices components
extractTimeSlices(x::AbstractVector, N, M) = @views reshape(x[1:end-1], N, M)
extractTimeSlices(x::BorderedArray,  N, M) = x.u

function POTrapScheme!(pb::PeriodicOrbitTrapProblem, dest, u1, u2, par, h::Number, tmp, linear::Bool = true)
	# this function implements the basic implicit scheme used for the time integration
	# because this function is called in a cyclic manner, we save in the variable tmp the value of F(u2) in order to avoid recomputing it in a subsequent call
	# basically tmp is F(u2)
	if linear
		dest .= tmp
		# tmp <- pb.F(u1, par)
		applyF(pb, tmp, u1, par) #TODO this line does not almost seem to be type stable in code_wartype, gives @_11::Union{Nothing, Tuple{Int64,Int64}}
		dest .= u1 .- u2 .- h .* (dest .+ tmp)
	else
		dest .-= h .* tmp
		# tmp <- pb.F(u1, par)
		applyF(pb, tmp, u1, par)
		dest .-= h .* tmp
	end
end

function POTrapSchemeJac!(pb::PeriodicOrbitTrapProblem, dest, u1, u2, du1, du2, par, h::Number, tmp)
	# this function implements the basic implicit scheme used for the time integration
	# useful for the matrix-free jacobian
	# basically tmp is dF(u2).du2 (see above for explanation)
	dest .= tmp
	# tmp <- apply(pb.J(u1, par), du1)
	applyJ(pb, tmp, u1, par, du1)
	dest .= du1 .- du2 .- h .* (dest .+ tmp)
end

"""
This function implements the functional for finding periodic orbits based on finite differences using the Trapezoidal rule. It works for inplace / out of place vector fields pb.F
"""
function POTrapFunctional!(pb::PeriodicOrbitTrapProblem, out, u0, par)
	M = pb.M
	N = pb.N
	T = extractPeriodFDTrap(u0)

	u0c  = extractTimeSlices(u0, N, M)
	outc = extractTimeSlices(out, N, M)

	# outc[:, M] plays the role of tmp until it is used just after the for-loop
	@views applyF(pb, outc[:, M], u0c[:, M-1], par)

	h = T * getTimeStep(pb, 1)
	@views POTrapScheme!(pb, outc[:, 1], u0c[:, 1], u0c[:, M-1], par, h/2, outc[:, M])

	for ii in 2:M-1
		h = T * getTimeStep(pb, ii)
		# this function avoids computing F(u0c[:, ii]) twice
		@views POTrapScheme!(pb, outc[:, ii], u0c[:, ii], u0c[:, ii-1], par, h/2, outc[:, M])
	end

	# closure condition ensuring a periodic orbit
	outc[:, M] .= @views u0c[:, M] .- u0c[:, 1]

	# this is for CuArrays.jl to work in the mode allowscalar(false)
	if onGpu(pb)
		return @views vcat(out[1:end-1], dot(u0c[:, 1], pb.ϕ) - dot(pb.xπ, pb.ϕ)) # this is the phase condition
	else
		out[end] = @views dot(u0c[:, 1], pb.ϕ) - dot(pb.xπ, pb.ϕ) #dot(u0c[:, 1] .- pb.xπ, pb.ϕ)
		return out
	end
end

"""
Matrix free expression of the Jacobian of the problem for computing periodic obits when evaluated at `u0` and applied to `du`.
"""
function POTrapFunctionalJac!(pb::PeriodicOrbitTrapProblem, out, u0, par, du)
	M = pb.M
	N = pb.N
	T  = extractPeriodFDTrap(u0)
	dT = extractPeriodFDTrap(du)

	u0c = extractTimeSlices(u0, N, M)
	outc = extractTimeSlices(out, N, M)
	duc = extractTimeSlices(du, N, M)

	# compute the cyclic part
	@views Jc(pb, outc, u0[1:end-1-N], par, T, du[1:end-N-1], outc[:, M])

	# outc[:, M] plays the role of tmp until it is used just after the for-loop
	tmp = @view outc[:, M]

	# we now compute the partial derivative w.r.t. the period T
	# the .+ is for the GPU
	# out .+= @views (pb(vcat(u0[1:end-1], T .+ δ)) .- pb(u0)) ./ δ .* dT
	@views applyF(pb, tmp, u0c[:, M-1], par)

	h = dT * getTimeStep(pb, 1)
	@views POTrapScheme!(pb, outc[:, 1], u0c[:, 1], u0c[:, M-1], par, h/2, tmp, false)
	for ii in 2:M-1
		h = dT * getTimeStep(pb, ii)
		@views POTrapScheme!(pb, outc[:, ii], u0c[:, ii], u0c[:, ii-1], par, h/2, tmp, false)
	end

	# closure condition ensuring a periodic orbit
	outc[:, M] .= @views duc[:, M] .- duc[:, 1]

	# this is for CuArrays.jl to work in the mode allowscalar(false)
	if onGpu(pb)
		return @views vcat(out[1:end-1], dot(duc[:, 1], pb.ϕ))
	else
		out[end] = @views dot(duc[:, 1], pb.ϕ)
		return out
	end
end

function (pb::PeriodicOrbitTrapProblem)(u0::AbstractVector, par)
	out = similar(u0)
	POTrapFunctional!(pb, out, u0, par)
end

function (pb::PeriodicOrbitTrapProblem)(u0::AbstractVector, par, du)
	out = similar(du)
	POTrapFunctionalJac!(pb, out, u0, par, du)
end

####################################################################################################
# Matrix free expression of matrices related to the Jacobian Matrix of the PO functional
"""
Function to compute the Matrix-Free version of Aγ, see docs for its expression.
"""
function Agamma!(pb::PeriodicOrbitTrapProblem, outc, u0::AbstractVector, par, du::AbstractVector)
	# u0 of size N * M + 1
	# du of size N * M
	M = pb.M
	N = pb.N
	T = extractPeriodFDTrap(u0)
	u0c = extractTimeSlices(u0, N, M)

	# compute the cyclic part
	@views Jc(pb, outc, u0[1:end-1-N], par, T, du[1:end-N], outc[:, M])

	# closure condition ensuring a periodic orbit
	duc = reshape(du, N, M)
	outc[:, M] .= @views duc[:, M] .- duc[:, 1]
	return nothing
end

"""
Function to compute the Matrix-Free version of the cyclic matrix Jc, see docs for its expression.
"""
function Jc(pb::PeriodicOrbitTrapProblem, outc::AbstractMatrix, u0::AbstractVector, par, T, du::AbstractVector, tmp)
	# tmp plays the role of buffer array
	# u0 of size N * (M - 1)
	# du of size N * (M - 1)
	# outc of size N * M
	M = pb.M
	N = pb.N

	u0c = reshape(u0, N, M-1)
	duc = reshape(du, N, M-1)

	@views applyJ(pb, tmp, u0c[:, M-1], par, duc[:, M-1])

	h = T * getTimeStep(pb, 1)
	@views POTrapSchemeJac!(pb, outc[:, 1], u0c[:, 1], u0c[:, M-1],
											duc[:, 1], duc[:, M-1], par, h/2, tmp)

	for ii in 2:M-1
		h = T * getTimeStep(pb, ii)
		@views POTrapSchemeJac!(pb, outc[:, ii], u0c[:, ii], u0c[:, ii-1],
												 duc[:, ii], duc[:, ii-1], par, h/2, tmp)
	end

	# we also return a Vector version of outc
	return vec(outc)
end

function Jc(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par, du::AbstractVector)
	M = pb.M
	N = pb.N

	T = extractPeriodFDTrap(u0)

	out  = similar(du)
	outc = reshape(out, N, M-1)
	tmp = similar(view(outc, :, 1))
	return @views Jc(pb, outc, u0[1:end-1-N], par, T, du, tmp)
end
####################################################################################################
"""
Matrix by blocks expression of the Jacobian for the PO functional computed at the space-time guess: `u0`
"""
function jacobianPOFDBlock(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par; γ = 1.0)
	# extraction of various constants
	M = pb.M
	N = pb.N

	Aγ = BlockArray(spzeros(M * N, M * N), N * ones(Int64, M),  N * ones(Int64, M))
	cylicPOFDBlock!(pb, u0, par, Aγ)

	In = spdiagm( 0 => ones(N))
	setblock!(Aγ, -γ * In, M, 1)
	setblock!(Aγ,  In,     M, M)
	return Aγ
end

"""
This function populates Jc with the cyclic matrix using the different Jacobians
"""
@views function cylicPOFDBlock!(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par, Jc::BlockArray)
	# extraction of various constants
	M = pb.M
	N = pb.N
	T = extractPeriodFDTrap(u0)

	In = spdiagm( 0 => ones(N))
	On = spzeros(N, N)

	u0c = extractTimeSlices(u0, N, M)
	outc = similar(u0c)

	tmpJ = @views pb.J(u0c[:, 1], par)

	h = T * getTimeStep(pb, 1)
	Jn = In - h/2 .* tmpJ
	setblock!(Jc, Jn, 1, 1)

	Jn = -In - h/2 .* pb.J(u0c[:, M-1], par)
	setblock!(Jc, Jn, 1, M-1)

	for ii in 2:M-1
		h = T * getTimeStep(pb, ii)
		Jn = -In - h/2 .* tmpJ
		setblock!(Jc, Jn, ii, ii-1)

		tmpJ .= @views pb.J(u0c[:, ii], par)

		Jn = In - h/2 .* tmpJ
		setblock!(Jc, Jn, ii, ii)
	end
	return Jc
end

function cylicPOFDBlock(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par)
	# extraction of various constants
	M = pb.M
	N = pb.N
	Jc = BlockArray(spzeros((M - 1) * N, (M - 1) * N), N * ones(Int64, M-1),  N * ones(Int64, M-1))
	cylicPOFDBlock!(pb, u0, par, Jc)
end

cylicPOTrapSparse(pb::PeriodicOrbitTrapProblem, orbitguess0, par) = blockToSparse(cylicPOFDBlock(pb, orbitguess0, par))

"""
This method returns the jacobian of the functional G encoded in PeriodicOrbitTrapProblem using a Sparse representation.
"""
function (pb::PeriodicOrbitTrapProblem)(::Val{:JacFullSparse}, u0::AbstractVector, par; γ = 1.0, δ = 1e-9)
	# extraction of various constants
	M = pb.M
	N = pb.N
	T = extractPeriodFDTrap(u0)
	AγBlock = jacobianPOFDBlock(pb, u0, par; γ = γ)

	# we now set up the last line / column
	@views ∂TGpo = (pb(vcat(u0[1:end-1], T + δ), par) .- pb(u0, par)) ./ δ

	# this is "bad" for performance. Get converted to SparseMatrix at the next line
	Aγ = blockToSparse(AγBlock) # most of the computing time is here!!
	@views Aγ = hcat(Aγ, ∂TGpo[1:end-1])
	Aγ = vcat(Aγ, spzeros(1, N * M + 1))

	Aγ[N*M+1, 1:N] .=  pb.ϕ
	Aγ[N*M+1, N*M+1] = ∂TGpo[end]
	return Aγ
end

"""
This method returns the jacobian of the functional G encoded in PeriodicOrbitTrapProblem using a Sparse representation and inplace update.
"""
@views function (pb::PeriodicOrbitTrapProblem)(::Val{:JacFullSparseInplace}, J0, u0::AbstractVector, par; γ = 1.0, δ = 1e-9)
		# update J0 inplace assuming that the sparsity pattern of J0 and dG(orbitguess0) are the same
		M = pb.M
		N = pb.N
		T = extractPeriodFDTrap(u0)

		In = spdiagm( 0 => ones(N))
		On = spzeros(N, N)

		u0c = extractTimeSlices(u0, N, M)
		outc = similar(u0c)

		tmpJ = @views pb.J(u0c[:, 1], par)

		h = T * getTimeStep(pb, 1)
		Jn = In - h/2 * tmpJ
		# setblock!(Jc, Jn, 1, 1)
		J0[1:N, 1:N] .= Jn

		Jn = -In - h/2 * pb.J(u0c[:, M-1], par)
		# setblock!(Jc, Jn, 1, M-1)
		J0[1:N, (M-2)*N+1:(M-1)*N] .= Jn

		for ii in 2:M-1
			h = T * getTimeStep(pb, ii)
			Jn = -In - h/2 * tmpJ
			# the next lines cost the most
			# setblock!(Jc, Jn, ii, ii-1)
			J0[(ii-1)*N+1:(ii)*N, (ii-2)*N+1:(ii-1)*N] .= Jn

			tmpJ = pb.J(u0c[:, ii], par)

			Jn = In - h/2 * tmpJ
			# setblock!(Jc, Jn, ii, ii)
			J0[(ii-1)*N+1:(ii)*N, (ii-1)*N+1:(ii)*N] .= Jn
		end

		# setblock!(Aγ, -γ * In, M, 1)
		# useless to update:
			# J0[(M-1)*N+1:(M)*N, (1-1)*N+1:(1)*N] .= -In
		# setblock!(Aγ,  In,     M, M)
		# useless to update:
			# J0[(M-1)*N+1:(M)*N, (M-1)*N+1:(M)*N] .= In

		# we now set up the last line / column
		∂TGpo = (pb(vcat(u0[1:end-1], T + δ), par) .- pb(u0, par)) ./ δ
		J0[:, end] .=  ∂TGpo

		# this following does not depend on u0, so it does not change
		# J0[N*M+1, 1:N] .=  pb.ϕ

		return J0
end


@views function (pb::PeriodicOrbitTrapProblem)(::Val{:JacFullSparseInplace}, J0, u0::AbstractVector, par, indx; γ = 1.0, δ = 1e-9)
	M = pb.M
	N = pb.N
	T = extractPeriodFDTrap(u0)

	In = spdiagm( 0 => ones(N))
	On = spzeros(N, N)

	u0c = extractTimeSlices(u0, N, M)
	outc = similar(u0c)

	tmpJ = pb.J(u0c[:, 1], par)

	h = T * getTimeStep(pb, 1)
	Jn = In - tmpJ * (h/2)
	# setblock!(Jc, Jn, 1, 1)
	J0.nzval[indx[1,1]] .= Jn.nzval

	Jn = -In - pb.J(u0c[:, M-1], par) * (h/2)
	# setblock!(Jc, Jn, 1, M-1)
	J0.nzval[indx[1,M-1]] .= Jn.nzval

	for ii in 2:M-1
		h = T * getTimeStep(pb, ii)
		Jn = -In - tmpJ * (h/2)
		# the next lines cost the most
		# setblock!(Jc, Jn, ii, ii-1)
		J0.nzval[indx[ii,ii-1]] .= Jn.nzval

		tmpJ = pb.J(u0c[:, ii], par)# * (h/2)

		Jn = In -  tmpJ * (h/2)
		# setblock!(Jc, Jn, ii, ii)
		J0.nzval[indx[ii,ii]] .= Jn.nzval
	end

	# setblock!(Aγ, -γ * In, M, 1)
	# useless to update:
		# J0[(M-1)*N+1:(M)*N, (1-1)*N+1:(1)*N] .= -In
	# setblock!(Aγ,  In,     M, M)
	# useless to update:
		# J0[(M-1)*N+1:(M)*N, (M-1)*N+1:(M)*N] .= In

	# we now set up the last line / column
	∂TGpo = (pb(vcat(u0[1:end-1], T + δ), par) .- pb(u0, par)) ./ δ
	J0[:, end] .=  ∂TGpo

	# this following does not depend on u0, so it does not change
	# J0[N*M+1, 1:N] .=  pb.ϕ

	return J0
end

function (pb::PeriodicOrbitTrapProblem)(::Val{:JacCyclicSparse}, u0::AbstractVector, par, γ = 1.0)
	# extraction of various constants
	N = pb.N
	AγBlock = jacobianPOFDBlock(pb, u0, par; γ = γ)

	# this is bad for performance. Get converted to SparseMatrix at the next line
	Aγ = blockToSparse(AγBlock) # most of the computing time is here!!
	return Aγ[1:end-N, 1:end-N]
end

function (pb::PeriodicOrbitTrapProblem)(::Val{:BlockDiagSparse}, u0::AbstractVector, par)
	# extraction of various constants
	M = pb.M
	N = pb.N
	T = extractPeriodFDTrap(u0)

	A_diagBlock = BlockArray(spzeros(M * N, M * N), N * ones(Int64, M),  N * ones(Int64, M))

	In = spdiagm( 0 => ones(N))

	u0c = reshape(u0[1:end-1], N, M)
	outc = similar(u0c)

	h = T * getTimeStep(pb, 1)
	@views Jn = In - h/2 .* pb.J(u0c[:, 1], par)
	setblock!(A_diagBlock, Jn, 1, 1)

	for ii in 2:M-1
		h = T * getTimeStep(pb, ii)
		@views Jn = In - h/2 .* pb.J(u0c[:, ii], par)
		setblock!(A_diagBlock, Jn, ii, ii)
	end
	setblock!(A_diagBlock, In, M, M)

	A_diag_sp = blockToSparse(A_diagBlock) # most of the computing time is here!!
	return A_diag_sp
end
####################################################################################################
# Utils

function getTimeDiff(pb::PeriodicOrbitTrapProblem, u0)
	M = pb.M
	N = pb.N
	T = extractPeriodFDTrap(u0)
	u0c = reshape(u0[1:end-1], N, M)

	res = Float64[]
	for ii in 1:M-1
		push!(res, norm(u0c[:,ii+1].-u0c[:,ii]) * T/M)
	end
	return res
end

"""
$(SIGNATURES)

Compute the amplitude of the periodic orbit associated to `x`. The keyword argument `ratio = 1` is used as follows. If `length(x) = 1 + ratio * n`, the call returns the amplitude over `x[1:n]`.
"""
function getAmplitude(prob::PeriodicOrbitTrapProblem, x::AbstractVector, p; ratio = 1)
	n = div(length(x)-1, ratio)
	_max = maximum(x[1:n])
	_min = minimum(x[1:n])
	return maximum(_max .- _min)
end

"""
$(SIGNATURES)

Compute the maximum of the periodic orbit associated to `x`. The keyword argument `ratio = 1` is used as follows. If `length(x) = 1 + ratio * n`, the call returns the amplitude over `x[1:n]`.
"""
function getMaximum(prob::PeriodicOrbitTrapProblem, x::AbstractVector, p; ratio = 1)
	n = div(length(x)-1, ratio)
	_max = maximum(x[1:n])
end

"""
$(SIGNATURES)

Compute the period of the periodic orbit associated to `x`.
"""
@inline getPeriod(prob::PeriodicOrbitTrapProblem, x, p) = extractPeriodFDTrap(x)

"""
$(SIGNATURES)

Compute the full trajectory associated to `x`. Mainly for plotting purposes.
"""
function getTrajectory(prob::PeriodicOrbitTrapProblem, x::AbstractVector, p)
	T = getPeriod(prob, x, p)
	M = getM(prob)
	N = div(length(x) - 1, M)
	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)
	return (t = cumsum(T .* collect(prob.mesh)), u = xc)
end
####################################################################################################
# The following struct encodes a jacobian of PeriodicOrbitTrapProblem which is a convenient composite type for the computation of Floquet multipliers. Therefore, it is only used in the method continuationPOTrap
mutable struct PeriodicOrbitTrapJacobianFull{Tpb, Tj, vectype, Tpar}
	pb::Tpb								# PeriodicOrbitTrapProblem
	J::Tj								# jacobian of the problem
	orbitguess0::vectype				# point at which the jacobian is computed
	par::Tpar							# parameter passed to F, J...
end

# computation of the jacobian, nothing to be done
(pojacfull::PeriodicOrbitTrapJacobianFull)(x, p) = return pojacfull

# linear solver for the PO functional, akin to a bordered linear solver
@with_kw mutable struct PeriodicOrbitTrapLS{Tl} <: AbstractLinearSolver
	linsolver::Tl = DefaultLS()			# linear solver
end

# linear solver for the jacobian
(pols::PeriodicOrbitTrapLS)(pojacfull::PeriodicOrbitTrapJacobianFull, rhs) = pols.linsolver(pojacfull.J, rhs)

(pols::PeriodicOrbitTrapLS)(pojacfull::PeriodicOrbitTrapJacobianFull, rhs1, rhs2) = pols.linsolver(pojacfull.J, rhs1, rhs2)

####################################################################################################
# Linear solvers of the linearized version of the functional G implemented by PeriodicOrbitTrapProblem

# composite type to encode the Aγ Operator and its associated cyclic matrix
@with_kw mutable struct AγOperator{Tvec, Tjc, T, Tpb, Tpar}
	N::Int64 = 0				    		# dimension of a time slice
	orbitguess::Tvec = zeros(1)				# point at which Aγ is evaluated, of size N * M + 1
	Jc::Tjc	= lu(spdiagm(0 => ones(1)))	    # lu factorisation of the cyclic matrix
	is_matrix_free::Bool = false	    	# whether we consider a sparse matrix representation or a Matrix Free one
	γ::T = 1.0				    			# factor γ can be used to compute Floquet multipliers
	prob::Tpb = nothing						# PO functional, used when is_matrix_free = true
	par::Tpar = nothing						# parameter passed to vector field F(x, par)
end

ismatrixfree(A::AγOperator) = A.is_matrix_free

# function to update the cyclic matrix
function (A::AγOperator)(pb::PeriodicOrbitTrapProblem, orbitguess::AbstractVector, par)
	if ismatrixfree(A) == false
		# we store the lu decomposition of the newly computed cyclic matrix
		A.Jc = SparseArrays.lu(cylicPOTrapSparse(pb, orbitguess, par))
	else
		copyto!(A.orbitguess, orbitguess)
		# update par for Matrix-Free
		A.par = par
	end
end

# linear solver designed specifically to deal with AγOperator
@with_kw struct AγLinearSolver{Tls} <: AbstractLinearSolver
	# Linear solver to invert the cyclic matrix Jc contained in Aγ
	linsolver::Tls = DefaultLS()
end

# this function is called whenever one wants to invert Aγ
function (ls::AγLinearSolver)(A::AγOperator, rhs)
	N = A.N
	if ismatrixfree(A) == false
		# we invert the cyclic part Jc of Aγ
		xbar, flag, numiter = @views ls.linsolver(A.Jc, rhs[1:end - N])
		!flag && @warn "Sparse solver for Aγ did not converge"
	else
		# we invert the cyclic part Jc of Aγ
		xbar, flag, numiter = @views ls.linsolver(dx -> Jc(A.prob, A.orbitguess, A.par, dx), rhs[1:end - N])
		!flag && @warn "Matrix Free solver for Aγ did not converge"
	end
	x = similar(rhs)
	x[1:end-N] .= xbar
	x[end-N+1:end] .= @views A.γ .* x[1:N] .+ rhs[end-N+1:end]
	return x ,flag, numiter
end

####################################################################################################
# The following structure encodes a jacobian of PeriodicOrbitTrapProblem which eases the use of PeriodicOrbitTrapBLS. It is made so that accessing to the cyclic matrix Jc or Aγ is easier. It is combined with a specific linear solver. It is also a convenient structure for the computation of Floquet multipliers. Therefore, it is only used in the method continuationPOTrap
@with_kw mutable struct PeriodicOrbitTrapJacobianBordered{Tpb <: PeriodicOrbitTrapProblem, T∂, vectype, Tpar}
	pb::Tpb								# PeriodicOrbitTrapProblem
	∂TGpo::T∂	   = nothing			# derivative of the PO functional G wrt T
	Aγ::AγOperator = AγOperator()		# Aγ Operator involved in the Jacobian of the PO functional
	orbitguess0::vectype = nothing		# point at which the jacobian is computed
	par::Tpar							# parameter passed to F, J...
end

# this function is called whenever the jacobian of G has to be updated
function (J::PeriodicOrbitTrapJacobianBordered)(orbitguess0::AbstractVector, par; δ = 1e-9)
	# u0 must be an orbit guess
	@views J.orbitguess0 .= orbitguess0[1:length(J.orbitguess0)]

	# we compute the derivative of the problem wrt the period TODO: remove this or improve!!
	T = extractPeriodFDTrap(orbitguess0)
	# TODO REMOVE CE vcat!
	@views J.∂TGpo .= (J.pb(vcat(orbitguess0[1:end-1], T + δ), par) .- J.pb(orbitguess0, par)) ./ δ

	# update Aγ
	J.Aγ(J.pb, orbitguess0, par)

	# return J, needed to properly call the linear solver.
	return J
end

####################################################################################################

# linear solver for the PO functional, akin to a bordered linear solver
@with_kw mutable struct PeriodicOrbitTrapBLS{Tl} <: AbstractLinearSolver
	linsolverbls::Tl = BorderingBLS(AγLinearSolver())	# linear solver
end

# Linear solver associated to PeriodicOrbitTrapJacobianBordered
function (ls::PeriodicOrbitTrapBLS)(J::PeriodicOrbitTrapJacobianBordered, rhs)
	N = J.pb.N

	# TODO REMOVE THIS HACK
	ϕ = zeros(length(rhs)-1)
	ϕ[1:N] .= J.pb.ϕ

	# we solve the bordered linear system as follows
	dX, dl, flag, liniter = @views ls.linsolverbls(J.Aγ, J.∂TGpo[1:end-1],
	 										 		  ϕ, J.∂TGpo[end],
													  rhs[1:end-1], rhs[end])
	return vcat(dX, dl), flag, sum(liniter)
end

# One could think that by implementing (ls::PeriodicOrbitTrapBLS)(J::PeriodicOrbitTrapJacobianBordered, rhs1, rhs2), we could speed up the computation of the linear Bordered system arising in the continuation process. However, we can note that this speed up would be observed only if a factorization of J.Aγ is available like an LU one. When such factorization is available, it is automatically stored as such in J.Aγ and so no speed up would be gained by implementing (ls::PeriodicOrbitTrapBLS)(J::PeriodicOrbitTrapJacobianBordered, rhs1, rhs2)

####################################################################################################
# newton wrappers
function _newton(probPO::PeriodicOrbitTrapProblem, orbitguess, par, options::NewtonPar, linearPO::Symbol = :BorderedLU; defOp::Union{Nothing, DeflationOperator{T, Tf, vectype}} = nothing, kwargs...) where {T, Tf, vectype}
	@assert linearPO in [:FullLU, :BorderedLU, :FullMatrixFree, :BorderedMatrixFree, :FullSparseInplace]
	N = probPO.N
	M = probPO.M

	if linearPO in [:FullLU, :FullMatrixFree, :FullSparseInplace]
		@assert orbitguess isa AbstractVector
		@assert length(orbitguess) == N * M + 1 "Error with size of the orbitguess"

		if linearPO == :FullLU
			jac = (x, p) -> probPO(Val(:JacFullSparse), x, p)
		elseif linearPO == :FullSparseInplace
			# sparse matrix to hold the jacobian
			_J =  probPO(Val(:JacFullSparse), orbitguess, par)
			_indx = getBlocks(_J, N, M)
			# inplace modification of the jacobian _J
			jac = (x, p) -> probPO(Val(:JacFullSparseInplace), _J, x, p)
		else
		 	jac = (x, p) -> ( dx -> probPO(x, p, dx))
		end

		if isnothing(defOp)
			return newton(probPO, jac, orbitguess, par, options; kwargs...)
		else
			return newton(probPO, jac, orbitguess, par, options, defOp; kwargs...)
		end

	else
		@assert orbitguess isa AbstractVector
		@assert length(orbitguess) == N * M + 1 "Error with size of the orbitguess"

		if linearPO == :BorderedLU
			Aγ = AγOperator(is_matrix_free = false, N = probPO.N, Jc = lu(spdiagm( 0 => ones(N * (M - 1)) )) )
			# linear solver
			lspo = PeriodicOrbitTrapBLS()
		else	# :BorderedMatrixFree
			Aγ = AγOperator(is_matrix_free = true, prob = probPO, N = probPO.N, orbitguess = zeros(N * M + 1), Jc = lu(spdiagm( 0 => ones(N * (M - 1)) )), par = par)
			# linear solver
			lspo = PeriodicOrbitTrapBLS(BorderingBLS(AγLinearSolver(options.linsolver)))
		end

		# create the jacobian
		jacPO = PeriodicOrbitTrapJacobianBordered(probPO, zeros(N * M + 1), Aγ, zeros(N * M + 1), par)

		if isnothing(defOp)
			return newton(probPO, jacPO, orbitguess, par, (@set options.linsolver = lspo); kwargs...)
		else
			return newton(probPO, jacPO, orbitguess, par, (@set options.linsolver = lspo), defOp; kwargs...)
		end
	end
end

"""
$(SIGNATURES)

This is the Newton-Krylov Solver for computing a periodic orbit using a functional G based on Finite Differences and a Trapezoidal rule.

# Arguments:
- `prob` a problem of type [`PeriodicOrbitTrapProblem`](@ref) encoding the functional G
- `orbitguess` a guess for the periodic orbit where `orbitguess[end]` is an estimate of the period of the orbit. It should be a vector of size `N * M + 1` where `M` is the number of time slices, `N` is the dimension of the phase space. This must be compatible with the numbers `N,M` in `prob`.
- `options` same as for the regular `newton` method
- `linearPO = :BorderedLU`. Specify the choice of the linear algorithm, which must belong to `[:FullLU, :FullSparseInplace, :BorderedLU, :FullMatrixFree, :BorderedMatrixFree, :FullSparseInplace]`. This is used to select a way of inverting the jacobian `dG` of the functional G.
    - For `:FullLU`, we use the default linear solver based on a sparse matrix representation of `dG`. This matrix is assembled at each newton iteration.
    - For `:FullSparseInplace`, this is the same as for `:FullLU` but the sparse matrix `dG` is updated inplace. This method allocates much less. In some cases, this is significantly faster than using `:FullLU`. Note that this method can only be used if the sparsity pattern of the jacobian is always the same.
    - For `:BorderedLU`, we take advantage of the bordered shape of the linear solver and use a LU decomposition to invert `dG` using a bordered linear solver. This is the default algorithm.
    - For `:FullMatrixFree`, a matrix free linear solver is used for `dG`: note that a preconditioner is very likely required here because of the cyclic shape of `dG` which affects negatively the convergence properties of GMRES.
    - For `:BorderedMatrixFree`, a matrix free linear solver is used but for `Jc` only (see docs): it means that `options.linsolver` is used to invert `Jc`. These two Matrix-Free options thus expose different part of the jacobian `dG` in order to use specific preconditioners. For example, an ILU preconditioner on `Jc` could remove the constraints in `dG` and lead to poor convergence. Of course, for these last two methods, a preconditioner is likely to be required.

# Output:
- solution
- history of residuals
- flag of convergence
- number of iterations
"""

newton(probPO::PeriodicOrbitTrapProblem, orbitguess, par, options::NewtonPar; linearPO::Symbol = :BorderedLU, kwargs...) = _newton(probPO, orbitguess, par, options, linearPO; defOp = nothing, kwargs...)

"""
	newton(probPO::PeriodicOrbitTrapProblem, orbitguess, options::NewtonPar, defOp::DeflationOperator{T, Tf, vectype}, linearPO = :BorderedLU; kwargs...) where {T, Tf, vectype}

This function is similar to `newton(probPO, orbitguess, options, linearPO; kwargs...)` except that it uses deflation in order to find periodic orbits different from the ones stored in `defOp`. We refer to the mentioned method for a full description of the arguments. The current method can be used in the vicinity of a Hopf bifurcation to prevent the Newton-Krylov algorithm from converging to the equilibrium point.
"""

newton(probPO::PeriodicOrbitTrapProblem, orbitguess::vectype, par, options::NewtonPar, defOp::DeflationOperator{T, Tf, vectype}; linearPO::Symbol = :BorderedLU, kwargs...) where {T, Tf, vectype} = _newton(probPO, orbitguess, par, options, linearPO; defOp = defOp, kwargs...)

####################################################################################################
# continuation wrapper

"""
	continuationPOTrap(probPO::PeriodicOrbitTrapProblem, orbitguess, par, lens::Lens, _contParams::ContinuationPar, linearAlgo::AbstractBorderedLinearSolver; linearPO = :BorderedLU, printSolution = (u,p) -> u[end], kwargs...)

This is the continuation routine for computing a periodic orbit using a functional G based on Finite Differences and a Trapezoidal rule.

# Arguments
- `prob::PeriodicOrbitTrapProblem` encodes the functional G
- `orbitguess` a guess for the periodic orbit where `orbitguess[end]` is an estimate of the period of the orbit. It could be a vector of size `N * M + 1` where `M` is the number of time slices, `N` is the dimension of the phase space. This must be compatible with the numbers `N, M` in `prob`.
- `p0` set of parameters passed to the vector field
- `contParams` same as for the regular [`continuation`](@ref) method
- `linearAlgo` same as in [`continuation`](@ref)
- `linearPO = :BorderedLU`. Same as `newton` when applied to `PeriodicOrbitTrapProblem`. More precisely:
    - For `:FullLU`, we use the default linear solver on a sparse matrix representation of `dG`. This matrix is assembled at each newton iteration.
    - For `:FullSparseInplace`, this is the same as for `:FullLU` but the sparse matrix `dG` is updated inplace. This method allocates much less. In some cases, this is significantly faster than using `:FullLU`. Note that this method can only be used if the sparsity pattern of the jacobian is always the same.
    - For `:BorderedLU`, we take advantage of the bordered shape of the linear solver and use LU decomposition to invert `dG` using a bordered linear solver. This is the default algorithm.
    - For `:FullMatrixFree`, a matrix free linear solver is used for `dG`: note that a preconditioner is very likely required here because of the cyclic shape of `dG` which affects negatively the convergence properties of GMRES.
    - For `:BorderedMatrixFree`, a matrix free linear solver is used but for `Jc` only (see docs): it means that `options.linsolver` is used to invert `Jc`. These two Matrix-Free options thus expose different part of the jacobian `dG` in order to use specific preconditioners. For example, an ILU preconditioner on `Jc` could remove the constraints in `dG` and lead to poor convergence. Of course, for these last two methods, a preconditioner is likely to be required.


Note that by default, the method prints the period of the periodic orbit as function of the parameter. This can be changed by providing your `printSolution` argument.
"""
function continuationPOTrap(probPO::PeriodicOrbitTrapProblem, orbitguess, par, lens::Lens, contParams::ContinuationPar, linearAlgo::AbstractBorderedLinearSolver; linearPO = :BorderedLU, printSolution = (u,p) -> u[end], kwargs...)
	@assert linearPO in [:FullLU, :FullMatrixFree, :BorderedLU, :BorderedMatrixFree, :FullSparseInplace]

	N = probPO.N
	M = probPO.M
	options = contParams.newtonOptions

	if computeEigenElements(contParams)
		contParams = @set contParams.newtonOptions.eigsolver = FloquetQaDTrap(contParams.newtonOptions.eigsolver)
	end

	if linearPO in [:FullLU, :FullMatrixFree, :FullSparseInplace]
		@assert length(orbitguess) == N * M + 1 "Error with size of the orbitguess"

		if linearPO == :FullLU
			jac = (x, p) -> probPO(Val(:JacFullSparse), x, p)
		elseif linearPO == :FullSparseInplace
			# sparse matrix to hold the jacobian
			_J =  probPO(Val(:JacFullSparse), orbitguess, par)
			_indx = getBlocks(_J, N, M)
			# inplace modification of the jacobian _J
			jac = (x, p) -> probPO(Val(:JacFullSparseInplace), _J, x, p, _indx)
		else
		 	jac = (x, p) ->   ( dx -> probPO(x, p, dx))
		end

		lspo = PeriodicOrbitTrapLS(options.linsolver)

			return continuation(
				probPO,
				(x, p) -> PeriodicOrbitTrapJacobianFull(probPO, jac(x, p), x, p),
				orbitguess, par, lens,
				(@set contParams.newtonOptions.linsolver = lspo);
				printSolution = printSolution,
				kwargs...)
	else
		@assert orbitguess isa AbstractVector
		@assert length(orbitguess) == N * M + 1 "Error with size of the orbitguess"

		if linearPO == :BorderedLU
			Aγ = AγOperator(is_matrix_free = false, N = N,
					Jc = lu(spdiagm( 0 => ones(N * (M - 1)) )) )
			# linear solver
			lspo = PeriodicOrbitTrapBLS()
		else
			Aγ = AγOperator(is_matrix_free = true, prob = probPO, N = N,
					orbitguess = zeros(N * M + 1),
					Jc = lu(spdiagm( 0 => ones(N * (M - 1)) )), par = par)
			# linear solver
			lspo = PeriodicOrbitTrapBLS(BorderingBLS(AγLinearSolver(options.linsolver)))
		end

		# create the jacobian
		jacPO = PeriodicOrbitTrapJacobianBordered(probPO, zeros(N * M + 1), Aγ, zeros(N * M + 1), par)

		return continuation(probPO, jacPO, orbitguess, par, lens,
			(@set contParams.newtonOptions.linsolver = lspo);
			printSolution = printSolution,
			kwargs...)
	end
end

"""
	continuation(probPO, orbitguess, p0::Real, _contParams::ContinuationPar; linearPO = :BorderedLU, printSolution = (u,p) -> u[end], linearAlgo = BorderingBLS(), kwargs...)

This is the continuation routine for computing a periodic orbit using a functional G based on Finite Differences and a Trapezoidal rule.

# Arguments
- `prob::PeriodicOrbitTrapProblem` encodes the functional G
- `orbitguess` a guess for the periodic orbit where `orbitguess[end]` is an estimate of the period of the orbit. It could be a vector of size `N * M + 1` where `M` is the number of time slices, `N` is the dimension of the phase space. This must be compatible with the numbers `N, M` in `prob`.
- `p0` set of parameters passed to the vector field
- `contParams` same as for the regular [`continuation`](@ref) method
- `linearAlgo` same as in [`continuation`](@ref)
- `linearPO = :BorderedLU`. Same as `newton` when applied to `PeriodicOrbitTrapProblem`. More precisely:
    - For `:FullLU`, we use the default linear solver on a sparse matrix representation of `dG`. This matrix is assembled at each newton iteration.
    - For `:FullSparseInplace`, this is the same as for `:FullLU` but the sparse matrix `dG` is updated inplace. This method allocates much less. In some cases, this is significantly faster than using `:FullLU`. Note that this method can only be used if the sparsity pattern of the jacobian is always the same.
    - For `:BorderedLU`, we take advantage of the bordered shape of the linear solver and use LU decomposition to invert `dG` using a bordered linear solver. This is the default algorithm.
    - For `:FullMatrixFree`, a matrix free linear solver is used for `dG`: note that a preconditioner is very likely required here because of the cyclic shape of `dG` which affects negatively the convergence properties of GMRES.
    - For `:BorderedMatrixFree`, a matrix free linear solver is used but for `Jc` only (see docs): it means that `options.linsolver` is used to invert `Jc`. These two Matrix-Free options thus expose different part of the jacobian `dG` in order to use specific preconditioners. For example, an ILU preconditioner on `Jc` could remove the constraints in `dG` and lead to poor convergence. Of course, for these last two methods, a preconditioner is likely to be required.


Note that by default, the method prints the period of the periodic orbit as function of the parameter. This can be changed by providing your `printSolution` argument.
"""
function continuation(probPO::PeriodicOrbitTrapProblem, orbitguess, par, lens::Lens, _contParams::ContinuationPar; linearPO = :BorderedLU, printSolution = (u,p) -> u[end], linearAlgo = BorderingBLS(), kwargs...)
	_linearAlgo = @set linearAlgo.solver = _contParams.newtonOptions.linsolver
	return continuationPOTrap(probPO, orbitguess, par, lens, _contParams, _linearAlgo; linearPO = linearPO, printSolution = printSolution, kwargs...)
end

####################################################################################################
# functions needed Branch switching from Hopf bifurcation point
function update(prob::PeriodicOrbitTrapProblem, F, dF, hopfpt, ζr::AbstractVector, M, orbitguess_a, period)
	# append period at the end of the initial guess
	orbitguess_v = reduce(vcat, orbitguess_a)
	orbitguess = vcat(vec(orbitguess_v), period) |> vec

	# update the problem
	probPO = setproperties(prob, N = length(ζr), M = M, F = F, J = dF, ϕ = ζr, xπ = hopfpt.x0)
	return probPO, orbitguess
end

####################################################################################################
# Branch switching from BP of PO
"""
$(SIGNATURES)

Branch switching at a Branch point of periodic orbits specified by a [`PeriodicOrbitTrapProblem`](@ref). This is still experimental. A deflated Newton-Krylov solver is used to improve the branch switching capabilities.

# Arguments
- `br` branch of periodic orbits computed with a [`PeriodicOrbitTrapProblem`](@ref)
- `ind_bif` index of the branch point
- `_contParams` parameters to be used by a regular [`continuation`](@ref)

# Optional arguments
- `Jt = (x, p) -> transpose(d_xF(x, p))` jacobian adjoint, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jt` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jt = (x, p) -> transpose(dF(x, p))`
- `δ` used internally to compute derivatives w.r.t the parameter `p`.
- `δp = 0.1` used to specify a particular guess for the parameter in the branch which is otherwise determined by `contParams.ds`. This allows to use a step larger than `contParams.dsmax`.
- `ampfactor = 1` factor which alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch
- `linearPO = :BorderedLU` linear solver used for the Newton-Krylov solver when applied to [`PeriodicOrbitTrapProblem`](@ref).
- `printSolution = (u,p) -> u[end]`, print method used in the bifurcation diagram, by default this prints the period of the periodic orbit.
- `linearAlgo = BorderingBLS()`, same as for [`continuation`](@ref)
- `kwargs` keywords arguments used for a call to the regular [`continuation`](@ref)
"""
function continuationPOTrapBPFromPO(br::BranchResult, ind_bif::Int, _contParams::ContinuationPar ; Jt = nothing, δ = 1e-8, δp = 0.1, ampfactor = 1, usedeflation = true, linearPO = :BorderedLU, printSolution = (u,p) -> u[end], linearAlgo = BorderingBLS(), kwargs...)
	verbose = get(kwargs, :verbosity, 0) > 0

	@assert br.functional isa PeriodicOrbitTrapProblem
	@assert abs(br.bifpoint[ind_bif].δ[1]) == 1

	bifpt = br.bifpoint[ind_bif]

	# let us compute the kernel
	λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
	verbose && print("--> computing nullspace...")
	ζ0 = geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[bifpt.idx].eigenvec, bifpt.ind_ev)
	verbose && println("Done!")
	# we normalize it by the sup norm because it could be too small/big in L2 norm
	ζ0 ./= norm(ζ0, Inf)

	pb = br.functional

	ζ_a = MonodromyQaDFD(Val(:ExtractEigenVector), pb, bifpt.x, set(br.params, br.param_lens, bifpt.param), real.(ζ0))
	ζ = reduce(vcat, ζ_a)

	orbitguess = copy(bifpt.x)
	orbitguess[1:end-1] .+= ampfactor .*  ζ

	newp = bifpt.param + δp

	pb(orbitguess, set(br.params, br.param_lens, newp))[end] |> abs > 1 && @warn "PO Trap constraint not satisfied"

	if usedeflation
		verbose && println("\n--> Attempt branch switching\n--> Compute point on the current branch...")
		optn = _contParams.newtonOptions
		# find point on the first branch
		sol0, _, flag, _ = newton(pb, bifpt.x, set(br.params, br.param_lens, newp), optn; linearPO = linearPO, kwargs...)

		# find the bifurcated branch using deflation
		deflationOp = DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]), 1.0, [sol0])
		verbose && println("\n--> Compute point on bifurcated branch...")
		solbif, _, flag, _ = newton(pb, orbitguess, set(br.params, br.param_lens, newp), (@set optn.maxIter = 10*optn.maxIter), deflationOp; linearPO = linearPO, kwargs...)
		@assert flag "Deflated newton did not converge"
		orbitguess .= solbif
	end

	# TODO
	# we have to adjust the phase constraint.
	# Right now, it can be quite large.

	# perform continuation
	branch, u, tau = continuation(br.functional, orbitguess, set(br.params, br.param_lens, newp), br.param_lens, _contParams; linearPO = linearPO, printSolution = printSolution, linearAlgo = linearAlgo, kwargs...)

	#create a branch
	bppo = Pitchfork(bifpt.x, bifpt.param, set(br.params, br.param_lens, bifpt.param), br.param_lens, ζ, ζ, nothing, :nothing)

	return Branch(setproperties(branch; type = :PeriodicOrbit, functional = br.functional), bppo), u, tau
end
