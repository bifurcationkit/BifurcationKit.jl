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
This composite type implements Finite Differences based on a Trapezoidal rule to locate periodic orbits. More details (maths, notations, linear systems) can be found [here](https://rveltz.github.io/BifurcationKit.jl/dev/periodicOrbitTrapeze/). The arguments are as follows
- `F(x,p)` vector field
- `J` is the jacobian of `F` at `(x, p)`. It can assume three forms.
    1. Either `J` is a function and `J(x, p)` returns a `::AbstractMatrix`. In this case, the default arguments of `contParams::ContinuationPar` will make `continuation` work.
    2. Or `J` is a function and `J(x, p)` returns a function taking one argument `dx` and returning `dr` of the same type as `dx`. In our notation, `dr = J * dx`. In this case, the default parameters of `contParams::ContinuationPar` will not work and you have to use a Matrix Free linear solver, for example `GMRESIterativeSolvers`,
    3. Or `J` is a function and `J(x, p)` returns a variable `j` which can assume any type. Then, you must implement a linear solver `ls` as a composite type, subtype of `AbstractLinearSolver` which is called like `ls(j, rhs)` and which returns the solution of the jacobian linear system. See for example `examples/SH2d-fronts-cuda.jl`. This linear solver is passed to `NewtonPar(linsolver = ls)` which itself passed to `ContinuationPar`. Similarly, you have to implement an eigensolver `eig` as a composite type, subtype of `AbstractEigenSolver`.
- `Jᵗ = nothing` jacobian transpose of `F` (optional), useful for continuation of Fold of periodic orbits. it should not be passed in case the jacobian is a (sparse) matrix as it is computed internally, and it would be computed twice in that case.
- `d2F = nothing` second derivative of F (optional), useful for continuation of Fold of periodic orbits. It has the definition `d2F(x,p,dx1,dx2)`.`
- `ϕ` used to set a section for the phase constraint equation
- `xπ` used in the section for the phase constraint equation
- `M::Int` number of time slices
- `linsolver: = DefaultLS()` linear solver for each time slice, i.e. to solve `J⋅sol = rhs`. This is only needed for the computation of the Floquet multipliers.
- `isinplace::Bool` whether `F` and `J` are inplace functions (Experimental). In this case, the functions `F` and `J` must have the following definitions `(o, x, p) ->  F(o, x, p)` and `(o, x, p, dx) -> J(o, x, p, dx)`.
- `ongpu::Bool` whether the computation takes place on the gpu (Experimental)
- `massmatrix` a mass matrix. You can pass for example a sparse matrix. Default: identity matrix.

The scheme is as follows. We first consider a partition of ``[0,1]`` given by ``0<s_0<\\cdots<s_m=1`` and one looks for `T = x[end]` such that

 ``M_a\\cdot\\left(x_{i} - x_{i-1}\\right) - \\frac{T\\cdot h_i}{2} \\left(F(x_{i}) + F(x_{i-1})\\right) = 0,\\ i=1,\\cdots,m-1``

with ``u_{0} := u_{m-1}`` and the periodicity condition ``u_{m} - u_{1} = 0`` and

where ``h_1 = s_i-s_{i-1}``. ``M_a`` is a mass matrix. Finally, the phase of the periodic orbit is constrained by using a section (but you could use your own)

 ``\\sum_i\\langle x_{i} - x_{\\pi,i}, \\phi_{i}\\rangle=0.``

# Orbit guess
You will see below that you can evaluate the residual of the functional (and other things) by calling `pb(orbitguess, p)` on an orbit guess `orbitguess`. Note that `orbitguess` must be of size M * N + 1 where N is the number of unknowns in the state space and `orbitguess[M*N+1]` is an estimate of the period ``T`` of the limit cycle. More precisely, using the above notations, `orbitguess` must be ``orbitguess = [x_{1},x_{2},\\cdots,x_{M}, T]``.


# Functional
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
@with_kw struct PeriodicOrbitTrapProblem{TF, TJ, TJt, Td2F, Td3F, vectype, Tls <: AbstractLinearSolver, Tmesh, Tmass} <: AbstractPOFDProblem
	# Function F(x, par)
	F::TF = nothing

	# Jacobian of F w.r.t. x
	J::TJ = nothing

	# Jacobian transpose of F w.r.t. x. This is mainly used for matrix-free computation of Folds of limit cycles
	Jᵗ::TJt = nothing

	# Hessian of F w.r.t. x, useful for continuation of Fold of periodic orbits
	d2F::Td2F = nothing

	# 3rd differential of F w.r.t. x, useful for branch switching from branch of periodic orbits
	d3F::Td3F = nothing

	# variables to define a Section for the phase constraint equation
	ϕ::vectype = nothing
	xπ::vectype = nothing

	# discretisation of the time interval
	M::Int = 0
	mesh::Tmesh = TimeMesh(M)

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

	# whether the problem is nonautonomous
	isautonomous::Bool = true

	# mass matrix
	massmatrix::Tmass = nothing
end

@inline getTimeStep(pb::AbstractPOFDProblem, i::Int) = getTimeStep(pb.mesh, i)
getTimes(pb::AbstractPOFDProblem) = cumsum(collect(pb.mesh))
@inline hasmassmatrix(pb::PeriodicOrbitTrapProblem) = ~isnothing(pb.massmatrix)
@inline function getMassMatrix(pb::PeriodicOrbitTrapProblem, returnArray = false)
	if returnArray == false
		return hasmassmatrix(pb) ? pb.massmatrix : spdiagm( 0 => ones(pb.N))
	else
		return hasmassmatrix(pb) ? pb.massmatrix : LinearAlgebra.I(pb.N)
	end
end

# for a dummy constructor, useful for specifying the "algorithm" to look for periodic orbits,
# just call PeriodicOrbitTrapProblem()

function PeriodicOrbitTrapProblem(F, J, d2F, ϕ::vectype, xπ::vectype, m::Union{Int, vecmesh}, ls::AbstractLinearSolver = DefaultLS(); isinplace = false, ongpu = false, adaptmesh = false, massmatrix = nothing) where {vectype, vecmesh <: AbstractVector}
	_length = ϕ isa AbstractVector ? length(ϕ) : 0
	M = m isa Number ? m : length(m) + 1

	return PeriodicOrbitTrapProblem(F = F, J = J, d2F = d2F, ϕ = ϕ, xπ = xπ, M = M, mesh = TimeMesh(m), N = _length ÷ M, linsolver = ls, isinplace = isinplace, ongpu = ongpu, adaptmesh = adaptmesh, massmatrix = massmatrix)
end

PeriodicOrbitTrapProblem(F, J, ϕ::vectype, xπ::vectype, m::Union{Int, vecmesh}, ls::AbstractLinearSolver = DefaultLS(); isinplace = false, ongpu = false, adaptmesh = false, massmatrix = nothing) where {vectype, vecmesh <: AbstractVector} = PeriodicOrbitTrapProblem(F, J, nothing, ϕ, xπ, m, ls; isinplace = isinplace, ongpu = ongpu, adaptmesh = adaptmesh, massmatrix = massmatrix)

function PeriodicOrbitTrapProblem(F, J, d2F, ϕ::vectype, xπ::vectype, m::Union{Int, vecmesh}, N::Int, ls::AbstractLinearSolver = DefaultLS(); isinplace = false, ongpu = false, adaptmesh = false, massmatrix = nothing) where {vectype, vecmesh <: AbstractVector}
	M = m isa Number ? m : length(m) + 1
	# we use 0 * ϕ to create a copy filled with zeros, this is useful to keep the types
	prob = PeriodicOrbitTrapProblem(F = F, J = J, d2F = d2F, ϕ = similar(ϕ, N*M), xπ = similar(xπ, N*M), M = M, mesh = TimeMesh(m), N = N, linsolver = ls, isinplace = isinplace, ongpu = ongpu, adaptmesh = adaptmesh, massmatrix = massmatrix)

	prob.xπ .= 0
	prob.ϕ .= 0

	prob.xπ[1:length(xπ)] .= xπ
	prob.ϕ[1:length(ϕ)] .= ϕ
	return prob
end

PeriodicOrbitTrapProblem(F, J, ϕ::vectype, xπ::vectype, m::Union{Int, vecmesh}, N::Int, ls::AbstractLinearSolver = DefaultLS(); isinplace = false, ongpu = false, adaptmesh = false, massmatrix = nothing) where {vectype, vecmesh <: AbstractVector} = PeriodicOrbitTrapProblem(F, J, nothing, ϕ, xπ, m, N, ls; isinplace = isinplace, ongpu = ongpu, adaptmesh = adaptmesh, massmatrix = massmatrix)

PeriodicOrbitTrapProblem(F, J, m::Union{Int, vecmesh}, N::Int, ls::AbstractLinearSolver = DefaultLS(); isinplace = false, ongpu = false, adaptmesh = false, massmatrix = nothing) where {vectype, vecmesh <: AbstractVector} = PeriodicOrbitTrapProblem(F, J, nothing, zeros(N*(m isa Number ? m : length(m) + 1)), zeros(N*(m isa Number ? m : length(m) + 1)), m, N, ls; isinplace = isinplace, ongpu = ongpu, adaptmesh = adaptmesh, massmatrix = massmatrix)

# these functions extract the last component of the periodic orbit guess
@inline extractPeriodFDTrap(x::AbstractVector) = x[end]
# @inline extractPeriodFDTrap(x::BorderedArray)  = x.T

# these functions extract the time slices components
extractTimeSlices(x::AbstractVector, N, M) = @views reshape(x[1:end-1], N, M)
# extractTimeSlices(x::BorderedArray,  N, M) = x.u
extractTimeSlices(pb::PeriodicOrbitTrapProblem, x) = extractTimeSlices(x, pb.N, pb.M)

function POTrapScheme!(pb::AbstractPOFDProblem, dest, u1, u2, du1, du2, par, h::Number, tmp, linear::Bool = true; applyf::Bool = true)
	# this function implements the basic implicit scheme used for the time integration
	# because this function is called in a cyclic manner, we save in the variable tmp the value of F(u2) in order to avoid recomputing it in a subsequent call
	# basically tmp is F(u2)
	if linear
		dest .= tmp
		if applyf
		# tmp <- pb.F(u1, par)
		applyF(pb, tmp, u1, par) #TODO this line does not almost seem to be type stable in code_wartype, gives @_11::Union{Nothing, Tuple{Int64,Int64}}
		else
			applyJ(pb, tmp, u1, par, du1)
		end
		if hasmassmatrix(pb)
			dest .= pb.massmatrix * (du1 .- du2) .- h .* (dest .+ tmp)
		else
			dest .= @. (du1 - du2) - h * (dest + tmp)
		end
	else
		dest .-= h .* tmp
		# tmp <- pb.F(u1, par)
		applyF(pb, tmp, u1, par)
		dest .-= h .* tmp
	end
end
POTrapScheme!(pb::AbstractPOFDProblem, dest, u1, u2, par, h::Number, tmp, linear::Bool = true; applyf::Bool = true) = POTrapScheme!(pb::AbstractPOFDProblem, dest, u1, u2, u1, u2, par, h::Number, tmp, linear; applyf = applyf)

"""
This function implements the functional for finding periodic orbits based on finite differences using the Trapezoidal rule. It works for inplace / out of place vector fields `pb.F`
"""
function POTrapFunctional!(pb::AbstractPOFDProblem, out, u, par)
	M, N = size(pb)
	T = extractPeriodFDTrap(u)

	uc  = extractTimeSlices(pb, u)
	outc = extractTimeSlices(pb, out)

	# outc[:, M] plays the role of tmp until it is used just after the for-loop
	@views applyF(pb, outc[:, M], uc[:, M-1], par)

	h = T * getTimeStep(pb, 1)
	@views POTrapScheme!(pb, outc[:, 1], uc[:, 1], uc[:, M-1], par, h/2, outc[:, M])

	for ii in 2:M-1
		h = T * getTimeStep(pb, ii)
		# this function avoids computing F(uc[:, ii]) twice
		@views POTrapScheme!(pb, outc[:, ii], uc[:, ii], uc[:, ii-1], par, h/2, outc[:, M])
	end

	# closure condition ensuring a periodic orbit
	outc[:, M] .= @views uc[:, M] .- uc[:, 1]

	# this is for CuArrays.jl to work in the mode allowscalar(false)
	if onGpu(pb)
		return @views vcat(out[1:end-1], dot(u[1:end-1], pb.ϕ) - dot(pb.xπ, pb.ϕ)) # this is the phase condition
	else
			out[end] = @views dot(u[1:end-1], pb.ϕ) - dot(pb.xπ, pb.ϕ) #dot(u0c[:, 1] .- pb.xπ, pb.ϕ)
		return out
	end
end

"""
Matrix free expression of the Jacobian of the problem for computing periodic obits when evaluated at `u` and applied to `du`.
"""
function POTrapFunctionalJac!(pb::AbstractPOFDProblem, out, u, par, du)
	M, N = size(pb)
	T  = extractPeriodFDTrap(u)
	dT = extractPeriodFDTrap(du)

	uc = extractTimeSlices(pb, u)
	outc = extractTimeSlices(pb, out)
	duc = extractTimeSlices(pb, du)

	# compute the cyclic part
	@views Jc(pb, outc, u[1:end-1-N], par, T, du[1:end-N-1], outc[:, M])

	# outc[:, M] plays the role of tmp until it is used just after the for-loop
	tmp = @view outc[:, M]

	# we now compute the partial derivative w.r.t. the period T
	@views applyF(pb, tmp, uc[:, M-1], par)

	h = dT * getTimeStep(pb, 1)
	@views POTrapScheme!(pb, outc[:, 1], uc[:, 1], uc[:, M-1], par, h/2, tmp, false)
	for ii in 2:M-1
		h = dT * getTimeStep(pb, ii)
		@views POTrapScheme!(pb, outc[:, ii], uc[:, ii], uc[:, ii-1], par, h/2, tmp, false)
	end

	# closure condition ensuring a periodic orbit
	outc[:, M] .= @views duc[:, M] .- duc[:, 1]

	# this is for CuArrays.jl to work in the mode allowscalar(false)
	if onGpu(pb)
		return @views vcat(out[1:end-1], dot(du[1:end-1], pb.ϕ))
	else
		out[end] = @views dot(du[1:end-1], pb.ϕ)
		return out
	end
end

(pb::PeriodicOrbitTrapProblem)(u::AbstractVector, par) = POTrapFunctional!(pb, similar(u), u, par)
(pb::PeriodicOrbitTrapProblem)(u::AbstractVector, par, du) = POTrapFunctionalJac!(pb, similar(du), u, par, du)

####################################################################################################
# Matrix free expression of matrices related to the Jacobian Matrix of the PO functional
"""
Function to compute the Matrix-Free version of Aγ, see docs for its expression.
"""
function Agamma!(pb::PeriodicOrbitTrapProblem, outc, u0::AbstractVector, par, du::AbstractVector; γ = 1)
	# u0 of size N * M + 1
	# du of size N * M
	M, N = size(pb)
	T = extractPeriodFDTrap(u0)
	u0c = extractTimeSlices(pb, u0)

	# compute the cyclic part
	@views Jc(pb, outc, u0[1:end-1-N], par, T, du[1:end-N], outc[:, M])

	# closure condition ensuring a periodic orbit
	duc = reshape(du, N, M)
	outc[:, M] .= @views duc[:, M] .- γ .* duc[:, 1]
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
	M, N = size(pb)

	u0c = reshape(u0, N, M-1)
	duc = reshape(du, N, M-1)

	@views applyJ(pb, tmp, u0c[:, M-1], par, duc[:, M-1])

	h = T * getTimeStep(pb, 1)
	@views POTrapScheme!(pb, outc[:, 1], u0c[:, 1], u0c[:, M-1],
										 duc[:, 1], duc[:, M-1], par, h/2, tmp, true; applyf = false)

	for ii in 2:M-1
		h = T * getTimeStep(pb, ii)
		@views POTrapScheme!(pb, outc[:, ii], u0c[:, ii], u0c[:, ii-1],
											  duc[:, ii], duc[:, ii-1], par, h/2, tmp, true; applyf = false)
	end

	# we also return a Vector version of outc
	return vec(outc)
end

function Jc(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par, du::AbstractVector)
	M, N = size(pb)
	T = extractPeriodFDTrap(u0)

	out  = similar(du)
	outc = reshape(out, N, M-1)
	tmp  = similar(view(outc, :, 1))
	return @views Jc(pb, outc, u0[1:end-1-N], par, T, du, tmp)
end
####################################################################################################
"""
Matrix by blocks expression of the Jacobian for the PO functional computed at the space-time guess: `u0`
"""
function jacobianPOTrapBlock(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par; γ = 1)
	# extraction of various constants
	M, N = size(pb)

	Aγ = BlockArray(spzeros(M * N, M * N), N * ones(Int64, M),  N * ones(Int64, M))
	cylicPOTrapBlock!(pb, u0, par, Aγ)

	In = spdiagm( 0 => ones(N))
	Aγ[Block(M, 1)] = -γ * In
	Aγ[Block(M, M)] = In
	return Aγ
end

"""
This function populates Jc with the cyclic matrix using the different Jacobians
"""
function cylicPOTrapBlock!(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par, Jc::BlockArray)
	# extraction of various constants
	M, N = size(pb)
	T = extractPeriodFDTrap(u0)

	In = getMassMatrix(pb)

	u0c = extractTimeSlices(pb, u0)
	outc = similar(u0c)

	tmpJ = @views pb.J(u0c[:, 1], par)

	h = T * getTimeStep(pb, 1)
	Jn = In - (h/2) .* tmpJ
	Jc[Block(1, 1)] = Jn

	# we could do a Jn .= -I .- ... but we want to allow the sparsity pattern to vary
	Jn = @views -In - (h/2) .* pb.J(u0c[:, M-1], par)
	Jc[Block(1, M-1)] = Jn

	for ii in 2:M-1
		h = T * getTimeStep(pb, ii)
		Jn = -In - (h/2) .* tmpJ
		Jc[Block(ii, ii-1)] = Jn

		tmpJ = @views pb.J(u0c[:, ii], par)

		Jn = In - (h/2) .* tmpJ
		Jc[Block(ii, ii)] = Jn
	end
	return Jc
end

function cylicPOTrapBlock(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par)
	# extraction of various constants
	M, N = size(pb)
	Jc = BlockArray(spzeros((M - 1) * N, (M - 1) * N), N * ones(Int64, M-1),  N * ones(Int64, M-1))
	cylicPOTrapBlock!(pb, u0, par, Jc)
end

cylicPOTrapSparse(pb::PeriodicOrbitTrapProblem, orbitguess0, par) = blockToSparse(cylicPOTrapBlock(pb, orbitguess0, par))

"""
This method returns the jacobian of the functional G encoded in PeriodicOrbitTrapProblem using a Sparse representation.
"""
function (pb::PeriodicOrbitTrapProblem)(::Val{:JacFullSparse}, u0::AbstractVector, par; γ = 1.0, δ = 1e-9)
	# extraction of various constants
	M, N = size(pb)
	T = extractPeriodFDTrap(u0)
	AγBlock = jacobianPOTrapBlock(pb, u0, par; γ = γ)

	# we now set up the last line / column
	@views ∂TGpo = (pb(vcat(u0[1:end-1], T + δ), par) .- pb(u0, par)) ./ δ

	# this is "bad" for performance. Get converted to SparseMatrix at the next line
	Aγ = blockToSparse(AγBlock) # most of the computing time is here!!
	@views Aγ = hcat(Aγ, ∂TGpo[1:end-1])
	Aγ = vcat(Aγ, spzeros(1, N * M + 1))

	Aγ[N*M+1, 1:length(pb.ϕ)] .=  pb.ϕ
	Aγ[N*M+1, N*M+1] = ∂TGpo[end]
	return Aγ
end

"""
This method returns the jacobian of the functional G encoded in PeriodicOrbitTrapProblem using an inplace update. In case where the passed matrix J0 is a sparse one, it updates J0 inplace assuming that the sparsity pattern of J0 and dG(orbitguess0) are the same.
"""
@views function (pb::PeriodicOrbitTrapProblem)(::Val{:JacFullSparseInplace}, J0::Tj, u0::AbstractVector, par; γ = 1.0, δ = 1e-9) where Tj
		M, N = size(pb)
		T = extractPeriodFDTrap(u0)

		In = getMassMatrix(pb, ~(Tj <: SparseMatrixCSC))

		u0c = extractTimeSlices(pb, u0)
		outc = similar(u0c)

		tmpJ = pb.J(u0c[:, 1], par)

		h = T * getTimeStep(pb, 1)
		Jn = In - (h/2) .* tmpJ
		# setblock!(Jc, Jn, 1, 1)
		J0[1:N, 1:N] .= Jn

		Jn .= -In .- (h/2) .* pb.J(u0c[:, M-1], par)
		# setblock!(Jc, Jn, 1, M-1)
		J0[1:N, (M-2)*N+1:(M-1)*N] .= Jn

		for ii in 2:M-1
			h = T * getTimeStep(pb, ii)
			@. Jn = -In - h/2 * tmpJ
			# the next lines cost the most
			# setblock!(Jc, Jn, ii, ii-1)
			J0[(ii-1)*N+1:(ii)*N, (ii-2)*N+1:(ii-1)*N] .= Jn

			tmpJ .= pb.J(u0c[:, ii], par)

			@. Jn = In - h/2 * tmpJ
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

		# this following does not depend on u0, so it does not change. However we update it in case the caller updated the section somewhere else
		J0[N*M+1, 1:length(pb.ϕ)] .=  pb.ϕ

		return J0
end


@views function (pb::PeriodicOrbitTrapProblem)(::Val{:JacFullSparseInplace}, J0, u0::AbstractVector, par, indx; γ = 1.0, δ = 1e-9, updateborder = true)
	M, N = size(pb)
	T = extractPeriodFDTrap(u0)

	In = getMassMatrix(pb)

	u0c = extractTimeSlices(pb, u0)
	outc = similar(u0c)

	tmpJ = pb.J(u0c[:, 1], par)

	h = T * getTimeStep(pb, 1)
	Jn = In - tmpJ * (h/2)

	# setblock!(Jc, Jn, 1, 1)
	J0.nzval[indx[1, 1]] .= Jn.nzval

	Jn .= -In .- pb.J(u0c[:, M-1], par) .* (h/2)
	# setblock!(Jc, Jn, 1, M-1)
	J0.nzval[indx[1, M-1]] .= Jn.nzval

	for ii in 2:M-1
		h = T * getTimeStep(pb, ii)
		@. Jn = -In - tmpJ * (h/2)
		# the next lines cost the most
		# setblock!(Jc, Jn, ii, ii-1)
		J0.nzval[indx[ii, ii-1]] .= Jn.nzval

		tmpJ .= pb.J(u0c[:, ii], par)# * (h/2)

		@. Jn = In -  tmpJ * (h/2)
		# setblock!(Jc, Jn, ii, ii)
		J0.nzval[indx[ii,ii]] .= Jn.nzval
	end

	# setblock!(Aγ, -γ * In, M, 1)
	# useless to update:
		# J0[(M-1)*N+1:(M)*N, (1-1)*N+1:(1)*N] .= -In
	# setblock!(Aγ,  In,     M, M)
	# useless to update:
		# J0[(M-1)*N+1:(M)*N, (M-1)*N+1:(M)*N] .= In

	if updateborder
		# we now set up the last line / column
		∂TGpo = (pb(vcat(u0[1:end-1], T + δ), par) .- pb(u0, par)) ./ δ
		J0[:, end] .=  ∂TGpo

		# this following does not depend on u0, so it does not change. However we update it in case the caller updated the section somewhere else
		J0[N*M+1, 1:length(pb.ϕ)] .=  pb.ϕ
	end

	return J0
end

function (pb::PeriodicOrbitTrapProblem)(::Val{:JacCyclicSparse}, u0::AbstractVector, par, γ = 1.0)
	# extraction of various constants
	N = pb.N
	AγBlock = jacobianPOTrapBlock(pb, u0, par; γ = γ)

	# this is bad for performance. Get converted to SparseMatrix at the next line
	Aγ = blockToSparse(AγBlock) # most of the computing time is here!!
	# the following line is bad but still less costly than the previous one
	return Aγ[1:end-N, 1:end-N]
end

function (pb::PeriodicOrbitTrapProblem)(::Val{:BlockDiagSparse}, u0::AbstractVector, par)
	# extraction of various constants
	M, N = size(pb)
	T = extractPeriodFDTrap(u0)

	A_diagBlock = BlockArray(spzeros(M * N, M * N), N * ones(Int64, M),  N * ones(Int64, M))

	In = getMassMatrix(pb)

	u0c = reshape(u0[1:end-1], N, M)
	outc = similar(u0c)

	h = T * getTimeStep(pb, 1)
	@views Jn = In - h/2 .* pb.J(u0c[:, 1], par)
	A_diagBlock[Block(1, 1)] = Jn

	for ii in 2:M-1
		h = T * getTimeStep(pb, ii)
		@views Jn = In - h/2 .* pb.J(u0c[:, ii], par)
		A_diagBlock[Block(ii, ii)]= Jn
	end
	A_diagBlock[Block(M, M)]= In

	A_diag_sp = blockToSparse(A_diagBlock) # most of the computing time is here!!
	return A_diag_sp
end
####################################################################################################
# Utils
"""
$(SIGNATURES)

Compute the full trajectory associated to `x`. Mainly for plotting purposes.
"""
@views function getTrajectory(prob::AbstractPOFDProblem, u::AbstractVector, p)
	T = getPeriod(prob, u, p)
	M, N = size(prob)
	uv = u[1:end-1]
	uc = reshape(uv, N, M)
	return (t = cumsum(T .* collect(prob.mesh)), u = uc)
end

"""
$(SIGNATURES)

Compute the period of the periodic orbit associated to `x`.
"""
@inline getPeriod(prob::PeriodicOrbitTrapProblem, x, p) = extractPeriodFDTrap(x)

"""
$(SIGNATURES)

Compute `norm(du/dt)`
"""
@views function getTimeDiff(pb::PeriodicOrbitTrapProblem, u)
	M, N = size(pb)
	T = extractPeriodFDTrap(u)
	uc = reshape(u[1:end-1], N, M)
	return [norm(uc[:,ii+1].-uc[:,ii]) * T/M for ii in 1:M-1]
end

"""
$(SIGNATURES)

Compute the amplitude of the periodic orbit associated to `x`. The keyword argument `ratio = 1` is used as follows. If `length(x) = 1 + ratio * n`, the call returns the amplitude over `x[1:n]`.
"""
@views function getAmplitude(prob::PeriodicOrbitTrapProblem, x::AbstractVector, p; ratio = 1)
	n = div(length(x)-1, ratio)
	_max = maximum(x[1:n])
	_min = minimum(x[1:n])
	return maximum(_max .- _min)
end

"""
$(SIGNATURES)

Compute the maximum of the periodic orbit associated to `x`. The keyword argument `ratio = 1` is used as follows. If `length(x) = 1 + ratio * n`, the call returns the amplitude over `x[1:n]`.
"""
@views function getMaximum(prob::PeriodicOrbitTrapProblem, x::AbstractVector, p; ratio = 1)
	n = div(length(x)-1, ratio)
	return maximum(x[1:n])
end

# this function updates the section during the continuation run
@views function updateSection!(prob::PeriodicOrbitTrapProblem, x, par; stride = 0)
	M, N = size(prob)
	xc = extractTimeSlices(prob, x)
	T = extractPeriodFDTrap(x)

	# update the reference
	prob.xπ .= x[1:end-1]

	# update the normals
	for ii=0:M-1
		# ii2 = (ii+1)<= M ? ii+1 : ii+1-M
		prob.ϕ[ii*N+1:ii*N+N] .= prob.F(xc[:, ii+1], par) ./ M
	end

	return true
end
####################################################################################################
# Linear solvers for the jacobian of the functional G implemented by PeriodicOrbitTrapProblem
# composite type to encode the Aγ Operator and its associated cyclic matrix
abstract type AbstractPOTrapAγOperator end

# Matrix Free implementation of the operator Aγ
@with_kw mutable struct AγOperatorMatrixFree{Tvec, Tpb, Tpar} <: AbstractPOTrapAγOperator
	orbitguess::Tvec = zeros(1)				# point at which Aγ is evaluated, of size N * M + 1
	prob::Tpb = nothing						# PO functional, used when is_matrix_free = true
	par::Tpar = nothing						# parameters,    used when is_matrix_free = true
end

# implementation of Aγ which catches the LU decomposition of the cyclic matrix
@with_kw mutable struct AγOperatorLU{Tjc, Tpb} <: AbstractPOTrapAγOperator
	N::Int64 = 0							# dimension of time slice
	Jc::Tjc	= lu(spdiagm(0 => ones(1)))	    # lu factorisation of the cyclic matrix
	prob::Tpb = nothing		# PO functional
end

@with_kw struct AγOperatorSparseInplace{Tjc, Tjcf, Tind, Tpb} <: AbstractPOTrapAγOperator
	Jc::Tjc	=  nothing		# cyclic matrix
	Jcfact::Tjcf = nothing	# factorisation of Jc
	indx::Tind = nothing	# indices associated to the sparsity of Jc
	prob::Tpb = nothing		# PO functional
end

# functions to update the cyclic matrix
function (A::AγOperatorMatrixFree)(orbitguess::AbstractVector, par)
	copyto!(A.orbitguess, orbitguess)
	# update par for Matrix-Free
	A.par = par
	return A
end

function (A::AγOperatorLU)(orbitguess::AbstractVector, par)
	# we store the lu decomposition of the newly computed cyclic matrix
	A.Jc = SparseArrays.lu(cylicPOTrapSparse(A.prob, orbitguess, par))
	A
end

function (A::AγOperatorSparseInplace)(orbitguess::AbstractVector, par)
	# compute the cyclic matrix
	A.prob(Val(:JacFullSparseInplace), A.Jc, orbitguess, par, A.indx; updateborder = false)
	# update the Lu decomposition
	lu!(A.Jcfact, A.Jc)
	return A
end

# linear solvers designed specifically for AbstractPOTrapAγOperator
# this function is called whenever one wants to invert Aγ
@with_kw struct AγLinearSolver{Tls} <: AbstractLinearSolver
	# Linear solver to invert the cyclic matrix Jc contained in Aγ
	linsolver::Tls = DefaultLS()
end

@views function _combineSolutionAγLinearSolver(rhs, xbar, N)
	x = similar(rhs)
	x[1:end-N] .= xbar
	x[end-N+1:end] .= x[1:N] .+ rhs[end-N+1:end]
	return x
end

@views function (ls::AγLinearSolver)(A::AγOperatorMatrixFree, rhs)
	# dimension of a time slice
	N = A.prob.N
	# we invert the cyclic part Jc of Aγ
	xbar, flag, numiter = ls.linsolver(dx -> Jc(A.prob, A.orbitguess, A.par, dx), rhs[1:end - N])
	!flag && @warn "Matrix Free solver for Aγ did not converge"
	return _combineSolutionAγLinearSolver(rhs, xbar, N), flag, numiter
end

@views function (ls::AγLinearSolver)(A::AγOperatorLU, rhs)
	# dimension of a time slice
	N = A.N
	xbar, flag, numiter = ls.linsolver(A.Jc, rhs[1:end - N])
	!flag && @warn "Sparse solver for Aγ did not converge"
	return _combineSolutionAγLinearSolver(rhs, xbar, N), flag, numiter
end

@views function (ls::AγLinearSolver)(A::AγOperatorSparseInplace, rhs)
	# dimension of a time slice
	N = A.prob.N
	# we invert the cyclic part Jc of Aγ
	xbar, flag, numiter = ls.linsolver(A.Jcfact, rhs[1:end - N])
	!flag && @warn "Sparse solver for Aγ did not converge"
	return _combineSolutionAγLinearSolver(rhs, xbar, N), flag, numiter
end

####################################################################################################
# The following structure encodes the jacobian of a PeriodicOrbitTrapProblem which eases the use of PeriodicOrbitTrapBLS. It is made so that accessing the cyclic matrix Jc or Aγ is easier. It is combined with a specific linear solver. It is also a convenient structure for the computation of Floquet multipliers. Therefore, it is only used in the method continuationPOTrap
@with_kw struct POTrapJacobianBordered{T∂, Tag <: AbstractPOTrapAγOperator}
	∂TGpo::T∂ = nothing		# derivative of the PO functional G w.r.t. T
	Aγ::Tag					# Aγ Operator involved in the Jacobian of the PO functional
end

# this function is called whenever the jacobian of G has to be updated
function (J::POTrapJacobianBordered)(orbitguess0::AbstractVector, par; δ = 1e-9)
	# we compute the derivative of the problem w.r.t. the period TODO: remove this or improve!!
	T = extractPeriodFDTrap(orbitguess0)
	# TODO REMOVE CE vcat!
	@views J.∂TGpo .= (J.Aγ.prob(vcat(orbitguess0[1:end-1], T + δ), par) .- J.Aγ.prob(orbitguess0, par)) ./ δ

	# update Aγ
	J.Aγ(orbitguess0, par)

	# return J, needed to properly call the linear solver.
	return J
end

####################################################################################################
# linear solver for the PO functional, akin to a bordered linear solver
@with_kw struct PeriodicOrbitTrapBLS{Tl} <: AbstractLinearSolver
	linsolverbls::Tl = BorderingBLS(AγLinearSolver())	# linear solver
end

# Linear solver associated to POTrapJacobianBordered
function (ls::PeriodicOrbitTrapBLS)(J::POTrapJacobianBordered, rhs)
	# we solve the bordered linear system as follows
	dX, dl, flag, liniter = @views ls.linsolverbls(J.Aγ, J.∂TGpo[1:end-1],
	 										J.Aγ.prob.ϕ, J.∂TGpo[end],
										   rhs[1:end-1], rhs[end])
	return vcat(dX, dl), flag, sum(liniter)
end

# One could think that by implementing (ls::PeriodicOrbitTrapBLS)(J::POTrapJacobianBLS, rhs1, rhs2), we could speed up the computation of the linear Bordered system arising in the continuation process. However, we can note that this speed up would be observed only if a factorization of J.Aγ is available like an LU one. When such factorization is available, it is automatically stored as such in J.Aγ and so no speed up would be gained by implementing (ls::PeriodicOrbitTrapBLS)(J::POTrapJacobianBLS, rhs1, rhs2)

####################################################################################################
const DocStrLinearPO = """
- `linearPO = :BorderedLU`. Specify the choice of the linear algorithm, which must belong to `[:FullLU, :FullSparseInplace, :BorderedLU, :FullMatrixFree, :BorderedMatrixFree, :FullSparseInplace]`. This is used to select a way of inverting the jacobian `dG` of the functional G.
    - For `:FullLU`, we use the default linear solver based on a sparse matrix representation of `dG`. This matrix is assembled at each newton iteration. This is the default algorithm.
    - For `:FullSparseInplace`, this is the same as for `:FullLU` but the sparse matrix `dG` is updated inplace. This method allocates much less. In some cases, this is significantly faster than using `:FullLU`. Note that this method can only be used if the sparsity pattern of the jacobian is always the same.
    - For `:Dense`, same as above but the matrix `dG` is dense. It is also updated inplace. This option is useful to study ODE of small dimension.
    - For `:BorderedLU`, we take advantage of the bordered shape of the linear solver and use a LU decomposition to invert `dG` using a bordered linear solver. This is the default algorithm.
    - For `:BorderedSparseInplace`, this is the same as for `:BorderedLU` but the cyclic matrix `dG` is updated inplace. This method allocates much less. In some cases, this is significantly faster than using `:FullLU`. Note that this method can only be used if the sparsity pattern of the jacobian is always the same.
    - For `:FullMatrixFree`, a matrix free linear solver is used for `dG`: note that a preconditioner is very likely required here because of the cyclic shape of `dG` which affects negatively the convergence properties of GMRES.
    - For `:BorderedMatrixFree`, a matrix free linear solver is used but for `Jc` only (see docs): it means that `options.linsolver` is used to invert `Jc`. These two Matrix-Free options thus expose different part of the jacobian `dG` in order to use specific preconditioners. For example, an ILU preconditioner on `Jc` could remove the constraints in `dG` and lead to poor convergence. Of course, for these last two methods, a preconditioner is likely to be required.
"""
##########################
# newton wrappers
function _newton(probPO::PeriodicOrbitTrapProblem, orbitguess, par, options::NewtonPar, linearPO::Symbol = :FullLU; defOp::Union{Nothing, DeflationOperator{T, Tf, vectype}} = nothing, kwargs...) where {T, Tf, vectype}
	@assert orbitguess[end] >= 0 "The guess for the period should be positive, I get $(orbitguess[end])"
	@assert linearPO in (:Dense, :FullLU, :BorderedLU, :FullMatrixFree, :BorderedMatrixFree, :FullSparseInplace, :BorderedSparseInplace)
	M, N = size(probPO)

	if linearPO in (:Dense, :FullLU, :FullMatrixFree, :FullSparseInplace)
		if linearPO == :FullLU
			jac = (x, p) -> probPO(Val(:JacFullSparse), x, p)
		elseif linearPO == :FullSparseInplace
			# sparse matrix to hold the jacobian
			_J =  probPO(Val(:JacFullSparse), orbitguess, par)
			_indx = getBlocks(_J, N, M)
			# inplace modification of the jacobian _J
			jac = (x, p) -> probPO(Val(:JacFullSparseInplace), _J, x, p, _indx)
		elseif linearPO == :Dense
			_J =  probPO(Val(:JacFullSparse), orbitguess, par) |> Array
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
		if linearPO == :BorderedLU
			Aγ = AγOperatorLU(N = N, Jc = lu(spdiagm( 0 => ones(N * (M - 1)) )), prob = probPO)
			# linear solver
			lspo = PeriodicOrbitTrapBLS()
		elseif linearPO == :BorderedSparseInplace
			_J =  probPO(Val(:JacCyclicSparse), orbitguess, par)
			_indx = getBlocks(_J, N, M-1)
			# inplace modification of the jacobian _J
			Aγ = AγOperatorSparseInplace(Jc = _J,  Jcfact = lu(_J), prob = probPO, indx = _indx)
			lspo = PeriodicOrbitTrapBLS()

		else	# :BorderedMatrixFree
			Aγ = AγOperatorMatrixFree(prob = probPO, orbitguess = zeros(N * M + 1), par = par)
			# linear solver
			lspo = PeriodicOrbitTrapBLS(BorderingBLS(AγLinearSolver(options.linsolver)))
		end

		jacPO = POTrapJacobianBordered(zeros(N * M + 1), Aγ)

		if isnothing(defOp)
			return newton(probPO, jacPO, orbitguess, par, (@set options.linsolver = lspo); kwargs...)
		else
			return newton(probPO, jacPO, orbitguess, par, (@set options.linsolver = lspo), defOp; kwargs...)
		end
	end
end

"""
$(SIGNATURES)

This is the Krylov-Newton Solver for computing a periodic orbit using a functional G based on Finite Differences and a Trapezoidal rule.

# Arguments:
- `prob` a problem of type [`PeriodicOrbitTrapProblem`](@ref) encoding the functional G
- `orbitguess` a guess for the periodic orbit where `orbitguess[end]` is an estimate of the period of the orbit. It should be a vector of size `N * M + 1` where `M` is the number of time slices, `N` is the dimension of the phase space. This must be compatible with the numbers `N,M` in `prob`.
- `par` parameters to be passed to the functional
- `options` same as for the regular `newton` method
$DocStrLinearPO

# Output:
- solution
- history of residuals
- flag of convergence
- number of iterations
"""
newton(probPO::PeriodicOrbitTrapProblem, orbitguess, par, options::NewtonPar; linearPO::Symbol = :FullLU, kwargs...) = _newton(probPO, orbitguess, par, options, linearPO; defOp = nothing, kwargs...)

"""
	newton(probPO::PeriodicOrbitTrapProblem, orbitguess, options::NewtonPar, defOp::DeflationOperator{T, Tf, vectype}, linearPO = :BorderedLU; kwargs...) where {T, Tf, vectype}

This function is similar to `newton(probPO, orbitguess, options, linearPO; kwargs...)` except that it uses deflation in order to find periodic orbits different from the ones stored in `defOp`. We refer to the mentioned method for a full description of the arguments. The current method can be used in the vicinity of a Hopf bifurcation to prevent the Newton-Krylov algorithm from converging to the equilibrium point.
"""
newton(probPO::PeriodicOrbitTrapProblem, orbitguess::vectype, par, options::NewtonPar, defOp::DeflationOperator{Tp, Tdot, T, vectype}; linearPO::Symbol = :FullLU, kwargs...) where {Tp, Tdot, T, vectype} = _newton(probPO, orbitguess, par, options, linearPO; defOp = defOp, kwargs...)

####################################################################################################
# continuation wrapper
"""
	continuationPOTrap(probPO::PeriodicOrbitTrapProblem, orbitguess, par, lens::Lens, _contParams::ContinuationPar, linearAlgo::AbstractBorderedLinearSolver; linearPO = :BorderedLU, recordFromSolution = (u, p) -> (period = u[end],), kwargs...)

This is the continuation routine for computing a periodic orbit using a functional G based on Finite Differences and a Trapezoidal rule.

# Arguments
- `prob::PeriodicOrbitTrapProblem` encodes the functional G
- `orbitguess` a guess for the periodic orbit where `orbitguess[end]` is an estimate of the period of the orbit. It could be a vector of size `N * M + 1` where `M` is the number of time slices, `N` is the dimension of the phase space. This must be compatible with the numbers `N, M` in `prob`.
- `p0` set of parameters passed to the vector field
- `contParams` same as for the regular [`continuation`](@ref) method
- `linearAlgo` same as in [`continuation`](@ref)
- `updateSectionEveryStep = 0` updates the section every `updateSectionEveryStep` step during continuation
$DocStrLinearPO

Note that by default, the method prints the period of the periodic orbit as function of the parameter. This can be changed by providing your `recordFromSolution` argument.
"""
function continuationPOTrap(prob::PeriodicOrbitTrapProblem, orbitguess, par, lens::Lens, contParams::ContinuationPar, linearAlgo::AbstractBorderedLinearSolver; linearPO = :FullLU, recordFromSolution = (u, p) -> (period = u[end],), updateSectionEveryStep = 0, kwargs...)
	@assert orbitguess[end] >= 0 "The guess for the period should be positive. We found T = $(orbitguess[end])"
	@assert linearPO in (:Dense, :FullLU, :FullMatrixFree, :BorderedLU, :BorderedMatrixFree, :FullSparseInplace, :BorderedSparseInplace)

	M, N = size(prob)
	options = contParams.newtonOptions

	if computeEigenElements(contParams)
		contParams = @set contParams.newtonOptions.eigsolver =
		 FloquetQaD(contParams.newtonOptions.eigsolver)
		 # FloquetLU(N,M)
	end

	_finsol = get(kwargs, :finaliseSolution, nothing)
	_finsol2 = isnothing(_finsol) ? (z, tau, step, contResult; k2...) ->
		begin
			modCounter(step, updateSectionEveryStep) && updateSection!(prob, z.u, setParam(contResult, z.p))
			true
		end :
		(z, tau, step, contResult; prob = prob, k2...) ->
			begin
				modCounter(step, updateSectionEveryStep) && updateSection!(prob, z.u, setParam(contResult, z.p))
				_finsol(z, tau, step, contResult; prob = prob, k2...)
			end

	if linearPO in (:Dense, :FullLU, :FullMatrixFree, :FullSparseInplace)

		if linearPO == :FullLU
			jac = (x, p) -> FloquetWrapper(prob, prob(Val(:JacFullSparse), x, p), x, p)
		elseif linearPO == :FullSparseInplace
			# sparse matrix to hold the jacobian
			_J =  prob(Val(:JacFullSparse), orbitguess, par)
			_indx = getBlocks(_J, N, M)
			# inplace modification of the jacobian _J
			jac = (x, p) -> (prob(Val(:JacFullSparseInplace), _J, x, p, _indx); FloquetWrapper(prob, _J, x, p));
		elseif linearPO == :Dense
			_J =  prob(Val(:JacFullSparse), orbitguess, par) |> Array
			jac = (x, p) -> (prob(Val(:JacFullSparseInplace), _J, x, p); FloquetWrapper(prob, _J, x, p));
		else
		 	jac = (x, p) -> FloquetWrapper(prob, x, p)
		end

		# we have to change the Bordered linearsolver to cope with our type FloquetWrapper
		linearAlgo = @set linearAlgo.solver = FloquetWrapperLS(linearAlgo.solver)

		br, z, τ = continuation(
			prob, jac,
			orbitguess, par, lens,
			(@set contParams.newtonOptions.linsolver = FloquetWrapperLS(options.linsolver)), linearAlgo; kwargs...,
			recordFromSolution = recordFromSolution,
			finaliseSolution = _finsol2,)
	else
		if linearPO == :BorderedLU
			Aγ = AγOperatorLU(N = N, Jc = lu(spdiagm( 0 => ones(N * (M - 1)) )), prob = prob)
			# linear solver
			lspo = PeriodicOrbitTrapBLS()
		elseif linearPO == :BorderedSparseInplace
			_J =  prob(Val(:JacCyclicSparse), orbitguess, par)
			_indx = getBlocks(_J, N, M-1)
			# inplace modification of the jacobian _J
			Aγ = AγOperatorSparseInplace(Jc = _J,  Jcfact = lu(_J), prob = prob, indx = _indx)
			lspo = PeriodicOrbitTrapBLS()

		else	# :BorderedMatrixFree
			Aγ = AγOperatorMatrixFree(prob = prob, orbitguess = zeros(N * M + 1), par = par)
			# linear solver
			lspo = PeriodicOrbitTrapBLS(BorderingBLS(AγLinearSolver(options.linsolver)))
		end

		jacBD = POTrapJacobianBordered(zeros(N * M + 1), Aγ)
		jacPO = (x, p) -> FloquetWrapper(prob, jacBD(x, p), x, p)

		# we change the linear solver
		contParams = @set contParams.newtonOptions.linsolver = FloquetWrapperLS(lspo)

		# we have to change the Bordered linearsolver to cope with our type FloquetWrapper
		linearAlgo = @set linearAlgo.solver = contParams.newtonOptions.linsolver

		br, z, τ = continuation(prob, jacPO, orbitguess, par, lens,
			contParams, linearAlgo;
			kwargs...,
			recordFromSolution = recordFromSolution,
			finaliseSolution = _finsol2,)
	end
	return setproperties(br; type = :PeriodicOrbit, functional = prob), z, τ
end

"""
$(SIGNATURES)

This is the continuation routine for computing a periodic orbit using a functional G based on Finite Differences and a Trapezoidal rule.

# Arguments
- `prob::PeriodicOrbitTrapProblem` encodes the functional G
- `orbitguess` a guess for the periodic orbit where `orbitguess[end]` is an estimate of the period of the orbit. It could be a vector of size `N * M + 1` where `M` is the number of time slices, `N` is the dimension of the phase space. This must be compatible with the numbers `N, M` in `prob`.
- `p0` set of parameters passed to the vector field
- `contParams` same as for the regular [`continuation`](@ref) method
- `linearAlgo` same as in [`continuation`](@ref)
$DocStrLinearPO
- `updateSectionEveryStep = 1` updates the section every when `mod(step, updateSectionEveryStep) == 1` during continuation

Note that by default, the method prints the period of the periodic orbit as function of the parameter. This can be changed by providing your `recordFromSolution` argument.
"""
function continuation(prob::PeriodicOrbitTrapProblem, orbitguess, par, lens::Lens, _contParams::ContinuationPar; linearPO = :BorderedLU, recordFromSolution = (u, p) -> (period = u[end],), linearAlgo = nothing, updateSectionEveryStep = 0, kwargs...)
	_linearAlgo = isnothing(linearAlgo) ?  BorderingBLS(_contParams.newtonOptions.linsolver) : linearAlgo
	return continuationPOTrap(prob, orbitguess, par, lens, _contParams, _linearAlgo; linearPO = linearPO, recordFromSolution = recordFromSolution, updateSectionEveryStep = updateSectionEveryStep, kwargs...)
end

####################################################################################################
# function needed for automatic Branch switching from Hopf bifurcation point
function problemForBS(prob::PeriodicOrbitTrapProblem, F, dF, par, hopfpt, ζr::AbstractVector, orbitguess_a, period)
	M = length(orbitguess_a)
	N = length(ζr)

	# append period at the end of the initial guess
	orbitguess_v = reduce(vcat, orbitguess_a)
	orbitguess = vcat(vec(orbitguess_v), period) |> vec

	# update the problem
	probPO = setproperties(prob, N = N, F = F, J = dF, ϕ = zeros(N*M), xπ = zeros(N*M))

	probPO.ϕ[1:N] .= ζr
	probPO.xπ[1:N] .= hopfpt.x0

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
- `Jᵗ = (x, p) -> transpose(d_xF(x, p))` jacobian adjoint, it should be implemented in an efficient manner. For matrix-free methods, `transpose` is not readily available and the user must provide a dedicated method. In the case of sparse based jacobian, `Jᵗ` should not be passed as it is computed internally more efficiently, i.e. it avoid recomputing the jacobian as it would be if you pass `Jᵗ = (x, p) -> transpose(dF(x, p))`
- `δ` used internally to compute derivatives w.r.t the parameter `p`.
- `δp = 0.1` used to specify a particular guess for the parameter in the branch which is otherwise determined by `contParams.ds`. This allows to use a step larger than `contParams.dsmax`.
- `ampfactor = 1` factor which alter the amplitude of the bifurcated solution. Useful to magnify the bifurcated solution when the bifurcated branch is very steep.
- `usedeflation = true` whether to use nonlinear deflation (see [Deflated problems](@ref)) to help finding the guess on the bifurcated branch
- `linearPO = :BorderedLU` linear solver used for the Newton-Krylov solver when applied to [`PeriodicOrbitTrapProblem`](@ref).
- `recordFromSolution = (u, p) -> u[end]`, print method used in the bifurcation diagram, by default this prints the period of the periodic orbit.
- `linearAlgo = BorderingBLS()`, same as for [`continuation`](@ref)
- `kwargs` keywords arguments used for a call to the regular [`continuation`](@ref)
- `updateSectionEveryStep = 1` updates the section every when `mod(step, updateSectionEveryStep) == 1` during continuation
"""
function continuationPOTrapBPFromPO(br::AbstractBranchResult, ind_bif::Int, _contParams::ContinuationPar ; Jᵗ = nothing, δ = 1e-8, δp = 0.1, ampfactor = 1, usedeflation = true, linearPO = :BorderedLU, recordFromSolution = (u,p) -> (period = u[end],), linearAlgo = nothing, updateSectionEveryStep = 1, kwargs...)

	@assert br.functional isa PeriodicOrbitTrapProblem
	@assert abs(br.specialpoint[ind_bif].δ[1]) == 1 "Only simple bifurcation points are handled"

	verbose = get(kwargs, :verbosity, 0) > 0
	_linearAlgo = isnothing(linearAlgo) ?  BorderingBLS(_contParams.newtonOptions.linsolver) : linearAlgo

	bifpt = br.specialpoint[ind_bif]

	# let us compute the kernel
	λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
	verbose && print("--> computing nullspace...")
	ζ0 = geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[bifpt.idx].eigenvec, bifpt.ind_ev)
	verbose && println("Done!")
	# we normalize it by the sup norm because it could be too small/big in L2 norm
	ζ0 ./= norm(ζ0, Inf)

	pb = br.functional
	M, N = size(pb)

	ζ_a = MonodromyQaD(Val(:ExtractEigenVector), pb, bifpt.x, setParam(br, bifpt.param), real.(ζ0))
	ζ = reduce(vcat, ζ_a)

	orbitguess = copy(bifpt.x)
	orbitguess[1:end-1] .+= ampfactor .*  ζ

	newp = bifpt.param + δp

	pb(orbitguess, setParam(br, newp))[end] |> abs > 1 && @warn "PO Trap constraint not satisfied"

	if usedeflation
		verbose && println("\n--> Attempt branch switching\n--> Compute point on the current branch...")
		optn = _contParams.newtonOptions
		# find point on the first branch
		sol0, _, flag, _ = newton(pb, bifpt.x, setParam(br, newp), optn; linearPO = linearPO, kwargs...)

		# find the bifurcated branch using deflation
		deflationOp = DeflationOperator(2, (x,y) -> dot(x[1:end-1], y[1:end-1]), 1.0, [sol0])
		verbose && println("\n--> Compute point on bifurcated branch...")
		solbif, _, flag, _ = newton(pb, orbitguess, setParam(br, newp), (@set optn.maxIter = 10*optn.maxIter), deflationOp; linearPO = linearPO, kwargs...)
		@assert flag "Deflated newton did not converge"
		orbitguess .= solbif
	end

	# TODO
	# we have to adjust the phase constraint.
	# Right now, it can be quite large.

	# perform continuation
	branch, u, tau = continuation(br.functional, orbitguess, setParam(br, newp), br.lens, _contParams; linearPO = linearPO, recordFromSolution = recordFromSolution, linearAlgo = _linearAlgo, kwargs...)

	#create a branch
	bppo = Pitchfork(bifpt.x, bifpt.param, setParam(br, bifpt.param), br.lens, ζ, ζ, nothing, :nothing)

	return Branch(setproperties(branch; type = :PeriodicOrbit, functional = br.functional), bppo), u, tau
end
