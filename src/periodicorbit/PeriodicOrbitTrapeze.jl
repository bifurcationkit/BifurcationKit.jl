using BlockArrays, SparseArrays

"""

$(TYPEDEF)

Structure to describe a (Time) mesh using the time steps t_{i+1} - t_{i}. If the time steps are constant, we do not record them but, instead, we save the number of time steps effectively yielding a `TimeMesh{Int64}`.
"""
struct TimeMesh{T}
    ds::T
end

TimeMesh(M::Int64) = TimeMesh{Int64}(M)

@inline can_adapt(ms::TimeMesh{Ti}) where Ti = !(Ti == Int64)
Base.length(ms::TimeMesh{Ti}) where Ti = length(ms.ds)
Base.length(ms::TimeMesh{Ti}) where {Ti <: Int} = ms.ds

# access the time steps
@inline get_time_step(ms, i::Int) = ms.ds[i]
@inline get_time_step(ms::TimeMesh{Ti}, i::Int) where {Ti <: Int} = 1.0 / ms.ds

Base.collect(ms::TimeMesh) = ms.ds
Base.collect(ms::TimeMesh{Ti}) where {Ti <: Int} = repeat([get_time_step(ms, 1)], ms.ds)
####################################################################################################
const _trapezoid_jacobian_type = [Dense(), AutoDiffDense(), FullLU(), FullMatrixFree(), BorderedLU(), BorderedMatrixFree(), FullSparseInplace(), BorderedSparseInplace(), AutoDiffMF()]

const DocStrjacobianPOTrap = """
Specify the choice of the jacobian (and linear algorithm), `jacobian` must belong to `[FullLU(), FullSparseInplace(), Dense(), AutoDiffDense(), BorderedLU(), BorderedSparseInplace(), FullMatrixFree(), BorderedMatrixFree(), FullMatrixFreeAD]`. This is used to select a way of inverting the jacobian `dG` of the functional G.
- For `jacobian = FullLU()`, we use the default linear solver based on a sparse matrix representation of `dG`. This matrix is assembled at each newton iteration. This is the default algorithm.
- For `jacobian = FullSparseInplace()`, this is the same as for `FullLU()` but the sparse matrix `dG` is updated inplace. This method allocates much less. In some cases, this is significantly faster than using `FullLU()`. Note that this method can only be used if the sparsity pattern of the jacobian is always the same.
- For `jacobian = Dense()`, same as above but the matrix `dG` is dense. It is also updated inplace. This option is useful to study ODE of small dimension.
- For `jacobian = AutoDiffDense()`, evaluate the jacobian using ForwardDiff
- For `jacobian = BorderedLU()`, we take advantage of the bordered shape of the linear solver and use a LU decomposition to invert `dG` using a bordered linear solver.
- For `jacobian = BorderedSparseInplace()`, this is the same as for `BorderedLU()` but the cyclic matrix `dG` is updated inplace. This method allocates much less. In some cases, this is significantly faster than using `:BorderedLU`. Note that this method can only be used if the sparsity pattern of the jacobian is always the same.
- For `jacobian = FullMatrixFree()`, a matrix free linear solver is used for `dG`: note that a preconditioner is very likely required here because of the cyclic shape of `dG` which affects negatively the convergence properties of GMRES.
- For `jacobian = BorderedMatrixFree()`, a matrix free linear solver is used but for `Jc` only (see docs): it means that `options.linsolver` is used to invert `Jc`. These two Matrix-Free options thus expose different part of the jacobian `dG` in order to use specific preconditioners. For example, an ILU preconditioner on `Jc` could remove the constraints in `dG` and lead to poor convergence. Of course, for these last two methods, a preconditioner is likely to be required.
- For `jacobian = FullMatrixFreeAD()`, the evaluation map of the differential is derived using automatic differentiation. Thus, unlike the previous two cases, the user does not need to pass a Matrix-Free differential.
"""

"""
$(TYPEDEF)

This composite type implements Finite Differences based on a Trapezoidal rule (Order 2 in time) to locate periodic orbits. More details (maths, notations, linear systems) can be found [here](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/periodicOrbitTrapeze/).

The scheme is as follows. We first consider a partition of ``[0,1]`` given by ``0<s_0<\\cdots<s_m=1`` and one looks for `T = x[end]` such that

 ``M_a\\cdot\\left(x_{i} - x_{i-1}\\right) - \\frac{T\\cdot h_i}{2} \\left(F(x_{i}) + F(x_{i-1})\\right) = 0,\\ i=1,\\cdots,m-1``

with ``u_{0} := u_{m-1}`` and the periodicity condition ``u_{m} - u_{1} = 0`` and

where ``h_1 = s_i-s_{i-1}``. ``M_a`` is a mass matrix. Finally, the phase of the periodic orbit is constrained by using a section (but you could use your own)

 ``\\sum_i\\langle x_{i} - x_{\\pi,i}, \\phi_{i}\\rangle=0.``

## Fields
$(TYPEDFIELDS)

## Constructors

The structure can be created by calling `PeriodicOrbitTrapProblem(;kwargs...)`. For example, you can declare such a problem without vector field by doing

    PeriodicOrbitTrapProblem(M = 100)

# Orbit guess
You will see below that you can evaluate the residual of the functional (and other things) by calling `pb(orbitguess, p)` on an orbit guess `orbitguess`. Note that `orbitguess` must be a vector of size M * N + 1 where N is the number of unknowns in the state space and `orbitguess[M*N+1]` is an estimate of the period ``T`` of the limit cycle. More precisely, using the above notations, `orbitguess` must be ``orbitguess = [x_{1},x_{2},\\cdots,x_{M}, T]``.

Note that you can generate this guess from a function solution using `generate_solution` or `generate_ci_problem`.

# Functional
 A functional, hereby called `G`, encodes this problem. The following methods are available

- `pb(orbitguess, p)` evaluates the functional G on `orbitguess`
- `pb(orbitguess, p, du)` evaluates the jacobian `dG(orbitguess).du` functional at `orbitguess` on `du`
- `pb(Val(:JacFullSparse), orbitguess, p)` return the sparse matrix of the jacobian `dG(orbitguess)` at `orbitguess` without the constraints. It is called `A_γ` in the docs.
- `pb(Val(:JacFullSparseInplace), J, orbitguess, p)`. Same as `pb(Val(:JacFullSparse), orbitguess, p)` but overwrites `J` inplace. Note that the sparsity pattern must be the same independently of the values of the parameters or of `orbitguess`. In this case, this is significantly faster than `pb(Val(:JacFullSparse), orbitguess, p)`.
- `pb(Val(:JacCyclicSparse), orbitguess, p)` return the sparse cyclic matrix Jc (see the docs) of the jacobian `dG(orbitguess)` at `orbitguess`
- `pb(Val(:BlockDiagSparse), orbitguess, p)` return the diagonal of the sparse matrix of the jacobian `dG(orbitguess)` at `orbitguess`. This allows to design Jacobi preconditioner. Use `blockdiag`.

# Jacobian
$DocStrjacobianPOTrap

!!! note "GPU call"
    For these methods to work on the GPU, for example with `CuArrays` in mode `allowscalar(false)`, we face the issue that the function `_extract_period_fdtrap` won't be well defined because it is a scalar operation. Note that you must pass the option `ongpu = true` for the functional to be evaluated efficiently on the gpu.
"""
@with_kw_noshow struct PeriodicOrbitTrapProblem{Tprob, vectype, Tls <: AbstractLinearSolver, T, Tmass, Tjac} <: AbstractPOFDProblem
    "a bifurcation problem"
    prob_vf::Tprob = nothing

    "used to set a section for the phase constraint equation, of size N*M"
    ϕ::vectype = nothing

    "used in the section for the phase constraint equation, of size N*M"
    xπ::vectype = nothing

    "number of time slices"
    M::Int = 0

    "Mesh, see `TimeMesh`"
    mesh::TimeMesh{T} = TimeMesh(M)

    "dimension of the problem in case of an `AbstractVector`"
    N::Int = 0

    "linear solver for each time slice, i.e. to solve `J⋅sol = rhs`. This is only needed for the computation of the Floquet multipliers in a full matrix-free setting."
    linsolver::Tls = DefaultLS()

    "whether the computation takes place on the gpu (Experimental)"
    ongpu::Bool = false

    isautonomous::Bool = true

    "a mass matrix. You can pass for example a sparse matrix. Default: identity matrix."
    massmatrix::Tmass = nothing

    "updates the section every `update_section_every_step` step during continuation"
    update_section_every_step::UInt = 1

    "symbol which describes the type of jacobian used in Newton iterations (see below)."
    jacobian::Tjac = Dense()

    @assert jacobian in _trapezoid_jacobian_type "$jacobian is not defined for `PeriodicOrbitTrapProblem`. Pick one in $_trapezoid_jacobian_type"
end

function Base.show(io::IO, pb::PeriodicOrbitTrapProblem)
    println(io, "┌─ Trapezoid functional for periodic orbits")
    println(io, "├─ time slices    : ", pb.M)
    println(io, "├─ dimension      : ", get_state_dim(pb))
    println(io, "├─ jacobian       : ", pb.jacobian)
    println(io, "├─ update section : ", pb.update_section_every_step)
    println(io, "├─ # unknowns without phase condition : ", length(pb) - 1)
    println(io, "└─ inplace        : ", isinplace(pb))
end

@inline isinplace(pb::PeriodicOrbitTrapProblem) = isnothing(pb.prob_vf) ? false : isinplace(pb.prob_vf)
@inline get_time_step(pb::AbstractPOFDProblem, i::Int) = get_time_step(pb.mesh, i)
get_times(pb::AbstractPOFDProblem) = cumsum(collect(pb.mesh))
@inline hasmassmatrix(pb::PeriodicOrbitTrapProblem{Tprob, vectype, Tls, T, Tmass}) where {Tprob, vectype, Tls, T, Tmass} = ~(Tmass == Nothing)
@inline getparams(pb::PeriodicOrbitTrapProblem) = getparams(pb.prob_vf)
@inline getlens(pb::PeriodicOrbitTrapProblem) = getlens(pb.prob_vf)
@inline getdelta(pb::PeriodicOrbitTrapProblem) = getdelta(pb.prob_vf)
setparam(pb::PeriodicOrbitTrapProblem, p) = set(getparams(pb), getlens(pb), p)
@inline get_state_dim(pb::PeriodicOrbitTrapProblem) = pb.N
@inline length(pb::PeriodicOrbitTrapProblem) = pb.M * get_state_dim(pb)

# type unstable!
@inline function get_mass_matrix(pb::PeriodicOrbitTrapProblem, return_type_Array = false)
    if return_type_Array == false
        return hasmassmatrix(pb) ? pb.massmatrix : spdiagm( 0 => ones(pb.N))
    else
        return hasmassmatrix(pb) ? pb.massmatrix : LinearAlgebra.I(pb.N)
    end
end
# these functions extract the last component of the periodic orbit guess
@inline _extract_period_fdtrap(pb::PeriodicOrbitTrapProblem, x::AbstractVector) = on_gpu(pb) ? x[end:end] : x[end]
# these functions extract the time slices components
get_time_slices(x::AbstractVector, N, M) = @views reshape(x[begin:end-1], N, M)
get_time_slices(pb::PeriodicOrbitTrapProblem, x) = get_time_slices(x, pb.N, pb.M)

"""
$(TYPEDSIGNATURES)

Compute the period of the periodic orbit associated to `x`.
"""
@inline getperiod(prob::PeriodicOrbitTrapProblem, x, p) = _extract_period_fdtrap(prob, x)

# for a dummy constructor, useful for specifying the "algorithm" to look for periodic orbits,
# just call PeriodicOrbitTrapProblem()

function PeriodicOrbitTrapProblem(prob,
                                    ϕ::vectype,
                                    xπ::vectype,
                                    m::Union{Int, AbstractVector}, 
                                    ls::AbstractLinearSolver = DefaultLS(); 
                                    ongpu = false, 
                                    massmatrix = nothing) where {vectype}
    _length = ϕ isa AbstractVector ? length(ϕ) : 0
    M = m isa Number ? m : length(m) + 1

    return PeriodicOrbitTrapProblem(;prob_vf = prob, ϕ, xπ, M, mesh = TimeMesh(m), N = _length ÷ M, linsolver = ls, ongpu, massmatrix)
end

function PeriodicOrbitTrapProblem(prob_vf,
                                    ϕ::vectype,
                                    xπ::vectype,
                                    m::Union{Int, AbstractVector},
                                    N::Int,
                                    ls::AbstractLinearSolver = DefaultLS();
                                    ongpu = false,
                                    massmatrix = nothing,
                                    update_section_every_step::Int = 0,
                                    jacobian = Dense()) where {vectype}
    M = m isa Number ? m : length(m) + 1
    # we use 0 * ϕ to create a copy filled with zeros, this is useful to keep the types
    prob = PeriodicOrbitTrapProblem(;prob_vf,
                                    ϕ = similar(ϕ, N*M),
                                    xπ = similar(xπ, N*M),
                                    M,
                                    mesh = TimeMesh(m),
                                    N,
                                    linsolver = ls,
                                    ongpu,
                                    massmatrix,
                                    update_section_every_step,
                                    jacobian)

    prob.xπ .= 0
    prob.ϕ .= 0

    prob.xπ[eachindex(xπ)] .= xπ
    prob.ϕ[eachindex(ϕ)] .= ϕ
    return prob
end

# PeriodicOrbitTrapProblem(F, J, ϕ::vectype, xπ::vectype, m::Union{Int, vecmesh}, N::Int, ls::AbstractLinearSolver = DefaultLS(); ongpu = false, adaptmesh = false, massmatrix = nothing) where {vectype, vecmesh <: AbstractVector} = PeriodicOrbitTrapProblem(F, J, nothing, ϕ, xπ, m, N, ls; isinplace = isinplace, ongpu = ongpu, massmatrix = massmatrix)

PeriodicOrbitTrapProblem(prob_vf,
                        m::Union{Int, AbstractVector},
                        N::Int,
                        ls::AbstractLinearSolver = DefaultLS();
                        ongpu = false,
                        adaptmesh = false,
                        massmatrix = nothing) = PeriodicOrbitTrapProblem(prob_vf, zeros(N*(m isa Number ? m : length(m) + 1)), zeros(N*(m isa Number ? m : length(m) + 1)), m, N, ls; ongpu = ongpu, massmatrix = massmatrix)


# do not type h::Number because this will annoy CUDA
function potrap_scheme!(pb::AbstractPOFDProblem, 
                        dest, 
                        u1, u2, 
                        du1, du2, 
                        par, h, 
                        tmp, 
                        linear::Bool = true; 
                        applyf::Bool = true)
    # this function implements the basic implicit scheme used for the time integration
    # because this function is called in a cyclic manner, we save the value of F(u2) 
    # in the variable tmp in order to avoid recomputing it in a subsequent call
    # basically tmp is F(u2)
    # applyf: if true use F and dF otherwise
    if linear
        dest .= tmp
        if applyf
            # tmp <- pb.F(u1, par)
            residual!(pb.prob_vf, tmp, u1, par) #TODO this line does not almost seem to be type stable in code_wartype, gives @_11::Union{Nothing, Tuple{Int64,Int64}}
        else
            applyJ(pb, tmp, u1, par, du1)
        end
        if hasmassmatrix(pb)
            dest .= pb.massmatrix * (du1 .- du2) .- h .* (dest .+ tmp)
        else
            @. dest = (du1 - du2) - h * (dest + tmp)
        end
    else # used for jvp
        dest .-= h .* tmp
        # tmp <- pb.F(u1, par)
        residual!(pb.prob_vf, tmp, u1, par)
        dest .-= h .* tmp
    end
end
potrap_scheme!(pb::AbstractPOFDProblem, dest, u1, u2, par, h, tmp, linear::Bool = true; applyf::Bool = true) = potrap_scheme!(pb, dest, u1, u2, u1, u2, par, h, tmp, linear; applyf)

"""
This function implements the functional for finding periodic orbits based on finite differences using the Trapezoidal rule. It works for inplace / out of place vector fields `pb.F`
"""
function residual!(pb::AbstractPOFDProblem, out, u, par)
        M, N = size(pb)
        T = getperiod(pb, u, nothing)

        uc = get_time_slices(pb, u)
        outc = get_time_slices(pb, out)

        # outc[:, M] plays the role of tmp until it is used just after the for-loop
        @views residual!(pb.prob_vf, outc[:, M], uc[:, M-1], par)

        h = T * get_time_step(pb, 1)
        # fastest is to do out[:, i] = x
        @views potrap_scheme!(pb, outc[:, 1], uc[:, 1], uc[:, M-1], par, h/2, outc[:, M])

        for ii in 2:M-1
            h = T * get_time_step(pb, ii)
            # this function avoids computing F(uc[:, ii]) twice
            @views potrap_scheme!(pb, outc[:, ii], uc[:, ii], uc[:, ii-1], par, h/2, outc[:, M])
        end

        # closure condition ensuring a periodic orbit
        outc[:, M] .= @views uc[:, M] .- uc[:, 1]

        # this is for CuArrays.jl to work in the mode allowscalar(false)
        if on_gpu(pb)
            return @views vcat(out[begin:end-1], LA.dot(u[begin:end-1], pb.ϕ) - LA.dot(pb.xπ, pb.ϕ)) # this is the phase condition
        else
            out[end] = @views LA.dot(u[begin:end-1], pb.ϕ) - LA.dot(pb.xπ, pb.ϕ)
            return out
        end
end

"""
Matrix free expression (jvp) of the Jacobian of the problem for computing periodic obits when evaluated at `u` and applied to `du`.
"""
function jvp!(pb::PeriodicOrbitTrapProblem, out, u, par, du)
    M, N = size(pb)
    T  = _extract_period_fdtrap(pb, u)
    dT = _extract_period_fdtrap(pb, du)

    uc = get_time_slices(pb, u)
    outc = get_time_slices(pb, out)
    duc = get_time_slices(pb, du)

    # compute the cyclic part
    @views Jc(pb, outc, u[begin:end-1-N], par, T, du[begin:end-N-1], outc[:, M])

    # outc[:, M] plays the role of tmp until it is used just after the for-loop
    tmp = @view outc[:, M]

    # we now compute the partial derivative w.r.t. the period T
    @views residual!(pb.prob_vf, tmp, uc[:, M-1], par)

    h = dT * get_time_step(pb, 1)
    @views potrap_scheme!(pb, outc[:, 1], uc[:, 1], uc[:, M-1], par, h/2, tmp, false)
    for ii in 2:M-1
        h = dT * get_time_step(pb, ii)
        @views potrap_scheme!(pb, outc[:, ii], uc[:, ii], uc[:, ii-1], par, h/2, tmp, false)
    end

    # closure condition ensuring a periodic orbit
    outc[:, M] .= @views duc[:, M] .- duc[:, 1]

    # this is for CuArrays.jl to work in the mode allowscalar(false)
    if on_gpu(pb)
        return @views vcat(out[begin:end-1], LA.dot(du[begin:end-1], pb.ϕ))
    else
        out[end] = @views LA.dot(du[begin:end-1], pb.ϕ)
        return out
    end
end


residual(pb::PeriodicOrbitTrapProblem, u::AbstractVector, par) = residual!(pb, similar(u), u, par)
jvp(pb::PeriodicOrbitTrapProblem, u::AbstractVector, par, du) = jvp!(pb, similar(du), u, par, du)

####################################################################################################
# Matrix free expression of matrices related to the Jacobian Matrix of the PO functional
"""
Function to compute the Matrix-Free version of Aγ, see docs for its expression.
"""
function Aγ!(pb::PeriodicOrbitTrapProblem, outc, u0::AbstractVector, par, du::AbstractVector; γ = 1)
    # u0 of size N * M + 1
    # du of size N * M
    M, N = size(pb)
    T = _extract_period_fdtrap(pb, u0)
    u0c = get_time_slices(pb, u0)

    # compute the cyclic part
    @views Jc(pb, outc, u0[begin:end-1-N], par, T, du[begin:end-N], outc[:, M])

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

    h = T * get_time_step(pb, 1)
    @views potrap_scheme!(pb, outc[:, 1], u0c[:, 1], u0c[:, M-1],
                                          duc[:, 1], duc[:, M-1], par, h/2, tmp, true; applyf = false)

    for ii in 2:M-1
        h = T * get_time_step(pb, ii)
        @views potrap_scheme!(pb, outc[:, ii], u0c[:, ii], u0c[:, ii-1],
                                               duc[:, ii], duc[:, ii-1], par, h/2, tmp, true; applyf = false)
    end

    # we also return a Vector version of outc
    return vec(outc)
end

function Jc(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par, du::AbstractVector)
    M, N = size(pb)
    T = _extract_period_fdtrap(pb, u0)
    out  = similar(du)
    outc = reshape(out, N, M-1)
    tmp  = similar(view(outc, :, 1))
    return @views Jc(pb, outc, u0[begin:end-1-N], par, T, du, tmp)
end
####################################################################################################
"""
Matrix by blocks expression of the Jacobian for the PO functional computed at the space-time guess: `u0`
"""
function jacobian_potrap_block(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par; γ = 1)
    # extraction of various constants
    M, N = size(pb)

    Aγ = BlockArray(spzeros(M * N, M * N), N * ones(Int64, M),  N * ones(Int64, M))
    cylic_potrap_block!(pb, u0, par, Aγ)

    Iₙ = spdiagm( 0 => ones(N))
    Aγ[Block(M, 1)] = (-γ) * Iₙ
    Aγ[Block(M, M)] = Iₙ
    return Aγ
end

"""
This function populates Jc with the cyclic matrix using the different Jacobians
"""
function cylic_potrap_block!(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par, Jc::BlockArray)
    # extraction of various constants
    M, N = size(pb)
    T = _extract_period_fdtrap(pb, u0)

    Iₙ = get_mass_matrix(pb)

    u0c = get_time_slices(pb, u0)
    outc = similar(u0c)

    tmpJ = @views jacobian(pb.prob_vf, u0c[:, 1], par)

    h = T * get_time_step(pb, 1)
    Jn = Iₙ - (h/2) .* tmpJ
    Jc[Block(1, 1)] = Jn

    # we could do a Jn .= -I .- ... but we want to allow the sparsity pattern to vary
    Jn = @views -Iₙ - (h/2) .* jacobian(pb.prob_vf, u0c[:, M-1], par)
    Jc[Block(1, M-1)] = Jn

    for ii in 2:M-1
        h = T * get_time_step(pb, ii)
        Jn = -Iₙ - (h/2) .* tmpJ
        Jc[Block(ii, ii-1)] = Jn

        tmpJ = @views jacobian(pb.prob_vf, u0c[:, ii], par)

        Jn = Iₙ - (h/2) .* tmpJ
        Jc[Block(ii, ii)] = Jn
    end
    return Jc
end

function cylic_potrap_block(pb::PeriodicOrbitTrapProblem, u0::AbstractVector, par)
    # extraction of various constants
    M, N = size(pb)
    Jc = BlockArray(spzeros((M - 1) * N, (M - 1) * N), N * ones(Int64, M-1),  N * ones(Int64, M-1))
    cylic_potrap_block!(pb, u0, par, Jc)
end

cylic_potrap_sparse(pb::PeriodicOrbitTrapProblem, orbitguess0, par) = block_to_sparse(cylic_potrap_block(pb, orbitguess0, par))

"""
This method returns the jacobian of the functional G encoded in PeriodicOrbitTrapProblem using a Sparse representation.
"""
function (pb::PeriodicOrbitTrapProblem)(::Val{:JacFullSparse}, u0::AbstractVector, par; γ = 1, δ = convert(eltype(u0), 1e-9))
    # extraction of various constants
    M, N = size(pb)
    T = _extract_period_fdtrap(pb, u0)
    AγBlock = jacobian_potrap_block(pb, u0, par; γ)

    # we now set up the last line / column
    @views ∂TGpo = (residual(pb, vcat(u0[begin:end-1], T + δ), par) .- residual(pb, u0, par)) ./ δ

    # this is "bad" for performance. Get converted to SparseMatrix at the next line
    Aγ = block_to_sparse(AγBlock) # most of the computing time is here!!
    @views Aγ = hcat(Aγ, ∂TGpo[begin:end-1])
    Aγ = vcat(Aγ, spzeros(1, N * M + 1))

    Aγ[N*M+1, eachindex(pb.ϕ)] .= pb.ϕ
    Aγ[N*M+1, N*M+1] = ∂TGpo[end]
    return Aγ
end

"""
This method returns the jacobian of the functional G encoded in PeriodicOrbitTrapProblem using an inplace update. In case where the passed matrix J0 is a sparse one, it updates J0 inplace assuming that the sparsity pattern of J0 and dG(orbitguess0) are the same.
"""
@views function (pb::PeriodicOrbitTrapProblem)(::Val{:JacFullSparseInplace}, J0::Tj, u0::AbstractVector, par; γ = 1, δ = convert(eltype(u0), 1e-9)) where Tj
        M, N = size(pb)
        T = _extract_period_fdtrap(pb, u0)

        Iₙ = get_mass_matrix(pb, ~(Tj <: SparseMatrixCSC))

        u0c = get_time_slices(pb, u0)
        outc = similar(u0c)

        tmpJ = jacobian(pb.prob_vf, u0c[:, 1], par)

        h = T * get_time_step(pb, 1)
        Jn = Iₙ - (h/2) .* tmpJ
        # setblock!(Jc, Jn, 1, 1)
        J0[1:N, 1:N] .= Jn

        Jn .= -Iₙ .- (h/2) .* jacobian(pb.prob_vf, u0c[:, M-1], par)
        # setblock!(Jc, Jn, 1, M-1)
        J0[1:N, (M-2)*N+1:(M-1)*N] .= Jn

        for ii in 2:M-1
            h = T * get_time_step(pb, ii)
            @. Jn = -Iₙ - h/2 * tmpJ
            # the next lines cost the most
            # setblock!(Jc, Jn, ii, ii-1)
            J0[(ii-1)*N+1:(ii)*N, (ii-2)*N+1:(ii-1)*N] .= Jn

            tmpJ .= jacobian(pb.prob_vf, u0c[:, ii], par)

            @. Jn = Iₙ - h/2 * tmpJ
            # setblock!(Jc, Jn, ii, ii)
            J0[(ii-1)*N+1:(ii)*N, (ii-1)*N+1:(ii)*N] .= Jn
        end

        # setblock!(Aγ, -γ * Iₙ, M, 1)
        # useless to update:
            # J0[(M-1)*N+1:(M)*N, (1-1)*N+1:(1)*N] .= -Iₙ
        # setblock!(Aγ,  Iₙ,     M, M)
        # useless to update:
            # J0[(M-1)*N+1:(M)*N, (M-1)*N+1:(M)*N] .= Iₙ

        # we now set up the last line / column
        ∂TGpo = (residual(pb,vcat(u0[begin:end-1], T + δ), par) .- residual(pb,u0, par)) ./ δ
        J0[:, end] .=  ∂TGpo

        # this following does not depend on u0, so it does not change. However we update it in case the caller updated the section somewhere else
        J0[N*M+1, eachindex(pb.ϕ)] .=  pb.ϕ

        return J0
end


@views function (pb::PeriodicOrbitTrapProblem)(::Val{:JacFullSparseInplace}, J0, u0::AbstractVector, par, indx; γ = 1, δ = convert(eltype(u0), 1e-9), updateborder::Bool = true)
    M, N = size(pb)
    T = _extract_period_fdtrap(pb, u0)

    Iₙ = get_mass_matrix(pb)

    u0c = get_time_slices(pb, u0)
    outc = similar(u0c)

    tmpJ = jacobian(pb.prob_vf, u0c[:, 1], par)

    h = T * get_time_step(pb, 1)
    Jn = Iₙ - tmpJ * (h/2)

    # setblock!(Jc, Jn, 1, 1)
    J0.nzval[indx[1, 1]] .= Jn.nzval

    Jn .= -Iₙ .- jacobian(pb.prob_vf, u0c[:, M-1], par) .* (h/2)
    # setblock!(Jc, Jn, 1, M-1)
    J0.nzval[indx[1, M-1]] .= Jn.nzval

    for ii in 2:M-1
        h = T * get_time_step(pb, ii)
        @. Jn = -Iₙ - tmpJ * (h/2)
        # the next lines cost the most
        # setblock!(Jc, Jn, ii, ii-1)
        J0.nzval[indx[ii, ii-1]] .= Jn.nzval

        tmpJ .= jacobian(pb.prob_vf, u0c[:, ii], par)# * (h/2)

        @. Jn = Iₙ -  tmpJ * (h/2)
        # setblock!(Jc, Jn, ii, ii)
        J0.nzval[indx[ii, ii]] .= Jn.nzval
    end

    # setblock!(Aγ, -γ * Iₙ, M, 1)
    # useless to update:
        # J0[(M-1)*N+1:(M)*N, (1-1)*N+1:(1)*N] .= -Iₙ
    # setblock!(Aγ,  Iₙ,     M, M)
    # useless to update:
        # J0[(M-1)*N+1:(M)*N, (M-1)*N+1:(M)*N] .= Iₙ

    if updateborder
        # we now set up the last line / column
        ∂TGpo = (residual(pb, vcat(u0[begin:end-1], T + δ), par) .- residual(pb, u0, par)) ./ δ
        J0[:, end] .=  ∂TGpo

        # this following does not depend on u0, so it does not change. However we update it in case the caller updated the section somewhere else
        J0[N*M+1, eachindex(pb.ϕ)] .=  pb.ϕ
    end

    return J0
end

function (pb::PeriodicOrbitTrapProblem)(::Val{:JacCyclicSparse}, u0::AbstractVector, par, γ = 1)
    # extraction of various constants
    N = pb.N
    AγBlock = jacobian_potrap_block(pb, u0, par; γ = γ)

    # this is bad for performance. Get converted to SparseMatrix at the next line
    Aγ = block_to_sparse(AγBlock) # most of the computing time is here!!
    # the following line is bad but still less costly than the previous one
    return Aγ[begin:end-N, begin:end-N]
end

function (pb::PeriodicOrbitTrapProblem)(::Val{:BlockDiagSparse}, u0::AbstractVector, par)
    # extraction of various constants
    M, N = size(pb)
    T = _extract_period_fdtrap(pb, u0)

    A_diagBlock = BlockArray(spzeros(M * N, M * N), N * ones(Int64, M),  N * ones(Int64, M))

    In = get_mass_matrix(pb)

    u0c = reshape(u0[begin:end-1], N, M)
    outc = similar(u0c)

    h = T * get_time_step(pb, 1)
    @views Jn = In - h/2 .* jacobian(pb.prob_vf, u0c[:, 1], par)
    A_diagBlock[Block(1, 1)] = Jn

    for ii in 2:M-1
        h = T * get_time_step(pb, ii)
        @views Jn = In - h/2 .* jacobian(pb.prob_vf, u0c[:, ii], par)
        A_diagBlock[Block(ii, ii)]= Jn
    end
    A_diagBlock[Block(M, M)]= In

    A_diag_sp = block_to_sparse(A_diagBlock) # most of the computing time is here!!
    return A_diag_sp
end
####################################################################################################
# Utils
"""
$(TYPEDSIGNATURES)

Compute the full periodic orbit associated to `x`. Mainly for plotting purposes.
"""
@views function get_periodic_orbit(prob::AbstractPOFDProblem, u, p)
    T = getperiod(prob, u, p)
    M, N = size(prob)
    uv = u[begin:end-1]
    uc = reshape(uv, N, M)
    return SolPeriodicOrbit(t = cumsum(T .* collect(prob.mesh)), u = uc)
end
get_periodic_orbit(prob::AbstractPOFDProblem, x, p::Real) = get_periodic_orbit(prob, x, setparam(prob, p))

# this function updates the section during the continuation run
@views function updatesection!(prob::PeriodicOrbitTrapProblem, x, par)
    @debug "Update section TRAP"
    M, N = size(prob)
    xc = get_time_slices(prob, x)
    T = _extract_period_fdtrap(prob, x)

    # update the reference point
    prob.xπ .= x[begin:end-1]

    # update the normals
    for ii in 0:M-1
        # ii2 = (ii+1)<= M ? ii+1 : ii+1-M
        residual!(prob.prob_vf, prob.ϕ[ii*N+1:ii*N+N], xc[:, ii+1], par)
        prob.ϕ[ii*N+1:ii*N+N] ./= M
    end
    return true
end
####################################################################################################
# Linear solvers for the jacobian of the functional G implemented by PeriodicOrbitTrapProblem
# composite type to encode the Aγ Operator and its associated cyclic matrix
abstract type AbstractPOTrapAγOperator end

# Matrix Free implementation of the operator Aγ
@with_kw mutable struct AγOperatorMatrixFree{Tvec, Tpb, Tpar} <: AbstractPOTrapAγOperator
    orbitguess::Tvec = zeros(1) # point at which Aγ is evaluated, of size N * M + 1
    prob::Tpb = nothing         # PO functional, used when is_matrix_free = true
    par::Tpar = nothing         # parameters,    used when is_matrix_free = true
end

# implementation of Aγ which catches the LU decomposition of the cyclic matrix
@with_kw mutable struct AγOperatorLU{Tjc, Tpb} <: AbstractPOTrapAγOperator
    N::Int64 = 0                           # dimension of time slice
    Jc::Tjc    = lu(spdiagm(0 => ones(1))) # lu factorisation of the cyclic matrix
    prob::Tpb = nothing                    # PO functional
end

@with_kw struct AγOperatorSparseInplace{Tjc, Tjcf, Tind, Tpb} <: AbstractPOTrapAγOperator
    Jc::Tjc    =  nothing     # cyclic matrix
    Jcfact::Tjcf = nothing    # factorisation of Jc
    indx::Tind = nothing      # indices associated to the sparsity of Jc
    prob::Tpb = nothing       # PO functional
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
    A.Jc = SparseArrays.lu(cylic_potrap_sparse(A.prob, orbitguess, par))
    A
end

function (A::AγOperatorSparseInplace)(orbitguess::AbstractVector, par)
    # compute the cyclic matrix
    A.prob(Val(:JacFullSparseInplace), A.Jc, orbitguess, par, A.indx; updateborder = false)
    # update the Lu decomposition
    LA.lu!(A.Jcfact, A.Jc)
    return A
end

@views function apply(A::AγOperatorSparseInplace, dx)
    out = similar(dx)
    M, N = size(A.prob)
    out1 = apply(A.Jc, dx[begin:end-N])
    return vcat(out1, -dx[begin:N] .+ dx[end-N+1:end])
end

# linear solvers designed specifically for AbstractPOTrapAγOperator
# this function is called whenever one wants to invert Aγ
@with_kw struct AγLinearSolver{Tls} <: AbstractLinearSolver
    # Linear solver to invert the cyclic matrix Jc contained in Aγ
    linsolver::Tls = DefaultLS()
end

@views function _combine_solution_Aγ_linearsolver(rhs, xbar, N)
    x = similar(rhs)
    x[begin:end-N] .= xbar
    x[end-N+1:end] .= x[begin:N] .+ rhs[end-N+1:end]
    return x
end

@views function (ls::AγLinearSolver)(A::AγOperatorMatrixFree, rhs)
    # dimension of a time slice
    N = A.prob.N
    # we invert the cyclic part Jc of Aγ
    xbar, flag, numiter = ls.linsolver(dx -> Jc(A.prob, A.orbitguess, A.par, dx), rhs[begin:end - N])
    !flag && @warn "Matrix Free solver for Aγ did not converge"
    return _combine_solution_Aγ_linearsolver(rhs, xbar, N), flag, numiter
end

@views function (ls::AγLinearSolver)(A::AγOperatorLU, rhs)
    # dimension of a time slice
    N = A.N
    xbar, flag, numiter = ls.linsolver(A.Jc, rhs[begin:end - N])
    !flag && @warn "Sparse solver for Aγ did not converge"
    return _combine_solution_Aγ_linearsolver(rhs, xbar, N), flag, numiter
end

@views function (ls::AγLinearSolver)(A::AγOperatorSparseInplace, rhs)
    # dimension of a time slice
    N = A.prob.N
    # we invert the cyclic part Jc of Aγ
    xbar, flag, numiter = ls.linsolver(A.Jcfact, rhs[begin:end - N])
    !flag && @warn "Sparse solver for Aγ did not converge"
    return _combine_solution_Aγ_linearsolver(rhs, xbar, N), flag, numiter
end

####################################################################################################
# The following structure encodes the jacobian of a PeriodicOrbitTrapProblem which eases the use of PeriodicOrbitTrapBLS. It is made so that accessing the cyclic matrix Jc or Aγ is easier. It is combined with a specific linear solver. It is also a convenient structure for the computation of Floquet multipliers. Therefore, it is only used in the method continuation_potrap
@with_kw struct POTrapJacobianBordered{T∂, Tag <: AbstractPOTrapAγOperator}
    ∂TGpo::T∂ = nothing # derivative of the PO functional G w.r.t. T
    Aγ::Tag             # Aγ Operator involved in the Jacobian of the PO functional
end

# this function is called whenever the jacobian of G has to be updated
function (J::POTrapJacobianBordered)(u0::AbstractVector, par; δ = convert(eltype(u0), 1e-9))
    T = _extract_period_fdtrap(J.Aγ.prob, u0)
    # we compute the derivative of the problem w.r.t. the period TODO: remove this or improve!!
    # TODO REMOVE vcat!
    @views J.∂TGpo .= (residual(J.Aγ.prob, vcat(u0[begin:end-1], T + δ), par) .- residual(J.Aγ.prob, u0, par)) ./ δ

    J.Aγ(u0, par) # update Aγ

    # return J, needed to properly call the linear solver.
    return J
end

# this is to use BorderingBLS with check_precision = true
#        ┌             ┐
#  J =   │  Aγ   ∂TGpo │
#        │  ϕ'     *   │
#        └             ┘
@views function apply(J::POTrapJacobianBordered, dx)
    # this function would be much more efficient if
    # we call J.Aγ.prob(x, par, dx) but we dont have (x, par)
    out1 = apply(J.Aγ, dx[begin:end-1])
    out1 .+= J.∂TGpo[begin:end-1] .* dx[end]
    return vcat(out1, dot(J.Aγ.prob.ϕ, dx[begin:end-1]) + dx[end] * J.∂TGpo[end])
end
####################################################################################################
# linear solver for the PO functional, akin to a bordered linear solver
@with_kw struct PeriodicOrbitTrapBLS{Tl} <: AbstractLinearSolver
    linsolverbls::Tl = BorderingBLS(solver = AγLinearSolver(), check_precision = false)    # linear solver
end

# Linear solver associated to POTrapJacobianBordered
function (ls::PeriodicOrbitTrapBLS)(J::POTrapJacobianBordered, rhs)
    # we solve the bordered linear system as follows
    dX, dl, flag, liniter = @views ls.linsolverbls(J.Aγ, J.∂TGpo[begin:end-1],
                                             J.Aγ.prob.ϕ, J.∂TGpo[end],
                                           rhs[begin:end-1], rhs[end])
    return vcat(dX, dl), flag, sum(liniter)
end

# One could think that by implementing (ls::PeriodicOrbitTrapBLS)(J::POTrapJacobianBLS, rhs1, rhs2), we could speed up the computation of the linear Bordered system arising in the continuation process. However, we can note that this speed up would be observed only if a factorization of J.Aγ is available like an LU one. When such factorization is available, it is automatically stored as such in J.Aγ and so no speed up would be gained by implementing (ls::PeriodicOrbitTrapBLS)(J::POTrapJacobianBLS, rhs1, rhs2)

##########################
# problem wrappers
residual(prob::WrapPOTrap, x, p) = residual(prob.prob, x, p)
residual!(prob::WrapPOTrap, args...) = residual!(prob.prob, args...)
jacobian(prob::WrapPOTrap, x, p) = prob.jacobian(x, p)
@inline save_solution(::WrapPOTrap, x, p) = x
get_periodic_orbit(prob::WrapPOTrap, u::AbstractVector, p) = get_periodic_orbit(prob.prob, u, p)
is_symmetric(::WrapPOTrap) = false
has_adjoint(::WrapPOTrap) = false
@inline getdelta(pb::WrapPOTrap) = getdelta(pb.prob)
##########################
# newton wrappers
function _newton_trap(trap::PeriodicOrbitTrapProblem,
                orbitguess,
                options::NewtonPar;
                defOp::Union{Nothing, DeflationOperator} = nothing,
                kwargs...)
    # this hack is for the test to work with CUDA
    @assert sum(_extract_period_fdtrap(trap, orbitguess)) >= 0 "The guess for the period should be positive"
    jacobianPO = trap.jacobian
    @assert jacobianPO in _trapezoid_jacobian_type "This jacobian is not defined. Please choose another one."
    M, N = size(trap)

    if jacobianPO in (Dense(), AutoDiffDense(), FullLU(), FullMatrixFree(), FullSparseInplace(), AutoDiffMF())
        if jacobianPO == FullLU()
            jac = (x, p) -> trap(Val(:JacFullSparse), x, p)
        elseif jacobianPO == FullSparseInplace()
            # sparse matrix to hold the jacobian
            _J =  trap(Val(:JacFullSparse), orbitguess, getparams(trap.prob_vf))
            _indx = get_blocks(_J, N, M)
            # inplace modification of the jacobian _J
            jac = (x, p) -> trap(Val(:JacFullSparseInplace), _J, x, p, _indx)
        elseif jacobianPO == Dense()
            _J =  trap(Val(:JacFullSparse), orbitguess, getparams(trap.prob_vf)) |> Array
            jac = (x, p) -> trap(Val(:JacFullSparseInplace), _J, x, p)
        elseif jacobianPO == AutoDiffDense()
            jac = (x, p) -> ForwardDiff.jacobian(z -> residual(trap, z, p), x)
        elseif jacobianPO == AutoDiffMF()
            jac = (x, p) -> dx -> ForwardDiff.derivative(t -> residual(trap, x .+ t .* dx, p), 0)
        else # FullMatrixFree()
            jac = (x, p) -> (dx -> jvp(trap, x, p, dx))
        end

        # define a problem to call newton
        prob = WrapPOTrap(trap, jac, orbitguess, getparams(trap.prob_vf), getlens(trap.prob_vf), nothing, nothing)

        if isnothing(defOp)
            return solve(prob, Newton(), options; kwargs...)
        else
            return solve(prob, defOp, options; kwargs...)
        end
    else # bordered linear solvers
        if jacobianPO == BorderedLU()
            Aγ = AγOperatorLU(N = N, Jc = LA.lu(spdiagm( 0 => ones(N * (M - 1)) )), prob = trap)
            # linear solver
            lspo = PeriodicOrbitTrapBLS()
        elseif jacobianPO == BorderedSparseInplace()
            _J =  trap(Val(:JacCyclicSparse), orbitguess, getparams(trap.prob_vf))
            _indx = get_blocks(_J, N, M-1)
            # inplace modification of the jacobian _J
            Aγ = AγOperatorSparseInplace(Jc = _J,  Jcfact = LA.lu(_J), prob = trap, indx = _indx)
            lspo = PeriodicOrbitTrapBLS()

        else # BorderedMatrixFree()
            Aγ = AγOperatorMatrixFree(prob = trap, orbitguess = zeros(N * M + 1), par = getparams(trap.prob_vf))
            # linear solver
            lspo = PeriodicOrbitTrapBLS(BorderingBLS(solver = AγLinearSolver(options.linsolver), check_precision = false))
        end

        jacPO = POTrapJacobianBordered(zeros(N * M + 1), Aγ)

        prob = WrapPOTrap(trap, jacPO, orbitguess, getparams(trap.prob_vf), getlens(trap.prob_vf), nothing, nothing)

        if isnothing(defOp)
            return solve(prob, Newton(), (@set options.linsolver = lspo); kwargs...)
        else
            return solve(prob, defOp, (@set options.linsolver = lspo); kwargs...)
        end
    end
end

"""
$(TYPEDSIGNATURES)

This is the Krylov-Newton Solver for computing a periodic orbit using a functional G based on Finite Differences and a Trapezoidal rule.

# Arguments:
- `prob` a problem of type [`PeriodicOrbitTrapProblem`](@ref) encoding the functional G
- `orbitguess` a guess for the periodic orbit where `orbitguess[end]` is an estimate of the period of the orbit. It should be a vector of size `N * M + 1` where `M` is the number of time slices, `N` is the dimension of the phase space. This must be compatible with the numbers `N, M` in `prob`.
- `par` parameters to be passed to the functional
- `options` same as for the regular `newton` method
$DocStrjacobianPOTrap
"""
newton(probPO::PeriodicOrbitTrapProblem,
        orbitguess,
        options::NewtonPar;
        kwargs...) = _newton_trap(probPO, orbitguess, options; defOp = nothing, kwargs...)

"""
    $(TYPEDSIGNATURES)

This function is similar to `newton(probPO, orbitguess, options, jacobianPO; kwargs...)` except that it uses deflation in order to find periodic orbits different from the ones stored in `defOp`. We refer to the mentioned method for a full description of the arguments. The current method can be used in the vicinity of a Hopf bifurcation to prevent the Newton-Krylov algorithm from converging to the equilibrium point.
"""
newton(probPO::PeriodicOrbitTrapProblem,
        orbitguess::vectype,
        defOp::DeflationOperator{Tp, Tdot, T, vectype},
        options::NewtonPar;
        kwargs...) where {Tp, Tdot, T, vectype} = _newton_trap(probPO, orbitguess, options; defOp = defOp, kwargs...)

####################################################################################################
# continuation wrapper
"""
    $(TYPEDSIGNATURES)

This is the continuation routine for computing a periodic orbit using a functional G based on Finite Differences and a Trapezoidal rule.

# Arguments
- `prob::PeriodicOrbitTrapProblem` encodes the functional G
- `orbitguess` a guess for the periodic orbit where `orbitguess[end]` is an estimate of the period of the orbit. It could be a vector of size `N * M + 1` where `M` is the number of time slices, `N` is the dimension of the phase space. This must be compatible with the numbers `N, M` in `prob`.
- `alg` continuation algorithm
- `contParams` same as for the regular [`continuation`](@ref) method
- `linear_algo` same as in [`continuation`](@ref)

# Keywords arguments
- `eigsolver` specify an eigen solver for the computation of the Floquet exponents, defaults to `FloquetQaD`

$DocStrjacobianPOTrap

Note that by default, the method prints the period of the periodic orbit as function of the parameter. This can be changed by providing your `record_from_solution` argument.
"""
function continuation_potrap(prob::PeriodicOrbitTrapProblem,
            orbitguess,
            alg::AbstractContinuationAlgorithm,
            contParams::ContinuationPar,
            linear_algo::AbstractBorderedLinearSolver;
            eigsolver = FloquetQaD(contParams.newton_options.eigsolver),
            record_from_solution = nothing,
            plot_solution = nothing,
            kwargs...)
    # this hack is for the test to work with CUDA
    @assert sum(_extract_period_fdtrap(prob, orbitguess)) >= 0 "The guess for the period should be positive"
    jacobianPO = prob.jacobian
    @assert jacobianPO in _trapezoid_jacobian_type "This jacobian is not defined. Please chose another one among $_trapezoid_jacobian_type."

    M, N = size(prob)
    options = contParams.newton_options

    if compute_eigenelements(contParams)
        contParams = @set contParams.newton_options.eigsolver =
         eigsolver
    end

    # change the user provided finalise function by passing prob in its parameters
    _finsol = modify_po_finalise(prob, kwargs, prob.update_section_every_step)
    # this is to remove this part from the arguments passed to continuation
    _kwargs = (record_from_solution = record_from_solution, plot_solution = plot_solution)
    _recordsol = modify_po_record(prob, getparams(prob.prob_vf), getlens(prob.prob_vf); _kwargs...)
    _plotsol = modify_po_plot(prob, getparams(prob.prob_vf), getlens(prob.prob_vf); _kwargs...)

    if jacobianPO in (Dense(), AutoDiffDense(), FullLU(), FullMatrixFree(), FullSparseInplace(), AutoDiffMF())
        if jacobianPO == FullLU()
            jac = (x, p) -> FloquetWrapper(prob, prob(Val(:JacFullSparse), x, p), x, p)
        elseif jacobianPO == FullSparseInplace()
            # sparse matrix to hold the jacobian
            _J =  prob(Val(:JacFullSparse), orbitguess, getparams(prob.prob_vf))
            _indx = get_blocks(_J, N, M)
            # inplace modification of the jacobian _J
            jac = (x, p) -> (prob(Val(:JacFullSparseInplace), _J, x, p, _indx); FloquetWrapper(prob, _J, x, p));
        elseif jacobianPO == Dense()
            _J =  prob(Val(:JacFullSparse), orbitguess, getparams(prob.prob_vf)) |> Array
            jac = (x, p) -> (prob(Val(:JacFullSparseInplace), _J, x, p); FloquetWrapper(prob, _J, x, p));
        elseif jacobianPO == AutoDiffDense()
            jac = (x, p) -> FloquetWrapper(prob, ForwardDiff.jacobian(z -> residual(prob, z, p), x), x, p)
        elseif jacobianPO == AutoDiffMF()
            jac = (x, p) -> FloquetWrapper(prob, dx -> ForwardDiff.derivative(t->residual(prob, x .+ t .* dx, p), 0), x, p)
        else
             jac = (x, p) -> FloquetWrapper(prob, x, p)
        end

        # we have to change the Bordered linearsolver to cope with our type FloquetWrapper
        linear_algo = @set linear_algo.solver = FloquetWrapperLS(linear_algo.solver)
        contParams2 = (@set contParams.newton_options.linsolver = FloquetWrapperLS(options.linsolver))
        alg = update(alg, contParams2, linear_algo)

        probwp = WrapPOTrap(prob, jac, orbitguess, getparams(prob.prob_vf), getlens(prob.prob_vf), _plotsol, _recordsol)

        br = continuation(probwp, alg,
            contParams2; 
            kwargs...,
            kind = PeriodicOrbitCont(),
            finalise_solution = _finsol,
            )
    else
        if jacobianPO == BorderedLU()
            Aγ = AγOperatorLU(N = N, Jc = LA.lu(spdiagm( 0 => ones(N * (M - 1)) )), prob = prob)
            # linear solver
            lspo = PeriodicOrbitTrapBLS()
        elseif jacobianPO == BorderedSparseInplace()
            _J =  prob(Val(:JacCyclicSparse), orbitguess, getparams(prob.prob_vf))
            _indx = get_blocks(_J, N, M-1)
            # inplace modification of the jacobian _J
            Aγ = AγOperatorSparseInplace(Jc = _J,  Jcfact = LA.lu(_J), prob = prob, indx = _indx)
            lspo = PeriodicOrbitTrapBLS()

        else # BorderedMatrixFree
            Aγ = AγOperatorMatrixFree(prob = prob, orbitguess = zeros(N * M + 1), par = getparams(prob.prob_vf))
            # linear solver
            lspo = PeriodicOrbitTrapBLS(BorderingBLS(solver = AγLinearSolver(options.linsolver), check_precision = false))
        end

        jacBD = POTrapJacobianBordered(zeros(N * M + 1), Aγ)
        jacPO = (x, p) -> FloquetWrapper(prob, jacBD(x, p), x, p)

        # we change the linear solver
        contParams = @set contParams.newton_options.linsolver = FloquetWrapperLS(lspo)

        # we have to change the Bordered linearsolver to cope with our type FloquetWrapper
        linear_algo = @set linear_algo.solver = contParams.newton_options.linsolver
        alg = update(alg, contParams, linear_algo)

        probwp = WrapPOTrap(prob, jacPO, orbitguess, getparams(prob.prob_vf), getlens(prob.prob_vf), _plotsol, _recordsol)

        br = continuation(probwp, alg,
            contParams;
            kwargs...,
            kind = PeriodicOrbitCont(),
            finalise_solution = _finsol)
    end
    return br
end

"""
$(TYPEDSIGNATURES)

This is the continuation routine for computing a periodic orbit using a functional G based on Finite Differences and a Trapezoidal rule.

# Arguments
- `prob::PeriodicOrbitTrapProblem` encodes the functional G
- `orbitguess` a guess for the periodic orbit where `orbitguess[end]` is an estimate of the period of the orbit. It could be a vector of size `N * M + 1` where `M` is the number of time slices, `N` is the dimension of the phase space. This must be compatible with the numbers `N, M` in `prob`.
- `alg` continuation algorithm
- `contParams` same as for the regular [`continuation`](@ref) method

# Keyword arguments

- `linear_algo` same as in [`continuation`](@ref)
$DocStrjacobianPOTrap

Note that by default, the method prints the period of the periodic orbit as function of the parameter. This can be changed by providing your `record_from_solution` argument.
"""
function continuation(prob::PeriodicOrbitTrapProblem,
                    orbitguess,
                    alg::AbstractContinuationAlgorithm,
                    _contParams::ContinuationPar;
                    record_from_solution = (u, p; k...) -> (period = u[end],),
                    linear_algo = nothing,
                    kwargs...)
    _linear_algo = isnothing(linear_algo) ?  BorderingBLS(solver = _contParams.newton_options.linsolver, check_precision = false) : linear_algo
    return continuation_potrap(prob, orbitguess, alg, _contParams, _linear_algo; record_from_solution, kwargs...)
end

####################################################################################################
# function needed for automatic Branch switching from Hopf bifurcation point
function re_make(prob::PeriodicOrbitTrapProblem, 
                prob_vf,
                hopfpt,
                ζr::AbstractVector,
                orbitguess_a,
                period; 
                kwargs...)
    M = length(orbitguess_a)
    N = length(ζr)

    # append period at the end of the initial guess
    orbitguess_v = reduce(vcat, orbitguess_a)
    orbitguess = vcat(vec(orbitguess_v), period) |> vec

    # update the problem
    probPO = setproperties(prob, N = N, prob_vf = prob_vf, ϕ = zeros(N*M), xπ = zeros(N*M))

    orbit = get(kwargs, :orbit, nothing)

    if isnothing(orbit)
        probPO.ϕ[1:N] .= real.(ζr)
        probPO.xπ[1:N] .= hopfpt.x0
    else
        probPO.xπ .= orbitguess[begin:end-1]
        _sol = get_periodic_orbit(probPO, orbitguess, nothing)
        probPO.ϕ .= reduce(vcat, [residual(prob_vf, _sol.u[:,i], getparams(prob_vf)) for i=1:probPO.M])
    end
    return probPO, orbitguess
end

using SciMLBase: AbstractTimeseriesSolution

"""
$(TYPEDSIGNATURES)

Generate a guess and a periodic orbit problem from a solution.

## Arguments
- `bifprob` a bifurcation problem to provide the vector field
- `sol` basically, and `ODEProblem`
- `tspan = (0, 1)` estimate of the time span (period) of the periodic orbit

## Output
- returns a `PeriodicOrbitTrapProblem` and an initial guess.
"""
function generate_ci_problem(pb::PeriodicOrbitTrapProblem,
                            bifprob::AbstractBifurcationProblem, 
                            sol::AbstractTimeseriesSolution,
                            tspan::Tuple; 
                            optimal_period::Bool = true,
                            ktrap...)
    u0 = sol(tspan[1])
    @assert u0 isa AbstractVector
    N = length(u0)

    par = sol.prob.p
    prob_vf = re_make(bifprob, params = par)
    probtrap = setproperties(pb; M = pb.M, N = N, prob_vf = prob_vf, xπ = copy(u0), ϕ = copy(u0), ktrap...)

    M, N = size(probtrap)
    resize!(probtrap.ϕ, N * M)
    resize!(probtrap.xπ, N * M)

    period = tspan[2] - tspan[1]

    # find best period candidate
    if optimal_period
        _times = LinRange(period * 0.8, period * 1.2, M)
        period = _times[argmin(norm(sol(tspan[1] + t) - sol(tspan[1])) for t in _times)]
    end

    ci = generate_solution(probtrap, t -> sol(tspan[1] + t * period / (2pi)), period)
    _sol = get_periodic_orbit(probtrap, ci, nothing)
    probtrap.xπ .= ci[begin:end-1]
    probtrap.ϕ .= reduce(vcat, [residual(bifprob, _sol.u[:,i], sol.prob.p) for i=1:probtrap.M])

    return probtrap, ci
end

generate_ci_problem(pb::PeriodicOrbitTrapProblem, bifprob::AbstractBifurcationProblem, sol::AbstractTimeseriesSolution, period::Real; ktrap...) = generate_ci_problem(pb, bifprob, sol, (zero(period), period); ktrap...)
####################################################################################################
