#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const _trapezoid_jacobian_type = (Dense(),
                                  AutoDiffDense(),
                                  FullLU(),
                                  FullMatrixFree(),
                                  BorderedLU(),
                                  BorderedMatrixFree(),
                                  FullSparseInplace(),
                                  BorderedSparseInplace(),
                                  AutoDiffMF())

const DocStrjacobianPOTrap = """
These methods only differ in the linear algebra used to invert the jacobian `dG` of the functional `G` (see [`Trapeze`](@ref)); the discretization is otherwise the same. The value of `jacobian` must belong to `$_trapezoid_jacobian_type`.
- For `jacobian = FullLU()`, we use the default linear solver based on a sparse matrix representation of `dG`. This matrix is assembled at each Newton iteration. This is the right choice when the sparsity pattern can change.
- For `jacobian = FullSparseInplace()`, this is the same as for `FullLU()` but the sparse matrix `dG` is updated inplace. This method allocates much less and, in some cases, is significantly faster than `FullLU()`. Note that this method can only be used if the sparsity pattern of the jacobian is always the same.
- For `jacobian = Dense()`, same as above but the matrix `dG` is dense, and it is also updated inplace. This option is useful to study ODEs of small dimension.
- For `jacobian = AutoDiffDense()`, the jacobian is evaluated using automatic differentiation (ForwardDiff).
- For `jacobian = BorderedLU()`, we take advantage of the bordered shape of `dG` and invert it with a bordered linear solver based on a LU decomposition of the cyclic matrix.
- For `jacobian = BorderedSparseInplace()`, this is the same as for `BorderedLU()` but the cyclic matrix `Jc` is updated inplace. This method allocates much less and, in some cases, is significantly faster than `BorderedLU()`. Note that this method can only be used if the sparsity pattern of the jacobian is always the same.
- For `jacobian = FullMatrixFree()`, a matrix-free linear solver (given by `options.linsolver`) is used to invert `dG`: note that a preconditioner is very likely required here because of the cyclic shape of `dG` which negatively affects the convergence properties of GMRES.
- For `jacobian = BorderedMatrixFree()`, a matrix-free linear solver is used as well but only for `Jc` (see the docs): `options.linsolver` is then used to invert `Jc`. These two matrix-free options thus expose different parts of the jacobian `dG` in order to apply specific preconditioners. For example, an ILU preconditioner on `Jc` could remove the constraints in `dG` and lead to poor convergence. Of course, for these last two methods, a preconditioner is likely to be required.
- For `jacobian = AutoDiffMF()`, the evaluation map of the differential is derived using automatic differentiation. Thus, unlike the previous two cases, the user does not need to pass a matrix-free differential.
"""

"""
$(TYPEDEF)

This composite type implements a finite-difference discretization based on the Trapeze rule (aka Crank-Nicolson, order 2 in time) to locate periodic orbits / BVP. More details (maths, notations, linear systems) can be found [here](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/periodicOrbitTrapeze/).

The scheme is as follows. We discretize the time on `M` slices ``x_{1},\\cdots,x_{M}``, each of dimension `N`, and one looks for the period `T = x[end]` such that the following Crank-Nicolson relations hold for ``i = 1, \\cdots, M-1``

 ``M_a\\cdot\\left(x_{i} - x_{i-1}\\right) - \\frac{T\\cdot h_i}{2} \\left(F(x_{i}) + F(x_{i-1})\\right) = 0,``

where we used the cyclic convention ``x_{0} := x_{M-1}`` and where ``h_i = s_i - s_{i-1}`` are the normalized steps of the mesh `mesh` (``s_0 = 0 < s_1 < \\cdots < s_M = 1``). The orbit is finally closed by the periodicity condition ``x_{M} - x_{1} = 0``. Here ``M_a`` is a mass matrix (the identity by default, see field `massmatrix`) and ``F`` stands for the residual of the vector field encoded in `prob_vf`.

The phase of the periodic orbit, which removes the indeterminacy due to the invariance of periodic orbits under time shifts, is constrained by using a section (but you could use your own)

 ``\\sum_i\\langle x_{i} - x_{\\pi,i}, \\phi_{i}\\rangle=0.``

The pair ``(\\phi, x_{\\pi})`` is stored in the fields `ϕ` and `xπ`: ``x_\\pi`` is a reference state on the orbit while ``\\phi`` are the (normalized) normal vectors of the section. It is updated automatically during continuation by `updatesection!` every `update_section_every_step` steps.

# Internal fields
$(TYPEDFIELDS)

# Methods

Here are some useful methods you can apply to `pb::Trapeze`:

- `length(pb)` gives the number `M * N` of unknowns of the time-discretized state (without the period `T`, which is stored as `x[end]`).
- `get_mesh_size(pb)` returns the number `M` of time slices.
- `get_state_dim(pb)` returns the dimension `N` of a time slice.
- `get_times(pb)` returns the normalized times `sᵢ` at which the orbit is discretized, i.e. the cumulative sum of the mesh steps.
- `get_time_slices(pb, x)` returns the state part of the guess `x` (i.e. `x[1:M*N]`, the period is dropped) reshaped as an `N x M` matrix.
- `get_time_step(pb, i)` returns the `i`-th normalized mesh step `hᵢ`.
- `get_mass_matrix(pb)` returns the mass matrix, defaulting to a sparse identity matrix if none was provided. Passing `true` as a second argument returns instead an identity matrix of the form `I(N)`.
- `hasmassmatrix(pb)` returns `true` if a mass matrix was provided.
- `getparams(pb)`, `getlens(pb)` and `setparam(pb, p)` give access to the parameters of the underlying vector field.
- `getperiod(pb, x)` returns the period `T = x[end]` of the guess `x`.
- `getdelta(pb)` returns the step `δ` used for finite differences.
- `generate_solution(pb, orbit, period)` generates a guess from a function `t -> orbit(t)` for `t ∈ [0, 2π]` and a period `period`.
- `generate_ci_problem(pb, bifprob, sol, tspan)` generates a `Trapeze` problem together with a guess from an `ODE` solution `sol`.
- `get_periodic_orbit(pb, x, pars)` computes the full periodic orbit, mainly for plotting purposes.

# Constructors

The structure can be created by calling `Trapeze(;kwargs...)`. For example, you can declare such a problem without vector field by doing

    Trapeze(M = 100)

A more realistic way to build the problem is to provide the bifurcation problem together with the number of time slices `M` (or a non-uniform mesh given by a vector of steps) and the dimension `N` of a time slice:

    Trapeze(prob_vf, M::Int, N::Int)
    Trapeze(prob_vf, ϕ, xπ, M::Int, N::Int, ls = DefaultLS(); kwargs...)

In the second form, `ϕ` and `xπ` (see above) provide the initial section; they are stored into vectors of length `N * M`, the extra entries (if the provided vectors are shorter) being set to `0`. The keyword `massmatrix` allows to specify a mass matrix. When the discretization is created with a vector field, the residual `F` of `prob_vf` and its jacobian are used to assemble the functional `G`.

# Orbit guess
An orbit guess `orbitguess` must be a vector of size `M * N + 1` where `N` is the number of unknowns in the state space and `orbitguess[M*N+1]` is an estimate of the period ``T`` of the limit cycle. More precisely, using the above notations, `orbitguess` must be ``orbitguess = [x_{1},x_{2},\\cdots,x_{M}, T]``.

Note that you can generate this guess from a function solution using `generate_solution` or from an `ODEProblem` solution using `generate_ci_problem`. You can evaluate the residual of the functional `G` on an orbit guess using `po_residual(pb, orbitguess, p)` and its jacobian with the methods listed in the `# Functional` section below.

# Functional
A functional, hereby called `G`, encodes this problem. The following methods are available

- `po_residual(pb, orbitguess, p)` evaluates the functional `G` on `orbitguess`
- `po_residual!(pb, out, orbitguess, p)` same as `po_residual` but writes the result into `out`
- `po_jvp(pb, orbitguess, p, du)` evaluates the jacobian `dG(orbitguess)⋅du` functional at `orbitguess` on `du`
- `po_jacobian_sparse(pb, orbitguess, p)` returns the sparse matrix of the jacobian `dG(orbitguess)` at `orbitguess`. It is called ``A_γ`` in the docs.
- `po_jacobian_sparse!(pb, J, orbitguess, p)`. Same as `po_jacobian_sparse` but overwrites `J` inplace. Note that the sparsity pattern must be the same independently of the values of the parameters or of `orbitguess`. In this case, this is significantly faster than `po_jacobian_sparse`.
- `jacobian_cyclic_sparse(pb, orbitguess, p)` returns the sparse cyclic matrix ``J_c`` (see the docs) of the jacobian `dG(orbitguess)` at `orbitguess`
- `jacobian_block_diag(pb, orbitguess, p)` returns the block diagonal of the sparse matrix of the jacobian `dG(orbitguess)` at `orbitguess`, i.e. the matrices ``I - (T\\,h_i/2)\\,J(x_i)`` associated to each slice (the last block being the identity). Its inverse is a natural block Jacobi preconditioner.

# Jacobian
$DocStrjacobianPOTrap

!!! note "GPU call"
    For these methods to work on the GPU, for example with `CuArrays` in mode `allowscalar(false)`, we face the issue that the function `_extract_period_fdtrap` won't be well defined because it is a scalar operation. Note that you must pass the option `ongpu = true` for the functional to be evaluated efficiently on the gpu.
"""
@with_kw_noshow struct Trapeze{Tprob, vectype, Tls <: AbstractLinearSolver, T, Tmass, Tjac} <: AbstractFiniteDifferencesDiscretization
    "Vector field (or bifurcation problem) whose residual `F` and jacobian are used to assemble the functional `G`. `nothing` is allowed to build a bare discretization."
    prob_vf::Tprob = nothing

    "Normal vectors `ϕ` of the phase constraint, of size `N * M` (see the section equation in the documentation of `Trapeze`)."
    ϕ::vectype = nothing

    "Reference point `xπ` of the phase constraint, of size `N * M` (see the section equation in the documentation of `Trapeze`)."
    xπ::vectype = nothing

    "Number of time slices."
    M::Int = 0

    "Mesh of (normalized) time steps, see `TimeMesh`."
    mesh::TimeMesh{T} = TimeMesh(M)

    "Dimension of the problem in case of an `AbstractVector` state space."
    N::Int = 0

    "Linear solver used to invert the jacobian of a single time slice, i.e. to solve `J⋅sol = rhs`. Only needed in a matrix-free setting (e.g. `BorderedMatrixFree()`) or for the computation of the Floquet multipliers."
    linsolver::Tls = DefaultLS()

    "Whether the computation takes place on the gpu (Experimental). When `true`, the functional returns `vcat(out[begin:end-1], phase_cond)` which is compatible with `CuArrays` in the mode `allowscalar(false)`."
    ongpu::Bool = false

    "Whether the vector field is autonomous, i.e. does not depend explicitly on time."
    isautonomous::Bool = true

    "Mass matrix ``M_a`` of the time discretization. You can pass for example a sparse matrix. Default: `nothing`, i.e. the identity matrix."
    massmatrix::Tmass = nothing

    "Frequency at which the phase constraint is updated during continuation, see `updatesection!`."
    update_section_every_step::UInt = 1

    "Type of jacobian used in Newton iterations (see the `# Jacobian` section in the documentation of `Trapeze`)."
    jacobian::Tjac = Dense()

    @assert jacobian in _trapezoid_jacobian_type "$jacobian is not defined for `Trapeze`. Pick one in $_trapezoid_jacobian_type"
end

function Base.show(io::IO, trap::Trapeze)
    println(io, "┌─ Trapeze method for periodic orbits (PO) / bvp")
    println(io, "├─ time slices    : ", trap.M)
    println(io, "├─ dimension      : ", get_state_dim(trap))
    println(io, "├─ jacobian       : ", trap.jacobian)
    println(io, "├─ update section : ", trap.update_section_every_step)
    println(io, "├─ # unknowns without phase condition (for PO) : ", length(trap) - 1)
    println(io, "└─ inplace        : ", isinplace(trap))
end

@inline isinplace(trap::Trapeze) = isnothing(trap.prob_vf) ? false : isinplace(trap.prob_vf)
@inline get_time_step(trap::Trapeze, i::Int) = get_time_step(trap.mesh, i)
get_times(trap::Trapeze) = cumsum(collect(trap.mesh))
@inline hasmassmatrix(::Trapeze{Tprob, vectype, Tls, T, Tmass}) where {Tprob, vectype, Tls, T, Tmass} = ~(Tmass == Nothing)
@inline getparams(trap::Trapeze) = getparams(trap.prob_vf)
@inline getlens(trap::Trapeze) = getlens(trap.prob_vf)
@inline getdelta(trap::Trapeze) = getdelta(trap.prob_vf)
setparam(trap::Trapeze, p) = set(getparams(trap), getlens(trap), p)
@inline get_state_dim(trap::Trapeze) = trap.N
@inline length(trap::Trapeze) = trap.M * get_state_dim(trap)

# type unstable!
@inline function get_mass_matrix(trap::Trapeze, return_type_Array = false)
    if return_type_Array == false
        return hasmassmatrix(trap) ? trap.massmatrix : SPA.spdiagm( 0 => ones(trap.N))
    else
        return hasmassmatrix(trap) ? trap.massmatrix : LinearAlgebra.I(trap.N)
    end
end
# these functions extract the last component of the periodic orbit guess
@inline _extract_period_fdtrap(trap::Trapeze, x::AbstractVector) = on_gpu(trap) ? x[end:end] : x[end]
# these functions extract the time slices components
get_time_slices(x::AbstractVector, N, M) = @views reshape(x[begin:end-1], N, M)
get_time_slices(trap::Trapeze, x) = get_time_slices(x, trap.N, trap.M)

"""
$(TYPEDSIGNATURES)

Compute the period of the periodic orbit associated to `x`.
"""
@inline getperiod(prob::Trapeze, x, p) = _extract_period_fdtrap(prob, x)

# for a dummy constructor, useful for specifying the "algorithm" to look for periodic orbits,
# just call Trapeze()

function Trapeze(prob_vf,
                                    ϕ::vectype,
                                    xπ::vectype,
                                    m::Union{Int, AbstractVector},
                                    ls::AbstractLinearSolver = DefaultLS();
                                    ongpu = false,
                                    massmatrix = nothing) where {vectype}
    _length = ϕ isa AbstractVector ? length(ϕ) : 0
    M = m isa Number ? m : length(m) + 1

    return Trapeze(;prob_vf, ϕ, xπ, M, mesh = TimeMesh(m), N = _length ÷ M, linsolver = ls, ongpu, massmatrix)
end

function Trapeze(prob_vf,
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
    trap = Trapeze(;prob_vf,
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

    trap.xπ .= 0
    trap.ϕ .= 0

    trap.xπ[eachindex(xπ)] .= xπ
    trap.ϕ[eachindex(ϕ)] .= ϕ
    return trap
end

Trapeze(prob_vf,
        m::Union{Int, AbstractVector},
        N::Int,
        ls::AbstractLinearSolver = DefaultLS();
        ongpu = false,
        adaptmesh = false,
        massmatrix = nothing) = Trapeze(prob_vf, zeros(N*(m isa Number ? m : length(m) + 1)), zeros(N*(m isa Number ? m : length(m) + 1)), m, N, ls; ongpu, massmatrix)


# do not type h::Number because this will annoy CUDA
"""
$(TYPEDSIGNATURES)

Low-level building block of the Crank-Nicolson scheme implemented by `Trapeze`. Given the two consecutive slices `u1`, `u2` and the (scaled) time step `h`, it stores in `dest` the residual

``M_a\\,(u_1 - u_2) - h\\,(F(u_1) + F(u_2))``

of the current time slice. The vector `tmp` is a buffer which, on entry, must contain ``F(u_2)`` (it is overwritten with ``F(u_1)`` or ``J(u_1)\\,du_1``); this avoids evaluating the vector field twice since the scheme is applied slice by slice in a cyclic way.

- if `applyf = Val(true)` (default), ``F(u_1)`` is evaluated with the vector field;
- if `applyf = Val(false)`, the jacobian action ``J(u_1)\\,du_1`` is used instead (this is the matrix-free expression of the jacobian);
- if `linear = Val(false)`, the mass term ``M_a\\,(du_1 - du_2)`` is dropped. This is used to accumulate the derivative of the residual with respect to the period `T` inside `po_jvp!`.

The 3-argument version (`u1`, `u2`, `h`) simply duplicates the slices for the directions `du1 = u1`, `du2 = u2`.
"""
function potrap_scheme!(trap,
                        dest,
                        u1, u2,
                        du1, du2,
                        par, h,
                        tmp,
                        linear::Val{is_linear} = Val(true);
                        applyf::Val{is_applyf} = Val(true)) where {is_linear, is_applyf}
    # this function implements the basic implicit scheme used for the time integration
    # because this function is called in a cyclic manner, we save the value of F(u2)
    # in the variable tmp in order to avoid recomputing it in a subsequent call
    # basically tmp is F(u2)
    # applyf: if true use F and dF otherwise
    if is_linear
        dest .= tmp
        if is_applyf
            # tmp <- trap.F(u1, par)
            residual!(trap.prob_vf, tmp, u1, par) #TODO this line does not almost seem to be type stable in code_wartype, gives @_11::Union{Nothing, Tuple{Int64,Int64}}
        else
            applyJ(trap, tmp, u1, par, du1)
        end
        if hasmassmatrix(trap)
            dest .= trap.massmatrix * (du1 .- du2) .- h .* (dest .+ tmp)
        else
            @. dest = (du1 - du2) - h * (dest + tmp)
        end
    else # used for jvp
        dest .-= h .* tmp
        # tmp <- trap.F(u1, par)
        residual!(trap.prob_vf, tmp, u1, par)
        dest .-= h .* tmp
    end
end
potrap_scheme!(trap, dest, u1, u2, par, h, tmp, linear = Val(true); applyf = Val(true)) = potrap_scheme!(trap, dest, u1, u2, u1, u2, par, h, tmp, linear; applyf)

"""
$(TYPEDSIGNATURES)

Evaluate, inplace, the functional `G` implemented by the `Trapeze` discretization at the guess `u` (of size `M * N + 1`) and store the result in `out`. The functional is given by the block vector

``G(u) = \\big[\\, M_a\\,(x_i - x_{i-1}) - \\tfrac{T\\,h_i}{2}(F(x_i) + F(x_{i-1}))\\ (i=1,\\cdots,M-1),\\; x_M - x_1,\\; \\langle x - x_\\pi, \\phi\\rangle\\,\\big]``

where ``T = x_{M N + 1}`` is the period. It works for inplace / out of place vector fields `pb.F`. On the CPU it writes into and returns `out`; in GPU mode (`ongpu = true`) it returns `vcat(out[begin:end-1], phase_cond)` to avoid scalar operations.
"""
@views function po_residual!(trap::Trapeze, out, u, par)
    T = getperiod(trap, u, nothing)
    M, N = size(trap)

    uc = get_time_slices(trap, u)
    outc = get_time_slices(trap, out)

    po_residual_bare!(trap, outc, uc, par, T)

    # closure condition ensuring a periodic orbit
    outc[:, M] .= uc[:, M] .- uc[:, 1]

    # this is for CuArrays.jl to work in the mode allowscalar(false)
    phase_cond = LA.dot(u[begin:end-1], trap.ϕ) - LA.dot(trap.xπ, trap.ϕ)
    if on_gpu(trap)
        return vcat(out[begin:end-1], phase_cond) # this is the phase condition
    else
        out[end] = phase_cond
        return out
    end
end
"""
$(TYPEDSIGNATURES)

Allocate a vector and evaluate the functional `G` at the guess `u`, see `po_residual!`.
"""
po_residual(trap::Trapeze, u, par) = po_residual!(trap, similar(u), u, par)

"""
$(TYPEDSIGNATURES)

Evaluate only the `M - 1` first blocks of `G` (i.e. the Crank-Nicolson residuals, see the documentation of [`Trapeze`](@ref)) into the columns `1:M-1` of `outc`. The periodicity condition ``x_M - x_1 = 0`` and the phase condition are **not** handled here; they are added by `po_residual!`.
"""
@views function po_residual_bare!(trap::Trapeze, outc, uc::AbstractMatrix, par, T)
    M, N = size(trap)

    # outc[:, M] plays the role of tmp until it is used just after the for-loop
    residual!(trap.prob_vf, outc[:, M], uc[:, M-1], par)

    h = T * get_time_step(trap, 1)
    # fastest is to do out[:, i] = x
    potrap_scheme!(trap, outc[:, 1], uc[:, 1], uc[:, M-1], par, h/2, outc[:, M])

    for ii in 2:M-1
        h = T * get_time_step(trap, ii)
        # this function avoids computing F(uc[:, ii]) twice
        potrap_scheme!(trap, outc[:, ii], uc[:, ii], uc[:, ii-1], par, h/2, outc[:, M])
    end
end

"""
$(TYPEDSIGNATURES)

Matrix-free expression (jvp) of the jacobian ``dG(u)\\cdot du`` of the PO functional, evaluated at `u` and applied to the direction `du` (of size `M * N + 1`, `du[end]` being the direction for the period `T`). The result is stored in `out`.
"""
@views function po_jvp!(trap::Trapeze, out, u, par, du)
    M, N = size(trap)
    T  = _extract_period_fdtrap(trap, u)
    dT = _extract_period_fdtrap(trap, du)

    uc = get_time_slices(trap, u)
    outc = get_time_slices(trap, out)
    duc = get_time_slices(trap, du)

    # compute the cyclic part
    Jc(trap, outc, u[begin:end-1-N], par, T, du[begin:end-N-1], outc[:, M])

    # outc[:, M] plays the role of tmp until it is used just after the for-loop
    tmp = outc[:, M]

    # we now compute the partial derivative w.r.t. the period T
    residual!(trap.prob_vf, tmp, uc[:, M-1], par)

    h = dT * get_time_step(trap, 1)
    potrap_scheme!(trap, outc[:, 1], uc[:, 1], uc[:, M-1], par, h/2, tmp, Val(false))
    for ii in 2:M-1
        h = dT * get_time_step(trap, ii)
        potrap_scheme!(trap, outc[:, ii], uc[:, ii], uc[:, ii-1], par, h/2, tmp, Val(false))
    end

    # closure condition ensuring a periodic orbit
    outc[:, M] .= duc[:, M] .- duc[:, 1]

    # this is for CuArrays.jl to work in the mode allowscalar(false)
    phase_cond = LA.dot(du[begin:end-1], trap.ϕ)
    if on_gpu(trap)
        return vcat(out[begin:end-1], phase_cond)
    else
        out[end] = phase_cond
        return out
    end
end

"""
$(TYPEDSIGNATURES)

Allocate a vector and evaluate the jacobian action ``dG(u)\\cdot du``, see `po_jvp!`.
"""
po_jvp(trap::Trapeze, u::AbstractVector, par, du) = po_jvp!(trap, similar(du), u, par, du)
jvp(wrap::PeriodicOrbitFunctionalTrap, u, par, du) = po_jvp(get_discretization(wrap), u, par, du)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Matrix free expression of matrices related to the Jacobian Matrix of the PO functional
"""
$(TYPEDSIGNATURES)

Matrix-free application of the block ``A_\\gamma`` of the jacobian ``dG`` (see `po_jacobian_block`) to the direction `du` (of size `M * N`): the cyclic part is applied with `Jc` and the closure block ``x_M - \\gamma\\,x_1`` is stored in the last column `outc[:, M]` of `outc`.
"""
function Aγ!(trap::Trapeze, outc, u0::AbstractVector, par, du::AbstractVector; γ = 1)
    # u0 of size N * M + 1
    # du of size N * M
    M, N = size(trap)
    T = _extract_period_fdtrap(trap, u0)
    u0c = get_time_slices(trap, u0)

    # compute the cyclic part
    @views Jc(trap, outc, u0[begin:end-1-N], par, T, du[begin:end-N], outc[:, M])

    # closure condition ensuring a periodic orbit
    duc = reshape(du, N, M)
    outc[:, M] .= @views duc[:, M] .- γ .* duc[:, 1]
    return nothing
end

"""
$(TYPEDSIGNATURES)

Matrix-free application of the cyclic matrix ``J_c`` of the PO functional to the direction `du`. The state part `u0` (of size ``N\\,(M-1)``) gathers the time slices ``x_1,\\cdots,x_{M-1}`` at which the jacobians are evaluated, `T` is the period, `du` is the direction (of size ``N\\,(M-1)``) and `outc` (of size ``N\\times M``) receives the result (only its first `M - 1` columns are used, the last one plays the role of buffer via `tmp`). A vector version of `outc` is returned. This is the building block used by the matrix-free linear solvers, e.g. `FullMatrixFree()` or `BorderedMatrixFree()`.
"""
function Jc(trap::Trapeze, outc::AbstractMatrix, u0::AbstractVector, par, T, du::AbstractVector, tmp)
    # tmp plays the role of buffer array
    # u0 of size N * (M - 1)
    # du of size N * (M - 1)
    # outc of size N * M
    M, N = size(trap)

    u0c = reshape(u0, N, M-1)
    duc = reshape(du, N, M-1)

    @views applyJ(trap, tmp, u0c[:, M-1], par, duc[:, M-1])

    h = T * get_time_step(trap, 1)
    @views potrap_scheme!(trap, outc[:, 1], u0c[:, 1], u0c[:, M-1],
                                          duc[:, 1], duc[:, M-1], par, h/2, tmp, Val(true); applyf = Val(false))

    for ii in 2:M-1
        h = T * get_time_step(trap, ii)
        @views potrap_scheme!(trap, outc[:, ii], u0c[:, ii], u0c[:, ii-1],
                                               duc[:, ii], duc[:, ii-1], par, h/2, tmp, Val(true); applyf = Val(false))
    end

    # we also return a Vector version of outc
    return vec(outc)
end

"""
$(TYPEDSIGNATURES)

Allocate the buffers and apply the cyclic matrix ``J_c(u_0)\\cdot du`` (of size ``N\\,(M-1)``) to the direction `du`, see `Jc`.
"""
function Jc(trap::Trapeze, u0::AbstractVector, par, du::AbstractVector)
    M, N = size(trap)
    T = _extract_period_fdtrap(trap, u0)
    out  = similar(du)
    outc = reshape(out, N, M-1)
    tmp  = similar(view(outc, :, 1))
    return @views Jc(trap, outc, u0[begin:end-1-N], par, T, du, tmp)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDSIGNATURES)

Return the block-by-block (sparse) expression of the matrix ``A_\\gamma``, i.e. the jacobian of the PO functional `G` w.r.t. the space unknowns only (the period column is **not** included). It is a block matrix with `M` blocks of size ``N\\times N`` whose cyclic part is filled by `po_cylic_block!` and whose last block row encodes the periodicity condition ``x_M - \\gamma\\,x_1 = 0`` (``\\gamma = 1`` for the exact jacobian). See `po_jacobian_sparse` for the full jacobian of `G`.
"""
function po_jacobian_block(trap::Trapeze, u0::AbstractVector, par; γ = 1)
    # extraction of various constants
    M, N = size(trap)

    Aγ = BA.BlockArray(SPA.spzeros(M * N, M * N), N * ones(Int64, M),  N * ones(Int64, M))
    po_cylic_block!(trap, u0, par, Aγ)

    Iₙ = SPA.spdiagm( 0 => ones(N))
    Aγ[BA.Block(M, 1)] = (-γ) * Iₙ
    Aγ[BA.Block(M, M)] = Iₙ
    return Aγ
end

"""
$(TYPEDSIGNATURES)

Fill the cyclic (block tridiagonal) part of the block matrix `Jc`, i.e. the jacobian of the Crank-Nicolson relations w.r.t. the space unknowns, using the analytic jacobians ``J(x_i)`` of the vector field evaluated at each slice. The blocks read ``M - (T\\,h_i/2)\\,J(x_i)`` on the diagonal and ``-M - (T\\,h_i/2)\\,J(x_{i-1})`` on the sub-diagonal (with the cyclic convention ``x_0 := x_{M-1}``).
"""
function po_cylic_block!(trap::Trapeze, u0::AbstractVector, par, Jc::BA.BlockArray)
    period = _extract_period_fdtrap(trap, u0)
    u0m = get_time_slices(trap, u0)
    _trac_cylic_block!(trap, u0m, period, par, Jc)
end

function _trac_cylic_block!(trap::Trapeze, u0m::AbstractMatrix, period, par, Jc::BA.BlockArray)
    # extraction of various constants
    M, N = size(trap)

    Iₙ = get_mass_matrix(trap)

    tmpJ = @views jacobian(trap.prob_vf, u0m[:, 1], par)

    h = period * get_time_step(trap, 1)
    Jn = Iₙ - (h/2) .* tmpJ
    Jc[BA.Block(1, 1)] = Jn

    # we could do a Jn .= -M .- ... but we want to allow the sparsity pattern to vary
    Jn = @views -Iₙ - (h/2) .* jacobian(trap.prob_vf, u0m[:, M-1], par)
    Jc[BA.Block(1, M-1)] = Jn

    for ii in 2:M-1
        h = period * get_time_step(trap, ii)
        Jn = -Iₙ - (h/2) .* tmpJ
        Jc[BA.Block(ii, ii-1)] = Jn

        tmpJ = @views jacobian(trap.prob_vf, u0m[:, ii], par)

        Jn = Iₙ - (h/2) .* tmpJ
        Jc[BA.Block(ii, ii)] = Jn
    end
    return Jc
end

"""
$(TYPEDSIGNATURES)

Return the cyclic (block tridiagonal) matrix ``J_c(u_0)`` of size ``N\\,(M-1)`` as a `BlockArray`, see `po_cylic_block!`.
"""
function po_cylic_block(trap::Trapeze, u0::AbstractVector, par)
    # extraction of various constants
    M, N = size(trap)
    Jc = BA.BlockArray(SPA.spzeros((M - 1) * N, (M - 1) * N), N * ones(Int64, M-1),  N * ones(Int64, M-1))
    po_cylic_block!(trap, u0, par, Jc)
end

"""
$(TYPEDSIGNATURES)

Return the cyclic matrix ``J_c`` (see `po_cylic_block`) converted to a standard sparse matrix. Used internally to build the LU factorizations of the bordered linear solvers.
"""
cylic_potrap_sparse(trap::Trapeze, orbitguess0, par) = block_to_sparse(po_cylic_block(trap, orbitguess0, par))

"""
$(TYPEDSIGNATURES)

Return the sparse matrix of the full jacobian ``dG(u_0)`` of the PO functional `G` at `u_0`. It is assembled from the block matrix ``A_\\gamma`` (see `po_jacobian_block`) to which the derivative ``\\partial_T G`` w.r.t. the period `T` is appended as a last column (computed by finite differences with step `δ`), together with the last row encoding the phase condition ``\\phi^\\top\\,x = 0``.
"""
function po_jacobian_sparse(trap::Trapeze, u0::AbstractVector, par; γ = 1, δ = getdelta(trap))
    # extraction of various constants
    M, N = size(trap)
    T = _extract_period_fdtrap(trap, u0)
    AγBlock = po_jacobian_block(trap, u0, par; γ)

    # we now set up the last line / column
    @views ∂TGpo = (po_residual(trap, vcat(u0[begin:end-1], T + δ), par) .- po_residual(trap, u0, par)) ./ δ

    # this is "bad" for performance. Get converted to SparseMatrix at the next line
    Aγ = block_to_sparse(AγBlock) # most of the computing time is here!!
    @views Aγ = hcat(Aγ, ∂TGpo[begin:end-1])
    Aγ = vcat(Aγ, SPA.spzeros(1, N * M + 1))

    Aγ[N*M+1, eachindex(trap.ϕ)] .= trap.ϕ
    Aγ[N*M+1, N*M+1] = ∂TGpo[end]
    return Aγ
end

"""
$(TYPEDSIGNATURES)

Inplace version of `po_jacobian_sparse`: the jacobian ``dG(u_0)`` is stored in the matrix `J0`, which must have the right size and sparsity pattern. When `J0` is sparse, this assumes that the sparsity pattern of `J0` and of ``dG(u_0)`` are the same (this is the case when only the values of the parameters or of `u_0` change). This method is then significantly faster than `po_jacobian_sparse` as it avoids reallocation.
"""
@views function po_jacobian_sparse!(trap::Trapeze, J0::Tj, u0::AbstractVector, par; γ = 1, δ = getdelta(trap)) where Tj
    M, N = size(trap)
    T = _extract_period_fdtrap(trap, u0)

    Iₙ = get_mass_matrix(trap, ~(Tj <: SPA.SparseMatrixCSC))

    u0m = get_time_slices(trap, u0)

    tmpJ = jacobian(trap.prob_vf, u0m[:, 1], par)

    h = T * get_time_step(trap, 1)
    Jn = Iₙ - (h/2) .* tmpJ
    # setblock!(Jc, Jn, 1, 1)
    J0[1:N, 1:N] .= Jn

    Jn .= -Iₙ .- (h/2) .* jacobian(trap.prob_vf, u0m[:, M-1], par)
    # setblock!(Jc, Jn, 1, M-1)
    J0[1:N, (M-2)*N+1:(M-1)*N] .= Jn

    for ii in 2:M-1
        h = T * get_time_step(trap, ii)
        @. Jn = -Iₙ - h/2 * tmpJ
        # the next lines cost the most
        # setblock!(Jc, Jn, ii, ii-1)
        J0[(ii-1)*N+1:(ii)*N, (ii-2)*N+1:(ii-1)*N] .= Jn

        tmpJ .= jacobian(trap.prob_vf, u0m[:, ii], par)

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
    ∂TGpo = (po_residual(trap,vcat(u0[begin:end-1], T + δ), par) .- po_residual(trap,u0, par)) ./ δ
    J0[:, end] .=  ∂TGpo

    # this following does not depend on u0, so it does not change. However we update it in case the caller updated the section somewhere else
    J0[N*M+1, eachindex(trap.ϕ)] .=  trap.ϕ

    return J0
end

"""
$(TYPEDSIGNATURES)

Same as the 3-argument version of `po_jacobian_sparse!` but using the precomputed `indx` of the blocks of the sparse matrix `J0` (as returned by `_get_blocks_from_sparse_matrix`). Only the values of `J0.nzval` are updated, making this the fastest variant. `updateborder = Val(false)` skips the update of the last column / last row (the period and phase parts), which is useful when only the cyclic part ``J_c`` needs to be refreshed, e.g. inside `AγOperatorSparseInplace`.
"""
@views function po_jacobian_sparse!(trap::Trapeze,
                            J0,
                            u0::AbstractVector,
                            par,
                            indx; 
                            updateborder::Val{_updateborder} = Val(true),
                            δ = getdelta(trap), 
                            kwargs...) where {_updateborder}
    period = _extract_period_fdtrap(trap, u0)
    u0m = get_time_slices(trap, u0)
    _trap_jacobian_sparse!(trap, J0, u0m, period, par, indx; kwargs...)
    if _updateborder
        # we now set up the last line / column
        M, N = size(trap)
        ∂TGpo = (po_residual(trap, vcat(u0[begin:end-1], period + δ), par) .- 
                 po_residual(trap, u0, par)) ./ δ
        J0[:, end] .= ∂TGpo

        # the following does not depend on u0, so it does not change. However we update it in case the caller updated the section somewhere else
        J0[N*M+1, eachindex(trap.ϕ)] .= trap.ϕ
    end
    return J0
end

@views function _trap_jacobian_sparse!(trap::Trapeze, J0, u0m::AbstractMatrix, period, par, indx; γ = 1)
    M, N = size(trap)
    Iₙ = get_mass_matrix(trap)

    tmpJ = jacobian(trap.prob_vf, u0m[:, 1], par)

    h = period * get_time_step(trap, 1)
    Jn = Iₙ - tmpJ * (h/2)

    # setblock!(Jc, Jn, 1, 1)
    J0.nzval[indx[1, 1]] .= Jn.nzval

    Jn .= -Iₙ .- jacobian(trap.prob_vf, u0m[:, M-1], par) .* (h/2)
    # setblock!(Jc, Jn, 1, M-1)
    J0.nzval[indx[1, M-1]] .= Jn.nzval

    for ii in 2:M-1
        h = period * get_time_step(trap, ii)
        @. Jn = -Iₙ - tmpJ * (h/2)
        # the next lines cost the most
        # setblock!(Jc, Jn, ii, ii-1)
        J0.nzval[indx[ii, ii-1]] .= Jn.nzval

        tmpJ .= jacobian(trap.prob_vf, u0m[:, ii], par)# * (h/2)

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

    return J0
end

"""
$(TYPEDSIGNATURES)

Return the sparse cyclic matrix ``J_c(u_0)`` of size ``N\\,(M-1)`` (see `po_cylic_block`) as a sparse matrix. This corresponds to the space part of ``dG`` without the closure block, the period column and the phase line; it is used to build the (preconditioned) bordered linear solvers.
"""
function jacobian_cyclic_sparse(trap::Trapeze, u0::AbstractVector, par, γ = 1)
    # extraction of various constants
    N = trap.N
    AγBlock = po_jacobian_block(trap, u0, par; γ)

    # this is bad for performance. Get converted to SparseMatrix at the next line
    Aγ = block_to_sparse(AγBlock) # most of the computing time is here!!
    # the following line is bad but still less costly than the previous one
    return Aγ[begin:end-N, begin:end-N]
end

"""
$(TYPEDSIGNATURES)

Return the block diagonal of the jacobian ``dG(u_0)``, i.e. a sparse matrix whose `M` diagonal blocks read ``I - (T\\,h_i/2)\\,J(x_i)`` (the last one is the identity). The inverse of this matrix is a natural block Jacobi preconditioner for the (preconditioned) matrix-free solvers of `Trapeze`.
"""
function jacobian_block_diag(trap::Trapeze, u0::AbstractVector, par)
    # extraction of various constants
    M, N = size(trap)
    T = _extract_period_fdtrap(trap, u0)

    A_diagBlock = BA.BlockArray(SPA.spzeros(M * N, M * N), N * ones(Int64, M),  N * ones(Int64, M))

    In = get_mass_matrix(trap)

    u0c = reshape(u0[begin:end-1], N, M)

    h = T * get_time_step(trap, 1)
    @views Jn = In - h/2 .* jacobian(trap.prob_vf, u0c[:, 1], par)
    A_diagBlock[BA.Block(1, 1)] = Jn

    for ii in 2:M-1
        h = T * get_time_step(trap, ii)
        @views Jn = In - h/2 .* jacobian(trap.prob_vf, u0c[:, ii], par)
        A_diagBlock[BA.Block(ii, ii)]= Jn
    end
    A_diagBlock[BA.Block(M, M)]= In

    A_diag_sp = block_to_sparse(A_diagBlock) # most of the computing time is here!!
    return A_diag_sp
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utils
"""
$(TYPEDSIGNATURES)

Compute the full periodic orbit associated to `x`. Mainly for plotting purposes.
"""
@views function get_periodic_orbit(trap::AbstractFiniteDifferencesDiscretization, u, p)
    T = getperiod(trap, u, p)
    M, N = size(trap)
    uv = u[begin:end-1]
    uc = reshape(uv, N, M)
    return BVPSolution(t = cumsum(T .* collect(trap.mesh)), u = uc)
end
get_periodic_orbit(prob::AbstractFiniteDifferencesDiscretization, x, p::Real) = get_periodic_orbit(prob, x, setparam(prob, p))

"""
$(TYPEDSIGNATURES)

Update the section used for the phase constraint at the current guess `x`: the reference point is set to ``x_\\pi = x`` (the state part of the guess) while the normals are taken as the (normalized) vector field evaluated at each time slice, i.e. ``\\phi_i = F(x_i)/M``. This is called automatically during continuation every `update_section_every_step` steps.
"""
@views function updatesection!(trap::Trapeze, x, pars)
    @debug "Update section TRAP"
    M, N = size(trap)
    xc = get_time_slices(trap, x)

    # update the reference point
    trap.xπ .= x[begin:end-1]

    # update the normals
    for ii in 0:M-1
        # ii2 = (ii+1)<= M ? ii+1 : ii+1-M
        residual!(trap.prob_vf, trap.ϕ[ii*N+1:ii*N+N], xc[:, ii+1], pars)
        trap.ϕ[ii*N+1:ii*N+N] ./= M
    end
    return true
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Linear solvers for the jacobian of the functional G implemented by Trapeze
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
    Jc::Tjc    = lu(SPA.spdiagm(0 => ones(1))) # lu factorisation of the cyclic matrix
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
    A.Jc = LA.lu(cylic_potrap_sparse(A.prob, orbitguess, par))
    A
end

function (A::AγOperatorSparseInplace)(orbitguess::AbstractVector, par)
    # compute the cyclic matrix
    po_jacobian_sparse!(A.prob, A.Jc, orbitguess, par, A.indx; updateborder = Val(false))
    # update the LU decomposition
    LA.lu!(A.Jcfact, A.Jc)
    return A
end

@views function apply(A::AγOperatorSparseInplace, dx)
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
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The following structure encodes the jacobian of a Trapeze which eases the use of PeriodicOrbitTrapBLS. It is made so that accessing the cyclic matrix Jc or Aγ is easier. It is combined with a specific linear solver. It is also a convenient structure for the computation of Floquet multipliers. Therefore, it is only used in the method continuation_potrap
@with_kw struct POTrapJacobianBordered{T∂, Tag <: AbstractPOTrapAγOperator}
    ∂TGpo::T∂ = nothing # derivative of the PO functional G w.r.t. T
    Aγ::Tag             # Aγ Operator involved in the Jacobian of the PO functional
end

# this function is called whenever the jacobian of G has to be updated
function (J::POTrapJacobianBordered)(u0::AbstractVector, par; δ = convert(VI.scalartype(u0), getdelta(J.Aγ.prob)))
    T = _extract_period_fdtrap(J.Aγ.prob, u0)
    # we compute the derivative of the problem w.r.t. the period TODO: remove this or improve!! # TODO REMOVE vcat!!
    @views J.∂TGpo .= (po_residual(J.Aγ.prob, vcat(u0[begin:end-1], T + δ), par) .- po_residual(J.Aγ.prob, u0, par)) ./ δ
    # update Aγ
    J.Aγ(u0, par)
    return J # needed to properly call the linear solver.
end

# this is used to apply the bordered structure of the jacobian with a BorderingBLS linear solver
#        ┌             ┐
#  J =   │  Aγ   ∂TGpo │
#        │  ϕ'     *   │
#        └             ┘
@views function apply(J::POTrapJacobianBordered, dx)
    # this function would be much more efficient if
    # we call J.Aγ.prob(x, par, dx) but we dont have (x, par)
    out1 = apply(J.Aγ, dx[begin:end-1])
    out1 .+= J.∂TGpo[begin:end-1] .* dx[end]
    return vcat(out1, LA.dot(J.Aγ.prob.ϕ, dx[begin:end-1]) + dx[end] * J.∂TGpo[end])
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
@inline save_solution(::PeriodicOrbitFunctionalTrap, x, p) = x
get_periodic_orbit(prob::PeriodicOrbitFunctionalTrap, u::AbstractVector, p) = get_periodic_orbit(get_discretization(prob), u, p)
is_symmetric(::PeriodicOrbitFunctionalTrap) = false
has_adjoint(::PeriodicOrbitFunctionalTrap) = false
##########################
function _generate_jacobian(trap::Trapeze, ::Dense, orbitguess, pars; k...)
    _J =  po_jacobian_sparse(trap, orbitguess, pars) |> Array
    return (Dense(), _J)
end

function _generate_jacobian(trap::Trapeze, ::FullSparseInplace, orbitguess, pars; k...)
    M, N = size(trap)
    # sparse matrix to hold the jacobian
    J =  po_jacobian_sparse(trap, orbitguess, getparams(trap.prob_vf))
    indx = _get_blocks_from_sparse_matrix(J, N, M)
    return (FullSparseInplace(), J, indx)
end

function _jacobian_po(wrap::PeriodicOrbitFunctionalTrap, J::Tuple{Dense, Tj}, x, p) where {Tj}
    _J = J[2]
    trap = get_discretization(wrap)
    po_jacobian_sparse!(trap, _J, x, p)
end

function _jacobian_po(wrap::PeriodicOrbitFunctionalTrap, J::Tuple{FullSparseInplace, Tj, Ti}, x, p) where {Tj, Ti}
    _J = J[2]
    _indx = J[3]
    trap = get_discretization(wrap)
    po_jacobian_sparse!(trap, _J, x, p, _indx)
end

_jacobian_po(wrap::PeriodicOrbitFunctionalTrap, J::FullLU, x, p) = po_jacobian_sparse(get_discretization(wrap), x, p)
POTrapJacobianBordered
_jacobian_po(::PeriodicOrbitFunctionalTrap, J::POTrapJacobianBordered, x, p) = J(x, p)
##########################
# newton wrappers
function _newton_po_from_disc(trap::Trapeze,
                        orbitguess,
                        options::NewtonPar;
                        defOp::Union{Nothing, DeflationOperator} = nothing,
                        kwargs...)
    # hack to test the use of CUDA
    @assert sum(_extract_period_fdtrap(trap, orbitguess)) >= 0 "The guess for the period should be positive"
    jacobianPO = trap.jacobian
    @assert jacobianPO in _trapezoid_jacobian_type "This jacobian is not defined. Please choose another one."
    M, N = size(trap)

    if jacobianPO in (Dense(), AutoDiffDense(), FullLU(), FullMatrixFree(), FullSparseInplace(), AutoDiffMF())
        jac = _generate_jacobian(trap, trap.jacobian, orbitguess, getparams(trap))
        wrap_prob = PeriodicOrbitFunctionalTrap(trap, jac, orbitguess, nothing, nothing)
        new_options = options # to prevent from duplicated code
    else # bordered linear solvers
        if jacobianPO === BorderedLU()
            Aγ = AγOperatorLU(N = N, Jc = LA.lu(SPA.spdiagm( 0 => ones(N * (M - 1)) )), prob = trap)
            # linear solver
            lspo = PeriodicOrbitTrapBLS()
        elseif jacobianPO === BorderedSparseInplace()
            _J =  jacobian_cyclic_sparse(trap, orbitguess, getparams(trap.prob_vf))
            _indx = _get_blocks_from_sparse_matrix(_J, N, M-1)
            # inplace modification of the jacobian _J
            Aγ = AγOperatorSparseInplace(Jc = _J,  Jcfact = LA.lu(_J), prob = trap, indx = _indx)
            lspo = PeriodicOrbitTrapBLS()

        else # BorderedMatrixFree()
            Aγ = AγOperatorMatrixFree(prob = trap, orbitguess = zeros(N * M + 1), par = getparams(trap.prob_vf))
            # linear solver
            lspo = PeriodicOrbitTrapBLS(BorderingBLS(solver = AγLinearSolver(options.linsolver), check_precision = false))
        end

        jacPO = POTrapJacobianBordered(zeros(N * M + 1), Aγ)
        wrap_prob = PeriodicOrbitFunctionalTrap(trap, jacPO, orbitguess, nothing, nothing)
        new_options = @set options.linsolver = lspo
    end

    if isnothing(defOp)
        return solve(wrap_prob, Newton(), new_options; kwargs...)
    else
        return solve(wrap_prob, defOp, new_options; kwargs...)
    end
end

"""
$(TYPEDSIGNATURES)

Locate a periodic orbit with a Newton solver applied to the finite-difference functional `G` of the [`Trapeze`](@ref) discretization, i.e. solve ``G(u) = 0``. The returned solution has `u[end] = T` equal to the period of the orbit.

# Arguments:
- `trap` a problem of type [`Trapeze`](@ref) encoding the functional `G`; its `jacobian` field selects the linear algebra used, see below.
- `orbitguess` a guess for the periodic orbit. See [`Trapeze`](@ref) for more details.
- `options` same as for the regular `newton` method.
$DocStrjacobianPOTrap
""" # TODO This is a bit of a hack. It should be a Functional not a discretization like Collocation
newton(trap::Trapeze,
        orbitguess,
        options::NewtonPar;
        kwargs...) = _newton_po_from_disc(trap, orbitguess, options; defOp = nothing, kwargs...)

"""
$(TYPEDSIGNATURES)

This function is similar to `newton` except that it uses deflation in order to find periodic orbits different from the ones stored in `defOp`. We refer to the mentioned method for a full description of the arguments. The current method can be used in the vicinity of a Hopf bifurcation to prevent the Newton algorithm from converging to the equilibrium point.
""" # TODO This is a bit of a hack. It should be a Functional not a discretization like Collocation
newton(trap::Trapeze,
        orbitguess::vectype,
        defOp::DeflationOperator{Tp, Tdot, T, vectype},
        options::NewtonPar;
        kwargs...) where {Tp, Tdot, T, vectype} = _newton_po_from_disc(trap, orbitguess, options; defOp, kwargs...)

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# continuation wrapper
"""
$(TYPEDSIGNATURES)

Continue a branch of periodic orbits computed with the finite-difference functional `G` of the [`Trapeze`](@ref) discretization.

# Arguments
- `trap` a problem of type [`Trapeze`](@ref) encoding the functional `G`.
- `orbitguess` a guess for a first periodic orbit. See [`Trapeze`](@ref) for more details.
- `alg` continuation algorithm.
- `contParams` same as for the regular [`continuation`](@ref) method.
- `linear_algo` same as in [`continuation`](@ref).

# Keywords arguments
- `eigsolver` specify an eigen solver for the computation of the Floquet exponents, defaults to `FloquetQaD`.
- `record_from_solution` function to record the solution, see [`continuation`](@ref).
- `plot_solution` function to plot the solution, see [`continuation`](@ref).

$DocStrjacobianPOTrap

Note that by default, the method prints the period of the periodic orbit as function of the parameter. This can be changed by providing your `record_from_solution` argument.
"""
function continuation_po(trap::Trapeze,
                            orbitguess,
                            alg::AbstractContinuationAlgorithm,
                            contParams::ContinuationPar,
                            linear_algo::AbstractBorderedLinearSolver;
                            eigsolver = FloquetQaD(contParams.newton_options.eigsolver),
                            record_from_solution = nothing,
                            plot_solution = nothing,
                            kwargs...)
    # this hack is for the test to work with CUDA
    @assert sum(_extract_period_fdtrap(trap, orbitguess)) >= 0 "The guess for the period should be positive"
    jacobianPO = trap.jacobian
    @assert jacobianPO in _trapezoid_jacobian_type "This jacobian is not defined.\nPlease chose another in $_trapezoid_jacobian_type."

    M, N = size(trap)
    options = contParams.newton_options

    # we need to specialize the eigensolver for the computation of Floquet coefficients
    if compute_eigenelements(contParams)
        contParams = @set contParams.newton_options.eigsolver =
         eigsolver
    end

    # this is to remove this part from the arguments passed to continuation
    _kwargs = (; plot_solution)
    record_po = RecordForPeriodicOrbits(record_from_solution, BifurcationKit.record_from_solution(trap.prob_vf))
    _plotsol = modify_po_plot(trap, getparams(trap.prob_vf), getlens(trap.prob_vf); _kwargs...)

    if jacobianPO in (Dense(), AutoDiffDense(), FullLU(), FullMatrixFree(), FullSparseInplace(), AutoDiffMF())
        jac = _generate_jacobian(trap, trap.jacobian, orbitguess, getparams(trap))
        probwp = PeriodicOrbitFunctionalTrap(trap, jac, orbitguess, _plotsol, record_po)
        kwargs_continuation = (kwargs...,
                                kind = PeriodicOrbitCont(),
                                linear_algo,)
    else
        if jacobianPO == BorderedLU()
            Aγ = AγOperatorLU(;N, Jc = LA.lu(SPA.spdiagm( 0 => ones(N * (M - 1)) )), prob = trap)
            # linear solver
            lspo = PeriodicOrbitTrapBLS()
        elseif jacobianPO == BorderedSparseInplace()
            _J =  jacobian_cyclic_sparse(trap, orbitguess, getparams(trap))
            _indx = _get_blocks_from_sparse_matrix(_J, N, M-1)
            # inplace modification of the jacobian _J
            Aγ = AγOperatorSparseInplace(;Jc = _J,  Jcfact = LA.lu(_J), prob = trap, indx = _indx)
            lspo = PeriodicOrbitTrapBLS()

        else # BorderedMatrixFree
            Aγ = AγOperatorMatrixFree(prob = trap, orbitguess = zeros(N * M + 1), par = getparams(trap.prob_vf))
            # linear solver
            lspo = PeriodicOrbitTrapBLS(BorderingBLS(solver = AγLinearSolver(options.linsolver), check_precision = false))
        end

        # we define a specific jacobian for this case
        jac = POTrapJacobianBordered(zeros(N * M + 1), Aγ)
        probwp = PeriodicOrbitFunctionalTrap(trap, jac, orbitguess, _plotsol, record_po)
        # we change the linear solver
        contParams = @set contParams.newton_options.linsolver = lspo
        # we have to change the Bordered linearsolver to cope with our lspo
        linear_algo = @set linear_algo.solver = contParams.newton_options.linsolver
        alg = update(alg, contParams, linear_algo)
        kwargs_continuation = (kwargs...,
                                kind = PeriodicOrbitCont(),
                                )
    end
    return continuation(probwp, alg,
                contParams;
                kwargs_continuation...,
                )
end

"""
$(TYPEDSIGNATURES)

Convenience wrapper around `continuation_po` to continue a branch of periodic orbits computed with the [`Trapeze`](@ref) finite-difference functional.

# Arguments
- `trap` a problem of type [`Trapeze`](@ref) encoding the functional `G`.
- `orbitguess` a guess for a first periodic orbit. See [`Trapeze`](@ref) for more details.
- `alg` continuation algorithm.
- `contParams` same as for the regular [`continuation`](@ref) method.

# Keyword arguments

- `linear_algo` same as in [`continuation`](@ref); it defaults to a bordered linear solver based on `contParams.newton_options.linsolver`.
- `record_from_solution` function used to record the solution on the branch; it defaults to `(u, p; k...) -> (period = u[end],)` so that the period is printed along the branch.
$DocStrjacobianPOTrap

Note that by default, the method prints the period of the periodic orbit as function of the parameter. This can be changed by providing your `record_from_solution` argument.
""" # TODO This is a bit of a hack. It should be a Functional not a discretization like Collocation
function continuation(trap::Trapeze,
                        orbitguess,
                        alg::AbstractContinuationAlgorithm,
                        _contParams::ContinuationPar;
                        record_from_solution = (u, p; k...) -> (period = u[end],),
                        linear_algo = nothing,
                        kwargs...)
    _linear_algo = isnothing(linear_algo) ?  BorderingBLS(solver = _contParams.newton_options.linsolver, check_precision = false) : linear_algo
    return continuation_po(trap, orbitguess, alg, _contParams, _linear_algo; record_from_solution, kwargs...)
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# function needed for automatic Branch switching from Hopf bifurcation point
function re_make(trap::Trapeze,
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
    probPO = setproperties(trap; N, prob_vf, ϕ = zeros(N * M), xπ = zeros(N * M))

    orbit = get(kwargs, :orbit, nothing)

    if isnothing(orbit)
        probPO.ϕ[1:N] .= real.(ζr)
        probPO.xπ[1:N] .= hopfpt.x0
    else
        probPO.xπ .= orbitguess[begin:end-1]
        _sol = get_periodic_orbit(probPO, orbitguess, nothing)
        probPO.ϕ .= reduce(vcat, [residual(prob_vf, _sol.u[:, i], getparams(prob_vf)) for i = 1:probPO.M])
    end
    return probPO, orbitguess
end

using SciMLBase: AbstractTimeseriesSolution

"""
$(TYPEDSIGNATURES)

Generate a guess for the periodic orbit and an updated `Trapeze` problem from an `ODE` solution.

## Arguments
- `trap` a `Trapeze` discretization, used for its mesh and default parameters (its number of time slices `M` is kept).
- `bifprob` a bifurcation problem to provide the vector field.
- `sol` an `AbstractTimeseriesSolution` (e.g. the output of `solve` on an `ODEProblem`) approximating the periodic orbit.
- `tspan` a `Tuple (t0, t1)` giving the time span (period) of the periodic orbit. If a single `period::Real` is passed instead, it is interpreted as `tspan = (0, period)`.

## Keywords
- `optimal_period::Bool = true`: when `true`, the period `tspan[2] - tspan[1]` is refined by minimizing the distance between ``x(t_0)`` and ``x(t_0 + T)`` for ``T`` close to the guess.
- `ktrap...` additional keywords passed to `setproperties` to update the discretization (e.g. `massmatrix`, `update_section_every_step`, `jacobian`).

## Output
- returns a `Trapeze` problem (whose `prob_vf` is set from `bifprob`, with the parameters of `sol`, and whose phase section `(ϕ, xπ)` is initialized from the solution) together with the corresponding initial guess for the periodic orbit (with the period appended as its last component). This guess can be fed directly to `newton` or `continuation`.
"""
function generate_ci_problem(trap::Trapeze,
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
    probtrap = setproperties(trap; M = trap.M, N, prob_vf, xπ = copy(u0), ϕ = copy(u0), ktrap...)

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
    probtrap.ϕ .= reduce(vcat, [residual(bifprob, _sol.u[:, i], sol.prob.p) for i = 1:probtrap.M])
    return probtrap, ci
end

generate_ci_problem(trap::Trapeze, bifprob::AbstractBifurcationProblem, sol::AbstractTimeseriesSolution, period::Real; ktrap...) = generate_ci_problem(trap, bifprob, sol, (zero(period), period); ktrap...)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
