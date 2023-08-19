using FastGaussQuadrature: gausslegendre
# using PreallocationTools: dualcache, get_tmp


"""
    cache = MeshCollocationCache(Ntst::Int, m::Int, Ty = Float64)

Structure to hold the cache for the collocation method.

$(TYPEDFIELDS)

# Constructor

    MeshCollocationCache(Ntst::Int, m::Int, Ty = Float64)

- `Ntst` number of time steps
- `m` degree of the collocation polynomials
- `Ty` type of the time variable
"""
struct MeshCollocationCache{T}
    "Coarse mesh size"
    Ntst::Int
    "Collocation degree, usually called m"
    degree::Int
    "Lagrange matrix"
    lagrange_vals::Matrix{T}
    "Lagrange matrix for derivative"
    lagrange_∂::Matrix{T}
    "Gauss nodes"
    gauss_nodes::Vector{T}
    "Gauss weights"
    gauss_weight::Vector{T}
    "Values for the coarse mesh, call τj. This can be adapted."
    mesh::Vector{T}
    "Values for collocation poinnts, call σj. These are fixed."
    mesh_coll::LinRange{T}
    "Full mesh containing both the coarse mesh and the collocation points."
    full_mesh::Vector{T}
end

function MeshCollocationCache(Ntst::Int, m::Int, Ty = Float64)
    τs = LinRange{Ty}( 0, 1, Ntst + 1) |> collect
    σs = LinRange{Ty}(-1, 1, m + 1)
    L, ∂L = getL(σs)
    zg, wg = gausslegendre(m)
    cache = MeshCollocationCache(Ntst, m, L, ∂L, zg, wg, τs, σs, zeros(Ty, 1 + m * Ntst))
    # put the mesh where we removed redundant timing
    cache.full_mesh .= get_times(cache)
    return cache
end

@inline Base.eltype(pb::MeshCollocationCache{T}) where T = T
@inline Base.size(pb::MeshCollocationCache) = (pb.degree, pb.Ntst)
@inline get_Ls(pb::MeshCollocationCache) = (pb.lagrange_vals, pb.lagrange_∂)
@inline getmesh(pb::MeshCollocationCache) = pb.mesh
@inline get_mesh_coll(pb::MeshCollocationCache) = pb.mesh_coll
get_max_time_step(pb::MeshCollocationCache) = maximum(diff(getmesh(pb)))
τj(σ, τs, j) = τs[j] + (1 + σ)/2 * (τs[j+1] - τs[j])
# get the sigma corresponding to τ in the interval (𝜏s[j], 𝜏s[j+1])
σj(τ, τs, j) = -(2*τ - τs[j] - τs[j + 1])/(-τs[j + 1] + τs[j])

# code from Jacobi.lagrange
function lagrange(i::Int, x, z)
    nz = length(z)
    l = one(z[1])
    for k in 1:(i-1)
        l = l * (x - z[k]) / (z[i] - z[k])
    end
    for k in (i+1):nz
        l = l * (x - z[k]) / (z[i] - z[k])
    end
    return l
end

dlagrange(i, x, z) = ForwardDiff.derivative(x -> lagrange(i, x, z), x)

# should accept a range, ie σs = LinRange(-1, 1, m + 1)
function getL(σs::AbstractVector)
    m = length(σs) - 1
    zs, = gausslegendre(m)
    L = zeros(m, m + 1); ∂L = zeros(m, m + 1)
    for j in 1:m+1
        for i in 1:m
             L[i, j] =  lagrange(j, zs[i], σs)
            ∂L[i, j] = dlagrange(j, zs[i], σs)
        end
    end
    return (;L, ∂L)
end

"""
$(SIGNATURES)

Return all the times at which the problem is evaluated.
"""
function get_times(pb::MeshCollocationCache)
    m, Ntst = size(pb)
    Ty = eltype(pb)
    ts = zero(Ty)
    tsvec = Ty[0]
    τs = pb.mesh
    σs = pb.mesh_coll
    for j in 1:Ntst
        for l in 1:m+1
            ts = τj(σs[l], τs, j)
            l>1 && push!(tsvec, τj(σs[l], τs, j))
        end
    end
    return vec(tsvec)
end

function update_mesh!(pb::MeshCollocationCache, mesh)
    pb.mesh .= mesh
    pb.full_mesh .= get_times(pb)
end
####################################################################################################
"""
cache to remove allocations from PeriodicOrbitOCollProblem
"""
struct POCollCache{T}
    gj::T
    gi::T
    ∂gj::T
    uj::T
end

function POCollCache(Ty::Type, n::Int, m::Int)
    gj  = (zeros(Ty, n, m), [n, m])
    gi  = (zeros(Ty, n, m), [n, m])
    ∂gj = (zeros(Ty, n, m), [n, m])
    uj  = (zeros(Ty, n, m+1), [n, (1 + m)])
    return POCollCache(gj, gi, ∂gj, uj)
end
####################################################################################################

"""
    pb = PeriodicOrbitOCollProblem(kwargs...)

This composite type implements an orthogonal collocation (at Gauss points) method of piecewise polynomials to locate periodic orbits. More details (maths, notations, linear systems) can be found [here](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/periodicOrbitCollocation/).

## Arguments
- `prob` a bifurcation problem
- `ϕ::AbstractVector` used to set a section for the phase constraint equation
- `xπ::AbstractVector` used in the section for the phase constraint equation
- `N::Int` dimension of the state space
- `mesh_cache::MeshCollocationCache` cache for collocation. See docs of `MeshCollocationCache`
- `updateSectionEveryStep` updates the section every `updateSectionEveryStep` step during continuation
- `jacobian::Symbol` symbol which describes the type of jacobian used in Newton iterations. Can only be `:autodiffDense`.
- `meshadapt::Bool = false` whether to use mesh adaptation
- `verboseMeshAdapt::Bool = true` verbose mesh adaptation information
- `K::Float64 = 500` parameter for mesh adaptation, control new mesh step size

## Methods

Here are some useful methods you can apply to `pb`

- `length(pb)` gives the total number of unknowns
- `size(pb)` returns the triplet `(N, m, Ntst)`
- `getmesh(pb)` returns the mesh `0 = τ0 < ... < τNtst+1 = 1`. This is useful because this mesh is born to vary by automatic mesh adaptation
- `get_mesh_coll(pb)` returns the (static) mesh `0 = σ0 < ... < σm+1 = 1`
- `get_times(pb)` returns the vector of times (length `1 + m * Ntst`) at the which the collocation is applied.
- `generate_solution(pb, orbit, period)` generate a guess from a function `t -> orbit(t)` which approximates the periodic orbit.
- `POSolution(pb, x)` return a function interpolating the solution `x` using a piecewise polynomials function

# Orbit guess
You will see below that you can evaluate the residual of the functional (and other things) by calling `pb(orbitguess, p)` on an orbit guess `orbitguess`. Note that `orbitguess` must be of size 1 + N * (1 + m * Ntst) where N is the number of unknowns in the state space and `orbitguess[end]` is an estimate of the period ``T`` of the limit cycle.

# Constructors
- `PeriodicOrbitOCollProblem(Ntst::Int, m::Int; kwargs)` creates an empty functional with `Ntst`and `m`.

Note that you can generate this guess from a function using `generate_solution`.

# Functional
 A functional, hereby called `G`, encodes this problem. The following methods are available

- `pb(orbitguess, p)` evaluates the functional G on `orbitguess`
"""
@with_kw_noshow struct PeriodicOrbitOCollProblem{Tprob <: Union{Nothing, AbstractBifurcationProblem}, Tjac <: AbstractJacobianType, vectype, Tmass, Tmcache <: MeshCollocationCache, Tcache} <: AbstractPODiffProblem
    # Function F(x, par)
    prob_vf::Tprob = nothing

    # variables to define a Section for the phase constraint equation
    ϕ::vectype = nothing
    xπ::vectype = nothing

    # dimension of the problem in case of an AbstractVector
    N::Int = 0

    # whether the time discretisation is adaptive
    adaptmesh::Bool = false

    # whether the problem is nonautonomous
    isautonomous::Bool = true

    # mass matrix
    massmatrix::Tmass = nothing

    # update the section every step
    updateSectionEveryStep::Int = 1

    # symbol to control the way the jacobian of the functional is computed
    jacobian::Tjac = AutoDiffDense()

    # collocation mesh cache
    mesh_cache::Tmcache = nothing

    # collocation mesh cache
    cache::Tcache = nothing

    #################
    # mesh adaptation
    meshadapt::Bool = false

    # verbose mesh adaptation information
    verboseMeshAdapt::Bool = false

    # parameter for mesh adaptation, control maximum mesh step size
    K::Float64 = 100
end

# trivial constructor
function PeriodicOrbitOCollProblem(Ntst::Int, m::Int, Ty = Float64; kwargs...)
    N = get(kwargs, :N, 1)
    PeriodicOrbitOCollProblem(; mesh_cache = MeshCollocationCache(Ntst, m, Ty),
                                    cache = POCollCache(Ty, N, m),
                                    kwargs...)
end

"""
$(SIGNATURES)

This function change the parameters `Ntst, m` for the collocation problem `pb` and return a new problem.
"""
function setCollocationSize(pb::PeriodicOrbitOCollProblem, Ntst, m)
    pb2 = @set pb.mesh_cache = MeshCollocationCache(Ntst, m, eltype(pb))
    resize!(pb2.ϕ, length(pb2))
    resize!(pb2.xπ, length(pb2))
    pb2
end

@inline get_mesh_size(pb::PeriodicOrbitOCollProblem) = pb.mesh_cache.Ntst

"""
The method `size` returns (n, m, Ntst) when applied to a `PeriodicOrbitOCollProblem`
"""
@inline Base.size(pb::PeriodicOrbitOCollProblem) = (pb.N, size(pb.mesh_cache)...)

@inline function length(pb::PeriodicOrbitOCollProblem)
    n, m, Ntst = size(pb)
    return n * (1 + m * Ntst)
end

@inline Base.eltype(pb::PeriodicOrbitOCollProblem) = eltype(pb.mesh_cache)
get_Ls(pb::PeriodicOrbitOCollProblem) = get_Ls(pb.mesh_cache)

@inline getparams(pb::PeriodicOrbitOCollProblem) = getparams(pb.prob_vf)
@inline getlens(pb::PeriodicOrbitOCollProblem) = getlens(pb.prob_vf)
@inline setparam(pb::PeriodicOrbitOCollProblem, p) = setparam(pb.prob_vf, p)

# these functions extract the time slices components
get_time_slices(x::AbstractVector, N, degree, Ntst) = reshape(x, N, degree * Ntst + 1)
# array of size Ntst ⋅ (m+1) ⋅ n
get_time_slices(pb::PeriodicOrbitOCollProblem, x) = @views get_time_slices(x[1:end-1], size(pb)...)
get_times(pb::PeriodicOrbitOCollProblem) = get_times(pb.mesh_cache)
"""
Returns the vector of size m+1,  0 = τ1 < τ1 < ... < τm+1 = 1
"""
getmesh(pb::PeriodicOrbitOCollProblem) = getmesh(pb.mesh_cache)
get_mesh_coll(pb::PeriodicOrbitOCollProblem) = get_mesh_coll(pb.mesh_cache)
get_max_time_step(pb::PeriodicOrbitOCollProblem) = get_max_time_step(pb.mesh_cache)
update_mesh!(pb::PeriodicOrbitOCollProblem, mesh) = update_mesh!(pb.mesh_cache, mesh)
@inline isinplace(pb::PeriodicOrbitOCollProblem) = isinplace(pb.prob_vf)
@inline is_symmetric(pb::PeriodicOrbitOCollProblem) = is_symmetric(pb.prob_vf)
@inline getdelta(pb::PeriodicOrbitOCollProblem) = getdelta(pb.prob_vf)

@inline getdelta(pb::WrapPOColl) = getdelta(pb.prob)
@inline has_adjoint(::WrapPOColl) = false #c'est dans problems.jl

function Base.show(io::IO, pb::PeriodicOrbitOCollProblem)
    N, m, Ntst = size(pb)
    println(io, "┌─ Collocation functional for periodic orbits")
    println(io, "├─ type               : Vector{", eltype(pb), "}")
    println(io, "├─ time slices (Ntst) : ", Ntst)
    println(io, "├─ degree      (m)    : ", m)
    println(io, "├─ dimension   (N)    : ", pb.N)
    println(io, "├─ inplace            : ", isinplace(pb))
    println(io, "├─ update section     : ", pb.updateSectionEveryStep)
    println(io, "├─ jacobian           : ", pb.jacobian)
    println(io, "├─ mesh adaptation    : ", pb.meshadapt)
    println(io, "└─ # unknowns         : ", pb.N * (1 + m * Ntst))
end

"""
$(SIGNATURES)

This function generates an initial guess for the solution of the problem `pb` based on the orbit `t -> orbit(t * period)` for t ∈ [0,1] and the `period`.
"""
function generate_solution(pb::PeriodicOrbitOCollProblem, orbit, period)
    n, _m, Ntst = size(pb)
    ts = get_times(pb)
    Nt = length(ts)
    ci = zeros(eltype(pb), n, Nt)
    for (l, t) in pairs(ts)
        ci[:, l] .= orbit(t * period)
    end
    return vcat(vec(ci), period)
end

using SciMLBase: AbstractTimeseriesSolution
"""
$(SIGNATURES)

Generate a periodic orbit problem from a solution.

## Arguments
- `pb` a `PeriodicOrbitOCollProblem` which provides basic information, like the number of time slices `M`
- `bifprob` a bifurcation problem to provide the vector field
- `sol` basically, and `ODEProblem
- `period` estimate of the period of the periodic orbit

## Output
- returns a `PeriodicOrbitOCollProblem` and an initial guess.
"""
function generate_ci_problem(pb::PeriodicOrbitOCollProblem,
                            bifprob::AbstractBifurcationProblem,
                            sol::AbstractTimeseriesSolution,
                            period)
    u0 = sol(0)
    @assert u0 isa AbstractVector
    N = length(u0)

    n, m, Ntst = size(pb)
    nunknows = N * (1 + m * Ntst)

    par = sol.prob.p
    prob_vf = re_make(bifprob, params = par)

    pbcoll = setproperties(pb, N = N, prob_vf = prob_vf, ϕ = zeros(nunknows), xπ = zeros(nunknows), cache = POCollCache(eltype(pb), N, m))

    ci = generate_solution(pbcoll, t -> sol(t), period)
    pbcoll.ϕ .= @view ci[1:end-1]

    return pbcoll, ci
end

# @views function phase_condition(prob::PeriodicOrbitOCollProblem, u)
#     dot(u[1:end-1], prob.ϕ) - dot(prob.xπ, prob.ϕ)
# end

"""
$(SIGNATURES)

[INTERNAL] Implementation of ∫_0^T < u(t), v(t) > dt.
# Arguments
- uj  n x (m + 1)
- vj  n x (m + 1)
"""
@views function ∫(pb::PeriodicOrbitOCollProblem, uc, vc, T = one(eltype(uc)))
    Ty = eltype(uc)
    phase = zero(Ty)

    n, m, Ntst = size(pb)
    L, ∂L = get_Ls(pb.mesh_cache)
    ω = pb.mesh_cache.gauss_weight
    mesh = pb.mesh_cache.mesh

    guj = zeros(Ty, n, m)
    uj  = zeros(Ty, n, m+1)

    gvj = zeros(Ty, n, m)
    vj  = zeros(Ty, n, m+1)

    rg = UnitRange(1, m+1)
    @inbounds for j in 1:Ntst
        uj .= uc[:, rg]
        vj .= vc[:, rg]
        mul!(guj, uj, L')
        mul!(gvj, vj, L')
        @inbounds for l in 1:m
            phase += dot(guj[:, l], gvj[:, l]) * ω[l] * (mesh[j+1] - mesh[j]) / 2
        end
        rg = rg .+ m
    end
    return phase * T
end

"""
$(SIGNATURES)

[INTERNAL] Implementation of phase condition ∫_0^T < u(t), ∂ϕ(t) > dt.
# Arguments
- uj   n x (m + 1)
- guj  n x m
"""
@views function phase_condition(pb::PeriodicOrbitOCollProblem, (u, uc), (L, ∂L), period)
    𝒯 = eltype(uc)
    phase = zero(𝒯)

    n, m, Ntst = size(pb)

    guj = zeros(𝒯, n, m)
    uj  = zeros(𝒯, n, m+1)

    vc = get_time_slices(pb.ϕ, size(pb)...)
    gvj = zeros(𝒯, n, m)
    vj  = zeros(𝒯, n, m+1)

    ω = pb.mesh_cache.gauss_weight

    rg = UnitRange(1, m+1)
    @inbounds for j in 1:Ntst
        uj .= uc[:, rg]
        vj .= vc[:, rg]
        mul!(guj, uj, L')
        mul!(gvj, vj, ∂L')
        @inbounds for l in 1:m
            phase += dot(guj[:, l], gvj[:, l]) * ω[l]
        end
        rg = rg .+ m
    end
    return phase / period
end

function _POOCollScheme!(pb::PeriodicOrbitOCollProblem, dest, ∂u, u, par, h, tmp)
    applyF(pb, tmp, u, par)
    dest .= @. ∂u - h * tmp
end

# function for collocation problem
@views function functional_coll_bare!(pb::PeriodicOrbitOCollProblem, out, u, period, (L, ∂L), pars)
    𝒯 = eltype(u)
    n, ntimes = size(u)
    m = pb.mesh_cache.degree
    Ntst = pb.mesh_cache.Ntst
    # we want slices at fixed times, hence gj[:, j] is the fastest
    # temporaries to reduce allocations
    # TODO VIRER CES TMP?
    gj  = zeros(𝒯, n, m)
    ∂gj = zeros(𝒯, n, m)
    uj  = zeros(𝒯, n, m+1)

    mesh = getmesh(pb)
    # range for locating time slices
    rg = UnitRange(1, m+1)
    for j in 1:Ntst
        uj .= u[:, rg]
        mul!(gj, uj, L')
        mul!(∂gj, uj, ∂L')
        # compute the collocation residual
        for l in 1:m
            # out[:, end] serves as buffer for now
            # @info "" j l rg[l] gj[:, l] period*(mesh[j+1]-mesh[j])/2
            _POOCollScheme!(pb, out[:, rg[l]], ∂gj[:, l], gj[:, l], pars, period * (mesh[j+1]-mesh[j]) / 2, out[:, end])
        end
        # carefull here https://discourse.julialang.org/t/is-this-a-bug-scalar-ranges-with-the-parser/70670/4"
        rg = rg .+ m
    end
    out
end

@views function functional_coll!(pb::PeriodicOrbitOCollProblem, out, u, period, (L, ∂L), pars)
    functional_coll_bare!(pb, out, u, period, (L, ∂L), pars)
    # add the periodicity condition
    out[:, end] .= u[:, end] .- u[:, 1]
end

@views function (prob::PeriodicOrbitOCollProblem)(u::AbstractVector, pars)
    uc = get_time_slices(prob, u)
    T = getperiod(prob, u, nothing)
    result = zero(u)
    resultc = get_time_slices(prob, result)
    functional_coll!(prob, resultc, uc, T, get_Ls(prob.mesh_cache), pars)
    # add the phase condition ∫_0^T < u(t), ∂ϕ(t) > dt
    result[end] = phase_condition(prob, (u, uc), get_Ls(prob.mesh_cache), T)
    return result
end

"""
$(SIGNATURES)

Compute the full periodic orbit associated to `x`. Mainly for plotting purposes.
"""
@views function get_periodic_orbit(prob::PeriodicOrbitOCollProblem, u, p)
    T = getperiod(prob, u, p)
    ts = get_times(prob)
    uc = get_time_slices(prob, u)
    return SolPeriodicOrbit(t = ts .* T, u = uc)
end

# simplified function to extract periodic orbit
get_periodic_orbit(prob::PeriodicOrbitOCollProblem, x, p::Real) = get_periodic_orbit(prob, x, setparam(prob, p))

# same function as above but for coping with mesh adaptation
@views function get_periodic_orbit(prob::PeriodicOrbitOCollProblem, x::NamedTuple{(:mesh, :sol, :_mesh), Tuple{Vector{Tp}, Vector{Tp}, Vector{Tp}}}, p) where Tp
    mesh = x.mesh
    u = x.sol
    T = getperiod(prob, u, p)
    uc = get_time_slices(prob, u)
    return SolPeriodicOrbit(t = mesh .* T, u = uc)
end

# function needed for automatic Branch switching from Hopf bifurcation point
function re_make(prob::PeriodicOrbitOCollProblem, prob_vf, hopfpt, ζr::AbstractVector, orbitguess_a, period; orbit = t->t, k...)
    M = length(orbitguess_a)
    N = length(ζr)

    _, m, Ntst = size(prob)
    nunknows = N * (1 + m * Ntst)

    # update the problem
    probPO = setproperties(prob, N = N, prob_vf = prob_vf, ϕ = zeros(nunknows), xπ = zeros(nunknows), cache = POCollCache(eltype(prob), N, m))

    probPO.xπ .= 0

    ϕ0 = generate_solution(probPO, t -> orbit(2pi*t/period + pi), period)
    probPO.ϕ .= @view ϕ0[1:end-1]

    # append period at the end of the initial guess
    orbitguess = generate_solution(probPO, t -> orbit(2pi*t/period), period)

    return probPO, orbitguess
end

residual(prob::WrapPOColl, x, p) = prob.prob(x, p)
jacobian(prob::WrapPOColl, x, p) = prob.jacobian(x, p)
@inline is_symmetric(prob::WrapPOColl) = is_symmetric(prob.prob)

# for recording the solution in a branch
function getsolution(wrap::WrapPOColl, x)
    if wrap.prob.meshadapt
        return (mesh = copy(get_times(wrap.prob)), sol = x, _mesh = copy(wrap.prob.mesh_cache.mesh))
    else
        return x
    end
end
####################################################################################################
const DocStrjacobianPOColl = """
- `jacobian` Specify the choice of the linear algorithm, which must belong to `(:autodiffDense, )`. This is used to select a way of inverting the jacobian dG
    - For `:autodiffDense`. The jacobian is formed as a dense Matrix. You can use a direct solver or an iterative one using `options`. The jacobian is formed inplace.
"""

function _newton_pocoll(probPO::PeriodicOrbitOCollProblem,
            orbitguess,
            options::NewtonPar;
            defOp::Union{Nothing, DeflationOperator{T, Tf, vectype}} = nothing,
            kwargs...) where {T, Tf, vectype}
    jacobianPO = probPO.jacobian
    @assert jacobianPO in
            (AutoDiffDense(), ) "This jacobian $jacobianPO is not defined. Please chose another one."

    if jacobianPO isa AutoDiffDense
        jac = (x, p) -> ForwardDiff.jacobian(z -> probPO(z, p), x)
    end

    prob = WrapPOColl(probPO, jac, orbitguess, getparams(probPO), getlens(probPO), nothing, nothing)

    if isnothing(defOp)
        return newton(prob, options; kwargs...)
        # return newton(probPO, jac, orbitguess, par, options; kwargs...)
    else
        # return newton(probPO, jac, orbitguess, par, options, defOp; kwargs...)
        return newton(prob, defOp, options; kwargs...)
    end
end

"""
$(SIGNATURES)

This is the Newton Solver for computing a periodic orbit using orthogonal collocation method.
Note that the linear solver has to be apropriately set up in `options`.

# Arguments

Similar to [`newton`](@ref) except that `prob` is a [`PeriodicOrbitOCollProblem`](@ref).

- `prob` a problem of type `<: PeriodicOrbitOCollProblem` encoding the shooting functional G.
- `orbitguess` a guess for the periodic orbit.
- `options` same as for the regular [`newton`](@ref) method.

# Optional argument
$DocStrjacobianPOColl
"""
newton(probPO::PeriodicOrbitOCollProblem,
            orbitguess,
            options::NewtonPar;
            kwargs...) = _newton_pocoll(probPO, orbitguess, options; defOp = nothing, kwargs...)

"""
    $(SIGNATURES)

This function is similar to `newton(probPO, orbitguess, options, jacobianPO; kwargs...)` except that it uses deflation in order to find periodic orbits different from the ones stored in `defOp`. We refer to the mentioned method for a full description of the arguments. The current method can be used in the vicinity of a Hopf bifurcation to prevent the Newton-Krylov algorithm from converging to the equilibrium point.
"""
newton(probPO::PeriodicOrbitOCollProblem,
                orbitguess,
                defOp::DeflationOperator,
                options::NewtonPar;
                kwargs...) =
    _newton_pocoll(probPO, orbitguess, options; defOp = defOp, kwargs...)


function build_jacobian(probPO::PeriodicOrbitOCollProblem, orbitguess, par; δ = convert(eltype(orbitguess), 1e-8))
    jacobianPO = probPO.jacobian
    @assert jacobianPO in (AutoDiffDense(),) "This jacobian is not defined. Please chose another one."
    _J = zeros(eltype(probPO), length(orbitguess), length(orbitguess))
    jac = (x, p) -> FloquetWrapper(probPO, ForwardDiff.jacobian!(_J, z -> probPO(z, p), x), x, p)
end

"""
$(SIGNATURES)

This is the continuation method for computing a periodic orbit using an orthogonal collocation method.

# Arguments

Similar to [`continuation`](@ref) except that `prob` is a [`PeriodicOrbitOCollProblem`](@ref). By default, it prints the period of the periodic orbit.

# Keywords arguments
- `eigsolver` specify an eigen solver for the computation of the Floquet exponents, defaults to `FloquetQaD`
"""
function continuation(probPO::PeriodicOrbitOCollProblem,
                    orbitguess,
                    alg::AbstractContinuationAlgorithm,
                    _contParams::ContinuationPar,
                    linearAlgo::AbstractBorderedLinearSolver;
                    δ = convert(eltype(orbitguess), 1e-8),
                    eigsolver = FloquetColl(),
                    kwargs...)

     jacPO = build_jacobian(probPO, orbitguess, getparams(probPO); δ = δ)

    linearAlgo = @set linearAlgo.solver = FloquetWrapperLS(linearAlgo.solver)
    options = _contParams.newtonOptions
    contParams = @set _contParams.newtonOptions.linsolver = FloquetWrapperLS(options.linsolver)

    # we have to change the Bordered linearsolver to cope with our type FloquetWrapper
    alg = update(alg, contParams, linearAlgo)

    if compute_eigenelements(contParams)
        contParams = @set contParams.newtonOptions.eigsolver = eigsolver
    end

    # change the user provided finalise function by passing prob in its parameters
    _finsol = modify_po_finalise(probPO, kwargs, probPO.updateSectionEveryStep)
    _recordsol = modify_po_record(probPO, kwargs, getparams(probPO.prob_vf), getlens(probPO.prob_vf))
    _plotsol = modify_po_plot(probPO, kwargs)

    probwp = WrapPOColl(probPO, jacPO, orbitguess, getparams(probPO), getlens(probPO), _plotsol, _recordsol)

    br = continuation(probwp, alg,
                    contParams;
                    kwargs...,
                    kind = PeriodicOrbitCont(),
                    finaliseSolution = _finsol)
    return br
end

"""
$(SIGNATURES)

Compute the maximum of the periodic orbit associated to `x`.
"""
function getmaximum(prob::PeriodicOrbitOCollProblem, x::AbstractVector, p)
    sol = get_periodic_orbit(prob, x, p).u
    return maximum(sol)
end

# this function updates the section during the continuation run
@views function updatesection!(prob::PeriodicOrbitOCollProblem, x, par; stride = 0)
    @debug "Update section Collocation"
    # update the reference point
    prob.xπ .= 0

    # update the "normals"
    prob.ϕ .= x[1:end-1]
    return true
end
####################################################################################################
# mesh adaptation method

# iterated derivatives
∂(f) = x -> ForwardDiff.derivative(f, x)
∂(f, n) = n == 0 ? f : ∂(∂(f), n-1)

@views function (sol::POSolution{ <: PeriodicOrbitOCollProblem})(t0)
    n, m, Ntst = size(sol.pb)
    xc = get_time_slices(sol.pb, sol.x)

    T = getperiod(sol.pb, sol.x, nothing)
    t = mod(t0, T) / T

    mesh = getmesh(sol.pb)
    indτ = searchsortedfirst(mesh, t) - 1
    if indτ <= 0
        return sol.x[1:n]
    elseif indτ > Ntst
        return xc[:, end]
    end
    # println("--> ", t, " belongs to ", (mesh[indτ], mesh[indτ+1])) # waste lots of ressources
    @assert mesh[indτ] <= t <= mesh[indτ+1] "Please open an issue on the website of BifurcationKit.jl"
    σ = σj(t, mesh, indτ)
    # @assert -1 <= σ <= 1 "Strange value of $σ"
    σs = get_mesh_coll(sol.pb)
    out = zeros(typeof(t), sol.pb.N)
    rg = (1:m+1) .+ (indτ-1) * m
    for l in 1:m+1
        out .+= xc[:, rg[l]] .* lagrange(l, σ, σs)
    end
    out
end

"""
$(SIGNATURES)

Perform mesh adaptation of the periodic orbit problem. Modify `pb` and `x` inplace if the adaptation is successfull.

Ascher, Uri M., Robert M. M. Mattheij, and Robert D. Russell. Numerical Solution of Boundary Value Problems for Ordinary Differential Equations. Society for Industrial and Applied Mathematics, 1995. https://doi.org/10.1137/1.9781611971231.

p. 368
"""
function compute_error!(pb::PeriodicOrbitOCollProblem, x::Vector{Ty};
                    normE = norm,
                    verbosity::Bool = false,
                    K = Inf,
                    kw...) where Ty
    n, m, Ntst = size(pb)
    period = getperiod(pb, x, nothing)
    # get solution
    sol = POSolution(deepcopy(pb), x)
    # derivative of degree m, indeed ∂(sol, m+1) = 0
    dmsol = ∂(sol, m)
    # we find the values of vm := ∂m(x) at the mid points
    mesh = getmesh(pb)
    meshT = mesh .* period
    vm = [ dmsol( (meshT[i] + meshT[i+1]) / 2 ) for i = 1:Ntst ]
    ############
    # Approx. IA
    # this is the function s^{(k)} in the above paper on page 63
    # we want to estimate sk = s^{(m+1)} which is 0 by definition, pol of degree m
    if isempty(findall(diff(meshT) .<= 0)) == false
        @error "[In mesh-adaptation]. The mesh is non monotonic! Please report the error to the website of BifurcationKit.jl"
        return (success = false, newmeshT = meshT, ϕ = meshT)
    end
    sk = Ty[]
    push!(sk, 2normE(vm[1])/(meshT[2]-meshT[1]))
    for i in 2:Ntst-1
        push!(sk, normE(vm[i]) / (meshT[i+1] - meshT[i-1]) +
                normE(vm[i+1]) / (meshT[i+2] - meshT[i]))
    end
    push!(sk, 2normE(vm[end]) / (meshT[end] - meshT[end-2]))

    ############
    # monitor function
    ϕ = sk.^(1/m)
    ϕ = max.(ϕ, maximum(ϕ) / K)
    @assert length(ϕ) == Ntst "Error. Please open an issue of the website of BifurcationKit.jl"
    # compute θ = ∫ϕ but also all intermediate values
    # these intermediate values are useful because the integral is piecewise linear
    # and equipartition is analytical
    # there are ntst values for the integrals, one for (0, mesh[2]), (mesh[2], mesh[3])...
    θs = zeros(Ty, Ntst); θs[1] = ϕ[1] * (meshT[2] - meshT[1])
    for i = 2:Ntst
        θs[i] = θs[i-1] + ϕ[i] * (meshT[i+1] - meshT[i])
    end
    θs = vcat(0, θs)
    θ = θs[end]

    ############
    # compute new mesh from equipartition
    newmeshT = zero(meshT); newmeshT[end] = 1
    c = θ / Ntst
    for i in 1:Ntst-1
        θeq = i * c
        # we have that θeq ∈ (θs[ind-1], θs[ind])
        ind = searchsortedfirst(θs, θeq)
        @assert 2 <= ind <= Ntst+1 "Error with 1 < $ind <= $(Ntst+1). Please open an issue on the website of BifurcationKit.jl"
        α = (θs[ind] - θs[ind-1]) / (meshT[ind] - meshT[ind-1])
        newmeshT[i+1] = meshT[ind-1] + (θeq - θs[ind-1]) / α
        @assert newmeshT[i+1] > newmeshT[i] "Error. Please open an issue on the website of BifurcationKit.jl"
    end
    newmesh = newmeshT ./ period; newmesh[end] = 1

    if verbosity
        h = maximum(diff(newmesh))
        printstyled(color = :magenta, 
        "   ┌─ Mesh adaptation",
        "\n   ├─── min(hi)       = ", minimum(diff(newmesh)),
        "\n   ├─── h = max(hi)   = ", h,
        "\n   ├─── K = max(h/hi) = ", maximum(h ./ diff(newmesh)),
        "\n   ├─── θ             = ", θ,
        "\n   ├─── min(ϕ)        = ", minimum(ϕ),
        "\n   ├─── max(ϕ)        = ", maximum(ϕ),
        "\n   └─── θ             = ", θ,
        "\n")
    end

    ############
    # modify meshes
    update_mesh!(pb, newmesh)

    ############
    # update solution
    newsol = generate_solution(pb, t -> sol(t), period)
    x .= newsol

    success = true
    return (;success, newmeshT, ϕ)
end
