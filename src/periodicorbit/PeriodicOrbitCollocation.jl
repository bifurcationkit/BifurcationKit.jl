import FastGaussQuadrature

"""
    cache = MeshCollocationCache(Ntst::Int, m::Int, Ty = Float64)

Structure to hold the cache for the collocation method. More precisely, it starts from a partition of [0, 1] based on the mesh points:

    0 = τ₁ < τ₂ < ... < τₙₜₛₜ₊₁ = 1

On each mesh interval [τⱼ, τⱼ₊₁] mapped to [-1, 1], a Legendre polynomial of degree m is formed. 

# Internal fields
$(TYPEDFIELDS)

# Constructor

    MeshCollocationCache(Ntst::Int, m::Int, 𝒯 = Float64)

- `Ntst` number of time steps.
- `m` degree of the collocation polynomials.
- `Ty` type of the time variable.
"""
struct MeshCollocationCache{𝒯}
    "Coarse mesh size."
    Ntst::Int
    "Collocation degree, usually named `m`."
    degree::Int
    "Lagrange matrix."
    lagrange_vals::Matrix{𝒯}
    "Lagrange matrix for derivative."
    lagrange_∂::Matrix{𝒯}
    "Gauss nodes."
    gauss_nodes::Vector{𝒯}
    "Gauss weights. Useful to compute integrals."
    gauss_weight::Vector{𝒯}
    "Values of the coarse mesh, call τj. This can be adapted."
    τs::Vector{𝒯}
    "Values of collocation points, call σj. These are fixed."
    σs::Vector{𝒯}
    "Full mesh containing both the coarse mesh and the collocation points."
    full_mesh::Vector{𝒯}
end

function MeshCollocationCache(Ntst::Int, m::Int, 𝒯 = Float64)
    τs = LinRange{𝒯}( 0, 1, Ntst + 1) |> collect
    σs = LinRange{𝒯}(-1, 1, m + 1) |> collect
    L, ∂L, zg, wg = compute_legendre_matrices(σs)
    cache = MeshCollocationCache{𝒯}(Ntst, m, L, ∂L, zg, wg, τs, σs, zeros(𝒯, 1 + m * Ntst))
    # save the mesh where we removed redundant timing
    cache.full_mesh .= get_times(cache)
    return cache
end

@inline Base.eltype(::MeshCollocationCache{𝒯}) where {𝒯} = 𝒯
@inline Base.size(cache::MeshCollocationCache) = (cache.degree, cache.Ntst)
@inline get_Ls(cache::MeshCollocationCache) = (cache.lagrange_vals, cache.lagrange_∂)
@inline getmesh(cache::MeshCollocationCache) = cache.τs
@inline get_mesh_coll(cache::MeshCollocationCache) = cache.σs
@inline get_full_mesh(cache::MeshCollocationCache) = cache.full_mesh
@inline get_gauss_nodes(cache::MeshCollocationCache) = cache.gauss_nodes
get_max_time_step(cache::MeshCollocationCache) = maximum(diff(getmesh(cache)))
@inline _τj(σ, τⱼ₊₁, τⱼ) = τⱼ + (1 + σ)/2 * (τⱼ₊₁ - τⱼ) # for σ ∈ [-1, 1], τj ∈ [τⱼ, τs[j+1]]
@inline τj(σ, τs, j) = _τj(σ, τs[j+1], τs[j])
"Get the `σ` corresponding to `τ` in the interval (τs[j], τs[j+1])"
@inline σj(τ, τs, j) = (2*τ - τs[j] - τs[j + 1])/(τs[j + 1] - τs[j]) # for τ ∈ [τs[j], τs[j+1]], σj ∈ [-1, 1]

"""
$(TYPEDSIGNATURES)

Evaluate Lagrange polynomial at `x`.
"""
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
function compute_legendre_matrices(σs::AbstractVector{𝒯}) where {𝒯}
    m = length(σs) - 1
    zg, wg = FastGaussQuadrature.gausslegendre(m)
    @assert length(zg) == m
    L  = zeros(𝒯, m + 1, m)
    ∂L = zeros(𝒯, m + 1, m)
    for j in 1:m+1
        for (i,z) in pairs(zg)
             L[j, i] =  lagrange(j, z, σs)
            ∂L[j, i] = dlagrange(j, z, σs)
        end
    end
    return (;L, ∂L, zg, wg)
end

"""
$(TYPEDSIGNATURES)

Return the times at which the problem is evaluated. 

!!! danger "This is a bit tricky"
    In order to remove the obvious continuity conditions at the coarse mesh border, we form the mesh by evaluating `_τj(σs[l], τs[j+1], τs[j])` for `l=2:m+1`. The only point not present is then `t=0` which is added to the list.
"""
function get_times(cache::MeshCollocationCache{𝒯}) where {𝒯}
    m, Ntst = size(cache)
    tsvec = zeros(𝒯, m * Ntst + 1)
    τs = cache.τs
    σs = cache.σs
    ind = 2
    @inbounds for j in 1:Ntst
        for l in 2:m+1
            t = _τj(σs[l], τs[j+1], τs[j])
            tsvec[ind] = t
            ind += 1
        end
    end
    return tsvec
end

function update_mesh!(cache::MeshCollocationCache, τs)
    cache.τs .= τs
    cache.full_mesh .= get_times(cache)
end
####################################################################################################
"""
Cache to remove allocations from PeriodicOrbitOCollProblem
"""
struct POCollCache{𝒯}
    gj::DiffCache{Matrix{𝒯},  Vector{𝒯}}
    gi::DiffCache{Matrix{𝒯},  Vector{𝒯}}
    ∂gj::DiffCache{Matrix{𝒯}, Vector{𝒯}}
    uj::DiffCache{Matrix{𝒯},  Vector{𝒯}}
    vj::DiffCache{Matrix{𝒯},  Vector{𝒯}}
    tmp::DiffCache{Vector{𝒯}, Vector{𝒯}}
    ∇phase::Vector{𝒯}
    In::Matrix{Bool}
end

"""
$(TYPEDSIGNATURES)

[Internal] In case `save_mem = true`, we do not allocate the identity matrix. Indeed think about `n = 100_000`.
"""
function POCollCache(𝒯::Type, Ntst::Int, n::Int, m::Int, save_mem = false)
    gj  = DiffCache(zeros(𝒯, n, m))
    gi  = DiffCache(zeros(𝒯, n, m))
    ∂gj = DiffCache(zeros(𝒯, n, m))
    uj  = DiffCache(zeros(𝒯, n, m + 1))
    vj  = DiffCache(zeros(𝒯, n, m + 1))
    tmp = DiffCache(zeros(𝒯, n))
    ∇phase = zeros(𝒯, n * (1 + m * Ntst))
    In = Array(LA.I(save_mem ? 1 : n))
    return POCollCache(gj, gi, ∂gj, uj, vj, tmp, ∇phase, In)
end
####################################################################################################
const _pocoll_jacobian_types = (AutoDiffDense(),
                                DenseAnalytical(),
                                FullSparse(),
                                DenseAnalyticalInplace(),
                                FullSparseInplace(),
                                AutoDiffMF())

"""
$(TYPEDEF)

This composite type implements an orthogonal collocation (at Gauss points) method of piecewise polynomials to locate periodic orbits. More details (maths, notations, linear systems) can be found [online](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/periodicOrbitCollocation/).

# Internal fields

$(TYPEDFIELDS)

# Methods

Here are some useful methods you can apply to `coll::PeriodicOrbitOCollProblem`:

- `length(coll)` gives the total number of unknowns.
- `size(coll)` returns the triplet `(N, m, Ntst)`.
- `getmesh(coll)` returns the mesh `0 = τ₁ < ... < τₙₜₛₜ₊₁ = 1`. This is useful because this mesh is bound to vary during automatic mesh adaptation.
- `get_mesh_coll(coll)` returns the (static) mesh `-1 = σ₁ < ... < σₘ₊₁ = 1`.
- `get_times(coll)` returns the vector of times (length `1 + m * Ntst`) at the which the collocation is applied.
- `generate_solution(coll, orbit, period)` generate a guess from a function `t -> orbit(t)` which approximates the periodic orbit.
- `POSolution(coll, x)` return a function interpolating the solution `x` using a piecewise polynomials function.

# Orbit guess
You can evaluate the residual of the functional (and other things) by calling `coll(orbitguess, p)` on an orbit guess `orbitguess`. Note that `orbitguess` must be of size 1 + N * (1 + m * Ntst) where N is the number of unknowns in the state space and `orbitguess[end]` is an estimate of the period `T` of the limit cycle.

Note that you can generate this guess from a function using `generate_solution` or `generate_ci_problem`.

# Jacobian

Specify the choice of the jacobian (and linear algorithm), `jacobian` must belong to $_pocoll_jacobian_types. This is used to
  select a way of inverting the jacobian dG of the functional G.

# Constructors
- `PeriodicOrbitOCollProblem(Ntst::Int, m::Int; kwargs)` creates an empty functional with `Ntst` and `m`.

# Functional
 A functional, hereby called `G`, encodes this problem. The following methods are available

- `residual(coll, orbitguess, p)` evaluates the functional G on `orbitguess`
- `residual!(coll, out, orbitguess, p)` evaluates the functional G on `orbitguess`
"""
@with_kw_noshow struct PeriodicOrbitOCollProblem{Tprob <: Union{Nothing, AbstractBifurcationProblem}, Tjac <: AbstractJacobianType, 𝒯, vectype, ∂vectype, Tmass} <: AbstractPODiffProblem
    "Bifurcation problem."
    prob_vf::Tprob = nothing

    "Used to set a section for the phase constraint equation."
    ϕ::vectype = nothing

    "Used in the section for the phase constraint equation."
    xπ::vectype = nothing

    "We store the derivative of ϕ, no need to recompute it each time."
    ∂ϕ::∂vectype = nothing

    "Dimension of the state space."
    N::Int = 0

    # whether the problem is nonautonomous
    isautonomous::Bool = true

    massmatrix::Tmass = nothing

    "Update the section every `update_section_every_step` step during continuation."
    update_section_every_step::UInt = 1

    "Describes the type of jacobian used in Newton/PALC/etc iterations."
    jacobian::Tjac = DenseAnalytical()

    "Cache for collocation. See docs of `MeshCollocationCache`."
    mesh_cache::MeshCollocationCache{𝒯} = nothing

    # cache for allocation free computations
    cache::POCollCache{𝒯} = nothing

    #################
    "Whether to use mesh adaptation."
    meshadapt::Bool = false

    "Verbose mesh adaptation information."
    verbose_mesh_adapt::Bool = false

    "Parameter for mesh adaptation, control new mesh step size. More precisely, we set max(hᵢ) / min(hᵢ) ≤ K if hᵢ denotes the time steps."
    K::Float64 = 100

    @assert jacobian in _pocoll_jacobian_types "This jacobian is not defined. Please chose another one in $_pocoll_jacobian_types."
end

# trivial constructor
function PeriodicOrbitOCollProblem(Ntst::Int, 
                                    m::Int,
                                    𝒯 = Float64;
                                    cache_In = false,
                                    kwargs...)
    N = get(kwargs, :N, 1)
    coll = PeriodicOrbitOCollProblem(; mesh_cache = MeshCollocationCache(Ntst, m, 𝒯),
                                    cache = POCollCache(𝒯, Ntst, N, m, cache_In),
                                    kwargs...)
    if ~isnothing(coll.ϕ)
        coll = set_collocation_size(coll, Ntst, m)
        updatesection!(coll, coll.ϕ, nothing)
    end
    coll
end

"""
$(TYPEDSIGNATURES)

This function changes the parameters `Ntst, m` for the collocation problem `pb` and return a new problem.
"""
function set_collocation_size(pb::PeriodicOrbitOCollProblem, Ntst, m)
    𝒯 = eltype(pb)
    pb2 = @set pb.mesh_cache = MeshCollocationCache(Ntst, m, eltype(pb))
    resize!(pb2.ϕ, length(pb2))
    resize!(pb2.xπ, length(pb2))
    @reset pb2.∂ϕ = zeros(eltype(pb), pb.N, Ntst * m)
    @reset pb2.cache.∇phase = zeros(𝒯, pb.N * (1 + m * Ntst))
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

@inline Base.eltype(::PeriodicOrbitOCollProblem{𝒯p, 𝒯j, 𝒯}) where {𝒯p, 𝒯j, 𝒯} = 𝒯

"""
    L, ∂L = get_Ls(pb::PeriodicOrbitOCollProblem)

Return the collocation matrices for evaluation and derivation.
"""
get_Ls(pb::PeriodicOrbitOCollProblem) = get_Ls(pb.mesh_cache)

@inline getparams(pb::PeriodicOrbitOCollProblem) = getparams(pb.prob_vf)
@inline getlens(pb::PeriodicOrbitOCollProblem) = getlens(pb.prob_vf)
@inline setparam(pb::PeriodicOrbitOCollProblem, p) = setparam(pb.prob_vf, p)

@inline getperiod(::PeriodicOrbitOCollProblem, x, par = nothing) = x[end]
getperiod(pb::PeriodicOrbitOCollProblem, x::POSolutionAndState, par = nothing) = getperiod(pb, x.sol, par)

# these functions extract the time slices components
get_time_slices(x::AbstractVector, N, degree, Ntst) = reshape(x, N, degree * Ntst + 1)
# array of size Ntst ⋅ (m+1) ⋅ n
get_time_slices(pb::PeriodicOrbitOCollProblem, x) = @views get_time_slices(x[begin:end-1], size(pb)...)
get_time_slices(pb::PeriodicOrbitOCollProblem, x::POSolutionAndState) = get_time_slices(pb, x.sol)
get_times(pb::PeriodicOrbitOCollProblem) = get_times(pb.mesh_cache)
get_gauss_nodes(pb::PeriodicOrbitOCollProblem) = get_gauss_nodes(pb.mesh_cache)

"""
Returns the vector of size Ntst + 1,  `0 = τ₁ < ... < τₙₜₛₜ₊₁ = 1`
"""
getmesh(coll::PeriodicOrbitOCollProblem) = getmesh(coll.mesh_cache)
get_mesh_coll(coll::PeriodicOrbitOCollProblem) = get_mesh_coll(coll.mesh_cache)
get_full_mesh(coll::PeriodicOrbitOCollProblem) = get_full_mesh(coll.mesh_cache)
get_max_time_step(coll::PeriodicOrbitOCollProblem) = get_max_time_step(coll.mesh_cache)
update_mesh!(coll::PeriodicOrbitOCollProblem, mesh) = update_mesh!(coll.mesh_cache, mesh)
@inline isinplace(coll::PeriodicOrbitOCollProblem) = isinplace(coll.prob_vf)
@inline is_symmetric(coll::PeriodicOrbitOCollProblem) = is_symmetric(coll.prob_vf)
@inline getdelta(coll::PeriodicOrbitOCollProblem) = getdelta(coll.prob_vf)
@inline get_state_dim(coll::PeriodicOrbitOCollProblem) = coll.N

function Base.show(io::IO, coll::PeriodicOrbitOCollProblem)
    N, m, Ntst = size(coll)
    println(io, "┌─ Collocation functional for periodic orbits")
    println(io, "├─ type               : Vector{", eltype(coll), "}")
    println(io, "├─ time slices (Ntst) : ", Ntst)
    println(io, "├─ degree      (m)    : ", m)
    println(io, "├─ dimension   (N)    : ", N)
    println(io, "├─ update section     : ", coll.update_section_every_step)
    println(io, "├─ jacobian           : ", coll.jacobian)
    println(io, "├─ mesh adaptation    : ", coll.meshadapt)
    if coll.meshadapt
        println(io, "├───── K              : ", coll.K)
    end
    println(io, "└─ # unknowns (without phase condition) : ", coll.N * (1 + m * Ntst))
end

"""
$(TYPEDSIGNATURES)

This function generates an initial guess for the solution of the problem `pb` based on the orbit `t -> orbit(t * period)` for t ∈ [0, 1] and the `period`. Used also in `generate_ci_problem`.
"""
function generate_solution(pb::PeriodicOrbitOCollProblem{𝒯p, 𝒯j, 𝒯}, orbit, period) where {𝒯p, 𝒯j, 𝒯}
    n, _m, Ntst = size(pb)
    ts = get_times(pb)
    Nt = length(ts)
    ci = zeros(𝒯, n, Nt)
    for (l, t) in pairs(ts)
        ci[:, l] .= orbit(t * period)
    end
    return vcat(vec(ci), period)
end

using SciMLBase: AbstractTimeseriesSolution
"""
$(TYPEDSIGNATURES)

Generate a guess and a periodic orbit problem from a solution.

## Arguments
- `pb` a `PeriodicOrbitOCollProblem`
- `bifprob` a bifurcation problem to provide the vector field
- `sol` basically an `ODEProblem` or a function `t -> sol(t)`
- `period` estimate of the period of the periodic orbit
- `cache_In = false` for caching in `MeshCollocationCache`
- `optimal_period = true` optimizes the period
- `use_adapted_mesh = false` adapt the mesh to minimize error. Should be used with mesh adaptation `meshadapt = true`.

## Output
- returns a `PeriodicOrbitOCollProblem` and an initial guess.
"""
function generate_ci_problem(pb::PeriodicOrbitOCollProblem,
                            bifprob::AbstractBifurcationProblem,
                            sol_ode::AbstractTimeseriesSolution,
                            period;
                            cache_In = false,
                            optimal_period::Bool = true,
                            use_adapted_mesh::Bool = false)
    if use_adapted_mesh || ~pb.meshadapt
        @warn "You initialize an adapted mesh but do not use mesh adaptation in the collocation problem!"
    end
    t0 = sol_ode.t[begin]
    u0 = sol_ode(t0)
    @assert u0 isa AbstractVector
    N = length(u0)
    𝒯 = eltype(u0)

    n, m, Ntst = size(pb)
    n_unknowns = N * (1 + m * Ntst)

    params = sol_ode.prob.p
    prob_vf = re_make(bifprob, params = params)

    coll = setproperties(pb;
                            N,
                            prob_vf,
                            ϕ  = zeros(𝒯, n_unknowns),
                            xπ = zeros(𝒯, n_unknowns),
                            ∂ϕ = zeros(𝒯, N, Ntst * m),
                            cache = POCollCache(eltype(pb), Ntst, N, m, cache_In))
    
    # find best period candidate
    if optimal_period
        _times = LinRange(period * 0.8, period * 1.2, 5 * Ntst)
        period = _times[argmin(norm(sol_ode(t + t0) - sol_ode(t0)) for t in _times)]
    end
    ci = copy(generate_solution(coll, t -> sol_ode(t0 + t), period))
    if use_adapted_mesh
        for _ = 1:10
        compute_error!(coll, ci, verbosity = pb.verbose_mesh_adapt)
        ci = generate_solution(coll, t -> sol_ode(t0 + t), period)
        end
    end
    updatesection!(coll, ci, nothing)
    return coll, ci
end

"""
$(TYPEDSIGNATURES)

[INTERNAL] Implementation of ∫_0^T < u(t), v(t) > dt.

```∫(pb, uc, vc, T = 1)```

# Arguments
- uj  n x (m + 1)
- vj  n x (m + 1)
"""
@views function ∫(pb::PeriodicOrbitOCollProblem, 
                    uc::AbstractMatrix, 
                    vc::AbstractMatrix,
                    period = one(eltype(uc)))
    𝒯y = promote_type(eltype(uc), eltype(vc)) 
    phase = zero(𝒯y)

    n, m, Ntst = size(pb)
    L, ∂L = get_Ls(pb.mesh_cache)
    ω = pb.mesh_cache.gauss_weight
    mesh = pb.mesh_cache.τs

    guj = zeros(𝒯y, n, m)
    uj  = zeros(𝒯y, n, m+1)

    gvj = zeros(𝒯y, n, m)
    vj  = zeros(𝒯y, n, m+1)

    rg = UnitRange(1, m+1)
    @inbounds for j in 1:Ntst
        LA.mul!(guj, uc[:, rg], L)
        LA.mul!(gvj, vc[:, rg], L)
        @inbounds for l in 1:m
            phase += LA.dot(guj[:, l], gvj[:, l]) * ω[l] * (mesh[j+1] - mesh[j]) / 2
        end
        rg = rg .+ m
    end
    return phase * period
end

function ∫(pb::PeriodicOrbitOCollProblem,
            u::AbstractVector,
            v::AbstractVector,
            period = one(eltype(uc)))
    uc = get_time_slices(pb, u)
    vc = get_time_slices(pb, v)
    return ∫(pb, uc, vc, period)
end

"""
$(TYPEDSIGNATURES)

[INTERNAL] Implementation of phase condition ∫_0^T < u(t), ∂ϕ(t) > dt. Note that it works for non uniform mesh.

# Arguments
- `Ls = (L, ∂L)` from `get_Ls`
- uj   n x (m + 1)
- guj  n x m
"""
function phase_condition(pb::PeriodicOrbitOCollProblem,
                        uc,
                        Ls,
                        period)
    𝒯 = eltype(uc)
    n, m, Ntst = size(pb)

    puj = get_tmp(pb.cache.gj, uc) # zeros(𝒯, n, m)
    uj  = get_tmp(pb.cache.uj, uc)  #zeros(𝒯, n, m+1)

    # vc = get_time_slices(pb.ϕ, size(pb)...)
    pvj = get_tmp(pb.cache.∂gj, uc) #zeros(𝒯, n, m)
    vj  = get_tmp(pb.cache.vj, uc)  #zeros(𝒯, n, m+1)

    _phase_condition(pb,
                    uc,
                    Ls,
                    (puj, uj, pvj, vj),
                    period)
end

# we do not check the indexing of uc, puj, ...
# these matrices were passed by the previous function
@views function _phase_condition(pb::PeriodicOrbitOCollProblem,
                                    uc,
                                    (L, ∂L),
                                    (pu, _, pϕ, _),
                                    period)
    𝒯 = eltype(uc)
    phase = zero(𝒯)
    n, m, Ntst = size(pb)
    ω = pb.mesh_cache.gauss_weight
    ϕc = get_time_slices(pb.ϕ, size(pb)...)
    rg = axes(uc, 2)[UnitRange(1, m+1)]

    @inbounds for j in 1:Ntst
        LA.mul!(pu, uc[:, rg], L) # pu : n x m
        LA.mul!(pϕ, ϕc[:, rg], ∂L)
        @inbounds for l in Base.OneTo(m)
            phase += LA.dot(pu[:, l], pϕ[:, l]) * ω[l]
        end
        rg = rg .+ m
    end
    return phase / period
end

function _POO_coll_scheme!(coll::PeriodicOrbitOCollProblem, dest, ∂u, u, par, h, tmp)
    residual!(coll.prob_vf, tmp, u, par)
    @. dest = ∂u - h * tmp
end

# functional for collocation problem
@views function po_residual_bare!(pb::PeriodicOrbitOCollProblem,
                                    out::AbstractMatrix, 
                                    u::AbstractMatrix{𝒯}, 
                                    period, 
                                    (L, ∂L), 
                                    pars;
                                    compute_phase::Val{CP} = Val(true)) where {𝒯, CP}
    # out is of size (n, m⋅Ntst + 1)
    n, ntimes = size(u)
    m = pb.mesh_cache.degree
    Ntst = pb.mesh_cache.Ntst
    ω = pb.mesh_cache.gauss_weight
    # we want slices at fixed times, hence pj[:, j] is the fastest
    # temporaries to reduce allocations
    pj  = get_tmp(pb.cache.gj, u)  #zeros(𝒯, n, m)
    ∂pj = get_tmp(pb.cache.∂gj, u) #zeros(𝒯, n, m)
    tmp = get_tmp(pb.cache.tmp, u)
    mesh = getmesh(pb)
    # range for locating time slices
    phase = zero(𝒯)
    rg = axes(out, 2)[UnitRange(1, m+1)] #Base.OneTo(m+1) 
    @inbounds for j in 1:Ntst
        dt = (mesh[j+1] - mesh[j]) / 2
        LA.mul!( pj, u[:, rg], L)  # size (n, m)
        LA.mul!(∂pj, u[:, rg], ∂L) # size (n, m)
        # compute the collocation residual
        for l in Base.OneTo(m)
            _POO_coll_scheme!(pb, out[:, rg[l]], ∂pj[:, l], pj[:, l], pars, period * dt, tmp)
        end
        if CP === true # statically knownm, should be removed by compiler
            @inbounds for l in Base.OneTo(m)
                phase += LA.dot(pj[:, l], pb.∂ϕ[:, (j-1)*m + l]) * ω[l]
            end
        end
        # carefull here https://discourse.julialang.org/t/is-this-a-bug-scalar-ranges-with-the-parser/70670/4"
        rg = rg .+ m
    end
    return phase / period
end

function po_residual!(pb::PeriodicOrbitOCollProblem, 
                                out::AbstractMatrix, 
                                u::AbstractMatrix, 
                                period, 
                                (L, ∂L), 
                                pars)
    phase = po_residual_bare!(pb, out, u, period, (L, ∂L), pars)
    # add the periodicity condition
    @views @. out[:, end] = u[:, end] - u[:, 1]
    return phase
end

function residual(prob::PeriodicOrbitOCollProblem, u::AbstractVector, pars)
    out = zero(u)
    residual!(prob, out, u, pars)
    out
end

function residual!(prob::PeriodicOrbitOCollProblem, result, u::AbstractVector, pars)
    uc = get_time_slices(prob, u)
    T = getperiod(prob, u, nothing)
    resultc = get_time_slices(prob, result)
    Ls = get_Ls(prob.mesh_cache)
    # add the phase condition ∫_0^T < u(t), ∂ϕ(t) > dt / T
    result[end] = po_residual!(prob, resultc, uc, T, Ls, pars)
    return result
end

"""
$(TYPEDSIGNATURES)

Compute the identity matrix associated with the collocation problem.
"""
function I(coll::PeriodicOrbitOCollProblem, u, par)
    T = getperiod(coll, u, par)
    N = get_state_dim(coll)
    Icoll = analytical_jacobian(coll, u, par; ρD = 0, ρF = 0, ρI = -1/T)
    Icoll[:, end] .= 0
    Icoll[end, :] .= 0
    Icoll[end-N:end-1, 1:N] .= 0
    Icoll[end-N:end-1, end-N:end-1] .= 0
    Icoll
end

"""
$(TYPEDSIGNATURES)

Compute the jacobian of the problem defining the periodic orbits by orthogonal collocation using an analytical formula. More precisely, it discretises

ρD * D - T * (ρF * F + ρI * I)

"""
@views function analytical_jacobian!(J,
                                    coll::PeriodicOrbitOCollProblem,
                                    u::AbstractVector{𝒯},
                                    pars; 
                                    _transpose::Val{TransposeBool} = Val(false),
                                    ρD = one(𝒯),
                                    ρF = one(𝒯),
                                    ρI = zero(𝒯)) where {𝒯, TransposeBool}
    n, m, Ntst = size(coll)
    nJ = length(coll) + 1
    L, ∂L = get_Ls(coll.mesh_cache) # L is of size (m+1, m)
    mesh = getmesh(coll)
    ω = coll.mesh_cache.gauss_weight
    period = getperiod(coll, u, nothing)
    phase = zero(𝒯)
    uc = get_time_slices(coll, u)
    pj = get_tmp(coll.cache.gi, u) # zeros(𝒯, n, m)
    In = coll.cache.In # this helps greatly the for loop for J0 below
    J0 = zeros(𝒯, n, n)

    # vector field
    VF = coll.prob_vf

    # put boundary condition
    J[nJ-n:nJ-1, nJ-n:nJ-1] .= In
    J[nJ-n:nJ-1, 1:n] .= (-1) .* In

    # loop over the mesh intervals
    rg = UnitRange(1, m+1)
    rgNx = UnitRange(1, n)
    rgNy = UnitRange(1, n)

    for j in 1:Ntst
        dt = (mesh[j+1] - mesh[j]) / 2
        α = period * dt
        LA.mul!(pj, uc[:, rg], L) # pj ≈ (L * uj')'
        # put the jacobian of the vector field
        for l in 1:m
            _rgX = rgNx .+ (l-1)*n
            if TransposeBool == false
                jacobian!(VF, J0, pj[:, l], pars)
            else
                J0 .= LA.transpose(jacobian(VF, pj[:, l], pars))
            end

            for l2 in 1:m+1
                J[_rgX, rgNy .+ (l2-1)*n ] .= @. (-α * L[l2, l] * ρF) * J0 +
                                                (ρD * ∂L[l2, l] - α * L[l2, l] * ρI) * In
            end
            # add derivative w.r.t. the period
            residual!(VF, J[_rgX, nJ], pj[:, l], pars)
            J[_rgX, nJ] .*= (-dt)

            phase += LA.dot(pj[:, l], coll.∂ϕ[:, (j-1)*m + l]) * ω[l]
        end
        rg = rg .+ m
        rgNx = rgNx .+ (m * n)
        rgNy = rgNy .+ (m * n)
    end

    J[end, begin:end-1] .= coll.cache.∇phase ./ period
    J[nJ, nJ] = -phase / period^2
    return J
end

function analytical_jacobian(coll::PeriodicOrbitOCollProblem, 
                            u::AbstractArray, 
                            pars; 
                            𝒯 = eltype(u), 
                            k...)
    if coll.jacobian isa AbstractJacobianSparseMatrix
        return analytical_jacobian_sparse( 
                        coll, 
                        u, 
                        pars;
                        𝒯, 
                        k...)
    else
        return analytical_jacobian!(zeros(𝒯, length(coll)+1, length(coll)+1), 
                        coll, 
                        u, 
                        pars; 
                        k...)
    end
end

function analytical_jacobian_sparse(coll::PeriodicOrbitOCollProblem,
                                    u::AbstractVector,
                                    pars; 
                                    k...)
    jacBlock = jacobian_poocoll_block(coll, u, pars; array_zeros = spzeros, k...)
    block_to_sparse(jacBlock)
end

function jacobian_poocoll_block(coll::PeriodicOrbitOCollProblem,
                                u::AbstractVector,
                                pars;
                                𝒯 = eltype(u),
                                array_zeros = zeros,
                                kwargs...) 
    n, m, Ntst = size(coll)
    blocks = n * ones(Int64, 1 + m * Ntst + 1); blocks[end] = 1
    n_blocks = length(blocks)
    J = BlockArray(array_zeros(𝒯, length(u), length(u)), blocks,  blocks)
    jacobian_poocoll_block!(J, coll, u, pars; kwargs...)
    return J
end

@views function jacobian_poocoll_block!(J::BlockArray,
                                coll::PeriodicOrbitOCollProblem,
                                u::AbstractVector{𝒯},
                                pars;
                                _transpose::Val{TransposeBool} = Val(false),
                                ρD = one(𝒯),
                                ρF = one(𝒯),
                                ρI = zero(𝒯)) where {𝒯, TransposeBool}
    n, m, Ntst = size(coll)
    n_blocks = size(J.blocks, 1)
    L, ∂L = get_Ls(coll.mesh_cache) # L is of size (m+1, m)
    ω = coll.mesh_cache.gauss_weight
    mesh = getmesh(coll)
    period = getperiod(coll, u, nothing)
    uc = get_time_slices(coll, u)
    pj = get_tmp(coll.cache.gi, u)   # zeros(𝒯, n, m)
    phase = zero(𝒯)

    # vector field
    VF = coll.prob_vf

    J0 = jacobian(VF, u[1:n], pars) # this works for sparse
    In = LA.I(n)

    # put boundary condition
    view(J, Block(1 + m * Ntst, 1 + m * Ntst)) .= In
    view(J, Block(1 + m * Ntst, 1)) .= (-1) .* In

    # loop over the mesh intervals
    rg = UnitRange(1, m+1)
    rgNx = UnitRange(1, n)
    iₚ = 0
    i∇ = 1

    for j in 1:Ntst
        dt = (mesh[j+1] - mesh[j]) / 2
        α = period * dt
        LA.mul!(pj, uc[:, rg], L) # pj ≈ (L * uj')'
        # put the jacobian of the vector field
        for l in 1:m
            if TransposeBool == false
                jacobian!(VF, J0, pj[:, l], pars)
            else
                J0 .= LA.transpose(jacobian(VF, pj[:, l], pars))
            end

            for l2 in 1:m+1
                view(J, Block(l + (j-1)*m, l2 + (j-1)*m)) .= @. (-α * L[l2, l] * ρF) * J0 +
                                                (ρD * ∂L[l2, l] - α * L[l2, l] * ρI) * In
            end
            # add derivative w.r.t. the period
            residual!(VF, view(J, Block(l + (j-1)*m, n_blocks)), pj[:, l], pars)
            view(J, Block(l + (j-1)*m, n_blocks)) .*= (-dt)

            phase += LA.dot(pj[:, l], coll.∂ϕ[:, iₚ + l]) * ω[l]
            view(J, Block(n_blocks, i∇)) .= coll.cache.∇phase[rgNx]' ./ period
            i∇ += 1; rgNx = rgNx .+ n
        end
        iₚ += m
        rg = rg .+ m
    end
    view(J, Block(n_blocks, i∇)) .= coll.cache.∇phase[rgNx]' ./ period # last bit
    J.blocks[end, end][1] = -phase / period^2
    return J
end

@views function jacobian_poocoll_sparse_indx!(coll::PeriodicOrbitOCollProblem,
                                        J::AbstractSparseMatrix,
                                        u::AbstractVector{𝒯},
                                        pars,
                                        indx; 
                                        _transpose::Val{TransposeBool} = Val(false),
                                        ρD = one(𝒯),
                                        ρF = one(𝒯),
                                        ρI = zero(𝒯),
                                        δ = convert(𝒯, 1e-9), 
                                        updateborder = true) where {𝒯, TransposeBool}
    n, m, Ntst = size(coll)
    @assert size(indx, 1) == 1 + m * Ntst + 1

    L, ∂L = get_Ls(coll.mesh_cache) # L is of size (m+1, m)
    ω = coll.mesh_cache.gauss_weight
    mesh = getmesh(coll)
    period = getperiod(coll, u, nothing)
    phase = zero(𝒯)
    uc = get_time_slices(coll, u)
    pj = zeros(𝒯, n, m)
    In = sparse(LA.I(n))
    J0 = jacobian(coll.prob_vf, uc[1:n], pars)
    tmpJ = copy(J0 + In)
    @assert J0 isa AbstractSparseMatrix

    # vector field
    VF = coll.prob_vf

    # put boundary condition
    J.nzval[indx[1 + m * Ntst, 1 + m * Ntst]] = In.nzval
    J.nzval[indx[1 + m * Ntst, 1]] = -In.nzval

    # loop over the mesh intervals
    rg = UnitRange(1, m+1)
    rgNx = UnitRange(1, n)
    rgNy = UnitRange(1, n)

    for j in 1:Ntst
        LA.mul!(pj, uc[:, rg], L) # pj ≈ (L * uj')'
        dt = (mesh[j+1]-mesh[j]) / 2
        α = period * dt
        # put the jacobian of the vector field
        for l in 1:m
            if ~TransposeBool
                @inbounds jacobian!(VF, J0, pj[:, l], pars)
            else
                J0 .= LA.transpose(jacobian(coll.prob_vf, pj[:,l], pars))
            end

            for l2 in 1:m+1
                tmpJ .= (-α * L[l2, l]) .* (ρF .* J0 + ρI * LA.I) .+ ρD * (∂L[l2, l] .* In)
                J.nzval[indx[ l + (j-1) * m, l2 + (j-1)*m] ] .= (tmpJ).nzval
            end
            # add derivative w.r.t. the period
            J[rgNx .+ (l-1)*n, end] .= residual(coll.prob_vf, pj[:,l], pars) .* (-dt)
            phase += LA.dot(pj[:, l], coll.∂ϕ[:, (j-1)*m + l]) * ω[l]
        end
        rg = rg .+ m
        rgNx = rgNx .+ (m * n)
    end

    J[end, begin:end-1] .= coll.cache.∇phase ./ period
    J[end, end] = -phase / period^2
    return J
end

"""
$(TYPEDSIGNATURES)

Compute the full periodic orbit associated to `x`. Mainly for plotting purposes.
"""
@views function get_periodic_orbit(prob::PeriodicOrbitOCollProblem, u, p)
    T = getperiod(prob, u, p)
    ts = get_times(prob)
    uc = get_time_slices(prob, u)
    return SolPeriodicOrbit(t = ts .* T, u = uc)
end

# same function as above but for coping with mesh adaptation
@views function get_periodic_orbit(coll::PeriodicOrbitOCollProblem, 
                x::Tx, 
                p) where { Tx <: POSolutionAndState}
    mesh = x.mesh
    u = x.sol
    T = getperiod(coll, u, p)
    uc = get_time_slices(coll, u)
    return SolPeriodicOrbit(t = mesh .* T, u = uc)
end

"""
$(TYPEDSIGNATURES)

Function needed for automatic branch switching from a Hopf bifurcation point.
"""
function re_make(coll::PeriodicOrbitOCollProblem,
                 prob_vf,
                 hopfpt,
                 ζr::AbstractVector,
                 orbitguess_a,
                 period; 
                 orbit = identity,
                 k...)
    N = length(ζr)
    _, m, Ntst = size(coll)
    n_unknows = N * (1 + m * Ntst) # careful, we need to do that here instead of length(coll)

    # update the problem
    new_coll = setproperties(coll; N, prob_vf, 
                ϕ = zeros(n_unknows), 
                xπ = zeros(n_unknows), 
                ∂ϕ = zeros(N, Ntst * m),
                cache = POCollCache(eltype(coll), Ntst, N, m)
                )

    ϕ0 = generate_solution(new_coll, t -> orbit(2pi * t / period + pi), period)
    updatesection!(new_coll, ϕ0, nothing)

    # append period at the end of the initial guess
    orbitguess = generate_solution(new_coll, t -> orbit(2pi*t/period), period)

    return new_coll, orbitguess
end

##########################
# problem wrappers
residual(prob::WrapPOColl, x, p) = residual(prob.prob, x, p)

@inline is_symmetric(prob::WrapPOColl) = is_symmetric(prob.prob)
@inline getdelta(pb::WrapPOColl) = getdelta(pb.prob)
@inline has_adjoint(::WrapPOColl) = false # it is in problems.jl

# for recording the solution in a branch
function save_solution(wrap::WrapPOColl, x, pars)
    if wrap.prob.meshadapt
        return POSolutionAndState(copy(get_times(wrap.prob)), 
                x, 
                copy(getmesh(wrap.prob.mesh_cache)),
                copy(wrap.prob.ϕ),
                )
    else
        return x
    end
end

####
function _generate_jacobian(coll::PeriodicOrbitOCollProblem, J::FullSparseInplace, orbitguess, pars; k...)
    _J = analytical_jacobian_sparse(coll, orbitguess, pars)
    indx = get_blocks(coll, _J)
    return (FullSparseInplace(), _J, indx)
end

function _generate_jacobian(coll::PeriodicOrbitOCollProblem, J::DenseAnalyticalInplace, orbitguess, pars; Jcoll_matrix = nothing, k...)
    _Jcoll_matrix = isnothing(Jcoll_matrix) ? analytical_jacobian(coll, orbitguess, pars) : Jcoll_matrix
    return (DenseAnalyticalInplace(), _Jcoll_matrix)
end
####

function jacobian(coll::PeriodicOrbitOCollProblem, J::Tuple{DenseAnalyticalInplace, Tj}, x, p) where {Tj}
    _Jcoll_matrix = J[2]
    analytical_jacobian!(_Jcoll_matrix, coll, x, p)
    return _Jcoll_matrix
end

function jacobian(coll::PeriodicOrbitOCollProblem, ::DenseAnalytical, x, p)
    return analytical_jacobian(coll, x, p)
end

function jacobian(coll::PeriodicOrbitOCollProblem, ::FullSparse, x, p)
    return analytical_jacobian_sparse(coll, x, p)
end

function jacobian(coll::PeriodicOrbitOCollProblem, J::Tuple{FullSparseInplace, Tj, Tind}, x, p) where {Tj, Tind}
    _J = J[2]
    indx = J[3]
    jacobian_poocoll_sparse_indx!(coll, _J, x, p, indx)
    return _J
end
####################################################################################################
const DocStringJacobianPOColl = """
- `jacobian` Specify the choice of the linear algorithm, which must belong to `(AutoDiffDense(), )`. This is used to select a way of inverting the jacobian dG
    - For `AutoDiffDense()`. The jacobian is formed as a dense Matrix. You can use a direct solver or an iterative one using `options`. The jacobian is formed inplace.
    - For `DenseAnalytical()` Same as for `AutoDiffDense` but the jacobian is formed using a mix of AD and analytical formula.
"""

function _newton_pocoll(probPO::PeriodicOrbitOCollProblem,
                        orbitguess,
                        options::NewtonPar;
                        defOp::Union{Nothing, DeflationOperator} = nothing,
                        kwargs...)
    jac = _generate_jacobian(probPO, probPO.jacobian, orbitguess, getparams(probPO))

    if options.linsolver isa COPLS
        @reset options.linsolver = COPLS(probPO)
    end

    prob = WrapPOColl(probPO, jac, orbitguess, getparams(probPO), getlens(probPO), nothing, nothing)

    if isnothing(defOp)
        return solve(prob, Newton(), options; kwargs...)
    else
        return solve(prob, defOp, options; kwargs...)
    end
end

"""
$(TYPEDSIGNATURES)

This is the Newton solver for computing a periodic orbit using orthogonal collocation method.
Note that the linear solver has to be apropriately set up in `options`.

# Arguments

Similar to [`newton`](@ref) except that `prob` is a [`PeriodicOrbitOCollProblem`](@ref).

- `prob` a problem of type `<: PeriodicOrbitOCollProblem` encoding the shooting functional G.
- `orbitguess` a guess for the periodic orbit.
- `options` same as for the regular [`newton`](@ref) method.

# Optional argument
$DocStringJacobianPOColl
"""
newton(probPO::PeriodicOrbitOCollProblem,
            orbitguess,
            options::NewtonPar;
            kwargs...) = _newton_pocoll(probPO, orbitguess, options; defOp = nothing, kwargs...)

"""
    $(TYPEDSIGNATURES)

This function is similar to `newton(probPO, orbitguess, options, jacobianPO; kwargs...)` except that it uses deflation in order to find periodic orbits different from the ones stored in `defOp`. We refer to the mentioned method for a full description of the arguments. The current method can be used in the vicinity of a Hopf bifurcation to prevent the Newton-Krylov algorithm from converging to the equilibrium point.
"""
function newton(probPO::PeriodicOrbitOCollProblem,
                orbitguess,
                defOp::DeflationOperator,
                options::NewtonPar;
                kwargs...)
    _newton_pocoll(probPO, orbitguess, options; defOp, kwargs...)
end

"""
$(TYPEDSIGNATURES)

This is the continuation method for computing a periodic orbit using an orthogonal collocation method.

# Arguments

Similar to [`continuation`](@ref) except that `prob` is a [`PeriodicOrbitOCollProblem`](@ref). By default, it prints the period of the periodic orbit.

# Keywords arguments
- `eigsolver` specify an eigen solver for the computation of the Floquet exponents, defaults to `FloquetQaD`
"""
function continuation(coll::PeriodicOrbitOCollProblem,
                    orbitguess,
                    alg::AbstractContinuationAlgorithm,
                    _contParams::ContinuationPar,
                    linear_algo::AbstractBorderedLinearSolver;
                    δ = convert(eltype(orbitguess), 1e-8),
                    eigsolver = FloquetColl(),
                    record_from_solution = nothing,
                    plot_solution = nothing,
                    kwargs...)
    options = _contParams.newton_options

    if linear_algo isa COPBLS
        Nbls = length(coll) + 2 # +1 for phase condition and +1 for PALC
        _Jcoll = analytical_jacobian(coll, orbitguess, getparams(coll))
        cache = COPCACHE(coll, Val(1))
        linear_algo = COPBLS(;
                        cache,
                        solver = nothing,
                        J = similar(_Jcoll, Nbls, Nbls)
                        )
        jacPO = _generate_jacobian(coll, coll.jacobian, orbitguess, getparams(coll); 
                        δ,
                        Jcoll_matrix = @view linear_algo.J[begin:end-1, begin:end-1]
                        )

        linear_algo.J .= 0
        lspo = COPLS(COPCACHE(coll, Val(0)))
        @reset options.linsolver = lspo
        if eigsolver isa FloquetColl
            @reset eigsolver.cache = COPCACHE(coll, Val(0))
        end
    else
        # linear_algo = @set linear_algo.solver = FloquetWrapperLS(linear_algo.solver)
        jacPO = _generate_jacobian(coll, coll.jacobian, orbitguess, getparams(coll); δ)
    end
    contParams = _contParams

    # we have to change the Bordered linear solver to cope with our type FloquetWrapper
    # alg = update(alg, contParams, linear_algo)

    if compute_eigenelements(contParams)
        contParams = @set contParams.newton_options.eigsolver = eigsolver
    end

    # change the user provided finalise function by passing prob in its parameters
    _finsol = modify_po_finalise(coll, kwargs, coll.update_section_every_step)
    # this is to remove this part from the arguments passed to continuation
    _kwargs = (record_from_solution = record_from_solution, plot_solution = plot_solution)
    _recordsol = modify_po_record(coll, getparams(coll.prob_vf), getlens(coll.prob_vf); _kwargs...)
    _plotsol = modify_po_plot(coll, getparams(coll.prob_vf), getlens(coll.prob_vf); _kwargs...)

    wrap_coll = WrapPOColl(coll, jacPO, orbitguess, getparams(coll), getlens(coll), _plotsol, _recordsol)

    br = continuation(wrap_coll, alg,
                      contParams;
                      kwargs...,
                      kind = PeriodicOrbitCont(),
                      finalise_solution = _finsol,
                      linear_algo
                      )
    return br
end

# this function updates the section during the continuation run
@views function updatesection!(coll::PeriodicOrbitOCollProblem, 
                                x::AbstractVector, 
                                par) # (2 allocations: 96 bytes)
    @debug "[collocation] update section"
    # update the reference point
    coll.xπ .= 0
    # update the "normals"
    coll.ϕ .= x[eachindex(coll.ϕ)]
    # update ∂ϕ
    ϕ = coll.ϕ
    L, ∂L = get_Ls(coll.mesh_cache)
    n, m, Ntst = size(coll)
    ϕc = get_time_slices(ϕ, n, m, Ntst) # (2 allocations: 96 bytes)
    pϕ = get_tmp(coll.cache.∂gj, ϕc) # zeros(𝒯, n, m)
    rg = axes(ϕc, 2)[UnitRange(1, m+1)] # (j-1)*m
    @inbounds for j in 1:Ntst
        LA.mul!(pϕ, ϕc[:, rg], ∂L)
        coll.∂ϕ[:, (j-1)*m .+ (1:m)] .= pϕ
        rg = rg .+ m
    end

    # update ∇phase
    ω = coll.mesh_cache.gauss_weight
    rg = 1:n
    coll.cache.∇phase .= 0
    @inbounds for j = 1:Ntst
        for k₁ = 1:m+1
            for l = 1:m
                coll.cache.∇phase[rg] .+= (L[k₁, l] * ω[l]) .* coll.∂ϕ[:, (j-1)*m + l]
            end
            if k₁ < m + 1
                rg = rg .+ n
            end
        end
    end
    @debug "[collocation] update section: done"
    return true
end
####################################################################################################
# mesh adaptation method
(sol::POSolution{ <: PeriodicOrbitOCollProblem})(t0) = __interpolate_posolution(sol.pb, t0, sol.x)

@views function __interpolate_posolution(coll::PeriodicOrbitOCollProblem, t0, x)
    n, m, Ntst = size(coll)
    xc = get_time_slices(coll, x)
    period = getperiod(coll, x, nothing)
    t = mod(t0, period) / period
    mesh = getmesh(coll)
    index_t = searchsortedfirst(mesh, t) - 1
    if index_t <= 0
        return x[1:n]
    elseif index_t > Ntst
        return xc[:, end]
    end
    @assert mesh[index_t] <= t <= mesh[index_t+1] "Please open an issue on the website of BifurcationKit.jl"
    σ = σj(t, mesh, index_t)
    # @assert -1 <= σ <= 1 "Strange value of $σ"
    σs = get_mesh_coll(coll)
    out = zeros(typeof(t), n)
    rg = (1:m+1) .+ (index_t - 1) * m
    for l in 1:m+1
        out .+= xc[:, rg[l]] .* lagrange(l, σ, σs)
    end
    out
end

"""
$(TYPEDSIGNATURES)

Perform mesh adaptation of the periodic orbit problem. Modify `pb` and `x` inplace if the adaptation is successfull.

See page 367 of [1] and also [2].

References:
[1] Ascher, Uri M., Robert M. M. Mattheij, and Robert D. Russell. Numerical Solution of Boundary Value Problems for Ordinary Differential Equations. Society for Industrial and Applied Mathematics, 1995. https://doi.org/10.1137/1.9781611971231.

[2] R. D. Russell and J. Christiansen, “Adaptive Mesh Selection Strategies for Solving Boundary Value Problems,” SIAM Journal on Numerical Analysis 15, no. 1 (February 1978): 59–80, https://doi.org/10.1137/0715004.
"""
function compute_error!(coll::PeriodicOrbitOCollProblem, x::AbstractVector{𝒯};
                        normE = norminf,
                        verbosity::Bool = false,
                        K = 𝒯(Inf),
                        par = nothing,
                        kw...) where 𝒯
    n, m, Ntst = size(coll) # recall that m = ncol
    period = getperiod(coll, x, nothing)
    # get solution, we copy x because it is overwritten at the end of this function
    sol = POSolution(deepcopy(coll), copy(x))
    # we need to estimate yᵐ⁺¹ where y is the true periodic orbit.
    # sol is the piecewise polynomial approximation of y.
    # However, sol is of degree m, hence ∂(sol, m+1) = 0
    # we thus estimate yᵐ⁺¹ using ∂(sol, m)
    dmsol = ∂(sol, Val(m))
    # we find the values of vm := ∂m(x) at the mid points
    τsT = getmesh(coll) .* period
    vm = [ dmsol( (τsT[i] + τsT[i+1]) / 2 ) for i = 1:Ntst ]
    ############
    # Approx. IA
    # this is the function s^{(k)} in the above paper [2] on page 63
    # we want to estimate sk = s^{(m+1)} which is 0 by definition, pol of degree m
    if isempty(findall(diff(τsT) .<= 0)) == false
        @error "[Mesh-adaptation]. The mesh is non monotonic!\nPlease report the error to the website of BifurcationKit.jl"
        return (success = false, newτsT = τsT, ϕ = τsT)
    end
    sk = zeros(𝒯, Ntst)
    sk[1] = 2normE(vm[1]) / (τsT[2] - τsT[1])
    for i in 2:Ntst-1
        sk[i] = normE(vm[i])   / (τsT[i+1] - τsT[i-1]) +
                normE(vm[i+1]) / (τsT[i+2] - τsT[i])
    end
    sk[Ntst] = 2normE(vm[end]) / (τsT[end] - τsT[end-2])
    ############
    # monitor function
    ϕ = sk.^(1/(m+1))
    # if the monitor function is too small, don't do anything
    if maximum(ϕ) < 1e-7
        return (;success = true, newmesh = nothing, ϕ)
    end
    ϕ = max.(ϕ, maximum(ϕ) / K)
    if length(ϕ) != Ntst
        error("Error. Please open an issue of the website of BifurcationKit.jl")
    end
    # compute θ = ∫ϕ but also all intermediate values
    # these intermediate values are useful because the integral is piecewise linear
    # and equipartition is analytical
    # there are ntst values for the integrals, one for (0, mesh[2]), (mesh[2], mesh[3])...
    θs = zeros(𝒯, Ntst); θs[1] = ϕ[1] * (τsT[2] - τsT[1])
    for i = 2:Ntst
        θs[i] = θs[i-1] + ϕ[i] * (τsT[i+1] - τsT[i])
    end
    pushfirst!(θs, zero(𝒯)) # θs = vcat(0, θs)
    θ = θs[end]
    ############
    # compute new mesh from equipartition
    newτsT = zero(τsT); newτsT[end] = 1
    c = θ / Ntst
    for i in 1:Ntst-1
        θeq = i * c
        # we have that θeq ∈ (θs[ind-1], θs[ind])
        ind = searchsortedfirst(θs, θeq)
        @assert 2 <= ind <= Ntst+1 "Error with 1 < $ind <= $(Ntst+1). Please open an issue on the website of BifurcationKit.jl"
        α = (θs[ind] - θs[ind-1]) / (τsT[ind] - τsT[ind-1])
        newτsT[i+1] = τsT[ind-1] + (θeq - θs[ind-1]) / α
        @assert newτsT[i+1] > newτsT[i] "Error. Please open an issue on the website of BifurcationKit.jl"
    end
    newmesh = newτsT ./ period
    newmesh[end] = 1

    if verbosity
        h = maximum(diff(newmesh))
        printstyled(color = :magenta, 
          "   ┌─ Mesh adaptation, new mesh hᵢ = time steps",
        "\n   ├─── min(hᵢ)       = ", minimum(diff(newmesh)),
        "\n   ├─── h = max(hᵢ)   = ", h,
        "\n   ├─── K = max(h/hᵢ) = ", maximum(h ./ diff(newmesh)),
        "\n   ├─── min(ϕ)        = ", minimum(ϕ),
        "\n   ├─── max(ϕ)        = ", maximum(ϕ),
        "\n   └─── θ             = ", θ,
        "\n")
    end
    ############
    # modify meshes
    update_mesh!(coll, newmesh)
    ############
    # update solution
    newsol = generate_solution(coll, sol, period)
    x .= newsol

    success = true
    return (;success, newτsT, ϕ)
end

# condensation of parameters in Ascher, Uri M., Robert M. M. Mattheij, and Robert D. Russell. Numerical Solution of Boundary Value Problems for Ordinary Differential Equations. Society for Industrial and Applied Mathematics, 1995. https://doi.org/10.1137/1.9781611971231.
####################################################################################################
"""
$(TYPEDSIGNATURES)

This function extracts the indices of the blocks composing the matrix J which is a M x M Block matrix where each block N x N has the same sparsity.
"""
function get_blocks(coll::PeriodicOrbitOCollProblem, Jac::SparseMatrixCSC)
    N, m, Ntst = size(coll)
    blocks = N * ones(Int64, 1 + m * Ntst + 1); blocks[end] = 1
    n_blocks = length(blocks)
    I, J, K = findnz(Jac)
    out = [Vector{Int}() for i in 1:n_blocks, j in 1:n_blocks];
    for k in eachindex(I)
        i, j = div(I[k]-1, N), div(J[k]-1, N)
        push!(out[1+i, 1+j], k)
    end
    out
end

