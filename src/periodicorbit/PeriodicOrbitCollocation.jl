using FastGaussQuadrature: gausslegendre

"""
    cache = MeshCollocationCache(Ntst::Int, m::Int, Ty = Float64)

Structure to hold the cache for the collocation method. More precisely, it starts from a partition of [0,1] based on the mesh points:

    0 = τ₁ < τ₂ < ... < τₙₜₛₜ₊₁ = 1

On each mesh interval [τⱼ, τⱼ₊₁] mapped to [-1,1], a Legendre polynomial of degree m is formed. 


$(TYPEDFIELDS)

# Constructor

    MeshCollocationCache(Ntst::Int, m::Int, Ty = Float64)

- `Ntst` number of time steps
- `m` degree of the collocation polynomials
- `Ty` type of the time variable
"""
struct MeshCollocationCache{𝒯}
    "Coarse mesh size"
    Ntst::Int
    "Collocation degree, usually called m"
    degree::Int
    "Lagrange matrix"
    lagrange_vals::Matrix{𝒯}
    "Lagrange matrix for derivative"
    lagrange_∂::Matrix{𝒯}
    "Gauss nodes"
    gauss_nodes::Vector{𝒯}
    "Gauss weights"
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

@inline Base.eltype(cache::MeshCollocationCache{𝒯}) where 𝒯 = 𝒯
@inline Base.size(cache::MeshCollocationCache) = (cache.degree, cache.Ntst)
@inline get_Ls(cache::MeshCollocationCache) = (cache.lagrange_vals, cache.lagrange_∂)
@inline getmesh(cache::MeshCollocationCache) = cache.τs
@inline get_mesh_coll(cache::MeshCollocationCache) = cache.σs
get_max_time_step(cache::MeshCollocationCache) = maximum(diff(getmesh(cache)))
_τj(σ, τⱼ₊₁, τⱼ) = τⱼ + (1 + σ)/2 * (τⱼ₊₁ - τⱼ) # for σ ∈ [-1,1], τj ∈ [τⱼ, τs[j+1]]
@inline τj(σ, τs, j) = _τj(σ, τs[j+1], τs[j])
# get the sigma corresponding to τ in the interval (τs[j], τs[j+1])
@inline σj(τ, τs, j) = (2*τ - τs[j] - τs[j + 1])/(τs[j + 1] - τs[j]) # for τ ∈ [τs[j], τs[j+1]], σj ∈ [-1, 1]

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
function compute_legendre_matrices(σs::AbstractVector{𝒯}) where 𝒯
    m = length(σs) - 1
    zs, ws = gausslegendre(m)
    L  = zeros(𝒯, m + 1, m)
    ∂L = zeros(𝒯, m + 1, m)
    for j in 1:m+1
        for i in 1:m
             L[j, i] =  lagrange(j, zs[i], σs)
            ∂L[j, i] = dlagrange(j, zs[i], σs)
        end
    end
    return (;L, ∂L, zg = zs, wg = ws)
end

"""
$(SIGNATURES)

Return all the times at which the problem is evaluated.
"""
@views function get_times(cache::MeshCollocationCache{𝒯}) where 𝒯
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
cache to remove allocations from PeriodicOrbitOCollProblem
"""
struct POCollCache{T}
    gj::DiffCache{Matrix{T}, Vector{T}}
    gi::DiffCache{Matrix{T}, Vector{T}}
    ∂gj::DiffCache{Matrix{T}, Vector{T}}
    uj::DiffCache{Matrix{T}, Vector{T}}
    vj::DiffCache{Matrix{T}, Vector{T}}
    tmp::DiffCache{Vector{T}, Vector{T}}
end

function POCollCache(𝒯::Type, n::Int, m::Int)
    gj  = DiffCache(zeros(𝒯, n, m))
    gi  = DiffCache(zeros(𝒯, n, m))
    ∂gj = DiffCache(zeros(𝒯, n, m))
    uj  = DiffCache(zeros(𝒯, n, m+1))
    vj  = DiffCache(zeros(𝒯, n, m+1))
    tmp = DiffCache(zeros(𝒯, n))
    return POCollCache(gj, gi, ∂gj, uj, vj, tmp)
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
- `update_section_every_step` updates the section every `update_section_every_step` step during continuation
- `jacobian = DenseAnalytical()` describes the type of jacobian used in Newton iterations. Can only be `AutoDiffDense(), DenseAnalytical(), FullSparse(), FullSparseInplace()`.
- `meshadapt::Bool = false` whether to use mesh adaptation
- `verbose_mesh_adapt::Bool = true` verbose mesh adaptation information
- `K::Float64 = 500` parameter for mesh adaptation, control new mesh step size. More precisely, we set max(hᵢ) / min(hᵢ) ≤ K if hᵢ denotes the time steps.

## Methods

Here are some useful methods you can apply to `pb`

- `length(pb)` gives the total number of unknowns
- `size(pb)` returns the triplet `(N, m, Ntst)`
- `getmesh(pb)` returns the mesh `0 = τ0 < ... < τNtst+1 = 1`. This is useful because this mesh is born to vary during automatic mesh adaptation
- `get_mesh_coll(pb)` returns the (static) mesh `0 = σ0 < ... < σm+1 = 1`
- `get_times(pb)` returns the vector of times (length `1 + m * Ntst`) at the which the collocation is applied.
- `generate_solution(pb, orbit, period)` generate a guess from a function `t -> orbit(t)` which approximates the periodic orbit.
- `POSolution(pb, x)` return a function interpolating the solution `x` using a piecewise polynomials function

# Orbit guess
You can evaluate the residual of the functional (and other things) by calling `pb(orbitguess, p)` on an orbit guess `orbitguess`. Note that `orbitguess` must be of size 1 + N * (1 + m * Ntst) where N is the number of unknowns in the state space and `orbitguess[end]` is an estimate of the period ``T`` of the limit cycle.

Note that you can generate this guess from a function using `generate_solution` or `generate_ci_problem`.

# Constructors
- `PeriodicOrbitOCollProblem(Ntst::Int, m::Int; kwargs)` creates an empty functional with `Ntst` and `m`.

# Functional
 A functional, hereby called `G`, encodes this problem. The following methods are available

- `residual(pb, orbitguess, p)` evaluates the functional G on `orbitguess`
- `residual!(pb, out, orbitguess, p)` evaluates the functional G on `orbitguess`
"""
@with_kw_noshow struct PeriodicOrbitOCollProblem{Tprob <: Union{Nothing, AbstractBifurcationProblem}, Tjac <: AbstractJacobianType, vectype, Tmass, Tmcache <: MeshCollocationCache, Tcache} <: AbstractPODiffProblem
    # Function F(x, par)
    prob_vf::Tprob = nothing

    # variables to define a Section for the phase constraint equation
    ϕ::vectype = nothing
    xπ::vectype = nothing

    # dimension of the problem in case of an AbstractVector
    N::Int = 0

    # whether the problem is nonautonomous
    isautonomous::Bool = true

    # mass matrix
    massmatrix::Tmass = nothing

    # update the section every step
    update_section_every_step::UInt = 1

    # variable to control the way the jacobian of the functional is computed
    jacobian::Tjac = DenseAnalytical()

    # collocation mesh cache
    mesh_cache::Tmcache = nothing

    # cache for allocation free computations
    cache::Tcache = nothing

    #################
    # mesh adaptation
    meshadapt::Bool = false

    # verbose mesh adaptation information
    verbose_mesh_adapt::Bool = false

    # parameter for mesh adaptation, control maximum mesh step size
    K::Float64 = 100
end

# trivial constructor
function PeriodicOrbitOCollProblem(Ntst::Int, 
                                    m::Int,
                                    𝒯 = Float64;
                                    kwargs...)
    # @assert iseven(Ntst) "Ntst must be even (otherwise issue with Floquet coefficients)"
    N = get(kwargs, :N, 1)
    PeriodicOrbitOCollProblem(; mesh_cache = MeshCollocationCache(Ntst, m, 𝒯),
                                    cache = POCollCache(𝒯, N, m),
                                    kwargs...)
end

"""
$(SIGNATURES)

This function change the parameters `Ntst, m` for the collocation problem `pb` and return a new problem.
"""
function set_collocation_size(pb::PeriodicOrbitOCollProblem, Ntst, m)
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
"""
    L, ∂L = get_Ls(pb)

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
get_time_slices(pb::PeriodicOrbitOCollProblem, x) = @views get_time_slices(x[1:end-1], size(pb)...)
get_time_slices(pb::PeriodicOrbitOCollProblem, x::POSolutionAndState) = get_time_slices(pb, x.sol)
get_times(pb::PeriodicOrbitOCollProblem) = get_times(pb.mesh_cache)
"""
Returns the vector of size m+1,  0 = τ₁ < τ₂ < ... < τₘ < τₘ₊₁ = 1
"""
getmesh(pb::PeriodicOrbitOCollProblem) = getmesh(pb.mesh_cache)
get_mesh_coll(pb::PeriodicOrbitOCollProblem) = get_mesh_coll(pb.mesh_cache)
get_max_time_step(pb::PeriodicOrbitOCollProblem) = get_max_time_step(pb.mesh_cache)
update_mesh!(pb::PeriodicOrbitOCollProblem, mesh) = update_mesh!(pb.mesh_cache, mesh)
@inline isinplace(pb::PeriodicOrbitOCollProblem) = isinplace(pb.prob_vf)
@inline is_symmetric(pb::PeriodicOrbitOCollProblem) = is_symmetric(pb.prob_vf)
@inline getdelta(pb::PeriodicOrbitOCollProblem) = getdelta(pb.prob_vf)

function Base.show(io::IO, pb::PeriodicOrbitOCollProblem)
    N, m, Ntst = size(pb)
    println(io, "┌─ Collocation functional for periodic orbits")
    println(io, "├─ type               : Vector{", eltype(pb), "}")
    println(io, "├─ time slices (Ntst) : ", Ntst)
    println(io, "├─ degree      (m)    : ", m)
    println(io, "├─ dimension   (N)    : ", pb.N)
    println(io, "├─ inplace            : ", isinplace(pb))
    println(io, "├─ update section     : ", pb.update_section_every_step)
    println(io, "├─ jacobian           : ", pb.jacobian)
    println(io, "├─ mesh adaptation    : ", pb.meshadapt)
    if pb.meshadapt
        println(io, "├───── K              : ", pb.K)
    end
    println(io, "└─ # unknowns (without phase condition) : ", pb.N * (1 + m * Ntst))
end

function get_matrix_phase_condition(coll::PeriodicOrbitOCollProblem)
    n, m, Ntst = size(coll)
    L, ∂L = get_Ls(coll.mesh_cache)
    ω = coll.mesh_cache.gauss_weight
    Ω = zeros(eltype(coll), m+1, m+1)
    for k₁ = 1:m+1
        for k₂ = 1:m+1
            for l = 1:m
                Ω[k₁, k₂] += ω[l] * L[k₁, l] * ∂L[k₂, l]
            end
        end
    end
    Ω
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
- `sol` basically an `ODEProblem` or a function `t -> sol(t)`
- `period` estimate of the period of the periodic orbit

## Output
- returns a `PeriodicOrbitOCollProblem` and an initial guess.
"""
function generate_ci_problem(pb::PeriodicOrbitOCollProblem,
                            bifprob::AbstractBifurcationProblem,
                            sol_ode::AbstractTimeseriesSolution,
                            period;
                            optimal_period::Bool = true)
    t0 = sol_ode.t[begin]
    u0 = sol_ode(t0)
    @assert u0 isa AbstractVector
    N = length(u0)

    n, m, Ntst = size(pb)
    n_unknows = N * (1 + m * Ntst)

    par = sol_ode.prob.p
    prob_vf = re_make(bifprob, params = par)

    pbcoll = setproperties(pb,
                            N = N,
                            prob_vf = prob_vf,
                            ϕ = zeros(n_unknows),
                            xπ = zeros(n_unknows),
                            cache = POCollCache(eltype(pb), N, m))
    
    # find best period candidate
    if optimal_period
        _times = LinRange(period * 0.8, period * 1.2, 5Ntst)
        period = _times[argmin(norm(sol_ode(t + t0) - sol_ode(t0)) for t in _times)]
    end

    ci = generate_solution(pbcoll, t -> sol_ode(t0 + t), period)
    pbcoll.ϕ .= @view ci[begin:end-1]

    return pbcoll, ci
end

"""
$(SIGNATURES)

[INTERNAL] Implementation of ∫_0^T < u(t), v(t) > dt.

```∫(pb, uc, vc, T = 1)```

# Arguments
- uj  n x (m + 1)
- vj  n x (m + 1)
"""
@views function ∫(pb::PeriodicOrbitOCollProblem, 
                    uc::AbstractMatrix, 
                    vc::AbstractMatrix,
                    T = one(eltype(uc)))
    Ty = promote_type(eltype(uc), eltype(vc)) 
    phase = zero(Ty)

    n, m, Ntst = size(pb)
    L, ∂L = get_Ls(pb.mesh_cache)
    ω = pb.mesh_cache.gauss_weight
    mesh = pb.mesh_cache.τs

    guj = zeros(Ty, n, m)
    uj  = zeros(Ty, n, m+1)

    gvj = zeros(Ty, n, m)
    vj  = zeros(Ty, n, m+1)

    rg = UnitRange(1, m+1)
    @inbounds for j in 1:Ntst
        mul!(guj, uc[:, rg], L)
        mul!(gvj, vc[:, rg], L)
        @inbounds for l in 1:m
            phase += dot(guj[:, l], gvj[:, l]) * ω[l] * (mesh[j+1] - mesh[j]) / 2
        end
        rg = rg .+ m
    end
    return phase * T
end

function ∫(pb::PeriodicOrbitOCollProblem,
            u::AbstractVector,
            v::AbstractVector,
            T = one(eltype(uc)))
    uc = get_time_slices(pb, u)
    vc = get_time_slices(pb, v)
    ∫(pb, uc, vc, T)
end

"""
$(SIGNATURES)

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
        mul!(pu, uc[:, rg], L) # pu : n x m
        mul!(pϕ, ϕc[:, rg], ∂L)
        @inbounds for l in Base.OneTo(m)
            phase += dot(pu[:, l], pϕ[:, l]) * ω[l]
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
@views function functional_coll_bare!(pb::PeriodicOrbitOCollProblem,
                                    out::AbstractMatrix, 
                                    u::AbstractMatrix{𝒯}, 
                                    period, 
                                    (L, ∂L), pars) where 𝒯
    # out is of size (n, m⋅Ntst + 1)
    n, ntimes = size(u)
    m = pb.mesh_cache.degree
    Ntst = pb.mesh_cache.Ntst
    # we want slices at fixed times, hence pj[:, j] is the fastest
    # temporaries to reduce allocations
    pj  = get_tmp(pb.cache.gj, u)  #zeros(𝒯, n, m)
    ∂pj = get_tmp(pb.cache.∂gj, u) #zeros(𝒯, n, m)
    tmp = get_tmp(pb.cache.tmp, u)
    mesh = getmesh(pb)
    # range for locating time slices
    rg = axes(out, 2)[UnitRange(1, m+1)]
    @inbounds for j in 1:Ntst
        dt = (mesh[j+1] - mesh[j]) / 2
        mul!( pj, u[:, rg], L)  # size (n, m)
        mul!(∂pj, u[:, rg], ∂L) # size (n, m)
        # compute the collocation residual
        for l in Base.OneTo(m)
            _POO_coll_scheme!(pb, out[:, rg[l]], ∂pj[:, l], pj[:, l], pars, period * dt, tmp)
        end
        # carefull here https://discourse.julialang.org/t/is-this-a-bug-scalar-ranges-with-the-parser/70670/4"
        rg = rg .+ m
    end
    out
end

function functional_coll!(pb::PeriodicOrbitOCollProblem, 
                                out::AbstractMatrix, 
                                u::AbstractMatrix, 
                                period, 
                                (L, ∂L), 
                                pars)
    functional_coll_bare!(pb, out, u, period, (L, ∂L), pars)
    # add the periodicity condition
    @views @. out[:, end] = u[:, end] - u[:, 1]
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
    functional_coll!(prob, resultc, uc, T, Ls, pars)
    # add the phase condition ∫_0^T < u(t), ∂ϕ(t) > dt / T
    result[end] = phase_condition(prob, uc, Ls, T)
    return result
end

"""
$(SIGNATURES)

Compute the identity matrix associated with the collocation problem.
"""
function LinearAlgebra.I(coll::PeriodicOrbitOCollProblem, u, par)
    T = getperiod(coll, u, par)
    N, _, _ = size(coll)
    Icoll = analytical_jacobian(coll, u, par; ρD = 0, ρF = 0, ρI = -1/T)
    Icoll[:, end] .= 0
    Icoll[end, :] .= 0
    Icoll[end-N:end-1, 1:N] .= 0
    Icoll[end-N:end-1, end-N:end-1] .= 0
    Icoll
end

"""
$(SIGNATURES)

Compute the jacobian of the problem defining the periodic orbits by orthogonal collocation using an analytical formula. More precisely, it discretises

ρD * D - T * (ρF * F + ρI * I)

"""
@views function analytical_jacobian!(J,
                                    coll::PeriodicOrbitOCollProblem,
                                    u::AbstractVector{𝒯},
                                    pars; 
                                    _transpose::Bool = false,
                                    ρD = one(𝒯),
                                    ρF = one(𝒯),
                                    ρI = zero(𝒯)) where {𝒯}
    n, m, Ntst = size(coll)
    nJ = length(coll) + 1
    L, ∂L = get_Ls(coll.mesh_cache) # L is of size (m+1, m)
    Ω = get_matrix_phase_condition(coll)
    mesh = getmesh(coll)
    period = getperiod(coll, u, nothing)
    uc = get_time_slices(coll, u)
    ϕc = get_time_slices(coll.ϕ, n, m, Ntst)
    pj = get_tmp(coll.cache.gi, u) # zeros(𝒯, n, m)
    ϕj = get_tmp(coll.cache.gj, u) # zeros(𝒯, n, m)
    uj = get_tmp(coll.cache.uj, u) # zeros(𝒯, n, m+1)
    In = I(n)
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
        mul!(pj, uc[:, rg], L) # pj ≈ (L * uj')'
        # put the jacobian of the vector field
        for l in 1:m
            _rgX = rgNx .+ (l-1)*n
            if _transpose == false
                jacobian!(VF, J0, pj[:, l], pars)
            else
                J0 .= transpose(jacobian(VF, pj[:, l], pars))
            end

            for l2 in 1:m+1
                J[_rgX, rgNy .+ (l2-1)*n ] .= @. (-α * L[l2, l] * ρF) * J0 +
                                                 (ρD * ∂L[l2, l] - α * L[l2, l] * ρI) * In
            end
            # add derivative w.r.t. the period
            # J[rgNx .+ (l-1)*n, end] .= residual(VF, pj[:,l], pars) .* (-dt)
            residual!(VF, J[_rgX, nJ], pj[:, l], pars)
            J[_rgX, nJ] .*= (-dt)
        end
        rg = rg .+ m
        rgNx = rgNx .+ (m * n)
        rgNy = rgNy .+ (m * n)
    end

    rg = 1:n
    J[end, 1:end-1] .= 0
    for j = 1:Ntst
        for k₁ = 1:m+1
            for k₂ = 1:m+1
                # J[end, rg] .+= Ω[k₁, k₂] .* ϕc[:, (j-1)*m + k₂]
                axpby!(Ω[k₁, k₂] / period, ϕc[:, (j-1)*m + k₂], 1, J[nJ, rg])
            end
            if k₁ < m + 1
                rg = rg .+ n
            end
        end
    end
    vj = get_tmp(coll.cache.vj, u)
    phase = _phase_condition(coll, uc, (L, ∂L), (pj, uj, ϕj, vj), period)
    J[nJ, nJ] = -phase / period
    return J
end

analytical_jacobian(coll::PeriodicOrbitOCollProblem, 
                            u::AbstractArray, 
                            pars; 
                            𝒯 = eltype(u), 
                            k...) = analytical_jacobian!(zeros(𝒯, length(coll)+1, length(coll)+1), 
                                                        coll, 
                                                        u, 
                                                        pars; 
                                                        k...)

function analytical_jacobian_sparse(coll::PeriodicOrbitOCollProblem,
                                    u::AbstractVector,
                                    pars; 
                                    k...)
    jacBlock = jacobian_poocoll_block(coll, u, pars; k...)
    block_to_sparse(jacBlock)
end

function jacobian_poocoll_block(coll::PeriodicOrbitOCollProblem,
                                u::AbstractVector,
                                pars;
                                kwargs...) 
    n, m, Ntst = size(coll)
    blocks = n * ones(Int64, 1 + m * Ntst + 1); blocks[end] = 1
    n_blocks = length(blocks)
    J = BlockArray(zeros(length(u), length(u)), blocks,  blocks)
    jacobian_poocoll_block!(J, coll, u, pars; kwargs...)
    return J
end

@views function jacobian_poocoll_block!(J,
                                coll::PeriodicOrbitOCollProblem,
                                u::AbstractVector{𝒯},
                                pars;
                                _transpose::Bool = false,
                                ρD = one(𝒯),
                                ρF = one(𝒯),
                                ρI = zero(𝒯)) where {𝒯}
    n, m, Ntst = size(coll)
    n_blocks = size(J.blocks, 1)
    # temporaries
    L, ∂L = get_Ls(coll.mesh_cache) # L is of size (m+1, m)
    Ω = get_matrix_phase_condition(coll)
    mesh = getmesh(coll)
    period = getperiod(coll, u, nothing)
    uc = get_time_slices(coll, u)
    ϕc = get_time_slices(coll.ϕ, n, m, Ntst)
    pj = get_tmp(coll.cache.gi, u)   # zeros(𝒯, n, m)
    tmp = get_tmp(coll.cache.tmp, u) # zeros(𝒯, n, m)
    ϕj = get_tmp(coll.cache.gj, u)   # zeros(𝒯, n, m)
    uj = get_tmp(coll.cache.uj, u)   # zeros(𝒯, n, m+1)

    In = I(n)
    J0 = jacobian(coll.prob_vf, u[1:n], pars)

    # put boundary condition
    J[Block(1 + m * Ntst, 1 + m * Ntst)] = In
    J[Block(1 + m * Ntst, 1)] = -In

    # loop over the mesh intervals
    rg = UnitRange(1, m+1)

    for j in 1:Ntst
        dt = (mesh[j+1] - mesh[j]) / 2
        α = period * dt
        mul!(pj, uc[:, rg], L) # pj ≈ (L * uj')'
        mul!(ϕj, ϕc[:, rg], ∂L)
        # put the jacobian of the vector field
        for l in 1:m
            if ~_transpose
                J0 .= jacobian(coll.prob_vf, pj[:,l], pars)
            else
                J0 .= transpose(jacobian(coll.prob_vf, pj[:,l], pars))
            end

            for l2 in 1:m+1
                J[Block( l + (j-1)*m ,l2 + (j-1)*m) ] = (-α * L[l2, l]) .* (ρF .* J0 + ρI * I) .+
                                                         ρD * (∂L[l2, l] .* In)
            end
            # add derivative w.r.t. the period
            J[Block(l + (j-1)*m, n_blocks)] = reshape(residual(coll.prob_vf, pj[:,l], pars) .* (-dt), n, 1)
        end
        rg = rg .+ m
    end

    rg = 1
    J[end, 1:end-1] .= 0
    for j = 1:Ntst
        for k₁ = 1:m+1
            for k₂ = 1:m+1
                J[Block(n_blocks, rg)] += reshape(Ω[k₁, k₂] .* ϕc[:, (j-1)*m + k₂], 1, n)
            end
            if k₁ < m + 1
                rg += 1
            end
        end
    end
    J[end, 1:end-1] ./= period

    J[Block(n_blocks, n_blocks)] = reshape([-phase_condition(coll, uc, (L, ∂L), period) / period],1,1)

    return J
end

@views function jacobian_poocoll_sparse_indx!(coll::PeriodicOrbitOCollProblem,
                                        J::AbstractSparseMatrix,
                                        u::AbstractVector{𝒯},
                                        pars,
                                        indx; 
                                        _transpose::Bool = false,
                                        ρD = one(𝒯),
                                        ρF = one(𝒯),
                                        ρI = zero(𝒯),
                                        δ = convert(𝒯, 1e-9), 
                                        updateborder = true) where {𝒯}
    n, m, Ntst = size(coll)
    # allocate the jacobian matrix
    blocks = n * ones(Int64, 1 + m * Ntst + 1); blocks[end] = 1
    n_blocks = length(blocks)
    @assert n_blocks == size(indx, 1)
    # J = BlockArray(spzeros(length(u), length(u)), blocks,  blocks)
    # temporaries
    L, ∂L = get_Ls(coll.mesh_cache) # L is of size (m+1, m)
    Ω = get_matrix_phase_condition(coll)
    mesh = getmesh(coll)
    period = getperiod(coll, u, nothing)
    uc = get_time_slices(coll, u)
    ϕc = get_time_slices(coll.ϕ, size(coll)...)
    pj = zeros(𝒯, n, m)
    ϕj = zeros(𝒯, n, m)
    uj = zeros(𝒯, n, m+1)
    In = sparse(I(n))
    J0 = jacobian(coll.prob_vf, uc[1:n], pars)
    tmpJ = copy(J0)
    @assert J0 isa AbstractSparseMatrix

    # put boundary condition
    J.nzval[indx[1 + m * Ntst, 1 + m * Ntst]] = In.nzval
    J.nzval[indx[1 + m * Ntst, 1]] = -In.nzval

    # loop over the mesh intervals
    rg = UnitRange(1, m+1)
    rgNx = UnitRange(1, n)
    rgNy = UnitRange(1, n)

    for j in 1:Ntst
        uj .= uc[:, rg]
        mul!(pj, uj, L) # pj ≈ (L * uj')'
        dt = (mesh[j+1]-mesh[j]) / 2
        α = period * dt
        mul!(ϕj, ϕc[:, rg], ∂L)
        # put the jacobian of the vector field
        for l in 1:m
            if ~_transpose
                J0 .= jacobian(coll.prob_vf, pj[:,l], pars)
            else
                J0 .= transpose(jacobian(coll.prob_vf, pj[:,l], pars))
            end

            for l2 in 1:m+1
                tmpJ .= (-α * L[l2, l]) .* (ρF .* J0 + ρI * I) .+ ρD * (∂L[l2, l] .* In)
                J.nzval[indx[ l + (j-1) * m ,l2 + (j-1)*m] ] .= sparse(tmpJ).nzval
            end
            # add derivative w.r.t. the period
            J[rgNx .+ (l-1)*n, end] .= residual(coll.prob_vf, pj[:,l], pars) .* (-dt)
        end
        rg = rg .+ m
        rgNx = rgNx .+ (m * n)
    end

    rg = 1:n
    J[end, 1:end-1] .= 0
    for j = 1:Ntst
        for k₁ = 1:m+1
            for k₂ = 1:m+1
                J[end, rg] .+= Ω[k₁, k₂] .* ϕc[:, (j-1)*m + k₂]
            end
            if k₁ < m + 1
                rg = rg .+ n
            end
        end
    end
    J[end, 1:end-1] ./= period
    J[end, end] = -phase_condition(coll, uc, (L, ∂L), period) / period
    return J
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

# function needed for automatic Branch switching from Hopf bifurcation point
function re_make(coll::PeriodicOrbitOCollProblem,
                 prob_vf,
                 hopfpt,
                 ζr::AbstractVector,
                 orbitguess_a,
                 period; 
                 orbit = t -> t,
                 k...)
    M = length(orbitguess_a)
    N = length(ζr)

    _, m, Ntst = size(coll)
    n_unknows = N * (1 + m * Ntst)

    # update the problem
    probPO = setproperties(coll, N = N, prob_vf = prob_vf, ϕ = zeros(n_unknows), xπ = zeros(n_unknows), cache = POCollCache(eltype(coll), N, m))

    probPO.xπ .= 0

    ϕ0 = generate_solution(probPO, t -> orbit(2pi*t/period + pi), period)
    probPO.ϕ .= @view ϕ0[1:end-1]

    # append period at the end of the initial guess
    orbitguess = generate_solution(probPO, t -> orbit(2pi*t/period), period)

    return probPO, orbitguess
end

##########################
# problem wrappers
residual(prob::WrapPOColl, x, p) = residual(prob.prob, x, p)
jacobian(prob::WrapPOColl, x, p) = prob.jacobian(x, p)
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
####################################################################################################
const DocStringJacobianPOColl = """
- `jacobian` Specify the choice of the linear algorithm, which must belong to `(AutoDiffDense(), )`. This is used to select a way of inverting the jacobian dG
    - For `AutoDiffDense()`. The jacobian is formed as a dense Matrix. You can use a direct solver or an iterative one using `options`. The jacobian is formed inplace.
    - For `DenseAnalytical()` Same as for `AutoDiffDense` but the jacobian is formed using a mix of AD and analytical formula.
"""

function _newton_pocoll(probPO::PeriodicOrbitOCollProblem,
                        orbitguess,
                        options::NewtonPar;
                        defOp::Union{Nothing, DeflationOperator{T, Tf, vectype}} = nothing,
                        kwargs...) where {T, Tf, vectype}
    jacobianPO = probPO.jacobian
    @assert jacobianPO in
            (AutoDiffDense(), DenseAnalytical(), FullSparse(), DenseAnalyticalInplace()) "This jacobian $jacobianPO is not defined. Please chose another one."

    if jacobianPO isa DenseAnalytical
        jac = (x, p) -> analytical_jacobian(probPO, x, p)
    elseif jacobianPO isa DenseAnalyticalInplace
        _J = analytical_jacobian(probPO, orbitguess, getparams(probPO))
        jac = (x, p) -> analytical_jacobian!(_J, probPO, x, p)
    elseif jacobianPO isa FullSparse
        jac = (x, p) -> analytical_jacobian_sparse(probPO, x, p)
    elseif jacobianPO isa FullSparseInplace
        _J = analytical_jacobian_sparse(probPO, orbitguess, par)
        jac = (x, p) -> analytical_jacobian!(_J, probPO, x, p)
    else
        jac = (x, p) -> ForwardDiff.jacobian(z -> residual(probPO, z, p), x)
    end

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
$(SIGNATURES)

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
    $(SIGNATURES)

This function is similar to `newton(probPO, orbitguess, options, jacobianPO; kwargs...)` except that it uses deflation in order to find periodic orbits different from the ones stored in `defOp`. We refer to the mentioned method for a full description of the arguments. The current method can be used in the vicinity of a Hopf bifurcation to prevent the Newton-Krylov algorithm from converging to the equilibrium point.
"""
function newton(probPO::PeriodicOrbitOCollProblem,
                orbitguess,
                defOp::DeflationOperator,
                options::NewtonPar;
                kwargs...)
    _newton_pocoll(probPO, orbitguess, options; defOp = defOp, kwargs...)
end

function generate_jacobian(coll::PeriodicOrbitOCollProblem, 
                        orbitguess, 
                        par; 
                        δ = convert(eltype(orbitguess), 1e-8),
                        Jcoll_matrix = nothing
                        )
    jacobianPO = coll.jacobian
    @assert jacobianPO in (AutoDiffDense(), DenseAnalytical(), FullSparse(), FullSparseInplace(), DenseAnalyticalInplace()) "This jacobian is not defined. Please chose another one."

    if jacobianPO isa DenseAnalytical
        jac = (x, p) -> FloquetWrapper(coll, analytical_jacobian(coll, x, p), x, p)
    elseif jacobianPO isa DenseAnalyticalInplace
        floquet_wrap = FloquetWrapper(coll, analytical_jacobian(coll, orbitguess, par), orbitguess, par)
        function jac(x, p)
            analytical_jacobian!(floquet_wrap.jacpb, floquet_wrap.pb, x, p)
            floquet_wrap.x .= x
            floquet_wrap.par = p
            floquet_wrap
        end
    elseif jacobianPO isa FullSparse
        jac = (x, p) -> FloquetWrapper(coll, analytical_jacobian_sparse(coll, x, p), x, p)
    elseif jacobianPO isa FullSparseInplace
        _J = analytical_jacobian_sparse(coll, orbitguess, par)
        indx = get_blocks(coll, _J)
        jac = (x, p) -> FloquetWrapper(coll, jacobian_poocoll_sparse_indx!(coll, _J, x, p, indx), x, p)
    else
        # if you use jacobian!, it has issues with DDEBifurcationKit
        jac = (x, p) -> FloquetWrapper(coll, ForwardDiff.jacobian(z -> residual(coll, z, p), x), x, p)
    end
end

"""
$(SIGNATURES)

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
    jacPO = generate_jacobian(coll, orbitguess, getparams(coll); δ)
    if linear_algo isa COPBLS
        _Jcoll = analytical_jacobian(coll, orbitguess, getparams(coll))
        linear_algo = COPBLS(coll)
        Nbls = length(coll) + 2
        floquet_wrap = jacPO(orbitguess, getparams(coll))
        linear_algo = COPBLS(
                        cache = linear_algo.cache,
                        solver = FloquetWrapperLS(linear_algo.solver),
                        J = similar(_Jcoll, Nbls, Nbls), 
                        )
        linear_algo.J .= 0
    else
        linear_algo = @set linear_algo.solver = FloquetWrapperLS(linear_algo.solver)
    end
    options = _contParams.newton_options
    contParams = @set _contParams.newton_options.linsolver = FloquetWrapperLS(options.linsolver)

    # we have to change the Bordered linearsolver to cope with our type FloquetWrapper
    alg = update(alg, contParams, linear_algo)

    if compute_eigenelements(contParams)
        contParams = @set contParams.newton_options.eigsolver = eigsolver
    end

    # change the user provided finalise function by passing prob in its parameters
    _finsol = modify_po_finalise(coll, kwargs, coll.update_section_every_step)
    # this is to remove this part from the arguments passed to continuation
    _kwargs = (record_from_solution = record_from_solution, plot_solution = plot_solution)
    _recordsol = modify_po_record(coll, _kwargs, getparams(coll.prob_vf), getlens(coll.prob_vf))
    _plotsol = modify_po_plot(coll, _kwargs)

    probwp = WrapPOColl(coll, jacPO, orbitguess, getparams(coll), getlens(coll), _plotsol, _recordsol)

    br = continuation(probwp, alg,
                      contParams;
                      kwargs...,
                      kind = PeriodicOrbitCont(),
                      finalise_solution = _finsol)
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
@views function updatesection!(coll::PeriodicOrbitOCollProblem, 
                                x::AbstractVector, 
                                par)
    @debug "Update section Collocation"
    # update the reference point
    coll.xπ .= 0

    # update the "normals"
    coll.ϕ .= x[eachindex(coll.ϕ)]
    return true
end
####################################################################################################
# mesh adaptation method

@views function (sol::POSolution{ <: PeriodicOrbitOCollProblem})(t0)
    n, m, Ntst = size(sol.pb)
    xc = get_time_slices(sol.pb, sol.x)

    T = getperiod(sol.pb, sol.x, nothing)
    t = mod(t0, T) / T

    mesh = getmesh(sol.pb)
    index_t = searchsortedfirst(mesh, t) - 1
    if index_t <= 0
        return sol.x[1:n]
    elseif index_t > Ntst
        return xc[:, end]
    end
    @assert mesh[index_t] <= t <= mesh[index_t+1] "Please open an issue on the website of BifurcationKit.jl"
    σ = σj(t, mesh, index_t)
    # @assert -1 <= σ <= 1 "Strange value of $σ"
    σs = get_mesh_coll(sol.pb)
    out = zeros(typeof(t), sol.pb.N)
    rg = (1:m+1) .+ (index_t-1) * m
    for l in 1:m+1
        out .+= xc[:, rg[l]] .* lagrange(l, σ, σs)
    end
    out
end

"""
$(SIGNATURES)

Perform mesh adaptation of the periodic orbit problem. Modify `pb` and `x` inplace if the adaptation is successfull.

See page 367 of [1] and also [2].

References:
[1] Ascher, Uri M., Robert M. M. Mattheij, and Robert D. Russell. Numerical Solution of Boundary Value Problems for Ordinary Differential Equations. Society for Industrial and Applied Mathematics, 1995. https://doi.org/10.1137/1.9781611971231.

[2] R. D. Russell and J. Christiansen, “Adaptive Mesh Selection Strategies for Solving Boundary Value Problems,” SIAM Journal on Numerical Analysis 15, no. 1 (February 1978): 59–80, https://doi.org/10.1137/0715004.
"""
function compute_error!(coll::PeriodicOrbitOCollProblem, x::AbstractVector{𝒯};
                        normE = norminf,
                        verbosity::Bool = false,
                        K = Inf,
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
        @error "[Mesh-adaptation]. The mesh is non monotonic! Please report the error to the website of BifurcationKit.jl"
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
    ϕ = sk.^(1/m)
    # if the monitor function is too small, don't do anything
    if maximum(ϕ) < 1e-7
        return (success = true, newmesh = nothing)
    end
    ϕ = max.(ϕ, maximum(ϕ) / K)
    @assert length(ϕ) == Ntst "Error. Please open an issue of the website of BifurcationKit.jl"
    # compute θ = ∫ϕ but also all intermediate values
    # these intermediate values are useful because the integral is piecewise linear
    # and equipartition is analytical
    # there are ntst values for the integrals, one for (0, mesh[2]), (mesh[2], mesh[3])...
    θs = zeros(𝒯, Ntst); θs[1] = ϕ[1] * (τsT[2] - τsT[1])
    for i = 2:Ntst
        θs[i] = θs[i-1] + ϕ[i] * (τsT[i+1] - τsT[i])
    end
    θs = vcat(0, θs)
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
          "   ┌─ Mesh adaptation, new mesh hi = time steps",
        "\n   ├─── min(hi)       = ", minimum(diff(newmesh)),
        "\n   ├─── h = max(hi)   = ", h,
        "\n   ├─── K = max(h/hi) = ", maximum(h ./ diff(newmesh)),
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
$(SIGNATURES)

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

