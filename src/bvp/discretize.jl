# discretize - Create a discretized BVP from model + discretizer
#
# This is the main user-facing function to combine a mathematical
# BVP model with a numerical discretization method.

using DocStringExtensions
import SciMLBase

"""
$(TYPEDSIGNATURES)

Discretize a BVP model using the specified method.

## Arguments
- `model::BVPModel`: Mathematical BVP formulation
- `disc::AbstractDiscretizer`: Discretization method

## Returns
- `DiscretizedBVP`: A discretized problem ready for Newton/continuation

## Example
```julia
# Define model
F(u, p) = [u[2], -p.ω² * u[1]]
model = PeriodicOrbitModel(F; n=2)

# Choose discretization
disc = Trapeze(M=100)

# Create discretized problem
bvp = discretize(model, disc)

# Use with continuation
prob = BifurcationProblem(bvp, x0, params, (@optic _.ω))
```
"""
function discretize end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shooting
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function discretize(model::BVPModel{ <: Union{SciMLBase.ODEProblem, SciMLBase.EnsembleProblem, SciMLBase.DAEProblem}}, 
                    disc::Shooting; 
                    kwargsDE...)
    cache = BK.Shooting(mesh_size(disc), model.F, disc.alg; parallel = is_parallel(disc), kwargsDE...)
    return DiscretizedBVP(model, disc, cache)
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trapezoid
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function discretize(model::BVPModel, disc::Trapeze)
    n = state_dimension(model)
    @assert n > 0 "State dimension must be specified in the model"

    M = disc.M

    # Create a BifurcationProblem wrapper for the vector field
    # This is needed by PeriodicOrbitTrapProblem
    # Note: We use a dummy params/lens since these are handled by BVPBifProblem
    # record_from_solution is required to avoid errors
    prob_vf = BK.BifurcationProblem(
        (u, p) -> model.F(u, p),    # Vector field
        zeros(n),                     # Dummy initial guess
        (dummy = 0.0,),               # Dummy params
        (BK.@optic _.dummy);             # Dummy lens
        inplace = false,              # Must match the signature above
        record_from_solution = (x, p; k...) -> nothing
    )

    # Create PeriodicOrbitTrapProblem for efficient residual computation
    # Phase constraint vectors (will be overwritten later)
    ϕ = zeros(n * M)
    xπ = zeros(n * M)

    po_trap = BK.Trapeze(;
        prob_vf,
        section = BK.SectionTrapeze(ϕ, xπ),
        M,
        mesh = disc.mesh,
        N = n,
    )

    cache = (
        po_trap = po_trap,          # BifurcationKit's trap problem for po_residual_bare!
        F_vals = zeros(n, M),       # Vector field at each slice # TODO utile? et type?
        temp = zeros(n),            # Temporary
    )

    return DiscretizedBVP(model, disc, cache)
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Collocation
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function discretize(model::BVPModel, disc::Collocation)
    n = state_dimension(model)
    @assert n > 0 "State dimension must be specified in the model"

    (;Ntst, m, K, meshadapt) = disc
    𝒯 = eltype(model)

    mesh_cache = MeshCollocationCache(Ntst, m, 𝒯)
    coll_cache = CollocationCache(𝒯, Ntst, n, m)

    cache = (; mesh_cache, coll_cache)
    return DiscretizedBVP(model, disc, cache)
end

function discretize(model::POModel, disc::Collocation)
    n = state_dimension(model)
    @assert n > 0 "State dimension must be specified in the model"

    (;Ntst, m) = disc

    # Allocate phase condition arrays for SectionCollocation
    𝒯 = eltype(model)
    ϕ = zeros(𝒯, n * n_mesh_pts(m, Ntst))
    ∂ϕ = zeros(𝒯, n, Ntst * m)
    section = BK.SectionCollocation(ϕ, ∂ϕ)

    # Use invoke to call the generic BVP discretize method, avoiding infinite recursion on POModel
    d_bvp = invoke(discretize, Tuple{BVPModel, typeof(disc)}, model, disc)

    return DiscretizedPO(d_bvp, section)
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utility: Generate initial guess
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
$(TYPEDSIGNATURES)

Generate an initial guess for the discretized BVP from an orbit function.

## Arguments
- `bvp::DiscretizedBVP`: The discretized problem
- `orbit`: Function `t -> u(t)` giving the orbit at time t

## Returns
- Initial guess vector X

## Example
```julia
x0 = generate_solution(bvp, t -> [cos(t), sin(t)])
```
"""
function generate_solution(d_bvp::DiscretizedBVP, orbit)
    return generate_solution(get_model(d_bvp), get_discretizer(d_bvp), get_cache(d_bvp), orbit)
end

function generate_solution(model::BVPModel, disc::Shooting, cache, orbit)
    n = state_dimension(model)
    M = disc.M
    t0, tf = get_time_interval(model)
    T = tf - t0
    # Sample at M shooting points
    X = zeros(n * M + 1)
    for i in 1:M
        t = (i - 1) / M * T
        X[(i-1)*n+1 : i*n] .= orbit(t)
    end
    return X
end

function generate_solution(model::BVPModel, disc::Trapeze, cache, orbit)
    n = state_dimension(model)
    M = disc.M
    t0, tf = get_time_interval(model)
    T = tf - t0
    # Sample at M time slices
    X = zeros(n * M + 1)
    for i in 1:M
        t = (i - 1) / (M - 1) * T
        X[(i-1)*n+1 : i*n] .= orbit(t)
    end
    return X
end

function generate_solution(model::BVPModel, disc::Collocation, cache, orbit)
    n = state_dimension(model)
    mesh_cache = cache.mesh_cache
    𝒯 = eltype(mesh_cache.τs)
    m, Ntst = size(mesh_cache)
    ts = get_times(mesh_cache)
    Nt = length(ts)
    t0, tf = get_time_interval(model)
    ci = zeros(𝒯, n, Nt)
    for (l, t) in pairs(ts)
        ci[:, l] .= orbit(t0 + t * (tf - t0))
    end
    return vec(ci)
end