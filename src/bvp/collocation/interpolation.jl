@views function __interpolate_posolution(mesh_cache::MeshCollocationCache, t0, xm::AbstractMatrix, period)
    m, Ntst = size(mesh_cache)
    t = mod(t0, period) / period
    mesh = getmesh(mesh_cache)
    index_t = searchsortedfirst(mesh, t) - 1
    if index_t <= 0
        return xm[:, 1]
    elseif index_t > Ntst
        return xm[:, end]
    end
    @assert mesh[index_t] <= t <= mesh[index_t+1] "Please open an issue on the website of BifurcationKit.jl"
    σ = σj(t, mesh, index_t)
    σs = mesh_cache.σs
    out = zeros(typeof(t), size(xm, 1))
    rg = (1:m+1) .+ (index_t - 1) * m
    for l in 1:m+1
        out .+= xm[:, rg[l]] .* lagrange(l, σ, σs)
    end
    out
end
