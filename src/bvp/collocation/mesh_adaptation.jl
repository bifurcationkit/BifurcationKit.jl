function _compute_error!(mesh_cache::MeshCollocationCache, n::Int, sol, x::AbstractVector{𝒯}, ΔT;
                        normE = BK.norminf,
                        verbosity::Bool = false,
                        K = 𝒯(Inf),
                        par = nothing,
                        kw...) where 𝒯
    m, Ntst = size(mesh_cache) # recall that m = ncol
    # we need to estimate yᵐ⁺¹ where y is the true periodic orbit.
    # sol is the piecewise polynomial approximation of y.
    # However, sol is of degree m, hence ∂(sol, m+1) = 0
    # we thus estimate yᵐ⁺¹ using ∂(sol, m)
    dmsol = BK.∂(sol, Val(m))
    # we find the values of vm := ∂m(x) at the mid points
    τsT = getmesh(mesh_cache) .* ΔT
    vm = [ dmsol( (τsT[i] + τsT[i+1]) / 2 ) for i = 1:Ntst ]
    ############
    # Approx. IA
    # we compute the error bound
    # We also do not divide by (m+1)!
    errT = [normE(vm[i]) * (τsT[i+1] - τsT[i])^(m+1) for i = 1:Ntst]
    ############
    # Approx. IIA
    # compute the monitoring function
    hT = diff(τsT)
    sk = errT ./ hT .^ (m+1)
    # filter the function sk
    for _ = 1:2
        sk = sum(sk) / Ntst .* ones(Ntst) .+ 0*sk
    end
    # we define the integration weights
    # I need to implement cumsum with trapezoidal rule
    ϕ = sk.^(1/(m+1))
    # if the monitor function is too small, don't do anything
    if maximum(ϕ) < 1e-7
        return (;success = true, newmesh = nothing, ϕ)
    end
    ϕ = max.(ϕ, maximum(ϕ) / K)
    if length(ϕ) != Ntst
        error("Error. Please open an issue of the website of BifurcationKit.jl")
    end
    # the integration is done using the left rectangle rule
    # we compute the integral of the monitor function
    cIϕ = [0; cumsum(ϕ .* hT)]
    Iϕ = cIϕ[end]
    newτsT = similar(τsT)
    newτsT[1] = 0; newτsT[end] = ΔT
    # compute the new uniform mesh
    for i in 2:Ntst
        newτsT[i] = find_t(cIϕ, τsT, (i-1) * Iϕ / Ntst)
    end
    newmesh = newτsT ./ ΔT
    if verbosity
        printstyled(color=:green, " \n\t--> New mesh max / min step: ", maximum(diff(newmesh)), " / ", minimum(diff(newmesh)), 
        "\n")
    end
    return (; newmesh, newτsT, ϕ)
end
