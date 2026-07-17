# ─────────────────────────────────────────────────────────────────────────────
# COP (Condensation of Parameters) — New interface on BVP.BVP.DiscretizedPO
#
# This file implements the same algorithms as cop_legacy.jl but operates
# on the new generic interface: BVP.DiscretizedPO{<:BVP.POModel, <:BVP.Collocation}.
#
# Key differences from cop_legacy.jl:
#   - COPCACHE constructor accepts BVP.DiscretizedPO instead of Collocation
#   - Size extraction uses BVP getters (state_dimension, disc.m, disc.Ntst)
#   - Identity matrix accessed via BVP.get_coll_cache(d_po).In
#   - Type 𝒯 inferred from eltype(BVP.get_mesh_cache(d_po))
#   - COPLS / COPBLS accept both Collocation (legacy) and BVP.DiscretizedPO (new)
# ─────────────────────────────────────────────────────────────────────────────

"""
$(TYPEDSIGNATURES)

[New interface] Build a `COPCACHE` from a `BVP.DiscretizedPO{<:BVP.POModel, <:BVP.Collocation}`.

The type parameter `dim` is the border size: `0` for Newton, `1` for PALC.
"""
function COPCACHE(d_po::BVP.DiscretizedPO{<:BVP.POModel, <:BVP.Collocation},
                  ::Val{dim0} = Val(0);
                  𝒯 = eltype(BVP.get_mesh_cache(d_po))) where {dim0}
    if ~(dim0 isa Int64)
        error("You must pass an integer.")
    end
    dim::Int = dim0

    disc  = BVP.get_discretizer(d_po)
    n     = BVP.state_dimension(d_po)
    m     = disc.m
    Ntst  = disc.Ntst
    Npo   = length(d_po)  # n*(m*Ntst+1) + 1  (includes period T)

    Jcoll_tmp = zeros(𝒯, Npo + dim, Npo + dim)
    Jext_tmp  = zeros(𝒯, Ntst * n + n + 1 + dim, Ntst * n + n + 1 + dim)

    new_cache = COPCACHE{dim, 𝒯, typeof(d_po)}(
        Jcoll_tmp,
        Jext_tmp,
        d_po,
        zeros(𝒯, size(Jext_tmp, 1)),
        zeros(𝒯, size(Jext_tmp, 1)),
        zeros(𝒯, n * m)
    )
    return new_cache
end

# ─────────────────────────────────────────────────────────────────────────────
# New constructors for COPLS / COPBLS — dispatch alongside legacy ones
# ─────────────────────────────────────────────────────────────────────────────

COPLS(d_po::BVP.DiscretizedPO{<:BVP.POModel, <:BVP.Collocation}) = COPLS(COPCACHE(d_po, Val(0)))
COPBLS(d_po::BVP.DiscretizedPO{<:BVP.POModel, <:BVP.Collocation}) = COPBLS(; cache = COPCACHE(d_po, Val(1)))

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers: size extraction for BVP.DiscretizedPO (new interface)
# ─────────────────────────────────────────────────────────────────────────────

@inline function _cop_dims(d_po::BVP.DiscretizedPO{<:BVP.POModel, <:BVP.Collocation})
    disc = BVP.get_discretizer(d_po)
    n    = BVP.state_dimension(d_po)
    m    = disc.m
    Ntst = disc.Ntst
    return n, m, Ntst
end

# ─────────────────────────────────────────────────────────────────────────────
# _copy_to_coll! — BVP.DiscretizedPO dispatch
# ─────────────────────────────────────────────────────────────────────────────

@views function _copy_to_coll!(d_po::BVP.DiscretizedPO{<:BVP.POModel, <:BVP.Collocation}, 𝑱, J, ::Val{dim}) where {dim}
    N, m, Ntst = _cop_dims(d_po)
    nbcoll = N * m
    In = BVP.get_coll_cache(d_po).In
    rgᵢ = 1:(nbcoll + N)
    @inbounds for iₜ = 1:Ntst
        𝑱[rgᵢ, rgᵢ] .= J[rgᵢ, rgᵢ]
        rgᵢ = rgᵢ .+ nbcoll
    end

    if dim >= 0
        𝑱[:, end-dim:end] .= J[:, end-dim:end]
        𝑱[end-dim:end, :] .= J[end-dim:end, :]
    end

    # put periodic boundary condition
    𝑱[end-N-dim:end-1-dim, end-N-dim:end-1-dim] .= In
    𝑱[end-N-dim:end-1-dim, 1:N] .= (-1) .* In
    return
end

# ─────────────────────────────────────────────────────────────────────────────
# condensation_of_parameters2! — BVP.DiscretizedPO dispatch
# ─────────────────────────────────────────────────────────────────────────────

function condensation_of_parameters2!(cop_cache::COPCACHE{dim},
                                d_po::BVP.DiscretizedPO{<:BVP.POModel, <:BVP.Collocation},
                                J,
                                In,
                                rhs0) where {dim}
    rhs = rhs0
    𝑱 = cop_cache.Jcoll
    α_values = cop_cache.α_values
    N, m, Ntst = _cop_dims(d_po)
    nbcoll = N * m
    Npo = length(d_po)

    if true
        _copy_to_coll!(d_po, 𝑱, J, Val(dim))
    end

    rgₖ = 1:nbcoll
    rgᵢ = 1:(nbcoll + N)
    for iₜ = 1:Ntst
        @inbounds for k = rgₖ
            colₖ = k + N
            rglast = Iterators.flatten((rgᵢ, Npo:Npo+dim))
            ##########
            # pivoting step
            Jmax = abs(𝑱[k, colₖ])
            iₚ = k
            @inbounds for l = k+1:last(rgₖ)
                absl = abs(𝑱[l, colₖ])
                if absl > Jmax
                    iₚ = l
                    Jmax = absl
                end
            end

            if iₚ != k
                @inbounds for jj in rglast
                    𝑱[k, jj], 𝑱[iₚ, jj] = 𝑱[iₚ, jj], 𝑱[k, jj]
                end
                rhs[k], rhs[iₚ] = rhs[iₚ], rhs[k]
            end
            ##########

            𝑱ₖ = 𝑱[k, colₖ]
            inv𝑱 = inv(𝑱ₖ)

            @inbounds for i = k:last(rgₖ)
                𝑱[i, colₖ] *= inv𝑱
            end
            @inbounds 𝑱[end, colₖ] *= inv𝑱
            if dim >= 1
                @inbounds 𝑱[end-1, colₖ] *= inv𝑱
            end
            if dim >= 2
                @inbounds 𝑱[end-2, colₖ] *= inv𝑱
            end

            @inbounds α = 𝑱[end, colₖ]
            @inbounds β = 𝑱[end-1, colₖ]
            @inbounds γ = 𝑱[end-2, colₖ]

            @inbounds for j in rglast
                    𝑱[end,   j] -= α * 𝑱[k, j]
                if dim >= 1
                    𝑱[end-1, j] -= β * 𝑱[k, j]
                end
                if dim >= 2
                    𝑱[end-2, j] -= γ * 𝑱[k, j]
                end
            end

            rhsk = rhs[k]
            rhs[end] -= α * rhsk
            if dim >= 1
                rhs[end-1] -= β * rhsk
            end
            if dim >= 2
                rhs[end-2] -= γ * rhsk
            end

            @inbounds for i=k+1:last(rgₖ)
                α_values[i - k] = 𝑱[i, colₖ]
                rhs[i] -= α_values[i - k] * rhsk
            end

            for j = Iterators.flatten((rgᵢ, Npo:Npo+dim))
                @inbounds 𝑱kj = 𝑱[k, j]
                for i = k+1:last(rgₖ)
                    @inbounds 𝑱[i, j] -= α_values[i - k] * 𝑱kj
                end
            end
            𝑱[k, colₖ] = 𝑱ₖ
        end

        rgₖ = rgₖ  .+ nbcoll
        rgᵢ = rgᵢ .+ nbcoll
    end

    return rhs
end

# ─────────────────────────────────────────────────────────────────────────────
# _solve_for_internal_variables — BVP.DiscretizedPO dispatch
# ─────────────────────────────────────────────────────────────────────────────

@views function _solve_for_internal_variables(d_po::BVP.DiscretizedPO{<:BVP.POModel, <:BVP.Collocation},
                                         Jcond,
                                         rhs::Vector{𝒯},
                                         sol_ext,
                                         ::Val{δn}) where {𝒯, δn}
    N, m, Ntst = _cop_dims(d_po)
    nbcoll = N * m

    ΔT = sol_ext[end - δn]
    Δp = sol_ext[end]

    r2 = N+1:(m)*N
    r1 = 1:(m-1)*N
    rsol = 1:(m-1)*N
    rN_left = 1:N
    rN = 1:N

    sol_cop = copy(rhs)
    rhs_tmp = zeros(𝒯, (m-1) * N)
    sol_tmp = copy(rhs_tmp)

    sol_cop[1:N] .= sol_ext[1:N]

    for iₜ in 1:Ntst
        Jtemp = LA.UpperTriangular(Jcond[r1, r2])
        left_part = Jcond[r1, rN_left]
        right_part = Jcond[r1, r2[end]+1:r2[end]+N]

        if δn == 0
            rhs_tmp .= @. rhs[rsol] - ΔT * Jcond[r1, end]
        elseif δn == 1
            rhs_tmp .= @. rhs[rsol] - ΔT * Jcond[r1, end-1] - Δp * Jcond[r1, end]
        elseif δn == 2
            rhs_tmp .= @. rhs[rsol] -
                          sol_ext[end] * Jcond[r1, end] -
                        sol_ext[end-1] * Jcond[r1, end-1] -
                        sol_ext[end-2] * Jcond[r1, end-2]
        else
            throw("This version of the current function is not yet implemented. δn = $δn")
        end
        LA.mul!(rhs_tmp, left_part,  sol_ext[rN],      -1, 1)
        LA.mul!(rhs_tmp, right_part, sol_ext[rN .+ N], -1, 1)

        LA.ldiv!(sol_tmp, Jtemp, rhs_tmp)

        sol_cop[rsol .+ N] .= sol_tmp
        sol_cop[rsol[end]+N+1:rsol[end]+2N] .= sol_ext[rN .+ N]

        r1 = r1 .+ nbcoll
        r2 = r2 .+ nbcoll
        rN_left = rN_left .+ nbcoll
        rsol = rsol .+ nbcoll
        rN = rN .+ N
    end
    sol_cop[end - δn:end] .= sol_ext[end - δn:end]
    return sol_cop
end

# ─────────────────────────────────────────────────────────────────────────────
# solve_cop — BVP.DiscretizedPO dispatch
# ─────────────────────────────────────────────────────────────────────────────

"""
$(TYPEDSIGNATURES)

[New interface] Solve the COP linear system for a `BVP.DiscretizedPO{<:BVP.POModel, <:BVP.Collocation}`.
"""
@views function solve_cop(d_po::BVP.DiscretizedPO{<:BVP.POModel, <:BVP.Collocation},
                          J,
                          rhs0,
                          cop_cache::COPCACHE{dim};
                          _DEBUG::Val{debug} = Val(false),
                          _USELU::Val{uselu} = Val(false)) where {dim, debug, uselu}
    @assert size(J, 1) == size(J, 2) == length(rhs0) "The right hand side does not have the right dimension or the jacobian is not square.\nsize(J) = $(size(J)) and \nlength(rhs0) = $(length(rhs0))\n"
    N, m, Ntst = _cop_dims(d_po)
    nbcoll = N * m
    Npo = length(d_po)
    nⱼ = size(J, 1)
    δn = nⱼ - Npo
    @assert δn >= 0
    @assert δn == dim "δn = $δn and dim = $dim should be equal!\nPass a proper COPCACHE."

    Jext = cop_cache.Jext
    @assert size(Jext, 1) == size(Jext, 2) == (Ntst+1)*N+1+δn "Error with matrix of external variables. Please report this issue on the website of BifurcationKit.\nδn = $δn\nsize(Jext) = $(size(Jext))\n(Ntst+1)*N+1+δn = $((Ntst+1)*N+1+δn)\n\n"
    𝒯 = eltype(BVP.get_mesh_cache(d_po))
    Iₙ = BVP.get_coll_cache(d_po).In

    rhs = condensation_of_parameters2!(cop_cache, d_po, J, Iₙ, rhs0)
    Jcop = cop_cache.Jcoll

    if debug === true
        P = Matrix{𝒯}(LinearAlgebra.I(nⱼ))
        Jtmp = zeros(𝒯, nbcoll + δn + 1, nbcoll)
        Fₚ = lu(P); Jcop = Fₚ \ J; rhs = Fₚ \ rhs0
    end

    rhs_ext = build_external_system!(Jext, Jcop, rhs, cop_cache.rhs_ext, Iₙ, Ntst, nbcoll, Npo, δn, N, m)

    if uselu
        F = LA.lu(Jext)
        sol_ext = F \ rhs_ext
    else
        _gaussian_elimination_external_pivoted!(Jext, rhs_ext, N, Ntst, δn)
        sol_ext = _backward_substitution_pivoted(Jext, rhs_ext, cop_cache.sol_ext, N, Ntst, Val(dim))
    end

    return _solve_for_internal_variables(d_po, Jcop, rhs, sol_ext, Val(dim))
end

# ─────────────────────────────────────────────────────────────────────────────
# COPLS / COPBLS callable — BVP.DiscretizedPO dispatch
# ─────────────────────────────────────────────────────────────────────────────

function (ls::COPLS{dim, 𝒯, Tp})(Jc, rhs) where {dim, 𝒯, Tp <: BVP.DiscretizedPO}
    res = solve_cop(ls.cache.coll, Jc, rhs, ls.cache)
    return res, true, 1
end

function (ls::COPBLS{dim, 𝒯, Tp, Ts, Tj})(Jc, dR,
                      dzu, dzp::T,
                      R::AbstractVecOrMat, n::T,
                      ξu::T = one(T), ξp::T = one(T);
                      shift::Tsh = nothing,
                      Mass::Tm = LinearAlgebra.I,
                      dotp = nothing,
                      applyξu! = nothing) where {dim, 𝒯, Tp <: BVP.DiscretizedPO, Ts, Tj, T <: Number, Tsh, Tm}
    if isnothing(shift)
        A = Jc
    else
        A = Jc + shift * Mass
    end
    rhs = vcat(R, n)
    d_po = ls.cache.coll

    if true  # always copy for BVP.DiscretizedPO (no DenseAnalyticalInplace yet)
        _fast_copy_bordered!(ls.J, A)
    end

    ls.J[begin:end-1, end] .= dR
    ls.J[end, begin:end-1] .= conj.(dzu .* ξu)
    ls.J[end, end] = dzp * ξp

    if isnothing(applyξu!) == false
        applyξu!(@view(ls.J[end, begin:end-1]))
    end

    res = solve_cop(d_po, ls.J, rhs, ls.cache)
    return (@view res[begin:end-1]), res[end], true, 1
end