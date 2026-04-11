"""
$(TYPEDEF)

Cache for the linear solver based on condensation of parameters (COP) [1].

!!! danger "`dim` type parameter"
    When using the cache solve a linear problem associated to a matrix `A`, the type parameter `dim` is such that `length(coll) + 1 + dim = size(A, 1)`

# Internal fields

$(TYPEDFIELDS)

# Constructor

```
COPCACHE(coll::PeriodicOrbitOCollProblem, Val(0))
```

## Reference(s)

[1] Govaerts, Willy, Yuri A. Kuznetsov, and Annick Dhooge. “Auto94p.” SIAM Journal on Scientific Computing 27, no. 1 (January 1, 2005): 231–52. https://doi.org/10.1137/030600746.

"""
struct COPCACHE{dim, 𝒯, Tp}
    "cache to hold the factorized form of the matrix collocation matrix J"
    Jcoll::Matrix{𝒯}
    "cache to hold the linear system for the external variables"
    Jext::Matrix{𝒯}
    "collocation problem. It is needed here because linear solver requires it."
    coll::Tp
    "r.h.s. of external problem."
    rhs_ext::Vector{𝒯}
    "solution of external problem."
    sol_ext::Vector{𝒯}
    "alpha values, buffer used in COP."
    α_values::Vector{𝒯}

    function COPCACHE(coll::PeriodicOrbitOCollProblem, 
                        ::Val{dim0} = Val(0); 
                        𝒯 = eltype(coll)) where {dim0}
        if ~(dim0 isa Int64)
            error("You must pass an integer.")
        end
        dim::Int = dim0
        N, m, Ntst = size(coll)
        n = N
        nbcoll = n * m
        Npo = length(coll) + 1

        Jcoll_tmp = zeros(𝒯, Npo + dim, Npo + dim)
        Jext_tmp  = zeros(𝒯, Ntst * N  + N + 1 + dim, Ntst * N  + N + 1 + dim)

        nⱼ = size(Jcoll_tmp, 1)

        new{dim, 𝒯, typeof(coll)}(
                                    Jcoll_tmp,
                                    Jext_tmp,
                                    coll,
                                    zeros(𝒯, size(Jext_tmp, 1)),
                                    zeros(𝒯, size(Jext_tmp, 1)),
                                    zeros(𝒯, N * m)
                                    )
    end
end
_getdim(::COPCACHE{dim}) where {dim} = dim

"""
$TYPEDEF

Linear solver based on the condensation of parameters.

# Internal fields

$TYPEDFIELDS

# Constructors

- `COPBLS()`
- `COPBLS(coll::PeriodicOrbitOCollProblem; cache::COPCACHE, solver = nothing, J = nothing)`

# Related

See `solve_cop`.
"""
struct COPLS{dim, 𝒯, Tp} <: AbstractDirectLinearSolver
    cache::COPCACHE{dim, 𝒯, Tp}
end

"""
$TYPEDEF

Bordered linear solver based on the condensation of parameters. `dim` in the struct definition is the size of the border counting the phase condition. It is thus `dim = 1` for COPLS and `dim = 2` for the case of arclength continuation of periodic orbits as there are two constraints: the phase and the arclength.

# Internal fields

$TYPEDFIELDS

# Constructors

- `COPBLS()`
- `COPBLS(coll::PeriodicOrbitOCollProblem; N = 0, cache::COPCACHE, solver = nothing, J = nothing)`

# Related

See `solve_cop`.
"""
struct COPBLS{dim, 𝒯, Tp, Ts, Tj} <: AbstractBorderedLinearSolver
    "Cache for the COP method. It is a subtype of COPCACHE."
    cache::COPCACHE{dim, 𝒯, Tp}
    "Linear solver. Defaults to `nothing`."
    solver::Ts
    "Cache for the bordered jacobian matrix."
    J::Tj

    function COPBLS(coll = PeriodicOrbitOCollProblem(2, 2; N = 0);
                    cache::COPCACHE{dim, 𝒯, Tp} = COPCACHE(coll, Val(1)), 
                    solver::Ts = nothing, 
                    J::Tj = nothing) where {dim, 𝒯, Tp, Ts, Tj}
        new{dim, 𝒯, Tp, Ts, Tj}(cache, solver, J)
    end
end
@inline _getdim(cop::COPBLS{dim}) where {dim} = _getdim(cop.cache)

COPLS(coll::PeriodicOrbitOCollProblem) = COPLS(COPCACHE(coll, Val(0)))
COPBLS(coll::PeriodicOrbitOCollProblem) = COPBLS(; cache = COPCACHE(coll, Val(1)))
COPLS() = COPLS(PeriodicOrbitOCollProblem(2, 2; N = 0))

"""
$(TYPEDSIGNATURES)

Solve the linear system associated with the collocation problem for computing periodic orbits. It returns the solution to the equation `J * sol = rhs0`. It can also solve a bordered version of the above problem and the border size `δn` is inferred at run time.

## Arguments
- `coll::PeriodicOrbitOCollProblem` collocation problem
- `J::Matrix`
- `rhs0::Vector`

## Optional arguments
- `_DEBUG = false` use a debug mode in which the condensation of parameters is performed without an analytical formula.
- `_USELU = false` use LU factorization instead of gaussian elimination and backward substitution to solve the linear problem.
"""
@views function solve_cop(coll::PeriodicOrbitOCollProblem, 
                          J, 
                          rhs0, 
                          cop_cache::COPCACHE{dim}; 
                          _DEBUG::Val{debug} = Val(false), 
                          _USELU::Val{uselu} = Val(false)) where {dim, debug, uselu}
    @assert size(J, 1) == size(J, 2) == length(rhs0) "The right hand side does not have the right dimension or the jacobian is not square. \nsize(J) = $(size(J)) and \nlength(rhs0) = $(length(rhs0))\n"
    N, m, Ntst = size(coll)
    nbcoll = N * m
    # size of the periodic orbit problem counting the phase condition.
    Npo = length(coll) + 1 # We use this to tackle the case where size(J, 1) > Npo
    nⱼ = size(J, 1)
    δn =  nⱼ - Npo # this allows to compute the border side
    @assert δn >= 0
    @assert δn == dim "δn = $δn and dim = $dim should be equal!\nPass a proper COPCACHE."

    # matrix to contain the linear system for the external variables
    Jext = cop_cache.Jext
    @assert size(Jext, 1) == size(Jext, 2) == (Ntst+1)*N+1+δn "Error with matrix of external variables. Please report this issue on the website of BifurcationKit.\nδn = $δn\nsize(Jext) = $(size(Jext))\n(Ntst+1)*N+1+δn = $((Ntst+1)*N+1+δn)\n\n"
    𝒯 = eltype(coll)
    Iₙ = coll.cache.In

    rhs = condensation_of_parameters2!(cop_cache, coll, J, Iₙ, rhs0)
    Jcop = cop_cache.Jcoll

    if debug === true
        P = Matrix{𝒯}(LinearAlgebra.I(nⱼ))
        Jtmp = zeros(𝒯, nbcoll + δn + 1, nbcoll)
        Fₚ = lu(P); Jcop = Fₚ \ J; rhs = Fₚ \ rhs0
    end

    # we build the linear system for the external variables in Jext and rhs_ext
    rhs_ext = build_external_system!(Jext, Jcop, rhs, cop_cache.rhs_ext, Iₙ, Ntst, nbcoll, Npo, δn, N, m)

    if uselu
        F = LA.lu(Jext)
        sol_ext = F \ rhs_ext
    else
        # gaussian elimination plus backward substitution to invert Jext
        _gaussian_elimination_external_pivoted!(Jext, rhs_ext, N, Ntst, δn)
        sol_ext = _backward_substitution_pivoted(Jext, rhs_ext,cop_cache.sol_ext, N, Ntst, Val(dim))
    end

    return _solve_for_internal_variables(coll, Jcop, rhs, sol_ext, Val(dim))
end

"""
$(TYPEDSIGNATURES)

Copy the matrix J into 𝑱.
"""
@views function _copy_to_coll!(coll, 𝑱, J, ::Val{dim}) where {dim}
    nj = size(J, 1)
    N, m, Ntst = size(coll)
    nbcoll = N * m
    In = coll.cache.In
    rgᵢ = 1:(nbcoll+N)
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

function condensation_of_parameters2!(cop_cache::COPCACHE{dim}, 
                                coll::PeriodicOrbitOCollProblem, 
                                J, 
                                In, # identify
                                rhs0) where {dim}
    rhs = rhs0
    𝑱 = cop_cache.Jcoll
    α_values = cop_cache.α_values
    n𝑱 = size(𝑱, 1)
    # for newton (dim == 0), we copy the matrix with a fast method TODO REMOVE. Otherwise (dim>0), the cache already contains the matrix J
    if true#dim === 0
        _copy_to_coll!(coll, 𝑱, J, Val(dim))
    end

    N, m, Ntst = size(coll)
    nbcoll = N * m
    Npo = length(coll) + 1

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

            # Swap rows k and p if needed
            if iₚ != k
                @inbounds for jj in rglast
                    𝑱[k, jj], 𝑱[iₚ, jj] = 𝑱[iₚ, jj], 𝑱[k, jj]
                end
                rhs[k], rhs[iₚ] = rhs[iₚ], rhs[k]
            end
            ##########

            𝑱ₖ = 𝑱[k, colₖ]
            inv𝑱 = inv(𝑱ₖ)

            # scale column
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

            # last column
            @inbounds α = 𝑱[end, colₖ]
            @inbounds β = 𝑱[end-1, colₖ]
            @inbounds γ = 𝑱[end-2, colₖ]

            @inbounds for j in rglast
                    𝑱[end,   j] -= α * 𝑱[k, j] # EVERYTHING IS HERE 6%
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

            # precompute α for all i
            @inbounds for i=k+1:last(rgₖ)
                α_values[i - k] = 𝑱[i, colₖ]
                rhs[i] -= α_values[i - k] * rhsk
            end

            # Loop over j first, then i
            for j = Iterators.flatten((rgᵢ, Npo:Npo+dim))
                @inbounds 𝑱kj = 𝑱[k, j]
                for i = k+1:last(rgₖ)
                    @inbounds 𝑱[i, j] -= α_values[i - k] * 𝑱kj  # EVERYTHING IS HERE 10%
                end
            end
            𝑱[k, colₖ] = 𝑱ₖ
        end
        
        rgₖ = rgₖ  .+ nbcoll
        rgᵢ = rgᵢ .+ nbcoll
    end

    return rhs
end

@views function build_external_system!(Jext::Matrix{𝒯},
                                       Jcond::Matrix{𝒯},
                                       rhs::Vector{𝒯},
                                       rhs_ext::Vector{𝒯},
                                       In,
                                       Ntst::Int,
                                       nbcoll::Int,
                                       Npo::Int,
                                       δn::Int,
                                       N::Int,
                                       m::Int) where {𝒯}
    r1 = 1:N
    r2 = N*(m-1)+1:(m*N)
    rN = 1:N

    # building the external variables
    fill!(Jext, 0)
    Jext[end-δn-N:end-δn-1, end-δn-N:end-δn-1] .= In
    Jext[end-δn-N:end-δn-1, 1:N] .= (-1) .* In
    Jext[end-δn:end, end-δn:end] .= Jcond[end-δn:end, end-δn:end]

    # we solve for the external unknowns
    for _ in 1:Ntst
        Jext[rN, rN] .= Jcond[r2, r1]
        Jext[rN, rN .+ N] .= Jcond[r2, r1 .+ nbcoll]

        Jext[rN, end-δn:end] .= Jcond[r2, Npo:(Npo+δn)]

        Jext[end-δn:end, rN] .= Jcond[Npo:(Npo+δn), r1]
        Jext[end-δn:end, rN .+ N] .= Jcond[Npo:(Npo+δn), r1 .+ nbcoll]

        rhs_ext[rN] .= rhs[r2]

        r1 = r1 .+ nbcoll
        r2 = r2 .+ nbcoll
        rN = rN .+ N
    end
    rhs_ext[rN] .= rhs[r1]
    rhs_ext[end-δn:end] .= rhs[end-δn:end]
    return rhs_ext
end

@views function _solve_for_internal_variables(coll::PeriodicOrbitOCollProblem,
                                         Jcond,
                                         rhs::Vector{𝒯}, 
                                         sol_ext, 
                                         ::Val{δn}) where {𝒯, δn}
    N, m, Ntst = size(coll)
    nbcoll = N * m

    # solver for the internal unknowns
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

        # rhs_tmp = rhs[rsol] - left_part * sol_ext[rN] - right_part * sol_ext[rN .+ N] - ΔT * Jcond[r1, end]
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

# ~/.julia/juliaup/julia-1.10.2+0.aarch64.apple.darwin14/share/julia/stdlib/v1.10/LinearAlgebra/src/lu.jl:134
@inbounds function _gaussian_elimination_external_pivoted!(J::AbstractMatrix{𝒯},
                                                           rhs,
                                                           n::Int,
                                                           Ntst::Int,
                                                           δn::Int ) where 𝒯
    st = 0
    nⱼ = size(J, 1)
    maxρ = zero(𝒯)
    iₚ = 0
    info = 0

    for nt = 1:Ntst-1
        for i = st+1:st+n

            # find the pivot
            iₚ = i
            Jmax = abs(J[i, i+n])
            for l = i:i+2n
                absl = abs(J[l, i+n])
                if absl > Jmax
                    iₚ = l
                    Jmax = absl
                end
            end

            if !iszero(J[iₚ, i+n])
                if iₚ != i
                    # rg = 1:nⱼ
                    rg = Iterators.flatten((1:n, st+1+n:st+3n, nⱼ-δn:nⱼ))
                    # swap rows
                    for j in rg
                        J[i, j], J[iₚ, j] = J[iₚ, j], J[i, j]
                    end
                    rhs[i], rhs[iₚ] = rhs[iₚ], rhs[i]
                end

                invpivot = inv(J[i, i+n])

                rg = i+1:nⱼ
                rg = Iterators.flatten((i+1:st+2n, nⱼ-δn:nⱼ))
                rhsi = rhs[i]
                @inbounds for l in rg
                    ρ = J[l, i+n] * invpivot
                    rhs[l] -= rhsi * ρ
                    # rg = 1:nⱼ
                    rgₖ = Iterators.flatten((1:n, st+1+n:st+3n, nⱼ-δn:nⱼ))
                    @inbounds for jₖ in rgₖ
                        J[l, jₖ] -= J[i, jₖ] * ρ
                    end
                end
            else
                info = i
            end
        end
        st += n
    end
    return J, rhs
end

@views function _backward_substitution_pivoted(Jext::Matrix{𝒯},
                                                rhs_ext,
                                                sol_ext,
                                                n::Int,
                                                Ntst::Int,
                                                ::Val{δn}) where {𝒯, δn}
    Jext_gauss = hcat(Jext[end-2n-δn:end,1:n], Jext[end-2n-δn:end, end-n-δn:end])
    rhs_ext_gauss = rhs_ext[end-2n-δn:end]
    sol_ext_gauss = Jext_gauss \ rhs_ext_gauss

    # backward substitution
    x₀ = sol_ext_gauss[1:n]
    xₘ = sol_ext_gauss[(1:n) .+ n]
    ΔT = sol_ext_gauss[end-δn]
    Δp = sol_ext_gauss[end]

    sol_ext[1:n] .= x₀
    sol_ext[end-δn-n:end] .= sol_ext_gauss[end-δn-n:end]

    rhs_tmp = zeros(𝒯, n)
    st = (Ntst-2)*n
    for iₜ in Ntst-1:-1:1
        if δn == 0
            rhs_tmp .= @. rhs_ext[(1:n) .+ st] - ΔT * Jext[(1:n) .+ st, end]
        elseif δn == 1
            rhs_tmp .= @. rhs_ext[(1:n) .+ st] -
                            ΔT * Jext[(1:n) .+ st, end-1] -
                            Δp * Jext[(1:n) .+ st, end] 
        elseif δn == 2
            rhs_tmp .= @. rhs_ext[(1:n) .+ st] -
                            sol_ext_gauss[end-2] * Jext[(1:n) .+ st, end-2] -
                            sol_ext_gauss[end-1] * Jext[(1:n) .+ st, end-1] -
                            sol_ext_gauss[end]   * Jext[(1:n) .+ st, end] 
        else
            throw("Case not handled")
        end
        LA.mul!(rhs_tmp, Jext[(1:n) .+ st, 1:n], x₀, -1, 1)
        LA.mul!(rhs_tmp, Jext[(1:n) .+ st, (1:n) .+ st .+ 2n], sol_ext[(1:n) .+ st .+ 2n], -1, 1)
        LA.ldiv!(sol_ext[(1:n) .+ st .+ n], LA.UpperTriangular(Jext[(1:n) .+ st, (1:n) .+ st .+ n]), rhs_tmp)
        st -= n
    end
    return sol_ext
end
####################################################################################################
function (ls::COPLS)(Jc, rhs)
    res = solve_cop(ls.cache.coll, Jc, rhs, ls.cache)
    return res, true, 1
end

# There is see https://github.com/JuliaLang/julia/pull/56657, which might improve the performance considerably. Unfortunately this seemed to increase allocations in certain cases, and I haven't got to looking into these.
function _fast_copy_bordered!(x, y)
    for (xcol, ycol) ∈ zip(eachcol(x), eachcol(y))
        @views xcol[begin:end - 1] .= ycol
    end
    x
end

# solve in dX, dl
# ┌                           ┐┌  ┐   ┌   ┐
# │ (shift⋅I + J)     dR      ││dX│ = │ R │
# │   ξu * dz.u'   ξp * dz.p  ││dl│   │ n │
# └                           ┘└  ┘   └   ┘
function (ls::COPBLS)(Jc, dR,
                      dzu, dzp::𝒯, 
                      R::AbstractVecOrMat, n::𝒯,
                      ξu::𝒯 = one(𝒯), ξp::𝒯 = one(𝒯);
                      shift::Ts = nothing, 
                      Mass::Tm = LinearAlgebra.I,
                      dotp = nothing,
                      applyξu! = nothing)  where {𝒯 <: Number, Ts, Tm}
    if isnothing(shift)
        A = Jc
    else
        A = Jc + shift * Mass
    end
    rhs = vcat(R, n)
    coll = ls.cache.coll

    # the copy is necessary in all cases except DenseAnalyticalInplace()
    # which is full inplace. Indeed, in this case pointer(Jc) == pointer(ls.J)
    if ~(coll.jacobian == DenseAnalyticalInplace())
        # we improve on the following situation.
        # ls.J[begin:end-1,begin:end-1] .= A 
        # is quite slow, it would be 8x faster to do ls.J .= A
        # hence we made the following function
        _fast_copy_bordered!(ls.J, A)
    end

    # why not use ls.cache.Jcoll? Because aliasing is dangerous
    ls.J[begin:end-1, end] .= dR
    ls.J[end, begin:end-1] .= conj.(dzu .* ξu)
    ls.J[end, end] = dzp * ξp

    # apply a linear operator to ξu
    if isnothing(applyξu!) == false
        applyξu!(@view(ls.J[end, begin:end-1]))
    end

    res = solve_cop(ls.cache.coll, ls.J, rhs, ls.cache)
    return (@view res[begin:end-1]), res[end], true, 1
end