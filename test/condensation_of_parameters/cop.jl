# using Revise, Plots
using Test
using BifurcationKit, LinearAlgebra
const BK = BifurcationKit

@views function condensation_of_parameters!(cop_cache::BK.COPCACHE{dim}, 
                                            coll::PeriodicOrbitOCollProblem, 
                                            J, 
                                            In, 
                                            rhs0::Vector) where {dim}
    N, m, Ntst = size(coll)
    n = N
    nbcoll = N * m
    Npo = length(coll) + 1
    nⱼ = size(J, 1)
    is_bordered = nⱼ == Npo
    δn =  nⱼ - Npo # this allows to compute the border side
    # δn = 0 for newton
    # δn = 1 for palc
    @assert δn >= 0
    @assert δn == dim "We found instead: δn = $δn == dim = $dim"

    𝒯 = eltype(coll)

    # cache to hold the factorized form of the matrix J
    Jcop = cop_cache.Jcoll
    # cache to hold the linear operator for the external variables
    Jext = cop_cache.Jext
    @assert size(Jext, 1) == size(Jext, 2) == Ntst*n+n+1+δn "Error with matrix of external variables. Please report this issue on the website of BifurcationKit. δn = $δn"

    Jcop[end, :] .= 0
    Jcop[:, end] .= 0
    Jcop[end, end] = J[end, end]

    # put periodic boundary condition
    Jcop[end-N-δn:end-1-δn, end-N-δn:end-1-δn] .= In
    Jcop[end-N-δn:end-1-δn, 1:N] .= (-1) .* In

    rg = 1:nbcoll
    rN = 1:N

    # the goal of the condensation of the parameters method is to remove the internal variables
    # by using gaussian elimination in each collocation block while removing the internal constraints
    # as well. 

    # recall that if F = lu(J) then
    # F.L * F.U = F.P * J
    # hence 𝐅𝐬⁻¹ = (P⁻¹ * L)⁻¹ = L⁻¹ * P
    # Now 𝐅𝐬 is with shape
    # ┌     ┐
    # │ A 0 │
    # │ c 1 │
    # └     ┘
    # This makes it easy to identify 𝐅𝐬⁻¹ which is also lower triangular by blocks. In particular c⁻¹ = c * A⁻¹, (computed with c' \ A)
    # Writing Jpo as
    # ┌       ┐
    # │ J  bⱼ │
    # │ cⱼ dⱼ │
    # └       ┘
    # we can identify 𝐅𝐬⁻¹⋅Jpo and the last row of this product, namely
    # c * A⁻¹ * J + cⱼ
    # last_row_𝐅𝐬⁻¹_analytical = zeros(𝒯, δn + 1, nⱼ) # last row of 𝐅𝐬⁻¹
    # last_row_𝐅𝐬 = zeros(𝒯, δn + 1, nⱼ) # last row of 𝐅𝐬

    (; blockⱼ,
        blockₙ,
        blockₙ₂,
        Lₜ,
        Uₜ,
        last_row_𝐅𝐬⁻¹_analytical,
        last_row_𝐅𝐬) = cop_cache
    
    rhs = zero(rhs0)
    p = zeros(Int, nbcoll + 1 + δn)
    pinv = zeros(Int, nbcoll + 1 + δn)

    d = zero(𝒯)
    for k in 1:Ntst
        blockⱼ[1:nbcoll, :] .= J[rg, rg .+ n]
        blockⱼ[nbcoll+1:(nbcoll + 1 + δn), :] .= J[Npo:(Npo+δn), rg .+ n]

        # the pivoting strategy is to ensure that the constraints 
        # get not mixed up with the collocation blocks
        F = lu!(blockⱼ, RowNonZero())
        @assert issuccess(F) "Failed LU factorization! Please report to the website of BifurcationKit."

        # get p .= F.p and pinv = invperm(p)
        _ipiv2perm!(p, F.ipiv, size(F, 1))
        _invperm!(pinv, p)

        @assert p[nbcoll+1] == nbcoll+1 "Pivoting strategy failed!! Please report to the website of BifurcationKit. You may try the default linear solver `defaultLS` as a backup."
        if dim > 0
            @assert p[nbcoll+2] == nbcoll+2 "Pivoting strategy failed!! Please report to the website of BifurcationKit. You may try the default linear solver `defaultLS` as a backup."
        end

        # Lₜ = LowerTriangular(F.L) # zero allocation?
        Lₜ.data .= blockⱼ[1:nbcoll, :]
        Uₜ.data .= Lₜ.data
        for i in axes(Lₜ, 1); Lₜ[i, i] = one(𝒯); end

        # we put the blocks in Jcop
        Jcop[rg, rg .+ N] .= Uₜ #UpperTriangular(F.factors[1:nbcoll, 1:nbcoll])

        # Jcop[rg, rN] .= P[rg, rg] \ J[rg, rN]
        # we have: P[rg, rg] = F.L[pinv[1:end-1-δn],:]
        # when δn = 0, we have blockₙ[1:nbcoll, 1:N] .= J[rg, rN][p_free,:]
        blockₙ[1:nbcoll, 1:N] .= J[rg[p[1:nbcoll]], rN]
        ldiv!(blockₙ₂, Lₜ, blockₙ)
        copyto!(Jcop[rg, rN], blockₙ₂)

        # last_row_𝐅𝐬[:, rg] .= F.L[pinv[end-δn:end], :] #!!! Allocates a lot !!!
        copyto!(last_row_𝐅𝐬[end, rg], F.factors[pinv[end], :])
        if dim == 1
            last_row_𝐅𝐬[end-1, rg] .= F.factors[pinv[end-δn], :]
        else
            # TODO!! We must improve this !! All allocations happens here
            last_row_𝐅𝐬[:, rg] .= F.L[pinv[end-δn:end], :]
            # last_row_𝐅𝐬[:, rg] .= F.factors[pinv[end-δn:end], :]
        end

        # condense RHS
        ldiv!(rhs[rg], Lₜ, rhs0[rg[p[1:nbcoll]]])

        # Jcop[end-δn:end, rg] .= -(last_row_𝐅𝐬[end-δn:end, rg] * Jcop[rg, rg]) .+ J[end-δn:end, rg]
        Jcop[end-δn:end, rg] .= J[end-δn:end, rg]
        mul!(Jcop[end-δn:end, rg], 
            last_row_𝐅𝐬[end-δn:end, rg], 
            Jcop[rg, rg], -1, 1)

        # ldiv!(Jcop[rg, end-δn:end] , Lₜ, F.P[1:end-1-δn,1:end-1-δn] * J[rg, end-δn:end])
        ldiv!(Jcop[rg, end-δn:end], 
                Lₜ, 
                J[rg[p[1:end-1-δn]], end-δn:end])

        ###
        # last_row_𝐅𝐬⁻¹_analytical[:, rg] .= -F.L[pinv[end-δn:end], :] / ( F.P'*F.L)[1:end-1-δn, :]
        LinearAlgebra._rdiv!(last_row_𝐅𝐬⁻¹_analytical[:, rg], 
                                last_row_𝐅𝐬[:, rg], 
                                Lₜ)
        last_row_𝐅𝐬⁻¹_analytical[:, rg] .*= -1
        ###

        if k>=2
            # correction = P[Npo, rg .- nbcoll]' * Jcop[rg .- nbcoll, rN]
            mul!(Jcop[end-δn:end, rN], 
                last_row_𝐅𝐬[:, rg .- nbcoll], 
                Jcop[rg .- nbcoll, rN], -1, 1)
        end

        rg = rg .+ nbcoll
        rN = rN .+ nbcoll
    end
    rhs[end-N-δn:end-1, :] .= rhs0[end-N-δn:end-1, :]
    return rhs
end

let
# ####################################################################################################
par_sl = (r = 0.1, μ = 0., ν = 1.0, c3 = 1.0)
par_hopf = (@set par_sl.r = 0.1)
# ####################################################################################################
Ntst = 80
m = 4
N = 3
#####################################################
_al = I + 10. .* rand(N, N)
# prob_ana = BifurcationProblem((x,p)->x, zeros(N), par_hopf, (@optic _.r) ; J = (x,p) -> I(N))
prob_ana = BifurcationProblem((x,p)->_al*x, zeros(N), par_hopf, (@optic _.r) ; J = (x,p) -> _al)
coll = PeriodicOrbitOCollProblem(Ntst, m; 
                                    prob_vf = prob_ana, 
                                    N,
                                    ϕ = rand(N*( 1 + m * Ntst)), 
                                    xπ = rand(N*( 1 + m * Ntst)))
_ci = generate_solution(coll, t->cos(t) .* ones(N), 2pi);
#####################################################
Jco = BK.analytical_jacobian(coll, _ci, par_sl);
@test size(Jco, 1) == length(coll) + 1
#####################################################
_rhs = rand(size(Jco, 1))
sol_bs = Jco \ _rhs;

Jco_tmp = zero(Jco)
Jext_tmp= zeros(Ntst*N+N+1, Ntst*N+N+1)
cop_cache = BK.COPCACHE(coll)
sol_cop = BK.solve_cop(coll, copy(Jco), copy(_rhs), cop_cache; _USELU = Val(true));
@test sol_bs ≈ sol_cop
cop_cache = BK.COPCACHE(coll)
@test BK._getdim(cop_cache) == 0
sol_cop = BK.solve_cop(coll, copy(Jco), copy(_rhs), cop_cache; _USELU = Val(false));
@test sol_bs ≈ sol_cop
# test _copy_to_coll!
cop_cache.Jcoll .= 0
BK._copy_to_coll!(coll, cop_cache.Jcoll, Jco, Val(0)) # 10.833 μs (1 allocation: 448 bytes)
@test cop_cache.Jcoll ≈ Jco

# test case of a bordered system to test PALC like linear problem
_t1 = rand(size(Jco, 1)); _t2 = rand(size(Jco, 1)+1)'; _t1[end-N:end-1] .= 0
Jco_bd = vcat(hcat(Jco, _t1), _t2) |> Array
_rhs_bd = rand(size((Jco_bd), 1))
sol_bs_bd = (Jco_bd) \ _rhs_bd;
cop_cache = BK.COPCACHE(coll, Val(1))
@test BK._getdim(cop_cache) == 1
cop_cache.Jcoll .= Jco_bd # the cache is updated inplace in normal use
sol_cop_bd = BK.solve_cop(coll, copy(Jco_bd), copy(_rhs_bd), cop_cache; _USELU = Val(false));
@test sol_bs_bd ≈ sol_cop_bd
# test _copy_to_coll!
cop_cache.Jcoll .= 0
BK._copy_to_coll!(coll, cop_cache.Jcoll, Jco_bd, Val(1)) # 11.334 μs (1 allocation: 448 bytes)
@test cop_cache.Jcoll ≈ Jco_bd

# test case of a bordered system with dim = 2
dim = 2
_t1 = rand(size(Jco, dim), dim); _t2 = rand(size(Jco, 1)+dim, dim)'; _t1[end-N:end-1, :] .= 0
Jco_bd = vcat(hcat(Jco, _t1), _t2) |> Array
_rhs_bd = rand(size((Jco_bd), 1))
sol_bs_bd = (Jco_bd) \ _rhs_bd;
cop_cache = BK.COPCACHE(coll, Val(dim))
@test BK._getdim(cop_cache) == 2
cop_cache.Jcoll .= Jco_bd # the cache is updated inplace in normal use
sol_cop_bd = BK.solve_cop(coll, copy(Jco_bd), copy(_rhs_bd), cop_cache; _USELU = Val(false));
@test sol_bs_bd ≈ sol_cop_bd
# test _copy_to_coll!
cop_cache.Jcoll .= 0
BK._copy_to_coll!(coll, cop_cache.Jcoll, Jco_bd, Val(2)) # 12.041 μs (1 allocation: 448 bytes)
@test cop_cache.Jcoll ≈ Jco_bd
end
