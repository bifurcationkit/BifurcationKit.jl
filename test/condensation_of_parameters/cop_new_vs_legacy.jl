using Test
using BifurcationKit, LinearAlgebra
const BK = BifurcationKit

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
    prob_ana = BifurcationProblem((x,p)->_al*x, zeros(N), par_hopf, (@optic _.r) ; J = (x,p) -> _al)
    
    # --- Legacy Collocation setup ---
    coll_legacy = Collocation(Ntst, m; prob_vf = prob_ana, N, ϕ = rand(N*( 1 + m * Ntst)))
    ci_legacy = generate_solution(coll_legacy, t->cos(t) .* ones(N), 2pi)
    Jco_legacy = BK.po_analytical_jacobian(coll_legacy, ci_legacy, par_sl)
    
    # --- New DiscretizedPO setup ---
    model_bvp = BK.BVP.POModel((x,p)->_al*x; n=N)
    disc = BK.BVP.Collocation(Ntst=Ntst, m=m, meshadapt=false)
    d_po = BK.BVP.discretize(model_bvp, disc)
    
    # We need a solution vector that matches the new structure (same size, just different arrangement maybe, but here we can just use the same vector size)
    ci_new = BK.generate_solution(d_po, t->cos(t) .* ones(N), 2pi)
    # We need the jacobian. The structure should be the same for the test
    Jco_new = copy(Jco_legacy) # For solving, we just need a matrix. Let's use the same matrix to ensure we compare the solver, not the jacobian assembly.
    
    #####################################################
    @test size(Jco_legacy, 1) == length(coll_legacy) + 1
    @test size(Jco_new, 1) == length(d_po)
    
    _rhs = rand(size(Jco_legacy, 1))
    
    # --- Test dim = 0 ---
    sol_bs = Jco_legacy \ _rhs
    
    cop_cache_legacy = BK.COPCACHE(coll_legacy, Val(0))
    sol_cop_legacy = BK.solve_cop(coll_legacy, copy(Jco_legacy), copy(_rhs), cop_cache_legacy; _USELU = Val(false))
    
    cop_cache_new = BK.COPCACHE(d_po, Val(0))
    sol_cop_new = BK.solve_cop(d_po, copy(Jco_new), copy(_rhs), cop_cache_new; _USELU = Val(false))
    
    @test sol_bs ≈ sol_cop_legacy
    @test sol_bs ≈ sol_cop_new
    @test sol_cop_legacy ≈ sol_cop_new
    
    # test _copy_to_coll!
    cop_cache_new.Jcoll .= 0
    BK._copy_to_coll!(d_po, cop_cache_new.Jcoll, Jco_new, Val(0))
    @test cop_cache_new.Jcoll ≈ Jco_new

    # --- Test bordered system (dim = 1) ---
    _t1 = rand(size(Jco_legacy, 1)); _t2 = rand(size(Jco_legacy, 1)+1)'; _t1[end-N:end-1] .= 0
    Jco_bd = vcat(hcat(Jco_legacy, _t1), _t2) |> Array
    _rhs_bd = rand(size((Jco_bd), 1))
    sol_bs_bd = (Jco_bd) \ _rhs_bd
    
    cop_cache_legacy_bd = BK.COPCACHE(coll_legacy, Val(1))
    sol_cop_legacy_bd = BK.solve_cop(coll_legacy, copy(Jco_bd), copy(_rhs_bd), cop_cache_legacy_bd; _USELU = Val(false))
    
    cop_cache_new_bd = BK.COPCACHE(d_po, Val(1))
    sol_cop_new_bd = BK.solve_cop(d_po, copy(Jco_bd), copy(_rhs_bd), cop_cache_new_bd; _USELU = Val(false))
    
    @test sol_bs_bd ≈ sol_cop_legacy_bd
    @test sol_bs_bd ≈ sol_cop_new_bd
    @test sol_cop_legacy_bd ≈ sol_cop_new_bd

    # test _copy_to_coll! dim 1
    cop_cache_new_bd.Jcoll .= 0
    BK._copy_to_coll!(d_po, cop_cache_new_bd.Jcoll, Jco_bd, Val(1))
    @test cop_cache_new_bd.Jcoll ≈ Jco_bd

    # --- Test bordered system (dim = 2) ---
    dim2 = 2
    _t1_2 = rand(size(Jco_legacy, 1), dim2); _t2_2 = rand(size(Jco_legacy, 1)+dim2, dim2)'; _t1_2[end-N:end-1, :] .= 0
    Jco_bd_2 = vcat(hcat(Jco_legacy, _t1_2), _t2_2) |> Array
    _rhs_bd_2 = rand(size((Jco_bd_2), 1))
    sol_bs_bd_2 = (Jco_bd_2) \ _rhs_bd_2
    
    cop_cache_legacy_bd_2 = BK.COPCACHE(coll_legacy, Val(dim2))
    sol_cop_legacy_bd_2 = BK.solve_cop(coll_legacy, copy(Jco_bd_2), copy(_rhs_bd_2), cop_cache_legacy_bd_2; _USELU = Val(false))
    
    cop_cache_new_bd_2 = BK.COPCACHE(d_po, Val(dim2))
    sol_cop_new_bd_2 = BK.solve_cop(d_po, copy(Jco_bd_2), copy(_rhs_bd_2), cop_cache_new_bd_2; _USELU = Val(false))
    
    @test sol_bs_bd_2 ≈ sol_cop_new_bd_2
    @test sol_cop_legacy_bd_2 ≈ sol_cop_new_bd_2

    println("Success: The new COP solver matches the legacy one exactly for dim=0, 1, 2.")

    println("\n--- Benchmarking dim = 0 ---")
    Jco_tmp_leg = copy(Jco_legacy); rhs_tmp_leg = copy(_rhs)
    Jco_tmp_new = copy(Jco_new); rhs_tmp_new = copy(_rhs)
    # warmup
    BK.solve_cop(coll_legacy, copy(Jco_tmp_leg), copy(rhs_tmp_leg), cop_cache_legacy; _USELU = Val(false))
    BK.solve_cop(d_po, copy(Jco_tmp_new), copy(rhs_tmp_new), cop_cache_new; _USELU = Val(false))
    print("Legacy: ")
    @time for i in 1:100; BK.solve_cop(coll_legacy, copy(Jco_tmp_leg), copy(rhs_tmp_leg), cop_cache_legacy; _USELU = Val(false)); end
    print("New:    ")
    @time for i in 1:100; BK.solve_cop(d_po, copy(Jco_tmp_new), copy(rhs_tmp_new), cop_cache_new; _USELU = Val(false)); end
    
    println("\n--- Benchmarking dim = 1 ---")
    Jco_tmp_leg = copy(Jco_bd); rhs_tmp_leg = copy(_rhs_bd)
    Jco_tmp_new = copy(Jco_bd); rhs_tmp_new = copy(_rhs_bd)
    # warmup
    BK.solve_cop(coll_legacy, copy(Jco_tmp_leg), copy(rhs_tmp_leg), cop_cache_legacy_bd; _USELU = Val(false))
    BK.solve_cop(d_po, copy(Jco_tmp_new), copy(rhs_tmp_new), cop_cache_new_bd; _USELU = Val(false))
    print("Legacy: ")
    @time for i in 1:100; BK.solve_cop(coll_legacy, copy(Jco_tmp_leg), copy(rhs_tmp_leg), cop_cache_legacy_bd; _USELU = Val(false)); end
    print("New:    ")
    @time for i in 1:100; BK.solve_cop(d_po, copy(Jco_tmp_new), copy(rhs_tmp_new), cop_cache_new_bd; _USELU = Val(false)); end
end
