# using Revise, Plots
using Test
using BifurcationKit, LinearAlgebra
const BK = BifurcationKit
# ####################################################################################################
par_sl = (r = 0.1, μ = 0., ν = 1.0, c3 = 1.0)
par_hopf = (@set par_sl.r = 0.1)
# ####################################################################################################
Ntst = 80
m = 4
N = 3
#####################################################
const _al = I + 10. .* rand(N, N)
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

_rhs = rand(size(Jco, 1))
sol_bs = Jco \ _rhs;

Jco_tmp = zero(Jco)
Jext_tmp= zeros(Ntst*N+N+1, Ntst*N+N+1)
cop_cache = BK.COPCACHE(coll)
sol_cop = BK.solve_cop(coll, copy(Jco), copy(_rhs), cop_cache; _USELU = Val(true));
@test sol_bs ≈ sol_cop

# test case of a bordered system to test PALC like linear problem
_t1 = rand(size(Jco, 1)); _t2 = rand(size(Jco, 1)+1)'; _t1[end-N:end-1] .= 0
Jco_bd = vcat(hcat(Jco, _t1),_t2) |> Array
_rhs_bd = rand(size((Jco_bd), 1))
sol_bs_bd = (Jco_bd) \ _rhs_bd;
cop_cache_bd = BK.COPCACHE(prob_col, Val(1))
sol_cop_bd = BK.solve_cop(prob_col, Jco_bd, _rhs_bd, cop_cache_bd; _USELU = Val(false));
@test sol_bs_bd ≈ sol_cop_bd