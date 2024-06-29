# using Revise, Plots
using Test
using BifurcationKit, ForwardDiff, LinearAlgebra
const BK = BifurcationKit
# ####################################################################################################
par_sl = (r = 0.1, μ = 0., ν = 1.0, c3 = 1.0)
par_hopf = (@set par_sl.r = 0.1)
# ####################################################################################################
Ntst = 80
m = 4
N = 3
#####################################################
const _al = I + 10. .*rand(N,N)
# prob_ana = BifurcationProblem((x,p)->x, zeros(N), par_hopf, (@lens _.r) ; J = (x,p) -> I(N))
prob_ana = BifurcationProblem((x,p)->_al*x, zeros(N), par_hopf, (@lens _.r) ; J = (x,p) -> _al)
prob_col = PeriodicOrbitOCollProblem(Ntst, m; prob_vf = prob_ana, N = N, ϕ = rand(N*( 1 + m * Ntst)), xπ = rand(N*( 1 + m * Ntst)))
_ci = generate_solution(prob_col, t->cos(t) .* ones(N), 2pi);
#####################################################
Jcofd = ForwardDiff.jacobian(z->prob_col(z, par_sl), _ci);
Jco = BK.analytical_jacobian(prob_col, _ci, par_sl);

_rhs = rand(size(Jco, 1))
sol_bs = Jco \ _rhs;

using BifurcationKit
Jco_tmp = zero(Jco)
Jext_tmp= zeros(Ntst*N+N+1, Ntst*N+N+1)
cop_cache = BK.COPCACHE(prob_col)
sol_cop = BK.solve_cop(prob_col, Jco, _rhs, cop_cache; _USELU = true);
@test sol_bs ≈ sol_cop

# test case of a bordered system to test PALC like linear problem
_t1 = rand(size(Jco, 1)); _t2 = rand(size(Jco, 1)+1)'; _t1[end-N:end-1] .= 0
Jco_bd = vcat(hcat(Jco, _t1),_t2) |> Array
_rhs_bd = rand(size((Jco_bd), 1))
sol_bs_bd = (Jco_bd) \ _rhs_bd;
cop_cache_bd = BK.COPCACHE(prob_col, 1)
sol_cop_bd = BK.solve_cop(prob_col, Jco_bd, _rhs_bd, cop_cache_bd; _USELU = false);
@test sol_bs_bd ≈ sol_cop_bd