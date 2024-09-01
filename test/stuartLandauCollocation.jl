# using Revise, AbbreviatedStackTraces, Plots
using Test
using BifurcationKit, LinearAlgebra, ForwardDiff, SparseArrays
const BK = BifurcationKit
##################################################################
# The goal of these tests is to test all combinations of options
##################################################################
# no allocation
function Fsl!(f, u, p, t = 0)
    (;r, μ, ν, c3) = p
    u1 = u[1]
    u2 = u[2]
    ua = u1^2 + u2^2
    f[1] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2)
    f[2] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1)
    return f
end
####################################################################################################
par_sl = (r = 0.1, μ = 0., ν = 1.0, c3 = 1.0)
u0 = [.001, .001]
par_hopf = (@set par_sl.r = 0.1)
probsl = BifurcationProblem(Fsl!, u0, par_hopf, (@optic _.r))
probsl_ip = BifurcationProblem(Fsl!, u0, par_hopf, (@optic _.r), inplace = true)
####################################################################################################
# continuation, Hopf bifurcation point detection
optconteq = ContinuationPar(ds = -0.01, detect_bifurcation = 3, p_min = -0.5, n_inversion = 8)
br = continuation(probsl, PALC(), optconteq)
####################################################################################################
# test hopf predictor
hp = get_normal_form(br, 1)
pred = predictor(hp, 0.1)
@test pred.orbit(0)[1] ≈ sqrt(0.1/1)
####################################################################################################
Ntst = 4
m = 4
N = 3
const Mf = rand(N, N)
prob1 = BK.BifurcationProblem((x,p) -> Mf * x.^2, zeros(1), nothing)
prob_col = BK.PeriodicOrbitOCollProblem(Ntst, m, prob_vf = prob1, N = N, ϕ = ones(N * ( 1 + m * Ntst)), xπ = zeros(N * ( 1 + m * Ntst)))
size(prob_col)
length(prob_col)
BK.get_times(prob_col)
BK.get_max_time_step(prob_col)
size(prob_col.mesh_cache)
BK.update_mesh!(prob_col, prob_col.mesh_cache.τs)
PeriodicOrbitOCollProblem(10, 2) |> BK.get_mesh_size
BK.get_Ls(prob_col)
show(prob_col)

_orbit(t) = [cos(2pi * t), 0, 0] * sqrt(par_sl.r / par_sl.c3)
_ci = BK.generate_solution(prob_col, _orbit, 1.)
BK.get_periodic_orbit(prob_col, _ci, par_sl)
BK.getmaximum(prob_col, _ci, par_sl)
@test BK.∂(sin, 2)(0.) == 0
prob_col(_ci, par_sl) #|> scatter
BK.get_time_slices(prob_col, _ci)

# interpolate solution
sol = BK.POSolution(prob_col, _ci)
sol(rand())
####################################################################################################
prob_col = PeriodicOrbitOCollProblem(200, 5, prob_vf = probsl, N = 1000)
_ci = BK.generate_solution(prob_col, t -> cos(t) .* ones(1000), 2pi)
BK.get_times(prob_col)
sol = BK.POSolution(prob_col, _ci)
sol(0.1)
####################################################################################################
# test precision of phase condition, it must work for non uniform mesh
# recall that it is 1/T int(f,g')
@views function phaseCond(pb::PeriodicOrbitOCollProblem, u, v)
    𝒯 = eltype(u)
    phase = zero(𝒯)

    uc = BK.get_time_slices(pb, u)
    vc = BK.get_time_slices(pb, v)

    n, m, Ntst = size(pb)

    T = BK.getperiod(pb, u, nothing)

    guj = zeros(𝒯, n, m)
    uj  = zeros(𝒯, n, m+1)

    gvj = zeros(𝒯, n, m)
    vj  = zeros(𝒯, n, m+1)

    L, ∂L = BK.get_Ls(pb.mesh_cache)
    ω = pb.mesh_cache.gauss_weight

    rg = UnitRange(1, m+1)
    @inbounds for j in 1:Ntst
        uj .= uc[:, rg]
        vj .= vc[:, rg]
        mul!(guj, uj, L)
        mul!(gvj, vj, ∂L)
        @inbounds for l in 1:m
            # for mul!(gvj, vj, L')
            # phase += dot(guj[:, l], gvj[:, l]) * ω[l] * (mesh[j+1] - mesh[j]) / 2
            phase += dot(guj[:, l], gvj[:, l]) * ω[l]
        end
        rg = rg .+ m
    end
    return phase / T
end

let
    for Ntst in 2:10:100
        # @info "Ntst" Ntst
        prob_col = PeriodicOrbitOCollProblem(Ntst, 10, prob_vf = probsl, N = 1)
        # test non uniform mesh
        BK.update_mesh!(prob_col, sort(vcat(0,rand(Ntst-1),1)))

        _ci1 = BK.generate_solution(prob_col, t -> [1], 2pi)
        _ci2 = BK.generate_solution(prob_col, t -> [t], 2pi)
        @test phaseCond(prob_col, _ci1, _ci2) ≈ 1 atol = 1e-10
        # @info phaseCond(prob_col, _ci1, _ci2)/pi

        _ci1 = BK.generate_solution(prob_col, t -> [cos(t)], 2pi)
        _ci2 = BK.generate_solution(prob_col, t -> [sin(t)], 2pi)
        @test phaseCond(prob_col, _ci1, _ci2) ≈ 1/2 atol = 2e-6
        # @info phaseCond(prob_col, _ci1, _ci2)/pi-1

        _ci1 = BK.generate_solution(prob_col, t -> [cos(t)], 2pi)
        _ci2 = BK.generate_solution(prob_col, t -> [cos(t)], 2pi)
        @test phaseCond(prob_col, _ci1, _ci2) / pi ≈ 0 atol = 1e-11
        # @info phaseCond(prob_col, _ci1, _ci2) / pi

        _ci1 = BK.generate_solution(prob_col, t -> [cos(t)], 2pi)
        _ci2 = BK.generate_solution(prob_col, t -> [t], 2pi)
        @test phaseCond(prob_col, _ci1, _ci2) / pi ≈ 0 atol = 1e-5
        # @info phaseCond(prob_col, _ci1, _ci2) / pi
    end
end


prob_col = PeriodicOrbitOCollProblem(22, 10, prob_vf = probsl, N = 1)
_ci1 = BK.generate_solution(prob_col, t -> [cos(2pi*t)], 1)
_ci2 = BK.generate_solution(prob_col, t -> [cos(2pi*t)], 1)
@test BK.∫(prob_col, BK.get_time_slices(prob_col, _ci1), BK.get_time_slices(prob_col, _ci2)) ≈ 0.5

prob_col = PeriodicOrbitOCollProblem(22, 10, prob_vf = probsl, N = 1)
_ci1 = BK.generate_solution(prob_col, t -> [cos(2pi*t)], 3)
_ci2 = BK.generate_solution(prob_col, t -> [cos(2pi*t)], 3)
@test BK.∫(prob_col, BK.get_time_slices(prob_col, _ci1), BK.get_time_slices(prob_col, _ci2), 3) ≈ 3/2
@test BK.∫(prob_col, _ci1, _ci2, 3) ≈ 3/2 # test vector form
####################################################################################################
Ntst = 50
m = 4
N = 2
prob_col = BK.PeriodicOrbitOCollProblem(Ntst, m; 
                                prob_vf = probsl,
                                N = 2,
                                ϕ = zeros(N*( 1 + m * Ntst)),
                                xπ = zeros(N*( 1 + m * Ntst)))
prob_col.ϕ[2] = 1 #phase condition

_orbit(t) = [cos(t), sin(t)] * sqrt(par_sl.r/par_sl.c3)
_ci = BK.generate_solution(prob_col, _orbit, 2pi)
prob_col(_ci, par_sl)
@test prob_col(_ci, par_sl)[1:end-1] |> norminf < 1e-7

prob_coll_ip = @set prob_col.prob_vf = probsl_ip

@time prob_col(_ci, par_sl);
@time prob_coll_ip(_ci, par_sl);

# test precision of generated solution
_sol = BK.get_periodic_orbit(prob_col, _ci, nothing)
for (i, t) in pairs(_sol.t)
    @test _sol.u[:, i] ≈ _orbit(t)
end

args = (
    plot_solution = (x,p; k...) -> begin
        outt = get_periodic_orbit(prob_col, x, p)
        plot!(vec(outt.t), outt.u[1, :]; k...)
    end,
    finalise_solution = (z, tau, step, contResult; k...) -> begin
        return true
    end,)

optcontpo = ContinuationPar(optconteq; detect_bifurcation = 2, tol_stability = 1e-7)
@reset optcontpo.ds = -0.01
@reset optcontpo.newton_options.verbose = false

prob_col2 = (@set prob_coll_ip.prob_vf.params = par_sl)
@reset prob_col2.jacobian = BK.AutoDiffDense()
sol_po = newton(prob_col2, _ci, optcontpo.newton_options);

# test solution
solc = BK.POSolution(prob_col2, sol_po.u)
# plot([t for t in LinRange(0,2pi,100)], [solc(t)[1] for t in LinRange(0,2pi,100)])
let
    mesh = BK.getmesh(prob_col2)
    solpo = get_periodic_orbit(prob_col2, sol_po.u, nothing)
    for (i, t) in pairs(solpo.t)
        @test solc(t) ≈ solpo.u[:, i]
    end
end

# 0.131970 seconds (32.51 k allocations: 309.737 MiB, 14.97% gc time)
@reset prob_col2.update_section_every_step = 1
br_po = @time continuation(prob_col2, _ci, PALC(tangent = Bordered()), optcontpo;
    verbosity = 0, plot = false,
    args...,
    );

br_po = @time continuation(prob_col2, _ci, PALC(tangent = Bordered()), optcontpo;
    verbosity = 0, plot = false,
    args...,
    linear_algo  = COPBLS(),
    );

newton(prob_col2, _ci, NewtonPar())
newton(prob_col2, _ci, NewtonPar(linsolver = COPLS()))
####################################################################################################
# test analytical jacobian
Ntst = 10
m = 3
N = 4
nullvf(x,p) = zero(x)
prob0 = BifurcationProblem(nullvf, zeros(N), par_hopf, (@optic _.r))
prob_col = BK.PeriodicOrbitOCollProblem(Ntst, m; prob_vf = prob0, N = N, ϕ = rand(N*( 1 + m * Ntst)), xπ = rand(N*( 1 + m * Ntst)))
_ci = BK.generate_solution(prob_col, t->cos(t) .* ones(N), 2pi);
prob_col(_ci, par_sl);
Jcofd = ForwardDiff.jacobian(z->prob_col(z, par_sl), _ci);
D = @time BK.analytical_jacobian(prob_col, _ci, par_sl); #0.000186 seconds (461 allocations: 164.859 KiB)
@test norminf(Jcofd - D) < 1e-15

# same but with linear vector field
Ntst = 140
m = 4
N = 5
const _al = I(N) + 0.1 .*rand(N,N)
idvf(x,p) = _al*x
prob_ana = BifurcationProblem(idvf, zeros(N), par_hopf, (@optic _.r) ; J = (x,p) -> _al)
prob_col = BK.PeriodicOrbitOCollProblem(Ntst, m; prob_vf = prob_ana, N = N, ϕ = rand(N*( 1 + m * Ntst)), xπ = rand(N*( 1 + m * Ntst)))
_ci = BK.generate_solution(prob_col, t->cos(t) .* ones(N), 2pi);
Jcofd = ForwardDiff.jacobian(z -> prob_col(z, par_sl), _ci);
Jco = BK.analytical_jacobian(prob_col, _ci, par_sl); # 0.004388 seconds (573 allocations: 60.124 MiB)
@test norminf(Jcofd - Jco) < 1e-15

# same but with Stuart-Landau vector field
N = 2
@assert N == 2 "must be the dimension of the SL"
Ntst = 3
m = 2
prob_col = BK.PeriodicOrbitOCollProblem(Ntst, m; prob_vf = probsl, N = N, ϕ = rand(N*( 1 + m * Ntst)), xπ = rand(N*( 1 + m * Ntst)))
_ci = BK.generate_solution(prob_col, t->cos(t) .* ones(N), 2pi);
Jcofd = ForwardDiff.jacobian(z->prob_col(z, par_sl), _ci);
Jco = @time BK.analytical_jacobian(prob_col, _ci, par_sl);
@test norminf(Jcofd - Jco) < 1e-14
BK.analytical_jacobian(prob_col, _ci, par_sl; _transpose = true, ρF = 1.);
# test for the case of sparse arrays
# jacobian using BlockArray
const _asp = sparse(I(N) + 0.1 .*sprand(N,N,0.1))
prob_ana =       BifurcationProblem((x,p)->_asp*x, zeros(N), par_hopf, (@optic _.r) ; J = (x,p) -> _asp)
prob_ana_dense = BifurcationProblem((x,p)->_asp*x, zeros(N), par_hopf, (@optic _.r) ; J = (x,p) -> Array(_asp))
prob_col_dense = BK.PeriodicOrbitOCollProblem(Ntst, m; prob_vf = prob_ana_dense, N = N, ϕ = rand(N*( 1 + m * Ntst)), xπ = rand(N*( 1 + m * Ntst)))
prob_col       = BK.PeriodicOrbitOCollProblem(Ntst, m; prob_vf = prob_ana,       N = N, ϕ = copy(prob_col_dense.ϕ), xπ = copy(prob_col_dense.xπ))
_ci = BK.generate_solution(prob_col, t->cos(t) .* ones(N), 2pi);
Jco_sp = BK.analytical_jacobian_sparse(prob_col, _ci, par_sl);
Jco = BK.analytical_jacobian(prob_col_dense, _ci, par_sl);
@assert norminf(Jco - Array(Jco_sp)) == 0
Jco2 = copy(Jco) |> sparse;
Jco2 .= 0
_indx = BifurcationKit.get_blocks(prob_col, Jco2);
@time BifurcationKit.jacobian_poocoll_sparse_indx!(prob_col, Jco2, _ci, par_sl, _indx);
@test norminf(Jco - Jco2) < 1e-14
####################################################################################################
# test Hopf aBS
let
    for jacPO in (BK.AutoDiffDense(), BK.DenseAnalytical(), BK.FullSparse())
        useGEV = jacPO in (BK.AutoDiffDense(), BK.DenseAnalytical())

        br_po_gev = continuation(br, 1, (@set ContinuationPar(optcontpo; ds = 0.01, save_sol_every_step = 1, max_steps = 10, p_max=0.8).newton_options.verbose = false),
            PeriodicOrbitOCollProblem(20, 5; jacobian = jacPO, update_section_every_step = 1);
            δp = 0.1,
            usedeflation = true,
            eigsolver = useGEV ? BK.FloquetCollGEV(DefaultEig(),(20*5+1)*2,2) : BK.FloquetColl(),
            )

        br_po = continuation(br, 1, (@set ContinuationPar(optcontpo; ds = 0.01, save_sol_every_step=1, max_steps = 10, p_max=0.8).newton_options.verbose = false),
            PeriodicOrbitOCollProblem(20, 5; jacobian = jacPO, update_section_every_step = 1);
            δp = 0.1,
            usedeflation = true,
            eigsolver = BK.FloquetColl(),
            )

        # test mesh adaptation
        br_po_adapt = continuation(br, 1, (@set ContinuationPar(optcontpo; ds = 0.01, save_sol_every_step=1, max_steps = 10, p_max=0.8).newton_options.verbose = false),
            PeriodicOrbitOCollProblem(20, 5; jacobian = jacPO, update_section_every_step = 1, meshadapt = true);
            δp = 0.1,
            usedeflation = true,
            eigsolver = BK.FloquetColl(),
            )

        # we test that the 2 methods give the same floquet exponents
        for i=1:length(br_po)-1
            @test BK.eigenvals(br_po, i) ≈ BK.eigenvals(br_po_gev, i)
        end
    end
end

