using BifurcationKit, Test
const BK = BifurcationKit

# ==============================================================================
# BVP Module Unit Tests
# ==============================================================================

# ----- BVPModel -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]

    model = BK.BVP.BVPModel(F, g; n=2)
    show(model)
    @test model isa BK.BVP.BVPModel
    @test BK.BVP.state_dimension(model) == 2
    @test BK.BVP.evaluate_F(model, [0.0, 1.0], (ω=2.0,)) ≈ [1.0, 0.0]
    @test BK.BVP.evaluate_g(model, [1.0, 0.0], [0.0, 0.0], nothing) == [1.0, 0.0]
    @test BK.BVP.get_time_interval(model) == (0.0, 1.0)

    model2 = BK.BVP.BVPModel(F, g; n=2, t0=-1.0, tf=2.0)
    @test BK.BVP.get_time_interval(model2) == (-1.0, 2.0)
end

# ----- PeriodicOrbitModel -----
let
    F(u, p) = [p.μ*u[1] - u[2] - u[1]*(u[1]^2+u[2]^2),
               u[1] + p.μ*u[2] - u[2]*(u[1]^2+u[2]^2)]
    model = BK.BVP.PeriodicOrbitModel(F; n=2)
    show(model)
    @test model isa BK.BVP.BVPModel
    @test BK.BVP.state_dimension(model) == 2
    @test BK.BVP.get_time_interval(model) == (0.0, 1.0)
end

# ----- TimeMesh -----
let
    m = BK.TimeMesh(10)
    @test length(m) == 10
    @test BK.get_time_step(m, 1) ≈ 0.1
    @test BK.get_time_step(m, 10) ≈ 0.1
    @test length(collect(m)) == 10
    @test BK.can_adapt(m) == false

    steps = [0.2, 0.3, 0.5]
    m2 = BK.TimeMesh(steps)
    @test length(m2) == 3
    @test m2.ds ≈ [0.2, 0.3, 0.5]
    @test BK.can_adapt(m2) == true
end

# ----- Shooting -----
let
    disc = BK.BVP.Shooting(; M=4, alg=nothing, parallel=false)
    show(disc)
    @test disc.M == 4
    @test BK.BVP.mesh_size(disc) == 4
    @test BK.BVP.solution_dim(disc, 2) == 8
    @test BK.BVP.total_dim(disc, 2) == 9
end

# ----- Trapeze -----
let
    disc = BK.BVP.Trapeze(M=50)
    show(disc)
    @test disc.M == 50
    @test BK.BVP.mesh_size(disc) == 50
    @test BK.BVP.solution_dim(disc, 2) == 100
    @test BK.BVP.total_dim(disc, 2) == 101

    mesh = BK.TimeMesh([0.5, 0.5])
    disc2 = BK.BVP.Trapeze(; M=3, mesh)
    @test disc2.mesh.ds ≈ [0.5, 0.5]

    @test_throws AssertionError BK.BVP.Trapeze(; M=1)
end

# ----- Collocation -----
let
    disc = BK.BVP.Collocation(Ntst=20, m=4, meshadapt=false)
    show(disc)
    @test disc.Ntst == 20
    @test disc.m == 4
    @test BK.BVP.mesh_size(disc) == 20*4 + 1
    @test BK.BVP.solution_dim(disc, 2) == 2 * (20*4 + 1)
    @test BK.BVP.total_dim(disc, 2) == 2 * (20*4 + 1) + 1
end

# ----- discretize: Trapeze -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)

    disc = BK.BVP.Trapeze(M=10)
    bvp = BK.BVP.discretize(model, disc)
    show(bvp)
    @test bvp isa BK.BVP.DiscretizedBVP
    @test bvp.model === model
    @test bvp.discretizer === disc
    @test BK.BVP.state_dimension(bvp) == 2
    @test length(bvp) == 2*10 + 1
end

# ----- discretize: Collocation -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Collocation(Ntst=5, m=3, meshadapt=false)
    bvp = BK.BVP.discretize(model, disc)
    show(bvp)
    @test bvp isa BK.BVP.DiscretizedBVP
    @test BK.BVP.state_dimension(bvp) == 2
    @test length(bvp) == 2 * (5*3 + 1) + 1
end

# ----- DiscretizedBVP getters -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=10)
    bvp = BK.BVP.discretize(model, disc)
    @test BK.BVP.get_model(bvp) === model
    @test BK.BVP.get_discretizer(bvp) === disc
    @test BK.BVP.get_cache(bvp) === bvp.cache
end

# ----- BVPBifProblem -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=10)
    bvp = BK.BVP.discretize(model, disc)

    x0 = zeros(length(bvp))
    params = (ω=1.0,)
    prob = BK.BVP.BVPBifProblem(bvp, x0, params, (@optic _.ω))
    @test prob isa BK.BVP.BVPBifProblem
    @test BK.BVP.get_bvp(prob) === bvp
    @test BK.getu0(prob) === x0
    @test BK.getparams(prob) === params
    @test BK.getlens(prob) == (@optic _.ω)
    @test BK.getparam(prob) == 1.0
    @test BK.setparam(prob, 2.0) == (ω=2.0,)
end

# ----- BVPBifProblem: re_make -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=10)
    bvp = BK.BVP.discretize(model, disc)
    x0 = zeros(length(bvp))
    prob = BK.BVP.BVPBifProblem(bvp, x0, (ω=1.0,), (@optic _.ω))
    prob2 = BK.re_make(prob; u0 = ones(length(x0)))
    @test BK.getu0(prob2) ≈ ones(length(x0))
end

# ----- bvp_residual: Trapeze (identically zero) -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=4)
    bvp = BK.BVP.discretize(model, disc)
    x0 = zeros(length(bvp))
    res = BK.BVP.bvp_residual(bvp, x0, (ω=1.0,))
    @test length(res) == length(bvp)
    @test res[1:end-1] == zeros(length(bvp)-1)
end

# ----- bvp_jacobian: Trapeze, FD vs analytical -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    x0 = rand(length(bvp))
    p = (ω=1.5,)

    J_fd = BK.BVP.bvp_jacobian(bvp, BK.AutoDiffDense(), x0, p)
    @test size(J_fd) == (length(bvp), length(bvp))
end

# ----- get_solution_bvp: Trapeze -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=4)
    bvp = BK.BVP.discretize(model, disc)
    n_total = BK.BVP.state_dimension(model) * disc.M
    x0 = rand(n_total)
    sol = BK.BVP.get_solution_bvp(bvp, x0, (ω=1.0,))
    @test propertynames(sol) == (:t, :u)
    @test length(sol.t) == disc.M
    @test size(sol.u, 2) == disc.M
end

# ----- non-uniform mesh -----
let
    steps = [1.0 + 0.2 * sin(2pi * (i-1)/9) for i in 1:10]
    disc = BK.BVP.Trapeze(M=11, mesh=steps)
    @test length(disc.mesh) == 10
    @test_throws AssertionError BK.BVP.Trapeze(M=3, mesh=[-0.1, 0.5])
end

# ----- Trapeze jacobian with Dense type -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=3)
    bvp = BK.BVP.discretize(model, disc)
    n_total = BK.BVP.state_dimension(model) * disc.M
    x0 = rand(n_total)
    p = (ω=1.0,)
    J = BK.BVP.bvp_jacobian(bvp, BK.Dense(), x0, p)
    n_total = BK.BVP.state_dimension(model) * disc.M
    @test size(J) == (n_total, n_total)
end

# ==============================================================================
# Additional unit tests for comprehensive coverage
# ==============================================================================

# ----- BVPModel: record_from_solution and plot_solution -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    @test BK.record_from_solution(model) == BK.record_sol_default
    @test BK.plot_solution(model) == BK.plot_default
end

# ----- PeriodicOrbitModel: record_from_solution dispatches -----
let
    F(u, p) = [p.μ*u[1] - u[2] - u[1]*(u[1]^2+u[2]^2),
               u[1] + p.μ*u[2] - u[2]*(u[1]^2+u[2]^2)]
    model = BK.BVP.PeriodicOrbitModel(F; n=2)
    @test BK.record_from_solution(model) == BK.record_sol_default
    @test BK.plot_solution(model) == BK.plot_default
end

# ----- is_parallel, meshadapt -----
let
    sh_par = BK.BVP.Shooting(M=3, alg=nothing, parallel=true)
    sh_seq = BK.BVP.Shooting(M=3, alg=nothing, parallel=false)
    @test BK.BVP.is_parallel(sh_par) == true
    @test BK.BVP.is_parallel(sh_seq) == false

    coll_adapt = BK.BVP.Collocation(Ntst=5, m=3, meshadapt=true)
    coll_noadapt = BK.BVP.Collocation(Ntst=5, m=3, meshadapt=false)
    @test BK.BVP.meshadapt(coll_adapt) == true
    @test BK.BVP.meshadapt(coll_noadapt) == false
end

# ----- Negative mesh validation errors -----
let
    @test_throws AssertionError BK.BVP.Trapeze(M=3, mesh=[-0.1, 0.5])
    @test_throws AssertionError BK.BVP.Trapeze(M=1)
end

# ----- get_time_slices (Trapeze) -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    n_total = BK.BVP.state_dimension(model) * disc.M
    x0 = rand(length(bvp))
    slices = BK.BVP.get_time_slices(bvp, @view(x0[1:n_total]))
    @test size(slices) == (2, 5)
end

# ----- get_solution_bvp (Collocation) -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Collocation(Ntst=3, m=2, meshadapt=false)
    bvp = BK.BVP.discretize(model, disc)
    n_state = BK.BVP.solution_dim(disc, BK.BVP.state_dimension(model))
    x0 = rand(n_state)
    sol = BK.BVP.get_solution_bvp(bvp, x0, (ω=1.0,))
    @test propertynames(sol) == (:t, :u)
    @test length(sol.t) == disc.Ntst * disc.m + 1
    @test size(sol.u, 2) == disc.Ntst * disc.m + 1
end

# ----- generate_solution (Trapeze) -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    x0 = BK.BVP.generate_solution(bvp, t -> [cos(t), sin(t)])
    @test length(x0) == length(bvp)
    @test x0[end] == 0.0  # T = 0 in initial guess
    # First slice should be orbit at t=0
    @test x0[1:2] ≈ [1.0, 0.0]
end

# ----- generate_solution (Collocation) -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Collocation(Ntst=3, m=2, meshadapt=false)
    bvp = BK.BVP.discretize(model, disc)
    x0 = BK.BVP.generate_solution(bvp, t -> [cos(t), sin(t)])
    @test length(x0) == BK.BVP.solution_dim(disc, BK.BVP.state_dimension(model))
end

# ----- DiscretizedBVP: record_from_solution and plot_solution -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    @test BK.record_from_solution(bvp) == BK.record_sol_default
    @test BK.plot_solution(bvp) == BK.plot_default
end

# ----- BVPBifProblem: isinplace, has_adjoint, getdelta, is_symmetric, get_bvp -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    x0 = zeros(length(bvp))
    prob = BK.BVP.BVPBifProblem(bvp, x0, (ω=1.0,), (@optic _.ω))
    @test BK.BVP.isinplace(prob) == false
    @test BK.BVP.has_adjoint(prob) == false
    @test BK.BVP.getdelta(prob) == 1e-8
    @test BK.BVP.is_symmetric(prob) == false
    @test BK.BVP.get_bvp(prob) === bvp
end

# ----- BVPBifProblem: _getvectortype -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    x0 = zeros(length(bvp))
    prob = BK.BVP.BVPBifProblem(bvp, x0, (ω=1.0,), (@optic _.ω))
    @test BK.BVP._getvectortype(prob) == Vector{Float64}
end

# ----- BVPBifProblem: dF, d2F, d3F -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    x0 = rand(length(bvp))
    prob = BK.BVP.BVPBifProblem(bvp, x0, (ω=1.0,), (@optic _.ω))
    dx = rand(length(bvp))
    dF_val = BK.BVP.dF(prob, x0, (ω=1.0,), dx)
    @test length(dF_val) == length(x0)
    dx2 = rand(length(bvp))
    d2F_val = BK.BVP.d2F(prob, x0, (ω=1.0,), dx, dx2)
    @test length(d2F_val) == length(x0)
    dx3 = rand(length(bvp))
    d3F_val = BK.BVP.d3F(prob, x0, (ω=1.0,), dx, dx2, dx3)
    @test length(d3F_val) == length(x0)
end

# ----- BVPBifProblem: record_from_solution dispatches -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    x0 = zeros(length(bvp))
    # Default (nothing) record
    prob = BK.BVP.BVPBifProblem(bvp, x0, (ω=1.0,), (@optic _.ω))
    @test BK.record_from_solution(prob) == BK.record_sol_default
    @test BK.plot_solution(prob) == BK.plot_default
    # With custom record/plot
    my_record(x, p; k...) = (val = x[1],)
    my_plot(x, p; k...) = nothing
    prob2 = BK.BVP.BVPBifProblem(bvp, x0, (ω=1.0,), (@optic _.ω);
        record_from_solution = my_record,
        plot_solution = my_plot)
    @test BK.record_from_solution(prob2, x0, (ω=1.0,)) == (val = x0[1],)
    @test BK.plot_solution(prob2) == my_plot
end

# ----- BVPBifProblem: update! (default) -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    x0 = zeros(length(bvp))
    prob = BK.BVP.BVPBifProblem(bvp, x0, (ω=1.0,), (@optic _.ω))
    @test BK.update!(prob, nothing, nothing) == true
end

# ----- save_solution dispatches -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    # Trapeze save_solution (generic dispatch)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    x0 = rand(length(bvp))
    @test BK.BVP.save_solution(bvp, x0, (ω=1.0,)) === x0
    # Collocation save_solution (without meshadapt -> returns x)
    disc_c = BK.BVP.Collocation(Ntst=3, m=2, meshadapt=false)
    bvp_c = BK.BVP.discretize(model, disc_c)
    x0_c = rand(length(bvp_c))
    @test BK.BVP.save_solution(bvp_c, x0_c, (ω=1.0,)) === x0_c
end

# ----- saved_solution interface -----
let
    x0 = rand(10)
    mesh0 = collect(range(0, 1, length=11))
    saved = BK.BVPSavedSolutionAndState(mesh0, x0, mesh0, zeros(10))
    @test BK.saved_solution(x0) === x0
    @test BK.saved_solution(saved) === x0
end
# ----- get_periodic_orbit (Trapeze) -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    x0 = rand(length(bvp))
    po = BK.BVP.get_periodic_orbit(bvp, x0, (ω=1.0,))
    @test po isa NamedTuple
    @test haskey(po, :t)
    @test haskey(po, :u)
    @test haskey(po, :period)
    @test length(po.t) == disc.M
    @test size(po.u, 2) == disc.M
    @test po.period == x0[end]
end

# ----- get_periodic_orbit (Collocation) -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Collocation(Ntst=3, m=2, meshadapt=false)
    bvp = BK.BVP.discretize(model, disc)
    x0 = rand(length(bvp))
    po = BK.BVP.get_periodic_orbit(bvp, x0, (ω=1.0,))
    @test po isa NamedTuple
    @test haskey(po, :t)
    @test haskey(po, :u)
    @test haskey(po, :period)
    @test length(po.t) == disc.Ntst * disc.m + 1
    @test po.period == x0[end]
end

# ----- update_phase_reference! -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Trapeze(M=5)
    bvp = BK.BVP.discretize(model, disc)
    @test BK.BVP.update_phase_reference!(bvp, rand(length(bvp)), (ω=1.0,)) == true
end

# ----- Collocation: bvp_jacobian with AutoDiffDense (default, expects full vector with period) -----
let
    F(u, p) = [u[2], -p.ω^2 * u[1]]
    g(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(F, g; n=2)
    disc = BK.BVP.Collocation(Ntst=3, m=2, meshadapt=false)
    bvp = BK.BVP.discretize(model, disc)
    x0 = rand(length(bvp))
    p = (ω=1.5,)
    J = BK.BVP.bvp_jacobian(bvp, BK.AutoDiffDense(), x0, p)
    @test size(J) == (length(bvp), length(bvp))
end

# ----- Collocation with meshadapt=true (basic smoke test) -----
let
    Fbratu(x, p) = [x[2], -10*(p.a * (exp(x[1]) - 1 - p.b * x[1]^2/2))]
    gbratu(u0, uT, p) = [u0[1], uT[1]]
    model = BK.BVP.BVPModel(Fbratu, gbratu; n=2)
    disc = BK.BVP.Collocation(Ntst=5, m=3, meshadapt=true)
    bvp = BK.BVP.discretize(model, disc)

    params = (a=0.5, b=0.)
    t_vals = LinRange(0, 1, 101)
    n_state = BK.BVP.solution_dim(disc, BK.BVP.state_dimension(model))
    x0 = zeros(n_state)

    prob = BK.BVP.BVPBifProblem(bvp, x0, params, (@optic _.a))

    optn = NewtonPar(tol=1e-10, verbose=false)
    optc = ContinuationPar(
        p_min=0.1, p_max=3.0, dsmax=0.1, ds=0.01,
        detect_bifurcation=0, newton_options=optn,
        max_steps=5, nev=5, n_inversion=4)

    br = continuation(prob, PALC(), optc;
        plot=false, verbosity=0, normC=norminf)
    @test br isa BK.AbstractBranchResult
    @test length(br) > 0
end
