using Test
using BifurcationKit, LinearAlgebra, ForwardDiff, SparseArrays
const BK = BifurcationKit
using Core.Compiler: return_type

function Fsl!(f, u, p, t = 0)
    (;r, μ, ν, c3) = p
    u1 = u[1]; u2 = u[2]
    ua = u1^2 + u2^2
    f[1] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2)
    f[2] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1)
    return f
end
Fsl(u, p) = Fsl!(similar(u), u, p, 0)

function Jsl!(J, u, p, t = 0)
    (; r, μ, ν, c3) = p
    u1 = u[1]; u2 = u[2]
    ua = u1^2 + u2^2
    A = c3*u1 - μ*u2; B = c3*u2 + μ*u1
    J[1,1] = r - (2*u1*A + ua*c3)
    J[1,2] = -ν - (2*u2*A - ua*μ)
    J[2,1] =  ν - (2*u1*B + ua*μ)
    J[2,2] = r - (2*u2*B + ua*c3)
    return J
end
Jsl(u, p, t = 0) = Jsl!(zeros(2,2), u, p, t)

par_sl = (r = 0.1, μ = 0., ν = 1.0, c3 = 1.0)
u0 = [.001, .001]
par_hopf = (@set par_sl.r = 0.1)
probsl = BifurcationProblem(Fsl!, u0, par_hopf, (@optic _.r), J=Jsl, J! = Jsl!)
probsl_ip = BifurcationProblem(Fsl!, u0, par_hopf, (@optic _.r), inplace = true, J! = Jsl!)

optconteq = ContinuationPar(ds = -0.01, detect_bifurcation = 3, p_min = -0.5, n_inversion = 8)
br = continuation(probsl, PALC(), optconteq)

let
    hp = get_normal_form(br, 1)
    pred = predictor(hp, 0.1)
    @test pred.orbit(0)[1] ≈ sqrt(0.1)
end

########################################################################
# CollocationDisc API: basic mesh, interpolation, residual
# Uses CollocationDisc → POModel → discretize → DiscretizedPO
########################################################################
let
    return
    Ntst = 4; m = 4
    coll_disc = BK.CollocationDisc(;Ntst, m)
    po_model = BK.POModel(Fsl; n = 2)
    @test BK.discretize(po_model, coll_disc) isa BK.DiscretizedPO
    po_d = BK.discretize(po_model, coll_disc) # init section (discretiser dependent) dans DiscretizedBVP.cache
    # contient:
    # - section
    # - mesh

    # plutot:
    # struct DiscretizedPO <: AbstractDiscretizedPO
    #     d_bvp::DiscretizedBVP
    #     section
    #     mesh
    # end
    # get_discretized_bvp(d::AbstractDiscretizedBVP) = d
    # get_discretized_bvp(d::DiscretizedPO) = d.d_bvp
    # get_discretizer(d::AbstractDiscretizedBVP) = get_discretizer(get_discretized_bvp(d))

    # @test size(po_d) == (2, m, Ntst)
    # @test length(po_d) == 2 * (1 + m * Ntst)
    BK.get_times(po_d)
    # BK.get_max_time_step(po_d)
    BK.get_gauss_nodes(po_d)
    # BK.update_mesh!(po_d, po_d.cache.po_coll.mesh_cache.τs)
    @test BK.lagrange(1, 1.0, 1:10) == 1
    @test BK.lagrange(1, 2.0, 1:10) == 0

    BK.get_mesh_size(coll_disc)
    BK.get_Ls(po_d)

    _orbit(t) = [cos(2pi * t), 0] * sqrt(par_sl.r / par_sl.c3)
    _ci = BK.generate_solution(po_d, _orbit, 1.)
    BK.get_periodic_orbit(po_d, _ci, par_sl)
    @test BK.∂(sin, Val(2))(0.) == 0
    BK.po_residual(po_d, _ci, par_sl)
    BK.get_time_slices(po_d, _ci)

    sol = BK.POInterpolation(po_d, _ci)
    sol(rand())
end

########################################################################
# Large N interpolation test via DiscretizedPO
########################################################################
let
    N = 1000
    coll_disc = BK.CollocationDisc(; Ntst=200, m=5)
    po_model = BK.POModel(Fsl; n=N)
    po_d = BK.discretize(po_model, coll_disc)

    _ci = BK.generate_solution(po_d, t -> cos(t) .* ones(N), 2pi)
    BK.get_times(po_d)
    sol = BK.POInterpolation(po_d, _ci)
    @test sol(0.1) ≈ cos(0.1) .* ones(N)
    for (i,t) in pairs(BK.get_times(po_d))
        @test sol(t)[1] ≈ cos(t)
    end
    for t in BK.getmesh(po_d)
        @test t in BK.get_times(po_d)
    end
end

########################################################################
# Phase condition: low-level, uses internal Collocation from cache
########################################################################
@views function phaseCond(pb::BK.Collocation, u, v)
    𝒯 = eltype(u)
    phase = zero(𝒯)
    uc = BK.get_time_slices(pb, u); vc = BK.get_time_slices(pb, v)
    n, m, Ntst = size(pb)
    T = BK.getperiod(pb, u, nothing)
    guj = zeros(𝒯, n, m); uj  = zeros(𝒯, n, m+1)
    gvj = zeros(𝒯, n, m); vj  = zeros(𝒯, n, m+1)
    L, ∂L = BK.get_Ls(pb.mesh_cache)
    ω = pb.mesh_cache.gauss_weight
    rg = UnitRange(1, m+1)
    @inbounds for j in 1:Ntst
        uj .= uc[:, rg]; vj .= vc[:, rg]
        mul!(guj, uj, L); mul!(gvj, vj, ∂L)
        @inbounds for l in 1:m
            phase += dot(guj[:, l], gvj[:, l]) * ω[l]
        end
        rg = rg .+ m
    end
    return phase / T
end

let
    return
    for Ntst in 2:10:100
        coll_disc = BK.CollocationDisc(; Ntst, m=10)
        po_model = BK.POModel(Fsl; n=1)
        po_d = BK.discretize(po_model, coll_disc)
        _coll = po_d.cache.po_coll

        BK.update_mesh!(po_d, sort(vcat(0,rand(Ntst-1),1)))

        _ci1 = BK.generate_solution(po_d, t -> [1], 2pi)
        _ci2 = BK.generate_solution(po_d, t -> [t], 2pi)
        @test phaseCond(_coll, _ci1, _ci2) ≈ 1 atol = 1e-10

        _ci1 = BK.generate_solution(po_d, t -> [cos(t)], 2pi)
        _ci2 = BK.generate_solution(po_d, t -> [sin(t)], 2pi)
        @test phaseCond(_coll, _ci1, _ci2) ≈ 1/2 atol = 1e-5

        _ci1 = BK.generate_solution(po_d, t -> [cos(t)], 2pi)
        _ci2 = BK.generate_solution(po_d, t -> [cos(t)], 2pi)
        @test phaseCond(_coll, _ci1, _ci2) / pi ≈ 0 atol = 1e-11

        _ci1 = BK.generate_solution(po_d, t -> [cos(t)], 2pi)
        _ci2 = BK.generate_solution(po_d, t -> [t], 2pi)
        @test phaseCond(_coll, _ci1, _ci2) / pi ≈ 0 atol = 1e-5
    end
end

########################################################################
# Integration ∫ via DiscretizedPO
########################################################################
let
    return
    coll_disc = BK.CollocationDisc(; Ntst=22, m=10)
    po_model = BK.POModel(Fsl; n=1)
    po_d = BK.discretize(po_model, coll_disc)

    _ci1 = BK.generate_solution(po_d, t -> [cos(2pi*t)], 1)
    _ci2 = BK.generate_solution(po_d, t -> [cos(2pi*t)], 1)
    @test BK.∫(po_d, BK.get_time_slices(po_d, _ci1), BK.get_time_slices(po_d, _ci2)) ≈ 0.5

    _ci1 = BK.generate_solution(po_d, t -> [cos(2pi*t)], 3)
    _ci2 = BK.generate_solution(po_d, t -> [cos(2pi*t)], 3)
    @test BK.∫(po_d, BK.get_time_slices(po_d, _ci1), BK.get_time_slices(po_d, _ci2), 3) ≈ 3/2
    @test BK.∫(po_d, _ci1, _ci2, 3) ≈ 3/2
end

########################################################################
# po_residual, Newton, continuation, Floquet
# CollocationDisc provides config; Collocation created with proper ϕ/xπ
########################################################################
let
    Ntst = 50; m = 4; N = 2

    coll_disc = BK.CollocationDisc(;Ntst, m)
    po_model = BK.POModel(Fsl; n = N)
    po_d = BK.discretize(po_model, coll_disc)
    _coll = po_d.cache.po_coll

    n_unk = N * (1 + m * Ntst)
    _coll.section.ϕ[2] = 1
    BK.updatesection!(_coll, _coll.section.ϕ, nothing)

    _orbit(t) = [cos(t), sin(t)] * sqrt(par_sl.r/par_sl.c3)
    _ci = BK.generate_solution(po_d, _orbit, 2pi)
    BK.po_residual(po_d, _ci, par_sl) # TODO create a POBifProblem and call residual on it
    @test BK.po_residual(po_d, _ci, par_sl)[1:end-1] |> norminf < 1e-7

    _coll_ip = @set _coll.prob_vf = probsl_ip

    @time BK.po_residual(po_d, _ci, par_sl);
    @time BK.po_residual(_coll_ip, _ci, par_sl);

    _sol = BK.get_periodic_orbit(_coll, _ci, nothing)
    for (i, t) in pairs(_sol.t)
        @test _sol.u[:, i] ≈ _orbit(t)
    end

    args = (
        plot_solution = (x,p; k...) -> begin
        outt = get_periodic_orbit(_coll, x, p)
        plot!(vec(outt.t), outt.u[1, :]; k...)
        end,
        finalise_solution = (z, tau, step, contResult; k...) -> begin
            return true
        end,
    )

    optcontpo = ContinuationPar(optconteq; detect_bifurcation = 2, tol_stability = 1e-7)
    @reset optcontpo.ds = -0.01
    @reset optcontpo.newton_options.verbose = true

    _coll2 = @set _coll.prob_vf = BifurcationProblem(Fsl, u0, par_sl, (@optic _.r), J=Jsl)
    @reset _coll2.jacobian = BK.AutoDiffDense()
    sol_po = newton(_coll2, _ci, optcontpo.newton_options)

    solc = BK.POInterpolation(_coll2, sol_po.u)
    let
        mesh = BK.getmesh(_coll2)
        solpo = get_periodic_orbit(_coll2, sol_po.u, nothing)
        for (i, t) in pairs(solpo.t)
            @test solc(t) ≈ solpo.u[:, i]
        end
    end

    _coll3 = @set _coll2.update_section_every_step = UInt(1)
    br_po = @time continuation(_coll3, _ci, PALC(tangent = Bordered()), optcontpo;
            verbosity = 0, plot = false,
            args...,
            )

    br_po = @time continuation(_coll3, _ci, PALC(tangent = Bordered()), optcontpo;
            verbosity = 0, plot = false,
            args...,
            linear_algo  = COPBLS(),
            )

    for k in 1:length(br_po)-1
        local _eigvals = br_po[k].eigenvals
        μ1_bk = minimum(real, _eigvals)
        μ1 = -2*br_po[k].param*(br_po[k].period)
        @test isapprox(μ1_bk, μ1, atol = 1e-5 )
    end

    newton(_coll2, _ci, NewtonPar())
    newton(_coll2, _ci, NewtonPar(linsolver = COPLS()))

    Jw = @time (BK.jacobian(BK.getprob(br_po), br_po.sol[3].x, @set par_sl.r = br_po.sol[3].p))
    J = (Jw) |> copy
    @test BK._eig_floquet_coll((J),2,4,50,2)[1] ≈ BK._eig_floquet_coll(sparse(J),2,4,50,2)[1] atol=1e-10
    @test BK._eig_floquet_coll_small_n((J),2,4,50,2,BK.COPCACHE(BK.get_discretization(BK.getprob(br_po)), Val(0)))[1] ≈ BK._eig_floquet_coll((J),2,4,50,2)[1] atol=1e-10
    @test BK._eig_floquet_coll_small_n((J),2,4,50,2,BK.COPCACHE(BK.get_discretization(BK.getprob(br_po)), Val(0)))[1] ≈ FloquetGEV(DefaultEig(),size(J,1)-1,2)(BK.get_discretization(BK.getprob(br_po)),Jw,2)[1]
end

########################################################################
# Analytical Jacobian via DiscretizedPO (forwarding methods)
########################################################################
let
    Ntst = 10; m = 3; N = 4
    nullvf(x,p) = zero(x)

    coll_disc = BK.CollocationDisc(;Ntst, m)
    po_model = BK.POModel(nullvf; n=N)
    po_d = BK.discretize(po_model, coll_disc)

    _ci = BK.generate_solution(po_d, t->cos(t) .* ones(N), 2pi)
    BK.po_residual(po_d, _ci, par_sl)
    Jcofd = ForwardDiff.jacobian(z -> BK.po_residual(po_d, z, par_sl), _ci)
    D = @time BK.po_analytical_jacobian(po_d, _ci, par_sl)
    @test norminf(Jcofd - D) < 1e-14

    Ntst = 140; m = 4; N = 5
    _al = I(N) + 0.1 .* rand(N,N)
    idvf(x,p) = _al*x

    coll_disc = BK.CollocationDisc(;Ntst, m)
    po_model = BK.POModel(idvf; n=N)
    po_d = BK.discretize(po_model, coll_disc)

    _ci = BK.generate_solution(po_d, t->cos(t) .* ones(N), 2pi)
    Jcofd = ForwardDiff.jacobian(z -> BK.po_residual(po_d, z, par_sl), _ci)
    Jco = BK.po_analytical_jacobian(po_d, _ci, par_sl)
    @test norminf(Jcofd - Jco) < 1e-14

    N = 2; @assert N == 2 "must be the dimension of the SL"
    Ntst = 3; m = 2

    coll_disc = BK.CollocationDisc(;Ntst, m)
    po_model = BK.POModel(Fsl; n=N)
    po_d = BK.discretize(po_model, coll_disc)

    _ci = BK.generate_solution(po_d, t->cos(t) .* ones(N), 2pi)
    Jcofd = ForwardDiff.jacobian(z->BK.po_residual(po_d, z, par_sl), _ci)
    Jco = @time BK.po_analytical_jacobian(po_d, _ci, par_sl)
    Jco_bk = @time BK.po_jacobian_block(po_d, _ci, par_sl)
    @test norminf(Jcofd - Jco) < 1e-14
    @test norminf(Jcofd - Jco_bk) < 1e-14

    BK.po_analytical_jacobian(po_d, _ci, par_sl; _transpose = Val(true), ρF = 1)

    _asp = sparse(I(N) + 0.1 .* sprand(N,N,0.1))

    coll_disc = BK.CollocationDisc(;Ntst, m)
    po_model_dense = BK.POModel((u, p) -> _asp * u; n=N)
    po_model = BK.POModel((u, p) -> _asp * u; n=N)
    po_d_dense = BK.discretize(po_model_dense, coll_disc)
    po_d = BK.discretize(po_model, coll_disc)

    _ci = BK.generate_solution(po_d, t->cos(t) .* ones(N), 2pi)
    Jco_sp = BK.po_analytical_jacobian_sparse(po_d, _ci, par_sl)
    Jco = BK.po_analytical_jacobian(po_d_dense, _ci, par_sl)
    @test norminf(Jco - Array(Jco_sp)) < 1e-15
    Jco2 = copy(Jco) |> sparse
    Jco2 .= 0
    _indx = BK.get_blocks(po_d, Jco2)
    @time BK.jacobian_poocoll_sparse_indx!(po_d, Jco2, _ci, par_sl, _indx)
    @test norminf(Jco - Jco2) < 1e-14
end

########################################################################
# Hopf predictor + continuation (uses Collocation directly,
# which is AbstractBoundaryValueDiscretization)
########################################################################
let
    optcontpo = ContinuationPar(optconteq; detect_bifurcation = 2, tol_stability = 1e-7)
    _cont_po =(@set ContinuationPar(optcontpo; ds = 0.01, max_steps = 10, p_max = 0.8).newton_options.verbose = false)
    _hp = BK.get_normal_form(br, 1; detailed = Val(true))
    pred = BK.predictor(_hp, 0.1)
    @test isconcretetype(return_type( (pred.orbit), typeof((0.1)))) == false
    BK._continuation(_hp, br.prob, _cont_po,
                    BK.Collocation(20, 5; jacobian = BK.DenseAnalytical()))
end

########################################################################
# aBS (automatic branch switching) from Hopf point
# Uses Collocation directly (AbstractBoundaryValueDiscretization)
########################################################################
let
    optcontpo = ContinuationPar(optconteq; detect_bifurcation = 2, tol_stability = 1e-7)
    for jacPO in (BK.DenseAnalytical(), BK.AutoDiffDense(), BK.FullSparse(), BK.DenseAnalyticalInplace()), use_nf in (true, false)
        useGEV = jacPO in (BK.AutoDiffDense(), BK.DenseAnalytical())
        _cont_po =(@set ContinuationPar(optcontpo; ds = 0.01, max_steps = 10, p_max = 0.8).newton_options.verbose = false)
        for lspo in (BK.MatrixBLS(), BK.COPBLS())
            for eig in (EigArnoldiMethod(;sigma=0.1), EigArpack(0.1), DefaultEig())
                br_po_gev = continuation(br, 1, _cont_po,
                    BK.Collocation(20, 5; jacobian = jacPO);
                    δp = 0.1,
                    use_normal_form = use_nf,
                    usedeflation = true,
                    eigsolver = useGEV ? BK.FloquetGEV(eig,(20*5+1)*2,2) : BK.FloquetColl(),
                )
                issorted(br_po_gev.eig[1].eigenvals, by = real)

                br_po = continuation(br, 1, _cont_po,
                    BK.Collocation(20, 5; jacobian = jacPO);
                    δp = 0.1,
                    use_normal_form = use_nf,
                    usedeflation = true,
                    eigsolver = BK.FloquetColl(),
                )
                issorted(br_po.eig[1].eigenvals, by = real)

                if eig isa DefaultEig
                    for i=1:length(br_po)-1
                        @test BK.eigenvals(br_po, i) ≈ BK.eigenvals(br_po_gev, i)
                    end
                end

                for k in 1:length(br_po)-1
                    _eigvals = br_po[k].eigenvals
                    μ1_bk = minimum(real, _eigvals)
                    valid = minimum(abs, _eigvals) < 1e-9
                    μ1 = -2*br_po[k].param*(br_po[k].period)
                    @test isapprox(μ1_bk, μ1, atol = 1e-5) || (eig isa EigArnoldiMethod) || (eig isa EigArpack) || ~valid
                end
            end

            br_po_adapt = continuation(br, 1, _cont_po,
                BK.Collocation(20, 5; jacobian = jacPO, meshadapt = true);
                δp = 0.1,
                use_normal_form = use_nf,
                linear_algo = lspo,
                usedeflation = true,
                eigsolver = BK.FloquetColl(),
            )
            issorted(br_po_adapt.eig[1].eigenvals, by = real)
        end
    end
end
