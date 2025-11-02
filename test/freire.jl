using BifurcationKit, Test
const BK = BifurcationKit

function freire!(du, u, p, t = 0)
    x, y, z = u
    (;β, ν, γ, a₃, b₃, r) = p
    
    δ = y - x
    δ³ = δ^3
    
    du[1] = (-(β + ν) * x + β * y - a₃ * x^3 + b₃ * δ³) / r
    du[2] = β * x - (β + γ) * y - z - b₃ * δ³
    du[3] = y
    du
end

par_freire = (γ = -0.6, β = 0.5, a₃ = 0.328578, b₃ = 0.933578, r = 0.6, ϵ = 0.01, ν = -0.9)
prob = BK.BifurcationProblem(freire!, zeros(3), par_freire, (@optic _.ν))
br = continuation(prob, PALC(), ContinuationPar(dsmax = 0.05, n_inversion = 8))
##################################################################################
import OrdinaryDiffEq as ODE
begin
    probsh = ODE.ODEProblem(freire!, zeros(3), (0, 1), par_freire; abstol = 1e-12, reltol = 1e-10)
    br_po = continuation(br, 1,
                ContinuationPar(br.contparams, ds = -0.001, dsmax = 0.01, tol_stability = 1e-4, p_min = -0.7), 
                ShootingProblem(15, probsh, ODE.Rodas5(), parallel = true);
                δp = 0.001, 
    )
    @test br_po.specialpoint[1].type == :bp
    @test br_po.specialpoint[2].type == :bp
    # plot(br, br_po)
    
    bp = get_normal_form(br_po, 2, detailed = true)
    @test bp.nf.nf.a ≈ 1e-6 atol = 1e-5
    @test bp.nf.nf.b1 ≈ 200 rtol = 1e-2
    @test bp.nf.nf.b2 ≈ 4e-4 atol = 1e-2
    @test bp.nf.nf.b3 ≈ 20811 rtol = 1e-5
    
    br_po_bp = continuation(deepcopy(br_po), 2;
                    δp = -0.001, ampfactor = 0.01,
                    use_normal_form = false, detailed = false,
    )
    
    @test br_po_bp.specialpoint[1].type == :ns
    @test br_po_bp.specialpoint[2].type == :pd

    ns = get_normal_form(br_po_bp, 1; detailed = true)
    @test ns.nf.type == :SubCritical
    @test ns.nf.nf.b ≈ 68.8696641601362 + 288.2521772452238im rtol = 1e-2

    pd = get_normal_form(br_po_bp, 2; detailed = true)
    @test pd.nf.type == :SuperCritical
    @test pd.nf.nf.a ≈ -1254 rtol = 1e-2
    @test pd.nf.nf.b3 ≈ 5548 rtol = 1e-2

    # plot(br, br_po, br_po_bp, xlims = (-0.7,-0.5))
end
##################################################################################
begin
    br_po = continuation(br, 1, 
                ContinuationPar(br.contparams, ds = -0.001, dsmax = 0.01, tol_stability = 1e-4, p_min = -0.7), 
                PeriodicOrbitOCollProblem(30,4; jacobian = BK.DenseAnalyticalInplace());
                δp = 0.001,
    )

    @test br_po.specialpoint[1].type == :bp
    @test br_po.specialpoint[2].type == :bp

    bp = get_normal_form(br_po, 2; detailed = true, prm = true)
    @test bp.nf.nf.a ≈ 1e-5 atol = 1e-5
    @test bp.nf.nf.b1 ≈ 219 rtol = 1e-2
    @test bp.nf.nf.b2 ≈ -4e-4 atol = 1e-2
    @test bp.nf.nf.b3 ≈ 1657 rtol = 1e-4

    bppo = get_normal_form(br_po, 2; detailed = true, prm = false)
    BK.predictor(bppo, 0.01, 0.1)

    br_po_bp = continuation(deepcopy(br_po), 2; 
                    δp = -0.001, ampfactor = 0.01,
                    use_normal_form = true, detailed = false,
    )
    @test br_po_bp.specialpoint[1].type == :ns
    @test br_po_bp.specialpoint[2].type == :pd
    get_normal_form(br_po_bp, 1; detailed = true)
    get_normal_form(br_po_bp, 1; detailed = true, prm = true)
    get_normal_form(br_po_bp, 2; detailed = true)
    get_normal_form(br_po_bp, 2; detailed = true, prm = true)
end
##################################################################################
begin
    br_po = continuation(br, 1, 
                ContinuationPar(br.contparams, ds = -0.001, dsmax = 0.01, tol_stability = 1e-4, p_min = -0.7), 
                PeriodicOrbitTrapProblem(M = 100; jacobian = BK.Dense());
                δp = 0.001, 
    )

    @test br_po.specialpoint[1].type == :bp
    @test br_po.specialpoint[2].type == :bp
    # plot(br, br_po, xlims = (-0.7,-0.5))

    bppo = get_normal_form(br_po, 2; detailed = true)
    BK.predictor(bppo,0.01,0.1)

    br_po_bp = continuation(deepcopy(br_po), 2; 
                    δp = -0.001, ampfactor = 0.01,
                    use_normal_form = true, detailed = false,
                    prm = false
    )

    @test br_po_bp.specialpoint[1].type == :ns
    @test br_po_bp.specialpoint[2].type == :pd

    get_normal_form(br_po_bp, 1; detailed = true)
    get_normal_form(br_po_bp, 2; detailed = true)
    # plot(br_po, br_po_bp)
end