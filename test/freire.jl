using Revise
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
    
    bp = get_normal_form(br_po, 2, detailed = Val(true))
    @test bp.nf.nf.a ≈ 1e-6 atol = 1e-5
    @test bp.nf.nf.b1 ≈ 200 rtol = 1e-2
    @test bp.nf.nf.b2 ≈ 4e-4 atol = 1e-2
    @test bp.nf.nf.b3 ≈ 20811 rtol = 1e-5
    
    br_po_bp = continuation(deepcopy(br_po), 2;
                    δp = -0.001, ampfactor = 0.01,
                    use_normal_form = false, detailed = Val(false),
    )
    
    @test br_po_bp.specialpoint[1].type == :ns
    @test br_po_bp.specialpoint[2].type == :pd

    ns = get_normal_form(br_po_bp, 1; detailed = Val(true))
    @test ns.nf.type == :SubCritical
    # @test ns.nf.nf.b ≈ 68.8696641601362 + 288.2521772452238im rtol = 1e-2

    pd = get_normal_form(br_po_bp, 2; detailed = Val(true))
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

    bp = get_normal_form(br_po, 2; detailed = Val(true), prm = Val(true))
    @test bp.nf.nf.a ≈ 1e-5 atol = 1e-5
    @test bp.nf.nf.b1 ≈ 219 rtol = 1e-2
    @test bp.nf.nf.b2 ≈ -4e-4 atol = 1e-2
    @test bp.nf.nf.b3 ≈ 1657 rtol = 1e-4

    bppo = get_normal_form(br_po, 2; detailed = Val(true), prm = Val(false))
    BK.predictor(bppo, 0.01, 0.1)

    br_po_bp = continuation(deepcopy(br_po), 2; 
                    δp = -0.001, ampfactor = 0.01,
                    use_normal_form = true, detailed = Val(false),
    )
    @test br_po_bp.specialpoint[1].type == :ns
    @test br_po_bp.specialpoint[2].type == :pd

    ns_p = get_normal_form(br_po_bp, 1; detailed = Val(true), prm = Val(true))
    # @test ns_p.nf.nf.b/(imag(ns_p.nf.nf.b)) ≈ (-6.724846398058799 - 52.018420910967144im)/(- 52.018420910967144) rtol = 1e-1
    @test ns_p.nf.type == :SuperCritical

    ns = get_normal_form(br_po_bp, 1; detailed = Val(true), prm = Val(false))
    @test ns.nf.nf.a ≈ 1.2086369211684012 + 2.0403410963243346e-17im rtol = 1e-2
    # @test ns.nf.nf.d ≈ 0.03767524458489463 - 1.0067498546014593im rtol = 1e-2
    # @test ns_p.nf.type == ns.nf.type

    pd_p = get_normal_form(br_po_bp, 2; detailed = Val(true), prm = Val(true))
    # @test pd.nf.nf.a ≈ 21 rtol = 1e-1 # TODO THIS DOES NOT WORK!! WHY??
    # @test pd_p.nf.nf.b3 ≈ 724 rtol = 1e-2
    @test pd_p.nf.type == :SuperCritical

    pd = get_normal_form(br_po_bp, 2; detailed = Val(true), prm = Val(false))
    @test pd.nf.type == :SuperCritical == pd_p.nf.type
    @test pd.nf.nf.a₀₁ ≈ 2.06888 rtol = 1e-1
    @test pd.nf.nf.a ≈ 0.3043419215670348 rtol = 1e-1
    @test pd.nf.nf.c₁₁ ≈ -247.7313562613303 rtol = 1e-1
    @test pd.nf.nf.b3 ≈ -6.736 rtol = 1e-2
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

    bppo = get_normal_form(br_po, 2; detailed = Val(true))
    BK.predictor(bppo,0.01,0.1)

    br_po_bp = continuation(deepcopy(br_po), 2; 
                    δp = -0.001, ampfactor = 0.01,
                    use_normal_form = true, detailed = Val(false),
                    prm = Val(false)
    )

    @test br_po_bp.specialpoint[1].type == :ns
    @test br_po_bp.specialpoint[2].type == :pd

    get_normal_form(br_po_bp, 1; detailed = Val(true))
    get_normal_form(br_po_bp, 2; detailed = Val(true))
    # plot(br_po, br_po_bp)
end