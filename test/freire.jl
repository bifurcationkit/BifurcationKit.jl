# using Revise
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
using OrdinaryDiffEq
probsh = ODEProblem(freire!, zeros(3), (0., 1.), par_freire; abstol = 1e-12, reltol = 1e-10)
br_po = continuation(br, 1,
            ContinuationPar(br.contparams, ds = -0.001, dsmax = 0.01, tol_stability = 1e-4, p_min = -0.7), 
            ShootingProblem(15, probsh, Rodas5(), parallel = true);
            δp = 0.001, 
            )
@test br_po.specialpoint[1].type == :bp
@test br_po.specialpoint[2].type == :bp
# plot(br, br_po)

get_normal_form(br_po, 2, detailed = true)

br_po_bp = continuation(deepcopy(br_po), 2;
                        δp = -0.001, ampfactor = 0.01,
                        use_normal_form = false, detailed = false,
                        )

@test br_po_bp.specialpoint[1].type == :ns
@test br_po_bp.specialpoint[2].type == :pd
# plot(br, br_po, br_po_bp, xlims = (-0.7,-0.5))
##################################################################################
br_po = continuation(br, 1, 
            ContinuationPar(br.contparams, ds = -0.001, dsmax = 0.01, tol_stability = 1e-4, p_min = -0.7), 
            PeriodicOrbitOCollProblem(30,4; jacobian = BK.DenseAnalyticalInplace());
            δp = 0.001, 
            )

@test br_po.specialpoint[1].type == :bp
@test br_po.specialpoint[2].type == :bp

get_normal_form(br_po, 2; detailed = true, prm = true)
bppo = get_normal_form(br_po, 2; detailed = true, prm = false)
BK.predictor(bppo, 0.01, 0.1)

br_po_bp = continuation(deepcopy(br_po), 2; 
                        δp = -0.001, ampfactor = 0.01,
                        use_normal_form = true, detailed = false,
                        )
@test br_po_bp.specialpoint[1].type == :ns
@test br_po_bp.specialpoint[2].type == :pd
##################################################################################
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
# plot(br_po, br_po_bp)