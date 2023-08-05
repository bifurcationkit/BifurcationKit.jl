# using Revise, Plots
using Test, ForwardDiff, Parameters, LinearAlgebra
using BifurcationKit, Test
const BK = BifurcationKit
###################################################################################################
function Pop!(du, X, p, t = 0)
    @unpack r,K,a,ϵ,b0,e,d = p
    x, y, u, v = X
    p = a * x / (b0 * (1 + ϵ * u) + x)
    du[1] = r * (1 - x/K) * x - p * y
    du[2] = e * p * y - d * y
    s = u^2 + v^2
    du[3] = u-2pi * v - s * u
    du[4] = 2pi * u + v - s * v
    du
end

par_pop = ( K = 1., r = 2π, a = 4π, b0 = 0.25, e = 1., d = 2π, ϵ = 0.2, )

z0 = [0.1,0.1,1,0]

prob = BifurcationProblem(Pop!, z0, par_pop, (@lens _.b0); recordFromSolution = (x, p) -> (x = x[1], y = x[2], u = x[3]))

opts_br = ContinuationPar(pMin = 0., pMax = 20.0, ds = 0.002, dsmax = 0.01, nInversion = 6, detectBifurcation = 3, maxBisectionSteps = 25, nev = 4, maxSteps = 20000)
@set! opts_br.newtonOptions.verbose = false

################################################################################
using OrdinaryDiffEq
prob_de = ODEProblem(Pop!, z0, (0,600.), par_pop)
alg = Rodas5()
# alg = Vern9()
sol = solve(prob_de, alg)
prob_de = ODEProblem(Pop!, sol.u[end], (0,5.), par_pop, reltol = 1e-8, abstol = 1e-10)
sol = solve(prob_de, Rodas5())
################################################################################
argspo = (recordFromSolution = (x, p) -> begin
        xtt = BK.getPeriodicOrbit(p.prob, x, p.p)
        return (max = maximum(xtt[1,:]),
                min = minimum(xtt[1,:]),
                period = getPeriod(p.prob, x, p.p))
    end,)
################################################################################
probcoll, ci = generateCIProblem(PeriodicOrbitOCollProblem(26, 3; updateSectionEveryStep = 0), prob, sol, 2.)

solpo = newton(probcoll, ci, NewtonPar(verbose = false))
@test BK.converged(solpo)

_sol = BK.getPeriodicOrbit(probcoll, solpo.u,1)

opts_po_cont = setproperties(opts_br, maxSteps = 40, saveEigenvectors = true, tolStability = 1e-8)
@set! opts_po_cont.newtonOptions.verbose = false
brpo_fold = continuation(probcoll, ci, PALC(), opts_po_cont;
    verbosity = 0, plot = false,
    argspo...
    )
# pt = getNormalForm(brpo_fold, 1)

prob2 = @set probcoll.prob_vf.lens = @lens _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
    verbosity = 0, plot = false,
    argspo...
    )
# pt = getNormalForm(brpo_pd, 1, prm = false)

# codim 2 Fold
opts_pocoll_fold = ContinuationPar(brpo_fold.contparams, detectBifurcation = 3, maxSteps = 3, pMin = 0., pMax=1.2, nInversion = 4)
@set! opts_pocoll_fold.newtonOptions.tol = 1e-12
fold_po_coll1 = continuation(brpo_fold, 1, (@lens _.ϵ), opts_pocoll_fold;
        verbosity = 0, plot = false,
        detectCodim2Bifurcation = 0,
        startWithEigen = false,
        bothside = true,
        jacobian_ma = :minaug,
        bdlinsolver = BorderingBLS(solver = DefaultLS(), checkPrecision = false),
        )
@test fold_po_coll1.kind isa BK.FoldPeriodicOrbitCont

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detectBifurcation = 3, maxSteps = 3, pMin = -1., dsmax = 1e-2, ds = 1e-3)
@set! opts_pocoll_pd.newtonOptions.tol = 1e-9
pd_po_coll = continuation(brpo_pd, 1, (@lens _.b0), opts_pocoll_pd;
        verbosity = 0, plot = false,
        detectCodim2Bifurcation = 1,
        startWithEigen = false,
        usehessian = false,
        jacobian_ma = :minaug,
        normC = norminf,
        callbackN = BK.cbMaxNorm(10),
        bothside = true,
        )

@test pd_po_coll.kind isa BK.PDPeriodicOrbitCont

#####
# find the NS case
par_pop2 = @set par_pop.b0 = 0.4
sol2 = solve(remake(prob_de, p = par_pop2, u0 = [0.1,0.1,1,0], tspan=(0,1000)), Rodas5())
sol2 = solve(remake(sol2.prob, tspan = (0,10), u0 = sol2[end]), Rodas5())

probcoll, ci = generateCIProblem(PeriodicOrbitOCollProblem(26, 3; updateSectionEveryStep = 0), reMake(prob, params = sol2.prob.p), sol2, 1.2)

brpo_ns = continuation(probcoll, ci, PALC(), ContinuationPar(opts_po_cont; maxSteps = 20, ds = -0.001);
    verbosity = 0, plot = false,
    argspo...,
    )

getNormalForm(brpo_ns, 1)

prob2 = @set probcoll.prob_vf.lens = @lens _.ϵ
brpo_pd = continuation(prob2, ci, PALC(), ContinuationPar(opts_po_cont, dsmax = 5e-3);
    verbosity = 0, plot = false,
    argspo...,
    bothside = true,
    )
getNormalForm(brpo_pd, 2)

# codim 2 PD
opts_pocoll_pd = ContinuationPar(brpo_pd.contparams, detectBifurcation = 3, maxSteps = 2, pMin = 1.e-2, dsmax = 1e-2, ds = 1e-3)
@set! opts_pocoll_pd.newtonOptions.tol = 1e-10
pd_po_coll2 = continuation(brpo_pd, 2, (@lens _.b0), opts_pocoll_pd;
        verbosity = 0, plot = false,
        detectCodim2Bifurcation = 1,
        startWithEigen = false,
        usehessian = false,
        jacobian_ma = :minaug,
        normN = norminf,
        callbackN = BK.cbMaxNorm(1),
        bothside = true,
        )

opts_pocoll_ns = ContinuationPar(brpo_pd.contparams, detectBifurcation = 3, maxSteps = 2, pMin = 0., dsmax = 1e-2, ds = 1e-3)
ns_po_coll = continuation(brpo_ns, 1, (@lens _.ϵ), opts_pocoll_ns;
        verbosity = 0, plot = false,
        detectCodim2Bifurcation = 1,
        startWithEigen = false,
        usehessian = false,
        jacobian_ma = :minaug,
        normN = norminf,
        callbackN = BK.cbMaxNorm(10),
        bothside = true,
        )

#####
# test of the implementation of the jacobian for the PD case
using ForwardDiff
_probpd = pd_po_coll2.prob
_x = pd_po_coll2.sol[end].x
_solpo = pd_po_coll2.sol[end].x.u
_p1 = pd_po_coll2.sol[end].x.p
_p2 = pd_po_coll2.sol[end].p
_param= BK.setParam(pd_po_coll2, _p1)
_param = set(_param, (@lens _.ϵ), _p2)

# _Jpdad = ForwardDiff.jacobian(x -> BK.residual(_probpd, x, _param), vcat(_x.u, _x.p))
_Jpdad = BK.finiteDifferences(x -> BK.residual(_probpd, x, _param), vcat(_x.u, _x.p))

_Jma = zero(_Jpdad)
_duu = rand(length(_x.u))
_sol = BK.PDMALinearSolver(_solpo, _p1, _probpd.prob, _param, _duu, 1.; debugArray = _Jma )
_solfd = _Jpdad \ vcat(_duu, 1)

@test norm(_Jpdad - _Jma, Inf) < 1e-6
@test norm(_solfd[1:end-1] - _sol[1], Inf) < 1e-3
@test abs(_solfd[end] - _sol[2]) < 1e-3
#########
# test of the implementation of the jacobian for the NS case
_probns = ns_po_coll.prob
_x = ns_po_coll.sol[end].x
_solpo = ns_po_coll.sol[end].x.u
_p1 = ns_po_coll.sol[end].x.p
_p2 = ns_po_coll.sol[end].p
_param= BK.setParam(ns_po_coll, _p1[1])
_param = set(_param, (@lens _.ϵ), _p2)

# _Jnsad = ForwardDiff.jacobian(x -> BK.residual(_probns, x, _param), vcat(_x.u, _x.p))
_Jnsad = BK.finiteDifferences(x -> BK.residual(_probns, x, _param), vcat(_x.u, _x.p))

_Jma = zero(_Jnsad)
_dp = rand()
_sol = BK.NSMALinearSolver(_solpo, _p1[1], _p1[2], _probns.prob, _param, _duu, _dp, 1.; debugArray = _Jma )
_solfd = _Jnsad \ vcat(_duu, _dp, 1)

@test norm(_Jnsad - _Jma, Inf) < 1e-6
@test norm(_solfd[1:end-2] - _sol[1], Inf) < 1e-3
@test abs(_solfd[end-1] - _sol[2]) < 1e-3
@test abs(_solfd[end] - _sol[3]) < 1e-3