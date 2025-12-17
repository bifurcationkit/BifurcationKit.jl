# using Revise, Plots
using ForwardDiff, Test
using BifurcationKit, LinearAlgebra
import OrdinaryDiffEq as ODE
const BK = BifurcationKit
const FD = ForwardDiff

function Fsl!(f, u, p, t = 0)
    (;r, μ, ν, c3, c5) = p
    u1 = u[1]
    u2 = u[2]
    ua = u1^2 + u2^2
    f[1] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
    f[2] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2
    return f
end

Fsl(x, p) = Fsl!(similar(x), x, p)
dFsl(x, dx, p) = FD.derivative(t -> Fsl(x .+ t .* dx, p), zero(dx[1]*x[1]))

function FslMono!(f, x, p, t)
    u = x[1:2]
    du = x[3:4]
    Fsl!(f[1:2], u, p, t)
    f[3:4] .= dFsl(u, du, p)
end
####################################################################################################
par_sl = (r = 0.5, μ = 0., ν = 1.0, c3 = 1.0, c5 = 0.0,)
par_hopf = (@set par_sl.r = 0.1)
u0 = [.001, .001]

prob_vf = BifurcationKit.BifurcationProblem(Fsl!, u0, par_hopf, (@optic _.r))

optconteq = ContinuationPar(ds = -0.01, p_min = -0.5, n_inversion = 8)
br = continuation(prob_vf, PALC(), optconteq)
####################################################################################################
prob = ODE.ODEProblem(Fsl!, u0, (0., 100.), par_hopf)
probMono = ODE.ODEProblem(FslMono!, vcat(u0, u0), (0., 100.), par_hopf)
BK._apply_vector_field(ODE.ODEProblem(Fsl!, u0, (0., 100.), par_sl), zeros(2), u0, par_sl)
BK._apply_vector_field(EnsembleProblem(ODE.ODEProblem((x,p,t)->Fsl!(similar(x), x, p), u0, (0., 100.), par_sl)), u0, par_sl)
####################################################################################################
sol = ODE.solve(prob, ODE.KenCarp4(), abstol=1e-9, reltol=1e-6)
# plot(sol[1,:], sol[2,:])

# test generation of initial guess from ODESolution
generate_ci_problem(PeriodicOrbitTrapProblem(M = 10), prob_vf, sol, 1.)
generate_ci_problem(PeriodicOrbitOCollProblem(10, 2), prob_vf, sol, 1.)
generate_ci_problem(ShootingProblem(M=10), prob_vf, prob, sol, 1.)
generate_ci_problem(PoincareShootingProblem(M=10), prob_vf, prob, sol, 1.)
####################################################################################################
section(x, T) = x[1] #* x[end]
section(x, T, dx, dT) = dx[1] #* x[end]
# standard simple shooting
M = 1
dM = 1
_pb = ShootingProblem(prob, KenCarp4(), 1, section; abstol = 1e-10, reltol=1e-9)
BifurcationKit.has_monodromy_DE(_pb.flow)

initpo = [0.13, 0., 6.]
res = _pb(initpo, par_hopf)

# test the flowDE interface
_pb_par = ShootingProblem(prob, KenCarp4(), 1, section; abstol=1e-10, reltol=1e-9, parallel = true)
_flow = _pb_par.flow; @reset _flow.vjp = (args...; kw...) -> nothing
BK.vjp(_flow, initpo, par_hopf, initpo, 0.1)
BK.jvp(_pb_par.flow, initpo, par_hopf, initpo, 0.1)

# test of the differential of the shooting method

_dx = rand(3)
resAD = FD.derivative(z -> _pb(initpo .+ z .* _dx, par_hopf), 0.)
resFD = (_pb(initpo .+ 1e-8 .* _dx, par_hopf) - _pb(initpo, par_hopf)) .* 1e8
resAN = BK.jvp(_pb, initpo, par_hopf, _dx; δ = 1e-8)
@test norm(resAN - resFD, Inf) < 5e-5
@test norm(resAN - resAD, Inf) < 5e-5
####################################################################################################
# test shooting interface M = 1
@info "Single Shooting"
_pb = ShootingProblem(prob, ODE.Rodas4(), [initpo[1:end-1]]; abstol=1e-10, reltol=1e-9, lens = (@optic _.r))
res = BK.residual(_pb, initpo, par_hopf)
res = BK.jvp(_pb, initpo, par_hopf, initpo)
@test _pb.flow.odeprob.p == _pb.par

# test the jacobian of the functional in the case M=1
_Jad = FD.jacobian( x -> _pb(x, par_hopf), initpo)
_Jana = _pb(Val(:JacobianMatrix), initpo, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-7

_pb2 = ShootingProblem(prob, ODE.Rodas4(), probMono, ODE.Rodas4(autodiff=false), [initpo[1:end-1]]; abstol = 1e-10, reltol = 1e-9)
res = BK.residual(_pb2, initpo, par_hopf)
res = BK.jvp(_pb2, initpo, par_hopf, initpo)
@test BK.issimple(_pb2)
@test _pb2.flow.odeprob.p == _pb2.par

# we test this using Newton - Continuation
optn = NewtonPar(verbose = false, tol = 1e-9,  max_iterations = 20)
# deflationOp = BK.DeflationOperator(2, (x,y) -> dot(x[1:end-1], y[1:end-1]), 1.0, [zeros(3)])
outpo = newton(_pb, initpo, optn; normN = norminf)
@test BK.converged(outpo)
@test outpo.prob.prob.jacobian isa BK.AutoDiffDense

BK.getperiod(_pb, outpo.u, par_hopf)
BK.get_periodic_orbit(_pb, outpo.u, par_hopf)

opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds= -0.01, p_max = 4.0, max_steps = 5, detect_bifurcation = 2, nev = 2, newton_options = (@set optn.tol = 1e-7), tol_stability = 1e-5)
br_pok2 = continuation(_pb, outpo.u, PALC(tangent = Bordered()),
    opts_po_cont;
    # verbosity = 0, plot = false,
    normC = norminf)
@test br_pok2.prob isa BK.WrapPOSh
@test br_pok2.prob.prob.jacobian isa BK.AutoDiffDense
@test br_pok2.period[1] ≈ 2pi rtol = 1e-7
_sol = BK.get_po_solution(_pb, outpo.u, BK.getparams(_pb))
_sol(0.1)
# plot(br_pok2)

# test of all matrix-based jacobians 
let
    for jacPO in (BK.AutoDiffDense(), BK.FiniteDifferences(), BK.AutoDiffDenseAnalytical())
        for eig in (FloquetQaD(optn.eigsolver), FloquetGEV(optn.eigsolver, 2*_pb.M, 2))
            println("*"^50)
            br_po = continuation((@set _pb.jacobian = jacPO), outpo.u, PALC(tangent = Bordered()),
            opts_po_cont;
            # verbosity = 0, plot = false,
            eigsolver = eig,
            normC = norminf)
        end
    end
end
####################################################################################################
# test automatic branch switching
@info "Single Shooting aBS"
_probsh = ShootingProblem(1, prob, ODE.KenCarp4();  abstol = 1e-10, reltol = 1e-9, lens = (@optic _.r))
br_pok2 = continuation(br, 1, opts_po_cont, _probsh; normC = norminf, verbosity = 0, autodiff_nf = false)

@test br_pok2.prob.prob.jacobian isa BK.AutoDiffDense
@test br_pok2.prob isa BK.WrapPOSh
@test br_pok2.period[1] ≈ 2pi rtol = 1e-7

# idem with deflation
br_pok2 = continuation(br, 1, opts_po_cont, _probsh; normC = norminf, usedeflation = true)
@test br_pok2.prob.prob.jacobian isa BK.AutoDiffDense
@test br_pok2.prob isa BK.WrapPOSh
@test br_pok2.period[1] ≈ 2pi rtol = 1e-7

# test matrix-free computation of floquet coefficients
eil = EigKrylovKit(dim = 2, x₀=rand(2))
opts_po_contMF = @set opts_po_cont.newton_options.eigsolver = eil
opts_po_contMF = @set opts_po_cont.detect_bifurcation = 0
br_pok2 = continuation(br,1, opts_po_contMF, _probsh; normC = norminf)
@test br_pok2.prob.prob.jacobian isa BK.AutoDiffDense
@test br_pok2.prob isa BK.WrapPOSh
@test br_pok2.period[1] ≈ 2pi rtol = 1e-7

# case with 2 sections
br_pok2_s2 = continuation(br, 1, (@set opts_po_cont.newton_options.verbose = false), ShootingProblem(2, prob, ODE.KenCarp4();  abstol = 1e-10, reltol = 1e-9, lens = (@optic _.r)); normC = norminf)
@test br_pok2_s2.prob.prob.jacobian isa BK.AutoDiffDense
@test br_pok2_s2.prob isa BK.WrapPOSh
@test br_pok2_s2.period[1] ≈ 2pi rtol = 1e-7
####################################################################################################
# test shooting interface M > 1
initpo = [0.13, 0., 6.]
_pb = ShootingProblem(prob, ODE.KenCarp4(), [initpo[1:end-1],initpo[1:end-1],initpo[1:end-1]]; abstol =1e-10, reltol=1e-9)
initpo = [0.13, 0, 0, 0.13, 0, 0.13 , 6.3]
res = BK.residual(_pb, initpo, par_hopf)
res = BK.jvp(_pb, initpo, par_hopf, initpo)
# test the jacobian of the functional
_Jad = FD.jacobian( x -> _pb(x, par_hopf), initpo)
_Jana = _pb(Val(:JacobianMatrix), initpo, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-7
####################################################################################################
# test shooting interface M > 1, parallel
initpo = [0.13, 0., 6.]
_pb = ShootingProblem(prob, ODE.KenCarp4(), [initpo[1:end-1],initpo[1:end-1],initpo[1:end-1]]; abstol =1e-10, reltol=1e-9, parallel = true)
initpo = [0.13, 0, 0, 0.13, 0, 0.13 , 6.3]
res = _pb(initpo, par_hopf)
res = BK.jvp(_pb, initpo, par_hopf, initpo)
# test the jacobian of the functional
_Jad = FD.jacobian( x -> _pb(x, par_hopf), initpo)
_Jana = _pb(Val(:JacobianMatrix), initpo, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-7

# test flowDE interface
_pb2_par = ShootingProblem(prob, Rodas4(), probMono, Rodas4(autodiff=false), [initpo[1:end-1],initpo[1:end-1],initpo[1:end-1]]; abstol = 1e-10, reltol = 1e-9, parallel = true)
BK.jvp(_pb2_par.flow, initpo, par_hopf, initpo, 0.1)

####################################################################################################
@info "Single Poincaré Shooting"
# Single Poincaré Shooting with hyperplane parametrization
normals = [[-1., 0.]]
centers = [zeros(2)]

probPsh = PoincareShootingProblem(2, prob, Rodas4(), probMono, Rodas4(autodiff=false); abstol=1e-10, reltol=1e-9, jacobian = BK.AutoDiffDenseAnalytical())
@test probPsh.par == probPsh.flow.prob1.p

probPsh = PoincareShootingProblem(2, prob, Rodas4(); rtol = abstol=1e-10, reltol=1e-9, jacobian = BK.AutoDiffDenseAnalytical())
@test probPsh.par == probPsh.flow.prob.p

probPsh = PoincareShootingProblem(prob, Rodas4(),
        probMono, Rodas4(autodiff=false),
        normals, centers; abstol = 1e-10, reltol = 1e-9,
        jacobian = BK.AutoDiffDenseAnalytical())

initpo_bar = BK.R(probPsh, [0, 0.4], 1)

BK.E(probPsh, [1.0,], 1)
initpo_bar = [0.4]

@info "Test evaluation"
probPsh(initpo_bar, par_hopf)

# test of the analytical formula for jacobian of the functional
_Jad = BifurcationKit.finite_differences( x -> probPsh(x, par_hopf), initpo_bar)
_Jana = probPsh(Val(:JacobianMatrix), initpo_bar, par_hopf)
@test norm(_Jad - _Jana, Inf) < 3e-3

@info "Test newton"
ls = DefaultLS()
eil = EigKrylovKit(dim = 1, x₀ = rand(1))
optn = NewtonPar(verbose = false, tol = 1e-8,  max_iterations = 140, linsolver = ls, eigsolver = eil)
deflationOp = BK.DeflationOperator(2, dot, 1.0, [zero(initpo_bar)])
outpo = newton(probPsh, initpo_bar, optn; normN = norminf)
@test BK.converged(outpo)

BK.getperiod(probPsh, outpo.u, par_hopf)
BK.get_periodic_orbit(probPsh, outpo.u, par_hopf)

probPsh = PoincareShootingProblem(prob, Rodas4(),
        # probMono, Rodas4(autodiff=false),
        normals, centers; abstol = 1e-10, reltol = 1e-9,
        lens = (@optic _.r))

BK.residual(probPsh, outpo.u, par_hopf)
BK.jvp(probPsh, outpo.u, par_hopf, outpo.u)
# probPsh([0.30429879744900434], par_hopf)
# probPsh([0.30429879744900434], (r = 0.09243096156871472, μ = 0.0, ν = 1.0, c3 = 1.0, c5 = 0.0))
# BK.evolve(probPsh.flow,[0.0, 0.30429879744900434], (r = 0.094243096156871472, μ = 0.0, ν = 1.0, c3 = 1.0, c5 = 0.0), Inf64) # this gives an error in DiffEqBase

@info "Test continuation"
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.015, ds= 0.01, p_max = 4.0, max_steps = 5, newton_options = setproperties(optn; tol = 1e-7, eigsolver = eil), detect_bifurcation = 0)
br_pok2 = continuation(probPsh, outpo.u, PALC(),
    opts_po_cont; verbosity = 0,
    plot = false, normC = norminf)
# plot(br_pok2)
BK.setparam(br_pok2.prob, 1.)
BK.getperiod(br_pok2.prob.prob, br_pok2.sol[1].x, br_pok2.sol[1].p)
BK.get_time_slices(br_pok2.prob.prob, br_pok2.sol[1].x)
BK.get_periodic_orbit(br_pok2, 1)
####################################################################################################
@info "Multiple Poincaré Shooting"
# normals = [[-1., 0.], [1, -1]]
# centers = [zeros(2), zeros(2)]
# initpo_bar = [1.04, -1.04/√2]

normals = [[-1., 0.], [1, 0]]
centers = [zeros(2), zeros(2)]
initpo_bar = [0.2, -0.2]

probPsh = PoincareShootingProblem(prob, ODE.KenCarp4(), normals, centers; abstol=1e-10, reltol=1e-9, lens = (@optic _.r), jacobian = BK.AutoDiffDenseAnalytical())
# version with analytical jacobian
probPsh2 = PoincareShootingProblem(prob, ODE.KenCarp4(), normals, centers; abstol=1e-10, reltol=1e-9, δ = 0, lens = (@optic _.r), jacobian = BK.AutoDiffDenseAnalytical())

# test of the analytical formula for jacobian of the functional
_Jad = BifurcationKit.finite_differences( x-> probPsh(x, par_hopf), initpo_bar)
# _Jphifd = BifurcationKit.finiteDifferences(x->probPsh.flow(x, par_hopf, Inf64), [0,0.4]; δ=1e-8)
_Jana = probPsh(Val(:JacobianMatrix), initpo_bar, par_hopf)
@test norm(_Jad - _Jana, Inf) < 1e-5

ls = DefaultLS()
eil = EigKrylovKit(dim = 1, x₀=rand(1))
optn = NewtonPar(verbose = false, tol = 1e-9,  max_iterations = 140, linsolver = ls, eigsolver = eil)
deflationOp = DeflationOperator(2.0, dot, 1.0, [zero(initpo_bar)])

outpo = newton(probPsh2, initpo_bar, optn; normN = norminf)
@test BK.converged(outpo)

outpo = newton(probPsh, initpo_bar, optn; normN = norminf)
@test BK.converged(outpo)

getperiod(probPsh, outpo.u, par_hopf)

opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.025, ds= -0.01, p_max = 4.0, max_steps = 5, newton_options = (@set optn.tol = 1e-9), detect_bifurcation = 3, nev = 2)
br_pok2 = continuation(probPsh, outpo.u, PALC(tangent = Bordered()),
        opts_po_cont; verbosity = 0,
        plot = false, normC = norminf)
@test br_pok2.prob isa BK.WrapPOSh
@test br_pok2.period[1] ≈ 2pi rtol = 1e-7
####################################################################################################
# @info "Multiple Poincaré Shooting 2"
# normals = [[-1., 0.], [1, 0], [0, 1]]
# centers = [zeros(2), zeros(2), zeros(2)]
# initpo = [[0., 0.4], [0, -.3], [0.3, 0]]
#
# probPsh = PoincareShootingProblem(prob, ODE.KenCarp4(), normals, centers; abstol=1e-10, reltol=1e-9, lens = (@optic _.r), jacobian = :autodiffDenseAnalytical)
#
# initpo_bar = reduce(vcat, [BK.R(probPsh, initpo[ii], ii) for ii in eachindex(centers)])
# # same with projection function
# initpo_bar = reduce(vcat, BK.projection(probPsh, initpo))
#
# # test of the other projection function
# BK.projection(probPsh, reduce(hcat, initpo)')
#
# probPsh(initpo_bar, par_hopf; verbose = true)
#
# # test of the analytical formula for jacobian of the functional
# _Jad = BifurcationKit.finiteDifferences( x -> probPsh(x, par_hopf), initpo_bar)
# _Jana = probPsh(Val(:JacobianMatrix), initpo_bar, par_hopf)
# @test norm(_Jad - _Jana, Inf) < 5e-5
#
# outpo = newton(probPsh, initpo_bar, optn; normN = norminf)
# BK.converged(outpo)
#
# for ii=eachindex(normals)
#     BK.E(probPsh, [outpo.u[ii]], ii)
# end
#
# getPeriod(probPsh, outpo.u, par_hopf)
#
# opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.025, ds= -0.005, p_max = 4.0, max_steps = 10, newton_options = setproperties(optn; tol = 1e-8), detect_bifurcation = 3)
#     br_hpsh = continuation(probPsh, outpo.u, PALC(),
#         opts_po_cont; normC = norminf)
#
# @test br_hpsh.prob isa BK.WrapPOSh
# @test br_hpsh.period[1] ≈ 2pi rtol = 1e-7
####################################################################################################
@info "Multiple Poincaré Shooting aBS"
# test automatic branch switching with most possible options
# calls with analytical jacobians
br_psh = continuation(br, 1, (@set opts_po_cont.ds = 0.005), PoincareShootingProblem(2, prob, KenCarp4(); abstol=1e-10, reltol=1e-9, parallel = true, lens = @optic _.r); normC = norminf)
@test br_psh.prob isa BK.WrapPOSh
@test br_psh.period[1] ≈ 2pi rtol = 1e-7

# test Iterative Floquet eigen solver
@reset opts_po_cont.newton_options.eigsolver.dim = 20
@reset opts_po_cont.newton_options.eigsolver.x₀ = rand(2)
br_sh = continuation(br, 1, ContinuationPar(opts_po_cont; ds = 0.005, save_sol_every_step = 1), ShootingProblem(2, prob, KenCarp4(); abstol=1e-10, reltol=1e-9, lens = @optic _.r); normC = norminf)
@test br_psh.prob isa BK.WrapPOSh
@test br_psh.period[1] ≈ 2pi rtol = 1e-7

# test MonodromyQaD
# BK.MonodromyQaD(br_sh.functional, br.sol)

ls = GMRESIterativeSolvers(reltol = 1e-7, N = length(initpo_bar), maxiter = 500, verbose = false)
@reset opts_po_cont.detect_bifurcation = 0
@reset opts_po_cont.newton_options.linsolver = ls
@reset opts_po_cont.save_sol_every_step = 1

for M in (1,2), jacobianPO in (BK.AutoDiffMF(), BK.MatrixFree(), BK.AutoDiffDenseAnalytical(), BK.FiniteDifferences())
    @info M, jacobianPO

    # specific to Poincaré Shooting
    jacPOps = jacobianPO isa BK.AutoDiffMF ? BK.FiniteDifferences() : jacobianPO
    _parallel = jacPOps isa BK.MatrixFree ? false : false

    local br_psh = continuation(br, 1,(@set opts_po_cont.ds = 0.005), 
            PoincareShootingProblem(M, prob, Rodas4P(); abstol=1e-10, reltol=1e-9, parallel = _parallel, jacobian = jacPOps, update_section_every_step = 2); 
            normC = norminf,
            linear_algo = BorderingBLS(solver = (@set ls.N = M), check_precision = false),
            verbosity = 0)

    local br_ssh = continuation(br, 1, (@set opts_po_cont.ds = 0.005),
            ShootingProblem(M, prob, Rodas4P(); abstol=1e-10, reltol=1e-9, parallel = _parallel, jacobian = jacobianPO, update_section_every_step = 2); 
            normC = norminf,
            linear_algo = BorderingBLS(solver = (@set ls.N = 2M + 1), check_precision = false), 
            verbosity = 0)

    # test different versions of newton
    newton(br_ssh.prob.prob, br_ssh.sol[1].x, br_ssh.contparams.newton_options)
end
