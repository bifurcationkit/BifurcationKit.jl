# using Revise
using BifurcationKit
using Test, LinearAlgebra
# using Plots
const BK = BifurcationKit
####################################################################################################
function COm(u, p)
    (;q1,q2,q3,q4,q5,q6,k) = p
    x, y, s = u
    z = 1-x-y-s
    out = similar(u, promote_type(eltype(u), eltype(q2), eltype(k)))
    out[1] = 2 * q1 * z^2 - 2 * q5 * x^2 - q3 * x * y
    out[2] = q2 * z - q6 * y - q3 * x * y
    out[3] = q4 * z - k * q4 * s
    out
end

par_com = (q1 = 2.5, q2 = 2.0, q3 = 10., q4 = 0.0675, q5 = 1., q6 = 0.1, k = 0.4)
z0 = [0.07,0.2,05]

prob = BifurcationProblem(COm, z0, par_com, (@optic _.q2); record_from_solution = (x, p; k...) -> (x = x[1], y = x[2], s = x[3]))

opts_br = ContinuationPar(p_min = 0.6, p_max = 2.5, ds = 0.002, dsmax = 0.01, n_inversion = 4, detect_bifurcation = 3, max_bisection_steps = 25, nev = 2, max_steps = 20000)

# @reset opts_br.newton_options.verbose = true
alg = PALC()
br = @time continuation(prob, alg, opts_br;
    plot = false, verbosity = 0, normC = norminf,
    bothside = true)

# plot(br, plotfold=false, markersize=4, legend=:topleft)
####################################################################################################
hp = newton(br, 2; options = NewtonPar( opts_br.newton_options; max_iterations = 10), start_with_eigen = false)
hp = newton(br, 2; options = NewtonPar( opts_br.newton_options; max_iterations = 10), start_with_eigen = true)

hpnf = get_normal_form(br, 5)
@test hpnf.nf.b |> real ≈ 1.070259e+01 rtol = 1e-3

hpnf = get_normal_form(br, 2)
@test hpnf.nf.b |> real ≈ 4.332247e+00 rtol = 1e-2

show(stdout, BK.FoldProblemMinimallyAugmented(prob))
BK.HopfProblemMinimallyAugmented(prob)
BK.PeriodDoublingProblemMinimallyAugmented(prob)
BK.NeimarkSackerProblemMinimallyAugmented(prob)
####################################################################################################
# different tests for the Fold point
@reset opts_br.newton_options.verbose = false
@reset opts_br.newton_options.max_iterations = 10

sn = newton(br, 3; options = opts_br.newton_options, bdlinsolver = MatrixBLS())
# printstyled(color=:red, "--> guess for SN, p = ", br.specialpoint[2].param, ", psn = ", sn[1].p)
    # plot(br);scatter!([sn.x.p], [sn.x.u[1]])
@test BK.converged(sn) && sn.itlineartot == 6
@test sn.u.u ≈ [0.05402941507127516, 0.3022414400400177, 0.45980653206336225] rtol = 1e-4
@test sn.u.p ≈ 1.0522002878699546 rtol = 1e-4

sn = newton(br, 3; options = opts_br.newton_options, bdlinsolver = MatrixBLS())
@test BK.converged(sn) && sn.itlineartot == 6

sn = newton(br, 3; options = opts_br.newton_options, bdlinsolver = MatrixBLS(), start_with_eigen = true)
@test BK.converged(sn) && sn.itlineartot == 6

for eigen_start in (true, false), _jac in (BK.AutoDiff(), BK.FiniteDifferences(), BK.MinAugMatrixBased(), BK.MinAug())
    # @info "" eigen_start _jac
    sn_br = continuation(br, 3, (@optic _.k), ContinuationPar(opts_br, p_max = 1., p_min = 0., detect_bifurcation = 1, max_steps = 50, save_sol_every_step = 1, detect_event = 2), 
            bdlinsolver = MatrixBLS(), 
            start_with_eigen = eigen_start, 
            update_minaug_every_step = 1,
            detect_codim2_bifurcation = 2,
            jacobian_ma = _jac
            )
    @test sn_br.kind isa BK.FoldCont
    @test sn_br.specialpoint[1].type == :bt
    @test sn_br.specialpoint[1].param ≈ 0.9716038596420551 rtol = 1e-5
    @test ~isnothing(sn_br.eig)

    # we test the jacobian and problem update
    par_sn = BK.setparam(br, BK.getp(sn_br.sol[end].x))
    par_sn = BK.set(par_sn, BK.getlens(sn_br), sn_br.sol[end].p)
    _J = BK.jacobian(prob, BK.getvec(sn_br.sol[end].x), par_sn)
    _eigvals, eigvec, = eigen(_J)
    ind = argmin(abs.(_eigvals))
    @test _eigvals[ind] ≈ 0 atol = 1e-10
    ζ = eigvec[:, ind]
    @test sn_br.prob.prob.b ./ norm(sn_br.prob.prob.b) ≈ ζ * sign(ζ[1])*sign(sn_br.prob.prob.b[1])

    _eigvals, eigvec, = eigen(_J')
    ind = argmin(abs.(_eigvals))
    ζstar = eigvec[:, ind]
    @test sn_br.prob.prob.a ≈ ζstar * sign(ζstar[1])*sign(sn_br.prob.prob.a[1])
end
####################################################################################################
# different tests for the Hopf point
hppt = get_normal_form(br, 2)
@test hppt.nf.a ≈ 2.546719962189168 + 1.6474887797814664im
@test hppt.nf.b ≈ 4.3536804635557855 + 15.441272421860365im

@reset opts_br.newton_options.verbose = false

hp = BK.newton_hopf(br, 2; options = opts_br.newton_options, start_with_eigen = true)
# printstyled(color=:red, "--> guess for HP, p = ", br.specialpoint[1].param, ", php = ", hp[1].p)
# plot(br);scatter!([hp[1].p[1]], [hp[1].u[1]])
@test hp.converged && hp.itlineartot == 8

hp = BK.newton_hopf(br, 2; options = opts_br.newton_options, start_with_eigen = false, bdlinsolver = MatrixBLS())
@test hp.converged && hp.itlineartot == 12

hp = BK.newton_hopf(br, 2; options = opts_br.newton_options, start_with_eigen = true, bdlinsolver = MatrixBLS(), verbose = true)
@test hp.converged && hp.itlineartot == 8

# we check that we truly have a bifurcation point.
pb = hp.prob.prob
ω = hp.u.p[2]
par_hp = BK.set(BK.getparams(br), BK.getlens(br), hp.u.p[1])
_J = BK.jacobian(pb.prob_vf.VF, hp.u.u, par_hp)
_eigvals, eigvec, = eigen(_J)
ind = argmin(abs.(_eigvals .- Complex(0, ω)))
@test real(_eigvals[ind]) ≈ 0 atol=1e-9
@test abs(imag(_eigvals[ind])) ≈ abs(hp.u.p[2]) atol=1e-9
ζ = eigvec[:, ind]
# reminder: pb.b should be a null vector of (J+iω)
@test pb.b ≈ ζ atol = 1e-3

hp = newton(br, 2;
    options = NewtonPar( opts_br.newton_options; max_iterations = 10),
    start_with_eigen = true,
    bdlinsolver = MatrixBLS())
# printstyled(color=:red, "--> guess for HP, p = ", br.specialpoint[1].param, ", php = ", hp.p)
# plot(br);scatter!([hp.p[1]], [hp.u[1]])

hp = newton(br, 2; options = NewtonPar( opts_br.newton_options; max_iterations = 10),start_with_eigen=true)

for eigen_start in (true, false), _jac in (BK.AutoDiff(), BK.MinAugMatrixBased(), BK.MinAug())
    # @info "" eigen_start _jac
    hp_br = continuation(br, 2, (@optic _.k), 
            ContinuationPar(opts_br, ds = -0.001, p_max = 1., p_min = 0., detect_bifurcation = 1, max_steps = 50, save_sol_every_step = 1, detect_event = 2), bdlinsolver = MatrixBLS(), 
            start_with_eigen = eigen_start, 
            update_minaug_every_step = 1, 
            verbosity = 0, 
            detect_codim2_bifurcation = 2,
            jacobian_ma = _jac,
            plot=false)
    @test hp_br.kind isa BK.HopfCont
    @test hp_br.specialpoint[1].type == :gh
    @test hp_br.specialpoint[2].type == :gh

    @test hp_br.specialpoint[1].param ≈ 0.305873681159479 rtol = 1e-5
    @test hp_br.specialpoint[2].param ≈ 0.23255761094689315 atol = 1e-4

    @test ~isnothing(hp_br.eig)
end
