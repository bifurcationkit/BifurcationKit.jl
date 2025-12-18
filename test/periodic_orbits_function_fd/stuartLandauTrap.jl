# using Revise, Plots
using Test
using BifurcationKit, LinearAlgebra, ForwardDiff, SparseArrays
const BK = BifurcationKit
##################################################################
# The goal of these tests is to test all combinations of options
##################################################################
function Fsl!(f, u, p, t)
    (;r, μ, ν, c3) = p
    u1 = u[1]
    u2 = u[2]

    ua = u1^2 + u2^2

    f[1] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2)
    f[2] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1)
    return f
end

Fsl(x, p) = Fsl!(similar(x), x, p, 0.)
####################################################################################################
par_sl = (r = 0.5, μ = 0., ν = 1.0, c3 = 1.0)
u0 = [.001, .001]
par_hopf = (@set par_sl.r = 0.1)
prob = BK.BifurcationProblem(Fsl, u0, par_hopf, (@optic _.r))
####################################################################################################
# continuation, Hopf bifurcation point detection
optconteq = ContinuationPar(ds = -0.01, detect_bifurcation = 3, p_min = -0.5, n_inversion = 8)
br = continuation(prob, PALC(), optconteq)
####################################################################################################
prob2 = BK.BifurcationProblem(Fsl, u0, par_hopf, (@optic _.r); J = (x, p) -> sparse(ForwardDiff.jacobian(z -> Fsl(z, p), x)))# we put sparse to try the different linear solvers
poTrap = PeriodicOrbitTrapProblem(
                        prob2,
                        [1., 0],
                        zeros(2),
                        100, 2; update_section_every_step = 1)

show(poTrap)
BK.isinplace(poTrap)
try
    BK.has_hessian(poTrap)
catch
end

# guess for the periodic orbit
orbitguess_f = reduce(vcat, [√(getparams(poTrap).r) .* [cos(θ), sin(θ)] for θ in LinRange(0, 2pi, poTrap.M)])
push!(orbitguess_f, 2pi)

optn_po = NewtonPar(tol = 1e-9, verbose = true)
opts_po_cont = ContinuationPar(dsmax = 0.02, ds = 0.001, p_max = 2.2, max_steps = 3, newton_options = optn_po, save_sol_every_step = 1, detect_bifurcation = 1)

let
    lsdef = DefaultLS()
    lsit = GMRESKrylovKit()
    for (ind, jacobianPO) in enumerate((BK.Dense(),
                                        BK.AutoDiffDense(),
                                        BK.FullLU(),
                                        BK.BorderedLU(),
                                        BK.FullSparseInplace(),
                                        BK.BorderedSparseInplace(),
                                        BK.FullMatrixFree(),
                                        BK.AutoDiffMF(),
                                        BK.BorderedMatrixFree()))
        _ls = ind > 6 ? lsit : lsdef
        @info jacobianPO, ind, _ls
        outpo_f = newton((@set poTrap.jacobian = jacobianPO),
            orbitguess_f, (@set optn_po.linsolver = _ls);
            normN = norminf)
        @test BK.converged(outpo_f)

        for eig in (FloquetQaD(optn_po.eigsolver),)
            br_po = continuation(
                (@set poTrap.jacobian = jacobianPO), 
                outpo_f.u,
                PALC(),
                (@set opts_po_cont.newton_options.linsolver = _ls);
                # verbosity = 0, plot = false,
                eigsolver = eig,
                linear_algo = BorderingBLS(solver = _ls, check_precision = false),
                normC = norminf)
            BK.get_periodic_orbit(br_po, 1)
            @test _test_sorted(BK.eigenvals(br_po, 1))

            for k in 1:length(br_po)-1
                _eigvals = br_po[k].eigenvals
                μ1_bk = minimum(real, _eigvals)
                valid = minimum(abs, _eigvals) < 1e-9 # Floquet exponent not precise?
                μ1 = -2*br_po[k].param*(br_po[k].period)
                # @error "" k _eigvals μ1_bk μ1 eig
                # we have large atol because the method is not that precise, ~1/M
                @test isapprox(μ1_bk, μ1, atol = 5e-2) || (eig isa EigArnoldiMethod) || (eig isa EigArpack) || ~valid
            end
        end
    end
end

let poTrap = poTrap
    @reset poTrap.jacobian = BK.Dense()
    outpo_f = newton(poTrap,
            orbitguess_f, (@set optn_po.linsolver = DefaultLS());
            normN = norminf)
    br_po = continuation(
                poTrap, 
                outpo_f.u,
                PALC(bls =  BorderingBLS(solver = DefaultLS(), check_precision = false)),
                (@set opts_po_cont.newton_options.linsolver = DefaultLS());
                # verbosity = 2, plot = false,
                # eigsolver = FloquetGEV(optn_po.eigsolver, length(poTrap), 2),
                normC = norminf)

        for k in 1:length(br_po)-1
            _eigvals = br_po[k].eigenvals
            μ1_bk = minimum(real, _eigvals)
            μ1 = -2*br_po[k].param*(br_po[k].period)
            # we have large atol because the method is not that precise, ~1/M
            @test μ1_bk ≈ μ1 atol = 5e-2
        end


    outpo_f = @time newton(poTrap, orbitguess_f, optn_po);
    outpo = reshape(outpo_f.u[1:end-1], 2, poTrap.M)

    # computation of the Jacobian at out_pof
    _J1 = poTrap(Val(:JacFullSparse), outpo_f.u, par_hopf)
    _Jfd = ForwardDiff.jacobian(z-> BK.residual(poTrap, z, par_hopf), outpo_f.u)

    # test of the jacobian against automatic differentiation
    @test norm(_Jfd - Array(_J1), Inf) < 1e-7

    # tests for constructor of Floquet routines
    for eig in (EigArpack(), EigArnoldiMethod(), EigKrylovKit())
        BK._check_floquet_options(eig)
    end
    FloquetQaD(EigKrylovKit()) |> FloquetQaD

    # test of the Floquet exponents
    eig = FloquetQaD(DefaultEig())
    # eig(Array(_J1), 2)
end
