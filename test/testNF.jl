# using Revise
using Plots
using Test
using BifurcationKit, LinearAlgebra
const BK = BifurcationKit

Fbp(x, p) = [x[1] * (3.23 .* p.μ - p.x2 * x[1] + p.x3 * x[1]^2) + x[2], 
            -x[2] + p.γ * x[1]^2]
####################################################################################################
let
    opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, p_max = 0.4, p_min = -0.5, detect_bifurcation = 3, newton_options = NewtonPar(tol = 1e-14), max_steps = 100, n_inversion = 8)

    prob = ODEBifProblem(Fbp, [0., 0], (μ = -0.2, ν = 0, x2 = 1.12, x3 = 0.234, γ = 4.4323), (@optic _.μ))
    br = continuation(prob, PALC(), opts_br; normC = norminf)
    plot(br)

    @test br.specialpoint[1].interval[1] < 0
    @test br.specialpoint[1].interval[2] > 0

    ##################################################
    # normal form computation
    # we set the bifurcation point for exact computations
    @reset br.specialpoint[1].param = 0.
    bp = BK.get_normal_form(br, 1; verbose = true, detailed = false, ζs = [1., 0], ζs_ad = [1., 1])
    # on that case, the correction is Ψ(x⋅ζ) = [0, γ⋅x²]
    @test bp.nf.Ψ20 ≈ [0, 2prob.params.γ]
    # normal form
    nf = bp.nf

    @test nf.a01 ≈ 0         atol = 1e-10
    @test nf.b11 ≈ 3.23      atol = 1e-10
    @test nf.b20/2 ≈ -prob.params.x2 + prob.params.γ   atol = 1e-10
    @test nf.b30/6 ≈ prob.params.x3   atol = 1e-10

    # test normal form predictor
    pred = predictor(bp, 0.1)
    @test norm(pred.x0) < 1e-10
    # @test pred.x1[1] ≈ 3.23 * 0.1 / prob.params.x2 rtol=1e-5

    bp = BK.get_normal_form(br, 1; verbose=false)
    @test BK.istranscritical(bp) == true
    @test BK.type(bp) == :Transcritical
    @test BK.type(nothing) == nothing

    prob2 = @set prob.VF.J = (x, p) -> BK.finite_differences(z -> Fbp(z, p), x)
    bp = BK.get_normal_form(prob2, br, 1; verbose = false, autodiff = false)
    @test BK.istranscritical(bp) == true
    show(bp)
end
####################################################################################################
# same but when the eigenvalues are not saved in the branch but computed on the fly
let
    prob = ODEBifProblem(Fbp, [0., 0], (μ = -0.2, ν = 0, x2 = 1.12, x3 = 0.234, γ = 0.), (@optic _.μ))
    opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, p_max = 0.4, p_min = -0.5, detect_bifurcation = 3, newton_options = NewtonPar(tol = 1e-14), max_steps = 100, n_inversion = 8)
    br_noev = BK.continuation(prob, PALC(), (@set opts_br.save_eigenvectors = false); normC = norminf)
    @test BK.haseigenvector(br_noev) == false
    bp = BK.get_normal_form(br_noev, 1; verbose=false, autodiff = true)
    bp = BK.get_normal_form(br_noev, 1; verbose=false)
    BK._predictor(bp, 0.01)
    nf = bp.nf
    @test nf.a01   ≈ 0       atol = 1e-10
    @test nf.b11   ≈ 3.23    atol = 1e-10
    @test nf.b20/2 ≈ -1.12   atol = 1e-10
    @test nf.b30/6 ≈ 0.234   atol = 1e-10
    ######################################################
    # Automatic branch switching
    br2 = continuation(br_noev, 1, setproperties(opts_br; p_max = 0.2, ds = 0.01, max_steps = 14))
    @test br2 isa Branch
    @test BK.haseigenvalues(br2) == true
    @test BK.haseigenvector(br2) == true
    BK.eigenvals(br2, 1, true)
    BK._getfirstusertype(br2)
    @test length(br2) == 12
    get_normal_form(br2, 1)
    plot(br_noev, br2)

    br3 = continuation(br_noev, 1, ContinuationPar(opts_br; ds = -0.01); usedeflation = true)
    @test isnothing(BK.multicontinuation(br_noev, 1))

    br4 = continuation(br_noev, 1, ContinuationPar(opts_br; p_max = 0.2, ds = 0.01, max_steps = 14); bothside = true)

    # automatic bifurcation diagram (Transcritical)
    bdiag = bifurcationdiagram(prob, PALC(), 2,
        ContinuationPar(opts_br; p_min = -.2, p_max = .2, ds = 0.01, newton_options = NewtonPar(tol = 1e-12), max_steps = 15);
        normC = norminf)

    plot(bdiag)

    # same from the non-trivial branch
    prob = BK.ODEBifProblem(Fbp, [-0.5, 0.], (μ = -0.2, ν = 0, x2 = 1.12, x3 = 1.0, γ = 0), (@optic _.μ))
    br = continuation(prob, PALC(), ContinuationPar(opts_br, n_inversion = 10); normC = norminf)
    bp = BK.get_normal_form(br, 1; verbose=false)
    nf = bp.nf
    @test nf.a01   ≈ 0                 atol = 1e-6
    @test nf.b11   ≈ 3.23              atol = 1e-10
    @test nf.b20/2 ≈ -prob.params.x2   atol = 1e-6
    @test nf.b30/6 ≈  prob.params.x3   atol = 1e-10
    br2 = continuation(br, 1, ContinuationPar(opts_br; p_max = 0.2, ds = 0.01, max_steps = 14); bothside = true)
    plot(br, br2)
end
####################################################################################################
# Case of the pitchfork like
let
    par_pf = setproperties((μ = -0.2, ν = 0, x2 = 1.12, x3 = 0.234, γ = 0.) ; x2 = 0.0, x3 = -1.0, γ = 1.422)
    prob_pf = ODEBifProblem(Fbp, [0., 0], par_pf, (@optic _.μ);record_from_solution = (x,p;k...)->(x[1], norm(x)))
    opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, p_max = 0.4, p_min = -0.5, detect_bifurcation = 3, newton_options = NewtonPar(tol = 1e-14), max_steps = 100, n_inversion = 8)
    brp = BK.continuation(prob_pf, PALC(tangent=Bordered()), opts_br; normC = norminf)
    bpp = BK.get_normal_form(brp, 1; verbose = true)
    show(bpp)
    BK.type(bpp)
    BK._predictor(bpp, 0.01)

    nf = bpp.nf
    @test nf.a01 ≈ 0                     atol = 1e-10
    @test nf.a02 ≈ 0                     atol = 1e-10
    @test nf.b11 ≈ 3.23                  atol = 1e-10
    @test nf.b20/2 ≈ prob_pf.params.γ    atol = 1e-4
    @test nf.b30/6 ≈ prob_pf.params.x3   atol = 1e-10

    # test predictor
    pred = predictor(bpp, 0.1)
    @test norminf(pred.x0) < 1e-6

    # test automatic branch switching
    br2 = continuation(brp, 1, ContinuationPar(opts_br; max_steps = 19, dsmax = 0.01, ds = 0.001, detect_bifurcation = 2))
    plot(brp, br2)

    # test methods for aBS
    BK.from(br2) |> BK.type
    BK.from(br2) |> BK.istranscritical
    BK.type(nothing)
    BK.show(stdout, br2)
    BK.propertynames(br2)

    # automatic bifurcation diagram (Pitchfork)
    bdiag = bifurcationdiagram(prob_pf, PALC(#=tangent=Bordered()=#), 2,
        setproperties(opts_br; p_min = -1.0, p_max = .5, ds = 0.01, dsmax = 0.05, n_inversion = 6, detect_bifurcation = 3, max_bisection_steps = 30, newton_options = NewtonPar(tol = 1e-12), max_steps = 15);
        verbosediagram = true, normC = norminf)
    BK.getalg(bdiag)
    BK.get_solution(bdiag, 1)
    bdiag[1]
    plot(bdiag)
end
####################################################################################################
# Automatic branch switching, degenerate case. The quadratic parameter makes it difficult for aBS 
# based on the order one (in parameter) predictor
let
    Fdegenerate(x, p) = [x[1]^2 - p[1]^2]
    prob = BK.BifurcationProblem(Fdegenerate, [1.], (-1.0), 1; record_from_solution = (x,p;k...)->x[1])
    br = continuation(prob, PALC(), ContinuationPar(n_inversion = 6); normC = norminf)
    br1 = continuation(br, 1, verbosity = 3, bothside = true)
    bp = get_normal_form(br, 1; verbose = true)
    show(bp)
    predictor(bp, 0.1)
    # plot(br, br1)
end
####################################################################################################
let
    function symmetrize3!(a)
        n1, n2, n3 = size(a)
        @assert n1 == n2 == n3
        n = n1

        for i in 1:n, j in 1:n, k in 1:n
            v = (
                a[i,j,k] + a[i,k,j] +
                a[j,i,k] + a[j,k,i] +
                a[k,i,j] + a[k,j,i]
            ) / 6
            a[i,j,k] = a[i,k,j] = a[j,i,k] = a[j,k,i] = a[k,i,j] = a[k,j,i] = v
        end

        return a
    end
    P = rand(2,2)
    vf = BK.NdBranchPoint(zeros(3), 0., 0., (μ = -0., ), (@optic _.μ), 
                            [[1., 0], [0., 1]], 
                            [[1., 0], [0., 1]], 
                            (a01 = zeros(2),
                                a02 = zeros(2),
                                b11 = P\diagm([-1,-.123])*P,
                                b20 = rand(2,2,2),
                                b30 = rand(2,2,2,2)),
                            :none)
    vf.nf.b20[1,:,:] .= Symmetric(vf.nf.b20[1,:,:])
    vf.nf.b20[2,:,:] .= Symmetric(vf.nf.b20[2,:,:])
    symmetrize3!(@view vf.nf.b30[1,:,:,:])
    symmetrize3!(@view vf.nf.b30[2,:,:,:])
    prob2d = BK.BifurcationProblem((x,p)->vf(Val(:reducedForm), x, p[1]), [0., 0], [-.1], 1)
    opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, p_max = 0.4, p_min = -0.5, detect_bifurcation = 3, newton_options = NewtonPar(tol = 1e-14), max_steps = 100, n_inversion = 8)
    br = continuation(prob2d, PALC(), ContinuationPar(opts_br; n_inversion = 8, dsmax=0.01))
    @reset br.specialpoint[1].param = 0.

    for give_ad in (true, false)
        bp2d = BK.get_normal_form(br, 1; 
                                        ζs = [[1, 0.], [0, 1.]],
                                        ζs_ad = give_ad ? [[1, 0], [0, 1]] : nothing, 
                                        autodiff = true, 
                                        verbose = true)
        @info give_ad
        @test vf.nf.a01 ≈ bp2d.nf.a01 atol = 1e-10
        # @test vf.nf.a02 ≈ bp2d.nf.a02 atol = 1e-10
        @test vf.nf.b11 ≈ bp2d.nf.b11 atol = 1e-10
        @test vf.nf.b20 ≈ bp2d.nf.b20 atol = 1e-10
        @test vf.nf.b30 ≈ bp2d.nf.b30 atol = 1e-10
    end
end
####################################################################################################
function Fbp2d!(out, x, p)
    out[1] = p.α * x[1] * (3.23 .* p.μ + p.A * x[1]^2 + p.B * x[2]^2) + x[3]
    out[2] = p.α * x[2] * (3.23 .* p.μ + p.C * x[1]^2 + p.A * x[2]^2)
    out[3] = -x[3] + p.γ * (x[1]^3 + x[2]^2)
    return out
end
Fbp2d(x, p) = Fbp2d!(similar(x .* p.μ), x, p)
# on that case, the correction is Ψ(x₁⋅ζ₁+x₂⋅ζ₂) = [0, 0, γ⋅(x₁³ + x₂²)]

let
    # test inplace, out-of-place etc
    for α in (-1.,), saveev in (true, ), _F in (Fbp2d!, ), autodiff in (true, ), γ in (0., 10.)

        par_2d = (μ = -0.2, ν = 0., α = α, γ = γ, A = 0.123, B = 0.234, C = 0.456)
        prob2d = BK.BifurcationProblem(_F, [0., 0, 0], par_2d, (@optic _.μ))
        opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, p_max = 0.4, p_min = -0.5, detect_bifurcation = 3, newton_options = NewtonPar(tol = 1e-14), max_steps = 100, n_inversion = 8)
        br = continuation(prob2d, PALC(), ContinuationPar(opts_br; n_inversion = 8, save_eigenvectors = saveev);
            plot = false, verbosity = 0, normC = norminf)
        # we have to be careful to have the same basis as for Fbp2d or the NF will not match Fbp2d
        @reset br.specialpoint[1].param = 0.
        bp2d = BK.get_normal_form(br, 1; verbose = true, detailed = Val(false));
        bp2d = BK.get_normal_form(br, 1; verbose = true);
        bp2d = BK.get_normal_form(br, 1; 
                                    ζs = [[1, 0, 0.], [0, 1, 0.]],
                                    ζs_ad = [[1, 0, 1], [0., 1, 0]], 
                                    autodiff, 
                                    verbose = false)

        @test bp2d.ζ[1] ≈ [1, 0, 0.]
        @test bp2d.ζ[2] ≈ [0, 1, 0.]

        show(bp2d)
        BK.type(bp2d)
        BK._get_string(bp2d)
        length(bp2d)
        bp2d(rand(2), 0.2)
        bp2d(Val(:reducedForm), rand(2), 0.2)
        predictor(bp2d, 0.01)


        _o1 = bp2d(Val(:reducedForm), [1., 0], 0.0)


        bp2d.nf.a02 .= 0
        for _x0 in (vcat(rand(2),0), [1.,0.,0], [0.,1.,0])
            _o1 = bp2d(Val(:reducedForm), _x0[1:2], 0.0)
            _o2 = Fbp2d(_x0, @set par_2d.μ = 0.)[1:2] + par_2d.γ * [_x0[1]^3 + _x0[2]^2,0]
            @test norminf(_o1 - _o2) < 1e-9
        end

        @test bp2d.nf.b30[1,1,1,1] / 6 ≈ prob2d.params.α * prob2d.params.A +
                                         prob2d.params.γ                       atol = 1e-10
        @test bp2d.nf.b30[1,1,2,2] / 2 ≈ prob2d.params.α * prob2d.params.B     atol = 1e-10
        @test bp2d.nf.b30[1,1,1,2] / 2 ≈ prob2d.params.α * 0.0                 atol = 1e-10
        @test bp2d.nf.b30[2,1,1,2] / 2 ≈ prob2d.params.α * prob2d.params.C     atol = 1e-10
        @test norminf(bp2d.nf.b20[:,:,1]) < 1e-10
        @test bp2d.nf.b20[1,:,:]/2 ≈ [0 0; 0 prob2d.params.γ]                  atol = 1e-10
        @test norminf(bp2d.nf.b11 - prob2d.params.α * 3.23 * I) ≈ 0            atol = 1e-10
        @test norminf(bp2d.nf.a01) < 1e-10
    end
end
####################################################################################################
# vector field to test nearby secondary bifurcations
let
    FbpSecBif(u, p) = @. -u * (p + u * (2-5u)) * (p -0.15 - u * (2+20u))
    prob = BK.BifurcationProblem(FbpSecBif, [0.0], -0.2,  (@optic _))
    opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, p_max = 0.4, p_min = -0.5, detect_bifurcation = 3, newton_options = NewtonPar(tol = 1e-14), max_steps = 100, n_inversion = 8)
    br_snd1 = BK.continuation(prob, PALC(),
        ContinuationPar(opts_br; p_min = -1.0, p_max = .3, ds = 0.001, dsmax = 0.005, n_inversion = 8, detect_bifurcation=3); normC = norminf)

    plot(br_snd1)

    br_snd2 = BK.continuation(
        br_snd1, 1,
        ContinuationPar(opts_br; p_min = -1.2, p_max = 0.2, ds = 0.001, detect_bifurcation = 3, max_steps=31, n_inversion = 8, newton_options = NewtonPar(opts_br.newton_options; verbose = false), dsmin_bisection =1e-18, tol_bisection_eigenvalue=1e-11, max_bisection_steps=20); normC = norminf,
        # finalise_solution = (z, tau, step, contResult) ->
        #     (Base.display(contResult.eig[end].eigenvals) ;true)
        )

    plot(br_snd1, br_snd2, putbifptlegend=false)

    bdiag = bifurcationdiagram(prob, PALC(), 2,
        ContinuationPar(opts_br; p_min = -1.0, p_max = .3, ds = 0.001, dsmax = 0.005, n_inversion = 8, detect_bifurcation = 3, dsmin_bisection =1e-18, tol_bisection_eigenvalue=1e-11, max_bisection_steps=20);
        normC = norminf)

    # plot(bdiag; putbifptlegend=false, markersize=2, plotfold=false, title = "#branch = $(size(bdiag))")

    # test calls for aBD
    BK.level(bdiag)
    BK.hasbranch(bdiag)
    BK.from(bdiag.child[1].γ)
    BK.get_branches_from_BP(bdiag, 2)
    BK.get_contresult(br_snd2)
    BK.get_contresult(get_branch(bdiag,(1,)).γ)
    size(bdiag)
    get_branch(bdiag, (1,))
    show(stdout, bdiag)
end
####################################################################################################
# test of the pitchfork-D6 normal form
function FbpD6(x, p)
    return [ p.μ * x[1] + (p.a * x[2] * x[3] - p.b * x[1]^3 - p.c * (x[2]^2 + x[3]^2) * x[1]),
             p.μ * x[2] + (p.a * x[1] * x[3] - p.b * x[2]^3 - p.c * (x[3]^2 + x[1]^2) * x[2]),
             p.μ * x[3] + (p.a * x[1] * x[2] - p.b * x[3]^3 - p.c * (x[2]^2 + x[1]^2) * x[3])]
end

begin
    probD6 = BK.BifurcationProblem(FbpD6, zeros(3), (μ = -0.2, a = 0.3, b = 1.5, c = 2.9), (@optic _.μ),)
    opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, p_max = 0.4, p_min = -0.5, detect_bifurcation = 3, newton_options = NewtonPar(tol = 1e-14), max_steps = 100, n_inversion = 8)
    br = BK.continuation(probD6, PALC(), setproperties(opts_br; n_inversion = 6, ds = 0.001); normC = norminf)

    # plot(br;  plotfold = false)
    # we have to be careful to have the same basis as for Fbp2d or the NF will not match Fbp2d
    bp2d = BK.get_normal_form(br, 1; ζs = [[1, 0, 0.], [0, 1, 0.], [0, 0, 1.]])
    BK._get_string(bp2d)
    BK.type(bp2d)

    @test bp2d.nf.a01 == zeros(3)
    @test bp2d.nf.b11 ≈ I(3)
    @test bp2d.nf.b30[1,1,1,1] / 6 ≈ -probD6.params.b atol = 1e-10
    @test bp2d.nf.b30[1,1,2,2] / 2 ≈ -probD6.params.c atol = 1e-10
    @test bp2d.nf.b20[1,2,3] ≈ probD6.params.a atol = 1e-10

    # test the evaluation of the normal form
    x0 = rand(3); @test norm(FbpD6(x0, BK.setparam(br, 0.001))  - bp2d(Val(:reducedForm), x0, 0.001), Inf) < 1e-12

    br1 = BK.continuation(br, 1,
        setproperties(opts_br; n_inversion = 4, dsmax = 0.005, ds = 0.001, max_steps = 100, p_max = 1.); normC = norminf, verbosedeflation = false)
    # plot(br1..., br, plotfold=false, putbifptlegend=false)

    bp2d = BK.get_normal_form(br, 1)
    # res = predictor(bp2d, 0.001;  verbose = false, perturb = identity, ampfactor = 1, nbfailures = 4)
    # deflationOp = DeflationOperator(2, 1.0, [zeros(3)]; autodiff = true)

    bdiag = bifurcationdiagram(probD6, PALC(), 3,
        (args...) -> setproperties(opts_br; p_min = -0.250, p_max = .4, ds = 0.001, dsmax = 0.005, n_inversion = 4, detect_bifurcation = 3, dsmin_bisection =1e-18, tol_bisection_eigenvalue=1e-11, max_bisection_steps=20);
        normC = norminf)

    # plot(bdiag; putspecialptlegend=false, markersize=2,plotfold=false);title!("#branch = $(size(bdiag))")
end

####################################################################################################
# test of the Hopf normal form
function Fsl2!(f, u, p, t)
    (;r, μ, ν, c3, c5) = p
    u1 = u[1]
    u2 = u[2]
    ua = u1^2 + u2^2

    f[1] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2) - c5 * ua^2 * u1
    f[2] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1) - c5 * ua^2 * u2

    return f
end

Fsl2(x, p) = Fsl2!(similar(x .* p.r), x, p, 0.)

let
    for _F in (Fsl2, Fsl2!), autodiff in  (true, false)
        par_sl = (r = -0.1, μ = 0.132, ν = 1.0, c3 = 1.123, c5 = 0.2)
        probsl2 = BK.BifurcationProblem(Fsl2, zeros(2), par_sl, (@optic _.r))

        # detect hopf bifurcation
        opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.02, ds = 0.01, p_max = 0.1, p_min = -0.3, detect_bifurcation = 3, nev = 2, max_steps = 100)

        br = BK.continuation(probsl2, PALC(), opts_br; normC = norminf)

        hp = BK.get_normal_form(br, 1; detailed = Val(false))
        hp = BK.get_normal_form(br, 1; autodiff, start_with_eigen = Val(false))
        hp = BK.get_normal_form(br, 1; autodiff)
        BK.type(hp)

        nf = hp.nf
        BK.type(hp)

        @test nf.a ≈ 1  atol = 1e-9
        @test nf.b/2 ≈ (-par_sl.c3 + im*par_sl.μ)  atol = 1e-14

        # same but when the eigenvalues are not saved in the branch but computed on the fly instead
        br = BK.continuation(probsl2, PALC(), ContinuationPar(opts_br, save_eigenvectors = false); normC = norminf)
        hp = BK.get_normal_form(br, 1)
        show(hp)
        nf = hp.nf
        @test nf.a ≈ 1  atol = 1e-9
        @test nf.b/2 ≈ (-par_sl.c3 + im*par_sl.μ)  atol = 1e-14
    end
end
####################################################################################################
# test for the Cusp normal form
let
    Fcusp(x, p) = [p.β1 + p.β2 * x[1] + p.c * x[1]^3]
    par = (β1 = 0.0, β2 = -0.01, c = 3.)
    prob = BK.BifurcationProblem(Fcusp, [0.01], par, (@optic _.β1))
    br = continuation(prob, PALC(), opts_br;)

    sn_codim2 = continuation(br, 1, (@optic _.β2), ContinuationPar(opts_br, detect_bifurcation = 1, save_sol_every_step = 1, max_steps = 40) ;
        update_minaug_every_step = 1,
        bdlinsolver = MatrixBLS(),
        jacobian_ma = BK.MinAug(),
        )
    # find the cusp point
    ind = findall(map(x->x.type == :cusp, sn_codim2.specialpoint))
    cuspnf = get_normal_form(sn_codim2, ind[1])
    show(cuspnf)
    BK.type(cuspnf)
    @test cuspnf.nf.c == par.c
end
####################################################################################################
# test for the Bogdanov-Takens normal form
function Fbt!(out, x, p)
    out[1] = x[2]
    out[2] = p.β1 + p.β2 * x[2] + p.a * x[1]^2 + p.b * x[1] * x[2]
    out
end
Fbt(x, p) = Fbt!(similar(x .* p.β1), x, p)

let
    for _F in (Fbt!, Fbt)
        par = (β1 = 0.01, β2 = -0.1, a = -1., b = 1.)
        prob  = BK.BifurcationProblem(_F, [0.01, 0.01], par, (@optic _.β1))
        opt_newton = NewtonPar(tol = 1e-9, max_iterations = 40, verbose = false)
        opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds = 0.01, p_max = 0.5, p_min = -0.5, detect_bifurcation = 3, nev = 2, newton_options = opt_newton, max_steps = 80, n_inversion = 8, save_sol_every_step = 1)

        br = continuation(prob, PALC(), opts_br; bothside = true, verbosity = 0)

        sn_codim2 = continuation(br, 2, (@optic _.β2), ContinuationPar(opts_br, detect_bifurcation = 1, save_sol_every_step = 1, max_steps = 40) ;
            detect_codim2_bifurcation = 2,
            update_minaug_every_step = 1,
            bdlinsolver = MatrixBLS()
            )
        @test sn_codim2.specialpoint[1].type == :bt
        @test sn_codim2.specialpoint[1].param ≈ 0 atol = 1e-6
        @test length(unique(sn_codim2.BT)) == length(sn_codim2)

        hopf_codim2 = continuation(br, 3, (@optic _.β2), ContinuationPar(opts_br, detect_bifurcation = 1, save_sol_every_step = 1, max_steps = 40, max_bisection_steps = 25) ; plot = false, verbosity = 0,
            detect_codim2_bifurcation = 2,
            update_minaug_every_step = 1,
            bothside = true,
            bdlinsolver = MatrixBLS(),
            )

        @test length(hopf_codim2.specialpoint) == 3
        @test hopf_codim2.specialpoint[2].type == :bt
        @test hopf_codim2.specialpoint[2].param ≈ 0 atol = 1e-6
        @test length(unique(hopf_codim2.BT)) == length(hopf_codim2)-1
        # plot(sn_codim2, hopf_codim2, branchlabel = ["Fold", "Hopf"])

        btpt = get_normal_form(sn_codim2, 1; nev = 2, autodiff = false)
        show(btpt)
        BK.type(btpt)
        @test norm(btpt.nf.b * sign(sum(btpt.ζ[1])) - par.b, Inf) < 1e-5
        @test norm(btpt.nf.a * sign(sum(btpt.ζ[1])) - par.a, Inf) < 1e-5
        @test isapprox(abs.(btpt.ζ[1]), [1, 0])
        @test isapprox(abs.(btpt.ζ[2]), [0, 1];rtol = 1e-6)
        @test isapprox(abs.(btpt.ζ★[1]), [1, 0];rtol = 1e-6)

        @test isapprox(btpt.nfsupp.K2, [0, 0]; atol = 1e-5)
        @test isapprox(btpt.nfsupp.d, 0; atol = 1e-3)
        @test isapprox(btpt.nfsupp.e, 0; atol = 1e-3)
        @test isapprox(btpt.nfsupp.a1, 0; atol = 1e-3)
        @test isapprox(btpt.nfsupp.b1, 0; atol = 1e-3)

        btpt1 = get_normal_form(sn_codim2, 1; nev = 2, autodiff = false)
        @test mapreduce(isapprox, &, btpt.nf, btpt1.nf)
        @test mapreduce(isapprox, &, btpt.nfsupp, btpt1.nfsupp)

        HC = BK.predictor(btpt, Val(:HopfCurve), 0.)
        HC.hopf(0.)
        SN = BK.predictor(btpt, Val(:FoldCurve), 0.)
        Hom = BK.predictor(btpt, Val(:HomoclinicCurve), 0.)
        Hom.orbit(0,0)

        # branch switching from BT from Fold
        opt = sn_codim2.contparams
        @reset opt.newton_options.verbose = false
        @reset opt.max_steps = 20
        hp_fromBT = continuation(sn_codim2, 1, opt;
            verbosity = 0, plot = false,
            δp = 1e-4,
            update_minaug_every_step = 1,
            )
        # plot(sn_codim2, hp_fromBT)

        # update the BT point using newton and MA formulation
        solbt = BK.newton_bt(sn_codim2, 1; options = NewtonPar(sn_codim2.contparams.newton_options, verbose = true), start_with_eigen = true, jacobian_ma = BK.AutoDiff())
        @assert BK.converged(solbt)
        _prob_bt = solbt.prob.VF.F
        BK.has_hessian(_prob_bt)
        BK.is_symmetric(_prob_bt)
        BK.has_adjoint(_prob_bt)
        BK.has_adjoint_MF(_prob_bt)
        BK.isinplace(_prob_bt)
        BK.getvec(zeros(2), _prob_bt)
        BK.getp(zeros(2), _prob_bt)
    end
end
####################################################################################################
# test of the Bautin normal form
function Fsl2!(f, u, p, t = 0)
    (;r, μ, ν, c3, c5) = p
    u1, u2 = u
    ua = u1^2 + u2^2
    f[1] = r * u1 - ν * u2 + ua * (c3 * u1 - μ * u2) + c5 * ua^2 * u1
    f[2] = r * u2 + ν * u1 + ua * (c3 * u2 + μ * u1) + c5 * ua^2 * u2
    return f
end
Fsl2(x, p) = Fsl2!(similar(x.*p.r), x, p, 0.)

let
    for _F in (Fsl2, Fsl2!)
        par_sl = (r = -0.5, μ = 0., ν = 1.0, c3 = 0.1, c5 = 0.3)
        prob = BK.BifurcationProblem(_F, [0.01, 0.01], par_sl, (@optic _.r))

        opt_newton = NewtonPar(tol = 1e-9, max_iterations = 40, verbose = false)
        opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.01, ds = 0.01, p_max = 0.5, p_min = -0.5, detect_bifurcation = 3, nev = 2, newton_options = opt_newton, max_steps = 80, n_inversion = 8, save_sol_every_step = 1)

        @reset opts_br.newton_options.verbose = false
        @reset opts_br.newton_options.tol = 1e-12
        opts_br = setproperties(opts_br;n_inversion = 10, max_bisection_steps = 25)

        br = continuation(prob, PALC(), opts_br)

        hopf_codim2 = continuation(br, 1, (@optic _.c3), ContinuationPar(opts_br, detect_bifurcation = 0, save_sol_every_step = 1, max_steps = 15, p_min = -2., p_max = 2., ds = -0.001) ;
            detect_codim2_bifurcation = 2,
            start_with_eigen = true,
            update_minaug_every_step = 1,
            bdlinsolver = MatrixBLS(),
            )
        @test hopf_codim2.specialpoint[1].type == :gh

        bautin = BK.get_normal_form(hopf_codim2, 1; nev = 2)
        show(bautin)
        BK.type(bautin)

        @test bautin.nf.l2 ≈ par_sl.c5 * 4 atol = 1e-6
    end
end
####################################################################################################
# test of the Zero-Hopf normal form
function Fzh!(f, u, p)
    (;β1, β2, G200, G011, G300, G111, G110, G210, G021) = p
    w0, u1, u2 = u
    ua = u1^2 + u2^2
    w1 = complex(u1, u2)

    f[1] = β1 + G200/2 * w0^2 + G011 * ua + G300/6 * w0^3 + G111 * w0 * ua

    tmp = (β2 + complex(0,1)) * w1 + G110 * w0 * w1 + G210/2 * w0^2 * w1 + G021/2 * w1 * ua

    f[2] = real(tmp)
    f[3] = imag(tmp)

    return f
end
Fzh(u, p) = Fzh!(similar(u), u, p)
let
    for _F in (Fzh!, Fzh)
        par_zh = (β1 = 0.1, β2 = -0.3, G200 = 1., G011 = 2., G300 = 3., G111 = 4., G110 = 5., G210 = -1., G021 = 7.)
        prob = BK.BifurcationProblem(_F, [0.05, 0.0, 0.0], par_zh, (@optic _.β1))
        br = continuation(prob, PALC(), setproperties(opts_br, ds = -0.001, dsmax = 0.0091, max_steps = 70, detect_bifurcation=3, n_inversion = 2), verbosity = 0)

        _cparams = br.contparams
        opts2 = @set _cparams.newton_options.verbose = false
        opts2 = setproperties(opts2 ; n_inversion = 10, ds = 0.001)
        br_codim2 = continuation(br, 2, (@optic _.β2), opts2; verbosity = 0, start_with_eigen = true, detect_codim2_bifurcation = 0, update_minaug_every_step = 1)

        @test br_codim2.specialpoint[1].type == :zh
        zh = get_normal_form(br_codim2, 1, autodiff = false, detailed = Val(true))
        @test zh.nf.G200 ≈ par_zh.G200
        @test zh.nf.G110 ≈ par_zh.G110
        @test zh.nf.G011/2 ≈ par_zh.G011
        BK.type(zh)

        pred = BK.predictor(zh, Val(:FoldCurve), 0.1)
        pred.EigenVec(0.1)
        pred.EigenVecAd(0.1)
        pred.fold(0.1)
    end
end
####################################################################################################
# test of the Hopf-Hopf normal form
function Fhh!(f, u, p)
    (;β1, β2, ω1, ω2, G2100, G1011, G3100, G2111, G1022, G1110, G0021, G2210, G1121, G0032) = p
    w1 = complex(u[1], u[2])
    w2 = complex(u[3], u[4])

    ua1 = abs2(w1)
    ua2 = abs2(w2)


    tmp1 = (β1 + complex(0, ω1)) * w1 + G2100/2 * w1 * ua1 + G1011 * w1 * ua2 + G3100/12 * w1 * ua1^2 + G2111/2 * w1 * ua1 * ua2 + G1022/4 * w1 * ua2^2

    f[1] = real(tmp1)
    f[2] = imag(tmp1)

    tmp2 = (β2 + complex(0, ω2)) * w2 + G1110 * w2 * ua2 + G0021/2 * w2 * ua2 + G2210/4 * w2 * ua1^2 + G1121/2 * w2 * ua1 * ua2 + G0032/12 * w2 * ua2^2

    f[3] = real(tmp2)
    f[4] = imag(tmp2)

    return f
end
Fhh(u, p) = Fhh!(similar(u .* p.β1), u, p)

let
    for _F in (Fhh!, Fhh)
        par_hh = (β1 = 0.1, β2 = -0.1, ω1 = 0.1, ω2 = 0.3, G2100 = 1., G1011 = 2., G3100 = 3., G2111 = 4., G1022 = 5., G1110 = 6., G0021 = 7., G2210 = 8., G1121 = 9., G0032 = 10. )
        prob = BK.BifurcationProblem(_F, zeros(4), par_hh, (@optic _.β1))
        br = continuation(prob, PALC(), setproperties(opts_br, ds = -0.001, dsmax = 0.0051, max_steps = 30, detect_bifurcation = 3, n_inversion = 2))
        _cparams = br.contparams
        opts2 = @set _cparams.newton_options.verbose = false
        opts2 = setproperties(opts2 ; n_inversion = 10, ds = 0.001)
        br_codim2 = continuation(br, 1, (@optic _.β2), opts2; verbosity = 0, start_with_eigen = true, detect_codim2_bifurcation = 2, update_minaug_every_step = 1)

        @test br_codim2.specialpoint[1].type == :hh
        hh = get_normal_form(br_codim2, 1, autodiff = false, detailed = Val(true))
        BK.type(hh)
        # @test hh.nf.G2100 == par_hh.G2100
        # @test hh.nf.G0021 == par_hh.G0021
        # @test hh.nf.G1110 == par_hh.G1110
        # @test hh.nf.G1011 == par_hh.G1011
    end
end