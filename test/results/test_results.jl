using Test, BifurcationKit
const BK = BifurcationKit

# Simple Test problem (Pitchfork bifurcation) to generate a ContResult and a Branch object
f(u, p) = @. p.r * u - u^3

@testset "ContResult" begin
    prob = BK.BifurcationProblem(f, zeros(1), (r = -1.0,), (@optic _.r))
    opt = BK.ContinuationPar(p_min=-1.0, p_max=1.0)
    @reset opt.newton_options.verbose = true
    contres = BK.continuation(prob, PALC(), opt)
    @test typeof(contres) <: BK.ContResult

    # geteigensolver on ContResult
    @test BK.geteigensolver(contres) == BK.geteigensolver(opt)

    bp = contres.specialpoint[1] # pitchfork bifurcation

    # Test slicing of ContResult object
    @test contres[1:bp.step+1].specialpoint[1].step == bp.step
    @test contres[bp.step+1:end].specialpoint[1].step == bp.step

    # Slicing and indexing should match
    @test contres[bp.step:bp.step][1].param == contres[bp.step].param

    # Recursive slicing should work
    @test contres[bp.step:end][1:1][1].param == contres[bp.step:end][1].param

    # Slicing should still work when not evey sol/eig is saved
    opt = BK.ContinuationPar(opt; detect_bifurcation=1, save_sol_every_step=2, save_eig_every_step=3)
    contres = BK.continuation(prob, PALC(), opt)
    contres[end]
    contres[begin]
    @assert length(contres) != length(contres.sol) != length(contres.eig)
    @test length(contres[1:end]) == length(contres)
    @test length(contres[1:end].sol) == length(contres.sol)
    @test length(contres[1:end].eig) == length(contres.eig)
    BK.getparams(contres, 1)
end

@testset "Branch" begin
    prob = BK.BifurcationProblem(f, zeros(1), (r = -1.0,), (@optic _.r))
    # Test slicing of Branch object
    opt = BK.ContinuationPar(p_min=-1.0, p_max=1.0)
    contres = BK.continuation(prob, PALC(), opt)
    branch = BK.continuation(contres, 1)
    @assert typeof(branch) <: BK.Branch
    bp = branch.specialpoint[1] # pitchfork bifurcation

    # Test slicing of Branch object
    @test branch[1:bp.step+1].specialpoint[1].step == bp.step
    @test branch[bp.step+1:end].specialpoint[1].step == bp.step

    # Slicing and indexing should match
    @test branch[bp.step:bp.step][1].param == branch[bp.step].param

    # Recursive slicing should work
    @test branch[bp.step:end][1:1][1].param == branch[bp.step:end][1].param
end

@testset "cat/merge for AbstractBranchResult" begin
    prob = BK.BifurcationProblem(f, zeros(1), (r = -1.0,), (@optic _.r))
    opt = BK.ContinuationPar(p_min=-2.0, p_max=1.0)
    contres = BK.continuation(prob, PALC(), opt)
    opt2 = BK.ContinuationPar(p_min=-2.0, p_max=-1.0)
    contres2 = BK.continuation(BK.re_make(prob; params = (r = -2.0,)), PALC(), opt2)
    new_branch = BK._merge(contres2, contres)
end

@testset "return codes" begin
    mk() = BK.BorderedArray([0.0], 0.0)

    s = BK.ContState(z_pred = mk(), τ = mk(), z = mk(), z_old = mk(),
                     retcode = BK.ReturnCode.MaxSteps)
    @test BK.return_code(s) == BK.ReturnCode.MaxSteps
    @test BK.successful_retcode(s) == BK.successful_retcode(BK.ReturnCode.MaxSteps)
    @test copy(s).retcode == BK.ReturnCode.MaxSteps

    s2 = BK.ContState(z_pred = mk(), τ = mk(), z = mk(), z_old = mk())
    copyto!(s2, s)
    @test s2.retcode == BK.ReturnCode.MaxSteps

    prob = BK.BifurcationProblem(f, zeros(1), (r = -1.0,), (@optic _.r))
    contres = BK.continuation(prob, PALC(), BK.ContinuationPar(p_min = -1.0, p_max = 1.0))
    @test BK.return_code(contres) isa BK.ReturnCode.T
    @test BK.successful_retcode(contres) == BK.successful_retcode(BK.return_code(contres))

    # a dsmin failure is reported as DsLessThanMin and is not overwritten by Finalized
    s3 = BK.ContState(z_pred = mk(), τ = mk(), z = mk(), z_old = mk(),
                      ds = 0.01, converged = false)
    BK._step_size_control!(s3, BK.ContinuationPar(dsmin = 0.01), 0)
    @test s3.stopcontinuation
    @test BK.return_code(s3) == BK.ReturnCode.DsLessThanMin
    @test !BK.successful_retcode(s3)

    it = BK.ContIterable(prob, PALC(), BK.ContinuationPar(p_min = -1.0, p_max = 1.0))
    @test !BK.not_done(it, s3)
    @test BK.return_code(s3) == BK.ReturnCode.DsLessThanMin

    # not_done(..., false) must not modify the return code
    s4 = BK.ContState(z_pred = mk(), τ = mk(), z = mk(), z_old = mk(),
                      stopcontinuation = true, retcode = BK.ReturnCode.EmptyReturnCode)
    @test !BK.not_done(it, s4, false)
    @test s4.retcode == BK.ReturnCode.EmptyReturnCode
    @test !BK.not_done(it, s4)
    @test s4.retcode == BK.ReturnCode.Finalized

    # the accessor is defined on states, not on continuation kinds
    @test_throws MethodError BK.return_code(BK.EquilibriumCont())
end





