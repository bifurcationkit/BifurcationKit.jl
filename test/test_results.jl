using Test, BifurcationKit
const BK = BifurcationKit

# Simple Test problem (Pitchfork bifurcation) to generate a ContResult and a Branch object
function f(u, p)
    return p.r .* u - u .^ 3
end
p = (r=-1.0,)
u0 = [0.0]
prob = BK.BifurcationProblem(f, u0, p, (@optic _.r))

@testset "ContResult" begin
    opt = BK.ContinuationPar(p_min=-1.0, p_max=1.0)
    contres = BK.continuation(prob, PALC(), opt)
    @assert typeof(contres) <: BK.ContResult
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
    @assert length(contres) != length(contres.sol) != length(contres.eig)
    @test length(contres[1:end]) == length(contres)
    @test length(contres[1:end].sol) == length(contres.sol)
    @test length(contres[1:end].eig) == length(contres.eig)
end

@testset "Branch" begin
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




