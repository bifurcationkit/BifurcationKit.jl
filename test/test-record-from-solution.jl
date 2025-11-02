using Test, BifurcationKit
const BK = BifurcationKit

# Same problem as for test_results.jl
function f(u, p)
    return @. p.r * u - u^3
end

function record_from_solution(x, p; state, iter)
    @assert !isnothing(state)  # Ensuring state and it are always defined as kwargs  
    @assert !isnothing(iter)
    return (; set(iter.prob.params, iter.prob.lens, p)...) # Return the value of all the parameters
end

@testset "RecordFromSolution" begin
    # Creating the problem with a dumy parameter to check if the record_from_solution can access it
    prob = BK.BifurcationProblem(f, zeros(1), (r=-1.0, s=0.2), (@optic _.r); record_from_solution)
    opt = BK.ContinuationPar(p_min=-1.0, p_max=1.0)
    contres = BK.continuation(prob, PALC(), opt)
    @test all(hasfield.(typeof.(contres.branch[:]), :s)) # Ensuring that :s was registered correctly
end