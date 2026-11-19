using BifurcationKit, Test

residual(x, p) = x .- p
residual!(out, x, p) = (out .= x .- p)
out_of_place(x::Vector{Float32}, level::Float32) = BifurcationProblem(residual, x, level; inplace = false)
in_place(x::Vector{Float32}, level::Float32) = BifurcationProblem(residual!, x, level; inplace = true)

@testset "BifurcationProblem infers a concrete type from a literal inplace" begin
    for build in (out_of_place, in_place)
        inferred = only(Base.return_types(build, (Vector{Float32}, Float32)))
        @test isconcretetype(inferred)
    end
end
