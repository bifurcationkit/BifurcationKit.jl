# using Revise
using Test

@testset "BifurcationKit" begin

	@testset "Linear Solvers" begin
		include("precond.jl") #OK
		include("test_linear.jl") #OK
	end

	@testset "Newton" begin
		include("test_newton.jl") #OK
		# include("test-bordered-problem.jl") #OK
	end

	@testset "Continuation" begin
		include("test_bif_detection.jl") #OK
		include("test-cont-non-vector.jl") # ok
		include("simple_continuation.jl") # ok
		include("testNF.jl")
	end

	@testset "Events / User function" begin
		include("event.jl") #ok
	end

	@testset "Fold Codim 2" begin
		include("testJacobianFoldDeflation.jl") #ok
		include("codim2.jl") #ok
	end

	@testset "Hopf Codim 2" begin
		include("testHopfMA.jl") #ok
		include("lorenz84.jl") #ok
		include("COModel.jl")
	end

	@testset "Periodic orbits" begin
		include("test_potrap.jl") #ok
		include("test_SS.jl") #ok
		include("poincareMap.jl") #ok
		include("stuartLandauSH.jl") #ok
		include("stuartLandauTrap.jl") #ok
		include("stuartLandauCollocation.jl") #ok
		# for testing period doubling:
		include("testLure.jl")
	end

	@testset "Wave" begin
		include("test_wave.jl") #ok
	end
end
