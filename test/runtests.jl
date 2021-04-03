# using Revise
using Test

# test linear, newton
begin
	include("precond.jl")
	include("test_linear.jl")
	include("test_newton.jl")
end

# basic tests of continuation
begin
	include("test_bif_detection.jl")
	include("test-cont-non-vector.jl")
	include("simple_continuation.jl")
	include("test-bordered-problem.jl")
	include("testNF.jl")
end

# test periodic orbits
include("test_SS.jl")
include("poincareMap.jl")
include("stuartLandauSH.jl")
include("stuartLandauTrap.jl")


include("testJacobianFoldDeflation.jl")

begin
	include("testHopfMA.jl")
end

begin
	include("test_potrap.jl")
end
