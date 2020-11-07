# using Revise
using Test

# test linear, newton
include("precond.jl")
include("test_linear.jl")
include("test_newton.jl")

# basic tests of continuation
include("test_bif_detection.jl")
include("test-cont-non-vector.jl")
include("simple_continuation.jl")
include("test-bordered-problem.jl")
include("testNF.jl")

# test periodic orbits
include("test_SS.jl")
include("poincareMap.jl")
include("stuartLandauSH.jl")
include("stuartLandauTrap.jl")


include("testJacobianFoldDeflation.jl")
include("testHopfMA.jl")
include("test_potrap.jl")
