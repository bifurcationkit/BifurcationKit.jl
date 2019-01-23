using Pkg, Test
cd(Pkg.dir("PseudoArcLengthContinuation")*"/test/")
include("test_linear.jl")
include("simple_continuation.jl")
include("test_newton.jl")
include("testHopfMA.jl")
include("testJacobianFoldDeflation.jl")
