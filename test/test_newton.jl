# using Revise, Test
using BifurcationKit, LinearAlgebra, Setfield
const BK = BifurcationKit

function test_newton(x0)
	Ty = eltype(x0)

	F(x, p) = x.^3 .- 1
	Jac(x, p) = diagm(0 => 3 .* x.^2)

	opts = NewtonPar( tol = Ty(1e-8), verbose = true)
	sol, hist, flag, _ = newton(F, Jac, x0, nothing, opts, normN = x->norm(x,Inf), callback = BK.cbMaxNorm(100.0))
end
####################################################################################################
# we test the regular newton algorithm
# simple case
test_newton(ones(10) .+ rand(10) * 0.1)

# test types for newton
sol, = test_newton(Float16.(ones(10) .+ rand(10) * 0.1))
@test eltype(sol) == Float16
####################################################################################################

function test_newton_palc(x0, p0)
	Ty = eltype(x0)
	N = length(x0)

	θ = Ty(0.2)
	dotθ = BK.DotTheta(dot)

	F(x, p) = x.^3 .- 13 .* x .- p
	Jac(x, p) = diagm(0 => 3 .* x.^2 .- 13)

	z0 = BorderedArray(x0, p0)
	τ0 = BorderedArray(rand(Ty, N), convert(typeof(p0), 0.2))
	zpred = BorderedArray(x0, convert(typeof(p0), 0.3))
	optn = NewtonPar{Ty, DefaultLS, DefaultEig}(verbose = true, tol = Ty(1e-6))
	optc = ContinuationPar{Ty, DefaultLS, DefaultEig}(newtonOptions = optn, ds = 0.001, theta = θ)

	iter = ContIterable(F, Jac, x0, p0, (@lens _), optc)
	state = iterate(iter)[1]
	sol, hist, flag, _ = newtonPALC(iter, state)
end

sol, = test_newton_palc(ones(10) .+ rand(10) * 0.01, 1.)

#test type
sol, = test_newton_palc(Float32.(ones(10) .+ rand(10) * 0.001), Float32(1.))
@test typeof(sol) == BorderedArray{Vector{Float32}, Float32}
####################################################################################################
# test of  deflated problems
_T = Float32
F4def = (x, p) -> @. (x-1) * (x-2)
J4def = (x, p) -> BK.finiteDifferences(z->F4def(z,p), x)
deflationOp = DeflationOperator(_T(2), dot, _T(1), [[_T(1)]])
@test firstindex(deflationOp) == 1
@test lastindex(deflationOp) == 1
@test eltype(deflationOp) == _T
@test deflationOp(rand(_T,1)) isa _T
defpb = DeflatedProblem(F4def, J4def, deflationOp)
@test defpb(rand(_T, 1), nothing) |> eltype == _T
@test defpb(rand(_T, 1), nothing, rand(_T, 1)) |> eltype == _T

push!(deflationOp, rand(_T,1))
@test deflationOp(zeros(_T, 1)) isa _T
@test deflationOp(rand(_T, 1), rand(_T, 1)) isa _T
copy(deflationOp)

#  test custom distance
deflationOp2 = DeflationOperator(_T(2), BifurcationKit.CustomDist((u,v)->norm(u-v)), _T(1), deflationOp.roots)
@test deflationOp2(zeros(_T, 1)) isa _T
length(deflationOp2)
deflationOp2(rand(_T, 1), rand(_T, 1))
