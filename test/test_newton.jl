#using Revise, Test
using PseudoArcLengthContinuation, LinearAlgebra, Setfield
const PALC = PseudoArcLengthContinuation

function test_newton(x0)
	Ty = eltype(x0)

	F(x, p) = x.^3 .- 1
	Jac(x, p) = diagm(0 => 3 .* x.^2)

	opts = PALC.NewtonPar( tol = Ty(1e-8))
	sol, hist, flag, _ = PALC.newton(F, Jac, x0, nothing, opts, normN = x->norm(x,Inf))
end
######################################################################
# we test the regular newton algorithm
# simple case
test_newton(ones(10) .+ rand(10) * 0.1)

# test types for newton
sol, _, _ = test_newton(Float16.(ones(10) .+ rand(10) * 0.1))
@test eltype(sol) == Float16
######################################################################
function test_newton_palc(x0, p0)
	Ty = eltype(x0)
	N = length(x0)

	θ = Ty(0.2)
	dotθ = PALC.DotTheta(dot)

	F(x, p) = x.^3 .- p
	Jac(x, p) = diagm(0 => 3 .* x.^2)

	z0 = BorderedArray(x0, p0)
	τ0 = BorderedArray(rand(Ty, N), convert(typeof(p0), 0.2))
	zpred = BorderedArray(x0, convert(typeof(p0), 0.3))
	optn = NewtonPar{Ty, DefaultLS, DefaultEig}()
	optc = ContinuationPar{Ty, DefaultLS, DefaultEig}(newtonOptions = optn)
	sol, hist, flag, _ = @time PALC.newtonPALC(F, Jac, p0, (@lens _), z0, τ0, zpred, Ty(0.02), θ, optc, dotθ)
end

sol, _, _, _ = test_newton_palc(ones(10) .+ rand(10) * 0.1, 1.)

#test type
sol, _, _, _ = test_newton_palc(Float32.(ones(10) .+ rand(10) * 0.1), Float32(1.))
@test typeof(sol) == BorderedArray{Vector{Float32}, Float32}
