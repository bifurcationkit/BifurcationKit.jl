using PseudoArcLengthContinuation
const Cont  =  PseudoArcLengthContinuation

function test_newton()
	println("--> Test Newton method")
	N = 100
	x0 = ones(N) .+ rand(N) * 0.1
	F(x) = x.^3 .- 1.0
	Jac(x) = diagm(0 => 3.0 * x.^2)

	opts = Cont.NewtonPar(verbose = false, maxIter = 8)
	sol, hist, flag, _ = @time Cont.newton(F, Jac, x0, opts)
	sol, hist, flag, _ = @time Cont.newton(F, Jac, x0, opts, normN = x->norm(x,Inf64))

end

test_newton()
