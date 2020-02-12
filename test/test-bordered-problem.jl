# using Revise
# using Plots
using Test, PseudoArcLengthContinuation, KrylovKit, LinearAlgebra, Setfield
const PALC = PseudoArcLengthContinuation

# test for constrained problems

####################################################################################################
# we test a dummy problem with constraint

N = (x; a = 0.5, b = 0.01) -> 1 + (x + a*x^2)/(1 + b*x^2)

function F_chan(x, α, β = 0.01)
	f = similar(x)
	n = length(x)
	f[1] = x[1] - β
	f[n] = x[n] - β
	for i=2:n-1
		f[i] = (x[i-1] - 2 * x[i] + x[i+1]) * (n-1)^2 + α * N(x[i], b = β)
	end
	return f
end

n = 101
a = 3.3
ig = [(i-1)*(n-i)/n^2+0.1 for i=1:n]
optnewton = NewtonPar(verbose = true)
sol, hist, flag = newton(x ->  F_chan(x, 3.3), ig, optnewton)

_tau = BorderedArray(rand(n), 1.0)
g(x, p, tau = _tau) = dot(x, tau.u) + (p - 3.) * tau.p
pb = BorderedProblem((x, p) -> F_chan(x, p), (x, p) -> g(x, p))

# test functional with AbstractVector form
pb(vcat(sol, 3.1))
# test functional with BorderedVector form
pb(BorderedArray(sol, 3.1))

# test of newton functional, with BorderedVector / vector
optnewbd = (@set optnewton.linsolver = MatrixBLS(optnewton.linsolver))
newtonBordered(pb, BorderedArray(sol, 3.1), optnewbd )
newtonBordered(pb, vcat(sol, 3.1), optnewbd )
optnewbd = (@set optnewton.linsolver = BorderingBLS(optnewton.linsolver))
newtonBordered(pb, BorderedArray(sol, 3.1),  optnewbd)
newtonBordered(pb, vcat(sol, 3.1), optnewbd )

# test of _newtonPALC
Jchan = (x0, p0) -> PALC.finiteDifferences(x -> F_chan(x, p0), x0)
PALC._newtonPALC( (x,p) -> F_chan(x, p), Jchan, BorderedArray(sol, 3.1), _tau, BorderedArray(sol, 3.1), ContinuationPar(newtonOptions = optnewbd), PALC.DotTheta())

# test of newton functional, with BorderedVector
prob = β -> BorderedProblem((x, p) -> F_chan(x, p, β), (x, p) -> g(x, p))

PseudoArcLengthContinuation.continuationBordered(prob, BorderedArray(sol, 3.1), 0.01, ContinuationPar(newtonOptions = optnewbd, maxSteps = 4), verbosity = 0)
####################################################################################################
# problem with 2 constraints
g2 = (x, p) -> [g(x,p[1]), p[2] - 0.01]

dpF = (x0, p0) -> PALC.finiteDifferences(p ->  F_chan(x0, p[1], p[2]), p0)
dpg = (x0, p0) -> PALC.finiteDifferences(p ->  g2(x0, p), p0)

nestedpb = BorderedProblem(F = (x, p) -> F_chan(x, p[1], p[2]), g = (x, p) -> g2(x, p), dpF = dpF, dpg = dpg, npar = 2)
z0nested = BorderedArray(sol, [3.1, 0.015])
nestedpb(z0nested)

z0nested = vcat(sol, [3.1, 0.015])
nestedpb(z0nested)

optnewnested = @set optnewbd.linsolver = MatrixBLS()
newtonBordered(nestedpb, z0nested, optnewnested)
