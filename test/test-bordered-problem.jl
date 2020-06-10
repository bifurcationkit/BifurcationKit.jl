# using Revise, Plots
using Test, BifurcationKit, KrylovKit, LinearAlgebra, Setfield
const BK = BifurcationKit

# test for constrained problems

####################################################################################################
# we test a dummy problem with constraint

N = (x; a = 0.5, b = 0.01) -> 1 + (x + a*x^2)/(1 + b*x^2)

function F_chan(x, p)
	α, β = p
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
par = (3.3, 0.01)
ig = [(i-1)*(n-i)/n^2+0.1 for i=1:n]
optnewton = NewtonPar(verbose = true)
sol, hist, flag = newton(F_chan, ig, par, optnewton)

_tau = BorderedArray(rand(n), 1.0)
g(x, p, tau = _tau) = dot(x, tau.u) + (p[1] - 3.) * tau.p
pb = BorderedProblem(F_chan, g, @lens _[1])

# test functional with AbstractVector form
pb(vcat(sol, 3.1), par)
pb(vcat(sol, 3.1), par, vcat(sol, 3.1))
# # test functional with BorderedVector form
# pb(BorderedArray(sol, 3.1), par)
#
# # test of newton functional, with BorderedVector / vector
# optnewbd = (@set optnewton.linsolver = MatrixBLS(optnewton.linsolver))
# newtonBordered(pb, BorderedArray(sol, 3.1), par, optnewbd )
# newtonBordered(pb, vcat(sol, 3.1), par, optnewbd )
# optnewbd = (@set optnewton.linsolver = BorderingBLS(optnewton.linsolver))
# newtonBordered(pb, BorderedArray(sol, 3.1),  optnewbd)
# newtonBordered(pb, vcat(sol, 3.1), optnewbd )
#
# # test of _newtonBK
# Jchan = (x0, p0) -> BK.finiteDifferences(x -> F_chan(x, p0), x0)
# BK._newtonBK( (x,p) -> F_chan(x, p), Jchan, BorderedArray(sol, 3.1), _tau, BorderedArray(sol, 3.1), ContinuationPar(newtonOptions = optnewbd), BK.DotTheta())
#
# # test of newton functional, with BorderedVector
# prob = β -> BorderedProblem((x, p) -> F_chan(x, p, β), (x, p) -> g(x, p))
#
# PseudoArcLengthContinuation.continuationBordered(prob, BorderedArray(sol, 3.1), 0.01, ContinuationPar(newtonOptions = optnewbd, maxSteps = 4), verbosity = 0)
# ####################################################################################################
# # problem with 2 constraints
# g2 = (x, p) -> [g(x,p[1]), p[2] - 0.01]
#
# dpF = (x0, p0) -> BK.finiteDifferences(p ->  F_chan(x0, p[1], p[2]), p0)
# dpg = (x0, p0) -> BK.finiteDifferences(p ->  g2(x0, p), p0)
#
# nestedpb = BorderedProblem(F = (x, p) -> F_chan(x, p[1], p[2]), g = (x, p) -> g2(x, p), dpF = dpF, dpg = dpg, npar = 2)
# z0nested = BorderedArray(sol, [3.1, 0.015])
# nestedpb(z0nested)
#
# z0nested = vcat(sol, [3.1, 0.015])
# nestedpb(z0nested)
#
# optnewnested = @set optnewbd.linsolver = MatrixBLS()
# newtonBordered(nestedpb, z0nested, optnewnested)
