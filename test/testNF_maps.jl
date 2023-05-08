# using Revise
# using Plots, Test
using BifurcationKit, LinearAlgebra, Setfield, SparseArrays, ForwardDiff, Parameters
const BK = BifurcationKit
norminf(x) = norm(x, Inf)
####################################################################################################
struct EigMaps{T} <: BK.AbstractEigenSolver
	solver::T
end

function (eig::EigMaps)(J, nev; kwargs...)
	λs, evs, cv, it = eig.solver(J + I, nev; kwargs)
	return log.(Complex.(λs)), evs, cv, it
end
####################################################################################################
opt_newton = NewtonPar(tol = 1e-9, maxIter = 20, verbose = false)
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, pMax = 0.4, pMin = -0.5, detectBifurcation = 3, nev = 2, newtonOptions = opt_newton, maxSteps = 100, nInversion = 4, tolBisectionEigenvalue = 1e-8, dsminBisection = 1e-9)
####################################################################################################
# case of the period doubling
Fpd(u, p) = @. (-1+p.μ * p.a) * u + p.c * u^3
pars_pd = (μ = -0.2, a = 0.456, c = -1.234)

probMap = BK.BifurcationProblem((x, p) -> Fpd(x, p) .- x, [0.0], pars_pd, (@lens _.μ); recordFromSolution = (x, p) -> x[1])

@set! opts_br.newtonOptions.eigsolver = EigMaps(DefaultEig())
br = continuation(probMap, PALC(), opts_br; normC = norminf, verbosity = 0)

prob = BK.BifurcationProblem(Fpd, [0.0], pars_pd, (@lens _.μ); recordFromSolution = (x, p) -> x[1])

pd = BK.PeriodDoubling(br.specialpoint[1].x, br.specialpoint[1].param, (@set pars_pd.μ = br.specialpoint[1].param), BK.getLens(br), [1.], [1.], nothing, :none)

nf = BK.periodDoublingNormalForm(prob, pd, DefaultLS(), verbose = false)
@test nf.nf.a ≈ pars_pd.a
@test nf.nf.b3 ≈ pars_pd.c
show(nf)
####################################################################################################
# case of the Neimark-Sacker
function Fns!(f, u, p, t)
	@unpack θ, μ, c3, a = p
	z = complex(u[1], u[2])
	dz = z * cis(θ) * (1 + a * μ + c3 * abs2(z))

	f[1] = real(dz)
	f[2] = imag(dz)

	return f
end
Fns(x, p) = Fns!(similar(x), x, p, 0.)
pars_ns = (a = 1.123, μ = -0.1, θ = 0.1, c3 = -1.123 - 0.456im)

prob_ns = BK.BifurcationProblem((x, p) -> Fns(x, p) .- x, 0.01rand(2), pars_ns, (@lens _.μ); recordFromSolution = (x, p) -> norminf(x),)

br = BK.continuation(prob_ns, PALC(), opts_br; normC = norminf, verbosity = 0)

prob = BK.BifurcationProblem(Fns, [0.0], pars_pd, (@lens _.μ); recordFromSolution = (x, p) -> x[1])

ns = BK.NeimarkSacker(br.specialpoint[1].x, br.specialpoint[1].param, (abs∘imag)(eigenvals(br, br.specialpoint[1].idx)[1]), (@set pars_ns.μ = br.specialpoint[1].param), BK.getLens(br), [1.], [1.], nothing, :none)

nf = BK.neimarkSackerNormalForm(prob, br, 1; nev = 2, verbose = true)
@test nf.nf.a ≈ pars_ns.a
@test nf.nf.b ≈ pars_ns.c3
show(nf)
####################################################################################################
