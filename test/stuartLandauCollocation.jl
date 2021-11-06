# using Revise, Plots
using Test
	using BifurcationKit, Parameters, Setfield, LinearAlgebra, ForwardDiff, SparseArrays
	const BK = BifurcationKit
##################################################################
# The goal of these tests is to test all combinaisons of options
##################################################################

norminf = x -> norm(x, Inf)

function Fsl!(f, u, p, t)
	@unpack r, μ, ν, c3 = p
	u1 = u[1]
	u2 = u[2]

	ua = u1^2 + u2^2

	f[1] = r * u1 - ν * u2 - ua * (c3 * u1 - μ * u2)
	f[2] = r * u2 + ν * u1 - ua * (c3 * u2 + μ * u1)
	return f
end

Fsl(x, p) = Fsl!(similar(x), x, p, 0.)

####################################################################################################
par_sl = (r = 0.1, μ = 0., ν = 1.0, c3 = 1.0)
u0 = [.001, .001]
par_hopf = (@set par_sl.r = 0.1)
####################################################################################################
# continuation, Hopf bifurcation point detection
optconteq = ContinuationPar(ds = -0.01, detectBifurcation = 3, pMin = -0.5, nInversion = 8)
br, = continuation(Fsl, u0, par_hopf, (@lens _.r), optconteq)
####################################################################################################
Ntst = 3
m = 4
N = 3
coll_cache = BK.POOrthogonalCollocationCache(Ntst, m)
const Mf = rand(N,N)
prob_col = BK.PeriodicOrbitOCollProblem(F = (x,p) -> Mf * x.^2, N = N, coll_cache = coll_cache, ϕ = ones(N*( 1 + m * Ntst)), xπ = zeros(N*( 1 + m * Ntst)))
size(prob_col)
length(prob_col)
BK.getTimes(prob_col)
size(coll_cache)

_orbit(t) = [cos(2pi*t)] * sqrt(par_sl.r/par_sl.c3)
_ci = BK.generateSolution(prob_col, _orbit, 1.)
prob_col(_ci, par_sl) #|> scatter
BK.getTimeSlices(prob_col, _ci)
# interpolate solution
sol = BK.POOcollSolution(prob_col, _ci)
sol(rand())

# using ForwardDiff
# J(x,p) = ForwardDiff.jacobian(u -> prob_col(u,  p), x)
# _J = J(vcat(vec(_ci), 1),  par_sl)
# 	heatmap(_J .!= 0, yflip = true)
####################################################################################################
Ntst = 50
m = 4
N = 2
coll_cache = BK.POOrthogonalCollocationCache(Ntst, m)
prob_col = BK.PeriodicOrbitOCollProblem(F = Fsl, N = 2, coll_cache = coll_cache, ϕ = zeros(N*( 1 + m * Ntst)), xπ = zeros(N*( 1 + m * Ntst)))
prob_col.ϕ[2] = 1

_orbit(t) = [cos(t), sin(t)] * sqrt(par_sl.r/par_sl.c3)
_ci = BK.generateSolution(prob_col, _orbit, 2pi)
prob_col(_ci, par_sl)
@test prob_col(_ci, par_sl) |> norminf < 1e-7

# using ForwardDiff
# J(x,p) = ForwardDiff.jacobian(u -> prob_col(u,  p), x)
# _J = J(vcat(vec(_ci), 2pi),  par_sl)
# 	heatmap(_J .!=0, yflip = true)

args = (tangentAlgo = BorderedPred(),
	# linearPO = :autodifSparse,
	# linearAlgo = MatrixBLS(),
	recordFromSolution = (x,p) -> (norminf(x[1:end-1])),
	plotSolution = (x,p; k...) -> begin
		outt = getPeriodicOrbit(prob_col, x, p)
		plot!(vec(outt.t), outt.u[1,:]; k...)
	end,
	finaliseSolution = (z, tau, step, contResult; k...) -> begin
		# BK.haseigenvalues(contResult) && Base.display(contResult.eig[end].eigenvals)
		return true
	end,)

optcontpo = setproperties(optconteq; detectBifurcation = 2, precisionStability = 1e-7)
@set! optcontpo.ds = -0.01
@set! optcontpo.newtonOptions.verbose = false

br_po, = @time continuation(prob_col, _ci, par_sl, (@lens _.r), optcontpo;
	verbosity = 0, plot = false,
	args...
	)
