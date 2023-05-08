# using Revise
# using Plots
using Test
using BifurcationKit, Parameters, Setfield, LinearAlgebra, ForwardDiff, SparseArrays
const BK = BifurcationKit
##################################################################
# The goal of these tests is to test all combinations of options
##################################################################

norminf(x) = norm(x, Inf)

function Fsl!(f, u, p, t = 0)
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
probsl = BK.BifurcationProblem(Fsl, u0, par_hopf, (@lens _.r))
probsl_ip = BK.BifurcationProblem(Fsl!, u0, par_hopf, (@lens _.r), inplace = true)
####################################################################################################
# continuation, Hopf bifurcation point detection
optconteq = ContinuationPar(ds = -0.01, detectBifurcation = 3, pMin = -0.5, nInversion = 8)
br = continuation(probsl, PALC(), optconteq)
####################################################################################################
Ntst = 4
m = 4
N = 3
const Mf = rand(N, N)
prob1 = BK.BifurcationProblem((x,p) -> Mf * x.^2, zeros(1), nothing)
prob_col = BK.PeriodicOrbitOCollProblem(Ntst, m, prob_vf = prob1, N = N, ϕ = ones(N * ( 1 + m * Ntst)), xπ = zeros(N * ( 1 + m * Ntst)))
size(prob_col)
length(prob_col)
BK.getTimes(prob_col)
BK.getMaxTimeStep(prob_col)
size(prob_col.mesh_cache)
BK.updateMesh!(prob_col, prob_col.mesh_cache.mesh)
PeriodicOrbitOCollProblem(10, 2) |> BK.getMeshSize
BK.getLs(prob_col)
show(prob_col)

_orbit(t) = [cos(2pi * t), 0, 0] * sqrt(par_sl.r / par_sl.c3)
_ci = BK.generateSolution(prob_col, _orbit, 1.)
BK.getPeriodicOrbit(prob_col, _ci, par_sl)
BK.getMaximum(prob_col, _ci, par_sl)
BK.∂(sin, 2)(0.)
prob_col(_ci, par_sl) #|> scatter
BK.getTimeSlices(prob_col, _ci)
# interpolate solution
sol = BK.POSolution(prob_col, _ci)
sol(rand())

# using ForwardDiff
# J(x,p) = ForwardDiff.jacobian(u -> prob_col(u,  p), x)
# _J = J(vcat(vec(_ci), 1),  par_sl)
# 	heatmap(_J .!= 0, yflip = true)
####################################################################################################
prob_col = PeriodicOrbitOCollProblem(200, 5, prob_vf = probsl, N = 1000)
_ci = BK.generateSolution(prob_col, t -> cos(t) .* ones(1000), 2pi)
BK.getTimes(prob_col)
sol = BK.POSolution(prob_col, _ci)
sol(0.1)
####################################################################################################
# test precision of phase condition, it must work for non uniform mesh
# recall that it is 1/T int(f,g')
@views function phaseCond(pb::PeriodicOrbitOCollProblem, u, v)
	Ty = eltype(u)
	phase = zero(Ty)

	uc = BK.getTimeSlices(pb, u)
	vc = BK.getTimeSlices(pb, v)

	n, m, Ntst = size(pb)

	T = getPeriod(pb, u, nothing)

	guj = zeros(Ty, n, m)
	uj  = zeros(Ty, n, m+1)

	gvj = zeros(Ty, n, m)
	vj  = zeros(Ty, n, m+1)

	L, ∂L = BK.getLs(pb.mesh_cache)
	ω = pb.mesh_cache.gauss_weight

	rg = UnitRange(1, m+1)
	@inbounds for j in 1:Ntst
		uj .= uc[:, rg]
		vj .= vc[:, rg]
		mul!(guj, uj, L')
		mul!(gvj, vj, ∂L')
		@inbounds for l in 1:m
			# for mul!(gvj, vj, L')
			# phase += dot(guj[:, l], gvj[:, l]) * ω[l] * (mesh[j+1] - mesh[j]) / 2 * T
			phase += dot(guj[:, l], gvj[:, l]) * ω[l]
		end
		rg = rg .+ m
	end
	return phase / T
end

for Ntst in 2:10:100
	# @info "Ntst" Ntst
	prob_col = PeriodicOrbitOCollProblem(Ntst, 10, prob_vf = probsl, N = 1)

	_ci1 = BK.generateSolution(prob_col, t -> [1], 2pi)
	_ci2 = BK.generateSolution(prob_col, t -> [t], 2pi)
	@test phaseCond(prob_col, _ci1, _ci2) ≈ 1 atol = 1e-10
	# @info phaseCond(prob_col, _ci1, _ci2)/pi

	_ci1 = BK.generateSolution(prob_col, t -> [cos(t)], 2pi)
	_ci2 = BK.generateSolution(prob_col, t -> [sin(t)], 2pi)
	@test phaseCond(prob_col, _ci1, _ci2) ≈ 1/2 atol = 2e-8
	# @info phaseCond(prob_col, _ci1, _ci2)/pi-1

	_ci1 = BK.generateSolution(prob_col, t -> [cos(t)], 2pi)
	_ci2 = BK.generateSolution(prob_col, t -> [cos(t)], 2pi)
	@test phaseCond(prob_col, _ci1, _ci2) / pi ≈ 0 atol = 1e-11
	# @info phaseCond(prob_col, _ci1, _ci2) / pi

	_ci1 = BK.generateSolution(prob_col, t -> [cos(t)], 2pi)
	_ci2 = BK.generateSolution(prob_col, t -> [t], 2pi)
	@test phaseCond(prob_col, _ci1, _ci2) / pi ≈ 0 atol = 1e-5
	# @info phaseCond(prob_col, _ci1, _ci2) / pi
end


prob_col = PeriodicOrbitOCollProblem(22, 10, prob_vf = probsl, N = 1)
_ci1 = BK.generateSolution(prob_col, t -> [cos(2pi*t)], 1)
_ci2 = BK.generateSolution(prob_col, t -> [cos(2pi*t)], 1)
@test BK.∫(prob_col, BK.getTimeSlices(prob_col, _ci1), BK.getTimeSlices(prob_col, _ci2)) ≈ 0.5

prob_col = PeriodicOrbitOCollProblem(22, 10, prob_vf = probsl, N = 1)
_ci1 = BK.generateSolution(prob_col, t -> [cos(2pi*t)], 3)
_ci2 = BK.generateSolution(prob_col, t -> [cos(2pi*t)], 3)
@test BK.∫(prob_col, BK.getTimeSlices(prob_col, _ci1), BK.getTimeSlices(prob_col, _ci2), 3) ≈ 3/2

####################################################################################################
Ntst = 50
m = 4
N = 2
prob_col = BK.PeriodicOrbitOCollProblem(Ntst, m; prob_vf = probsl, N = 2, ϕ = zeros(N*( 1 + m * Ntst)), xπ = zeros(N*( 1 + m * Ntst)))
prob_col.ϕ[2] = 1 #phase condition

_orbit(t) = [cos(t), sin(t)] * sqrt(par_sl.r/par_sl.c3)
_ci = BK.generateSolution(prob_col, _orbit, 2pi)
@time prob_col(_ci, par_sl)
@test prob_col(_ci, par_sl)[1:end-1] |> norminf < 1e-7

prob_coll_ip = @set prob_col.prob_vf = probsl_ip

@time prob_col(_ci, par_sl)
@time prob_coll_ip(_ci, par_sl)

# test precision of generated solution
_sol = getPeriodicOrbit(prob_col, _ci, nothing)
for (i, t) in pairs(_sol.t)
	@test _sol.u[:, i] ≈ _orbit(t)
end

args = (
	plotSolution = (x,p; k...) -> begin
		outt = getPeriodicOrbit(prob_col, x, p)
		plot!(vec(outt.t), outt.u[1, :]; k...)
	end,
	finaliseSolution = (z, tau, step, contResult; k...) -> begin
		return true
	end,)

optcontpo = setproperties(optconteq; detectBifurcation = 2, tolStability = 1e-7)
@set! optcontpo.ds = -0.01
@set! optcontpo.newtonOptions.verbose = false

prob_col2 = (@set prob_coll_ip.prob_vf.params = par_sl)
@set! prob_col2.jacobian = BK.AutoDiffDense()
sol_po = newton(prob_col2, _ci, optcontpo.newtonOptions)

# test Solution
solc = BK.POSolution(prob_col2, sol_po.u)
# plot([t for t in LinRange(0,2pi,100)], [solc(t)[1] for t in LinRange(0,2pi,100)])
let
	mesh = BK.getMesh(prob_col2)
	solpo = getPeriodicOrbit(prob_col2, sol_po.u, nothing)
	for (i, t) in pairs(solpo.t)
		@test solc(t) ≈ solpo.u[:, i]
	end
end

# 3.855762 seconds (1.24 M allocations: 3.658 GiB, 12.54% gc time)
@set! prob_col2.updateSectionEveryStep = 1
br_po = @time continuation(prob_col2, _ci, PALC(tangent = Bordered()), optcontpo;
	verbosity = 0, plot = false,
	args...,
	)
####################################################################################################
# test  Hopf aBS
br_po_gev = continuation(br, 1, (@set ContinuationPar(optcontpo; ds = 0.01, saveSolEveryStep=1, maxSteps = 10).newtonOptions.verbose = false),
	PeriodicOrbitOCollProblem(20, 5; jacobian = BK.AutoDiffDense(), updateSectionEveryStep = 1);
	δp = 0.1,
	usedeflation = true,
	eigsolver = BK.FloquetCollGEV(DefaultEig(),(20*5+1)*2,2),
	)

br_po = continuation(br, 1, (@set ContinuationPar(optcontpo; ds = 0.01, saveSolEveryStep=1, maxSteps = 10).newtonOptions.verbose = false),
	PeriodicOrbitOCollProblem(20, 5; jacobian = BK.AutoDiffDense(), updateSectionEveryStep = 1);
	δp = 0.1,
	usedeflation = true,
	eigsolver = BK.FloquetColl(),
	)

# we test that the 2 methods give the same floquet exponents
for i=1:length(br_po)-1
	@info i
	@test BK.eigenvals(br_po, i) ≈ BK.eigenvals(br_po_gev, i)
end

# test mesh adaptation
br_po = continuation(br, 1, (@set ContinuationPar(optcontpo; ds = 0.01, saveSolEveryStep=1, maxSteps = 2).newtonOptions.verbose = false),
	PeriodicOrbitOCollProblem(20, 5; jacobian = BK.AutoDiffDense(), updateSectionEveryStep = 1, meshadapt = true);
	δp = 0.1,
	usedeflation = true,
	eigsolver = BK.FloquetColl(),
	)
