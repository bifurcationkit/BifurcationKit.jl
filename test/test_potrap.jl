# using Revise
using Test, BifurcationKit, LinearAlgebra, Setfield, SparseArrays, ForwardDiff
const BK = BifurcationKit

n = 250*150
M = 30
par = nothing
sol0 = rand(2n)				# 585.977 KiB
orbitguess_f = rand(2n*M+1)	# 17.166MiB
pb = PeriodicOrbitTrapProblem(
			(x, p) -> x.^2,
			(x, p) -> (dx -> 2 .* dx),
			rand(2n*M),
			rand(2n*M),
			rand(M-1))

pbg = PeriodicOrbitTrapProblem(
			(x, p) -> x.^2,
			(x, p) -> (dx -> 2 .* dx),
			pb.ϕ,
			pb.xπ,
			pb.mesh.ds ; ongpu = true)

pbi = PeriodicOrbitTrapProblem(
			(o, x, p) -> o .= x.^2,
			((o, x, p, dx) -> o .= 2 .* dx),
			pb.ϕ,
			pb.xπ,
			pb.mesh.ds ; isinplace = true)
@test BK.isInplace(pb) == false
# @time BK.POTrapFunctional(pb, res, orbitguess_f)
# @time BK.POTrapFunctional(pbi, res, orbitguess_f)
res = @time pb(orbitguess_f, par)
resg = @time pbg(orbitguess_f, par)
resi = @time pbi(orbitguess_f, par)
@test res == resi
@test res == resg

res = @time pb(orbitguess_f, par, orbitguess_f)
resg = @time pbg(orbitguess_f, par, orbitguess_f)
resi = @time pbi(orbitguess_f, par, orbitguess_f)
@test res == resi
@test res == resg

@time BK.POTrapFunctional!(pbi, resi, orbitguess_f, par)
@time BK.POTrapFunctionalJac!(pbi, resi, orbitguess_f, par, orbitguess_f)
@test res == resi

# @code_warntype BK.POTrapFunctional!(pbi, resi, orbitguess_f)

# using BenchmarkTools
# @btime pb($orbitguess_f, $par); 								# 17.825 ms (62 allocations: 34.33 MiB)
# @btime pbi($orbitguess_f, $par); 								# 12.768 ms (2 allocations: 17.17 MiB)
# @btime pb($orbitguess_f, $par, $orbitguess_f) 					# 28.427 ms (122 allocations: 51.50 MiB)
# @btime pbi($orbitguess_f, $par, $orbitguess_f)  				# 14.170 ms (2 allocations: 17.17 MiB)
# @btime BK.POTrapFunctional!($pbi, $resi, $orbitguess_f, $par) 	# 7.117 ms (0 allocations: 0 bytes)
# @btime BK.POTrapFunctionalJac!($pbi, $resi, $orbitguess_f, $par, $orbitguess_f) #  13.117 ms (0 allocations: 0 bytes)

#
# using IterativeSolvers, LinearMaps
#
# Jmap = LinearMap{Float64}(dv -> pbi(orbitguess_f, par, dv), 2n*M+1 ; ismutating = false)
# gmres(Jmap, orbitguess_f; verbose = false, maxiter = 1)
# @time gmres(Jmap, orbitguess_f; verbose = false, maxiter = 10)

# Jmap! = LinearMap{Float64}((o, dv) -> BK.POTrapFunctionalJac!(pbi, o, orbitguess_f, par, dv), 2n*M+1 ; ismutating = true)
# gmres(Jmap!, orbitguess_f; verbose = false, maxiter = 1)
# @time gmres(Jmap!, orbitguess_f; verbose = false, maxiter = 10)
#
# @code_warntype BK.POTrapFunctional!(pbi, resi, orbitguess_f, par)
# @profiler BK.POTrapFunctionalJac!(pbi, resi, orbitguess_f, par, orbitguess_f)
#
# Jmap2! = LinearMap{Float64}((o, dv) -> pbi(o, orbitguess_f, par, dv), 2n*M+1 ; ismutating = true)
# gmres(Jmap2!, orbitguess_f; verbose = false, maxiter = 1)
# @time gmres(Jmap2!, orbitguess_f; verbose = false, maxiter = 10)
#
# Jmap3! = LinearMap{Float64}((o, dv) -> (o .= pbi( orbitguess_f, dv)), 2n*M+1 ; ismutating = true)
# gmres(Jmap3!, orbitguess_f; verbose = false, maxiter = 1)
# @time gmres(Jmap3!, orbitguess_f; verbose = false, maxiter = 10)
#
# using ProfileView, Profile
# @profview gmres!(res, Jmap!, orbitguess_f; verbose = false, maxiter = 1)
# @profview gmres!(res, Jmap!, orbitguess_f; verbose = false, maxiter = 10)


####################################################################################################
# test whether we did not make any mistake in the improved version of the PO functional
function _functional(poPb, u0, p)
	M, N = size(poPb)
	T = u0[end]
	h = T * BK.getTimeStep(poPb, 1)
	Mass = BifurcationKit.hasmassmatrix(poPb) ? poPb.massmatrix : I(poPb.N)

	u0c = BK.extractTimeSlices(poPb, u0)
	outc = similar(u0c)

	outc[:, 1] .= Mass * (u0c[:, 1] .- u0c[:, M-1]) .- (h/2) .* (poPb.F(u0c[:, 1], p) .+ poPb.F(u0c[:, M-1], p))

	for ii = 2:M-1
		h = T * BK.getTimeStep(poPb, ii)
		outc[:, ii] .= Mass * (u0c[:, ii] .- u0c[:, ii-1]) .- (h/2) .* (poPb.F(u0c[:, ii], p) .+ poPb.F(u0c[:, ii-1], p))
	end

	# closure condition ensuring a periodic orbit
	outc[:, M] .= u0c[:, M] .- u0c[:, 1]

	return vcat(vec(outc),
			dot(u0[1:end-1] .- poPb.xπ, poPb.ϕ)) # this is the phase condition
end

function _dfunctional(poPb, u0, p, du)
	# jacobian of the functional

	M, N = size(poPb)
	T = u0[end]
	dT = du[end]
	h = T * BK.getTimeStep(poPb, 1)
	dh = dT * BK.getTimeStep(poPb, 1)
	Mass = BifurcationKit.hasmassmatrix(poPb) ? poPb.massmatrix : I(poPb.N)

	u0c = BK.extractTimeSlices(poPb, u0)
	duc = BK.extractTimeSlices(poPb, du)
	outc = similar(u0c)

	outc[:, 1] .= Mass * (duc[:, 1] .- duc[:, M-1]) .- (h/2) .* (poPb.J(u0c[:, 1], p)(duc[:, 1]) .+ poPb.J(u0c[:, M-1], p)(duc[:, M-1]))

	for ii = 2:M-1
		h = T * BK.getTimeStep(poPb, ii)
		dh = dT * BK.getTimeStep(poPb, ii)
		outc[:, ii] .= Mass * (duc[:, ii] .- duc[:, ii-1]) .- (h/2) .* (poPb.J(u0c[:, ii], p)(duc[:, ii]) .+ poPb.J(u0c[:, ii-1], p)(duc[:, ii-1]))
	end

	dh = dT * BK.getTimeStep(poPb, 1)
	outc[:, 1] .-=  dh/2 .* (poPb.F(u0c[:, 1], p) .+ poPb.F(u0c[:, M-1], p))
	for ii = 2:M-1
		dh = dT * BK.getTimeStep(poPb, ii)
		outc[:, ii] .-= dh/2 .* (poPb.F(u0c[:, ii], p) .+ poPb.F(u0c[:, ii-1], p))
	end

	# closure condition ensuring a periodic orbit
	outc[:, M] .= duc[:, M] .- duc[:, 1]

	return vcat(vec(outc),
			dot(du[1:end-1], poPb.ϕ)) # this is the phase condition

end


res = @time pb(orbitguess_f, par)
_res = _functional(pb, orbitguess_f, par)
@test res ≈ _res

_du = rand(length(orbitguess_f))
res = @time pb(orbitguess_f, par, _du)
_res = _dfunctional(pb, orbitguess_f, par, _du)
@test res ≈ _res

# with mass matrix
pbmass = @set pb.massmatrix = spdiagm( 0 => rand(pb.N))
res = @time pbmass(orbitguess_f, par)
_res = _functional(pbmass, orbitguess_f, par)
@test res ≈ _res

_du = rand(length(orbitguess_f))
res = @time pbmass(orbitguess_f, par, _du)
_res = _dfunctional(pbmass, orbitguess_f, par, _du)
@test res ≈ _res
####################################################################################################
# test whether the analytical version of the Jacobian is right
n = 50
pbsp = PeriodicOrbitTrapProblem(
			(x, p) -> cos.(x),
			(x, p) -> spdiagm(0 => -sin.(x)),
			rand(2n*10),
			rand(2n*10),
			10)
orbitguess_f = rand(2n*10+1)
dorbit = rand(2n*10+1)
Jfd = sparse( ForwardDiff.jacobian(x -> pbsp(x, par), orbitguess_f) )
Jan = pbsp(Val(:JacFullSparse), orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

@time pbsp(Val(:JacFullSparseInplace), Jan, orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

#test for :Dense
@time pbsp(Val(:JacFullSparseInplace), Array(Jan), orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

# test for inplace
Jan = pbsp(Val(:JacFullSparse), orbitguess_f, par)
Jan2 = copy(Jan); _indx = BK.getBlocks(Jan2, pbsp.N, pbsp.M)
pbsp(Val(:JacFullSparseInplace), Jan2, orbitguess_f, par, _indx)
@test norm(Jan2 - Jan, Inf) < 1e-6

Jan = pbsp(Val(:JacCyclicSparse), orbitguess_f, par)
@test norm(Jfd[1:size(Jan,1),1:size(Jan,1)] - Jan, Inf) < 1e-6

# we update the section
BK.updateSection!(pbsp, rand(2n*10+1), par)
Jfd2 = sparse( ForwardDiff.jacobian(x -> pbsp(x, par), orbitguess_f) )
Jan2 = pbsp(Val(:JacFullSparse), orbitguess_f, par)
@test Jan2 != Jan
@test norm(Jfd2 - Jan2, Inf) < 1e-6
@time pbsp(Val(:JacFullSparseInplace), Jan2, orbitguess_f, par)
@test norm(Jfd2 - Jan2, Inf) < 1e-6
Jan = pbsp(Val(:JacCyclicSparse), orbitguess_f, par)
@test norm(Jfd2[1:size(Jan2,1),1:size(Jan2,1)] - Jan2, Inf) < 1e-6

##########################
#### idem with mass matrix
pbsp_mass = @set pbsp.massmatrix = massmatrix = spdiagm( 0 => rand(pbsp.N))
Jfd = sparse( ForwardDiff.jacobian(x -> pbsp_mass(x, par), orbitguess_f) )
Jan = pbsp_mass(Val(:JacFullSparse), orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

@time pbsp_mass(Val(:JacFullSparseInplace), Jan, orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

#test for :Dense
@time pbsp_mass(Val(:JacFullSparseInplace), Array(Jan), orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

# test for inplace
Jan = pbsp_mass(Val(:JacFullSparse), orbitguess_f, par)
Jan2 = copy(Jan); _indx = BK.getBlocks(Jan2, pbsp_mass.N, pbsp_mass.M)
pbsp_mass(Val(:JacFullSparseInplace), Jan2, orbitguess_f, par, _indx)
@test norm(Jan2 - Jan, Inf) < 1e-6

Jan = pbsp_mass(Val(:JacCyclicSparse), orbitguess_f, par)
@test norm(Jfd[1:size(Jan,1),1:size(Jan,1)] - Jan, Inf) < 1e-6

# we update the section
BK.updateSection!(pbsp_mass, rand(2n*10+1), par)
Jfd2 = sparse( ForwardDiff.jacobian(x -> pbsp_mass(x, par), orbitguess_f) )
Jan2 = pbsp_mass(Val(:JacFullSparse), orbitguess_f, par)
@test Jan2 != Jan
@test norm(Jfd2 - Jan2, Inf) < 1e-6
@time pbsp_mass(Val(:JacFullSparseInplace), Jan2, orbitguess_f, par)
@test norm(Jfd2 - Jan2, Inf) < 1e-6
Jan = pbsp_mass(Val(:JacCyclicSparse), orbitguess_f, par)
@test norm(Jfd2[1:size(Jan2,1),1:size(Jan2,1)] - Jan2, Inf) < 1e-6

####################################################################################################
# test whether the inplace version of computation of the Jacobian is right
n = 1000
pbsp = PeriodicOrbitTrapProblem(
			(x, p) -> x.^2,
			(x, p) -> spdiagm(0 => 2 .* x),
			rand(2n*M),
			rand(2n*M),
			M)

sol0 = rand(2n)
orbitguess_f = rand(2n*M+1)
Jpo = pbsp(Val(:JacFullSparse), orbitguess_f, par)
Jpo2 = copy(Jpo)
pbsp(Val(:JacFullSparseInplace), Jpo2, orbitguess_f, par)
@test nnz(Jpo2 - Jpo) == 0

# version with indices in the full matrix
_indx = BK.getBlocks(Jpo, 2n, M)
pbsp(Val(:JacFullSparseInplace), Jpo2, orbitguess_f, par, _indx)
@test nnz(Jpo2 - Jpo) == 0

# version with indices in the cyclic matrix
Jpo = pbsp(Val(:JacCyclicSparse), orbitguess_f, par)
_indx = BK.getBlocks(Jpo, 2n, M-1)
Jpo2 = copy(Jpo); Jpo2.nzval .*= 0
pbsp(Val(:JacFullSparseInplace), Jpo2, orbitguess_f, par, _indx; updateborder = false)
@test nnz(Jpo2 - Jpo) == 0

##########################
#### idem with mass matrix
pbsp_mass = @set pbsp.massmatrix = massmatrix = spdiagm( 0 => rand(pbsp.N))
Jpo = pbsp_mass(Val(:JacFullSparse), orbitguess_f, par)
Jpo2 = copy(Jpo)
pbsp_mass(Val(:JacFullSparseInplace), Jpo2, orbitguess_f, par)
@test nnz(Jpo2 - Jpo) == 0

# version with indices in the full matrix
_indx = BK.getBlocks(Jpo, 2n, M)
pbsp_mass(Val(:JacFullSparseInplace), Jpo2, orbitguess_f, par, _indx)
@test nnz(Jpo2 - Jpo) == 0

# version with indices in the cyclic matrix
Jpo = pbsp_mass(Val(:JacCyclicSparse), orbitguess_f, par)
_indx = BK.getBlocks(Jpo, 2n, M-1)
Jpo2 = copy(Jpo); Jpo2.nzval .*= 0
pbsp_mass(Val(:JacFullSparseInplace), Jpo2, orbitguess_f, par, _indx; updateborder = false)
@test nnz(Jpo2 - Jpo) == 0
####################################################################################################
# test of the version with inhomogenous time discretisation
M = 10
pbsp = PeriodicOrbitTrapProblem(
			(x, p) -> cos.(x),
			(x, p) -> spdiagm(0 => -sin.(x)),
			rand(2n*M),
			rand(2n*M),
			M)

pbspti = PeriodicOrbitTrapProblem(
			(x, p) -> cos.(x),
			(x, p) -> spdiagm(0 => -sin.(x)),
			pbsp.ϕ,
			pbsp.xπ,
			ones(9) ./ 10)

BK.getM(pbspti)
orbitguess_f = rand(2n*10+1)
BK.getAmplitude(pbspti, orbitguess_f, par)
BK.getMaximum(pbspti, orbitguess_f, par)
BK.getPeriod(pbspti, orbitguess_f, par)
BK.getTrajectory(pbspti, orbitguess_f, par)

@test pbspti.xπ ≈ pbsp.xπ
@test pbspti.ϕ ≈ pbsp.ϕ
pbspti(orbitguess_f, par)
@test pbsp(orbitguess_f, par) ≈ pbspti(orbitguess_f, par)
