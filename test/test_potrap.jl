# using Revise
using Test, BifurcationKit, LinearAlgebra, SparseArrays, ForwardDiff
const BK = BifurcationKit

n = 250*150
M = 30
par = nothing
sol0 = rand(2n)             # 585.977 KiB
orbitguess_f = rand(2n*M+1) # 17.166MiB

prob = BK.BifurcationProblem((x, p) -> x.^2, sol0, par; J = (x, p) -> (dx -> 2 .* dx))
probi = BK.BifurcationProblem((o, x, p) -> o .= x.^2, sol0, par; J = ((o, x, p, dx) -> o .= 2 .* dx), inplace = true)

pb = PeriodicOrbitTrapProblem(
            prob,
            rand(2n*M),
            rand(2n*M),
            rand(M-1))

pbg = PeriodicOrbitTrapProblem(
            prob,
            pb.ϕ,
            pb.xπ,
            pb.mesh.ds ; ongpu = true)

pbi = PeriodicOrbitTrapProblem(
            probi,
            pb.ϕ,
            pb.xπ,
            pb.mesh.ds)
@test BK.isinplace(pb) == false
# BK.POTrapFunctional(pb, res, orbitguess_f)
# BK.POTrapFunctional(pbi, res, orbitguess_f)
res = BK.residual(pb, orbitguess_f, par)
resg = BK.residual(pbg, orbitguess_f, par)
resi = BK.residual(pbi, orbitguess_f, par)
@test res == resi
@test res == resg

res = BK.jvp(pb, orbitguess_f, par, orbitguess_f)
resg = BK.jvp(pbg, orbitguess_f, par, orbitguess_f)
resi = BK.jvp(pbi, orbitguess_f, par, orbitguess_f)
@test res == resi
@test res == resg

BK.residual!(pbi, resi, orbitguess_f, par)
BK.jvp!(pbi, resi, orbitguess_f, par, orbitguess_f)
@test res == resi

# @code_warntype BK.potrap_functional!(pbi, resi, orbitguess_f)

# using BenchmarkTools
# @btime BK.residual($pb, $orbitguess_f, $par);                    # 6.825 ms (62 allocations: 34.33 MiB)
# @btime BK.residual($pbi, $orbitguess_f, $par);                   # 4.768 ms (2 allocations: 17.17 MiB)
# @btime BK.jvp($pb, $orbitguess_f, $par, $orbitguess_f);          # 8.427 ms (122 allocations: 51.50 MiB)
# @btime BK.jvp($pbi, $orbitguess_f, $par, $orbitguess_f);         # 5.170 ms (2 allocations: 17.17 MiB)
# @btime BK.residual!($pbi, $resi, $orbitguess_f, $par);           # 7.117 ms (0 allocations: 0 bytes)
# @btime BK.jvp!($pbi, $resi, $orbitguess_f, $par, $orbitguess_f); # 3.900 ms (0 allocations: 0 bytes)

#
# using IterativeSolvers, LinearMaps
#
# Jmap = LinearMap{Float64}(dv -> pbi(orbitguess_f, par, dv), 2n*M+1 ; ismutating = false)
# gmres(Jmap, orbitguess_f; verbose = false, maxiter = 1)
# gmres(Jmap, orbitguess_f; verbose = false, maxiter = 10)

# Jmap! = LinearMap{Float64}((o, dv) -> BK.POTrapFunctionalJac!(pbi, o, orbitguess_f, par, dv), 2n*M+1 ; ismutating = true)
# gmres(Jmap!, orbitguess_f; verbose = false, maxiter = 1)
# gmres(Jmap!, orbitguess_f; verbose = false, maxiter = 10)
#
# @code_warntype BK.POTrapFunctional!(pbi, resi, orbitguess_f, par)
# @profiler BK.POTrapFunctionalJac!(pbi, resi, orbitguess_f, par, orbitguess_f)
#
# Jmap2! = LinearMap{Float64}((o, dv) -> pbi(o, orbitguess_f, par, dv), 2n*M+1 ; ismutating = true)
# gmres(Jmap2!, orbitguess_f; verbose = false, maxiter = 1)
# gmres(Jmap2!, orbitguess_f; verbose = false, maxiter = 10)
#
# Jmap3! = LinearMap{Float64}((o, dv) -> (o .= pbi( orbitguess_f, dv)), 2n*M+1 ; ismutating = true)
# gmres(Jmap3!, orbitguess_f; verbose = false, maxiter = 1)
# gmres(Jmap3!, orbitguess_f; verbose = false, maxiter = 10)
#
# using ProfileView, Profile
# @profview gmres!(res, Jmap!, orbitguess_f; verbose = false, maxiter = 1)
# @profview gmres!(res, Jmap!, orbitguess_f; verbose = false, maxiter = 10)


####################################################################################################
# test whether we did not make any mistake in the improved version of the PO functional
function _functional(poPb, u0, p)
    M, N = size(poPb)
    T = u0[end]
    h = T * BK.get_time_step(poPb, 1)
    Mass = BifurcationKit.hasmassmatrix(poPb) ? poPb.massmatrix : I(poPb.N)

    u0c = BK.get_time_slices(poPb, u0)
    outc = similar(u0c)

    outc[:, 1] .= Mass * (u0c[:, 1] .- u0c[:, M-1]) .- (h/2) .* (poPb.prob_vf.VF.F(u0c[:, 1], p) .+ poPb.prob_vf.VF.F(u0c[:, M-1], p))

    for ii = 2:M-1
        h = T * BK.get_time_step(poPb, ii)
        outc[:, ii] .= Mass * (u0c[:, ii] .- u0c[:, ii-1]) .- (h/2) .* (poPb.prob_vf.VF.F(u0c[:, ii], p) .+ poPb.prob_vf.VF.F(u0c[:, ii-1], p))
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
    h = T * BK.get_time_step(poPb, 1)
    dh = dT * BK.get_time_step(poPb, 1)
    Mass = BifurcationKit.hasmassmatrix(poPb) ? poPb.massmatrix : I(poPb.N)

    u0c = BK.get_time_slices(poPb, u0)
    duc = BK.get_time_slices(poPb, du)
    outc = similar(u0c)

    outc[:, 1] .= Mass * (duc[:, 1] .- duc[:, M-1]) .- (h/2) .* (poPb.prob_vf.VF.J(u0c[:, 1], p)(duc[:, 1]) .+ poPb.prob_vf.VF.J(u0c[:, M-1], p)(duc[:, M-1]))

    for ii = 2:M-1
        h = T * BK.get_time_step(poPb, ii)
        dh = dT * BK.get_time_step(poPb, ii)
        outc[:, ii] .= Mass * (duc[:, ii] .- duc[:, ii-1]) .- (h/2) .* (poPb.prob_vf.VF.J(u0c[:, ii], p)(duc[:, ii]) .+ poPb.prob_vf.VF.J(u0c[:, ii-1], p)(duc[:, ii-1]))
    end

    dh = dT * BK.get_time_step(poPb, 1)
    outc[:, 1] .-=  dh/2 .* (poPb.prob_vf.VF.F(u0c[:, 1], p) .+ poPb.prob_vf.VF.F(u0c[:, M-1], p))
    for ii = 2:M-1
        dh = dT * BK.get_time_step(poPb, ii)
        outc[:, ii] .-= dh/2 .* (poPb.prob_vf.VF.F(u0c[:, ii], p) .+ poPb.prob_vf.VF.F(u0c[:, ii-1], p))
    end

    # closure condition ensuring a periodic orbit
    outc[:, M] .= duc[:, M] .- duc[:, 1]

    return vcat(vec(outc),
            dot(du[1:end-1], poPb.ϕ)) # this is the phase condition

end

res = BK.residual(pb, orbitguess_f, par)
_res = _functional(pb, orbitguess_f, par)
@test res ≈ _res

_du = rand(length(orbitguess_f))
res = BK.jvp(pb, orbitguess_f, par, _du)
_res = _dfunctional(pb, orbitguess_f, par, _du)
@test res ≈ _res

# with mass matrix
pbmass = @set pb.massmatrix = spdiagm( 0 => rand(pb.N))
res = BK.residual(pbmass, orbitguess_f, par)
_res = _functional(pbmass, orbitguess_f, par)
@test res ≈ _res

_du = rand(length(orbitguess_f))
res = BK.jvp(pbmass, orbitguess_f, par, _du)
_res = _dfunctional(pbmass, orbitguess_f, par, _du)
@test res ≈ _res
####################################################################################################
# test whether the analytical version of the Jacobian is right
n = 50
prob = BK.BifurcationProblem((x, p) -> cos.(x), sol0, par; J = (x, p) -> spdiagm(0 => -sin.(x)))
pbsp = PeriodicOrbitTrapProblem(
            prob,
            rand(2n*10),
            rand(2n*10),
            10)
orbitguess_f = rand(2n*10+1)
dorbit = rand(2n*10+1)
Jfd = sparse( ForwardDiff.jacobian(x -> BK.residual(pbsp,x, par), orbitguess_f) )
Jan = pbsp(Val(:JacFullSparse), orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

pbsp(Val(:JacFullSparseInplace), Jan, orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

#test for :Dense
pbsp(Val(:JacFullSparseInplace), Array(Jan), orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

# test for inplace
Jan = pbsp(Val(:JacFullSparse), orbitguess_f, par)
Jan2 = copy(Jan); _indx = BK.get_blocks(Jan2, pbsp.N, pbsp.M)
pbsp(Val(:JacFullSparseInplace), Jan2, orbitguess_f, par, _indx)
@test norm(Jan2 - Jan, Inf) < 1e-6

Jan = pbsp(Val(:JacCyclicSparse), orbitguess_f, par)
@test norm(Jfd[1:size(Jan,1),1:size(Jan,1)] - Jan, Inf) < 1e-6

# we update the section
BK.updatesection!(pbsp, rand(2n*10+1), par)
Jfd2 = sparse( ForwardDiff.jacobian(x -> BK.residual(pbsp, x, par), orbitguess_f) )
Jan2 = pbsp(Val(:JacFullSparse), orbitguess_f, par)
@test Jan2 != Jan
@test norm(Jfd2 - Jan2, Inf) < 1e-6
pbsp(Val(:JacFullSparseInplace), Jan2, orbitguess_f, par)
@test norm(Jfd2 - Jan2, Inf) < 1e-6
Jan = pbsp(Val(:JacCyclicSparse), orbitguess_f, par)
@test norm(Jfd2[1:size(Jan2,1),1:size(Jan2,1)] - Jan2, Inf) < 1e-6

##########################
#### idem with mass matrix
pbsp_mass = @set pbsp.massmatrix = massmatrix = spdiagm( 0 => rand(pbsp.N))
Jfd = sparse( ForwardDiff.jacobian(x -> BK.residual(pbsp_mass, x, par), orbitguess_f) )
Jan = pbsp_mass(Val(:JacFullSparse), orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

pbsp_mass(Val(:JacFullSparseInplace), Jan, orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

#test for :Dense
pbsp_mass(Val(:JacFullSparseInplace), Array(Jan), orbitguess_f, par)
@test norm(Jfd - Jan, Inf) < 1e-6

# test for inplace
Jan = pbsp_mass(Val(:JacFullSparse), orbitguess_f, par)
Jan2 = copy(Jan); _indx = BK.get_blocks(Jan2, pbsp_mass.N, pbsp_mass.M)
pbsp_mass(Val(:JacFullSparseInplace), Jan2, orbitguess_f, par, _indx)
@test norm(Jan2 - Jan, Inf) < 1e-6

Jan = pbsp_mass(Val(:JacCyclicSparse), orbitguess_f, par)
@test norm(Jfd[1:size(Jan,1), 1:size(Jan,1)] - Jan, Inf) < 1e-6

# we update the section
BK.updatesection!(pbsp_mass, rand(2n*10+1), par)
Jfd2 = sparse( ForwardDiff.jacobian(x -> BK.residual(pbsp_mass, x, par), orbitguess_f) )
Jan2 = pbsp_mass(Val(:JacFullSparse), orbitguess_f, par)
@test Jan2 != Jan
@test norm(Jfd2 - Jan2, Inf) < 1e-6
pbsp_mass(Val(:JacFullSparseInplace), Jan2, orbitguess_f, par)
@test norm(Jfd2 - Jan2, Inf) < 1e-6
Jan = pbsp_mass(Val(:JacCyclicSparse), orbitguess_f, par)
@test norm(Jfd2[1:size(Jan2,1),1:size(Jan2,1)] - Jan2, Inf) < 1e-6

####################################################################################################
# test whether the inplace version of computation of the Jacobian is right
n = 1000
prob = BK.BifurcationProblem((x, p) -> x.^2, sol0, par; J = (x, p) -> spdiagm(0 => 2 .* x))
pbsp = PeriodicOrbitTrapProblem(
            prob,
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
_indx = BK.get_blocks(Jpo, 2n, M)
pbsp(Val(:JacFullSparseInplace), Jpo2, orbitguess_f, par, _indx)
@test nnz(Jpo2 - Jpo) == 0

# version with indices in the cyclic matrix
Jpo = pbsp(Val(:JacCyclicSparse), orbitguess_f, par)
_indx = BK.get_blocks(Jpo, 2n, M-1)
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
_indx = BK.get_blocks(Jpo, 2n, M)
pbsp_mass(Val(:JacFullSparseInplace), Jpo2, orbitguess_f, par, _indx)
@test nnz(Jpo2 - Jpo) == 0

# version with indices in the cyclic matrix
Jpo = pbsp_mass(Val(:JacCyclicSparse), orbitguess_f, par)
_indx = BK.get_blocks(Jpo, 2n, M-1)
Jpo2 = copy(Jpo); Jpo2.nzval .*= 0
pbsp_mass(Val(:JacFullSparseInplace), Jpo2, orbitguess_f, par, _indx; updateborder = false)
@test nnz(Jpo2 - Jpo) == 0
####################################################################################################
# test of the version with inhomogeneous time discretisation
M = 10
prob = BK.BifurcationProblem((x, p) -> cos.(x), sol0, par; J = (x, p) -> spdiagm(0 => -sin.(x)))
pbsp = PeriodicOrbitTrapProblem(
            prob,
            rand(2n*M),
            rand(2n*M),
            M)

pbspti = PeriodicOrbitTrapProblem(
            prob,
            pbsp.ϕ,
            pbsp.xπ,
            ones(9) ./ 10)

BK.get_mesh_size(pbspti)
orbitguess_f = rand(2n*10+1)
BK.getperiod(pbspti, orbitguess_f, par)
BK.get_periodic_orbit(pbspti, orbitguess_f, par)

@test pbspti.xπ ≈ pbsp.xπ
@test pbspti.ϕ ≈ pbsp.ϕ
BK.residual(pbspti, orbitguess_f, par)
@test BK.residual(pbsp, orbitguess_f, par) ≈ BK.residual(pbspti, orbitguess_f, par)