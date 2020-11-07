# using Revise, Test, Plots
using BifurcationKit, LinearAlgebra, Setfield
const BK = BifurcationKit

# display internal information about the branch
function displayBr(contRes)
	println("#"^50)
	for ii in 1:length(contRes.branch)
		println("- $ii --------------")
		println("step = ", contRes[ii][end])
		println("eiv = "); display(contRes.eig[ii].eigenvals)
		println("stab = ", contRes[ii].stable)
		println("n_uns = ", contRes[ii].n_unstable)
	end
end

function teststab(br)
	# test if stability works
	# test that stability corresponds
	out = true
	for ii in eachindex(br.branch)
		n_u = BK.isstable(br.contparams, br.eig[ii].eigenvals)
		# test that the stability matches the one in eig
		br[ii].n_unstable != n_u[2] && (println( "$ii did not work!!",br[i].n_unstable ,", ", n_u[2]); @assert 1==0)
		out = out && br[ii].n_unstable == n_u[2]
		out = out && br[ii].stable  == n_u[1]
		# we test that step = ii-1
		out = out && br.branch[ii][end] == ii-1
		# test that the field `step` match in the structure
		out = out && br.branch[ii][end] == br.eig[ii].step
	end
	@assert out "Basic structure of the branch is broken"
	# test about bifurcation points
	for bp in br.bifpoint
		id = bp.idx
		# @show id, br.n_unstable[id], br.n_unstable[id-1]
		# test that the states marked as bifurcation points are always after true bifurcation points
		out = out && abs(br[id].n_unstable - br[id-1].n_unstable) > 0
	end
	out
end

NL(x) = -x^3
dNL(x) = -3x^2

function Ftb(x,p)
	return -x .+ (p.L * x) .* p.λ .+ NL.(x)
end

function Jtb(x, p)
	J = copy(p.L .* p.λ)
	for i in eachindex(x)
		J[i,i] += dNL(x[i]) - 1
	end
	return J
end

par = (L = Diagonal([1.0/ii for ii in 1:5 for jj in 1:ii]), λ = .0)
append!(par.L.diag, [1/6. 1/6.5 1/6.75 1/6.875])

# ensemble of bifurcation points
bifpoints = unique(1 ./par.L.diag);
dimBif = [ii for ii in 1:5]; append!(dimBif, [1 1 1 1])

x0 = zeros(size(par.L,1))

optc = ContinuationPar(pMin = -1., pMax = 10., ds = 0.1, maxSteps = 150, detectBifurcation = 2, saveEigenvectors = false)
	br1, _ = continuation(Ftb, Jtb, x0, par, (@lens _.λ), optc; plot=false, verbosity = 0)
@test teststab(br1)

br2, _ = continuation(Ftb, Jtb, x0, par, (@lens _.λ), setproperties(optc; detectBifurcation = 3, pMax = 10.3, nInversion = 4, tolBisectionEigenvalue = 1e-7); plot=false, verbosity = 0)
@test teststab(br2)

bifpoint2 = [bp.param for bp in br2.bifpoint]
@test bifpoint2 > bifpoints
@test norm(bifpoints - bifpoint2, Inf) < 3e-3
dimBif2 = [abs(bp.δ[1]) for bp in br2.bifpoint]
@test dimBif2 == dimBif
