# using Revise
# using Test, Plots
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

function testBranch(br)
    # test if stability works
    # test that stability corresponds
    out = true
    for ii in eachindex(br.branch)
        @test br.eig[ii].eigenvals == eigenvals(br, br.eig[ii].step, false)
        # compute number of unstable eigenvalues
        isstable, n_u, n_i = BK.isStable(br.contparams, br.eig[ii].eigenvals)
        # test that the stability matches the one in eig
        @test br[ii].n_unstable == n_u
        @test br[ii].stable  == isstable
        # we test that step = ii-1
        @test br.branch[ii][end] == ii-1
        # test that the field `step` match in the structure
        @test br.branch[ii][end] == br.eig[ii].step
    end
    # test about bifurcation points
    for bp in br.specialpoint
        if bp.type != :endpoint
            id = bp.idx
            # test that the states marked as bifurcation points are always after true bifurcation points
            # for reversed branch, we test with id+1
            @test (abs(br[id].n_unstable - br[id-1].n_unstable) > 0) ||
            (abs(br[id].n_unstable - br[id+1].n_unstable) > 0)
            # test that the bifurcation point belongs to the interval
            @test bp.interval[1] <= bp.param <= bp.interval[2]
            # test that bp.param = br[id].param
            @test bp.param == br[id].param
        end
    end
end

NL(x) = -x^3
dNL(x) = -3x^2
Ftb(x,p) = -x .+ (p.L * x) .* p.λ .+ NL.(x)

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
specialpoints = unique(1 ./par.L.diag);
dimBif = [ii for ii in 1:5]; append!(dimBif, [1 1 1 1])

x0 = zeros(size(par.L, 1))

optc = ContinuationPar(pMin = -1., pMax = 10., ds = 0.1, maxSteps = 150, detectBifurcation = 2, saveEigenvectors = false)
prob = BK.BifurcationProblem(Ftb, x0, par, (@lens _.λ); J = Jtb)
alg = PALC()
br1 = continuation(prob, alg, optc; verbosity = 0)
testBranch(br1)

br1r = BK._reverse(br1)
testBranch(BK._reverse(br1))


br2 = continuation(prob, alg, setproperties(optc; detectBifurcation = 3, pMax = 10.3, nInversion = 4, tolBisectionEigenvalue = 1e-7); plot = false, verbosity = 0)
testBranch(br2)
for bp in br2.specialpoint
    @test bp.interval[1] <= bp.param <= bp.interval[2]
end

specialpoint2 = [bp.param for bp in br2.specialpoint if bp.type != :endpoint]
@test specialpoint2 > specialpoints
@test norm(specialpoints - specialpoint2, Inf) < 3e-3
dimBif2 = [abs(bp.δ[1]) for bp in br2.specialpoint if bp.type != :endpoint]
@test dimBif2 == dimBif


# case where bisection "fails". Test whether the bifurcation point belongs to the specified interval
br3 = continuation(prob, alg, setproperties(optc; detectBifurcation = 3, pMax = 10.3, nInversion = 8, tolBisectionEigenvalue = 1e-7); verbosity = 0)
testBranch(br3)

# case where bisection "fails". Test whether the bifurcation point belongs to the specified interval
# in this case, we test if coming from above, and having no inversion, still leads to correct result
br4 = continuation((@set prob.params.λ = 0.95), alg, setproperties(optc; detectBifurcation = 3, pMax = 1.95, nInversion = 8, ds = 0.7, dsmax = 1.5, maxBisectionSteps = 1); verbosity = 0)
testBranch(br4)
####################################################################################################
# this example is to test failures in Newton and how it affects the bifurcation points labels
F = (x, p; k = 3) -> (@. p * x -  x^k/k)
Jac_m = (x, p; k = 2) -> diagm(0 => p .- x.^k)

opts = ContinuationPar(dsmax = 0.1, dsmin = 1e-5, ds = 0.001, maxSteps = 130, pMin = -3., pMax = 0.1, newtonOptions = NewtonPar(tol = 1e-8, verbose = false, maxIter = 4), detectBifurcation=3, nInversion=4)

prob4 = BK.BifurcationProblem(F, zeros(1), -0.1, (@lens _); J = Jac_m, recordFromSolution = (x,p)->x[1])
br4 = continuation(prob4, alg, opts; verbosity = 0, plot=false)
testBranch(br4)
####################################################################################################
function Ftb(X, p)
    p1, p2, k = p
    x, y = X
    out = similar(X)
    out[1] = p1 + x - y - x^k/k
    out[2] = p1 + y + x - 2y^k/k
    out
end

par = (p1 = -3., p2=-3., k=3)

opts = ContinuationPar(dsmax = 0.1, ds = 0.001, maxSteps = 135, pMin = -3., pMax = 4.0, newtonOptions = NewtonPar(maxIter = 5), detectBifurcation = 3, nInversion = 6, dsminBisection = 1e-9, maxBisectionSteps = 15, nev = 2)

prob = BK.BifurcationProblem(Ftb, -2ones(2), par, (@lens _.p1); recordFromSolution = (x,p)->x[1])
br = continuation(prob, alg, (@set opts.detectBifurcation = 3);
    plot = false, verbosity = 0,)
    show(br)
testBranch(br)

br = continuation(prob, alg, opts)
testBranch(br)
