# using Revise
using Test, BifurcationKit, LinearAlgebra, ForwardDiff
const BK = BifurcationKit

source_term(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)
limits(x,i,N, b) = (i<1||i>N) ? b : x[i]

function F_chan(x, p)
    α, β = p
    N = length(x)
    f = similar(x)
    N = length(x)
    ind(x,i) = limits(x,i,N,β)
    for i=1:N
        f[i] = (ind(x,i-1) - 2 * x[i] + ind(x,i+1)) * (N-1)^2 + α * source_term(x[i], b = β)
    end
    return f
end

par_chan = (α = 3.3, β = 0.01)

n = 101
sol0 = [(i-1)*(n-i)/n^2+0.1 for i=1:n]
opt_newton = NewtonPar(tol = 1e-9, verbose = false)
prob = BK.BifurcationProblem(F_chan, sol0, (α = 3.3, β = 0.01), (@lens _.α);
    plot_solution = (x, p; kwargs...) -> (plot!(x;label="", kwargs...)))
out = newton( prob, opt_newton)

# test with secant continuation
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, p_max = 4.1, max_steps = 120, newton_options = opt_newton, detect_bifurcation = 3)
br = continuation( prob, PALC(), opts_br0)
####################################################################################################
# deflation newton solver, test of jacobian expression
deflationOp = DeflationOperator(2, dot, 1.0, [out.u])

# for testing
show(deflationOp)
length(deflationOp)
deflationOp(2out.u, out.u)
push!(deflationOp, rand(n))
deleteat!(deflationOp, 2)

deflationOp = DeflationOperator(2, dot, 1.0, [out.u])
chanDefPb = DeflatedProblem(prob, deflationOp, DefaultLS())

opt_def = setproperties(opt_newton; tol = 1e-10, max_iterations = 1000, verbose = false)
outdef1 = newton((@set prob.u0 = out.u .* (1 .+0.01*rand(n))), deflationOp, opt_def)
# @test BK.converged(outdef1)
outdef1 = newton((@set prob.u0 = out.u .* (1 .+0.01*rand(n))), deflationOp, opt_def, Val(:autodiff))
@test BK.converged(outdef1)
####################################################################################################
# Fold continuation, test of Jacobian expression
outfold = newton(br, 2; start_with_eigen = true)
@test  BK.converged(outfold) && outfold.itnewton == 2
outfold = BK.newton_fold((@set br.prob.VF.isSymmetric = true), 2; start_with_eigen = true, issymmetric = true)
@test BK.converged(outfold) && outfold.itnewton == 2

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds= 0.01, p_max = 4.1, p_min = 0., newton_options = NewtonPar(verbose=false, tol = 1e-8), max_steps = 50, detect_bifurcation = 0)
outfoldco = continuation(br, 2, (@lens _.β), optcontfold; start_with_eigen = true, update_minaug_every_step = 1, plot = false)
outfoldco = continuation((@set br.prob.VF.isSymmetric = true), 2, (@lens _.β), optcontfold; start_with_eigen = true, update_minaug_every_step = 1)

# manual handling
indfold = 1
foldpt = foldpoint(br, indfold)
foldpb = FoldProblemMinimallyAugmented(
        (@set prob.VF.d2F = nothing), # this is for debug array
        br.specialpoint[indfold].x,
        br.specialpoint[indfold].x,
        opts_br0.newton_options.linsolver)
foldpb(foldpt, par_chan) |> norm

outfold = BK.newton_fold(prob, foldpt, par_chan, br.specialpoint[indfold].x, br.specialpoint[indfold].x, NewtonPar(verbose=false))
# @test BK.converged(outfold)
    # println("--> Fold found at α = ", outfold.p, " from ", br.specialpoint[indfold].param)

# user defined Fold Problem
indfold = 1

# we define the following wrappers to be able to use ForwardDiff
Bd2Vec(x) = vcat(x.u, x.p)
Vec2Bd(x) = BorderedArray(x[1:end-1], x[end])
foldpbVec(x,p) = Bd2Vec(foldpb(Vec2Bd(x),p))

rhs = rand(n+1)
Jac_fold_fdMA(u0) = ForwardDiff.jacobian( u -> foldpbVec(u, par_chan), u0)
J_fold_fd = Jac_fold_fdMA(Bd2Vec(foldpt))
res_fd =  J_fold_fd \ rhs

Jac_fold_MA(u0, p, pb::FoldProblemMinimallyAugmented) = (return (x=u0, params=p, prob = pb))
jacFoldSolver = BK.FoldLinearSolverMinAug()
debugTmpForσ = zeros(n+1,n+1) # temporary array for debugging σ
res_explicit = jacFoldSolver(Jac_fold_MA(foldpt, par_chan, foldpb), Vec2Bd(rhs); debugArray = debugTmpForσ)

Jac_fold_MA(foldpt, 0.01, foldpb)[2]

# test whether the Jacobian Matrix for the Fold problem is correct
@test debugTmpForσ[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1] |> norminf == 0

# we test the expression for σp
σp_fd = J_fold_fd[end,end]
σp_fd_ana = debugTmpForσ[end,end]
@test σp_fd ≈ σp_fd_ana atol = 1e-5

# we test the expression for σx
σx_fd = J_fold_fd[end,1:end-1]
σx_ana = debugTmpForσ[end,1:end-1]
@test σx_fd ≈ σx_ana rtol = 1e-2

σx_fd - σx_ana |> norminf
debugTmpForσ[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1] |> norminf

# check our solution of the bordered problem
res_exp = debugTmpForσ \ rhs
@test norm(res_exp - Bd2Vec(res_explicit[1]), Inf64) < 1e-10
#############################################
# same with foldpb.issymmetric = true
@set! foldpb.prob_vf.VF.isSymmetric = true
rhs = rand(n+1)
Jac_fold_fdMA(u0) = ForwardDiff.jacobian( u -> foldpbVec(u, par_chan), u0)
J_fold_fd = Jac_fold_fdMA(Bd2Vec(foldpt))
res_fd =  J_fold_fd \ rhs

Jac_fold_MA(u0, p, pb::FoldProblemMinimallyAugmented) = (return (x=u0, params=p, prob = pb))
jacFoldSolver = BK.FoldLinearSolverMinAug()
debugTmpForσ = zeros(n+1,n+1) # temporary array for debugging σ
res_explicit = jacFoldSolver(Jac_fold_MA(foldpt, par_chan, foldpb), Vec2Bd(rhs); debugArray = debugTmpForσ)

Jac_fold_MA(foldpt, 0.01, foldpb)[2]

# test whether the Jacobian Matrix for the Fold problem is correct
@test debugTmpForσ[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1] |> norminf == 0

# we test the expression for σp
σp_fd = J_fold_fd[end,end]
σp_fd_ana = debugTmpForσ[end,end]
@test σp_fd ≈ σp_fd_ana atol = 1e-5

# we test the expression for σx
σx_fd = J_fold_fd[end,1:end-1]
σx_ana = debugTmpForσ[end,1:end-1]
@test σx_fd ≈ σx_ana rtol = 1e-2

σx_fd - σx_ana |> norminf
debugTmpForσ[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1] |> norminf

# check our solution of the bordered problem
res_exp = debugTmpForσ \ rhs
@test norm(res_exp - Bd2Vec(res_explicit[1]), Inf64) < 1e-10
####################################################################################################
# Use of different eigensolvers
opt_newton = NewtonPar(tol = 1e-8, verbose = false, eigsolver = EigKrylovKit())
opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, p_max = 4.1, max_steps = 250, newton_options = opt_newton, detect_fold = true, detect_bifurcation = 1, nev = 15)

br = continuation(BK.re_make(prob;record_from_solution = (x,p)->norm(x,Inf64)), PALC(), opts_br0, plot = false, verbosity = 0)

opts_br0 = ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, p_max = 4.1, max_steps = 250, newton_options = NewtonPar(tol =1e-8), detect_fold = true, detect_bifurcation = 1, nev = 15)

br = continuation(prob, PALC(), opts_br0,plot = false, verbosity = 0)
