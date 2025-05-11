# using Revise
using Test, BifurcationKit, LinearAlgebra, ForwardDiff
const BK = BifurcationKit

_source_term(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)
_limits(x, i, N, b) = (i<1 || i>N) ? b : x[i]

function F_chan(x, p)
    (;α, β) = p
    N = length(x)
    f = similar(x)
    N = length(x)
    ind(x,i) = _limits(x,i,N,β)
    for i=1:N
        f[i] = (ind(x,i-1) - 2 * x[i] + ind(x,i+1)) * (N-1)^2 + α * _source_term(x[i], b = β)
    end
    return f
end

par_chan = (α = 3.3, β = 0.01)

n = 101
sol0 = [(i-1)*(n-i)/n^2+0.1 for i=1:n]
opt_newton = NewtonPar(tol = 1e-10)
prob = BK.BifurcationProblem(F_chan, sol0, (α = 3.3, β = 0.01), (@optic _.α))
out = BK.solve(prob, Newton(), opt_newton)

opts_br0 = ContinuationPar(p_max = 4., max_steps = 100, newton_options = opt_newton)
br = continuation(prob, PALC(), opts_br0)
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

opt_def = NewtonPar(opt_newton; tol = 1e-10, max_iterations = 1000)
outdef1 = BK.solve((@set prob.u0 = out.u .* (1 .+ 0.01*rand(n))), deflationOp, opt_def)
@test BK.converged(outdef1)
outdef1 = BK.solve((@set prob.u0 = out.u .* (1 .+ 0.01*rand(n))), deflationOp, opt_def, Val(:autodiff))
@test BK.converged(outdef1)
####################################################################################################
# Fold continuation, test of Jacobian expression
outfold = newton(br, 2; start_with_eigen = true)
@test  BK.converged(outfold) && outfold.itnewton == 2
outfold = BK.newton_fold((@set br.prob.VF.isSymmetric = true), 2; start_with_eigen = true, issymmetric = true)
@test BK.converged(outfold) && outfold.itnewton == 2

optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds= 0.01, p_max = 4.1, p_min = 0., newton_options = NewtonPar(tol = 1e-8), max_steps = 50, detect_bifurcation = 0)
outfoldco = continuation(br, 2, (@optic _.β), optcontfold; start_with_eigen = true, update_minaug_every_step = 1)
outfoldco = continuation((@set br.prob.VF.isSymmetric = true), 2, (@optic _.β), optcontfold; start_with_eigen = true, update_minaug_every_step = 1)

# manual handling
indfold = 1
foldpt = BK.fold_point(br, indfold)
foldpb = FoldProblemMinimallyAugmented(
        (@set prob.VF.d2F = nothing), # this is for debug array
        br.specialpoint[indfold].x,
        br.specialpoint[indfold].x,
        opts_br0.newton_options.linsolver)

outfold = BK.newton_fold(prob, foldpt, par_chan, br.specialpoint[indfold].x, br.specialpoint[indfold].x, NewtonPar())
# @test BK.converged(outfold)

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

# test against analytical jacobian
_fold_ma_problem = BK.FoldMAProblem(foldpb, BK. MinAugMatrixBased(), Bd2Vec(foldpt), par_chan, (@optic _.β), nothing, nothing)
J_ana = BK.jacobian(_fold_ma_problem, Bd2Vec(foldpt), par_chan)
@test norminf(J_ana - J_fold_fd) < 1e-5

###
Jac_fold_MA(u0, p, pb::FoldProblemMinimallyAugmented) = (return (x=u0, params=p, prob = pb))
res_explicit = BK.FoldLinearSolverMinAug()(Jac_fold_MA(foldpt, par_chan, foldpb), Vec2Bd(rhs))

# test whether the Jacobian Matrix for the Fold problem is correct
@test norminf(J_ana[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1]) == 0

# we test the expression for σp
σp_fd = J_fold_fd[end,end]
σp_fd_ana = J_ana[end,end]
@test σp_fd ≈ σp_fd_ana atol = 1e-5

# we test the expression for σx
σx_fd = J_fold_fd[end,1:end-1]
σx_ana = J_ana[end,1:end-1]
@test σx_fd ≈ σx_ana rtol = 1e-2

σx_fd - σx_ana |> norminf
J_ana[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1] |> norminf

# check our solution of the bordered problem
res_exp = J_ana \ rhs
@test norm(res_exp - Bd2Vec(res_explicit[1]), Inf64) < 1e-10
#############################################
@reset foldpb.prob_vf.VF.isSymmetric = true
_fold_ma_problem = BK.FoldMAProblem(foldpb, BK. MinAugMatrixBased(), Bd2Vec(foldpt), par_chan, (@optic _.β), nothing, nothing)
rhs = rand(n+1)
Jac_fold_fdMA(u0) = ForwardDiff.jacobian( u -> foldpbVec(u, par_chan), u0)
J_fold_fd = Jac_fold_fdMA(Bd2Vec(foldpt))
res_fd =  J_fold_fd \ rhs

Jac_fold_MA(u0, p, pb::FoldProblemMinimallyAugmented) = (return (x=u0, params=p, prob = pb))
jacFoldSolver = BK.FoldLinearSolverMinAug()
J_ana = BK.jacobian(_fold_ma_problem, Bd2Vec(foldpt), par_chan)
res_explicit = jacFoldSolver(Jac_fold_MA(foldpt, par_chan, foldpb), Vec2Bd(rhs))

Jac_fold_MA(foldpt, 0.01, foldpb)[2]

# test whether the Jacobian Matrix for the Fold problem is correct
@test J_ana[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1] |> norminf == 0

# we test the expression for σp
σp_fd = J_fold_fd[end,end]
σp_fd_ana = J_ana[end,end]
@test σp_fd ≈ σp_fd_ana atol = 1e-5

# we test the expression for σx
σx_fd = J_fold_fd[end,1:end-1]
σx_ana = J_ana[end,1:end-1]
@test σx_fd ≈ σx_ana rtol = 1e-2

σx_fd - σx_ana |> norminf
J_ana[1:end-1,1:end-1] - J_fold_fd[1:end-1,1:end-1] |> norminf

# check our solution of the bordered problem
res_exp = J_ana \ rhs
@test norm(res_exp - Bd2Vec(res_explicit[1]), Inf64) < 1e-10