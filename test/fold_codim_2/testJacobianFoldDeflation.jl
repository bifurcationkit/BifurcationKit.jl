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
# Fold continuation
let
    outfold = newton(br, 2; start_with_eigen = false)
    outfold = newton(br, 2; start_with_eigen = true)
    @test  BK.converged(outfold)
    outfold = BK.newton_fold((@set br.prob.VF.isSymmetric = true), 2; start_with_eigen = false, issymmetric = true)
    @test BK.converged(outfold)

    optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds= 0.01, p_max = 4.1, p_min = 0., newton_options = NewtonPar(tol = 1e-8), max_steps = 50, detect_bifurcation = 0)
    for eig_st in (true, false)
        outfoldco = continuation(br, 2, (@optic _.β), optcontfold; start_with_eigen = eig_st, update_minaug_every_step = 1)
        outfoldco = continuation((@set br.prob.VF.isSymmetric = true), 2, (@optic _.β), optcontfold; start_with_eigen = eig_st, update_minaug_every_step = 1)
        # test use of jacobian_adjoint
        outfoldco = continuation((@set br.prob.VF.Jᵗ = (x,p)->transpose(BK.jacobian(prob,x,p))), 2, (@optic _.β), optcontfold; start_with_eigen = eig_st, update_minaug_every_step = 1)
    end
end

# test of Jacobian expression
# manual handling
let
    indfold = 1
    foldpt = BK.fold_point(br, indfold)
    foldpb = FoldProblemMinimallyAugmented(
                    (@set prob.VF.d2F = nothing), # this is for debug array
                    br.specialpoint[indfold].x,
                    br.specialpoint[indfold].x,
                    opts_br0.newton_options.linsolver)

    outfold = BK.newton_fold(prob, foldpt, par_chan, br.specialpoint[indfold].x, br.specialpoint[indfold].x, NewtonPar(tol = 1e-10), normN = norminf)
    @test BK.converged(outfold)

    # we now use the newton refined point
    foldpt = outfold.u

    # we define the following wrappers to be able to use ForwardDiff
    Bd2Vec(x) = vcat(x.u, x.p)
    Vec2Bd(x) = BorderedArray(x[1:end-1], x[end])
    foldpbVec(x,p) = Bd2Vec(foldpb(Vec2Bd(x),p))

    rhs = rand(n+1)
    Jac_fold_fdMA(u0) = ForwardDiff.jacobian( u -> foldpbVec(u, par_chan), u0)
    J_fold_fwdiff = Jac_fold_fdMA(Bd2Vec(foldpt))
    res_fd =  J_fold_fwdiff \ rhs

    # test against analytical jacobian, compare to ForwardDiff
    # The main error comes from the borders which are evaluated by finite differences in the functional
    _fold_ma_problem = BK.FoldMAProblem(foldpb, BK. MinAugMatrixBased(), Bd2Vec(foldpt), par_chan, (@optic _.β), nothing, nothing)
    BK.has_adjoint(_fold_ma_problem)
    BK.is_symmetric(_fold_ma_problem)
    BK.residual!(_fold_ma_problem, zero(Bd2Vec(foldpt)) ,Bd2Vec(foldpt), par_chan)
    J_ana = BK.jacobian(_fold_ma_problem, Bd2Vec(foldpt), par_chan)

    # test whether the Jacobian Matrix for the Fold problem is correct
    @test norminf(J_ana[1:end-1,1:end-1] - J_fold_fwdiff[1:end-1,1:end-1]) == 0
    @test norminf(J_ana - J_fold_fwdiff) < 1e-5

    ###
    Jac_fold_MA(u0, p, pb::FoldProblemMinimallyAugmented) = (return (x=u0, params=p, prob = pb))
    res_explicit = BK.FoldLinearSolverMinAug()(Jac_fold_MA(foldpt, par_chan, foldpb), Vec2Bd(rhs))

    # we test the expression for σp
    σp_fd = J_fold_fwdiff[end,end]
    σp_fd_ana = J_ana[end,end]
    @test σp_fd ≈ σp_fd_ana atol = 1e-5

    # we test the expression for σx
    σx_fd = J_fold_fwdiff[end,1:end-1]
    σx_ana = J_ana[end,1:end-1]
    @test σx_fd ≈ σx_ana rtol = 1e-2

    σx_fd - σx_ana |> norminf
    J_ana[1:end-1,1:end-1] - J_fold_fwdiff[1:end-1,1:end-1] |> norminf

    # check our solution of the bordered problem
    res_exp = J_ana \ rhs
    @test norm(res_exp - Bd2Vec(res_explicit[1]), Inf64) < 1e-10

    # test the symmetric case
    @reset foldpb.prob_vf.VF.isSymmetric = true
    _fold_ma_problem = BK.FoldMAProblem(foldpb, BK. MinAugMatrixBased(), Bd2Vec(foldpt), par_chan, (@optic _.β), nothing, nothing)
    rhs = rand(n+1)
    Jac_fold_fdMA(u0) = ForwardDiff.jacobian( u -> foldpbVec(u, par_chan), u0)
    J_fold_fwdiff = Jac_fold_fdMA(Bd2Vec(foldpt))
    res_fd =  J_fold_fwdiff \ rhs

    jacFoldSolver = BK.FoldLinearSolverMinAug()
    J_ana = BK.jacobian(_fold_ma_problem, Bd2Vec(foldpt), par_chan)
    res_explicit = jacFoldSolver(Jac_fold_MA(foldpt, par_chan, foldpb), Vec2Bd(rhs))

    Jac_fold_MA(foldpt, 0.01, foldpb)[2]

    # test whether the Jacobian Matrix for the Fold problem is correct
    @test J_ana[1:end-1,1:end-1] - J_fold_fwdiff[1:end-1,1:end-1] |> norminf == 0

    # we test the expression for σp
    σp_fd = J_fold_fwdiff[end,end]
    σp_fd_ana = J_ana[end,end]
    @test σp_fd ≈ σp_fd_ana atol = 1e-5

    # we test the expression for σx
    σx_fd = J_fold_fwdiff[end,1:end-1]
    σx_ana = J_ana[end,1:end-1]
    @test σx_fd ≈ σx_ana rtol = 1e-2
    @test σx_fd ≈ σx_ana atol = 1e-4

    # check our solution of the bordered problem
    res_exp = J_ana \ rhs
    @test norm(res_exp - Bd2Vec(res_explicit[1]), Inf64) < 1e-10
end