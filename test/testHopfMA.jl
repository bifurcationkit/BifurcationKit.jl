# using Revise
using Test, BifurcationKit, LinearAlgebra, SparseArrays, ForwardDiff
const BK = BifurcationKit

f1(u, v) = u^2 * v

function Fbru!(f, x, p)
    (; α, β, D1, D2, l) = p
    n = div(length(x), 2)
    h = 1.0 / n; h2 = h*h
    c1 = D1 / l^2 / h2
    c2 = D2 / l^2 / h2

    u = @view x[1:n]
    v = @view x[n+1:2n]

    # Dirichlet boundary conditions
    f[1]   = c1 * (α      - 2u[1] + u[2] ) + α - (β + 1) * u[1] + f1(u[1], v[1])
    f[end] = c2 * (v[n-1] - 2v[n] + β / α)           + β * u[n] - f1(u[n], v[n])

    f[n]   = c1 * (u[n-1] - 2u[n] +  α  )  + α - (β + 1) * u[n] + f1(u[n], v[n])
    f[n+1] = c2 * (β / α  - 2v[1] + v[2])            + β * u[1] - f1(u[1], v[1])

    for i=2:n-1
          f[i] = c1 * (u[i-1] - 2u[i] + u[i+1]) + α - (β + 1) * u[i] + f1(u[i], v[i])
        f[n+i] = c2 * (v[i-1] - 2v[i] + v[i+1])           + β * u[i] - f1(u[i], v[i])
    end
    return f
end

Fbru(x, p) = Fbru!(similar(x), x, p)

function Jbru_sp(x, p)
    (; α, β, D1, D2, l) = p
    # compute the Jacobian using a sparse representation
    n = div(length(x), 2)
    h = 1.0 / n; h2 = h*h

    c1 = D1 / p.l^2 / h2
    c2 = D2 / p.l^2 / h2

    u = @view x[1:n]
    v = @view x[n+1:2n]

    diag   = zeros(eltype(x), 2n)
    diagp1 = zeros(eltype(x), 2n-1)
    diagm1 = zeros(eltype(x), 2n-1)

    diagpn = zeros(eltype(x), n)
    diagmn = zeros(eltype(x), n)

    @. diagmn = β - 2 * u * v
    @. diagm1[1:n-1] = c1
    @. diagm1[n+1:end] = c2

    @. diag[1:n]    = -2c1 - (β + 1) + 2 * u * v
    @. diag[n+1:2n] = -2c2 - u * u

    @. diagp1[1:n-1] = c1
    @. diagp1[n+1:end] = c2

    @. diagpn = u * u
    J = spdiagm(0 => diag, 1 => diagp1, -1 => diagm1, n => diagpn, -n => diagmn)
    return J
end

Jbru_ana(x, p) = ForwardDiff.jacobian(z->Fbru(z,p),x)

n = 10
par_bru = (α = 2., β = 5.45, D1 = 0.008, D2 = 0.004, l = 0.3)
sol0 = vcat(par_bru.α * ones(n), par_bru.β/par_bru.α * ones(n))
prob = BifurcationKit.BifurcationProblem(Fbru!, sol0, par_bru, (@optic _.l); J = Jbru_ana)

# test that the jacobian is well computed
@test Jbru_sp(sol0, par_bru) - Jbru_ana(sol0, par_bru) |> sparse |> nnz == 0

opt_newton = NewtonPar(tol = 1e-11, verbose = false)
out = BK.solve(BK.re_make(prob, u0 = sol0 .* (1 .+ 0.01rand(2n))), Newton(), opt_newton)

opts_br0 = ContinuationPar(p_max = 1.8, newton_options = opt_newton, detect_bifurcation = 3, nev = 16, n_inversion = 4)
br = continuation(BK.re_make(prob; u0 = out.u, params = (@set par_bru.l = 0.3)), PALC(), opts_br0,)
###################################################################################################
# Hopf continuation with automatic procedure
outhopf = newton(br, 1; start_with_eigen = false)
outhopf = newton(br, 1; start_with_eigen = true)
outhopfco = continuation(br, 1, (@optic _.β), optconthopf; start_with_eigen = true, update_minaug_every_step = 1, jacobian_ma = :minaug)
optconthopf = ContinuationPar(dsmin = 0.001, dsmax = 0.15, ds= 0.01, p_max = 6.8, p_min = 0., newton_options = opt_newton, max_steps = 5, detect_bifurcation = 2)

# Continuation of the Hopf Point using Dense method
ind_hopf = 1
hopfpt = BK.hopf_point(br, ind_hopf)
bifpt = br.specialpoint[ind_hopf]
hopfvariable = HopfProblemMinimallyAugmented(
                    (@set prob.VF.d2F = nothing),
                    conj.(br.eig[bifpt.idx].eigenvecs[:, bifpt.ind_ev]),
                    (br.eig[bifpt.idx].eigenvecs[:, bifpt.ind_ev]),
                    opts_br0.newton_options.linsolver)

Bd2Vec(x) = vcat(x.u, x.p)
Vec2Bd(x) = BorderedArray(x[1:end-2], x[end-1:end])
hopfpbVec(x, p) = Bd2Vec(hopfvariable(Vec2Bd(x),p))

# finite differences Jacobian
Jac_hopf_fdMA(u0, p) = ForwardDiff.jacobian( u-> hopfpbVec(u, p), u0)
# ``analytical'' jacobian
Jac_hopf_MA(u0, p, pb::HopfProblemMinimallyAugmented) = (return (x=u0, params=p,hopfpb=pb))

rhs = rand(length(hopfpt))
jac_hopf_fd = Jac_hopf_fdMA(Bd2Vec(hopfpt), par_bru)
sol_fd = jac_hopf_fd \ rhs

# test against analytical jacobian
_hopf_ma_problem = BK.HopfMAProblem(hopfvariable, BK. MinAugMatrixBased(), Bd2Vec(hopfpt), par_bru, (@optic _.β), prob.plotSolution, prob.recordFromSolution)
J_ana = BK.jacobian(_hopf_ma_problem, Bd2Vec(hopfpt), par_bru)
@test norminf(J_ana - jac_hopf_fd) < 1e-3

# create a linear solver
hopfls = BK.HopfLinearSolverMinAug()
sol_ma,  = hopfls(Jac_hopf_MA(hopfpt, par_bru, hopfvariable), BorderedArray(rhs[1:end-2],rhs[end-1:end]))

# we test the expression for σp
σp_fd = Complex(jac_hopf_fd[end-1,end-1], jac_hopf_fd[end, end-1])
σp_fd_ana = Complex(J_ana[end-1,end-1], J_ana[end, end-1])
@test σp_fd ≈ σp_fd_ana rtol = 1e-4

# we test the expression for σω
σω_fd = Complex(jac_hopf_fd[end-1,end], jac_hopf_fd[end, end])
σω_fd_ana = Complex(J_ana[end-1,end], J_ana[end, end])
@test σω_fd ≈ σω_fd_ana rtol = 1e-4

# we test the expression for σx
σx_fd = jac_hopf_fd[end-1, 1:end-2] + Complex(0,1) * jac_hopf_fd[end, 1:end-2]
σx_fd_ana = J_ana[end-1, 1:end-2] + Complex(0,1) * J_ana[end, 1:end-2]
@test σx_fd ≈ σx_fd_ana rtol = 1e-3

outhopf = newton(br, 1)
@test BK.converged(outhopf)

pb_hopf_perso = BK.BifurcationProblem((u, p) -> hopfvariable(u, p),
                hopfpt, par_bru;
                J = (x, p) -> Jac_hopf_MA(x, p, hopfvariable),)
outhopf = BK.solve(pb_hopf_perso, Newton(), NewtonPar(verbose = false, linsolver = BK.HopfLinearSolverMinAug()))
@test BK.converged(outhopf)

# version with analytical Hessian = 2 P(du2) P(du1) QU + 2 PU P(du1) Q(du2) + 2PU P(du2) Q(du1)
function d2F(x, p1, du1, du2)
    n = div(length(x), 2)
    out = 2 .* x[n+1:end] .* du1[1:n] .* du2[1:n] .+
                       2 .* x[1:n] .* du1[1:n] .* du2[n+1:end] .+
                       2 .* x[1:n] .* du2[1:n] .* du1[1:n]
    return vcat(out, -out)
end

# add specific hessian
br_d2f = (@set br.prob.VF.d2F = d2F)

outhopf = newton(br_d2f, 1)
@test BK.converged(outhopf)

br_hopf = continuation(br, ind_hopf, (@optic _.β),
            ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, p_max = 6.5, p_min = 0.0, a = 2., max_steps = 3, newton_options = NewtonPar(verbose = false)), 
            jacobian_ma = :minaug)

br_hopf = continuation(br_d2f, ind_hopf, (@optic _.β), 
            ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, p_max = 6.5, p_min = 0.0, a = 2., max_steps = 3, newton_options = NewtonPar(verbose = false)), 
            jacobian_ma = :minaug)
####################################################################################################
ind_hopf = 1
hopfpt = BK.hopf_point(br, ind_hopf)

l_hopf = hopfpt.p[1]
ωH     = hopfpt.p[2] |> abs
M = 20

orbitguess = zeros(2n, M)
phase = []; scalphase = []
vec_hopf = geteigenvector(opt_newton.eigsolver, br.eig[br.specialpoint[ind_hopf].idx][2], br.specialpoint[ind_hopf].ind_ev-1)
for ii=1:M
    t = (ii-1)/(M-1)
    orbitguess[:, ii] .= real.(hopfpt.u + 26 * 0.1 * vec_hopf * exp(-2pi * complex(0, 1) * (t - .252)))
end

orbitguess_f = vcat(vec(orbitguess), 2pi/ωH) |> vec

# test guess using function
l_hopf, Th, orbitguess2, hopfpt, vec_hopf = BK.guess_from_hopf(br, ind_hopf, opt_newton.eigsolver, M, 2.6; phase = 0.252)

prob = BifurcationKit.BifurcationProblem(Fbru!, sol0, par_bru, (@optic _.l);
        J = Jbru_sp)

poTrap = PeriodicOrbitTrapProblem(prob, real.(vec_hopf), hopfpt.u, M, 2n)

jac_PO_fd = BK.finite_differences(x -> BK.residual(poTrap, x, (@set par_bru.l = l_hopf + 0.01)), orbitguess_f)
jac_PO_sp = poTrap(Val(:JacFullSparse), orbitguess_f, (@set par_bru.l = l_hopf + 0.01))

# test of the Jacobian for PeriodicOrbit via Finite differences VS the FD associated jacobian
# test jacobian expression for Periodic Orbit solve problem
@test norm(jac_PO_fd - jac_PO_sp, Inf64) < 1e-4

# test various jacobians and methods
jac_PO_sp =  poTrap(Val(:BlockDiagSparse), orbitguess_f, (@set par_bru.l = l_hopf + 0.01))
# BK.Jc(poTrap, reshape(orbitguess_f[1:end-1], 2n, M), par_bru, reshape(orbitguess_f[1:end-1], 2n, M))
# BK.Jc(poTrap, orbitguess_f, par_bru, orbitguess_f)

# newton to find Periodic orbit
_prob = BK.BifurcationProblem((x, p) -> BK.residual(poTrap, x, p), copy(orbitguess_f), (@set par_bru.l = l_hopf + 0.01); J = (x, p) ->  poTrap(Val(:JacFullSparse),x,p))
opt_po = NewtonPar(tol = 1e-8, max_iterations = 150)
outpo_f = @time BK.solve(_prob, Newton(), opt_po)
@test BK.converged(outpo_f)
    # println("--> T = ", outpo_f[end])
# flag && printstyled(color=:red, "--> T = ", outpo_f[end], ", amplitude = ", BK.amplitude(outpo_f, n, M; ratio = 2),"\n")

outpo_f = newton(poTrap, orbitguess_f, opt_po; jacobianPO = :FullLU)
BK.converged(outpo_f)

# jacobian of the functional
Jpo2 = poTrap(Val(:JacCyclicSparse), orbitguess_f, (@set par_bru.l = l_hopf + 0.01))

# calcul des exposants de Floquet
floquetES = FloquetQaD(DefaultEig())

# calcul des exposants de Floquet, extract full vector
pbwrap = BK.WrapPOTrap(poTrap, :dense, orbitguess_f, par_bru, nothing, nothing, nothing)
floquetES(Val(:ExtractEigenVector), pbwrap, orbitguess_f, par_bru, orbitguess_f[1:2n])

# continuation of periodic orbits using :BorderedLU linear algorithm
opts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.05, ds= 0.001, p_max = 2.3, max_steps = 3, newton_options = NewtonPar(verbose = false), detect_bifurcation = 1)
br_pok2 = continuation((@set poTrap.jacobian = :BorderedLU), orbitguess_f, PALC(), opts_po_cont)

# test of simple calls to newton / continuation
deflationOp = DeflationOperator(2.0, (x,y) -> dot(x[1:end-1], y[1:end-1]), 1.0, [zero(orbitguess_f)])
# opt_po = NewtonPar(tol = 1e-8, verbose = false, max_iterations = 15)
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.03, ds= 0.01, p_max = 3.0, max_steps = 3, newton_options = (@set opt_po.verbose = false), nev = 2, tol_stability = 1e-8, detect_bifurcation = 1)
for linalgo in [:FullLU, :BorderedLU, :FullSparseInplace]
    @show linalgo
    # with deflation
    @time newton((@set poTrap.jacobian = linalgo),
            copy(orbitguess_f), deflationOp, opt_po; normN = norminf)
    # classic Newton-Krylov
    outpo_f = @time newton((@set poTrap.jacobian = linalgo),
            copy(orbitguess_f), opt_po; normN = norminf)
    # continuation
    br_pok2 = @time continuation((@set poTrap.jacobian = linalgo),
            copy(orbitguess_f), PALC(),
            opts_po_cont; normC = norminf)
end

# test of Matrix free computation of Floquet exponents
eil = EigKrylovKit(x₀ = rand(2n))
ls = GMRESKrylovKit()
ls = DefaultLS()
opt_po = NewtonPar(tol = 1e-8, max_iterations = 10, linsolver = ls, eigsolver = eil)
opts_po_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.03, ds= 0.01, p_max = 3.0, max_steps = 3, newton_options = (@set opt_po.verbose = false), nev = 2, tol_stability = 1e-8, detect_bifurcation = 2)
br_pok2 = continuation(poTrap, outpo_f.u, PALC(), opts_po_cont; normC = norminf)
