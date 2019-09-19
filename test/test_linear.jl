using Test, PseudoArcLengthContinuation, LinearAlgebra, SparseArrays, Arpack
const Cont = PseudoArcLengthContinuation
####################################################################################################
# test the type BorderedArray and the different methods associated to it
z_pred = BorderedArray(rand(10),1.0)
tau_pred = BorderedArray(rand(10),2.0)
Cont.minus!(z_pred, tau_pred)
Cont.eltype(z_pred)

axpy!(2. /3, tau_pred, z_pred)
axpby!(2. /3, tau_pred, 1.0, z_pred)
dot(z_pred, tau_pred)
Cont.dottheta(z_pred, tau_pred, 0.1)
Cont.dottheta(z_pred.u, tau_pred.u, 1.0, 1.0, 0.1)

z = BorderedArray(z_pred, rand(10))
z2 = BorderedArray(z_pred, rand(10))
zero(z2);zero(z_pred)
@test length(z_pred) == 11

copyto!(z,z2)
Cont.minus(z.u,z2.u);Cont.minus!(z.u,z2.u)
Cont.minus(1.,2.);Cont.minus!(1.,2.)
rmul!(z_pred, 1.0)
rmul!(z_pred, true)
mul!(z_pred, tau_pred, 1.0)

z_predC = BorderedArray(ComplexF64.(z_pred.u), ComplexF64.(z_pred.u))
z3 = similar(z_predC, ComplexF64)
mul!(z3, z3, 1.0)
####################################################################################################
# test the bordered linear solvers
println("--> Test linear Bordered solver")
J0 = rand(10,10) * 0.1 + I
rhs = rand(10)
sol_explicit = J0 \ rhs

linBdsolver = Cont.BorderingBLS(Default())
sol_bd1, sol_bd2, _ = linBdsolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end])
@test sol_explicit[1:end-1] ≈ sol_bd1

ls = GMRES_KrylovKit{Float64}(rtol = 1e-9, dim = 9, verbose = 2)
linBdsolver = Cont.MatrixFreeBLS(ls)
sol_bd_2, _, cv, _ = linBdsolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end])
@test cv
@test sol_bd_2 ≈ sol_explicit[1:end-1]

@test sol_explicit[1:end-1] ≈ sol_bd1
@test sol_explicit[end] ≈ sol_bd2
####################################################################################################
# test the linear solvers for matrix free formulations
J0 = I + sprand(100,100,0.1)
Jmf = x -> J0*x
x0 = rand(100)
ls = Default()
out = ls(J0, x0)

ls = GMRES_KrylovKit{Float64}(rtol = 1e-9, dim = 100)
outkk = ls(J0, x0)
@test out[1] ≈ outkk[1]
outkk = ls(Jmf, x0)
@test out[1] ≈ outkk[1]

ls = GMRES_IterativeSolvers{Float64}(N = 100, tol = 1e-9)
outit = ls(J0, x0)
@test out[1] ≈ outit[1]
####################################################################################################
# test the eigen solvers for matrix free formulations
eil = Cont.eig_IterativeSolvers(tol = 1e-9)
out = Arpack.eigs(J0, nev = 20, which = :LR)

eil = Cont.eig_KrylovKit(tol = 1e-9)
outkk = eil(J0, 20)
getEigenVector(eil, outkk[2], 2)

eil = Cont.eig_MF_KrylovKit(tol = 1e-9, x₀ = x0)
outkkmf = eil(Jmf, 20)
getEigenVector(eil, outkkmf[2], 2)

eil = Cont.Default_eig_sp()
outdefault = eil(J0, 20)
@test out[1] ≈ outdefault[1]
