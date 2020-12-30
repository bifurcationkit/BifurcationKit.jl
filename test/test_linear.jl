# using Revise
using Test, BifurcationKit, LinearAlgebra, SparseArrays, Arpack
const BK = BifurcationKit
####################################################################################################
# test the type BorderedArray and the different methods associated to it
z_pred = BorderedArray(rand(10),1.0)
tau_pred = BorderedArray(rand(10),2.0)
BK.minus!(z_pred, tau_pred)
BK.eltype(z_pred)

axpy!(2. /3, tau_pred, z_pred)
axpby!(2. /3, tau_pred, 1.0, z_pred)
dot(z_pred, tau_pred)

dottheta = BK.DotTheta((x,y)->dot(x,y)/length(x))

dottheta(z_pred, 0.1)
dottheta(z_pred, tau_pred, 0.1)
dottheta(z_pred.u, tau_pred.u, 1.0, 1.0, 0.1)

z = BorderedArray(z_pred, rand(10))
z2 = BorderedArray(z_pred, rand(10))
zero(z2);zero(z_pred)
@test length(z_pred) == 11

copyto!(z,z2)
BK.minus(z.u,z2.u);BK.minus!(z.u,z2.u)
BK.minus(1.,2.);BK.minus!(1.,2.)
rmul!(z_pred, 1.0)
rmul!(z_pred, true)
mul!(z_pred, tau_pred, 1.0)

z_predC = BorderedArray(ComplexF64.(z_pred.u), ComplexF64.(z_pred.u))
z3 = similar(z_predC, ComplexF64)
mul!(z3, z3, 1.0)

z_sim = BorderedArray(rand(3), rand(2))
z_sim2 = similar(z_sim)
typeof(z_sim) == typeof(z_sim2)
####################################################################################################
# test the bordered linear solvers
println("--> Test linear Bordered solver")
J0 = rand(100,100) * 0.9 - I
rhs = rand(100)
sol_explicit = J0 \ rhs

linBdsolver = BK.BorderingBLS(solver = DefaultLS(), checkPrecision=true)
sol_bd1u, sol_bd1p, _, _ = linBdsolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end])
@test sol_explicit[1:end-1] ≈ sol_bd1u
@test sol_explicit[end] ≈ sol_bd1p

ls = GMRESIterativeSolvers(reltol = 1e-9, N = length(rhs)-1)
linBdsolver = BK.BorderingBLS(ls)
sol_bd2u, sol_bd2p, _, _ = linBdsolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end])
@test sol_explicit[1:end-1] ≈ sol_bd2u
@test sol_explicit[end] ≈ sol_bd2p

ls = GMRESKrylovKit(dim = length(rhs)-1)
linBdsolver = BK.BorderingBLS(ls)
sol_bd2u, sol_bd2p, _, _ = linBdsolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end])
@test sol_explicit[1:end-1] ≈ sol_bd2u
@test sol_explicit[end] ≈ sol_bd2p

linBdsolver = BK.MatrixBLS(ls)
sol_bd3u, sol_bd3p, _, _ = linBdsolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end])
@test sol_explicit[1:end-1] ≈ sol_bd3u
@test sol_explicit[end] ≈ sol_bd3p

 BK.MatrixFreeBLS(nothing   )
linBdsolver = BK.MatrixFreeBLS(ls)
sol_bd3u, sol_bd3p, _, _ = linBdsolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end])
@test sol_explicit[1:end-1] ≈ sol_bd3u
@test sol_explicit[end] ≈ sol_bd3p

linBdsolver = BK.MatrixFreeBLS(GMRESIterativeSolvers(reltol = 1e-9, N = size(J0, 1)))
sol_bd4u, sol_bd4p, _, _ = linBdsolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end])
@test sol_explicit[1:end-1] ≈ sol_bd4u
@test sol_explicit[end] ≈ sol_bd4p

# test the bordered linear solvers as used in newtonPseudoArcLength
xiu = rand()
xip = rand()

linBdsolver = BK.BorderingBLS(DefaultLS())
sol_bd1u, sol_bd1p, _, _ = linBdsolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end], xiu, xip)

linBdsolver = BK.MatrixFreeBLS(ls)
sol_bd2u, sol_bd2p, _, _ = linBdsolver(J0[1:end-1,1:end-1], J0[1:end-1,end], J0[end,1:end-1], J0[end,end], rhs[1:end-1], rhs[end], xiu, xip)

@test sol_bd1u ≈ sol_bd2u
@test sol_bd1p ≈ sol_bd2p
####################################################################################################
# test the linear solvers for matrix free formulations
J0 = I + sprand(100,100,0.1)
Jmf = x -> J0 * x
x0 = rand(100)
ls = DefaultLS()
out = ls(J0, x0)

# speciric linear solver built from a bordered linear solver
linSolver = BK.LSFromBLS()
sol_lin0 = linSolver(J0, x0)
@test sol_lin0[1] ≈ out[1]
sol_lin0 = linSolver(J0, x0, 2x0)
@test sol_lin0[1] ≈ out[1]
@test sol_lin0[2] ≈ 2out[1]

ls = GMRESKrylovKit(rtol = 1e-9, dim = 100)
outkk = ls(J0, x0)
@test out[1] ≈ outkk[1]
outkk = ls(Jmf, x0)
@test out[1] ≈ outkk[1]
outkk = ls(Jmf, x0; a₀ = 0., a₁ = 1.)
outkk = ls(Jmf, x0; a₀ = 0., a₁ = 1.5)
outkk = ls(Jmf, x0; a₀ = 1., a₁ = 1.)
outkk = ls(Jmf, x0; a₀ = 1., a₁ = 1.5)
outkk = ls(Jmf, x0; a₀ = 0.5, a₁ = 1.5)

# test preconditioner
Pl = lu(J0*0.9)
ls = GMRESKrylovKit(rtol = 1e-9, dim = 100, Pl = Pl)
outkk = ls(J0, x0)
@test out[1] ≈ outkk[1]
outkk = ls(Jmf, x0)
@test out[1] ≈ outkk[1]
outkk = ls(Jmf, x0; a₀ = 0.5, a₁ = 1.5)

ls = GMRESIterativeSolvers(N = 100, reltol = 1e-9)
outit = ls(J0, x0)
@test out[1] ≈ outit[1]
outkk = ls(J0, x0; a₀ = 0., a₁ = 1.)
outit = ls(J0, x0; a₀ = 0., a₁ = 1.5)
outit = ls(J0, x0; a₀ = 1., a₁ = 1.)
outit = ls(J0, x0; a₀ = 1., a₁ = 1.5)
outit = ls(J0, x0; a₀ = 0.5, a₁ = 1.5)

ls = GMRESIterativeSolvers!(N = 100, reltol = 1e-9)
Jom = (o,x) -> mul!(o,J0,x)
outit = ls(Jom, x0)
@test out[1] ≈ outit[1]
####################################################################################################
# test the shifted linear systems
rhs = rand(size(J0, 1))
sol0 = J0\rhs;

ls0 = GMRESIterativeSolvers(N = size(J0,1), reltol = 1e-10)
sol1, _ = @time ls0(J0, rhs)
@test norm(sol0 .- sol1, Inf) < 1e-8

h = 0.81
sol0 = (I - h.*J0)\rhs
sol1 = (I/h - J0)\rhs
@test norm(sol0 - sol1/h, Inf) < 1e-8

sol0,_ = ls0(I - h*J0, rhs)
sol1,_ = ls0(J0, rhs; a₀ = 1.0, a₁ = -h)
@test norm(sol0 - sol1,Inf) < 1e-8

ls0 = GMRESKrylovKit(atol = 1e-10)
sol0,_ = ls0(I - h*J0, rhs)
sol1,_ = ls0(J0, rhs; a₀ = 1.0, a₁ = -h)
@test norm(sol0 - sol1,Inf) < 1e-8

sol0,_ = ls0(I - h*J0, rhs)
sol1,_ = ls0(J0, rhs; a₀ = 1.0/h, a₁ = -1.)
@test norm(sol0 - sol1/h,Inf) < 1e-8


sol0,_ = ls0(I - h*J0, rhs)
sol1,_ = ls0(J0, rhs; a₀ = 1., a₁ = -h)
@test norm(sol0 - sol1,Inf) < 1e-8

####################################################################################################
# test the eigen solvers for matrix free formulations
# eil = BK.EigIterativeSolvers(tol = 1e-9)
out = Arpack.eigs(J0, nev = 20, which = :LR)

eil = BK.EigKrylovKit(tol = 1e-9)
outkk = eil(J0, 20)
geteigenvector(eil, outkk[2], 2)

eil = BK.EigKrylovKit(tol = 1e-9, x₀ = x0)
outkkmf = eil(Jmf, 20)
geteigenvector(eil, outkkmf[2], 2)

eil = BK.EigArpack(v0 = copy(x0))
outdefault = eil(J0, 20)
@test out[1] ≈ outdefault[1]
outdefault = eil(x ->J0*x, 20)
@test out[1] ≈ outdefault[1]

eil = BK.EigArnoldiMethod(;x₀ = x0)
outam = eil(J0, 20)
outam = eil(Jmf, 20)
geteigenvector(eil, outam[2], 2)

eil = BK.EigArnoldiMethod(;x₀ = x0, sigma = 1.)
outam = eil(J0, 20)
outam = eil(Jmf, 20)
geteigenvector(eil, outam[2], 2)
