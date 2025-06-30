# using Revise
using Test, BifurcationKit, LinearAlgebra, SparseArrays, Arpack
const BK = BifurcationKit
####################################################################################################
BK.closesttozero(rand(10))
BK.norm2sqr(rand(2))
BK.print_ev(rand(2))
BK._print_line(1,nothing,1)
####################################################################################################
# test VectorInterface
# x0 = rand(5)
# y0 = rand(5)
# x0b = BorderedArray(x0[begin:end-1], x0[end])
# y0b = BorderedArray(y0[begin:end-1], y0[end])

# @test BK.zerovector(x0b) |> norm == 0
# out = BK.add(y0b,x0b,1.1,-0.1)
# @test (vcat(out.u,out.p) == 1.1 .*x0 .-0.1 .*y0)
# out = BK.add!(copy(y0b),x0b,1.1,-0.1)
# @test (vcat(out.u,out.p) == 1.1 .*x0 .-0.1 .*y0)
# out = BK.add!!(copy(y0b),x0b,1.1,-0.1)
# @test (vcat(out.u,out.p) == 1.1 .*x0 .-0.1 .*y0)
####################################################################################################
# test the type BorderedArray and the different methods associated to it
let
    z_pred = BorderedArray(rand(10), 1.0)
    tau_pred = BorderedArray(rand(10), 2.0)
    BK.minus!(z_pred, tau_pred)
    BK.eltype(z_pred)

    axpy!(2. /3, tau_pred, z_pred)
    axpby!(2. /3, tau_pred, 1.0, z_pred)
    dot(z_pred, tau_pred)

    dottheta = BK.DotTheta((x, y) -> dot(x, y) / length(x))

    dottheta(z_pred, 0.1)
    dottheta(z_pred, tau_pred, 0.1)
    dottheta(z_pred.u, tau_pred.u, 1.0, 1.0, 0.1)

    z = BorderedArray(z_pred, rand(10))
    z2 = BorderedArray(z_pred, rand(10))
    zero(z2); zero(z_pred)
    @test length(z_pred) == 11

    copyto!(z,z2)
    BK.minus(z.u,z2.u); BK.minus!(z.u,z2.u)
    BK.minus(1.,2.); BK.minus!(1.,2.)
    rmul!(z_pred, 1.0)
    rmul!(z_pred, true)
    mul!(z_pred, tau_pred, 1.0)

    z_predC = BorderedArray(ComplexF64.(z_pred.u), ComplexF64.(z_pred.u))
    z3 = similar(z_predC, ComplexF64)
    mul!(z3, z3, 1.0)

    z_sim = BorderedArray(rand(3), rand(2))
    z_sim2 = similar(z_sim)
    typeof(z_sim) == typeof(z_sim2)
end
####################################################################################################
# test _axpy_op
let
    J0 = rand(100, 100)
    dx = rand(size(J0, 1))
    _o = rand(ComplexF64, size(J0, 1))
    a₀ = rand(ComplexF64)
    a₁ = -1.432
    BK._axpy(J0, 0, a₁)
    BK._axpy(J0, 1, 1)
    out1 = BK._axpy_op(J0, dx, a₀, a₁)
    out2 = a₀ * dx + a₁ * J0 * dx
    @test out1 ≈ out2
    @test a₀ * I + a₁ * J0  ≈ BK._axpy(J0, a₀, a₁)
    BK._axpy_op!(_o, J0, dx, a₀, a₁)
    BK._axpy_op!(_o, J0, dx, 0, 1)
    BK._axpy_op!(_o, J0, dx, 0, a₁)
    BK._axpy_op!(_o, J0, dx, 1, 1)
    BK._axpy_op!(_o, J0, dx, 1, a₁)
end
####################################################################################################
# test of MatrixFreeBLSmap
let
    J0 = rand(100, 100)
    map_bls = BK.MatrixFreeBLSmap(J0, rand(size(J0,1)), rand(size(J0,1)), rand(), rand(), dot)
    x_bd = BorderedArray(rand(100), rand())
    x_bd_v = vcat(copy(x_bd.u), x_bd.p)
    o1 = map_bls(x_bd)
    o2 = map_bls(x_bd_v)
    @test o1.u ≈ o2[begin:end-1]
    @test o1.p ≈ o2[end]
end
####################################################################################################
let
    # test of the linear  solvers
    J0 = rand(100,100) * 0.1 - I
    rhs = rand(100)
    sol_explicit = J0 \ rhs

    ls = DefaultLS()
    _sol, = ls(J0, rhs)
    @test _sol == sol_explicit

    ls = BK.DefaultPILS()
    _sol, = ls(J0, rhs)
    @test _sol == sol_explicit

    ls = GMRESIterativeSolvers(N = 100, reltol = 1e-16)
    _sol, = ls(J0, rhs)
    @test _sol ≈ sol_explicit

    ls = GMRESKrylovKit(rtol = 1e-16)
    _sol, = ls(J0, rhs)
    @test _sol ≈ sol_explicit

    # Case with a shift
    sol_explicit = (0.9J0 + 0.1I) \ rhs

    ls = DefaultLS()
    _sol, = ls(J0, rhs; a₀ = 0.1, a₁ = 0.9)
    @test _sol == sol_explicit

    ls = GMRESIterativeSolvers(N = 100, reltol = 1e-16)
    _sol, = ls(J0, rhs; a₀ = 0.1, a₁ = 0.9)
    @test _sol ≈ sol_explicit

    _sol, = ls(x -> J0 * x, rhs; a₀ = 0.1, a₁ = 0.9)
    @test _sol ≈ sol_explicit

    ls = GMRESKrylovKit(rtol = 1e-16)
    _sol, = ls(J0, rhs; a₀ = 0.1, a₁ = 0.9)
    @test _sol ≈ sol_explicit

    _sol, = ls(x -> J0 * x, rhs; a₀ = 0.1, a₁ = 0.9)
    @test _sol ≈ sol_explicit

    ls = KrylovLS(rtol = 1e-16, Pl = I)
    _sol, = ls(J0, rhs; a₀ = 0.1, a₁ = 0.9)
    @test _sol ≈ sol_explicit

    ls = KrylovLSInplace(rtol = 1e-16, Pl = I, m = 100, n = 100, verbose = 0 )
    _sol, = ls(J0, rhs; a₀ = 0.1, a₁ = 0.9)
    @test _sol ≈ sol_explicit

    ls = KrylovLSInplace(rtol = 1e-16, Pl = I, m = 100, n = 100, verbose = 0, is_inplace = true )
    _sol, = ls(J0, rhs; a₀ = 0.1, a₁ = 0.9)
    @test _sol ≈ sol_explicit
end

J0 = rand(100,100) * 0.1 - I
rhs = rand(100)
ls = KrylovLSInplace(rtol = 1e-16, m = 100, n = 100, verbose = 2)
_sol, = ls(J0, rhs; )
sol_explicit = (J0) \ rhs
@test _sol ≈ sol_explicit
####################################################################################################
# test the bordered linear solvers
let
    J0 = rand(5,5) * 0.9 - I
    rhs = rand(5)
    sol_explicit = (J0 + 0.2spdiagm(0 => vcat(ones(4),0))) \ rhs

    args_bls = (J0[begin:end-1,1:end-1], J0[begin:end-1,end], J0[end,1:end-1], J0[end,end], rhs[begin:end-1], rhs[end])

    linBdsolver = BK.BorderingBLS(solver = DefaultLS(), check_precision=true)
    sol_bd1u, sol_bd1p, _, _ = linBdsolver(args_bls...; shift = 0.2)
    @test sol_explicit[begin:end-1] ≈ sol_bd1u
    @test sol_explicit[end] ≈ sol_bd1p

    ls = GMRESIterativeSolvers(reltol = 1e-11, N = length(rhs)-1)
    linBdsolver = BK.BorderingBLS(ls)
    sol_bd2u, sol_bd2p, _, _ = linBdsolver(args_bls...; shift = 0.2)
    @test sol_explicit[begin:end-1] ≈ sol_bd2u
    @test sol_explicit[end] ≈ sol_bd2p

    ls = GMRESKrylovKit(dim = length(rhs) - 1, rtol = 1e-11)
    linBdsolver = BK.BorderingBLS(ls)
    sol_bd2u, sol_bd2p, _, _ = linBdsolver(args_bls...; shift = 0.2)
    @test sol_explicit[begin:end-1] ≈ sol_bd2u
    @test sol_explicit[end] ≈ sol_bd2p

    linBdsolver = BK.MatrixBLS(ls)
    sol_bd3u, sol_bd3p, _, _ = linBdsolver(args_bls...; shift = 0.2)
    @test sol_explicit[begin:end-1] ≈ sol_bd3u
    @test sol_explicit[end] ≈ sol_bd3p

    BK.MatrixFreeBLS(nothing)
    linBdsolver = BK.MatrixFreeBLS(ls, true)
    sol_bd3u, sol_bd3p, _, _ = linBdsolver(args_bls...; shift = 0.2)
    @test sol_explicit[begin:end-1] ≈ sol_bd3u
    @test sol_explicit[end] ≈ sol_bd3p rtol = 1e-6

    linBdsolver = BK.MatrixFreeBLS(GMRESIterativeSolvers(reltol = 1e-9, N = size(J0, 1)))
    sol_bd4u, sol_bd4p, _, _ = linBdsolver(args_bls...; shift = 0.2)
    @test sol_explicit[begin:end-1] ≈ sol_bd4u
    @test sol_explicit[end] ≈ sol_bd4p

    # test the bordered linear solvers as used in newtonPALC
    xiu = rand()
    xip = rand()

    linBdsolver = BK.BorderingBLS(DefaultLS())
    sol_bd1u, sol_bd1p, _, _ = linBdsolver(args_bls..., xiu, xip)

    linBdsolver = BK.MatrixFreeBLS(ls)
    sol_bd2u, sol_bd2p, _, _ = linBdsolver(args_bls..., xiu, xip)

    @test sol_bd1u ≈ sol_bd2u
    @test sol_bd1p ≈ sol_bd2p
end
####################################################################################################
# test the bordered linear solvers in the case of blocks
let
    J0 = rand(100,100) * 0.2 - I
    rhs = rand(100)
    m = 3
    J = J0[begin:end-m, 1:end-m]
    c = J0[end-m+1:end, end-m+1:end]
    a = Tuple(J0[begin:end-m, k] for k in size(J0,1)-m+1:size(J0,1))
    b = Tuple(J0[k, 1:end-m] for k in size(J0,1)-m+1:size(J0,1))
    @assert J0 ≈ vcat(hcat(J, hcat(a...)), hcat(adjoint(hcat(b...)), c))

    rhs = zeros(100)
    sol_explicit = J0 \ rhs

    ls = DefaultLS()

    linBdsolver = BK.MatrixBLS(ls)
    sol_bd1u, sol_bd1p, _, _ = BK.solve_bls_block(linBdsolver, J, a, b, c, rhs[begin:end-m], rhs[end-m+1:end])
    @test sol_explicit[begin:end-m] ≈ sol_bd1u
    @test sol_explicit[end-m+1:end] ≈ sol_bd1p

    for ls in (DefaultLS(), GMRESKrylovKit(), GMRESIterativeSolvers(reltol = 1e-11, N = size(J,1)))
        linBdsolver = BK.BorderingBLS(ls)
        sol_bd2u, sol_bd2p, success, it = BK.solve_bls_block(linBdsolver, J, a, b, c, rhs[begin:end-m], rhs[end-m+1:end])
        @assert success
        @test sol_explicit[begin:end-m] ≈ sol_bd2u
        @test sol_explicit[end-m+1:end] ≈ sol_bd2p
    end

    # same but matrix-free
    Jmf = x -> J * x
    for ls in (GMRESKrylovKit(), GMRESIterativeSolvers(reltol = 1e-11, N = size(J,1)))
        linBdsolver = BK.BorderingBLS(ls)
        sol_bd2u, sol_bd2p, success, it = BK.solve_bls_block(linBdsolver, Jmf, a, b, c, rhs[begin:end-m], rhs[end-m+1:end])
        @assert success
        @test sol_explicit[begin:end-m] ≈ sol_bd2u
        @test sol_explicit[end-m+1:end] ≈ sol_bd2p
    end

    # test MatrixFreeBLSmap evaluation
    blsmap = BK.MatrixFreeBLSmap(Jmf, a, b, c, nothing, dot)
    rhs_bd = BorderedArray(rhs[begin:end-m], rhs[end-m+1:end])
    lhs = J0 * rhs
    lhs2 = blsmap(rhs_bd)
    @test lhs[begin:end-m] ≈ lhs2.u
    @test lhs[end-m+1:end] ≈ lhs2.p
    lhs2 = blsmap(rhs)
    @test lhs[begin:end-m] ≈ lhs2[begin:end-m]
    @test lhs[end-m+1:end] ≈ lhs2[end-m+1:end]

    # test MatrixFreeBLS
    for ls in (GMRESKrylovKit(), GMRESIterativeSolvers(reltol = 1e-11, N = size(J0,1)))
        linBdsolver = BK.MatrixFreeBLS(ls)
        sol_bd2u, sol_bd2p, success, it = BK.solve_bls_block(linBdsolver, Jmf, a, b, c, rhs[begin:end-m], rhs[end-m+1:end])
        @assert success
        @test sol_explicit[begin:end-m] ≈ sol_bd2u
        @test sol_explicit[end-m+1:end] ≈ sol_bd2p
    end
end
####################################################################################################
# test the bordered linear solvers, Complex case
# we test the linear system with MA formulation of Hopf bifurcation
let
    J = [0 1 0; -1 0 0; 0 0 1.]
    V, Ve = eigen(J)
    λ = V[2]
    v = Ve[:,2]
    V, Ve = eigen(J')
    λ2 = V[1]
    @assert λ ≈ conj(λ2)
    w = Ve[:,1]
    @test det(J - λ*I(3)) ≈ 0

    λ *= 1.01
    J0 = [J - λ*I(3) w;(v') 0]
    @test ~(det(J0) ≈ 0)
    @test abs(det([[J - λ*I(3) w];[conj(v') 0]]))  < 1e-12

    n = size(J, 1)
    rhs = zeros(ComplexF64, n+1); rhs[end] = 1
    sol_explicit = J0 \ rhs

    # extract the border
    J11 = J0[begin:end-1,1:end-1]
    J12 = J0[begin:end-1,end]
    J21 = J0[end,1:end-1] |> conj
    J22 = J0[end,end]

    @test norm([[J11 J12]; [J21' J22]] - J0) ≈ 0

    J0_b = hcat(J11, J12); J0_b = vcat(J0_b, hcat(J21' , J22))
        @test norm(J0_b - J0) ≈ 0

    args_bls = (J11, J12, J21, J22, rhs[begin:end-1], rhs[end])
    args_bls_shift = (J, J12, J21, J22, rhs[begin:end-1], rhs[end])

    linBdsolver = BK.MatrixBLS()
        sol_bd3u, sol_bd3p, _, _ = @time linBdsolver(args_bls...)
        @test sol_explicit[begin:end-1] ≈ sol_bd3u
        @test sol_explicit[end] ≈ sol_bd3p

    linBdsolver = BK.MatrixBLS()
        sol_bd3u, sol_bd3p, _, _ = @time linBdsolver(args_bls_shift..., shift = -λ)
        @test sol_explicit[begin:end-1] ≈ sol_bd3u
        @test sol_explicit[end] ≈ sol_bd3p

    # without the shift the first block is singular
    linBdsolver = BK.BorderingBLS(solver = DefaultLS(), check_precision=true)
    sol_bd1u, sol_bd1p, _, _ = linBdsolver(args_bls...)
    @test sol_explicit[begin:end-1] ≈ sol_bd1u
    @test sol_explicit[end] ≈ sol_bd1p

    sol_bd1u, sol_bd1p, _, _ = linBdsolver(args_bls_shift..., shift = -λ)
    @test sol_explicit[begin:end-1] ≈ sol_bd1u
    @test sol_explicit[end] ≈ sol_bd1p

    ls = GMRESIterativeSolvers(reltol = 1e-11, N = length(rhs)-1)
    linBdsolver = BK.BorderingBLS(ls)
    sol_bd2u, sol_bd2p, _, _ = linBdsolver(args_bls...)
    @test sol_explicit[begin:end-1] ≈ sol_bd2u
    @test sol_explicit[end] ≈ sol_bd2p

    sol_bd2u, sol_bd2p, _, _ = linBdsolver(args_bls_shift..., shift = -λ)
    @test sol_explicit[begin:end-1] ≈ sol_bd2u
    @test sol_explicit[end] ≈ sol_bd2p

    ls = GMRESKrylovKit(dim = length(rhs) - 1, rtol = 1e-11)
    linBdsolver = BK.BorderingBLS(ls)
    sol_bd2u, sol_bd2p, _, _ = linBdsolver(args_bls...)
    @test sol_explicit[begin:end-1] ≈ sol_bd2u
    @test sol_explicit[end] ≈ sol_bd2p

    sol_bd2u, sol_bd2p, _, _ = linBdsolver(x->J*x, J12, J21, J22, rhs[begin:end-1], rhs[end]; shift = -λ)
    @test sol_explicit[begin:end-1] ≈ sol_bd2u
    @test sol_explicit[end] ≈ sol_bd2p

    BK.MatrixFreeBLS(nothing)
    linBdsolver = BK.MatrixFreeBLS(ls)
    sol_bd3u, sol_bd3p, _, _ = linBdsolver(args_bls...)
    @test sol_explicit[begin:end-1] ≈ sol_bd3u
    @test sol_explicit[end] ≈ sol_bd3p rtol = 1e-6

    sol_bd3u, sol_bd3p, _, _ = linBdsolver(args_bls_shift..., shift = -λ)
    @test sol_explicit[begin:end-1] ≈ sol_bd3u
    @test sol_explicit[end] ≈ sol_bd3p rtol = 1e-6

    linBdsolver = BK.MatrixFreeBLS(GMRESIterativeSolvers(reltol = 1e-9, N = size(J0, 1)))
    sol_bd4u, sol_bd4p, _, _ = linBdsolver(args_bls...)
    @test sol_explicit[begin:end-1] ≈ sol_bd4u
    @test sol_explicit[end] ≈ sol_bd4p

    # test the bordered linear solvers as used in newtonPALC
    ξu = 1.
    ξp = rand()

    linBdsolver = BK.BorderingBLS(DefaultLS())
    sol_bd1u, sol_bd1p, _, _ = linBdsolver(args_bls..., ξu, ξp; shift = 0.2)

    linBdsolver = BK.MatrixFreeBLS(ls)
    sol_bd2u, sol_bd2p, _, _ = linBdsolver(args_bls..., ξu, ξp; shift = 0.2)

    @test sol_bd1u ≈ sol_bd2u
    @test sol_bd1p ≈ sol_bd2p
end
####################################################################################################
# test the bordered linear solvers, Complex case
# random case
let
    n = 5
    J0 = rand(n,n) * 0.9 - I
    J0 = hcat(J0, rand(ComplexF64, n))
    J0 = vcat(J0, rand(ComplexF64, n+1)')
    rhs = rand(ComplexF64, size(J0, 1))
    sol_explicit = J0 \ rhs

    # extract the border
    J11 = real.(J0[begin:end-1,1:end-1])
    J12 = J0[begin:end-1,end]
    J21 = J0[end,1:end-1] |> conj
    J22 = J0[end,end]

    @test norm([[J11 J12]; [J21' J22]] - J0) ≈ 0

    J0_b = hcat(J11, J12); J0_b = vcat(J0_b, hcat(J21' , J22))
        @test norm(J0_b - J0) ≈ 0

    linBdsolver = BK.MatrixBLS()
    sol_bd3u, sol_bd3p, _, _ = @time linBdsolver(J11, J12, J21, J22, rhs[begin:end-1], rhs[end])
    @test sol_explicit[begin:end-1] ≈ sol_bd3u
    @test sol_explicit[end] ≈ sol_bd3p

    linBdsolver = BK.BorderingBLS(solver = DefaultLS(), check_precision=true)
    sol_bd1u, sol_bd1p, _, _ = linBdsolver(J11, J12, J21, J22, rhs[begin:end-1], rhs[end])
    @test sol_explicit[begin:end-1] ≈ sol_bd1u
    @test sol_explicit[end] ≈ sol_bd1p

    ls = GMRESIterativeSolvers(reltol = 1e-11, N = length(rhs)-1)
    linBdsolver = BK.BorderingBLS(ls)
    sol_bd2u, sol_bd2p, _, _ = linBdsolver(J11, J12, J21, J22, rhs[begin:end-1], rhs[end])
    @test sol_explicit[begin:end-1] ≈ sol_bd2u
    @test sol_explicit[end] ≈ sol_bd2p

    ls = GMRESKrylovKit(dim = length(rhs) - 1, rtol = 1e-11)
    linBdsolver = BK.BorderingBLS(ls)
    sol_bd2u, sol_bd2p, _, _ = linBdsolver(J11, J12, J21, J22, rhs[begin:end-1], rhs[end])
    @test sol_explicit[begin:end-1] ≈ sol_bd2u
    @test sol_explicit[end] ≈ sol_bd2p

    BK.MatrixFreeBLS(nothing)
    linBdsolver = BK.MatrixFreeBLS(ls)
    sol_bd3u, sol_bd3p, _, _ = linBdsolver(J11, J12, J21, J22, rhs[begin:end-1], rhs[end])
    @test sol_explicit[begin:end-1] ≈ sol_bd3u
    @test sol_explicit[end] ≈ sol_bd3p rtol = 1e-6

    linBdsolver = BK.MatrixFreeBLS(GMRESIterativeSolvers(reltol = 1e-9, N = size(J0, 1)))
    sol_bd4u, sol_bd4p, _, _ = linBdsolver(J11, J12, J21, J22, rhs[begin:end-1], rhs[end])
    @test sol_explicit[begin:end-1] ≈ sol_bd4u rtol = 1e-6
    @test sol_explicit[end] ≈ sol_bd4p rtol = 1e-6

    # test the bordered linear solvers as used in newtonPALC
    xiu = rand()
    xip = rand()

    linBdsolver = BK.BorderingBLS(DefaultLS())
    sol_bd1u, sol_bd1p, _, _ = linBdsolver(J11, J12, J21, J22, rhs[begin:end-1], rhs[end], xiu, xip)

    linBdsolver = BK.MatrixFreeBLS(ls)
    sol_bd2u, sol_bd2p, _, _ = linBdsolver(J11, J12, J21, J22, rhs[begin:end-1], rhs[end], xiu, xip)

    @test sol_bd1u ≈ sol_bd2u
    @test sol_bd1p ≈ sol_bd2p
end
####################################################################################################
# test the linear solvers for matrix free formulations
let
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
    @test out[1] ≈ outkk[1] rtol = 1e-9
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

    ls = GMRESIterativeSolvers(N = 100, reltol = 1e-9, ismutating = true)
    outit = ls(J0, x0)
    @test out[1] ≈ outit[1]
end
####################################################################################################
# test the shifted linear systems
let
    J0 = I + sprand(100,100,0.1)
    rhs = rand(size(J0, 1))
    sol0 = J0\rhs;

    ls0 = GMRESIterativeSolvers(N = size(J0,1), reltol = 1e-10)
    sol1, _ = ls0(J0, rhs)
    @test norm(sol0 .- sol1, Inf) < 1e-8

    h = 0.81
    sol0 = (I - h.*J0)\rhs
    sol1 = (I/h - J0)\rhs
    @test norm(sol0 - sol1/h, Inf) < 1e-8

    sol0,_ = ls0(I - h*J0, rhs)
    sol1,_ = ls0(J0, rhs; a₀ = 1.0, a₁ = -h)
    @test norm(sol0 - sol1, Inf) < 1e-8

    ls0 = GMRESKrylovKit(atol = 1e-10)
    sol0,_ = ls0(I - h*J0, rhs)
    sol1,_ = ls0(J0, rhs; a₀ = 1.0, a₁ = -h)
    @test norm(sol0 - sol1, Inf) < 1e-8

    sol0,_ = ls0(I - h*J0, rhs)
    sol1,_ = ls0(J0, rhs; a₀ = 1.0/h, a₁ = -1.)
    @test norm(sol0 - sol1/h, Inf) < 1e-8


    sol0,_ = ls0(I - h*J0, rhs)
    sol1,_ = ls0(J0, rhs; a₀ = 1., a₁ = -h)
    @test norm(sol0 - sol1, Inf) < 1e-8
end
####################################################################################################
# test biorthogonalise
let
    vs = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]
    BK.biorthogonalise(vs, deepcopy(vs), true)

    vss = [[0.0, 0.0, 1.0],
           [0.0, 1.0, 0.0],
           [1.0, 0.0, 0.0]]
    BK.biorthogonalise(vs, vss, true)
end
####################################################################################################
# test the eigen solvers for matrix free formulations
# test that the top eigenvalues are sorted by decrasing real part
_test_sorted(x) = issorted(x, by = real, rev = true)

# test ordering of eigenvalues
let 
    J0 = [0.688714  0.363181   0.956579  0.967328  0.950136
            0.860239  0.0481349  0.705687  0.236736  0.921345
            0.740663  0.659207   0.365235  0.123933  0.810514
            0.998672  0.717179   0.609523  0.907024  0.307781
            0.259797  0.0059453  0.105637  0.218516  0.356943]
    _vals, _vecs, cv = DefaultEig()(J0, 5)
    @test all(_vals .≈ [      2.750124876460063 + 0.0im
                         0.2338099902832191 - 0.3203002738693372im
                         0.2338099902832191 + 0.3203002738693372im
                       -0.42584697851325004 - 0.17961985097997188im
                       -0.42584697851325004 + 0.17961985097997188im])
    
    # it must be properly normalized
    @test norminf(_vecs - [-0.582081+0.0im   0.145221-0.0794783im   0.145221+0.0794783im    0.697271-0.0im          0.697271+0.0im
                            -0.37944+0.0im  -0.347233+0.0865446im  -0.347233-0.0865446im   -0.183677+0.381804im    -0.183677-0.381804im
                           -0.363176+0.0im  -0.620374-0.0im        -0.620374+0.0im         -0.434315-0.286744im    -0.434315+0.286744im
                            -0.60577+0.0im   0.474679-0.336788im    0.474679+0.336788im    -0.211553-0.0631766im   -0.211553+0.0631766im
                           -0.135474+0.0im   0.177718+0.298898im    0.177718-0.298898im   -0.0950868+0.0752506im  -0.0950868-0.0752506im]) < 1e-6
end

let
    x0 = rand(100)
    J0 = I + sprand(100,100,0.1)
    Jmf = x -> J0 * x

    @test _test_sorted(DefaultEig()(J0, 10)[1])

    B = sparse(I(length(x0)))
    out = Arpack.eigs(J0, nev = 20, which = :LR)
    resgev = BK.gev(DefaultEig(), J0, B, 10)
    @test issorted(resgev[1], by = real)

    eil = BK.EigKrylovKit(tol = 1e-9)
    outkk = eil(J0, 20)
    @test _test_sorted(outkk[1])
    geteigenvector(eil, outkk[2], 2)

    eil = BK.EigKrylovKit(tol = 1e-9, x₀ = x0)
    outkkmf = eil(Jmf, 20)
    @test _test_sorted(outkkmf[1])
    geteigenvector(eil, outkkmf[2], 2)

    eil = BK.EigArpack(v0 = copy(x0))
    outdefault = eil(J0, 20)
    @test _test_sorted(outdefault[1])
    @test out[1] ≈ outdefault[1]
    outdefault = eil(x ->J0*x, 20)
    @test _test_sorted(outdefault[1])
    @test out[1] ≈ outdefault[1]
    resgev = BK.gev(eil, J0, B, 10)
    @test issorted(resgev[1], by = real, rev = true)

    eil = BK.EigArnoldiMethod(;x₀ = x0)
    outam = eil(J0, 20)
    @test _test_sorted(outam[1])
    outam = eil(Jmf, 20)
    @test _test_sorted(outam[1])
    geteigenvector(eil, outam[2], 2)

    eil = BK.EigArnoldiMethod(;x₀ = x0, sigma = 1.)
    outam = eil(J0, 20)
    @test _test_sorted(outam[1])
    outam = eil(Jmf, 20)
    @test _test_sorted(outam[1])
    geteigenvector(eil, outam[2], 2)
    resgev = BK.gev(eil, J0, B, 10)
    # @test issorted(resgev[1], by = real, rev = true)
end
####################################################################################################
# test Shift-Invert
eig = BifurcationKit.ShiftInvert(0.1, DefaultLS(), EigArnoldiMethod(x₀=rand(10)))
J = I + 0.1rand(10,10)
res1 = eig(J, 10)
res1[1]
@test norminf(eigvals(J, sortby = real) - res1[1])<1e-9