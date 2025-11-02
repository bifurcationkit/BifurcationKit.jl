# this test is designed to test the ability of the package to use a state space that is not an AbstractArray.
# using Revise
using Test, Random
using BifurcationKit
const BK = BifurcationKit
####################################################################################################
# test Bordered Arrays
let
    _a = BorderedArray(zeros(2), zeros(2))
    _b = BorderedArray(zeros(2), zeros(2))
    BK.VI.scale!(_a, _b, 1.)
    BK.VI.add!(_a, _b, 1., 1.)
    BK.getvec(_a)
    BK.getvec(_a.u)
    BK.getvec(BorderedArray(rand(2),1.))
    BK.getp(BorderedArray(rand(2),1.))
end
####################################################################################################
# We start with a simple Fold problem
using LinearAlgebra

function F0(x::Vector, r)
    out = r .+  x .- x.^3
end

let
    opt_newton0 = NewtonPar(tol = 1e-11)
    prob = BK.BifurcationProblem(F0, [0.8], 1., (@optic _);
            record_from_solution = (x, p; k...) -> x[1],
            J = (x, r) -> diagm(0 => 1 .- 3 .* x.^2),
            Jᵗ = (x, r) -> diagm(0 => 1 .- 3 .* x.^2),
            d2F = (x, r, v1, v2) -> -6 .* x .* v1 .* v2,)
    sol0 = BK.solve(prob, Newton(), opt_newton0)

    opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.07, ds= -0.02, p_max = 4.1, p_min = -1., newton_options = NewtonPar(tol = 1e-8), detect_bifurcation = 0, max_steps = 150)

    BK.is_stable(opts_br0, nothing)

    br0 = continuation(prob, PALC(), opts_br0)

    @test br0.param[end] == -1

    solfold = newton(br0, 2)
    @test BK.converged(solfold)
end
####################################################################################################
# Here is a more involved example
function Fb(x::BorderedArray, p)
    r, s = p
    BorderedArray(r .+  s .* x.u .- (x.u).^3, x.p - 0.0)
end

# there is no finite differences defined, so we need to provide a linear solver
# we could also have used GMRES. We define a custom Jacobian which will be used for evaluation and jacobian inverse
struct Jacobian
    x
    r
    s
end

# We express the jacobian operator
function (J::Jacobian)(dx)
    BorderedArray( (J.s .- 3 .* ((J.x).u).^2) .* dx.u, dx.p)
end

struct linsolveBd <: BK.AbstractBorderedLinearSolver end

function (l::linsolveBd)(J, dx)
    x = J.x
    r = J.r
    out = BorderedArray(dx.u ./ (J.s .- 3 .* (x.u).^2), dx.p)
    out, true, 1
end

let
    sol0 = BorderedArray([0.8], 0.0)

    opt_newton = NewtonPar(tol = 1e-9, verbose = false, linsolver = linsolveBd())
    prob = BK.BifurcationProblem(Fb, sol0, (1., 1.), (@optic _[1]); J = (x, r) -> Jacobian(x, r[1], r[2]), record_from_solution = (x,p;k...) -> x.u[1])
    sol = BK.solve(prob, Newton(), opt_newton)
    @test BK.converged(sol)

    opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= -0.01, p_max = 4.1, p_min = -1., newton_options = setproperties(opt_newton; max_iterations = 10), detect_bifurcation = 0, max_steps = 50, save_sol_every_step = 1)

    br = continuation(prob, PALC(), opts_br; linear_algo = BorderingBLS(opt_newton.linsolver))

    BK.get_solx(br, 1)
    BK.get_solp(br, 1)

    # plot(br)

    prob2 = BK.BifurcationProblem(Fb, sol0, (1., 1.), (@optic _[1]);
                J  = (x, r) -> Jacobian(x, r[1], r[2]),
                Jᵗ = (x, r) -> Jacobian(x, r[1], r[2]),
                d2F = (x, r, v1, v2) -> BorderedArray(-6 .* x.u .* v1.u .* v2.u, 0.),)

    br = continuation(prob2, PALC(), opts_br; linear_algo = BorderingBLS(opt_newton.linsolver))

    solfold = newton(br, 1; bdlinsolver = BorderingBLS(solver = opt_newton.linsolver, dot = BK.VI.inner))
    @test BK.converged(solfold)

    try
        outfoldco = continuation(br, 1, (@optic _[2]), ContinuationPar(opts_br, max_steps = 4); 
                        # verbosity = 2,
                        start_with_eigen = false,
                        bdlinsolver = BorderingBLS(opt_newton.linsolver), 
                        jacobian_ma = BK.MinAug())
    catch

    end

    # try with newtonDeflation
    # test with Newton deflation 1
    deflationOp = DeflationOperator(2, 1.0, [BK.VI.zerovector(sol.u)])
    soldef0 = BorderedArray([0.1], 0.0)
    soldef1 = BK.solve(BK.re_make(prob, u0 = soldef0), deflationOp, opt_newton)

    push!(deflationOp, soldef1.u)

    Random.seed!(1231)
    # test with Newton deflation 2
    soldef2 = BK.solve(BK.re_make(prob, u0 = BK.VI.scale(soldef0, rand())), deflationOp, opt_newton)
end
####################################################################################################
# Here is a more involved example
function Fb2(x::BorderedArray, p)
    r, s = p
    BorderedArray(BK.VI.MinimalMVec(r .+  s .* x.u.vec .- (x.u.vec).^3), x.p - 0.0)
end

# there is no finite differences defined, so we need to provide a linear solver
# we could also have used GMRES. We define a custom Jacobian which will be used for evaluation and jacobian inverse
struct Jacobian2
    x
    r
    s
end

# We express the jacobian operator
function (J::Jacobian2)(dx)
    BorderedArray( BK.VI.MinimalMVec((J.s .- 3 .* ((J.x).u.vec).^2) .* dx.u.vec), dx.p)
end

struct linsolveBd2 <: BK.AbstractBorderedLinearSolver end

function (l::linsolveBd2)(J, rhs)
    x = J.x
    r = J.r
    out = BorderedArray(BK.VI.MinimalMVec(rhs.u.vec ./ (J.s .- 3 .* (x.u.vec).^2)), rhs.p)
    out, true, 1
end

let
    sol0 = BorderedArray(BK.VI.MinimalMVec([0.8]), 0.0)
    opt_newton = NewtonPar(tol = 1e-9, verbose = false, linsolver = linsolveBd2())
    prob = BK.BifurcationProblem(Fb2, sol0, (1., 1.), (@optic _[1]); J = (x, r) -> Jacobian2(x, r[1], r[2]), record_from_solution = (x,p;k...) -> x.u.vec[1])
    sol = BK.solve(prob, Newton(), opt_newton)
    @test BK.converged(sol)

    opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= -0.01, p_max = 4.1, p_min = -1., newton_options = setproperties(opt_newton; max_iterations = 10), detect_bifurcation = 0, max_steps = 50, save_sol_every_step = 1)

    br = continuation(prob, PALC(), opts_br; linear_algo = BorderingBLS(opt_newton.linsolver))

    BK.get_solx(br, 1)
    BK.get_solp(br, 1)

    # plot(br)

    prob2 = BK.BifurcationProblem(Fb2, sol0, (1., 1.), (@optic _[1]);
                J  = (x, r) -> Jacobian2(x, r[1], r[2]),
                Jᵗ = (x, r) -> Jacobian2(x, r[1], r[2]),
                d2F = (x, r, v1, v2) -> BorderedArray(BK.VI.MinimalMVec(-6 .* x.u.vec .* v1.u.vec .* v2.u.vec), 0.),)

    br = continuation(prob2, PALC(), opts_br; linear_algo = BorderingBLS(opt_newton.linsolver))

    solfold = newton(br, 1; bdlinsolver = BorderingBLS(solver = opt_newton.linsolver, dot = BK.VI.inner))
    @test BK.converged(solfold)

    try
        outfoldco = continuation(br, 1, (@optic _[2]), ContinuationPar(opts_br, max_steps = 4); 
                        # verbosity = 2,
                        start_with_eigen = false,
                        bdlinsolver = BorderingBLS(opt_newton.linsolver), 
                        jacobian_ma = BK.MinAug())
    catch

    end

    # try with newtonDeflation
    # test with Newton deflation 1
    deflationOp = DeflationOperator(2, 1.0, [BK.VI.zerovector(sol.u)])
    soldef0 = BorderedArray(BK.VI.MinimalMVec([0.1]), 0.0)
    soldef1 = BK.solve(BK.re_make(prob, u0 = soldef0), deflationOp, opt_newton)

    push!(deflationOp, soldef1.u)

    Random.seed!(1231)
    # test with Newton deflation 2
    soldef2 = BK.solve(BK.re_make(prob, u0 = BK.VI.scale(soldef0, rand())), deflationOp, opt_newton)
end