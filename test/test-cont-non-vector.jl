# this test is designed to test the ability of the package to use a state space that is not an AbstractArray.
# using Revise
using Test, Random, Setfield
using BifurcationKit
const BK = BifurcationKit
####################################################################################################
# test Bordered Arrays
_a = BorderedArray(zeros(2), zeros(2))
_b = BorderedArray(zeros(2), zeros(2))
BK.mul!(_a, 1., _b)
BK.axpby!(1., _a, 1., _b)
BK.getVec(_a)
BK.getVec(_a.u)
BK.getVec(BorderedArray(rand(2),1.))
BK.getP(BorderedArray(rand(2),1.))
####################################################################################################
# We start with a simple Fold problem
using LinearAlgebra
function F0(x::Vector, r)
    out = r .+  x .- x.^3
end

opt_newton0 = NewtonPar(tol = 1e-11, verbose = false)
    prob = BK.BifurcationProblem(F0, [0.8], 1., (@lens _);
            recordFromSolution = (x, p) -> x[1],
            J = (x, r) -> diagm(0 => 1 .- 3 .* x.^2),
            Jᵗ = (x, r) -> diagm(0 => 1 .- 3 .* x.^2),
            d2F = (x, r, v1, v2) -> -6 .* x .* v1 .* v2,)
    sol0 = newton(prob, opt_newton0)

opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.07, ds= -0.02, pMax = 4.1, pMin = -1., newtonOptions = setproperties(opt_newton0; maxIter = 70, tol = 1e-8), detectBifurcation = 0, maxSteps = 150)

BK.isStable(opts_br0, nothing)

br0 = continuation(prob, PALC(), opts_br0)
@test br0.param[end] == -1

solfold = newton(br0, 2)
    # flag && printstyled(color=:red, "--> We found a Fold Point at α = ",outfold.p, ", from ", br0.specialpoint[2].param, "\n")

@test BK.converged(solfold)
####################################################################################################
# Here is a more involved example

function Fb(x::BorderedArray, p)
    r, s = p
    BorderedArray(r .+  s .* x.u .- (x.u).^3, x.p - 0.0)
end

# there is no finite differences defined, so we need to provide a linearsolve
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

sol0 = BorderedArray([0.8], 0.0)

opt_newton = NewtonPar(tol = 1e-11, verbose = false, linsolver = linsolveBd())
prob = BK.BifurcationProblem(Fb, sol0, (1., 1.), (@lens _[1]); J = (x, r) -> Jacobian(x, r[1], r[2]), recordFromSolution = (x,p) -> x.u[1])
sol = newton(prob, opt_newton)
@test BK.converged(sol)

opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= -0.01, pMax = 4.1, pMin = -1., newtonOptions = setproperties(opt_newton; maxIter = 70, tol = 1e-8), detectBifurcation = 0, maxSteps = 150, saveSolEveryStep = 1)

    br = continuation(prob, PALC(), opts_br; linearAlgo = BorderingBLS(opt_newton.linsolver))

BK.getSolx(br, 1)
BK.getSolp(br, 1)
@test br.param[end] == -1

# plot(br);title!("")

prob2 = BK.BifurcationProblem(Fb, sol0, (1., 1.), (@lens _[1]);
    J = (x, r) -> Jacobian(x, r[1], r[2]),
    Jᵗ = (x, r) -> Jacobian(x, r[1], r[2]),
    d2F = (x, r, v1, v2) -> BorderedArray(-6 .* x.u .* v1.u .* v2.u, 0.),)

br = continuation(prob2, PALC(), opts_br; linearAlgo = BorderingBLS(opt_newton.linsolver))
@test br.param[end] == -1

solfold = newton(br, 1; bdlinsolver = BorderingBLS(opt_newton.linsolver))
@test BK.converged(solfold)

outfoldco = continuation(br, 1, (@lens _[2]), opts_br; bdlinsolver = BorderingBLS(opt_newton.linsolver), jacobian_ma = :minaug)

# try with newtonDeflation
# test with Newton deflation 1
deflationOp = DeflationOperator(2, 1.0, [zero(sol.u)])
soldef0 = BorderedArray([0.1], 0.0)
soldef1 = newton(BK.reMake(prob, u0 = soldef0), deflationOp, opt_newton)

push!(deflationOp, soldef1.u)

Random.seed!(1231)
# test with Newton deflation 2
soldef2 = newton(BK.reMake(prob, u0 = rmul!(soldef0,rand())), deflationOp, opt_newton)

deflationOp[1]
length(deflationOp)
pop!(deflationOp)
empty!(deflationOp)

####################################################################################################
using KrylovKit, Parameters

function Fr(x, p)
    @unpack r, s = p
    out = similar(x)
    for ii=1:length(x)
        out[ii] .= @. r +  s * x[ii] - x[ii]^3
    end
    out
end

# there is no finite differences defined, so we need to provide a linearsolve
# we could also have used GMRES. We define a custom Jacobian which will be used for evaluation and jacobian inverse
struct JacobianR
    x
    s
end

# We express the jacobian operator
function (J::JacobianR)(dx)
    out = similar(dx)
    for ii=1:length(out)
        out[ii] .= (J.s .- 3 .* (J.x[ii]).^2) .* dx[ii]
    end
    return out
end

struct linsolveBd_r <: BK.AbstractBorderedLinearSolver end

function (l::linsolveBd_r)(J, dx)
    x = J.x
    out = similar(dx)
    for ii=1:length(out)
        out[ii] .= dx[ii] ./ (J.s .- 3 .* (x[ii]).^2)
    end
    out, true, 1
end

opt_newton0 = NewtonPar(tol = 1e-10, maxIter = 5, verbose = false, linsolver = linsolveBd_r())

prob = BK.BifurcationProblem(Fr,
        RecursiveVec([1 .+ 0.1*rand(1) for _ = 1:2]),
        (r = 1.0, s = 1.), (@lens _.r);
        delta = 1e-8,
        J  = (x, p) -> JacobianR(x, p.s),
        Jᵗ = (x, p) -> JacobianR(x, p.s),
        d2F = (x, r, v1, v2) -> RecursiveVec([-6 .* x[ii] .* v1[ii] .* v2[ii] for ii=1:length(x)]))

sol = newton(prob, opt_newton0)

Base.:copyto!(dest::RecursiveVec, in::RecursiveVec) = copy!(dest, in)

opts_br0 = ContinuationPar(dsmin = 0.00001, dsmax = 0.1, ds= -0.01, pMin = -1., pMax = 1.1, newtonOptions = opt_newton0, maxSteps = 200, detectBifurcation = 0)

br0 = continuation(BK.reMake(prob, u0 = sol.u),
            PALC(tangent=Secant()),
            opts_br0;
            linearAlgo = BorderingBLS(opt_newton0.linsolver)
            )

@test br0.param[end] == -1

br0 = continuation(BK.reMake(prob, u0 = sol.u),
    PALC(tangent=Secant()), opts_br0;
    linearAlgo = BorderingBLS(opt_newton0.linsolver),)

br0 = continuation(prob, PALC(tangent = Bordered()), opts_br0;
    linearAlgo = BorderingBLS(opt_newton0.linsolver))

outfold = newton(br0, 1; bdlinsolver = BorderingBLS(opt_newton0.linsolver))
@test BK.converged(outfold)


outfoldco = continuation(br0, 1, (@lens _.s), opts_br0,
    bdlinsolver = BorderingBLS(opt_newton0.linsolver), jacobian_ma = :minaug)

br0sec = @set br0.alg.tangent = Secant()
outfoldco = continuation(br0sec, 1, (@lens _.s), opts_br0,
    bdlinsolver = BorderingBLS(opt_newton0.linsolver),jacobian_ma = :minaug)

# try with newtonDeflation
# test with Newton deflation 1
deflationOp = DeflationOperator(2, 1.0, [sol.u])
soldef1 = newton(BK.reMake(prob, u0 = 0.1*(sol.u), params = (r=0., s=1.)),
    deflationOp, (@set opt_newton0.maxIter = 20))

push!(deflationOp, soldef1.u)
