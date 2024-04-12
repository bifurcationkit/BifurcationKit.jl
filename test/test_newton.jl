# using Revise
using Test, BifurcationKit, LinearAlgebra
const BK = BifurcationKit

function test_newton(x0)
    Ty = eltype(x0)
    F(x, p) = @. x^3 - 1
    J0 = zeros(length(x0), length(x0))
    function Jac(x, p)
        for i in eachindex(x)
            J0[i,i] = 3 * x[i]^2
        end
        J0
    end

    opts = NewtonPar( tol = Ty(1e-8), verbose = false)
    prob = BifurcationProblem(F, x0, nothing; J = Jac)
    newton(prob, opts; callback = BK.cbMaxNorm(100.0), normN = x->norm(x,Inf))
end
####################################################################################################
# we test the regular newton algorithm
# simple case
test_newton(ones(10) .+ rand(10) * 0.1)

# test types for newton
# test type
for T in (Float64, Float32, Float16)
    sol = test_newton(T.(ones(10) .+ rand(10) * 0.1)).u
    @test eltype(sol) == T
end
####################################################################################################
function test_newton_palc(x0::Vector{T}, p0::T) where T
    @assert eltype(x0) == T
    N = length(x0)

    θ = T(0.5)
    dotθ = BK.DotTheta(dot)

    F(x, p) = @. x^3 - 13 * x - p
    Jac(x, p) = diagm(0 => 3 .* x.^2 .- 13)

    z0 = BorderedArray(x0, p0)
    τ0 = BorderedArray(rand(T, N), convert(typeof(p0), 0.2))
    zpred = BorderedArray(x0, convert(typeof(p0), 0.3))
    optn = NewtonPar{T, DefaultLS, DefaultEig}(verbose = false, tol = T(1e-6))
    optc = ContinuationPar{T, DefaultLS, DefaultEig}(newton_options = optn, ds = 0.001,)

    prob = BifurcationProblem(F, x0, p0, (@lens _); J = Jac)
    iter = ContIterable(prob, PALC{Secant, MatrixBLS{Nothing}, T, BK.DotTheta}(θ = θ), optc)
    state = iterate(iter)[1]
    BK.newton_palc(iter, state)
end

test_newton_palc(-ones(10) .* 0.04, 0.5)

# test type
for T in (Float64, Float32, Float16)
    local sol = test_newton_palc(T.(-ones(10) .*0.04), T(0.5)).u
    @test typeof(sol) == BorderedArray{Vector{T}, T}
end
####################################################################################################
# test of deflated factor
using ForwardDiff
F4def(x, p) = @. (x-1) * (x-2)
J4def(x, p) = ForwardDiff.jacobian(z -> F4def(z, p), x)

for _T in (Float64, Float32, Float16)
    deflationOp = DeflationOperator(2, dot, _T(1), [[_T(1)]])
    @test firstindex(deflationOp) == 1
    @test lastindex(deflationOp) == 1
    @test eltype(deflationOp) == _T
    @test deflationOp(rand(_T, 1)) isa _T
end

# the test the value of the deflation factor
_T = Float64
deflationOp = DeflationOperator(2, dot, _T(1), [rand(2) for _ in 1:3])
let
    _x0 = rand(2)
    _res = 1
    for r in deflationOp.roots
        _res *= deflationOp.α + norm(_x0 - r)^(-2deflationOp.power)
    end
    @test _res ≈ deflationOp(_x0)[1] rtol = 1e-8
end

# test of deflated problem
deflationOp = DeflationOperator(2, dot, _T(1), [[_T(1)]])
prob = BifurcationProblem(F4def, [0.1], nothing; J = J4def)
defpb = DeflatedProblem(prob, deflationOp, nothing)
show(defpb)
BK.isinplace(defpb)
BK._getvectortype(defpb)
BK.is_symmetric(defpb)
BK.getlens(defpb)
BK.getparam(defpb)
BK.setparam(defpb, 0.)

@test defpb(rand(_T, 1), nothing) |> eltype == _T
@test defpb(rand(_T, 1), nothing, rand(_T, 1)) |> eltype == _T

push!(deflationOp, rand(_T,1))
@test deflationOp(zeros(_T, 1)) isa _T
@test deflationOp(rand(_T, 1), rand(_T, 1)) isa _T
copy(deflationOp)

# test of custom distance
deflationOp2 = DeflationOperator(_T(2), BifurcationKit.CustomDist((u,v)->norm(u-v)), _T(1), deflationOp.roots)
@test deflationOp2(zeros(_T, 1)) isa _T
length(deflationOp2)
deflationOp2(rand(_T, 1), rand(_T, 1))

# test of the jacobians
let
    n = 3
    deflationOp = DeflationOperator(2, dot, _T(1), [1 .+ 0.01rand(n) for _ in 1:3]; autodiff = true)
    rhs = rand(n)
    sol = rand(n)
    solverdf = BK.DeflatedProblemCustomLS(DefaultLS())
    probdf = DeflatedProblem(prob, deflationOp, Val(:Custom))
    Jdf = BK.jacobian((@set probdf.jactype = BK.AutoDiff()), sol, nothing)
    @test Jdf ≈ ForwardDiff.jacobian(z->probdf(z,nothing),sol)
    Jdf = BK.jacobian(probdf, sol, nothing)
    # test the value of the jacobian
    outj = probdf(sol, nothing, rhs)
    outfd = ForwardDiff.derivative(t->probdf(sol .+ t .* rhs, nothing), 0)
    @test outj ≈ outfd

    sol0, = solverdf(Jdf, rhs)
    Jfd = ForwardDiff.jacobian(z->probdf(z, nothing), sol)
    solfd = Jfd \ rhs
    @test sol0 ≈ solfd
end

# test of the different newton solvers
deflationOp = DeflationOperator(2, dot, _T(1), [[_T(1)]])

sol = newton(prob, deflationOp, NewtonPar())
@test BK.converged(sol)
sol = newton(prob, deflationOp, NewtonPar(), Val(:autodiff))
@test BK.converged(sol)
sol = newton(prob, deflationOp, NewtonPar(linsolver = GMRESKrylovKit()),)
@test BK.converged(sol)
sol = newton(prob, deflationOp, NewtonPar(linsolver = GMRESKrylovKit()), Val(:fullIterative))
@test BK.converged(sol)
####################################################################################################
# test newton adapted to branch switching
F4def(x, p) = @. (x-1) * (x-2)
prob = BifurcationProblem(F4def, [0.1], nothing)
sol1, sol0, flag = newton(prob, [1.2], [2.1], nothing, NewtonPar())
@test flag
@test BK.converged(sol0)
@test BK.converged(sol1)
@test sol0.u ≈ [1.]
@test sol1.u ≈ [2.]
