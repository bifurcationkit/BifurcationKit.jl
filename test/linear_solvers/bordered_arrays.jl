import BifurcationKit as BK
import LinearAlgebra as LA
import BifurcationKit.BorderedArray as BorderedArray
import Random

deepcollect(x::BK.BorderedArray) = vcat(x.u, x.p)
deepcollect(x::Number) = x
Random.seed!(1234)
let
    LA.norm(BorderedArray(rand(2), 0), 0)
    LA.norm(BorderedArray(rand(2), 0), 1)
    LA.norm(BorderedArray(rand(2), 0), 2)
    LA.norm(BorderedArray(rand(2), 0), Inf)
    LA.norm(BorderedArray(rand(2), 0), -Inf)
end

for (x,y) in [(BK.BorderedArray([1., 2.], 3.),   BK.BorderedArray([4., 5.],  6.)),
              (BK.BorderedArray([1., 2.], [3.]), BK.BorderedArray([4., 5.], [6.]))]

    # @error "" x y

    @testset "scalartype" begin
        @test Float64 == BK.VI.scalartype(x)
    end

    @testset "zerovector" begin
        z = BK.VI.zerovector(x)
        @test all(iszero, deepcollect(z))
        @test all(deepcollect(z) .=== zero(BK.VI.scalartype(x)))

        z1 = BK.VI.zerovector!!(deepcopy(x))
        @test all(deepcollect(z1) .=== zero(BK.VI.scalartype(x)))
        z2 = BK.VI.zerovector!(deepcopy(x))
        @test all(deepcollect(z2) .=== zero(BK.VI.scalartype(x)))
    end

    @testset "BK.VI.scale" begin
        α = randn()
        z = BK.VI.scale(x, α)
        @test all(deepcollect(z) .== α .* deepcollect(x))

        z2 = BK.VI.scale!!(deepcopy(x), α)
        @test deepcollect(z2) ≈ (α .* deepcollect(x))
        xcopy = deepcopy(x)
        z2 = BK.VI.scale!!(deepcopy(y), xcopy, α)
        @test deepcollect(z2) ≈ (α .* deepcollect(x))
        @test all(deepcollect(xcopy) .== deepcollect(x))

        z3 = BK.VI.scale!(deepcopy(x), α)
        @test deepcollect(z3) ≈ (α .* deepcollect(x))
        xcopy = deepcopy(x)
        z3 = BK.VI.scale!(BK.VI.zerovector(x), xcopy, α)
        @test deepcollect(z3) ≈ (α .* deepcollect(x))
        @test all(deepcollect(xcopy) .== deepcollect(x))

        α = randn(ComplexF64)
        z4 = BK.VI.scale(x, α)
        @test deepcollect(z4) ≈ (α .* deepcollect(x))
        xcopy = deepcopy(x)
        z5 = BK.VI.scale!!(xcopy, α)
        @test deepcollect(z5) ≈ (α .* deepcollect(x))
        @test all(deepcollect(xcopy) .== deepcollect(x))
        @test_throws InexactError BK.VI.scale!(xcopy, α)

        α = randn(ComplexF64)
        xcopy = deepcopy(x)
        z6 = BK.VI.scale!!(BK.VI.zerovector(x), xcopy, α)
        @test deepcollect(z6) ≈ (α .* deepcollect(x))
        @test all(deepcollect(xcopy) .== deepcollect(x))
        @test_throws InexactError BK.VI.scale!(BK.VI.zerovector(x), xcopy, α)

        xz = BK.VI.zerovector(x, ComplexF64)
        z6 = BK.VI.scale!!(xz, xcopy, α)
        @test deepcollect(z6) ≈ (α .* deepcollect(x))
        @test all(deepcollect(xcopy) .== deepcollect(x))

        z7 = BK.VI.scale!(xz, xcopy, α)
        @test deepcollect(z7) ≈ (α .* deepcollect(x))
        @test all(deepcollect(xcopy) .== deepcollect(x))

        ycomplex = BK.VI.zerovector(y, ComplexF64)
        α = randn(Float64)
        xcopy = deepcopy(x)
        z8 = BK.VI.scale!!(ycomplex, xcopy, α)
        @test z8 === ycomplex
        @test all(deepcollect(z8) .== α .* deepcollect(xcopy))
    end

    @testset "inner" begin
        s = BK.VI.inner(x, y)
        @test s ≈ BK.VI.inner(deepcollect(x), deepcollect(y))

        α, β = randn(ComplexF64, 2)
        s2 = BK.VI.inner(BK.VI.scale(x, α), BK.VI.scale(y, β))
        @test s2 ≈ BK.VI.inner(α * deepcollect(x), β * deepcollect(y))
    end

    @testset "add" begin
        α, β = randn(2)
        z = BK.VI.add(y, x)
        @test all(deepcollect(z) .== deepcollect(x) .+ deepcollect(y))
        z = BK.VI.add(y, x, α)
        # for some reason, on some Julia versions on some platforms, but only in test mode
        # there is a small floating point discrepancy, which makes the following test fail:
        # @test all(deepcollect(z) .== muladd.(deepcollect(x), α, deepcollect(y)))
        @test deepcollect(z) ≈ muladd.(deepcollect(x), α, deepcollect(y))
        z = BK.VI.add(y, x, α, β)
        # for some reason, on some Julia versions on some platforms, but only in test mode
        # there is a small floating point discrepancy, which makes the following test fail:
        # @test all(deepcollect(z) .== muladd.(deepcollect(x), α, deepcollect(y) .* β))
        @test deepcollect(z) ≈ muladd.(deepcollect(x), α, deepcollect(y) .* β)

        α, β = randn(2)
        xcopy = deepcopy(x)
        z2 = BK.VI.add!!(deepcopy(y), xcopy)
        @test deepcollect(z2) ≈ (deepcollect(x) .+ deepcollect(y))
        @test all(deepcollect(xcopy) .== deepcollect(x))
        z2 = BK.VI.add!!(deepcopy(y), xcopy, α)
        @test deepcollect(z2) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
        @test all(deepcollect(xcopy) .== deepcollect(x))
        z2 = BK.VI.add!!(deepcopy(y), xcopy, α, β)
        @test deepcollect(z2) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))
        @test all(deepcollect(xcopy) .== deepcollect(x))

        α, β = randn(2)
        z3 = BK.VI.add!(deepcopy(y), xcopy)
        @test deepcollect(z3) ≈ (deepcollect(y) .+ deepcollect(x))
        @test all(deepcollect(xcopy) .== deepcollect(x))
        z3 = BK.VI.add!(deepcopy(y), xcopy, α)
        @test all(deepcollect(xcopy) .== deepcollect(x))
        @test deepcollect(z3) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
        z3 = BK.VI.add!(deepcopy(y), xcopy, α, β)
        @test deepcollect(z3) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))
        @test all(deepcollect(xcopy) .== deepcollect(x))

        α, β = randn(ComplexF64, 2)
        z4 = BK.VI.add(y, x, α)
        @test deepcollect(z4) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
        z4 = BK.VI.add(y, x, α, β)
        @test deepcollect(z4) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))

        α, β = randn(ComplexF64, 2)
        z5 = BK.VI.add!!(deepcopy(y), xcopy, α)
        @test deepcollect(z5) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
        @test all(deepcollect(xcopy) .== deepcollect(x))
        z5 = BK.VI.add!!(deepcopy(y), xcopy, α, β)
        @test deepcollect(z5) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))
        @test all(deepcollect(xcopy) .== deepcollect(x))

        α, β = randn(ComplexF64, 2)
        z5 = BK.VI.add!!(deepcopy(y), xcopy, α)
        @test deepcollect(z5) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
        @test all(deepcollect(xcopy) .== deepcollect(x))
        z5 = BK.VI.add!!(deepcopy(y), xcopy, α, β)
        @test deepcollect(z5) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))
        @test all(deepcollect(xcopy) .== deepcollect(x))

        α, β = randn(ComplexF64, 2)
        @test_throws InexactError BK.VI.add!(deepcopy(y), xcopy, α)
        @test_throws InexactError BK.VI.add!(deepcopy(y), xcopy, α, β)
    end
end
