using Test, BifurcationKit, LinearAlgebra, SparseArrays, FFTW
const BK = BifurcationKit

# loading FFTW must trigger the extension which provides the `ldiv!` methods
@test Base.get_extension(BifurcationKit, :FFTExt) !== nothing

let
    n = 4
    M = 7
    par = nothing
    sol0 = rand(2n)

    # linear vector field: the cyclic blocks are then exactly M̄ / H̄ up to the time steps
    prob = BK.BifurcationProblem((x, p) -> 0.3 .* x, sol0, par; J = (x, p) -> 0.3 * sparse(I, 2n, 2n))

    ϕ  = rand(2n * M)
    xπ = rand(2n * M)
    pb = Trapeze(prob, ϕ, xπ, M)

    u0 = vcat(rand(2n * M), 1.2)
    P  = BK.POTrapCirculantPrec(pb, u0, par)

    N, m = P.N, P.m

    # explicit block-circulant operator: (C x)_i = M x_i - H x_{i-1}, with x_0 = x_m
    S = spzeros(m, m) # (S x)_i = x_{i-1}
    for i in 1:m
        S[i, mod1(i - 1, m)] = 1
    end
    C = kron(sparse(I, m, m), P.M) - kron(S, P.H)

    x = rand(N * m)
    y = P \ x
    @test norm(C * y - x) ≤ 1e-8 * norm(x)

    # inplace version on a larger (bordered) vector: the tail must be left untouched
    xb = vcat(x, rand(2))
    yb = similar(xb)
    ldiv!(yb, P, xb)
    @test norm(C * yb[1:N*m] - x) ≤ 1e-8 * norm(x)
    @test yb[N*m+1:end] == xb[N*m+1:end]

    # inplace version without extra components
    xc = copy(x)
    ldiv!(P, xc)
    @test norm(C * xc - x) ≤ 1e-8 * norm(x)

    # the preconditioner can be refreshed
    BK.update_preconditioner!(P, pb, u0, par)
    @test norm(C * (P \ x) - x) ≤ 1e-8 * norm(x)
end
