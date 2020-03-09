using Test
using KrylovKit, SparseArrays, ArnoldiMethod, LinearMaps, LinearAlgebra
using PseudoArcLengthContinuation

N = 110
A = I + sprand(N, N, 0.05)

T, vecs, vals, info = @time KrylovKit.schursolve(A, rand(N), 10, :LM, Arnoldi(krylovdim = 30, verbosity = 2))
vals = eigvals(Array(A))

P = PrecPartialSchurKrylovKit(A, rand(N), 4, :LM)
P = PrecPartialSchurArnoldiMethod(A, N, 4, LM())

Jmap = LinearMap(x-> A * (P \ x), N, N ; ismutating = false)

decomp, history = ArnoldiMethod.partialschur(Jmap, nev = 110, tol=1e-9, which=LM());

@test Base.setdiff(round.(vals, digits = 5), round.(decomp.eigenvalues, digits = 5)) |> length in [4,5]

#
# using Plots
# scatter(real.(vals), imag.(vals), label = "")
# 	scatter!(real.(decomp.eigenvalues), imag.(decomp.eigenvalues), label = "", marker=:cross)
