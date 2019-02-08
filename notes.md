#PseudoArcLengthContinuation.jl

## Ideas for general interface


[SparseMatrixAssemblers.jl](https://github.com/ettersi/SparseMatrixAssemblers.jl)

[BlockSparseMatrices.jl](https://github.com/KristofferC/BlockSparseMatrices.jl)

[LazyArrays.jl](https://github.com/JuliaArrays/LazyArrays.jl)

Impressive speedups by [MUMPS](https://github.com/JuliaSparse/MUMPS.jl)

http://lostella.github.io/2018/07/25/iterative-methods-done-right.html

https://discourse.julialang.org/t/how-to-deal-with-sparse-jacobians/19561

https://julialang.org/blog/2017/08/native-julia-implementations-of-iterative-solvers-for-numerical-linear-algebra

## Todo
1. bad initial guess for `a` in newtonFold
1. test de Fold avec Hessian, comparer avec FD
2. mettre muladd etc comme dans KrylovKit
3. example de contination de Fold pour ApproxFun
3. corriger Hopf pour faire du Matrix-Free
1. plotting from Bifurcations.jl
2. improve bordered newton by calling newton
- Coder Floquet
- shooting is wrong I/h

## Todo 2
- dans `detect_bifuraction` mettre un tag `:floquet`
- calculer Forme Normale Hopf
1. il faut passer en inplace pour Newton et Continuation
- utiliser `ArrayPartition` plutôt `BorderedVector`
- Preconditionner pour Periodic Orbit
- dans Continuation Fold, changer de structure pour ne pas allouer pour chaque parametre
- faire du GPU!!
- [GPU-gmres](https://github.com/sheldonucr/GPU-GMRES) et aussi B-Euler et Matrix Free
- mettre damping dans Newton
- coder Multifario
- coder MultiContinuation de Farrell 
- implementer Forcage periodic

## Message on discourse

Dear All,

I have accumulated, over the last years, some methods for performing pseudo-arclength continuation of solutions of PDE or large scale problems. I decided to package these methods and release it publicly in case others find it useful. I tried my best to write something customizable where one can easily specify a custom linear solver (GMRES, \, \ on GPU....) and eigen solver to adapt the method to the specificity of the problem under study. It works reasonably well although I was not very careful about allocations (``premature optimization is...``)

So the package can perform continuation of solutions, detection of codim 1 bifurcation (Branch point, Fold point, Hopf point) and continue them as function of another parameter. In the context of matrix free methods, there aren't so many codes which does this (pde2path, trilinos, ?)

I did not implement branch switching because I could not be bothered implementing the different normal forms (PR suggested ;) ) BUT I implemented a very **powerful** method described by [P Farrell](http://www.pefarrell.org/) which largely makes up for this and allows you to discover many more solutions than with branch switching.

Finally, I also provide some methods for computing periodic orbits and continuation of them. There are Matrix Free methods and one suitable for Sparse problems.

I did my best **not** to rely on AbstractVector so one can use Matrices as a state space, or use ApproxFun or ArrayFire...

Please have a look at [PseudoArcLengthContinuation.jl]() and at the examples. Feel free to suggest improvements, design choices and submit PR :)

> I dont understand why loading the package results in a lot of `WARNING: Method definition...`

|Feature|Matrix Free|Custom state|
|---|---|---|
| Newton | Y | Y |
| Newton + Deflation| Y | Y |
| Continuation (Natural, Secant, Tangent) | Y | Y |
| Branching point detection | Y | Y |
| Fold detection | Y | Y |
| Hopf detection | Y | Y |
| Fold continuation | Y | N |
| Hopf continuation | Y | N |
| Periodic Orbit Newton | Y | N |
| Periodic Orbit continuation | Y | N |


Best regards,







