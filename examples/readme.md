# Instructions for example files

Here are some comments for the example files in this folder. Some of these examples are the same as in the [tutorials](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/tutorials/) but more developed. Some are not shown in the [tutorials](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/tutorials/) because they are too computationally involved.

You may need to install some additional packages

## brusselator.jl

```julia
using Pkg
pkg"add Plots SparseArrays "
```

## brusselatorShooting.jl

```julia
using Pkg
pkg"add Plots SparseArrays LoopVectorization DifferentialEquations ForwardDiff SparseDiffTools"
```

## carrier.jl

```julia
using Pkg
pkg"add Plots SparseArrays BandedMatrices"
```

## cGL2d.jl

```julia
using Pkg
pkg"add Plots ForwardDiff IncompleteLU SparseArrays"
```

## cGL2d-Shooting.jl

```julia
using Pkg
pkg"add Plots SparseArrays"
```

## Chan.jl

```julia
using Pkg
pkg"add Plots SparseArrays"
```

## chan-af.jl

```julia
using Pkg
pkg"add Plots SparseArrays"
```

## codim2PO.jl

```julia
using Pkg
pkg"add Plots ForwardDiff OrdinaryDiffEq"
```

## codim2PO-sh-mf.jl

This is a Work In Progress to show how to detect codim 2 bifurcations of periodic orbits in fully matrix-free context.

```julia
using Pkg
pkg"add Test Plots ComponentArrays DifferentialEquations DifferentiationInterface Zygote ForwardDiff"
```

## COModel.jl

```julia
using Pkg
pkg"add Plots DifferentialEquations"
```

## mittleman.jl

```julia
using Pkg
pkg"add Plots SparseArrays"
```

## pd-1d.jl

```julia
using Pkg
pkg"add ForwardDiff DifferentialEquations SparseArrays"
```

## SHpde_snaking.jl

```julia
using Pkg
pkg"add Plots SparseArrays"
```

## SH2d-fronts.jl

```julia
using Pkg
pkg"add Plots SparseArrays LinearAlgebra IncompleteLU"
```

## SH2d-fronts-cuda.jl

You don't need CUDA to run the example though.

```julia
using Pkg
pkg"add  AbstractFFTs FFTW KrylovKit Plots CUDA"
```

## SH3d.jl

```julia
using Pkg
pkg"add KrylovKit GLMakie SparseArrays SuiteSparse"
```

## TMModel.jl

```julia
using Pkg
pkg"add Plots DifferentialEquations"
```
