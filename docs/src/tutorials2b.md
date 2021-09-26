# 2d Swift-Hohenberg equation (non-local) on the GPU, periodic BC (Advanced)

```@contents
Pages = ["tutorials2b.md"]
Depth = 3
```

Here we give an example where the continuation can be done **entirely** on the GPU, *e.g.* on a single V100 NIVDIA.


!!! info "Why this example?"
    This is not the simplest GPU example because we need a preconditioned linear solver and shift-invert eigen solver for this problem. On the other hand, you will be shown how to set up your own linear/eigen solver.

We choose the 2d Swift-Hohenberg as an example and consider a larger grid. See [2d Swift-Hohenberg equation: snaking, Finite Differences](@ref) for more details. Solving the sparse linear problem in $v$

$$-(I+\Delta)^2 v+(l +2\nu u-3u^2)v = rhs$$

with a **direct** solver becomes prohibitive. Looking for an iterative method, the conditioning of the jacobian is not good enough to have fast convergence, mainly because of the Laplacian operator. However, the above problem is equivalent to:

$$-v + L \cdot (d \cdot v) = L\cdot rhs$$

where

$$L := ((I+\Delta)^2 + I)^{-1}$$

is very well conditioned and

$$d := l+1+2\nu v-3v^2.$$

Hence, to solve the previous equation, only a **few** GMRES iterations are required.

> In effect, the preconditioned PDE is an example of nonlocal problem.


## Linear Algebra on the GPU

We plan to use `KrylovKit` on the GPU. We define the following types so it is easier to switch to `Float32` for example:

```julia
using Revise, CUDA

# this disable slow operations but errors if you use one of them
CUDA.allowscalar(false)

# type used for the arrays, can be Float32 is GPU requires it
TY = Float64

# put the AF = Array{TY} instead to make the code on the CPU
AF = CuArray{TY}
```

## Computing the inverse of the differential operator
The issue now is to compute $L$ but this is easy using Fourier transforms.


Hence, that's why we slightly modify the previous Example by considering **periodic** boundary conditions. Let us now show how to compute $L$. Although the code looks quite technical, it is based on two facts. First, the Fourier transform symbol associated to $L$ is

$$l_1 = 1+(1-k_x^2-k_y^2)^2$$

which is pre-computed in the composite type `SHLinearOp `. Then, the effect of `L` on `u` is as simple as `real.(ifft( l1 .* fft(u) ))` and the inverse `L\u` is `real.(ifft( fft(u) ./ l1 ))`. However, in order to save memory on the GPU, we use inplace FFTs to reduce temporaries which explains the following code.

```julia
using AbstractFFTs, FFTW, KrylovKit, Setfield, Parameters
using BifurcationKit, LinearAlgebra, Plots
const BK = BifurcationKit

# the following struct encodes the operator L1
# Making the linear operator a subtype of BK.AbstractLinearSolver is handy as it will be used
# in the Newton iterations.
struct SHLinearOp{Treal, Tcomp, Tl1, Tplan, Tiplan} <: BK.AbstractLinearSolver
	tmp_real::Treal         # temporary
	tmp_complex::Tcomp      # temporary
	l1::Tl1
	fftplan::Tplan
	ifftplan::Tiplan
end

# this is a constructor for the above struct
function SHLinearOp(Nx, lx, Ny, ly; AF = Array{TY})
	# AF is a type, it could be CuArray{TY} to run the following on GPU
	k1 = vcat(collect(0:Nx/2), collect(Nx/2+1:Nx-1) .- Nx)
	k2 = vcat(collect(0:Ny/2), collect(Ny/2+1:Ny-1) .- Ny)
	d2 = [(1-(pi/lx * kx)^2 - (pi/ly * ky)^2)^2 + 1. for kx in k1, ky in k2]
	tmpc = Complex.(AF(zeros(Nx, Ny)))
	return SHLinearOp(AF(zeros(Nx, Ny)), tmpc, AF(d2), plan_fft!(tmpc), plan_ifft!(tmpc))
end

import Base: *, \

# generic function to apply operator op to u
function apply(c::SHLinearOp, u, multiplier, op = *)
	c.tmp_complex .= Complex.(u)
	c.fftplan * c.tmp_complex
	c.tmp_complex .= op.(c.tmp_complex, multiplier)
	c.ifftplan * c.tmp_complex
	c.tmp_real .= real.(c.tmp_complex)
	return copy(c.tmp_real)
end

# action of L
*(c::SHLinearOp, u) = apply(c, u, c.l1)

# inverse of L
\(c::SHLinearOp, u) = apply(c, u, c.l1, /)
```

Before applying a Newton solver, we need to tell how to solve the linear equation arising in the Newton Algorithm.

```julia
# inverse of the jacobian of the PDE
function (sh::SHLinearOp)(J, rhs; shift = 0., tol =  1e-9)
	u, l, ν = J
	udiag = l .+ 1 .+ 2ν .* u .- 3 .* u.^2 .- shift
	res, info = KrylovKit.linsolve( du -> -du .+ sh \ (udiag .* du), sh \ rhs,
	tol = tol, maxiter = 6)
	return res, true, info.numops
end
```

Now that we have our operator `L`, we can encode our functional:

```julia
function F_shfft(u, p)
	@unpack l, ν, L = p
	return -(L * u) .+ ((l+1) .* u .+ ν .* u.^2 .- u.^3)
end
```


Let us now show how to build our operator `L` and an initial guess `sol0` using the above defined structures.

```julia
using LinearAlgebra, Plots

# to simplify plotting of the solution
plotsol(x; k...) = heatmap(reshape(Array(x), Nx, Ny)'; color=:viridis, k...)
plotsol!(x; k...) = heatmap!(reshape(Array(x), Nx, Ny)'; color=:viridis, k...)
norminf(x) = maximum(abs.(x))

# norm compatible with CUDA
norminf(x) = maximum(abs.(x))

Nx = 2^10
Ny = 2^10
lx = 8pi * 2
ly = 2*2pi/sqrt(3) * 2

X = -lx .+ 2lx/(Nx) * collect(0:Nx-1)
Y = -ly .+ 2ly/(Ny) * collect(0:Ny-1)

sol0 = [(cos(x) .+ cos(x/2) * cos(sqrt(3) * y/2) ) for x in X, y in Y]
		sol0 .= sol0 .- minimum(vec(sol0))
		sol0 ./= maximum(vec(sol0))
		sol0 = sol0 .- 0.25
		sol0 .*= 1.7

L = SHLinearOp(Nx, lx, Ny, ly, AF = AF)
J_shfft(u, p) = (u, p.l, p.ν)
# parameters of the PDE
par = (l = -0.15, ν = 1.3, L = L)
```

## Newton iterations and deflation

We are now ready to perform Newton iterations:

```julia
opt_new = NewtonPar(verbose = true, tol = 1e-6, maxIter = 100, linsolver = L)
	sol_hexa, hist, flag = @time newton(
		F_shfft, J_shfft,
		AF(sol0), par,
		opt_new, normN = norminf)

	println("--> norm(sol) = ", maximum(abs.(sol_hexa)))
	plotsol(sol_hexa)
```

You should see this:

```julia
┌─────────────────────────────────────────────────────┐
│ Newton Iterations      f(x)      Linear Iterations  │
├─────────────┬──────────────────────┬────────────────┤
│       0     │       3.3758e-01     │        0       │
│       1     │       8.0152e+01     │       12       │
│       2     │       2.3716e+01     │       28       │
│       3     │       6.7353e+00     │       22       │
│       4     │       1.9498e+00     │       17       │
│       5     │       5.5893e-01     │       14       │
│       6     │       1.0998e-01     │       12       │
│       7     │       1.1381e-02     │       11       │
│       8     │       1.6393e-04     │       11       │
│       9     │       7.3277e-08     │       10       │
└─────────────┴──────────────────────┴────────────────┘
  0.317790 seconds (42.67 k allocations: 1.256 MiB)
--> norm(sol) = 1.26017611779702
```

**Note that this is about the 10x faster than Example 2 but for a problem almost 100x larger! (On a V100 GPU)**

The solution is:

![](SH-GPU.png)

We can also use the deflation technique (see [`DeflationOperator`](@ref) and [`DeflatedProblem`](@ref) for more information) on the GPU as follows

```julia
deflationOp = DeflationOperator(2, dot, 1.0, [sol_hexa])

opt_new = @set opt_new.maxIter = 250
outdef, _, flag, _ = @time newton(
		F_shfft, J_shfft,
		0.4 .* sol_hexa .* AF([exp(-1(x+0lx)^2/25) for x in X, y in Y]),
		par, opt_new, deflationOp, normN = x-> maximum(abs.(x)))
	println("--> norm(sol) = ", norm(outdef))
	plotsol(outdef) |> display
	flag && push!(deflationOp, outdef)
```

and get:

![](SH-GPU-deflation.png)


## Computation of the branches

Finally, we can perform continuation of the branches on the GPU:

```julia
opts_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.007, ds= -0.005,
	pMax = 0., pMin = -1.0, theta = 0.5, plotEveryStep = 5,
	newtonOptions = setproperties(opt_new; tol = 1e-6, maxIter = 15), maxSteps = 100)

	br, = @time continuation(F_shfft, J_shfft,
		deflationOp[1], par, (@lens _.l), opts_cont;
		plot = true, verbosity = 3,
		plotSolution = (x, p; kwargs...)->plotsol!(x; color=:viridis, kwargs...),
		normC = x -> maximum(abs.(x))
		)
```

We did not detail how to compute the eigenvalues on the GPU and detect the bifurcations. It is based on a simple Shift-Invert strategy, please look at `examples/SH2d-fronts-cuda.jl`.

![](GPU-branch.png)

We have the following information about the branch of hexagons

```julia
julia> br
Branch number of points: 67
Branch of Equilibrium
Bifurcation points:
 (ind_ev = index of the bifurcating eigenvalue e.g. `br.eig[idx].eigenvals[ind_ev]`)
- #  1,    nd at p ≈ -0.21522461 ∈ (-0.21528614, -0.21522461), |δp|=6e-05, [converged], δ = ( 3,  0), step =  24, eigenelements in eig[ 25], ind_ev =   3
- #  2,    nd at p ≈ -0.21469007 ∈ (-0.21479652, -0.21469007), |δp|=1e-04, [converged], δ = ( 2,  0), step =  25, eigenelements in eig[ 26], ind_ev =   5
- #  3,    nd at p ≈ -0.21216919 ∈ (-0.21264341, -0.21216919), |δp|=5e-04, [converged], δ = ( 2,  0), step =  27, eigenelements in eig[ 28], ind_ev =   7
- #  4,    nd at p ≈ -0.21052576 ∈ (-0.21110899, -0.21052576), |δp|=6e-04, [converged], δ = ( 2,  0), step =  28, eigenelements in eig[ 29], ind_ev =   9
- #  5,    nd at p ≈ -0.20630678 ∈ (-0.21052576, -0.20630678), |δp|=4e-03, [converged], δ = ( 8,  0), step =  29, eigenelements in eig[ 30], ind_ev =  17
- #  6,    nd at p ≈ -0.19896508 ∈ (-0.19897308, -0.19896508), |δp|=8e-06, [converged], δ = ( 6,  0), step =  30, eigenelements in eig[ 31], ind_ev =  23
- #  7,    nd at p ≈ -0.18621673 ∈ (-0.18748234, -0.18621673), |δp|=1e-03, [converged], δ = ( 2,  0), step =  33, eigenelements in eig[ 34], ind_ev =  25
- #  8,    nd at p ≈ -0.17258147 ∈ (-0.18096574, -0.17258147), |δp|=8e-03, [converged], δ = ( 4,  0), step =  35, eigenelements in eig[ 36], ind_ev =  29
- #  9,    nd at p ≈ -0.14951737 ∈ (-0.15113148, -0.14951737), |δp|=2e-03, [converged], δ = (-4,  0), step =  39, eigenelements in eig[ 40], ind_ev =  29
- # 10,    nd at p ≈ -0.14047758 ∈ (-0.14130979, -0.14047758), |δp|=8e-04, [converged], δ = (-2,  0), step =  41, eigenelements in eig[ 42], ind_ev =  25
- # 11,    nd at p ≈ -0.11304882 ∈ (-0.11315916, -0.11304882), |δp|=1e-04, [converged], δ = (-4,  0), step =  45, eigenelements in eig[ 46], ind_ev =  23
- # 12,    nd at p ≈ -0.09074623 ∈ (-0.09085968, -0.09074623), |δp|=1e-04, [converged], δ = (-6,  0), step =  49, eigenelements in eig[ 50], ind_ev =  19
- # 13,    nd at p ≈ -0.07062574 ∈ (-0.07246519, -0.07062574), |δp|=2e-03, [converged], δ = (-4,  0), step =  52, eigenelements in eig[ 53], ind_ev =  13
- # 14,    nd at p ≈ -0.06235903 ∈ (-0.06238787, -0.06235903), |δp|=3e-05, [converged], δ = (-2,  0), step =  54, eigenelements in eig[ 55], ind_ev =   9
- # 15,    nd at p ≈ -0.05358077 ∈ (-0.05404312, -0.05358077), |δp|=5e-04, [converged], δ = (-2,  0), step =  56, eigenelements in eig[ 57], ind_ev =   7
- # 16,    nd at p ≈ -0.02494422 ∈ (-0.02586444, -0.02494422), |δp|=9e-04, [converged], δ = (-2,  0), step =  60, eigenelements in eig[ 61], ind_ev =   5
- # 17,    nd at p ≈ -0.00484022 ∈ (-0.00665356, -0.00484022), |δp|=2e-03, [converged], δ = (-2,  0), step =  63, eigenelements in eig[ 64], ind_ev =   3
- # 18,    nd at p ≈ +0.00057801 ∈ (-0.00122418, +0.00057801), |δp|=2e-03, [converged], δ = ( 5,  0), step =  64, eigenelements in eig[ 65], ind_ev =   6
- # 19,    nd at p ≈ +0.00320921 ∈ (+0.00141327, +0.00320921), |δp|=2e-03, [converged], δ = (10,  0), step =  65, eigenelements in eig[ 66], ind_ev =  16
Fold points:
- #  1, fold at p ≈ -0.21528694 ∈ (-0.21528694, -0.21528694), |δp|=-1e+00, [    guess], δ = ( 0,  0), step =  24, eigenelements in eig[ 24], ind_ev =   0
```
