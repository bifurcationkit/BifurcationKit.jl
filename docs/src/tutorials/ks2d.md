# 2d Kuramoto–Sivashinsky Equation on GPU (Advanced)

```@contents
Pages = ["ks2d.md"]
Depth = 3
```

!!! unknown "References"
    The following example is exposed in Evstigneev, Nikolay M., and Oleg I. Ryabkov. **Bifurcation Diagram of Stationary Solutions of the 2D Kuramoto-Sivashinsky Equation in Periodic Domains.** Journal of Physics: Conference Series 1730, no. 1 2021

Here we give an example where the **bifurcation diagram** (not just the continuation) can be done **entirely** on the GPU, *e.g.* on a single V100 NIVDIA.

We choose the 2d Kuramoto–Sivashinsky Equation as an example on the 2d torus.

$$-\alpha\left(u u_{x}+u u_{y}+\Delta u\right)- \Delta^{2} u=0,\quad x \in \mathbb{T}^{2}.\tag{E}$$

For the Newton algorithm, we need to solve the linear system in $v$:

$$-\alpha\left(v u_{x}+ v u_{y}+2 u v_{x}+ u v_{y}+\Delta u\right)- \Delta^{2} v = rhs.$$

The conditioning of the jacobian is not good enough to have fast convergence. However, the above problem is equivalent to:

$$v + L \cdot (A \cdot v) = L\cdot rhs$$

where

$$L := -(\Delta^2-\alpha\Delta)^{-1}$$

is very well conditioned and

$$A\cdot v := -\alpha\left(v u_{x}+v u_{y}+u v_{x}+u v_{y}\right) = -\alpha\nabla\cdot (uv).$$

Hence, to solve the previous equation, only a **few** GMRES iterations are required.

> In effect, the preconditioned PDE is an example of nonlocal problem.

## Structures for inplace fft
The issue now is to compute $L$ but this is easy using Fourier transforms.

The **periodic** boundary conditions imply that the Fourier transform symbol associated to $L$ is

$$l_1 = \alpha k_x^2+\alpha k_y^2+(kx^2+k_y^2)^2$$

which is pre-computed in the composite type `SHLinearOp `. Then, the effect of `L` on `u` is as simple as `real.(ifft( l1 .* fft(u) ))` and the inverse `L\u` is `real.(ifft( fft(u) ./ l1 ))`. However, in order to save memory on the GPU, we use inplace FFTs to reduce temporaries which explains the following code.

```julia
using Revise
using AbstractFFTs, FFTW, KrylovKit, Setfield, Parameters
using BifurcationKit, LinearAlgebra, Plots
const BK = BifurcationKit

# the following struct encodes the operator L1
# Making the linear operator a subtype of BK.AbstractLinearSolver is handy 
# as it will be used in the Newton iterations.
mutable struct KSLinearOp{TX, Treal, Tcomp, Td1, Td2, Td4, Tsymbol, Tplan, Tiplan, Tm} <: BK.AbstractLinearSolver
	X::TX					# x grid
	Y::TX					# y grid
	tmp_real::Treal         # temporary with real values
	tmp_complex::Tcomp      # temporary with complex values
	d1::Td1					# contains symbol for ∂
	d2::Td2					# contains symbol for ∂^2
	d4::Td4					# contains symbol for ∂^4
	symbolMul::Tsymbol
	symbolDiv::Tsymbol
	fftplan::Tplan		# FFT plan
	ifftplan::Tiplan	# inverse FFT plan
	periodic::Bool		# periodic BC?
	mask::Tm		# mask to invert L
	applymask::Bool		# apply the mask?
end

function KSLinearOp(N, l; AF = Array{TY})
	Nx, Ny = N
	lx, ly = l

	tmpc = AF(Complex.(zeros(Nx, Ny)))

	X = -lx .+ 2lx/Nx * collect(0:Nx-1)
	Y = -ly .+ 2ly/Ny * collect(0:Ny-1)
	# AF is a type, it could be CuArray{TY} to run the following on GPU
	Kx = vcat(collect(0:Nx/2), collect(Nx/2+1:Nx-1) .- Nx)
	Ky = vcat(collect(0:Ny/2), collect(Ny/2+1:Ny-1) .- Ny)
	plan = plan_fft!(tmpc)
	iplan = plan_ifft!(tmpc)

	mask = ones(Nx, Ny); mask[1,1] = 0

	d1 = [ Complex(0,(pi/lx * kx)   + (pi/ly * ky))  for kx in Kx, ky in Ky]
	d2 = [          -(pi/lx * kx)^2 - (pi/ly * ky)^2 for kx in Kx, ky in Ky]
	d4 = d2 .^ 2
	return KSLinearOp(X, Y, AF(zeros(Nx, Ny)), tmpc, AF(d1), AF(d2), AF(d4), 
				AF(4d4+d2), AF(4d4+d2), plan, iplan, true, AF(mask), true)
end

function apply!(out, c::KSLinearOp, u::AbstractMatrix{ <: Real}, multiplier, op = *)
	c.tmp_complex .= Complex.(u)
	c.fftplan * c.tmp_complex
	c.tmp_complex .= op.(c.tmp_complex, multiplier)
	if c.applymask
		# remove the constant component
		# c.tmp_complex[1,1] = 0 # ca casse le calcul des VPs...
		c.tmp_complex .*= c.mask
	end
	c.ifftplan * c.tmp_complex
	out .= real.(c.tmp_complex)
	return out
end
apply(c::KSLinearOp, u, multiplier, op = *) = apply!(similar(u), c, u, multiplier, op)
```

## Code for the functional
We can use this to encode the equation (E)

```julia
function F_ksfft(u, p)
	@unpack α, L = p
	# update the operator L
	updateMul!(L, α)
	out = similar(u)
	out .= u.^2
	divergence!(out, L, out)
	# uL = L * u
	mul!(L.tmp_real, L, u)
	out .+= L.tmp_real
	out .*= -1
	out
end
```

While we are at it, let's compute the other differentials

```julia
# second differential
function d2F_ksfft(u, p, du1, du2)
	@unpack α, L = p
	# update the operator L
	updateMul!(L, α)
	return -2 .* divergence(L, du2 .* du1)
end

# third differential
d3F_ksfft(u, p, du1, du2, du3) = zero(du1)
```

## Code for the jacobian
Finally, let us encode the jacobian

```julia
mutable struct JacobianKS{Tv, T, Tl}
	u::Tv	# current value of u at which jacobian is evaluated
	α::T	# parameter α
	L::Tl	# operator L
	applySym::Bool # whether to apply the Sym operator 
end

# update the jacobian
function (J::JacobianKS)(u, p)
	# we update the internal field
	J.u .= u
	J.α = p.α
	# update the operator L
	updateMul!(J.L, p.α)
	updateDiv!(J.L, p.α)
	# return the jacobian
	return J
end

# jacobian evaluation
function (J::JacobianKS)(out::T, du::T; _transpose::Bool = false) where T
	α = J.α
	ρ = _transpose ? -2 : 2
	# out = J.L * du
	mul!(out, J.L, du)
	# J.L.tmp_real = ρ .* u .* du
	J.L.tmp_real .= ρ .* J.u .* du
	# J.L.tmp_real = div(J.L.tmp_real)
	divergence!(J.L.tmp_real, J.L, J.L.tmp_real)
	out .= (-1) .* (out .+ J.L.tmp_real)
	return out
end
(J::JacobianKS)(du; _transpose::Bool = false) = (J::JacobianKS)(similar(du), du; _transpose = _transpose)
```

## Newton iterations and deflation

We have now everything to solve (E). Let us run a Krylov-Newton algorithm to find non trivial states:

```julia
# size of the interval
_L = 1
# we make it const because we use it for the norms
const N = (2^9, 2^9)
# domain
l = (pi * _L, pi)

# Operator L
# we make it const because we update its parameters
const L = KSLinearOp(N, l; AF = AF)

# Structure to compute eigenvalues
Leig = KSEigOp(L, 0.01) # for eigenvalues computation

# initial guess for Newton
sol0 = 0.5 .* [(cos(x-2.) .+ 0sin(y) ) for x in L.X, y in L.Y] |> AF

# Structure to hold the jacobian
J_ksfft = JacobianKS(copy(sol0), 0., L, true)

# we hold the differentials together
jet = (F_ksfft, J_ksfft, d2F_ksfft, d3F_ksfft)

# parameters for the problem
par = (α = 1/4.005, L = L)

# Linear solver: GMRES with left preconditioner given by L
LsL = GMRESKrylovKit(Pl = L, verbose = 0, dim = 100, maxiter = 5)
```

We now run the Krylov-Newton

```julia
opt_new = NewtonPar(verbose = true, tol = 1e-6, linsolver = LsL, eigsolver = Leig)
sol_hexa, hist, flag = @time newton(F_ksfft, J_ksfft,
	(Π(AF(sol0))), par, opt_new,
	normN = x -> norm(x) / sqrt(N[1]*N[2])
	)
println("--> norm(sol) = ", norminf(sol_hexa), ", mean = ", mean(sol_hexa))

plotsol(sol_hexa)
```
