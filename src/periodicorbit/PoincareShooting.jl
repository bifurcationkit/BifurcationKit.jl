using DiffEqBase

####################################################################################################
# Poincare shooting based on Sánchez, J., M. Net, B. Garcı́a-Archilla, and C. Simó. “Newton–Krylov Continuation of Periodic Orbits for Navier–Stokes Flows.” Journal of Computational Physics 201, no. 1 (November 20, 2004): 13–33. https://doi.org/10.1016/j.jcp.2004.04.018.

function sectionHyp!(out, x, normals, centers)
	for ii = 1:length(normals)
		out[ii] = dot(normals[ii], x) - dot(normals[ii], centers[ii])
	end
	out
end

# this composite type encodes a set of hyperplanes which are used as Poincaré sections
struct HyperplaneSections{Tn, Tc, Ti, Tnb, Tcb}
	M::Int64		# number of hyperplanes
	normals::Tn 	# normals to define hyperplanes
	centers::Tc 	# representative point on each hyperplane
	indices::Ti 	# indices to be removed in the operator Ek

	normals_bar::Tnb
	centers_bar::Tcb

	function HyperplaneSections(normals, centers)
		M = length(normals)
		indices = zeros(Int64, M)
		for ii=1:M
			indices[ii] = argmax(abs.(normals[ii]))
		end
		nbar = [R(normals[ii], indices[ii]) for ii=1:M]
		cbar = [R(centers[ii], indices[ii]) for ii=1:M]

		return new{typeof(normals), typeof(centers), typeof(indices), typeof(nbar), typeof(cbar)}(M, normals, centers, indices, nbar, cbar)
	end
end

(hyp::HyperplaneSections)(out, u) = sectionHyp!(out, u, hyp.normals, hyp.centers)

"""
	update!(hyp::HyperplaneSections, normals, centers)

Update the hyperplanes saved in `hyp`.
"""
function update!(hyp::HyperplaneSections, normals, centers)
	M = hyp.M
	@assert length(normals) == M "Wrong number of normals"
	@assert length(centers) == M "Wrong number of centers"
	hyp.normals .= normals
	hyp.centers .= centers
	for ii=1:M
		hyp.indices[ii] = argmax(abs.(normals[ii]))
		R!(hyp.nbar[ii], normals[ii], indices[ii])
		R!(hyp.cbar[ii], centers[ii], indices[ii])
	end
	return hyp
end


# Operateur Rk from the paper above
function R!(out, x::AbstractVector, k::Int)
	@views out[1:k-1] .= x[1:k-1]
	@views out[k:end] .= x[k+1:end]
	return out
end

R!(hyp::HyperplaneSections, out, x::AbstractVector, ii::Int) = R!(out, x, hyp.indices[ii])

function R(x::AbstractVector, k::Int)
	out = similar(x, length(x) - 1)
	R!(out, x, k)
end

function R(hyp::HyperplaneSections, x::AbstractVector, ii::Int)
	out = similar(x, length(x) - 1)
	R!(hyp, out, x, ii)
end

# differential of R
dR!(hyp::HyperplaneSections, out, dx::AbstractVector, ii::Int) = R!(hyp, out, dx, ii)

# Operateur Ek from the paper above
function E!(hyp::HyperplaneSections, out, xbar::AbstractVector, ii::Int)
	@assert length(xbar) == length(hyp.normals[1]) - 1 "Wrong size for the projector / expansion operators, length(xbar) = $(length(xbar)) and length(normal) = $(length(hyp.normals[1]))"
	k = hyp.indices[ii]
	nbar = hyp.normals_bar[ii]
	xcbar = hyp.centers_bar[ii]
	coord_k = hyp.centers[ii][k] - (dot(nbar, xbar) - dot(nbar, xcbar)) / hyp.normals[ii][k]

	@views out[1:k-1] .= xbar[1:k-1]
	@views out[k+1:end] .= xbar[k:end]
	out[k] = coord_k
	return out
end

function E(hyp::HyperplaneSections, xbar::AbstractVector, ii::Int)
	out = similar(xbar, length(xbar) + 1)
	E!(hyp, out, xbar, ii)
end

# differential of E!
function dE!(hyp::HyperplaneSections, out, dxbar::AbstractVector, ii::Int)
	k = hyp.indices[ii]
	nbar = hyp.normals_bar[ii]
	xcbar = hyp.centers_bar[ii]
	coord_k = - dot(nbar, dxbar) / hyp.normals[ii][k]

	@views out[1:k-1]   .= dxbar[1:k-1]
	@views out[k+1:end] .= dxbar[k:end]
	out[k] = coord_k
	return out
end

function dE(hyp::HyperplaneSections, dxbar::AbstractVector, ii::Int)
	out = similar(dxbar, length(dxbar) + 1)
	dE!(hyp, out, dxbar, ii)
end
####################################################################################################
"""
pb = PoincareShootingProblem(flow::Flow, M, section)

This composite type implements	 the Poincaré Shooting method to locate periodic orbits, basically using Poincaré return maps. The arguments are as follows
- `flow::Flow`: implements the flow of the Cauchy problem though the structure `Flow`.
- `M`: the number of return maps. If `M==1`, then the simple shooting is implemented and the multiple one otherwise.
- `section`: implements a Poincaré section condition. The evaluation `section(x)` must return a scalar number where `x` is a guess for the periodic orbit when `M=1`. Otherwise, one must implement a function `section(out, x)` which populates `out` with the `M` sections.

## Simplified constructors
- A simpler way is to create a functional is

`pb = PoincareShootingProblem(F, p, prob::ODEProblem, alg, section; kwargs...)`

for simple shooting or

`pb = PoincareShootingProblem(F, p, prob::ODEProblem, alg, M::Int, section; kwargs...)`

for multiple shooting . Here `F` is the vector field, `p` is a parameter (to be passed to the vector and the flow), `prob` is an `ODEProblem` which is used to create a flow using the ODE solver `alg` (for example `Tsit5()`). Finally, the arguments `kwargs` are passed to the ODE solver defining the flow. Look at `DifferentialEquations.jl` for more information.

- Another convenient call is

`pb = PoincareShootingProblem(F, p, prob::ODEProblem, alg, normals::AbstractVector, centers::AbstractVector; δ = 1e-8, kwargs...)`

where `normals` (resp. `centers`) is a list of normals (resp. centers) which defines a list of hyperplanes ``\\Sigma_i``. These hyperplanes are used to define partial Poincaré return maps. δ is a numerical value used for the Matrix-Free Jacobian by finite differences. If set to 0, analytical jacobian is used.

## Computing the functionals
You can then call `pb(orbitguess)` to apply the functional to a guess. Note that `orbitguess::AbstractVector` must be of size M * N where N is the number of unknowns in the state space and `M` is the number of Poincaré maps. Another accepted `guess` is such that `guess[i]` is the state of the orbit on the `i`th section. This last form allows for non-vector state space which can be convenient for 2d problems for example.

A functional, hereby called `G` encodes this shooting problem. For example, the following methods are available:
- `pb(orbitguess)` evaluates the functional G on `orbitguess`
- `pb(orbitguess, du)` evaluates the jacobian `dG(orbitguess).du` functional at `orbitguess` on `du`

!!! tip "Tip"
    You can use the function `getPeriod(pb, sol)` to get the period of the solution `sol`
"""
@with_kw struct PoincareShootingProblem{Tf <: Flow, Tsection <: HyperplaneSections} <: AbstractShootingProblem
	M::Int64 = 1				# number of Poincaré sections
	flow::Tf					# should be a Flow{TF, Tf, Td}
	section::Tsection			# Poincaré sections
	δ::Float64 = 0e-8			# Numerical value used for the Matrix-Free Jacobian by finite differences. If set to 0, analytical jacobian is used
	isparallel::Bool = false	# whether we use DE in Ensemble mode for multiple shooting
end

@inline getM(psh::PoincareShootingProblem) = psh.M
@inline isParallel(psh::PoincareShootingProblem) = psh.isparallel

function PoincareShootingProblem(F, p,
			prob::ODEProblem, alg,
			normals::AbstractVector, centers::AbstractVector;
			δ = 1e-8, interp_points = 50, isparallel = false, kwargs...)
	hyp = HyperplaneSections(normals, centers)
	pSection(out, u, t, integrator) = (hyp(out, u); out .= out .* (integrator.iter > 1))
	affect!(integrator, idx) = terminate!(integrator)
	# we put nothing option to have an upcrossing
	cb = VectorContinuousCallback(pSection, affect!, hyp.M; interp_points = interp_points, affect_neg! = nothing)
	# change the ODEProblem -> EnsembleProblem for the parallel case
	_M = length(normals)
	isparallel = _M==1 ? false : isparallel
	_pb = isparallel ? EnsembleProblem(prob) : prob
	return PoincareShootingProblem(flow = Flow(F, p, _pb, alg; callback = cb, kwargs...), M = hyp.M, section = hyp, δ = δ, isparallel = isparallel)
end

function PoincareShootingProblem(F, p,
				prob1::ODEProblem, alg1,
				prob2::ODEProblem, alg2,
				hyp::HyperplaneSections;
				δ = 1e-8, interp_points = 50, isparallel = false, kwargs...)
	pSection(out, u, t, integrator) = (hyp(out, u); out .= out .* (integrator.iter > 1))
	affect!(integrator, idx) = terminate!(integrator)
	# we put nothing option to have an upcrossing
	cb = VectorContinuousCallback(pSection, affect!, hyp.M; interp_points = interp_points, affect_neg! = nothing)
	# change the ODEProblem -> EnsembleProblem for the parallel case
	_M = hyp.M
	isparallel = _M==1 ? false : isparallel
	_pb1 = isparallel ? EnsembleProblem(prob1) : prob1
	_pb2 = isparallel ? EnsembleProblem(prob2) : prob2
	return PoincareShootingProblem(flow = Flow(F, p, _pb1, alg1, _pb2, alg2; callback = cb, kwargs...), M = hyp.M, section = hyp, δ = δ, isparallel = isparallel)
end

function PoincareShootingProblem(F, p,
				prob1::ODEProblem, alg1,
				prob2::ODEProblem, alg2,
				normals::AbstractVector, centers::AbstractVector;
				δ = 1e-8, interp_points = 50, isparallel = false, kwargs...)
	return PoincareShootingProblem(F, p,
					prob1, alg2, prob2, alg2,
					HyperplaneSections(normals, centers);
					δ = δ, interp_points = interp_points, isparallel = isparallel, kwargs...)
end

function getPeriod(psh::PoincareShootingProblem, x_bar)
	M = getM(psh)
	Nm1 = div(length(x_bar), M)

	# reshape the period orbit guess
	x_barc = reshape(x_bar, Nm1, M)

	# variable to hold the computed result
	xc = similar(x_bar, Nm1 + 1, M)
	outc = similar(xc)

	Th = eltype(x_barc)
	period = Th(0)

	# we extend the state space to be able to call the flow, so we fill xc
	if ~isParallel(psh)
		for ii=1:M
			E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
			# We need the callback to be active here!!!
			period += @views psh.flow(Val(:TimeSol), xc[:, ii], Inf64).t
		end
	else
		for ii=1:M
			E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
		end
		solOde =  psh.flow(Val(:TimeSol), xc, repeat([Inf64],M))
		for ii=1:M
			period += solOde[ii].t
		end
	end
	return period
end

function _getMax(psh::PoincareShootingProblem, x_bar::AbstractVector; ratio = 1)
	# this function extracts the amplitude of the cycle
	M = getM(psh)
	Nm1 = div(length(x_bar), M)

	# reshape the period orbit guess
	x_barc = reshape(x_bar, Nm1, M)
	xc = similar(x_bar, Nm1 + 1, M)

	Th = eltype(x_bar)
	mx = Th(0)

	if ~isParallel(psh)
		for ii=1:M
			E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
			# We need the callback to be active here!!!
			sol = @views psh.flow(Val(:Full), xc[:, ii], Inf64)
			mx = max(mx, maximum(sol[1:div(Nm1, ratio), :]))
		end
	else
		for ii=1:M
			E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
		end
		solOde =  psh.flow(Val(:Full), xc, repeat([Inf64], M) )
		for ii=1:M
			mx = max(mx, maximum(solOde[ii].u[1:div(Nm1, ratio), :]))
		end
	end

	return mx
end

# Poincaré (multiple) shooting with hyperplanes parametrization
function (psh::PoincareShootingProblem)(x_bar::AbstractVector; verbose = false)
	M = getM(psh)
	Nm1 = div(length(x_bar), M)

	# reshape the period orbit guess
	x_barc = reshape(x_bar, Nm1, M)

	# TODO the following declaration of xc allocates. It would be better to make it inplace
	xc = similar(x_bar, Nm1 + 1, M)

	# variable to hold the result of the computations
	outc = similar(xc)

	# we extend the state space to be able to call the flow, so we fill xc
	#TODO create the projections on the fly
	for ii=1:M
		E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
	end

	if ~isParallel(psh)
		for ii=1:M
			im1 = (ii == 1 ? M : ii - 1)
			# We need the callback to be active here!!!
			@views outc[:, ii] .= xc[:, ii] .- psh.flow(xc[:, im1], Inf64)
		end
	else
		solOde = psh.flow(xc, repeat([Inf64],M))
		for ii=1:M
			im1 = (ii == 1 ? M : ii - 1)
			# We need the callback to be active here!!!
			@views outc[:, ii] .= xc[:, ii] .- solOde[im1][2]
		end
	end

	# build the array to be returned
	out_bar = similar(x_bar)
	out_barc = reshape(out_bar, Nm1, M)
	for i=1:M
		R!(psh.section, view(out_barc, :, i), view(outc, :, i), i)
	end
	return out_bar
end

function diffPoincareMap(psh::PoincareShootingProblem, x, dx, ii::Int)
	normal = psh.section.normals[ii]
	abs(dot(normal, dx)) > 1e-12 && @warn "Vector does not belong to hyperplane!  dot(normal, dx) = $(abs(dot(normal, dx))) and $(dot(dx, dx))"
	# compute the Poincare map from x
	tΣ, solΣ = psh.flow(Val(:TimeSol), x, Inf64)
	z = psh.flow.F(solΣ)
	# solution of the variational equation at time tΣ
	# We need the callback to be INACTIVE here!!!
	y = psh.flow(x, dx, tΣ; callback = nothing).du
	out = y .- (dot(normal, y) / dot(normal, z)) .* z
end

# jacobian of the shooting functional
function (psh::PoincareShootingProblem)(x_bar::AbstractVector, dx_bar::AbstractVector)
	δ = psh.δ
	if δ > 0
		# mostly for debugging purposes
		return (psh(x_bar .+  δ .* dx_bar) .- psh(x_bar)) ./ δ
	end

	# otherwise analytical Jacobian
	M = getM(psh)
	Nm1 = div(length(x_bar), M)

	# reshape the period orbit guess
	x_barc  = reshape( x_bar, Nm1, M)
	dx_barc = reshape(dx_bar, Nm1, M)

	# variable to hold the computed result
	xc  = similar( x_bar, Nm1 + 1, M)
	dxc = similar(dx_bar, Nm1 + 1, M)
	outc = similar(xc)

	# we extend the state space to be able to call the flow, so we fill xc
	for ii=1:M
		 E!(psh.section,  view(xc, :, ii),  view(x_barc, :, ii), ii)
		dE!(psh.section, view(dxc, :, ii), view(dx_barc, :, ii), ii)
	end

	if ~isParallel(psh)
		for ii=1:M
			im1 = (ii == 1 ? M : ii - 1)
			@views outc[:, ii] .= dxc[:, ii] .- diffPoincareMap(psh, xc[:, im1], dxc[:, im1], im1)
		end
	else
		@assert 1==0 "Analytical Jacobian for parallel Poincare Shooting not implemented yet. Please use the option δ > 0."
	end

	# build the array to be returned
	out_bar = similar(x_bar)
	out_barc = reshape(out_bar, Nm1, M)
	for ii=1:M
		dR!(psh.section, view(out_barc, :, ii), view(outc, :, ii), ii)
	end
	return out_bar
end
