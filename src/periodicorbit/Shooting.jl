# related to
# https://github.com/JuliaDiffEq/DiffEqParamEstim.jl/blob/72114707e95e6a19dd264c8fdbb476e9fda6ee31/src/multiple_shooting_objective.jl
using KrylovKit, DiffEqBase

abstract type AbstractShootingProblem end

####################################################################################################
# this function takes into accound a parameter passed to the vector field
# Putting the options `save_start=false` seems to bug with Sundials
function flowTimeSol(x, p, tspan, pb::ODEProblem; alg = Euler(), kwargs...)
	_prob = DiffEqBase.remake(pb; u0 = x, tspan = tspan, p = p)
	# the use of concrete_solve makes it compatible with Zygote
	sol = DiffEqBase.concrete_solve(_prob, alg, save_everystep = false; kwargs...)
	return sol.t[end], sol[end]
end

flow(x, p, tspan, pb::ODEProblem; alg = Euler(), kwargs...) = flowTimeSol(x, p, tspan, pb; alg = alg, kwargs...)[2]
flow(x, tspan, pb::ODEProblem; alg = Euler(), kwargs...) =  flow(x, nothing, tspan, pb; alg = alg, kwargs...)
################
# this flow is used for computing the derivative of the flow, so pb encode the variational equation
function dflow(dx, x::AbstractVector, tspan, pb::ODEProblem; alg = Euler(), kwargs...)
	n = length(x)
	_prob = DiffEqBase.remake(pb; u0 = vcat(x, dx), tspan = tspan)
	# the use of concrete_solve makes it compatible with Zygote
	sol = DiffEqBase.concrete_solve(_prob, alg, save_everystep = false; kwargs...)[end]
	return sol[1:n], sol[n+1:end]
end

function dflow(dx, x::AbstractVector, p, tspan, pb::ODEProblem; alg = Euler(), kwargs...)
	n = length(x)
	_prob = DiffEqBase.remake(pb; u0 = vcat(x, dx), tspan = tspan, p = p)
	# the use of concrete_solve makes it compatible with Zygote
	sol = DiffEqBase.concrete_solve(_prob, alg, save_everystep = false; kwargs...)[end]
	return sol[1:n], sol[n+1:end]
end
dflow(dx, x::AbstractVector, tspan, pb::ODEProblem; alg = Euler(), kwargs...) = dflow(dx, x, nothing, tspan, pb; alg = alg, kwargs...)
################
# this function takes into accound a parameter passed to the vector field
function dflow_fd(x, dx, p, tspan, pb::ODEProblem; alg = Euler(), δ = 1e-9, kwargs...)
	sol1 = flow(x .+ δ .* dx, p, tspan, pb; alg = alg, kwargs...)
	sol2 = flow(x 			, p, tspan, pb; alg = alg, kwargs...)
	return sol2, (sol1 .- sol2) ./ δ
end
dflow_fd(x, dx, tspan, pb::ODEProblem; alg = Euler(), δ = 1e-9, kwargs...) = dflow_fd(x, dx, nothing, tspan, pb; alg = alg, δ = δ, kwargs...)
################
# this gives access to the full solution, convenient for Poincaré shooting
# this function takes into accound a parameter passed to the vector field and returns the full solution from the ODE solver. This is useful in Poincare Shooting to extract the period.
function flowFull(x, p, tspan, pb::ODEProblem; alg = Euler(), kwargs...)
	_prob = DiffEqBase.remake(pb; u0 = x, tspan = tspan, p = p)
	sol = DiffEqBase.solve(_prob, alg; kwargs...)
end
flowFull(x, tspan, pb::ODEProblem; alg = Euler(), kwargs...) = flowFull(x, nothing, tspan, pb; alg = alg, kwargs...)
####################################################################################################
# Structures related to computing ODE/PDE Flows
"""
	fl = Flow(F, flow, dflow)
This composite type encapsulates:
- the vector field `x -> F(x)` associated to a Cauchy problem,
- the flow (or semigroup) associated to the Cauchy problem `(x, t) -> flow(x, t)`. Only the last time point must be returned.
- the flow (or semigroup) associated to the Cauchy problem `(x, t) -> flow(x, t)`. The whole solution on the time interval (0,t) must be returned. This is not strictly necessary to provide this.
- the differential `dflow` of the flow w.r.t. `x`, `(x, dx, t) -> dflow(x, dx, t)`. One important thing is that we require `dflow(x, dx, t)` to return 2 vectors: the first is `flow(x, t)` and the second is the value of the derivative of the flow as the second vector.

There are some simple constructors for which you only have to pass a `prob::ODEProblem` from `DifferentialEquations.jl` and an ODE time stepper like `Tsit5()`. Hence, you can do for example

	fl = Flow(F, prob, Tsit5(); kwargs...)

If your vector field depends on parameters `p`, you can define a `Flow` using

	fl = Flow(F, p, prob, Tsit5(); kwargs...)

Finally, you can pass two `ODEProblem` where the second one is used to compute the variational equation:

	fl = Flow(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2; kwargs...)

"""
struct Flow{TF, Tf, Tts, Tff, Td}
	F::TF				# vector field F(x)
	flow::Tf			# flow(x,t)
	flowTimeSol::Tts	# flow(x,t)
	flowFull::Tff		# flow(x,t) returns full solution
	dflow::Td			# dflow(x,dx,t) returns (flow(x,t), dflow(x,t)⋅dx)

	function Flow(F::TF, fl::Tf, fts::Tts, flf::Tff, df::Td) where {TF, Tf, Tts, Tff, Td}
		new{TF, Tf, Tts, Tff, Td}(F, fl, fts, flf, df)
	end

	function Flow(F::TF, fl::Tf, df::Td) where {TF, Tf, Td}
		new{TF, Tf, Nothing, Nothing, Td}(F, fl, nothing, nothing, df)
	end
end

(fl::Flow)(x, tspan)     = fl.flow(x, tspan)
(fl::Flow)(x, dx, tspan) = fl.dflow(x, dx, tspan)
(fl::Flow)(::Val{:Full}, x, tspan) = fl.flowFull(x, tspan)
(fl::Flow)(::Val{:TimeSol}, x, tspan) = fl.flowTimeSol(x, tspan)

"""
Creates a Flow variable based on a `prob::ODEProblem` and ODE solver `alg`. The vector field `F` has to be passed, this will be resolved in the future as it can be recovered from `prob`. Also, the derivative of the flow is estimated with finite differences.
"""
# this constructor takes into accound a parameter passed to the vector field
function Flow(F, p, prob::ODEProblem, alg; kwargs...)
	return Flow(F,
		(x, t) ->			 	flow(x, p, (zero(t), t), prob; alg = alg, kwargs...),
		(x, t) ->		 flowTimeSol(x, p, (zero(t), t), prob; alg = alg, kwargs...),
		# we remove the callback in order to use this for the Jacobian in Poincare Shooting
		(x, t) -> 		 	flowFull(x, p, (zero(t), t), prob; alg = alg, kwargs..., callback = nothing),
		(x, dx, t) -> dflow_fd(x, dx, p, (zero(t), t), prob; alg = alg, kwargs...)
		)
end

function Flow(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2; kwargs...)
	return Flow(F,
		(x, t) -> 			flow(x, p, (zero(t), t), prob1, alg = alg1; kwargs...),
		(x, t) ->	  flowTimeSol(x, p, (zero(t), t), prob; alg = alg1, kwargs...),
		# we remove the callback in order to use this for the Jacobian in Poincare Shooting
		(x, t) -> 		flowFull(x, p, (zero(t), t), prob1, alg = alg1; kwargs..., callback = nothing),
		(x, dx, t) ->  dflow(dx, x, p, (zero(t), t), prob2, alg = alg2; kwargs...)
		)
end
####################################################################################################
# Standard Shooting functional
"""
	pb = ShootingProblem(flow::Flow, ds, section)

This composite type implements the Standard Simple / Multiple Standard Shooting  method to locate periodic orbits. The arguments are as follows
- `flow::Flow`: implements the flow of the Cauchy problem though the structure `Flow`.
- `ds`: vector of time differences for each shooting. Its length is written `M`. If `M==1`, then the simple shooting is implemented and the multiple one otherwise.
- `section`: implements a phase condition. The evaluation `section(x)` must return a scalar number where `x` is a guess for the periodic orbit. Note that the period `T` of the guess `x` is always included either as the last component of `T = x[end]` or as `T = x.p`. The type of `x` depends on what is passed to the newton solver.

You can then call `pb(orbitguess)` to apply the functional to a guess. Note that `orbitguess` must be of size M * N + 1 where N is the number of unknowns of the state space and `orbitguess[M * N + 1]` is an estimate of the period `T` of the limit cycle. This form of guess is convenient for the use of the linear solvers in `IterativeSolvers.jl` (for example) which accepts only `AbstractVector`s. Another accepted guess is of the form `BorderedArray(guess, T)` where `guess[i]` is the state of the orbit at the `i`th time slice. This last form allows for non-vector state space which can be convenient for 2d problems for example.

A functional, hereby called `G`, encodes the shooting problem. For example, the following methods are available:
- `pb(orbitguess)` evaluates the functional G on `orbitguess`
- `pb(orbitguess, du)` evaluates the jacobian `dG(orbitguess).du` functional at `orbitguess` on `du`

## Simplified constructors
- A simpler way to build a functional is to use
	pb = ShootingProblem(F, p, prob::ODEProblem, alg, M::Int, section; kwargs...)
or

	pb = ShootingProblem(F, p, prob::ODEProblem, alg, ds, section; kwargs...)
where `F` is the vector field, `p` is a parameter (to be passed to the vector field and the flow), `prob` is an `ODEProblem` which is used to create a flow using the ODE solver `alg` (for example `Tsit5()`). Finally, the arguments `kwargs` are passed to the ODE solver defining the flow. Look at `DifferentialEquations.jl` for more information. Note that, in this case, the derivative of the flow is computed internally using Finite Differences.
- The other way is an elaboration of the previous one
	pb = ShootingProblem(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2, M::Int, section; kwargs...)
or

	pb = ShootingProblem(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2, ds, section; kwargs...)
where we supply now two `ODEProblem`. The first one `prob1`, is used to define the flow associated to `F` while the second one is a problem associated to the derivative of the flow. Hence, `prob2` should implement the following vector field ``\\tilde F(x,\\phi) = (F(x),dF(x)\\cdot\\phi)``.
"""
@with_kw struct ShootingProblem{Tf <: Flow, Ts, Tsection} <: AbstractShootingProblem
	flow::Tf							# should be a Flow{TF, Tf, Td}
	ds::Ts = diff(LinRange(0, 1, 5))	# number of Poincaré sections
	section::Tsection					# sections for phase condition
end

# this constructor takes into accound a parameter passed to the vector field
ShootingProblem(F, p, prob::ODEProblem, alg, ds, section; kwargs...) = ShootingProblem(Flow(F, p, prob, alg; kwargs...), ds, section)

ShootingProblem(F, p, prob::ODEProblem, alg, M::Int, section; kwargs...) = ShootingProblem(Flow(F, p, prob, alg; kwargs...), diff(LinRange(0, 1, M + 1)), section)

# idem but with an ODEproblem to define the derivative of the flow
ShootingProblem(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2, ds, section; kwargs...) = ShootingProblem(Flow(F, p, prob1, alg1, prob2, alg2; kwargs...), ds, section)

ShootingProblem(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2, M::Int, section; kwargs...) = ShootingProblem(Flow(F, p, prob1, alg1, prob2, alg2; kwargs...), diff(LinRange(0, 1, M + 1)), section)

# this function extracts the last component of the periodic orbit
extractPeriodShooting(x::AbstractVector) = x[end]
extractPeriodShooting(x::BorderedArray)  = x.p

# Standard shooting functional
function (sh::ShootingProblem)(x::AbstractVector)
	# period of the cycle
	# Sundials does not like @views :(
	T = extractPeriodShooting(x)
	M = length(sh.ds)
	N = div(length(x) - 1, M)

	# extract the orbit guess and reshape it into a matrix as it's more convenient to handle it
	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)

	# variable to hold the computed result
	out = similar(x)
	outv = @view out[1:end-1]
	outc = reshape(outv, N, M)

	for ii in 1:M-1
		# yi = Flow(xi, dsi⋅T) - x[i+1]
		outc[:, ii] .= sh.flow(xc[:, ii], sh.ds[ii] * T) .- xc[:, ii+1]
	end
	# ym = Flow(xm, dsm⋅T) - x1
	outc[:, M] .= sh.flow(xc[:, M], sh.ds[M] * T) .- xc[:, 1]

	# add constraint
	out[end] = sh.section(x)

	return out
end

# jacobian of the shooting functional
function (sh::ShootingProblem)(x::AbstractVector, dx::AbstractVector; δ = 1e-8)
	# period of the cycle
	# Sundials does not like @views :(
	dT = extractPeriodShooting(dx)
	T  = extractPeriodShooting(x)
	M = length(sh.ds)
	N = div(length(x) - 1, M)

	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)

	dxv = @view dx[1:end-1]
	dxc = reshape(dxv, N, M)

	# variable to hold the computed result
	out = similar(x)
	outv = @view out[1:end-1]
	outc = reshape(outv, N, M)

	for ii = 1:M-1
		# call jacobian of the flow
		tmp = sh.flow(xc[:, ii], dxc[:, ii], sh.ds[ii] * T)
		outc[:, ii] .= tmp[2] .+ sh.flow.F(tmp[1]) .* sh.ds[ii] * dT .- dxc[:, ii+1]
	end

	# ym = Flow(xm, dsm⋅T) - x1
	tmp = sh.flow(xc[:, M], dxc[:, M], sh.ds[M] * T)
	outc[:, M] .= tmp[2] .+ sh.flow.F(tmp[1]) .* sh.ds[M] * dT .- dxc[:, 1]

	# add constraint
	out[end] = (sh.section(x .+ δ .* dx) - sh.section(x)) / δ

	return out
end

# shooting functional, this allows for Array state space
function (sh::ShootingProblem)(x::BorderedArray)
	# period of the cycle
	T = extractPeriodShooting(x)
	M = length(x.u)

	# variable to hold the computed result
	out = BorderedArray{typeof(x.u), typeof(x.p)}(similar(x.u), typeof(x.p)(0))

	for ii in 1:M-1
		# yi = Flow(xi, dsi⋅T) - x[i+1]
		out.u[ii] .= sh.flow(x.u[ii], sh.ds[ii] * T) .- x.u[ii+1]
	end
	# ym = Flow(xm, sm T) - x1
	out.u[M] .= sh.flow(x.u[M], sh.ds[M] * T) .- x.u[1]

	# add constraint
	out.p = sh.section(x)

	return out
end

# jacobian of the shooting functional, this allows for Array state space
function (sh::ShootingProblem)(x::BorderedArray, dx::BorderedArray; δ = 1e-8)
	# period of the cycle
	dT = extractPeriodShooting(dx)
	T  = extractPeriodShooting(x)
	M = length(x.u)

	# variable to hold the computed result
	out = BorderedArray{typeof(x.u), typeof(x.p)}(similar(x.u), typeof(x.p)(0))

	for ii = 1:M-1
		# call jacobian of the flow
		tmp = sh.flow(x.u[ii], dx.u[ii], sh.ds[ii] * T)
		out.u[ii] .= tmp[2] .+ sh.flow.F(tmp[1]) .* sh.ds[ii] * dT .- dx.u[ii+1]
	end

	# ym = Flow(xm, sm T) - x1
	tmp = sh.flow(x.u[M], dx.u[M], sh.ds[M] * T)
	out.u[M] .= tmp[2] .+ sh.flow.F(tmp[1]) .* sh.ds[M] * dT .- dx.u[1]

	# add constraint
	x_tmp = similar(x.u); copyto!(x_tmp, x.u)
	axpy!(δ , dx.u, x_tmp)
	out.p = (sh.section(BorderedArray(x_tmp, T + δ * dT)) - sh.section(x)) / δ

	return out
end

function _getMax(prob::ShootingProblem, x::AbstractVector; ratio = 1)
	# this function extracts the amplitude of the cycle
	T = extractPeriodShooting(x)
	M = length(prob.ds)
	N = div(length(x) - 1, M)
	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)

	# !!!! we could use @views but then Sundials will complain !!!
	sol = prob.flow(Val(:Full), xc[:, 1], T)
	mx = @views maximum(sol[1:div(N, ratio), :], dims = 1)
end

function getAmplitude(prob::ShootingProblem, x::AbstractVector; ratio = 1)
	mx = _getMax(prob, x; ratio = ratio)
	return maximum(mx) - minimum(mx)
end

function getMaximum(prob::ShootingProblem, x::AbstractVector; ratio = 1)
	mx = _getMax(prob, x; ratio = ratio)
	return maximum(mx)
end

function sectionShooting(x::AbstractVector, po::AbstractMatrix, p, F)
	res = 1.0
	N, M = size(po)
	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)

	for ii = 1:size(x, 2)
		res *= @views dot(xc[:, ii], F(po[:, ii], p)) .- dot(po[:, ii], F(po[:, ii], p))
	end
	res
end
####################################################################################################
"""
pb = PoincareShootingProblem(flow::Flow, M, section)

This composite type implements	 the Poincaré Shooting method to locate periodic orbits, basically using Poincaré return maps. The arguments are as follows
- `flow::Flow`: implements the flow of the Cauchy problem though the structure `Flow`.
- `M`: the number of return maps. If `M==1`, then the simple shooting is implemented and the multiple one otherwise.
- `section`: implements a Poincaré section condition. The evaluation `section(x)` must return a scalar number where `x` is a guess for the periodic orbit when `M=1`. Otherwise, one must implement a function `section(out, x)` which populates `out` with the `M` hyperplanes.

## Simplified constructors
A simpler way is to create a functional is `pb = PoincareShootingProblem(F, p, prob::ODEProblem, alg, section; kwargs...)` for simple shooting or `pb = PoincareShootingProblem(F, p, prob::ODEProblem, alg, M::Int, section; kwargs...)` for multiple shooting . Here `F` is the vector field, `p` is a parameter (to be passed to the vector and the flow), `prob` is an `ODEProblem` which is used to create a flow using the ODE solver `alg` (for example `Tsit5()`). Finally, the arguments `kwargs` are passed to the ODE solver defining the flow. Look at `DifferentialEquations.jl` for more information.

Another convenient call is `pb = PoincareShootingProblem(F, p, prob::ODEProblem, alg, normals::AbstractVector, centers::AbstractVector; δ = 1e-8, kwargs...)` where `normals` (resp. `centers`) is a list of normals (resp. centers) which define a list of hyperplanes ``\\Sigma_i``. These hyperplanes are used to define partial Poincaré return maps. δ is a numerical value used for the Matrix-Free Jacobian by finite differences. If set to 0, analytical jacobian is used. See docs for more information.

## Computing the functionals
You can then call `pb(orbitguess)` to apply the functional to a guess. Note that `orbitguess` must be of size M * N where N is the number of unknowns in the state space and `M` is the number of Poincaré maps. Another accepted `guess` is such that `guess[i]` is the state of the orbit on the `i`th section. This last form allows for non-vector state space which can be convenient for 2d problems for example.

A functional, hereby called `G` encodes this shooting problem. For example, the following methods are available:
- `pb(orbitguess)` evaluates the functional G on `orbitguess`
- `pb(orbitguess, du)` evaluates the jacobian `dG(orbitguess).du` functional at `orbitguess` on `du`

!!! tip "Tip"
    You can use the function `getPeriod(pb::PoincareShootingProblem, sol)` to get the period of the solution `sol`
"""
@with_kw struct PoincareShootingProblem{Tf <: Flow, Tsection} <: AbstractShootingProblem
	flow::Tf					# should be a Flow{TF, Tf, Td}
	M::Int64 = 1				# number of Poincaré sections
	section::Tsection			# Poincaré sections
	δ::Float64 = 0e-8			# Numerical value used for the Matrix-Free Jacobian by finite differences. If set to 0, analytical jacobian is used
end
####################################################################################################
# Poincare shooting based on Sánchez, J., M. Net, B. Garcı́a-Archilla, and C. Simó. “Newton–Krylov Continuation of Periodic Orbits for Navier–Stokes Flows.” Journal of Computational Physics 201, no. 1 (November 20, 2004): 13–33. https://doi.org/10.1016/j.jcp.2004.04.018.

function sectionHyp!(out, x, normals, centers)
	for ii = 1:length(normals)
		out[ii] = dot(normals[ii], x .- centers[ii])
	end
	out
end

# this composite type encodes a set of hyperplanes which are used as Poincaré sections
struct HyperplaneSections{Tn, Tc, Ti, Tnb, Tcb}
	M::Int			# number of hyperplanes
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

# Operateur Rk from the paper above
function R!(out, x::AbstractVector, k::Int)
	@views out[1:k-1] .= x[1:k-1]
	@views out[k:end] .= x[k+1:end]
	return out
end

R!(hyp::HyperplaneSections, out, x::AbstractVector, ii::Int) = R!(out, x, hyp.indices[ii])

function R(hyp::HyperplaneSections, x::AbstractVector, ii::Int)
	out = similar(x, length(x) - 1)
	R!(hyp, out, x, ii)
end

function R(x::AbstractVector, k::Int)
	out = similar(x, length(x) - 1)
	R!(out, x, k)
end


# differential of R
dR!(hyp::HyperplaneSections, out, dx::AbstractVector, ii::Int) = R!(hyp, out, dx, ii)

# Operateur Ek from the paper above
function E!(hyp::HyperplaneSections, out, xbar::AbstractVector, ii::Int)
	@assert length(xbar) == length(hyp.normals[1]) - 1 "Wrong size for the projector / expansion operators, length(xbar) = $(length(xbar)) and length(normal) = $(length(hyp.normals[1]))"
	k = hyp.indices[ii]
	nbar = hyp.normals_bar[ii]
	xcbar = hyp.centers_bar[ii]
	coord_k = hyp.centers[ii][k] - dot(nbar, xbar .- xcbar) / hyp.normals[ii][k]

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

	@views out[1:k-1] .= dxbar[1:k-1]
	@views out[k+1:end] .= dxbar[k:end]
	out[k] = coord_k
	return out
end

function dE(hyp::HyperplaneSections, dxbar::AbstractVector, ii::Int)
	out = similar(dxbar, length(dxbar) + 1)
	dE!(hyp, out, dxbar, ii)
end

struct HyperplanePoincareShootingProblem{Tpsh <: PoincareShootingProblem} <: AbstractShootingProblem
	psh::Tpsh
end

function getPeriod(hpsh::HyperplanePoincareShootingProblem, x_bar)
	sh = hpsh.psh
	M = sh.M
	Nm1 = div(length(x_bar), M)

	# reshape the period orbit guess
	x_barc = reshape(x_bar, Nm1, M)

	# variable to hold the computed result
	xc = similar(x_bar, Nm1 + 1, M)
	outc = similar(xc)

	Th = eltype(x_barc)
	period = Th(0)

	# we extend the state space to be able to call the flow, so we fill xc
	#TODO create the projections on the fly
	for ii=1:M
		E!(sh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
	end

	for ii in 1:M-1
		period += @views sh.flow(Val(:TimeSol), xc[:, ii+1], Inf64)[1]
	end
	period += @views sh.flow(Val(:TimeSol), xc[:, M], Inf64)[1]
end

function PoincareShootingProblem(F, p, prob::ODEProblem, alg, normals::AbstractVector, centers::AbstractVector; δ = 0e-8, interp_points = 50, kwargs...)
	hyp = HyperplaneSections(normals, centers)
	pSection(out, u, t, integrator) = (hyp(out, u); out .= out .* (integrator.iter > 1))
	affect!(integrator, idx) = terminate!(integrator)
	# we put nothing option to have an upcrossing
	cb = VectorContinuousCallback(pSection, affect!, hyp.M; interp_points = interp_points, affect_neg! = nothing)
	return HyperplanePoincareShootingProblem(
			PoincareShootingProblem(flow = Flow(F, p, prob, alg; callback = cb, kwargs...), M = hyp.M, section = hyp, δ = δ))
end

# Poincaré (multiple) shooting with hyperplanes parametrization
function (hpsh::HyperplanePoincareShootingProblem)(x_bar::AbstractVector; verbose = false)
	sh = hpsh.psh
	M = sh.M
	Nm1 = div(length(x_bar), M)

	# reshape the period orbit guess
	x_barc = reshape(x_bar, Nm1, M)

	# variable to hold the computed result
	xc = similar(x_bar, Nm1 + 1, M)
	outc = similar(xc)

	# we extend the state space to be able to call the flow, so we fill xc
	#TODO create the projections on the fly
	for ii=1:M
		E!(sh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
	end

	for ii in 1:M-1
		# yi = x[i+1] - Flow(xi)
		@views outc[:, ii] .= xc[:, ii+1] .- sh.flow(xc[:, ii], Inf64)
	end
	# ym = x1 - Flow(xm)
	@views outc[:, M] .= xc[:, 1] .- sh.flow(xc[:, M], Inf64)

	# build the array to be returned
	out_bar = similar(x_bar)
	out_barc = reshape(out_bar, Nm1, M)
	for ii=1:M
		R!(sh.section, view(out_barc, :, ii), view(outc, :, ii), ii)
	end
	return out_bar
end

function diffPoincareMap(hpsh, x, dx, ii::Int)
	sh = hpsh.psh
	normal = sh.section.normals[ii]
	abs(dot(normal, dx)) > 1e-12 * dot(dx, dx) && @warn "Vector does not belong to hyperplane!  dot(normal, dx) = $(abs(dot(normal, dx)))"
	# compute the Poincare map from x
	tΣ, solΣ = sh.flow(Val(:TimeSol), x, Inf64)
	z = sh.flow.F(solΣ)
	# @error "I dont want to have dflow avec le callback"
	y = sh.flow(x, dx, tΣ)[2]
	# @show size(normal) size(x) size(dx) size(y) size(z)
	out = y .- (dot(normal, y) / dot(normal, z)) .* z
end

# jacobian of the shooting functional
function (hpsh::HyperplanePoincareShootingProblem)(x_bar::AbstractVector, dx_bar::AbstractVector)
	δ = hpsh.psh.δ
	if δ > 0
		# mostly for debugging purposes
		return (hpsh(x_bar .+  δ .* dx_bar) .- hpsh(x_bar)) ./ δ
	end

	# otherwise analytical Jacobian
	sh = hpsh.psh
	M = sh.M
	Nm1 = div(length(x_bar), M)

	# reshape the period orbit guess
	x_barc = reshape(x_bar, Nm1, M)
	dx_barc = reshape(dx_bar, Nm1, M)

	# variable to hold the computed result
	xc  = similar( x_bar, Nm1 + 1, M)
	dxc = similar(dx_bar, Nm1 + 1, M)
	outc = similar(xc)

	# we extend the state space to be able to call the flow, so we fill xc
	for ii=1:M
		 E!(sh.section,  view(xc, :, ii),  view(x_barc, :, ii), ii)
		dE!(sh.section, view(dxc, :, ii), view(dx_barc, :, ii), ii)
	end

	for ii in 1:M-1
		@views outc[:, ii] .= dxc[:, ii+1] .- diffPoincareMap(hpsh, xc[:, ii], dxc[:, ii], ii)
	end
	@views outc[:, M] .= dxc[:, 1] .- diffPoincareMap(hpsh, xc[:, M], dxc[:, M], M)

	# build the array to be returned
	out_bar = similar(x_bar)
	out_barc = reshape(out_bar, Nm1, M)
	for ii=1:M
		dR!(sh.section, view(out_barc, :, ii), view(outc, :, ii), ii)
	end

	return out_bar
end

####################################################################################################
# if we use the same code as for newton (see below) in continuation, it is difficult to tell the eigensolver not to use the jacobian but instead the monodromy matrix. So we have to use a dedicated composite type for the jacobian to handle this case.

struct ShootingJacobian{Tpb <: AbstractShootingProblem, Torbitguess}
	pb::Tpb
	x::Torbitguess
end

# evaluation of the jacobian
(shjac::ShootingJacobian)(dx) = shjac.pb(shjac.x, dx)

####################################################################################################
# newton wrapper
"""
	newton(prob::T, orbitguess, options::NewtonPar; kwargs...) where {T <: AbstractShootingProblem}

This is the Newton Solver for computing a periodic orbit using Shooting method.

# Output:
- solution
- history of residuals
- flag of convergence
- number of iterations
"""
function newton(prob::T, orbitguess, options::NewtonPar; kwargs...) where {T <: AbstractShootingProblem}
	return newton(x -> prob(x),
			x -> (dx -> prob(x, dx)),
			orbitguess,
			options; kwargs...)
end

"""
	newton(prob::T, orbitguess, options::NewtonPar, defOp::DeflationOperator; kwargs...) where {T <: AbstractShootingProblem}

This is the deflated Newton Solver for computing a periodic orbit using Shooting method.

# Output:
- solution
- history of residuals
- flag of convergence
- number of iterations
"""
function newton(prob::Tpb, orbitguess, options::NewtonPar, defOp::DeflationOperator{T, Tf, vectype}; kwargs...) where {Tpb <: AbstractShootingProblem, T, Tf, vectype}
	return newton(x -> prob(x),
			x -> (dx -> prob(x, dx)),
			orbitguess,
			options, defOp; kwargs...)
end

####################################################################################################
# Continuation

function continuationPOShooting(prob, orbitguess, p0::Real, _contParams::ContinuationPar, linearAlgo::AbstractBorderedLinearSolver; printPeriod = true, kwargs...)

	contParams = check(_contParams)

	options = contParams.newtonOptions

	pb0 = prob(p0)

	if contParams.computeEigenValues
		if pb0 isa HyperplanePoincareShootingProblem && pb0.psh.M > 1
			@warn "Floquet multipliers wrongly computed"
		end
		contParams = @set contParams.newtonOptions.eigsolver = FloquetQaDShooting(contParams.newtonOptions.eigsolver)
	end

	if (pb0 isa PoincareShootingProblem) || (pb0 isa HyperplanePoincareShootingProblem)
		if printPeriod
			printSolutionPS = (x, p) -> getPeriod(prob(p), x)
			return continuation(
				(x, p) -> prob(p)(x),
				# (x, p) -> (dx -> prob(p)(x, dx)),
				(x, p) -> ShootingJacobian(prob(p), x),
				orbitguess, p0,
				contParams, linearAlgo;
				printSolution = printSolutionPS,
				kwargs...)
		end
	end

	return continuation(
		(x, p) -> prob(p)(x),
		(x, p) -> ShootingJacobian(prob(p), x),
		orbitguess, p0,
		contParams, linearAlgo;
		kwargs...)
end

"""
	continuationPOShooting(prob, orbitguess, p0::Real, contParams::ContinuationPar; printPeriod = true, kwargs...)

This is the continuation routine for computing a periodic orbit using a functional G based on a Shooting method.

# Arguments
- `p -> prob(p)` is a function or family such that `prob(p)::AbstractShootingProblem` encodes the functional G
- `orbitguess` a guess for the periodic orbit. For the type of `orbitguess`, please see the information concerning `ShootingProblem` and `PoincareShootingProblem`.
- `p0` initial parameter, must be a real number
- `contParams` same as for the regular `continuation` method
- `printPeriod` in the case of Poincaré Shooting, plot the period of the cycle.
"""
function continuationPOShooting(prob, orbitguess, p0::Real, contParams::ContinuationPar; linearAlgo = BorderingBLS(), printPeriod = true, kwargs...)
	_linearAlgo = @set linearAlgo.solver = contParams.newtonOptions.linsolver
	return continuationPOShooting(prob, orbitguess, p0, contParams, _linearAlgo; printPeriod = printPeriod, kwargs...)
end
