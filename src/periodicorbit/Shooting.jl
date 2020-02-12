# related to
# https://github.com/JuliaDiffEq/DiffEqParamEstim.jl/blob/72114707e95e6a19dd264c8fdbb476e9fda6ee31/src/multiple_shooting_objective.jl
using KrylovKit, DiffEqBase

abstract type AbstractShootingProblem end

####################################################################################################
function flow(x, tspan, pb::ODEProblem; alg = Euler(), kwargs...)
	_prob = DiffEqBase.remake(pb; u0 = x, tspan = tspan)
	# the use of concrete_solve makes it compatible with Zygote
	sol = DiffEqBase.concrete_solve(_prob, alg, save_everystep = false; kwargs...)[end]
	return sol
end

# this function takes into accound a parameter passed to the vector field
# Putting the options `save_start=false` seems to bug with Sundials
function flow(x, p, tspan, pb::ODEProblem; alg = Euler(), kwargs...)
	_prob = DiffEqBase.remake(pb; u0 = x, tspan = tspan, p = p)
	# the use of concrete_solve makes it compatible with Zygote
	sol = DiffEqBase.concrete_solve(_prob, alg, save_everystep = false; kwargs...)[end]
	return sol
end
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
################
function dflow_fd(x, dx, tspan, pb::ODEProblem; alg = Euler(), δ = 1e-9, kwargs...)
	sol1 = flow(x .+ δ .* dx, tspan, pb; alg = alg, kwargs...)
	sol2 = flow(x 			, tspan, pb; alg = alg, kwargs...)
	return sol2, (sol1 .- sol2) ./ δ
end

# this function takes into accound a parameter passed to the vector field
function dflow_fd(x, dx, p, tspan, pb::ODEProblem; alg = Euler(), δ = 1e-9, kwargs...)
	sol1 = flow(x .+ δ .* dx, p, tspan, pb; alg = alg, kwargs...)
	sol2 = flow(x 			, p, tspan, pb; alg = alg, kwargs...)
	return sol2, (sol1 .- sol2) ./ δ
end
################
# this gives access to the full solution, convenient for Poincaré shooting
function flowFull(x, tspan, pb::ODEProblem; alg = Euler(), kwargs...)
	_prob = DiffEqBase.remake(pb; u0 = x, tspan = tspan)
	sol = DiffEqBase.solve(_prob, alg; kwargs...)
end

# this function takes into accound a parameter passed to the vector field and returns the full solution from the ODE solver. This is useful in Poincare Shooting to extract the period.
function flowFull(x, p, tspan, pb::ODEProblem; alg = Euler(), kwargs...)
	_prob = DiffEqBase.remake(pb; u0 = x, tspan = tspan, p = p)
	sol = DiffEqBase.solve(_prob, alg; kwargs...)
end

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
struct Flow{TF, Tf, Tff, Td}
	F::TF				# vector field F(x)
	flow::Tf			# flow(x,t)
	flowFull::Tff		# flow(x,t) returns full solution
	dflow::Td			# dflow(x,dx,t) returns (flow(x,t), dflow(x,t)⋅dx)

	function Flow(F::TF, fl::Tf, flf::Tff, df::Td) where {TF, Tf, Tff, Td}
		new{TF, Tf, Tff, Td}(F, fl, flf, df)
	end

	function Flow(F::TF, fl::Tf, df::Td) where {TF, Tf, Td}
		new{TF, Tf, Nothing, Td}(F, fl, nothing, df)
	end
end

(fl::Flow)(x, tspan)     = fl.flow(x, tspan)
(fl::Flow)(x, dx, tspan) = fl.dflow(x, dx, tspan)

"""
Creates a Flow variable based on a `prob::ODEProblem` and ODE solver `alg`. The vector field `F` has to be passed, this will be resolved in the future as it can be recovered from `prob`. Also, the derivative of the flow is estimated with finite differences.
"""
function Flow(F, prob::ODEProblem, alg; kwargs...)
	return Flow(F,
		(x, t) -> 				flow(x, (zero(t), t), prob; alg = alg, kwargs...),
		(x, t) ->			flowFull(x, (zero(t), t), prob; alg = alg, kwargs...),
		(x, dx, t) -> dflow_fd((x, dx), (zero(t), t), prob; alg = alg, kwargs...)
		)
end

# this constructor takes into accound a parameter passed to the vector field
function Flow(F, p, prob::ODEProblem, alg; kwargs...)
	return Flow(F,
		(x, t) ->			 	flow(x, p, (zero(t), t), prob; alg = alg, kwargs...),
		(x, t) -> 		 	flowFull(x, p, (zero(t), t), prob; alg = alg, kwargs...),
		(x, dx, t) -> dflow_fd((x, dx), p, (zero(t), t), prob; alg = alg, kwargs...)
		)
end

function Flow(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2; kwargs...)
	return Flow(F,
		(x, t) -> 			flow(x, p, (zero(t), t), prob1, alg = alg1; kwargs...),
		(x, t) -> 		flowFull(x, p, (zero(t), t), prob1, alg = alg1; kwargs...),
		(x, dx, t) ->  dflow(dx, x, p, (zero(t), t), prob2, alg = alg2; kwargs...)
		)
end

"""
Creates a Flow variable based on a `prob1::ODEProblem` and ODE solver `alg`. The derivative of the flow is imlemented in using `prob2::ODEProblem`. The vector field `F` has to be passed, this will be resolved in the future as it can be recovered from `prob`.
"""
function Flow(F, prob::ODEProblem, alg; kwargs...)
	return Flow(F,
		(x, t) ->			  flow(x, (zero(t), t), prob; alg = alg, kwargs...),
		(x, t) ->		  flowFull(x, (zero(t), t), prob; alg = alg, kwargs...),
		(x, dx, t) -> dflow_fd(x, dx, (zero(t), t), prob; alg = alg, kwargs...)
		)
end

# this constructor takes into accound a parameter passed to the vector field
function Flow(F, p, prob::ODEProblem, alg; kwargs...)
	return Flow(F,
		(x, t) ->			  flow(x, p, (zero(t), t), prob; alg = alg, kwargs...),
		(x, t) -> 		  flowFull(x, p, (zero(t), t), prob; alg = alg, kwargs...),
		(x, dx, t) -> dflow_fd(x, dx, p, (zero(t), t), prob; alg = alg, kwargs...)
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
where `F` is the vector field, `p` is a parameter (to be passed to the vector and the flow), `prob` is an `ODEProblem` which is used to create a flow using the ODE solver `alg` (for example `Tsit5()`). Finally, the arguments `kwargs` are passed to the ODE solver defining the flow. Look at `DifferentialEquations.jl` for more information. Note that, in this case, the derivative of the flow is computed internally using Finite Differences.
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

ShootingProblem(F, prob::ODEProblem, alg, ds, section; kwargs...) =  ShootingProblem(Flow(F, prob, alg; kwargs...), ds, section)

ShootingProblem(F, prob::ODEProblem, alg, M::Int, section; kwargs...) = ShootingProblem(Flow(F, prob, alg; kwargs...), diff(LinRange(0, 1, M + 1)), section)

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

	# we could use @views but then Sundials might complain
	sol = prob.flow.flowFull((xc[:, 1]), T)
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

This composite type implements the Poincaré Shooting method to locate periodic orbits, basically using Poincaré return maps. The arguments are as follows
- `flow::Flow`: implements the flow of the Cauchy problem though the structure `Flow`.
- `M`: the number of return maps. If `M==1`, then the simple shooting is implemented and the multiple one otherwise.
- `section`: implements a Poincaré section condition. The evaluation `section(x)` must return a scalar number where `x` is a guess for the periodic orbit when `M=1`. Otherwise, one must implement a function `section(out, x)` which populates `out` with the `M` hyperplanes.

## Simplified constructors
A simpler way is to create a functional is `pb = PoincareShootingProblem(F, p, prob::ODEProblem, alg, section; kwargs...)` for simple shooting or `pb = PoincareShootingProblem(F, p, prob::ODEProblem, alg, M::Int, section; kwargs...)` for multiple shooting . Here `F` is the vector field, `p` is a parameter (to be passed to the vector and the flow), `prob` is an `ODEProblem` which is used to create a flow using the ODE solver `alg` (for example `Tsit5()`). Finally, the arguments `kwargs` are passed to the ODE solver defining the flow. Look at `DifferentialEquations.jl` for more information.

Another convenient call is `pb = PoincareShootingProblem(F, p, prob::ODEProblem, alg, normals::AbstractVector, centers::AbstractVector; kwargs...)` where `normals` (resp. `centers`) is a list of normals (resp. centers) which define a list of hyperplanes `\\Sigma_i`. These hyperplanes are used to define partial Poincaré return maps. See docs for more information.

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
end

# simple Poincaré shooting
function PoincareShootingProblem(F, prob::ODEProblem, alg, section; interp_points = 50, kwargs...)
	pSection(u, t, integrator) = section(u) .* (integrator.iter > 1)
	# we put the nothing option to have an upcrossing
	cb = ContinuousCallback(pSection, terminate!, nothing, interp_points = interp_points)
	return PoincareShootingProblem(Flow(F, prob, alg, callback=cb; kwargs...), 1, section)
end

# this function takes into account a parameter passed to the vector field
# simple shooting
function PoincareShootingProblem(F, p, prob::ODEProblem, alg, section; interp_points = 50, kwargs...)
	pSection(u, t, integrator) = section(u) * (integrator.iter > 1)
	# we put nothing option to have an upcrossing
	cb = ContinuousCallback(pSection, terminate!, nothing, interp_points = interp_points)
	return PoincareShootingProblem(Flow(F, p, prob, alg; callback = cb, kwargs...), 1, section)
end

# this function takes into account a parameter passed to the vector field
# multiple shooting
function PoincareShootingProblem(F, p, prob::ODEProblem, alg, M, section; interp_points = 50, kwargs...)
	if M==1
		return PoincareShootingProblem(F, p, prob::ODEProblem, alg, section; kwargs...)
	end

	pSection(out, u, t, integrator) = section(out, u) .* (integrator.iter > 1)
	affect!(integrator, idx) = terminate!(integrator)
	# we put the nothing option to have an upcrossing
	cb = VectorContinuousCallback(pSection, affect!, M;interp_points = interp_points, affect_neg! = nothing)
	return PoincareShootingProblem(Flow(F, p, prob, alg; callback = cb, kwargs...), M, section)
end

function getPeriod(prob::PoincareShootingProblem, x)
	# this function extracts the period of the cycle
	@warn "Only done for M=1"
	prob.flow.flowFull(x, Inf64).t[end]
end

# Poincaré shooting functional
function (sh::PoincareShootingProblem)(x::AbstractVector)
	# period of the cycle
	M = sh.M
	N = div(length(x), M)

	# reshape the period orbit guess
	xc = reshape(x, N, M)

	# variable to hold the computed result
	out = similar(x)
	outc = reshape(out, N, M)

	for ii in 1:M-1
		# yi = Flow(xi) - x[i+1]
		# here we compute the flow up to an infinite time hoping we intersect a Poincare section before :D
		@views outc[:, ii] .= sh.flow(xc[:, ii], Inf64) .- xc[:, ii+1]
	end
	# ym = Flow(xm) - x1
	@views outc[:, M] .= sh.flow(xc[:, M], Inf64) .- xc[:, 1]

	return out
end

# jacobian of the shooting functional
function (sh::PoincareShootingProblem)(x::AbstractVector, dx::AbstractVector; δ = 1e-8)
	return (sh(x .+ δ .* dx) .- sh(x)) ./ δ
end
####################################################################################################
# Poincare shooting based on Sánchez, J., M. Net, B. Garcı́a-Archilla, and C. Simó. “Newton–Krylov Continuation of Periodic Orbits for Navier–Stokes Flows.” Journal of Computational Physics 201, no. 1 (November 20, 2004): 13–33. https://doi.org/10.1016/j.jcp.2004.04.018.

function sectionHyp(out, x, normals, centers)
	for ii = 1:length(normals)
		out[ii] = dot(normals[ii], x .- centers[ii])
	end
	out
end

# this composite type encodes a set of hyperplanes which are used as Poincaré sections
struct HyperplaneSections
	M				# number of hyperplanes
	normals 		# normals to define hyperplanes
	centers 		# representative point on each hyperplane
	indices 		# indices to be removed in the operator Ek

	function HyperplaneSections(normals, centers)
		M = length(normals)
		indices = zeros(Int64, M)
		for ii=1:M
			indices[ii] = argmax(abs.(normals[ii]))
		end
		return new(M, normals, centers, indices)
	end
end

function (hyp::HyperplaneSections)(out, u)
	sectionHyp(out, u, hyp.normals, hyp.centers)
end

# Operateur Rk from the paper above
function R!(hyp::HyperplaneSections, out, x::AbstractVector, ii::Int)
	k = hyp.indices[ii]
	@views out[1:k-1] .= x[1:k-1]
	@views out[k:end] .= x[k+1:end]
	return out
end

function R(hyp::HyperplaneSections, x::AbstractVector, ii::Int)
	out = similar(x, length(x) - 1)
	R!(hyp, out, x, ii)
end

# Operateur Ek from the paper above
function E!(hyp::HyperplaneSections, out, xbar::AbstractVector, ii::Int)
	@assert length(xbar) == length(hyp.normals[1]) - 1 "Wrong size for the projector / expansion operators, length(xbar) = $(length(xbar)) and length(normal) = $(length(hyp.normals[1]))"
	k = hyp.indices[ii]
	nbar = R(hyp, hyp.normals[ii], ii)
	xcbar = R(hyp, hyp.centers[ii], ii)
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

struct HyperplanePoincareShootingProblem <: AbstractShootingProblem
	psh::PoincareShootingProblem
end

getPeriod(hpsh::HyperplanePoincareShootingProblem, x) = getPeriod(hpsh.psh)

function PoincareShootingProblem(F, p, prob::ODEProblem, alg, normals::AbstractVector, centers::AbstractVector; interp_points = 50, kwargs...)
	hyp = HyperplaneSections(normals, centers)
	pSection(out, u, t, integrator) = hyp(out, u) * (integrator.iter > 1)
	affect!(integrator, idx) = terminate!(integrator)
	# we put nothing option to have an upcrossing
	cb = VectorContinuousCallback(pSection, affect!, hyp.M; interp_points = interp_points, affect_neg! = nothing)
	return HyperplanePoincareShootingProblem(PoincareShootingProblem(Flow(F, p, prob, alg; callback = cb, kwargs...), hyp.M, hyp))
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
		# yi = Flow(xi) - x[i+1]
		@views outc[:, ii] .= xc[:, ii+1].- sh.flow(xc[:, ii], Inf64)
	end
	# ym = Flow(xm) - x1
	@views outc[:, M] .= xc[:, 1] .- sh.flow(xc[:, M], Inf64)

	# build the array to be returned
	out_bar = similar(x_bar)
	out_barc = reshape(out_bar, Nm1, M)
	for ii=1:M-1
		R!(hpsh.psh.section, view(out_barc, :, ii), view(outc, :, ii), ii+1)
	end
	R!(hpsh.psh.section, view(out_barc, :, M), view(outc, :, M), 1)

	return out_bar
end

# jacobian of the shooting functional
function (sh::HyperplanePoincareShootingProblem)(x::AbstractVector, dx::AbstractVector; δ = 1e-8)
	return (sh(x .+ δ .* dx) .- sh(x)) ./ δ
end

####################################################################################################
# if we use the same code as for newton (see below) in continuation, it is difficult to tell the eigensolver not to use the jacobian but instead the monodromy matrix. So we have to use a dedicated composite type for the jacobian to handle this case.

struct ShootingJacobian{Tpb <: AbstractShootingProblem, Torbitguess}
	pb::Tpb
	x::Torbitguess
end

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

	if contParams.computeEigenValues
		contParams = @set contParams.newtonOptions.eigsolver = FloquetQaDShooting(contParams.newtonOptions.eigsolver)
	end

	if (prob(p0) isa PoincareShootingProblem) || (prob(p0) isa HyperplanePoincareShootingProblem)
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
