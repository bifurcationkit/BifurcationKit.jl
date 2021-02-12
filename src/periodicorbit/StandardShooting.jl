using DiffEqBase: EnsembleProblem, ODEProblem

"""
$(SIGNATURES)

Compute the amplitude of the periodic orbit associated to `x`. The keyword argument `ratio = 1` is used as follows. If `length(x) = ratio * n`, the call returns the amplitude over `x[1:n]`.
"""
function getAmplitude(prob::AbstractShootingProblem, x::AbstractVector, p; ratio = 1)
	_max = _getExtremum(prob, x, p; ratio = ratio)
	_min = _getExtremum(prob, x, p; ratio = ratio, op = (min, minimum))
	return maximum(_max .- _min)
end

"""
$(SIGNATURES)

Compute the maximum of the periodic orbit associated to `x`. The keyword argument `ratio = 1` is used as follows. If `length(x) = ratio * n`, the call returns the amplitude over `x[1:n]`.
"""
function getMaximum(prob::AbstractShootingProblem, x::AbstractVector, p; ratio = 1)
	mx = _getExtremum(prob, x, p; ratio = ratio)
	return maximum(mx)
end
####################################################################################################
# Standard Shooting functional
"""
	pb = ShootingProblem(flow::Flow, ds, section; parallel = false)

Create a problem to implement the Standard Simple / Parallel Multiple Standard Shooting method to locate periodic orbits. More details (maths, notations, linear systems) can be found [here](https://rveltz.github.io/BifurcationKit.jl/dev/periodicOrbitShooting/). The arguments are as follows
- `flow::Flow`: implements the flow of the Cauchy problem though the structure [`Flow`](@ref).
- `ds`: vector of time differences for each shooting. Its length is written `M`. If `M==1`, then the simple shooting is implemented and the multiple one otherwise.
- `section`: implements a phase condition. The evaluation `section(x, T)` must return a scalar number where `x` is a guess for **one point** the periodic orbit and `T` is the period of the guess. The type of `x` depends on what is passed to the newton solver. See [`SectionSS`](@ref) for a type of section defined as a hyperplane.
- `parallel` whether the shooting are computed in parallel (threading). Available through the use of Flows defined by `EnsembleProblem`.

A functional, hereby called `G`, encodes the shooting problem. For example, the following methods are available:

- `pb(orbitguess, par)` evaluates the functional G on `orbitguess`
- `pb(orbitguess, par, du; δ = 1e-9)` evaluates the jacobian `dG(orbitguess).du` functional at `orbitguess` on `du`. The optional argument `δ` is used to compute a finite difference approximation of the derivative of the section.
- `pb`(Val(:JacobianMatrixInplace), J, x, par)` compute the jacobian of the functional analytically. This is based on ForwardDiff.jl. Useful mainly for ODEs.
- `pb(Val(:JacobianMatrix), x, par)` same as above but out-of-place.

You can then call `pb(orbitguess, par)` to apply the functional to a guess. Note that `orbitguess::AbstractVector` must be of size M * N + 1 where N is the number of unknowns of the state space and `orbitguess[M * N + 1]` is an estimate of the period `T` of the limit cycle. This form of guess is convenient for the use of the linear solvers in `IterativeSolvers.jl` (for example) which accepts only `AbstractVector`s. Another accepted guess is of the form `BorderedArray(guess, T)` where `guess[i]` is the state of the orbit at the `i`th time slice. This last form allows for non-vector state space which can be convenient for 2d problems for example, use `GMRESKrylovKit` for the linear solver in this case.

## Simplified constructors
- A simpler way to build the functional is to use
	pb = ShootingProblem(F, p, prob::Union{ODEProblem, EnsembleProblem}, alg, centers::AbstractVector; kwargs...)
where `F(x,p)` is the vector field, `p` is a parameter (to be passed to the vector field and the flow), `prob` is an `ODEProblem` (resp. `EnsembleProblem`) which is used to create a flow using the ODE solver `alg` (for example `Tsit5()`). `centers` is list of `M` points close to the periodic orbit, they will be used to build a constraint for the phase. `parallel = false` is an option to use Parallel simulations (Threading) to simulate the multiple trajectories in the case of multiple shooting. This is efficient when the trajectories are relatively long to compute. Finally, the arguments `kwargs` are passed to the ODE solver defining the flow. Look at `DifferentialEquations.jl` for more information. Note that, in this case, the derivative of the flow is computed internally using Finite Differences.

- Another way to create a Shooting problem with more options is the following where in particular, one can provide its own scalar constraint `section(x)::Number` for the phase
	pb = ShootingProblem(F, p, prob::Union{ODEProblem, EnsembleProblem}, alg, M::Int, section; parallel = false, kwargs...)
or

	pb = ShootingProblem(F, p, prob::Union{ODEProblem, EnsembleProblem}, alg, ds, section; parallel = false, kwargs...)
- The next way is an elaboration of the previous one
	pb = ShootingProblem(F, p, prob1::Union{ODEProblem, EnsembleProblem}, alg1, prob2::Union{ODEProblem, EnsembleProblem}, alg2, M::Int, section; parallel = false, kwargs...)
or

	pb = ShootingProblem(F, p, prob1::Union{ODEProblem, EnsembleProblem}, alg1, prob2::Union{ODEProblem, EnsembleProblem}, alg2, ds, section; parallel = false, kwargs...)
where we supply now two `ODEProblem`s. The first one `prob1`, is used to define the flow associated to `F` while the second one is a problem associated to the derivative of the flow. Hence, `prob2` must implement the following vector field ``\\tilde F(x,y,p) = (F(x,p),dF(x,p)\\cdot y)``.
"""
@with_kw struct ShootingProblem{Tf <: Flow, Ts, Tsection} <: AbstractShootingProblem
	M::Int64 = 0							# number of sections
	flow::Tf = Flow()						# should be a Flow
	ds::Ts = diff(LinRange(0, 1, M + 1))	# difference of times for multiple shooting
	section::Tsection = nothing				# sections for phase condition
	parallel::Bool = false					# whether we use DE in Ensemble mode for multiple shooting
end

# this constructor takes into accound a parameter passed to the vector field
# if M = 1, we disable parallel processing
function ShootingProblem(F, p, prob::ODEProblem, alg, ds, section; parallel = false, kwargs...)
	_M = length(ds)
	parallel = _M == 1 ? false : parallel
	_pb = parallel ? EnsembleProblem(prob) : prob
	return ShootingProblem(M = _M, flow = Flow(F, p, _pb, alg; kwargs...),
			ds = ds, section = section, parallel = parallel)
end

ShootingProblem(F, p, prob::ODEProblem, alg, M::Int, section; parallel = false, kwargs...) = ShootingProblem(F, p, prob, alg, diff(LinRange(0, 1, M + 1)), section; parallel = parallel, kwargs...)

ShootingProblem(F, p, prob::ODEProblem, alg, centers::AbstractVector; parallel = false, kwargs...) = ShootingProblem(F, p, prob, alg, diff(LinRange(0, 1, length(centers) + 1)), SectionSS(F(centers[1], p)./ norm(F(centers[1], p)), centers[1]); parallel = parallel, kwargs...)

# this is the "simplest" constructor to use in automatic branching from Hopf
ShootingProblem(M::Int, par, prob::ODEProblem, alg; parallel = false, kwargs...) = ShootingProblem(nothing, par, prob, alg, M, nothing; parallel = parallel, kwargs...)

ShootingProblem(M::Int, par, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2; parallel = false, kwargs...) = ShootingProblem(nothing, par, prob1, alg1, prob2, alg2, M, nothing; parallel = parallel, kwargs...)

# idem but with an ODEproblem to define the derivative of the flow
function ShootingProblem(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2, ds, section; parallel = false, kwargs...)
	_M = length(ds)
	parallel = _M == 1 ? false : parallel
	_pb1 = parallel ? EnsembleProblem(prob1) : prob1
	_pb2 = parallel ? EnsembleProblem(prob2) : prob2
	return ShootingProblem(M = _M, flow = Flow(F, p, _pb1, alg1, _pb2, alg2; kwargs...), ds = ds, section = section, parallel = parallel)
end

ShootingProblem(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2, M::Int, section; parallel = false, kwargs...) = ShootingProblem(F, p, prob1, alg1, prob2, alg2, diff(LinRange(0, 1, M + 1)), section; parallel = parallel, kwargs...)

ShootingProblem(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2, centers::AbstractVector; parallel = false, kwargs...) = ShootingProblem(F, p, prob1, alg1, prob2, alg2, diff(LinRange(0, 1, length(centers) + 1)), SectionSS(F(centers[1], p)./ norm(F(centers[1], p)), centers[1]); parallel = parallel, kwargs...)

@inline isSimple(sh::ShootingProblem) = sh.M == 1
@inline isParallel(sh::ShootingProblem) = sh.parallel

# this function extracts the last component of the periodic orbit
extractPeriodShooting(x::AbstractVector) = x[end]
extractPeriodShooting(x::BorderedArray)  = x.p

# this function updates the section during the continuation run
function updateSection!(prob::ShootingProblem, x, par)
	xt = extractTimeSlices(x, getM(prob))
	@views update!(prob.section, prob.flow.F(xt[:, 1], par), xt[:, 1])
	prob.section.normal ./= norm(prob.section.normal)
	return true
end

"""
$(SIGNATURES)

Compute the period of the periodic orbit associated to `x`.
"""
@inline getPeriod(sh::ShootingProblem, x, par = nothing) = extractPeriodShooting(x)

function extractTimeSlices(x::AbstractVector, M::Int)
	N = div(length(x) - 1, M)
	return @views reshape(x[1:end-1], N, M)
end
extractTimeSlices(x::BorderedArray, M::Int) = x.u

@inline extractTimeSlice(x::AbstractMatrix, ii::Int) = @view x[:, ii]
@inline extractTimeSlice(x::AbstractVector, ii::Int) = xc[ii]

####################################################################################################
# Standard shooting functional using AbstractVector, convenient for IterativeSolvers.
function (sh::ShootingProblem)(x::AbstractVector, par)
	# Sundials does not like @views :(
	T = extractPeriodShooting(x)
	M = getM(sh)
	N = div(length(x) - 1, M)

	# extract the orbit guess and reshape it into a matrix as it's more convenient to handle it
	xc = extractTimeSlices(x, M)

	# variable to hold the computed result
	out = similar(x)
	outc = extractTimeSlices(out, M)

	if ~isParallel(sh)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			# we can use views but Sundials will complain
			outc[:, ii] .= sh.flow(xc[:, ii], par, sh.ds[ii] * T) .- xc[:, ip1]
		end
	else
		solOde = sh.flow(xc, par, sh.ds .* T)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			# we can use views but Sundials will complain
			outc[:, ii] .= @views solOde[ii][2] .- xc[:, ip1]
		end
	end

	# add constraint
	out[end] = @views sh.section(xc[:, 1], T)

	return out
end

# shooting functional, this allows for AbstractArray state space
function (sh::ShootingProblem)(x::BorderedArray, par)
	# period of the cycle
	T = extractPeriodShooting(x)
	M = getM(sh)

	# extract the orbit guess and reshape it into a matrix as it's more convenient to handle it
	xc = extractTimeSlices(x, M)

	# variable to hold the computed result
	out = similar(x)

	if ~isParallel(sh)
		for ii in 1:M
			# we can use views but Sundials will complain
			ip1 = (ii == M) ? 1 : ii+1
			out.u[ii] .= sh.flow(xc[ii], par, sh.ds[ii] * T) .- xc[ip1]
		end
	else
		@assert 1==0 "Not implemented yet. Try to use AbstractVectors instead"
	end

	# add constraint
	out.p = sh.section(xc[1], T)

	return out
end

# jacobian of the shooting functional
function (sh::ShootingProblem)(x::AbstractVector, par, dx::AbstractVector; δ = 1e-9)
	# period of the cycle
	# Sundials does not like @views :(
	dT = extractPeriodShooting(dx)
	T  = extractPeriodShooting(x)
	M = getM(sh)
	N = div(length(x) - 1, M)

	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)

	dxv = @view dx[1:end-1]
	dxc = reshape(dxv, N, M)

	# variable to hold the computed result
	out = similar(x)
	outv = @view out[1:end-1]
	outc = reshape(outv, N, M)

	if ~isParallel(sh)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			# call jacobian of the flow
			tmp = sh.flow(xc[:, ii], par, dxc[:, ii], sh.ds[ii] * T)
			outc[:, ii] .= @views tmp.du .+ sh.flow.F(tmp.u, par) .* sh.ds[ii] .* dT .- dxc[:, ip1]
		end
	else
		# call jacobian of the flow
		solOde = sh.flow(xc, par, dxc, sh.ds .* T)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			outc[:, ii] .= solOde[ii].du .+ sh.flow.F(solOde[ii].u, par) .* sh.ds[ii] .* dT .- dxc[:, ip1]
		end
	end

	# add constraint
	out[end] = @views (sh.section(x[1:N] .+ δ .* dx[1:N], T + δ * dT ) - sh.section(x[1:N], T)) / δ

	return out
end

# jacobian of the shooting functional, this allows for Array state space
function (sh::ShootingProblem)(x::BorderedArray, par, dx::BorderedArray; δ = 1e-9)
	dT = extractPeriodShooting(dx)
	T  = extractPeriodShooting(x)
	M = getM(sh)

	# variable to hold the computed result
	out = BorderedArray{typeof(x.u), typeof(x.p)}(similar(x.u), typeof(x.p)(0))

	if ~isParallel(sh)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			# call jacobian of the flow
			tmp = sh.flow(x.u[ii], par, dx.u[ii], sh.ds[ii] * T)
			out.u[ii] .= tmp.du .+ sh.flow.F(tmp.u, par) .* sh.ds[ii] .* dT .- dx.u[ip1]
		end
	else
		@assert 1==0 "Not implemented yet. Try using AbstractVectors instead"
	end

	# add constraint
	x_tmp = similar(x.u); copyto!(x_tmp, x.u)
	axpy!(δ , dx.u, x_tmp)
	out.p = (sh.section(BorderedArray(x_tmp, T + δ * dT), T + δ * dT ) - sh.section(x, T)) / δ

	return out
end

# inplace computation of the matrix of the jacobian of the shooting problem, only serial for now
function (sh::ShootingProblem)(::Val{:JacobianMatrixInplace}, J::AbstractMatrix, x::AbstractVector, par)
	T = extractPeriodShooting(x)
	M = getM(sh)
	N = div(length(x) - 1, M)

	# extract the orbit guess and reshape it into a matrix as it's more convenient to handle it
	xc = extractTimeSlices(x, M)

	# jacobian of the flow
	dflow = (_J, _x, _T) -> ForwardDiff.jacobian!(_J, z -> sh.flow(Val(:SerialTimeSol), z, par, _T).u, _x)

	# put the matrices by blocks
	In = I(N)
	for ii=1:M
		@views dflow(J[(ii-1)*N+1:(ii-1)*N+N, (ii-1)*N+1:(ii-1)*N+N], xc[:, ii], sh.ds[ii] * T)
		# we put the identity matrices
		ip1 = (ii == M) ? 1 : ii+1
		if M == 1
			J[(ii-1)*N+1:(ii-1)*N+N, (ip1-1)*N+1:(ip1-1)*N+N] .+= -In
		else
			J[(ii-1)*N+1:(ii-1)*N+N, (ip1-1)*N+1:(ip1-1)*N+N] .= -In
		end
		# we fill the last column
		tmp = @views sh.flow(Val(:SerialTimeSol), xc[:, ii], par, sh.ds[ii] * T).u
		J[(ii-1)*N+1:(ii-1)*N+N, end] .= sh.flow.F(tmp, par) .* sh.ds[ii]
	end

	# we fill the last row
	@views ForwardDiff.gradient!(J[end, 1:N], z -> sh.section(z, T), x[1:N])
	J[end, end] = @views ForwardDiff.derivative(z -> sh.section(x[1:N], z), T)

	return J
end

# out of place version
(sh::ShootingProblem)(::Val{:JacobianMatrix}, x::AbstractVector, par) = sh(Val(:JacobianMatrixInplace), zeros(length(x), length(x)), x, par)
####################################################################################################

function _getExtremum(prob::ShootingProblem, x::AbstractVector, p; ratio = 1, op = (max, maximum))
	# this function extracts the amplitude of the cycle
	T = extractPeriodShooting(x)
	M = getM(prob)
	N = div(length(x) - 1, M)
	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)
	Th = eltype(x)
	n = div(N, ratio)

	# !!!! we could use @views but then Sundials will complain !!!
	if ~isParallel(prob)
		sol = prob.flow(Val(:Full), xc[:, 1], p, T)
		mx = @views op[2](sol[1:n, :], dims = 1)
	else # threaded version
		sol = prob.flow(Val(:Full), xc, p, prob.ds .* T)
		mx = op[2](sol[1][1:n, :] , dims = 2)
		for ii = 2:M
			mx = op[1].(mx, op[2](sol[ii][1:n, :], dims = 2))
		end
	end
	return mx
end

"""
$(SIGNATURES)

Compute the full trajectory associated to `x`. Mainly for plotting purposes.
"""
function getTrajectory(prob::ShootingProblem, x::AbstractVector, p)
	T = extractPeriodShooting(x)
	M = getM(prob)
	N = div(length(x) - 1, M)
	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)
	Th = eltype(x)

	# !!!! we could use @views but then Sundials will complain !!!
	if ~isParallel(prob)
		return prob.flow(Val(:Full), xc[:, 1], p, T)
	else # threaded version
		sol = prob.flow(Val(:Full), xc[:, 1:1], p, [T])
		return sol[1]
	end
end


####################################################################################################
# functions needed for Branch switching from Hopf bifurcation point
function problemForBS(prob::ShootingProblem, F, dF, par, hopfpt, ζr, orbitguess_a, period)
	# append period at the end of the initial guess
	orbitguess_v = reduce(vcat, orbitguess_a)
	orbitguess = vcat(vec(orbitguess_v), period) |> vec

	# update the problem but not the section if the user passed one
	probSh = setproperties(prob, section = isnothing(prob.section) ? SectionSS(F(orbitguess_a[1], hopfpt.params), copy(orbitguess_a[1])) : prob.section)
	probSh.section.normal ./= norm(probSh.section.normal)

	# be sure that the vector field is correctly inplace in the Flow structure
	probSh = @set probSh.flow.F = F

	return probSh, 	orbitguess
end
