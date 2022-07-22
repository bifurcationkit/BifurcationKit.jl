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

Create a problem to implement the Standard Simple / Parallel Multiple Standard Shooting method to locate periodic orbits. More details (maths, notations, linear systems) can be found [here](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/periodicOrbitShooting/). The arguments are as follows
- `flow::Flow`: implements the flow of the Cauchy problem though the structure [`Flow`](@ref).
- `ds`: vector of time differences for each shooting. Its length is written `M`. If `M == 1`, then the simple shooting is implemented and the multiple one otherwise.
- `section`: implements a phase condition. The evaluation `section(x, T)` must return a scalar number where `x` is a guess for **one point** on the periodic orbit and `T` is the period of the guess. Also, the method `section(x, T, dx, dT)` must be available and which returns the differential of `section`. The type of `x` depends on what is passed to the newton solver. See [`SectionSS`](@ref) for a type of section defined as a hyperplane.
- `parallel` whether the shooting is computed in parallel (threading). Available through the use of Flows defined by `EnsembleProblem` (this is automatically set up for you).
- `par` parameters of the model
- `lens` parameter axis
- `updateSectionEveryStep` updates the section every `updateSectionEveryStep` step during continuation
- `jacobian::Symbol` symbol which describes the type of jacobian used in Newton iterations (see below).

A functional, hereby called `G`, encodes the shooting problem. For example, the following methods are available:

- `pb(orbitguess, par)` evaluates the functional G on `orbitguess`
- `pb(orbitguess, par, du; δ = 1e-9)` evaluates the jacobian `dG(orbitguess)⋅du` functional at `orbitguess` on `du`. The optional argument `δ` is used to compute a finite difference approximation of the derivative of the section.
- `pb`(Val(:JacobianMatrixInplace), J, x, par)` compute the jacobian of the functional analytically. This is based on ForwardDiff.jl. Useful mainly for ODEs.
- `pb(Val(:JacobianMatrix), x, par)` same as above but out-of-place.

You can then call `pb(orbitguess, par)` to apply the functional to a guess. Note that `orbitguess::AbstractVector` must be of size `M * N + 1` where N is the number of unknowns of the state space and `orbitguess[M * N + 1]` is an estimate of the period `T` of the limit cycle. This form of guess is convenient for the use of the linear solvers in `IterativeSolvers.jl` (for example) which only accept `AbstractVector`s. Another accepted guess is of the form `BorderedArray(guess, T)` where `guess[i]` is the state of the orbit at the `i`th time slice. This last form allows for non-vector state space which can be convenient for 2d problems for example, use `GMRESKrylovKit` for the linear solver in this case.

Note that you can generate this guess from a function solution using `generateSolution`.

## Jacobian
$DocStrjacobianPOSh

## Simplified constructors
- A simpler way to build the functional is to use
	pb = ShootingProblem(prob::Union{ODEProblem, EnsembleProblem}, alg, centers::AbstractVector; kwargs...)
where `prob` is an `ODEProblem` (resp. `EnsembleProblem`) which is used to create a flow using the ODE solver `alg` (for example `Tsit5()`). `centers` is list of `M` points close to the periodic orbit, they will be used to build a constraint for the phase. `parallel = false` is an option to use Parallel simulations (Threading) to simulate the multiple trajectories in the case of multiple shooting. This is efficient when the trajectories are relatively long to compute. Finally, the arguments `kwargs` are passed to the ODE solver defining the flow. Look at `DifferentialEquations.jl` for more information. Note that, in this case, the derivative of the flow is computed internally using Finite Differences.

- Another way to create a Shooting problem with more options is the following where in particular, one can provide its own scalar constraint `section(x)::Number` for the phase
	pb = ShootingProblem(F, p, prob::Union{ODEProblem, EnsembleProblem}, alg, M::Int, section; parallel = false, kwargs...)
or

	pb = ShootingProblem(prob::Union{ODEProblem, EnsembleProblem}, alg, ds, section; parallel = false, kwargs...)
- The next way is an elaboration of the previous one
	pb = ShootingProblem(prob1::Union{ODEProblem, EnsembleProblem}, alg1, prob2::Union{ODEProblem, EnsembleProblem}, alg2, M::Int, section; parallel = false, kwargs...)
or

	pb = ShootingProblem(prob1::Union{ODEProblem, EnsembleProblem}, alg1, prob2::Union{ODEProblem, EnsembleProblem}, alg2, ds, section; parallel = false, kwargs...)
where we supply now two `ODEProblem`s. The first one `prob1`, is used to define the flow associated to `F` while the second one is a problem associated to the derivative of the flow. Hence, `prob2` must implement the following vector field ``\\tilde F(x,y,p) = (F(x,p), dF(x,p)\\cdot y)``.
"""
@with_kw_noshow struct ShootingProblem{Tf <: AbstractFlow, Ts, Tsection, Tpar, Tlens} <: AbstractShootingProblem
	M::Int64 = 0							# number of sections
	flow::Tf = Flow()						# should be a Flow
	ds::Ts = diff(LinRange(0, 1, M + 1))	# difference of times for multiple shooting
	section::Tsection = nothing				# sections for phase condition
	parallel::Bool = false					# whether we use DE in Ensemble mode for multiple shooting
	par::Tpar = nothing
	lens::Tlens = nothing
	updateSectionEveryStep::Int = 0
	jacobian::Symbol = :autodiffDense
end

@inline isSimple(sh::ShootingProblem) = getMeshSize(sh) == 1
@inline isParallel(sh::ShootingProblem) = sh.parallel
@inline getLens(sh::ShootingProblem) = sh.lens
getParams(prob::ShootingProblem) = prob.par

function Base.show(io::IO, sh::ShootingProblem)
	println(io, "┌─ Standard shooting functional for periodic orbits")
	println(io, "├─ time slices    : ", getMeshSize(sh))
	println(io, "├─ lens           : ", getLensSymbol(sh.lens))
	println(io, "├─ jacobian       : ", sh.jacobian)
	println(io, "├─ update section : ", sh.updateSectionEveryStep)
	if sh.flow isa FlowDE
		println(io, "├─ integrator     : ", typeof(sh.flow.alg).name.name)
	end
	println(io, "└─ parallel       : ", isParallel(sh))
end

# this function updates the section during the continuation run
function updateSection!(sh::ShootingProblem, x, par)
	xt = getTimeSlices(sh, x)
	@views update!(sh.section, vf(sh.flow, xt[:, 1], par), xt[:, 1])
	sh.section.normal ./= norm(sh.section.normal)
	return true
end

@views function getTimeSlices(sh::ShootingProblem, x::AbstractVector)
	M = getMeshSize(sh)
	N = div(length(x) - 1, M)
	return reshape(x[1:end-1], N, M)
end
getTimeSlices(::ShootingProblem ,x::BorderedArray) = x.u

@inline getTimeSlice(::ShootingProblem, x::AbstractMatrix, ii::Int) = @view x[:, ii]
@inline getTimeSlice(::ShootingProblem, x::AbstractVector, ii::Int) = xc[ii]
####################################################################################################
# Standard shooting functional using AbstractVector, convenient for IterativeSolvers.
function (sh::ShootingProblem)(x::AbstractVector, par)
	# Sundials does not like @views :(
	T = getPeriod(sh, x)
	M = getMeshSize(sh)
	N = div(length(x) - 1, M)

	# extract the orbit guess and reshape it into a matrix as it's more convenient to handle it
	xc = getTimeSlices(sh, x)

	# variable to hold the computed result
	out = similar(x)
	outc = getTimeSlices(sh, out)

	if ~isParallel(sh)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			# we can use views but Sundials will complain
			outc[:, ii] .= evolve(sh.flow, xc[:, ii], par, sh.ds[ii] * T).u .- xc[:, ip1]
		end
	else
		solOde = evolve(sh.flow, xc, par, sh.ds .* T)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			# we can use views but Sundials will complain
			outc[:, ii] .= @views solOde[ii][2] .- xc[:, ip1]
		end
	end

	# add constraint
	out[end] = @views sh.section(getTimeSlice(sh, xc, 1), T)
	return out
end

# shooting functional, this allows for AbstractArray state space
function (sh::ShootingProblem)(x::BorderedArray, par)
	# period of the cycle
	T = getPeriod(sh, x)
	M = getMeshSize(sh)

	# extract the orbit guess
	xc = getTimeSlices(sh, x)

	# variable to hold the computed result
	out = similar(x)

	if ~isParallel(sh)
		for ii in 1:M
			# we can use views but Sundials will complain
			ip1 = (ii == M) ? 1 : ii+1
			out.u[ii] .= evolve(sh.flow, xc[ii], par, sh.ds[ii] * T).u .- xc[ip1]
		end
	else
		@assert 1==0 "Not implemented yet. Try to use an AbstractVector instead"
	end

	# add constraint
	out.p = sh.section(getTimeSlice(sh, xc, 1), T)
	return out
end

# jacobian of the shooting functional
function (sh::ShootingProblem)(x::AbstractVector, par, dx::AbstractVector; δ =convert(eltype(x), 1e-8))
	# period of the cycle
	# Sundials does not like @views :(
	dT = getPeriod(sh, dx)
	T  = getPeriod(sh, x)
	M  = getMeshSize(sh)

	xc = getTimeSlices(sh, x)
	dxc = getTimeSlices(sh, dx)

	# variable to hold the computed result
	out = similar(x)
	outc = getTimeSlices(sh, out)

	if ~isParallel(sh)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			# call jacobian of the flow
			tmp = evolve(sh.flow, xc[:, ii], par, dxc[:, ii], sh.ds[ii] * T)
			outc[:, ii] .= @views tmp.du .+ vf(sh.flow, tmp.u, par) .* sh.ds[ii] .* dT .- dxc[:, ip1]
		end
	else
		# call jacobian of the flow
		solOde = evolve(sh.flow, xc, par, dxc, sh.ds .* T)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			outc[:, ii] .= solOde[ii].du .+ vf(sh.flow, solOde[ii].u, par) .* sh.ds[ii] .* dT .- dxc[:, ip1]
		end
	end

	# add constraint
	N = div(length(x) - 1, M)
	out[end] = @views sh.section(x[1:N], T, dx[1:N], dT)
	return out
end

# jacobian of the shooting functional, this allows for Array state space
function (sh::ShootingProblem)(x::BorderedArray, par, dx::BorderedArray; δ = convert(eltype(x.u), 1e-8))
	dT = getPeriod(sh, dx)
	T  = getPeriod(sh, x)
	M  = getMeshSize(sh)

	# variable to hold the computed result
	out = BorderedArray{typeof(x.u), typeof(x.p)}(similar(x.u), typeof(x.p)(0))

	if ~isParallel(sh)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			# call jacobian of the flow
			tmp = evolve(sh.flow, x.u[ii], par, dx.u[ii], sh.ds[ii] * T)
			out.u[ii] .= tmp.du .+ sh.flow.F(tmp.u, par) .* sh.ds[ii] .* dT .- dx.u[ip1]
		end
	else
		@assert 1==0 "Not implemented yet. Try using AbstractVectors instead"
	end

	# add constraint
	out.p = sh.section(x.u[1], T, dx.u[1], dT)
	return out
end

# inplace computation of the matrix of the jacobian of the shooting problem, only serial for now
function (sh::ShootingProblem)(::Val{:JacobianMatrixInplace}, J::AbstractMatrix, x::AbstractVector, par)
	T = getPeriod(sh, x)
	M = getMeshSize(sh)
	N = div(length(x) - 1, M)

	# extract the orbit guess and reshape it into a matrix as it's more convenient to handle it
	xc = getTimeSlices(sh, x)

	# jacobian of the flow
	dflow = (_J, _x, _T) -> ForwardDiff.jacobian!(_J, z -> evolve(sh.flow, Val(:SerialTimeSol), z, par, _T).u, _x)

	# put the matrices by blocks
	In = I(N)
	for ii in 1:M
		@views dflow(J[(ii-1)*N+1:(ii-1)*N+N, (ii-1)*N+1:(ii-1)*N+N], xc[:, ii], sh.ds[ii] * T)
		# we put the identity matrices
		ip1 = (ii == M) ? 1 : ii+1
		if M == 1
			J[(ii-1)*N+1:(ii-1)*N+N, (ip1-1)*N+1:(ip1-1)*N+N] .+= -In
		else
			J[(ii-1)*N+1:(ii-1)*N+N, (ip1-1)*N+1:(ip1-1)*N+N]  .= -In
		end
		# we fill the last column
		tmp = @views evolve(sh.flow, Val(:SerialTimeSol), xc[:, ii], par, sh.ds[ii] * T).u
		J[(ii-1)*N+1:(ii-1)*N+N, end] .= vf(sh.flow, tmp, par) .* sh.ds[ii]
	end

	# we fill the last row
	@views ForwardDiff.gradient!(J[end, 1:N], z -> sh.section(z, T), x[1:N])
	J[end, end] = @views ForwardDiff.derivative(z -> sh.section(x[1:N], z), T)

	return J
end

# out of place version
(sh::ShootingProblem)(::Val{:JacobianMatrix}, x::AbstractVector, par) = sh(Val(:JacobianMatrixInplace), zeros(eltype(x), length(x), length(x)), x, par)
####################################################################################################

function _getExtremum(prob::ShootingProblem, x::AbstractVector, p; ratio = 1, op = (max, maximum))
	# this function extracts the amplitude of the cycle
	T = getPeriod(prob, x)
	M = getMeshSize(prob)
	N = div(length(x) - 1, M)
	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)
	Th = eltype(x)
	n = div(N, ratio)

	# !!!! we could use @views but then Sundials will complain !!!
	if ~isParallel(prob)
		sol = evolve(prob.flow, Val(:Full), xc[:, 1], p, T)
		mx = @views op[2](sol[1:n, :], dims = 1)
	else # threaded version
		sol = evolve(prob.flow, Val(:Full), xc, p, prob.ds .* T)
		mx = op[2](sol[1][1:n, :] , dims = 2)
		for ii in 2:M
			mx = op[1].(mx, op[2](sol[ii][1:n, :], dims = 2))
		end
	end
	return mx
end

"""
$(SIGNATURES)

Compute the full periodic orbit associated to `x`. Mainly for plotting purposes.
"""
function getPeriodicOrbit(prob::ShootingProblem, x::AbstractVector, par)
	T = getPeriod(prob, x)
	M = getMeshSize(prob)
	N = div(length(x) - 1, M)
	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)
	Th = eltype(x)

	# !!!! we could use @views but then Sundials will complain !!!
	if ~isParallel(prob)
		sol = [evolve(prob.flow, Val(:Full), xc[:, ii], par, prob.ds[ii] * T) for ii in 1:M]
		time = sol[1].t; u = sol[1][:,:]
		for ii in 2:M
			append!(time, sol[ii].t .+ time[end])
			u = hcat(u, sol[ii][:,:])
		end
		return SolPeriodicOrbit(t = time, u = u)

	else # threaded version
		sol = evolve(prob.flow, Val(:Full), xc, par, prob.ds .* T)
		time = sol[1].t; u = sol[1][:,:]
		for ii in 2:M
			append!(time, sol[ii].t .+ time[end])
			u = hcat(u, sol[ii][:,:])
		end
		return SolPeriodicOrbit(t = time, u = u)
	end
end

####################################################################################################
# functions needed for Branch switching from Hopf bifurcation point
function reMake(prob::ShootingProblem, prob_vf, hopfpt, ζr, orbitguess_a, period; k...)
	# append period at the end of the initial guess
	orbitguess_v = reduce(vcat, orbitguess_a)
	orbitguess = vcat(vec(orbitguess_v), period) |> vec

	# update the problem but not the section if the user passed one
	probSh = setproperties(prob, section = isnothing(prob.section) ? SectionSS(residual(prob_vf, orbitguess_a[1], hopfpt.params), copy(orbitguess_a[1])) : prob.section, par = getParams(prob_vf), lens = getLens(prob_vf))
	probSh.section.normal ./= norm(probSh.section.normal)

	# be sure that the vector field is correctly inplace in the Flow structure
	# @set! probSh.flow.F = prob_vf.VF.F

	return probSh, orbitguess
end

function predictor(pb::ShootingProblem, bifpt, ampfactor, ζs, bptype::Symbol)
	@assert bptype in (:bp, :pd)
	Mv = getMeshSize(pb)
	# plot(reshape(orbitguess[1:end-1],2,Mv)')
	if bptype == :bp
		orbitguess = copy(bifpt.x)
		orbitguess[1:length(ζs)] .+= ampfactor .* ζs

		# plot(cumsum(pb.ds) .* orbitguess[end], reshape(orbitguess[1:end-1],3, pb.M)') |> display

		# plot!(cumsum(pb.ds) .* orbitguess[end], reshape(bifpt.x[1:end-1],3, pb.M)', linewidth = 4) |> display
	elseif bptype == :pd
		orbitguess = copy(bifpt.x)[1:end-1] .+ ampfactor .* ζs
		orbitguess =
			vcat(orbitguess, copy(bifpt.x)[1:end-1] .- ampfactor .* ζs, bifpt.x[end])
		if 	pb isa ShootingProblem
			@set! pb.M = 2pb.M
			# @show pb.ds cumsum(pb.ds)
			@set! pb.ds = _duplicate(pb.ds) ./ 2
			# @show pb.ds cumsum(pb.ds)
			orbitguess[end] *= 2
			# plot(cumsum(pb.ds) .* orbitguess[end], reshape(orbitguess[1:end-1],3, pb.M)', marker = :d) |> display
		end
	end
	return pb, orbitguess
end
