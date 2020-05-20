using DiffEqBase

function getAmplitude(prob::AbstractShootingProblem, x::AbstractVector, p; ratio = 1)
	mx = _getMax(prob, x, p; ratio = ratio)
	return maximum(mx) - minimum(mx)
end

function getMaximum(prob::AbstractShootingProblem, x::AbstractVector, p; ratio = 1)
	mx = _getMax(prob, x, p; ratio = ratio)
	return maximum(mx)
end

####################################################################################################
@views function sectionShooting(x::AbstractVector, normals::AbstractVector, centers::AbstractVector)
	res = eltype(x)(1)
	M = length(centers)
	N = div(length(x)-1, M)
	xv = x[1:end-1]
	xc = reshape(xv, N, M)

	for ii in 1:M
		# this avoids the temporary xc - centers
		res *= dot(xc[:, ii], normals[ii]) - dot(centers[ii], normals[ii])
	end
	res
end

# section for Standard Shooting
struct SectionSS{Tn, Tc}
	normals::Tn 	# normals to define hyperplanes
	centers::Tc 	# representative point on each hyperplane
end

(sect::SectionSS)(u) = sectionShooting(u, sect.normals, sect.centers)

# we update the field of Section, useful during continuation procedure for updating the section
function update!(sect::SectionSS, normals, centers)
	copyto!(sect.normals, normals)
	copyto!(sect.centers, centers)
	sect
end
####################################################################################################
# Standard Shooting functional
"""
	pb = ShootingProblem(flow::Flow, ds, section; isparallel = false)

This composite type implements the Standard Simple / Parallel Multiple Standard Shooting method to locate periodic orbits. The arguments are as follows
- `flow::Flow`: implements the flow of the Cauchy problem though the structure [`Flow`](@ref).
- `ds`: vector of time differences for each shooting. Its length is written `M`. If `M==1`, then the simple shooting is implemented and the multiple one otherwise.
- `section`: implements a phase condition. The evaluation `section(x)` must return a scalar number where `x` is a guess for the periodic orbit. Note that the period `T` of the guess `x` is always included either as the last component of `T = x[end]` or as `T = x.p`. The type of `x` depends on what is passed to the newton solver.
- `isparallel` whether the shooting are computed in parallel (threading). Only available through the use of Flows defined by `EnsembleProblem`.

You can then call `pb(orbitguess, par)` to apply the functional to a guess. Note that `orbitguess::AbstractVector` must be of size M * N + 1 where N is the number of unknowns of the state space and `orbitguess[M * N + 1]` is an estimate of the period `T` of the limit cycle. This form of guess is convenient for the use of the linear solvers in `IterativeSolvers.jl` (for example) which accepts only `AbstractVector`s. Another accepted guess is of the form `BorderedArray(guess, T)` where `guess[i]` is the state of the orbit at the `i`th time slice. This last form allows for non-vector state space which can be convenient for 2d problems for example, use `GMRESKrylovKit` for the linear solver in this case.

A functional, hereby called `G`, encodes the shooting problem. For example, the following methods are available:
- `pb(orbitguess, par)` evaluates the functional G on `orbitguess`
- `pb(orbitguess, par, du)` evaluates the jacobian `dG(orbitguess).du` functional at `orbitguess` on `du`

## Simplified constructors
- A simpler way to build a functional is to use
	pb = ShootingProblem(F, p, prob::Union{ODEProblem, EnsembleProblem}, alg, centers::AbstractVector; kwargs...)
where `F(x,p)` is the vector field, `p` is a parameter (to be passed to the vector field and the flow), `prob` is an `ODEProblem` (resp. `EnsembleProblem`) which is used to create a flow using the ODE solver `alg` (for example `Tsit5()`). `centers` is list of `M` points close to the periodic orbit, they will be used to build a constraint for the phase. `isparallel = false` is an option to use Parallel simulations (Threading) to simulate the multiple trajectories in the case of multiple shooting. This is efficient when the trajectories are relatively long to compute. Finally, the arguments `kwargs` are passed to the ODE solver defining the flow. Look at `DifferentialEquations.jl` for more information. Note that, in this case, the derivative of the flow is computed internally using Finite Differences.

- Another way with more options is the following where in particular, one can provide its own scalar constraint `section(x)::Number` for the phase
	pb = ShootingProblem(F, p, prob::Union{ODEProblem, EnsembleProblem}, alg, M::Int, section; isparallel = false, kwargs...)
or

	pb = ShootingProblem(F, p, prob::Union{ODEProblem, EnsembleProblem}, alg, ds, section; isparallel = false, kwargs...)
- The next way is an elaboration of the previous one
	pb = ShootingProblem(F, p, prob1::Union{ODEProblem, EnsembleProblem}, alg1, prob2::Union{ODEProblem, EnsembleProblem}, alg2, M::Int, section; isparallel = false, kwargs...)
or

	pb = ShootingProblem(F, p, prob1::Union{ODEProblem, EnsembleProblem}, alg1, prob2::Union{ODEProblem, EnsembleProblem}, alg2, ds, section; isparallel = false, kwargs...)
where we supply now two `ODEProblem`s. The first one `prob1`, is used to define the flow associated to `F` while the second one is a problem associated to the derivative of the flow. Hence, `prob2` must implement the following vector field ``\\tilde F(x,y,p) = (F(x,p),dF(x,p)\\cdot y)``.
"""
@with_kw struct ShootingProblem{Tf <: Flow, Ts, Tsection} <: AbstractShootingProblem
	M::Int64 = 0						# number of sections
	flow::Tf = Flow()					# should be a Flow{TF, Tf, Td}
	ds::Ts = diff(LinRange(0, 1, 5))	# difference of times for multiple shooting
	section::Tsection = nothing			# sections for phase condition
	isparallel::Bool = false			# whether we use DE in Ensemble mode for multiple shooting
end

# this constructor takes into accound a parameter passed to the vector field
# if M = 1, we disable parallel processing
function ShootingProblem(F, p, prob::ODEProblem, alg, ds, section; isparallel = false, kwargs...)
	_M = length(ds)
	isparallel = _M == 1 ? false : isparallel
	_pb = isparallel ? EnsembleProblem(prob) : prob
	return ShootingProblem(M = _M, flow = Flow(F, p, _pb, alg; kwargs...),
			ds = ds, section = section, isparallel = isparallel)
end

ShootingProblem(F, p, prob::ODEProblem, alg, M::Int, section; isparallel = false, kwargs...) = ShootingProblem(F, p, prob, alg, diff(LinRange(0, 1, M + 1)), section; isparallel = isparallel, kwargs...)

ShootingProblem(F, p, prob::ODEProblem, alg, centers::AbstractVector; isparallel = false, kwargs...) = ShootingProblem(F, p, prob, alg, diff(LinRange(0, 1, length(centers) + 1)), SectionSS([F(c, p) for c in centers], centers); isparallel = isparallel, kwargs...)

# this is the "simplest" constructor to use in automatic branching from Hopf
ShootingProblem(M::Int, par, prob::ODEProblem, alg; isparallel = false, kwargs...) = ShootingProblem(nothing, par, prob, alg, M, nothing; isparallel = isparallel, kwargs...)

ShootingProblem(M::Int, par, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2; isparallel = false, kwargs...) = ShootingProblem(nothing, par, prob1, alg1, prob2, alg2, M, nothing; isparallel = isparallel, kwargs...)

# idem but with an ODEproblem to define the derivative of the flow
function ShootingProblem(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2, ds, section; isparallel = false, kwargs...)
	_M = length(ds)
	isparallel = _M == 1 ? false : isparallel
	_pb1 = isparallel ? EnsembleProblem(prob1) : prob1
	_pb2 = isparallel ? EnsembleProblem(prob2) : prob2
	return ShootingProblem(M = _M, flow = Flow(F, p, _pb1, alg1, _pb2, alg2; kwargs...), ds = ds, section = section, isparallel = isparallel)
end

ShootingProblem(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2, M::Int, section; isparallel = false, kwargs...) = ShootingProblem(F, p, prob1, alg1, prob2, alg2, diff(LinRange(0, 1, M + 1)), section; isparallel = isparallel, kwargs...)

ShootingProblem(F, p, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2, centers::AbstractVector; isparallel = false, kwargs...) = ShootingProblem(F, p, prob1, alg1, prob2, alg2, diff(LinRange(0, 1, length(centers) + 1)), SectionSS([F(c, p) for c in centers], centers); isparallel = isparallel,
kwargs...)

isSimple(sh::ShootingProblem) = sh.M == 1
@inline isParallel(sh::ShootingProblem) = sh.isparallel

# this function extracts the last component of the periodic orbit
extractPeriodShooting(x::AbstractVector) = x[end]
extractPeriodShooting(x::BorderedArray)  = x.p
@inline getPeriod(sh::ShootingProblem, x, par = nothing) = extractPeriodShooting(x)

function extractTimeSlices(x::AbstractVector, M::Int)
	N = div(length(x) - 1, M)
	return @views reshape(x[1:end-1], N, M)
end
extractTimeSlices(x::BorderedArray, M::Int) = x.u

@inline extractTimeSlice(x::AbstractMatrix, ii::Int) = @view x[:, ii]
@inline extractTimeSlice(x::AbstractVector, ii::Int) = xc[ii]

# putSection(x::AbstractVector, s) = x[end] = s
# putSection(x::BorderedArray, s) = x.p = s

# Standard shooting functional using AbstractVector, convenient for IterativeSolvers.
function (sh::ShootingProblem)(x::AbstractVector, par)
	# period of the cycle
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
			outc[:, ii] .= solOde[ii][2] .- xc[:, ip1]
		end
	end

	# add constraint
	out[end] = sh.section(x)

	return out
end

# # shooting functional, this allows for Array state space
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
	out.p = sh.section(x)

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
			outc[:, ii] .= @views tmp.du .+ sh.flow.F(tmp.u, par) .* sh.ds[ii] * dT .- dxc[:, ip1]
		end
	else
		# call jacobian of the flow
		solOde = sh.flow(xc, par, dxc, sh.ds .* T)
		for ii in 1:M
			ip1 = (ii == M) ? 1 : ii+1
			outc[:, ii] .= solOde[ii].du .+ sh.flow.F(solOde[ii].u, par) .* sh.ds[ii] * dT .- dxc[:, ip1]
		end
	end

	# add constraint
	out[end] = (sh.section(x .+ δ .* dx) - sh.section(x)) / δ

	return out
end

# jacobian of the shooting functional, this allows for Array state space
function (sh::ShootingProblem)(x::BorderedArray, par, dx::BorderedArray; δ = 1e-9)
	# period of the cycle
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
			out.u[ii] .= tmp.du .+ sh.flow.F(tmp.u, par) .* sh.ds[ii] * dT .- dx.u[ip1]
		end
	else
		@assert 1==0 "Not implemented yet. Try to use AbstractVectors instead"
	end

	# add constraint
	x_tmp = similar(x.u); copyto!(x_tmp, x.u)
	axpy!(δ , dx.u, x_tmp)
	out.p = (sh.section(BorderedArray(x_tmp, T + δ * dT)) - sh.section(x)) / δ

	return out
end

function _getMax(prob::ShootingProblem, x::AbstractVector, p; ratio = 1)
	# this function extracts the amplitude of the cycle
	T = extractPeriodShooting(x)
	M = length(prob.ds)
	N = div(length(x) - 1, M)
	xv = @view x[1:end-1]
	xc = reshape(xv, N, M)
	Th = eltype(x)
	mx = Th(0)

	# !!!! we could use @views but then Sundials will complain !!!
	if ~isParallel(prob)
		sol = prob.flow(Val(:Full), xc[:, 1], p, T)
		mx = @views maximum(sol[1:div(N, ratio), :], dims = 1)
	else
		sol = prob.flow(Val(:Full), xc, p, prob.ds .* T)
		for ii = 1:M
			mx = max(mx, maximum(sol[ii].u[1:div(N, ratio), :]))
		end
	end
	return mx
end

####################################################################################################
# functions needed Branch switching from Hopf bifurcation point

function update(prob::ShootingProblem, F, dF, hopfpt, ζr, M, orbitguess_a, period)
	# append period at the end of the initial guess
	orbitguess_v = reduce(vcat, orbitguess_a)
	orbitguess = vcat(vec(orbitguess_v), period) |> vec

	# update the problem
	probSh = setproperties(prob, M = M, section = SectionSS([F(c, hopfpt.params) for c in orbitguess_a], orbitguess_a))

	# be sure that the vector field is correctly inplace in the Flow structure
	probSh = @set probSh.flow.F = F

	return probSh, 	orbitguess
end
