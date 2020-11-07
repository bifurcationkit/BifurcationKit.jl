using DiffEqBase

####################################################################################################
# Poincare shooting based on Sánchez, J., M. Net, B. Garcı́a-Archilla, and C. Simó. “Newton–Krylov Continuation of Periodic Orbits for Navier–Stokes Flows.” Journal of Computational Physics 201, no. 1 (November 20, 2004): 13–33. https://doi.org/10.1016/j.jcp.2004.04.018.

"""

`pb = PoincareShootingProblem(flow::Flow, M, sections; δ = 1e-8, interp_points = 50, parallel = false)`

This composite type implements the Poincaré Shooting method to locate periodic orbits by relying on Poincaré return maps. More details (maths, notations, linear systems) can be found [here](https://rveltz.github.io/BifurcationKit.jl/dev/periodicOrbitShooting/). The arguments are as follows
- `flow::Flow`: implements the flow of the Cauchy problem though the structure [`Flow`](@ref).
- `M`: the number of Poincaré sections. If `M==1`, then the simple shooting is implemented and the multiple one otherwise.
- `sections`: function or callable struct which implements a Poincaré section condition. The evaluation `sections(x)` must return a scalar number when `M==1`. Otherwise, one must implement a function `section(out, x)` which populates `out` with the `M` sections. See [`SectionPS`](@ref) for type of section defined as a hyperplane.
- `δ = 1e-8` used to compute the jacobian of the fonctional by finite differences. If set to `0`, an analytical expression of the jacobian is used instead.
- `interp_points = 50` number of interpolation point used to define the callback (to compute the hitting of the hyperplan section)
- `parallel = false` whether the shooting are computed in parallel (threading). Only available through the use of Flows defined by `EnsembleProblem`.

## Simplified constructors
- A simpler way is to create a functional is

`pb = PoincareShootingProblem(F, p, prob::ODEProblem, alg, section; kwargs...)`

for simple shooting or

`pb = PoincareShootingProblem(F, p, prob::Union{ODEProblem, EnsembleProblem}, alg, M::Int, section; kwargs...)`

for multiple shooting . Here `F(x,p)` is the vector field, `p` is a parameter (to be passed to the vector and the flow), `prob` is an `Union{ODEProblem, EnsembleProblem}` which is used to create a flow using the ODE solver `alg` (for example `Tsit5()`). Finally, the arguments `kwargs` are passed to the ODE solver defining the flow. We refere to `DifferentialEquations.jl` for more information.

- Another convenient call is

`pb = PoincareShootingProblem(F, p, prob::Union{ODEProblem, EnsembleProblem}, alg, normals::AbstractVector, centers::AbstractVector; δ = 1e-8, kwargs...)`

where `normals` (resp. `centers`) is a list of normals (resp. centers) which defines a list of hyperplanes ``\\Sigma_i``. These hyperplanes are used to define partial Poincaré return maps.

## Computing the functionals
A functional, hereby called `G` encodes this shooting problem. You can then call `pb(orbitguess, par)` to apply the functional to a guess. Note that `orbitguess::AbstractVector` must be of size M * N where N is the number of unknowns in the state space and `M` is the number of Poincaré maps. Another accepted `guess` is such that `guess[i]` is the state of the orbit on the `i`th section. This last form allows for non-vector state space which can be convenient for 2d problems for example.

- `pb(orbitguess, par)` evaluates the functional G on `orbitguess`
- `pb(orbitguess, par, du)` evaluates the jacobian `dG(orbitguess).du` functional at `orbitguess` on `du`
- `pb`(Val(:JacobianMatrixInplace), J, x, par)` compute the jacobian of the functional analytically. This is based on ForwardDiff.jl. Useful mainly for ODEs.
- `pb(Val(:JacobianMatrix), x, par)` same as above but out-of-place.


!!! tip "Tip"
    You can use the function `getPeriod(pb, sol, par)` to get the period of the solution `sol` for the problem with parameters `par`.
"""
@with_kw struct PoincareShootingProblem{Tf, Tsection <: SectionPS} <: AbstractShootingProblem
	M::Int64 = 0						# number of Poincaré sections
	flow::Tf = Flow()					# should be a Flow
	section::Tsection = SectionPS(M)	# Poincaré sections
	δ::Float64 = 0e-8					# Numerical value used for the Matrix-Free Jacobian by finite differences. If set to 0, analytical jacobian is used
	parallel::Bool = false				# whether we use DE in Ensemble mode for multiple shooting
end

@inline isParallel(psh::PoincareShootingProblem) = psh.parallel

function PoincareShootingProblem(F, p,
			prob::ODEProblem, alg,
			hyp::SectionPS;
			δ = 1e-8, interp_points = 50, parallel = false, kwargs...)
	pSection(out, u, t, integrator) = (hyp(out, u); out .= out .* (integrator.iter > 1))
	affect!(integrator, idx) = terminate!(integrator)
	# we put nothing option to have an upcrossing
	cb = VectorContinuousCallback(pSection, affect!, hyp.M; interp_points = interp_points, affect_neg! = nothing)
	# change the ODEProblem -> EnsembleProblem for the parallel case
	_M = hyp.M
	parallel = _M==1 ? false : parallel
	_pb = parallel ? EnsembleProblem(prob) : prob
	return PoincareShootingProblem(flow = Flow(F, p, _pb, alg; callback = cb, kwargs...), M = hyp.M, section = hyp, δ = δ, parallel = parallel)
end

# this is the "simplest" constructor to use in automatic branching from Hopf
# this is a Hack to pass the arguments to construct a Flow. Indeed, we need to provide the
# appropriate callback for Poincare Shooting to work
PoincareShootingProblem(M::Int, par, prob::ODEProblem, alg; parallel = false, section = SectionPS(M), kwargs...) = PoincareShootingProblem(M = M, flow = (par = par, prob = prob, alg = alg, kwargs = kwargs), parallel = parallel, section = section)

PoincareShootingProblem(M::Int, par, prob1::ODEProblem, alg1, prob2::ODEProblem, alg2; parallel = false, section = SectionPS(M), kwargs...) = PoincareShootingProblem(M = M, flow = (par = par, prob1 = prob1, alg1 = alg1, prob2 = prob2, alg2 = alg2, kwargs = kwargs), parallel = parallel, section = section)

function PoincareShootingProblem(F, p,
			prob::ODEProblem, alg,
			normals::AbstractVector, centers::AbstractVector;
			δ = 1e-8, interp_points = 50, parallel = false, kwargs...)
	return PoincareShootingProblem(F, p,
					prob, alg,
					SectionPS(normals, centers);
					δ = δ, interp_points = interp_points, parallel = parallel, kwargs...)
end

function PoincareShootingProblem(F, p,
				prob1::ODEProblem, alg1,
				prob2::ODEProblem, alg2,
				hyp::SectionPS;
				δ = 1e-8, interp_points = 50, parallel = false, kwargs...)
	pSection(out, u, t, integrator) = (hyp(out, u); out .= out .* (integrator.iter > 1))
	affect!(integrator, idx) = terminate!(integrator)
	# we put nothing option to have an upcrossing
	cb = VectorContinuousCallback(pSection, affect!, hyp.M; interp_points = interp_points, affect_neg! = nothing)
	# change the ODEProblem -> EnsembleProblem for the parallel case
	_M = hyp.M
	parallel = _M == 1 ? false : parallel
	_pb1 = parallel ? EnsembleProblem(prob1) : prob1
	_pb2 = parallel ? EnsembleProblem(prob2) : prob2
	return PoincareShootingProblem(flow = Flow(F, p, _pb1, alg1, _pb2, alg2; callback = cb, kwargs...), M = hyp.M, section = hyp, δ = δ, parallel = parallel)
end

function PoincareShootingProblem(F, p,
				prob1::ODEProblem, alg1,
				prob2::ODEProblem, alg2,
				normals::AbstractVector, centers::AbstractVector;
				δ = 1e-8, interp_points = 50, parallel = false, kwargs...)
	return PoincareShootingProblem(F, p,
					prob1, alg2, prob2, alg2,
					SectionPS(normals, centers);
					δ = δ, interp_points = interp_points, parallel = parallel, kwargs...)
end
####################################################################################################
R(pb::PoincareShootingProblem, x::AbstractVector, k::Int) = R(pb.section, x, k)
E(pb::PoincareShootingProblem, xbar::AbstractVector, k::Int) = E(pb.section, xbar, k)


"""
	update!(pb::PoincareShootingProblem, centers_bar; _norm = norm)

This function updates the normals and centers of the hyperplanes defining the Poincaré sections.
"""
@views function updateSection!(pb::PoincareShootingProblem, centers_bar, par; _norm = norm)
	M = getM(pb); Nm1 = div(length(centers_bar), M)
	centers_barc = reshape(centers_bar, Nm1, M)
	centers = [E(pb.section, centers_barc[:, ii], ii) for ii = 1:M]
	normals = [pb.flow.F(c, par) for c in centers]

	for ii in eachindex(normals)
		normals[ii] ./= _norm(normals[ii])
	end
	update!(pb.section, normals, centers)
end

"""
$(SIGNATURES)

Compute the period of the periodic orbit associated to `x_bar`.
"""
function getPeriod(psh::PoincareShootingProblem, x_bar, par)
	M = getM(psh); Nm1 = div(length(x_bar), M)

	# reshape the period orbit guess
	x_barc = reshape(x_bar, Nm1, M)

	# variable to hold the computed result
	xc = similar(x_bar, Nm1 + 1, M)
	outc = similar(xc)

	Th = eltype(x_barc)
	period = Th(0)

	# we extend the state space to be able to call the flow, so we fill xc
	if ~isParallel(psh)
		for ii in 1:M
			E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
			# We need the callback to be active here!!!
			period += @views psh.flow(Val(:TimeSol), xc[:, ii], par, Inf64).t
		end
	else
		for ii in 1:M
			E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
		end
		solOde =  psh.flow(Val(:TimeSol), xc, par, repeat([Inf64], M))
		period = sum(x->x.t, solOde)
	end
	return period
end

"""
$(SIGNATURES)

Compute the full trajectory associated to `x`. Mainly for plotting purposes.
"""
function getTrajectory(prob::PoincareShootingProblem, x_bar::AbstractVector, p)
	# this function extracts the amplitude of the cycle
	M = getM(prob); Nm1 = length(x_bar) ÷ M

	# reshape the period orbit guess
	x_barc = reshape(x_bar, Nm1, M)
	xc = similar(x_bar, Nm1 + 1, M)

	# !!!! we could use @views but then Sundials will complain !!!
	if ~isParallel(prob)
		T = getPeriod(prob, x_bar, p)
		E!(prob.section, view(xc, :, 1), view(x_barc, :, 1), 1)
		# We need the callback to be active here!!!
		sol1 =  @views prob.flow(Val(:Full), xc[:, 1], p, T; callback = nothing)
		return sol1
	else # threaded version
		T = getPeriod(prob, x_bar, p)
		E!(prob.section, view(xc, :, 1), view(x_barc, :, 1), 1)
		sol = @views prob.flow(Val(:Full), xc[:, 1:1], p, [T]; callback = nothing)
		return sol[1]
	end
end

function _getExtremum(psh::PoincareShootingProblem, x_bar::AbstractVector, par; ratio = 1, op = (max, maximum))
	# this function extracts the amplitude of the cycle
	M = getM(psh)
	Nm1 = div(length(x_bar), M)

	# reshape the period orbit guess
	x_barc = reshape(x_bar, Nm1, M)
	xc = similar(x_bar, Nm1 + 1, M)

	Th = eltype(x_bar)
	n = div(Nm1, ratio)

	if ~isParallel(psh)
		E!(psh.section, view(xc, :, 1), view(x_barc, :, 1), 1)
		# We need the callback to be active here!!!
		sol = @views psh.flow(Val(:Full), xc[:, 1], par, Inf64)
		mx = op[2](sol[1:n, :])
		for ii in 2:M
			E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
			# We need the callback to be active here!!!
			sol = @views psh.flow(Val(:Full), xc[:, ii], par, Inf64)
			mx = op[1](mx, op[2](sol[1:n, :]))
		end
	else
		for ii in 1:M
			E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
		end
		solOde =  psh.flow(Val(:Full), xc, par, repeat([Inf64], M) )
		mx = op[2](solOde[1][1:n, :])
		for ii in 1:M
			mx = op[1](mx, op[2](solOde[ii][1:n, :]))
		end
	end
	return mx
end

"""
$(SIGNATURES)

Compute the projection of each vector (`x[i]` is a `Vector`) on the Poincaré section.
"""
function projection(psh::PoincareShootingProblem, x::AbstractVector)
	# create initial guess. We have to pass it through the projection R
	M = getM(psh)
	orbitguess_bar = Vector{eltype(x)}(undef, 0)
	@assert M == length(psh.section.normals)
	for ii=1:M
		push!(orbitguess_bar, R(psh, x[ii], ii))
	end
	return orbitguess_bar
end

"""
$(SIGNATURES)

Compute the projection of each vector (`x[i, :]` is a `Vector`) on the Poincaré section.
"""
function projection(psh::PoincareShootingProblem, x::AbstractMatrix)
	# create initial guess. We have to pass it through the projection R
	M = getM(psh)
	m, n = size(x)
	orbitguess_bar = Matrix{eltype(x)}(undef, m, n-1)
	@assert M == length(psh.section.normals)
	for ii=1:M
		orbitguess_bar[ii, :] .= @views R(psh, x[ii, :], ii)
	end
	return orbitguess_bar
end
####################################################################################################
# Poincaré (multiple) shooting with hyperplanes parametrization
function (psh::PoincareShootingProblem)(x_bar::AbstractVector, par; verbose = false)
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
	for ii in 1:M
		E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
	end

	if ~isParallel(psh)
		for ii in 1:M
			im1 = ii == 1 ? M : ii - 1
			# We need the callback to be active here!!!
			outc[:, ii] .= xc[:, ii] .- psh.flow(xc[:, im1], par, Inf64)
		end
	else
		solOde = psh.flow(xc, par, repeat([Inf64], M))
		for ii in 1:M
			im1 = ii == 1 ? M : ii - 1
			# We need the callback to be active here!!!
			@views outc[:, ii] .= xc[:, ii] .- solOde[im1][2]
		end
	end

	# build the array to be returned
	out_bar = similar(x_bar)
	out_barc = reshape(out_bar, Nm1, M)
	for i in 1:M
		R!(psh.section, view(out_barc, :, i), view(outc, :, i), i)
	end
	return out_bar
end

"""
This function compute the derivative of the Poincare return map Π(x) = ϕ(t(x),x)
"""
function diffPoincareMap(psh::PoincareShootingProblem, x, par, dx, ii::Int)
	normal = psh.section.normals[ii]
	abs(dot(normal, dx)) > 1e-12 && @warn "Vector does not belong to hyperplane!  dot(normal, dx) = $(abs(dot(normal, dx))) and $(dot(dx, dx))"
	# compute the Poincare map from x
	tΣ, solΣ = psh.flow(Val(:TimeSol), x, par, Inf64)
	z = psh.flow.F(solΣ, par)
	# solution of the variational equation at time tΣ
	# We need the callback to be INACTIVE here!!!
	y = psh.flow(x, par, dx, tΣ; callback = nothing).du
	out = y .- (dot(normal, y) / dot(normal, z)) .* z
end

# jacobian of the shooting functional
function (psh::PoincareShootingProblem)(x_bar::AbstractVector, par, dx_bar::AbstractVector)
	δ = psh.δ
	if δ > 0
		# mostly for debugging purposes
		return (psh(x_bar .+  δ .* dx_bar, par) .- psh(x_bar, par)) ./ δ
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
	for ii in 1:M
		 E!(psh.section,  view(xc, :, ii),  view(x_barc, :, ii), ii)
		dE!(psh.section, view(dxc, :, ii), view(dx_barc, :, ii), ii)
	end

	if ~isParallel(psh)
		for ii in 1:M
			im1 = (ii == 1 ? M : ii - 1)
			@views outc[:, ii] .= dxc[:, ii] .- diffPoincareMap(psh, xc[:, im1], par, dxc[:, im1], im1)
		end
	else
		@assert 1==0 "Analytical Jacobian for parallel Poincare Shooting not implemented yet. Please use the option δ > 0 to use Matrix-Free jacobian or chose `:FiniteDifferencesDense` to compute jacobian based on finite differences."
	end

	# build the array to be returned
	out_bar = similar(x_bar)
	out_barc = reshape(out_bar, Nm1, M)
	for ii in 1:M
		dR!(psh.section, view(out_barc, :, ii), view(outc, :, ii), ii)
	end
	return out_bar
end

# inplace computation of the matrix of the jacobian of the shooting problem, only serial for now
function (psh::PoincareShootingProblem)(::Val{:JacobianMatrixInplace}, J::AbstractMatrix, x_bar::AbstractVector, par)
	M = getM(psh)
	Nm1 = div(length(x_bar), M)
	N = Nm1 + 1

	x_barc = reshape(x_bar, Nm1, M)

	# TODO the following declaration of xc allocates. It would be better to make it inplace
	xc = similar(x_bar, N, M)

	# we extend the state space to be able to call the flow, so we fill xc
	#TODO create the projections on the fly
	for ii in 1:M
		E!(psh.section, view(xc, :, ii), view(x_barc, :, ii), ii)
	end

	# jacobian of the flow
	dflow = (_J, _x, _T) -> ForwardDiff.jacobian!(_J, z -> psh.flow(Val(:SerialTimeSol), z, par, _T; callback = nothing).u, _x)

	# initialize some temporaries
	Jtmp = zeros(N, N)
	normal = copy(psh.section.normals[1])
	F = zeros(N)
	Rm = zeros(Nm1, N)
	Em = zeros(N, Nm1)

	# put the matrices by blocks
	In = I(Nm1)
	for ii=1:M
		im1 = ii == 1 ? M : ii - 1
		# we find the point on the next section
		tΣ, solΣ = psh.flow(Val(:SerialTimeSol), view(xc,:,im1), par, Inf64)
		# computation of the derivative of the return map
		F .= psh.flow.F(solΣ, par)
		normal .= psh.section.normals[ii]
		dflow(Jtmp, xc[:, im1], tΣ)
		Jtmp .= Jtmp .- F * normal' * Jtmp ./ dot(F, normal)
		# projection with Rm, Em
		ForwardDiff.jacobian!(Rm, x->R(psh.section, x, ii), zeros(N))
		ForwardDiff.jacobian!(Em, x->E(psh.section, x, im1), zeros(Nm1))
		J[(ii-1)*Nm1+1:(ii-1)*Nm1+Nm1, (im1-1)*Nm1+1:(im1-1)*Nm1+Nm1] .= -Rm * Jtmp * Em
		if M == 1
			J[(ii-1)*Nm1+1:(ii-1)*Nm1+Nm1, (ii-1)*Nm1+1:(ii-1)*Nm1+Nm1] .+= In
		else
			J[(ii-1)*Nm1+1:(ii-1)*Nm1+Nm1, (ii-1)*Nm1+1:(ii-1)*Nm1+Nm1] .= In
		end
	end
	return J
end

# out of place version
(psh::PoincareShootingProblem)(::Val{:JacobianMatrix}, x::AbstractVector, par) = psh(Val(:JacobianMatrixInplace), zeros(length(x), length(x)), x, par)
####################################################################################################
# functions needed for Branch switching from Hopf bifurcation point
function problemForBS(prob::PoincareShootingProblem, F, dF, hopfpt, ζr, centers, period)
	# create the section
	normals = [F(u, hopfpt.params) for u in centers]
	for n in normals; n ./= norm(n); end

	@assert isEmpty(prob.section) "Specifying its own section for aBS from a Hopf point is not allowed yet."

	@assert ~(prob.flow isa Flow) "Somehow, this method was not called as it should. `prob.flow` should be a Named Tuple, prob should be constructed with the simple constructor, not yielding a Flow for its flow field."

	# update the problem
	if length(prob.flow) == 4
		probPSh = PoincareShootingProblem(F, prob.flow.par, prob.flow.prob, prob.flow.alg, deepcopy(normals), deepcopy(centers); parallel = prob.parallel, prob.flow.kwargs...)
	else
		probPSh = PoincareShootingProblem(F, prob.flow.par, prob.flow.prob1, prob.flow.alg1, prob.flow.prob2, prob.flow.alg2, deepcopy(normals), deepcopy(centers); parallel = prob.parallel, prob.flow.kwargs...)
	end

	# create initial guess. We have to pass it through the projection R
	hyper = probPSh.section
	M = getM(probPSh)
	@assert length(normals) == M
	orbitguess_bar = zeros(length(centers[1])-1, M)
	for ii=1:length(normals)
		orbitguess_bar[:, ii] .= R(hyper, centers[ii], ii)
	end

	return probPSh, vec(orbitguess_bar)
end
