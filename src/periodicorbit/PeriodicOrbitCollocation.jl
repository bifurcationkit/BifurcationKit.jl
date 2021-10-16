using FastGaussQuadrature: gausslegendre

"""
	cache = POOrthogonalCollocationCache(Ntst::Int, m::Int, Ty = Float64)

Structure to hold the cache for the collocation method.

# Arguments
- `Ntst` number of time steps
- `m` degree of the polynomials
- `Ty` type of the time variable
"""
struct POOrthogonalCollocationCache{T}
	Ntst::Int
	degree::Int
    lagrange_vals::Matrix{T}
    lagrange_driv::Matrix{T}
	gauss_nodes::Vector{T}
    gauss_weight::Vector{T}
	# TODO how do we ensure that eltype(mesh) is what we expect?
	mesh::Vector{T} 		# τs, we need a vector here for mesh adaptation
	mesh_coll::LinRange{T} 	# σs
	full_mesh::Vector{T}
end

function POOrthogonalCollocationCache(Ntst::Int, m::Int, Ty = Float64)
	τs = LinRange{Ty}( 0, 1, Ntst + 1) |> collect
	σs = LinRange{Ty}(-1, 1, m + 1)
	L, ∂L = getL(σs)
	zg, wg = gausslegendre(m)
	prob = POOrthogonalCollocationCache(Ntst, m, L, ∂L, zg, wg, τs, σs, zeros(Ty, 1 + m * Ntst))
	# put the mesh where we removed redundant timing
	prob.full_mesh .= getTimes(prob)
	return prob
end

@inline Base.eltype(pb::POOrthogonalCollocationCache) = eltype(pb.lagrange_vals)
@inline Base.size(pb::POOrthogonalCollocationCache) = (pb.degree, pb.Ntst)
@inline getLs(pb::POOrthogonalCollocationCache) = (pb.lagrange_vals, pb.lagrange_driv)
@inline getMesh(pb::POOrthogonalCollocationCache) = pb.mesh
@inline getMeshColl(pb::POOrthogonalCollocationCache) = pb.mesh_coll
getMaxTimeStep(pb::POOrthogonalCollocationCache) = maximum(diff(getMesh(pb)))
τj(σ, τs, j) = τs[j] + (1 + σ)/2 * (τs[j+1] - τs[j])
# get the sigma corresponding to τ in the interval (𝜏s[j], 𝜏s[j+1])
σj(τ, τs, j) = -(2*τ - τs[j] - τs[j + 1])/(-τs[j + 1] + τs[j])

# code from Jacobi.lagrange
function lagrange(i, x, z)
    nz = length(z)
    l = one(z[1])
	for k = 1:(i-1)
        l = l * (x - z[k]) / (z[i] - z[k])
    end
    for k = (i+1):nz
        l = l * (x - z[k]) / (z[i] - z[k])
    end
    return l
end

dlagrange(i, x, z) = ForwardDiff.derivative(x -> lagrange(i, x, z), x)

# accept a range, ie σs = LinRange(-1, 1, m + 1)
function getL(σs::AbstractVector)
	m = length(σs) - 1
	zs, = gausslegendre(m)
	L = zeros(m, m+1); ∂L = zeros(m, m + 1)
	for j = 1:m+1
		for i=1:m
			L[i, j]  =  lagrange(j, zs[i], σs)
			∂L[i, j] = dlagrange(j, zs[i], σs)
		end
	end
	return (;L, ∂L)
end

"""
$(SIGNATURES)

Return the times at which is evaluated the collocation problem.
"""
function getTimes(pb::POOrthogonalCollocationCache)
	m, Ntst = size(pb)
	Ty = eltype(pb)
	ts = Matrix{Ty}(undef, Ntst, m + 1)
	tsvec = Ty[0]
	τs = pb.mesh
	σs = pb.mesh_coll
	for j=1:Ntst
		for l=1:m+1
			ts[j, l] = τj(σs[l], τs, j)
			l>1 && push!(tsvec, τj(σs[l], τs, j))
		end
	end
	return vec(tsvec)
end

function updateMesh!(pb::POOrthogonalCollocationCache, mesh)
	pb.mesh .= mesh
	pb.full_mesh .= getTimes(pb)
end
####################################################################################################

"""
	pb = PeriodicOrbitOCollProblem(kwargs...)

This composite type implements an orthogonal collocation (at Guass points) method of piecewise polynomials to locate periodic orbits. More details (maths, notations, linear systems) can be found [here](https://rveltz.github.io/BifurcationKit.jl/dev/periodicOrbitCollocation/).

## Arguments
- `F` vector field specified as a function of two arguments `F(x,p)`
- `J` is the jacobian of `F` at `(x, p)`. It can assume three forms.
    1. Either `J` is a function and `J(x, p)` returns a `::AbstractMatrix`. In this case, the default arguments of `contParams::ContinuationPar` will make `continuation` work.
    2. Or `J` is a function and `J(x, p)` returns a function taking one argument `dx` and returning `dr` of the same type as `dx`. In our notation, `dr = J * dx`. In this case, the default parameters of `contParams::ContinuationPar` will not work and you have to use a Matrix Free linear solver, for example `GMRESIterativeSolvers`,
    3. Or `J` is a function and `J(x, p)` returns a variable `j` which can assume any type. Then, you must implement a linear solver `ls` as a composite type, subtype of `AbstractLinearSolver` which is called like `ls(j, rhs)` and which returns the solution of the jacobian linear system. See for example `examples/SH2d-fronts-cuda.jl`. This linear solver is passed to `NewtonPar(linsolver = ls)` which itself passed to `ContinuationPar`. Similarly, you have to implement an eigensolver `eig` as a composite type, subtype of `AbstractEigenSolver`.
- `ϕ::AbstractVector` used to set a section for the phase constraint equation
- `xπ::AbstractVector` used in the section for the phase constraint equation
- `N::Int` dimension of the state space
- `coll_cache::POOrthogonalCollocationCache` cache for collocation. See docs of `POOrthogonalCollocationCache` .

## Methods

Here are some useful methods you can apply to `pb`

- `length(pb)` gives the total number of unknowns
- `size(pb)` returns the triplet `(N, m, Ntst)`
- `getMesh(pb)` returns the mesh `0 = τ0 < ... < τNtst+1 = 1`. This is useful because this mesh is born to vary by automatic mesh adaptation
- `getMeshColl(pb)` returns the (static) mesh `0 = σ0 < ... < σm+1 = 1`
- `getTimes(pb)` returns the vector of times (length `1 + m * Ntst`) at the which the collocation is applied.
- `generateSolution(pb, orbit, period)` generate a guess from a function `t -> orbit(t)` which approximates the periodic orbit.
- `POOcollSolution(pb, x)` return a function interpolating the solution `x` using a piecewise polynomials function


# Orbit guess
You will see below that you can evaluate the residual of the functional (and other things) by calling `pb(orbitguess, p)` on an orbit guess `orbitguess`. Note that `orbitguess` must be of size 1 + N * (1 + m * Ntst) where N is the number of unknowns in the state space and `orbitguess[end]` is an estimate of the period ``T`` of the limit cycle.

Note that you can generate this guess from a function using `generateSolution`.

# Functional
 A functional, hereby called `G`, encodes this problem. The following methods are available

- `pb(orbitguess, p)` evaluates the functional G on `orbitguess`

"""
@with_kw_noshow struct PeriodicOrbitOCollProblem{TF, TJ, vectype, Tmass, Tcache <: POOrthogonalCollocationCache} <: AbstractPeriodicOrbitProblem
	# Function F(x, par)
	F::TF = nothing

	# Jacobian of F w.r.t. x
	J::TJ = nothing

	# variables to define a Section for the phase constraint equation
	ϕ::vectype = nothing
	xπ::vectype = nothing

	# dimension of the problem in case of an AbstractVector
	N::Int = 0

	# whether the time discretisation is adaptive
	adaptmesh::Bool = false

	# whether the problem is nonautonomous
	isautonomous::Bool = true

	# mass matrix
	massmatrix::Tmass = nothing

	# collocation cache
	coll_cache::Tcache = nothing
end

# trivial constructor
function PeriodicOrbitOCollProblem(Ntst, m, N = 0)
	cache = POOrthogonalCollocationCache(Ntst, m)
	PeriodicOrbitOCollProblem(; N = N, coll_cache = cache)
end

# TODO rename this in num_mesh? or meshSize
@inline getM(pb::PeriodicOrbitOCollProblem) = pb.coll_cache.Ntst

@inline length(pb::PeriodicOrbitOCollProblem) = ( (n, m, Ntst) = size(pb); return n * (1 + m * Ntst) )
# the size is (n, m, Ntst)
@inline Base.size(pb::PeriodicOrbitOCollProblem) = (pb.N, size(pb.coll_cache)...)
@inline Base.eltype(pb::PeriodicOrbitOCollProblem) = eltype(pb.coll_cache)
getLs(pb::PeriodicOrbitOCollProblem) = getLs(pb.coll_cache)

# these functions extract the time slices components
getTimeSlices(x::AbstractVector, N, degree, Ntst) = @views reshape(x[1:end-1], N, (degree) * Ntst + 1)
# array of size Ntst ⋅ (m+1) ⋅ n
getTimeSlices(pb::PeriodicOrbitOCollProblem, x) = getTimeSlices(x, size(pb)...)
getTimes(pb::PeriodicOrbitOCollProblem) = getTimes(pb.coll_cache)
getMesh(pb::PeriodicOrbitOCollProblem) = getMesh(pb.coll_cache)
getMeshColl(pb::PeriodicOrbitOCollProblem) = getMeshColl(pb.coll_cache)
getMaxTimeStep(pb::PeriodicOrbitOCollProblem) = getMaxTimeStep(pb.coll_cache)
updateMesh!(pb::PeriodicOrbitOCollProblem, mesh) = updateMesh!(pb.coll_cache, mesh)

function Base.show(io::IO, pb::PeriodicOrbitOCollProblem)
	N, m, Ntst = size(pb)
	println(io, "┌─ Collocation problem for periodic orbits")
	println(io, "├─ type            : Vector{", eltype(pb), "}")
	println(io, "├─ time slices     : ", Ntst)
	println(io, "├─ degree          : ", m)
	println(io, "├─ space dimension : ", pb.N)
	println(io, "└─ # unknowns      : ", pb.N * (1 + m * Ntst))
end

"""
$(SIGNATURES)

This function generates an initial guess for the solution of the problem `pb` based on the orbit `t -> orbit(t)` for t ∈ [0,1] and the period `period`.
"""
function generateSolution(pb::PeriodicOrbitOCollProblem, orbit, period)
	n, _m, Ntst = size(pb)
	ts = getTimes(pb)
	Nt = length(ts)
	ci = zeros(eltype(pb), n, Nt)
	for (l, t) in pairs(ts)
		ci[:, l] .= orbit(t * period)
	end
	return vcat(vec(ci), period)
end

@views function (prob::PeriodicOrbitOCollProblem)(u::AbstractVector, pars)
	uc = getTimeSlices(prob, u)
	T = getPeriod(prob, u, pars)
	result = zero(u)
	resultc = getTimeSlices(prob, result)
	functionalColl!(prob, resultc, uc, T, getLs(prob.coll_cache), pars)
	# add  the phase condition
	result[end] = dot(u[1:end-1], prob.ϕ) - dot(prob.xπ, prob.ϕ)
	return result
end

# function or collocation problem
@views function functionalColl!(pb, out, u, period, (L, ∂L), pars)
	Ty = eltype(u)
	n, ntimes = size(u)
	m = pb.coll_cache.degree
	Ntst = pb.coll_cache.Ntst
	# on veut faire des slices a t fixes, donc des gj[:, j], c'est le plus rapide
	gj = zeros(Ty, n, m)
	∂gj = zeros(Ty, n, m)
	uj = zeros(Ty, n, m+1)
	mesh = getMesh(pb)
	rg = 1:m+1
	for j=1:Ntst
		uj .= u[:, rg]
		gj 	.=  uj * L'
		∂gj .=  uj * ∂L'
		# mul!(gj, uj, L')
		# mul!(∂gj, uj, ∂L')
		# compute the collocation residual
		for l=1:m
			out[:, rg[l]] .= ∂gj[:, l] .- (period * (mesh[j+1]-mesh[j]) / (2)) .* pb.F(gj[:, l], pars)
		end
		rg = rg .+ m
	end
	# add the periodicity condition
	out[:, end] .= u[:, end] .- u[:, 1]
end

"""
$(SIGNATURES)

Compute the full trajectory associated to `x`. Mainly for plotting purposes.
"""
@views function getTrajectory(prob::PeriodicOrbitOCollProblem, u::AbstractVector, p)
	T = getPeriod(prob, u, p)
	ts = getTimes(prob)
	uc = getTimeSlices(prob, u)
	n, m, Ntst = size(prob)
	return (t = ts .* T, u = uc)
end

# function needed for automatic Branch switching from Hopf bifurcation point
function reMake(prob::PeriodicOrbitOCollProblem, F, dF, par, hopfpt, ζr::AbstractVector, orbitguess_a, period; orbit = t->t)
	M = length(orbitguess_a)
	N = length(ζr)

	n, m, Ntst = size(prob)
	nunknows = N * (1 + m*Ntst)

	# update the problem
	probPO = setproperties(prob, N = N, F = F, J = dF, ϕ = zeros(nunknows), xπ = zeros(nunknows))

	probPO.ϕ[1:N] .= ζr
	probPO.xπ[1:N] .= hopfpt.x0

	# append period at the end of the initial guess
	orbitguess = generateSolution(probPO, t -> orbit(2pi*t/period), period)

	return probPO, orbitguess
end

####################################################################################################
"""
$(SIGNATURES)

This is the Newton Solver for computing a periodic orbit using (Standard / Poincaré) Shooting method.
Note that the linear solver has to be apropriately set up in `options`.

# Arguments

Similar to [`newton`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref). These two problems have specific options to be tuned, we refer to their link for more information and to the tutorials.

- `prob` a problem of type `<: AbstractShootingProblem` encoding the shooting functional G.
- `orbitguess` a guess for the periodic orbit. See [`ShootingProblem`](@ref) and See [`PoincareShootingProblem`](@ref) for information regarding the shape of `orbitguess`.
- `par` parameters to be passed to the functional
- `options` same as for the regular [`newton`](@ref) method.

# Optional argument
- `linearPO` Specify the choice of the linear algorithm, which must belong to `(:autodiffMF, :MatrixFree, :autodiffDense, :autodiffDenseAnalytical, :FiniteDifferences)`. This is used to select a way of inverting the jacobian dG
	- For `:MatrixFree`, we use an iterative solver (e.g. GMRES) to solve the linear system. The jacobian was specified by the user in `prob`.
	- For `:autodiffMF`, we use iterative solver (e.g. GMRES) to solve the linear system. We use Automatic Differentiation to compute the (matrix-free) derivative of `x -> prob(x, p)`.
	- For `:autodiffDense`. Same as for `:autodiffMF` but the jacobian is formed as a dense Matrix. You can use a direct solver or an iterative one using `options`.
	- For `:autodiffDenseAnalytical`. Same as for `:autodiffDense` but the jacobian is using a mix of AD and analytical formula.
	- For `:FiniteDifferencesDense`, same as for `:autodiffDense` but we use Finite Differences to compute the jacobian of `x -> prob(x, p)` using the `δ = 1e-8` which can be passed as an argument.

# Output:
- solution
- history of residuals
- flag of convergence
- number of iterations
"""
function newton(prob::PeriodicOrbitOCollProblem, orbitguess, par, options::NewtonPar;
		linearPO = :autodiffDense, δ = 1e-8, kwargs...)
	@assert linearPO in
			(:autodiffMF, :MatrixFree, :autodiffDense, :autodiffSparse, )

	if linearPO == :autodiffDense
		jac = (x, p) -> ForwardDiff.jacobian(z -> prob(z, p), x)
	end

	return newton(prob, jac, orbitguess, par, options; kwargs...)
end

"""
$(SIGNATURES)

This is the continuation method for computing a periodic orbit using a (Standard / Poincaré) Shooting method.

# Arguments

Similar to [`continuation`](@ref) except that `prob` is either a [`ShootingProblem`](@ref) or a [`PoincareShootingProblem`](@ref). By default, it prints the period of the periodic orbit.

# Optional argument
- `δ = 1e-8` used for finite differences
- `linearPO` Specify the choice of the linear algorithm, which must belong to `[:autodiffMF, :MatrixFree, :autodiffDense, :autodiffDenseAnalytical, :FiniteDifferences]`. This is used to select a way of inverting the jacobian dG
- `updateSectionEveryStep = 0` updates the section every `updateSectionEveryStep` step during continuation

## Choices for `linearPO`
- For `:MatrixFree`, we use an iterative solver (e.g. GMRES) to solve the linear system. The jacobian was specified by the user in `prob`.
- For `:autodiffMF`, we use iterative solver (e.g. GMRES) to solve the linear system. We use Automatic Differentiation to compute the (matrix-free) derivative of `x -> prob(x, p)`.
- For `:autodiffDense`. Same as for `:autodiffMF` but the jacobian is formed as a dense Matrix. You can use a direct solver or an iterative one using `options`.
- For `:FiniteDifferencesDense`, same as for `:autodiffDense` but we use Finite Differences to compute the jacobian of `x -> prob(x, p)` using the `δ = 1e-8` which can be passed as an argument.
- For `:autodiffDenseAnalytical`. Same as for `:autodiffDense` but the jacobian is using a mix of AD and analytical formula.
- For `:FiniteDifferences`, use Finite Differences to compute the jacobian of `x -> prob(x, p)` using the `δ = 1e-8` which can be passed as an argument.
"""
function continuation(
	prob::PeriodicOrbitOCollProblem,
	orbitguess, par, lens::Lens, _contParams::ContinuationPar,
	_linearAlgo::AbstractBorderedLinearSolver;
	linearPO = :autodiffDense,
	updateSectionEveryStep = 0, kwargs...)
	@assert linearPO in
			(:autodiffMF, :MatrixFree, :autodiffDense, :autodiffDenseAnalytical, :FiniteDifferencesDense, :FiniteDifferences, :Dense)

	jac = (x, p) -> FloquetWrapper(prob, ForwardDiff.jacobian(z -> prob(z, p), x), x, p)

	# we have to change the Bordered linearsolver to cope with our type FloquetWrapper
	linearAlgo = @set _linearAlgo.solver = FloquetWrapperLS(_linearAlgo.solver)

	options = _contParams.newtonOptions
	contParams = @set _contParams.newtonOptions.linsolver = FloquetWrapperLS(options.linsolver)

	if computeEigenElements(contParams)
		contParams = @set contParams.newtonOptions.eigsolver = FloquetLUColl(contParams.newtonOptions.eigsolver, length(prob), prob.N)
	end

	# change the user provided finalise function by passing prob in its parameters
	_finsol = modifyPOFinalise(prob, kwargs, updateSectionEveryStep)
	_recordsol = modifyPORecord(prob, kwargs, par, lens)
	_plotsol = modifyPOPlot(prob, kwargs)

	branch, u, τ = continuation(
					prob, jac,
					orbitguess, par, lens,
					contParams, linearAlgo;
					kwargs...,
					recordFromSolution = _recordsol,
					finaliseSolution = _finsol,
					plotSolution = _plotsol)
	return setproperties(branch; type = :PeriodicOrbit, functional = prob), u, τ
end

"""
$(SIGNATURES)

Compute the maximum of the periodic orbit associated to `x`.
"""
function getMaximum(prob::PeriodicOrbitOCollProblem, x::AbstractVector, p)
	sol = getTrajectory(prob, x, p).u
	return maximum(sol)
end

# this function updates the section during the continuation run
@views function updateSection!(prob::PeriodicOrbitOCollProblem, x, par; stride = 0)
	n, m, Ntst = size(prob)
	xc = getTimeSlices(prob, x)
	T = getPeriod(prob, x, par)

	# update the reference point
	prob.xπ .= x[1:end-1]

	# update the normals
	ϕc = @views reshape(prob.ϕ, n, (m) * Ntst + 1)
	for ii = 1: (m) * Ntst + 1
		ϕc[:, ii] .= prob.F(xc[:, ii], par) ./ (m*Ntst)
	end
	return true
end
####################################################################################################
# iterated derivatives
∂(f) = x -> ForwardDiff.derivative(f, x)
∂(f, n) = n == 0 ? f : ∂(∂(f), n-1)

struct POOcollSolution{Tpb, Tx}
	pb::Tpb
	x::Tx
end

@views function (sol::POOcollSolution)(t0)
	n,m,Ntst = size(sol.pb)
	xc = getTimeSlices(sol.pb, sol.x)

	T = getPeriod(sol.pb, sol.x, 0)
	t = t0 / T

	mesh = getMesh(sol.pb)
	indτ = searchsortedfirst(mesh, t) - 1
	if indτ <= 0
		return sol.x[1:n]
	elseif indτ > Ntst
		return xc[:, end]
	end
	# println("--> ", t, " belongs to ", (mesh[indτ], mesh[indτ+1])) # waste lots of ressources
	@assert mesh[indτ] <= t <= mesh[indτ+1] "Please open an issue on the website of BifurcationKit.jl"
	σ = σj(t, mesh, indτ)
	# @assert -1 <= σ <= 1 "Strange value of $σ"
	σs = getMeshColl(sol.pb)
	out = zeros(typeof(t), sol.pb.N)
	rg = (1:m+1) .+ (indτ-1)*m
	for l = 1:m+1
		out .+= xc[:, rg[l]] .* lagrange(l, σ, σs)
	end
	out
end
