abstract type AbstractBranchResult end
abstract type AbstractResult{Tkind, Tprob} <: AbstractBranchResult end

####################################################################################################
# functions used in RecordFromSolution
namedprintsol(x) = (x = x,)
namedprintsol(x::Real) = (x = x,)
namedprintsol(x::NamedTuple) = x
namedprintsol(x::Tuple) = (;zip((Symbol("x$i") for i in eachindex(x)), x)...)
mergefromuser(x, a::NamedTuple) = merge(namedprintsol(x), a)
####################################################################################################
# Structure to hold continuation result
"""
$(TYPEDEF)

Structure which holds the results after a call to [`continuation`](@ref).

You can see the propertynames of a result `br` by using `propertynames(br)` or `propertynames(br.branch)`.

# Fields

$(TYPEDFIELDS)

# Associated methods
- `length(br)` number of the continuation steps
- `eigenvals(br, ind)` returns the eigenvalues for the ind-th continuation step
- `eigenvec(br, ind, indev)` returns the indev-th eigenvector for the ind-th continuation step
- `br[k+1]` gives information about the k-th step. A typical run yields the following
```
julia> br[1]
(x = 0.0, param = 0.1, itnewton = 0, itlinear = 0, ds = -0.01, θ = 0.5, n_unstable = 2, n_imag = 2, stable = false, step = 0, eigenvals = ComplexF64[0.1 - 1.0im, 0.1 + 1.0im], eigenvecs = ComplexF64[0.7071067811865475 - 0.0im 0.7071067811865475 + 0.0im; 0.0 + 0.7071067811865475im 0.0 - 0.7071067811865475im])
```
which provides the value `param` of the parameter of the current point, its stability, information on the newton iterations, etc. The fields can be retrived using `propertynames(br.branch)`. This information is stored in `br.branch` which is a `StructArray`. You can thus extract the vector of parameters along the branch as
```
julia> br.param
10-element Vector{Float64}:
 0.1
 0.08585786437626905
 0.06464466094067263
 0.03282485578727799
-1.2623798512809007e-5
-0.07160718539365075
-0.17899902778635765
-0.3204203840236672
-0.4618417402609767
-0.5
```
- `getSolx(br, k)` returns the k-th solution on the branch
- `getSolp(br, k)` returns the parameter  value associated with k-th solution on the branch
- `getParams(br)` Parameters passed to continuation and used in the equation `F(x, par) = 0`.
- `setParam(br, p0)` set the parameter value `p0` according to `::Lens` for the parameters of the problem `br.prob`
- `getLens(br)` get the lens used for the computation of the branch
"""
@with_kw_noshow struct ContResult{Tkind <: AbstractContinuationKind, Tbr, Teigvals, Teigvec, Biftype, Tsol, Tparc, Tprob, Talg} <: AbstractResult{Tkind, Tprob}
	"holds the low-dimensional information about the branch. More precisely, `branch[i+1]` contains the following information `(recordFromSolution(u, param), param, itnewton, itlinear, ds, θ, n_unstable, n_imag, stable, step)` for each continuation step `i`.\n
  - `itnewton` number of Newton iterations
  - `itlinear` total number of linear iterations during corrector
  - `n_unstable` number of eigenvalues with positive real part for each continuation step (to detect stationary bifurcation)
  - `n_imag` number of eigenvalues with positive real part and non zero imaginary part for each continuation step (to detect Hopf bifurcation).
  - `stable`  stability of the computed solution for each continuation step. Hence, `stable` should match `eig[step]` which corresponds to `branch[k]` for a given `k`.
  - `step` continuation step (here equal `i`)"
	branch::StructArray{Tbr}

	"A vector with eigen-elements at each continuation step."
	eig::Vector{NamedTuple{(:eigenvals, :eigenvecs, :converged, :step), Tuple{Teigvals, Teigvec, Bool, Int64}}}

	"Vector of solutions sampled along the branch. This is set by the argument `saveSolEveryStep::Int64` (default 0) in [`ContinuationPar`](@ref)."
	sol::Tsol

	"The parameters used for the call to `continuation` which produced this branch. Must be a ContinationPar"
	contparams::Tparc

	"Type of solutions computed in this branch."
	kind::Tkind = EquilibriumCont()

	"Structure associated to the functional, useful for branch switching. For example, when computing periodic orbits, the functional `PeriodicOrbitTrapProblem`, `ShootingProblem`... will be saved here."
	prob::Tprob = nothing

	"A vector holding the set of detected bifurcation points. See [`SpecialPoint`](@ref) for a description of the fields."
	specialpoint::Vector{Biftype}

	"Continuation algorithm used for the computation of the branch"
	alg::Talg

	@assert contparams isa ContinuationPar "Only `::ContinuationPar` parameters are allowed"
end

# returns the number of steps in a branch
Base.length(br::AbstractBranchResult) = length(br.branch)
getParams(br::AbstractBranchResult) = getParams(br.prob)
getLens(br::AbstractBranchResult) = getLens(br.prob)
@inline getProb(br::AbstractBranchResult) = br.prob

# check whether the eigenvalues are saved in the branch
# this is a good test bifucause we always fill br.eig with a dummy vector :(
@inline haseigenvalues(br::ContResult) = hasstability(br)
@inline haseigenvalues(br::AbstractBranchResult) = haseigenvalues(br.γ)

# check whether the solution are saved in the branch
@inline hassolution(br::ContResult{Tkind, Tbr, Teigvals, Teigvec, Biftype, Tsol, Tparc, Tprob, Talg} ) where {Tkind, Tbr, Teigvals, Teigvec, Biftype, Tsol, Tparc, Tprob, Talg } = Tsol != Nothing
@inline hassolution(br::AbstractBranchResult) = hassolution(br.γ)

# check whether the eigenvectors are saved in the branch
@inline haseigenvector(br::ContResult{Tkind, Tbr, Teigvals, Teigvec, Biftype, Tsol, Tparc, Tprob, Talg} ) where {Tkind, Tbr, Teigvals, Teigvec, Biftype, Tsol, Tparc, Tprob, Talg } = Teigvec != Nothing
@inline haseigenvector(br::AbstractBranchResult) = haseigenvector(br.γ)

@inline hasstability(br::AbstractBranchResult) = typeof(br.branch).parameters[1].parameters[2].parameters[end-1] == Bool

getfirstusertype(br::AbstractBranchResult) = keys(br.branch[1])[1]
@inline getvectortype(br::AbstractBranchResult) = getVectorType(eltype(br.specialpoint))
@inline getvectoreltype(br::AbstractBranchResult) = eltype(getvectortype(br))
setParam(br::AbstractBranchResult, p0) = setParam(br.prob, p0)
Base.getindex(br::ContResult, k::Int) = (br.branch[k]..., eigenvals = haseigenvalues(br) ? br.eig[k].eigenvals : nothing, eigenvecs = haseigenvector(br) ? br.eig[k].eigenvecs : nothing)
Base.lastindex(br::ContResult) = length(br)
@inline function getSolx(br::ContResult, ind::Int)
	@assert hassolution(br) "You did not record the solution in the branch. Please set `saveSolEveryStep` in `ContinuationPar`"
	return br.sol[ind].x
end

@inline function getSolp(br::ContResult, ind::Int)
	@assert hassolution(br) "You did not record the solution in the branch. Please set `saveSolEveryStep` in `ContinuationPar`"
	return br.sol[ind].p
end

function Base.getproperty(br::ContResult, s::Symbol)
	if s in (:specialpoint, :contparams, :lens, :sol, :kind, :branch, :eig, :prob, :alg)
		getfield(br, s)
	else
		getproperty(br.branch, s)
	end
end

@inline kernelDim(br::ContResult, ind) = kernelDim(br.specialpoint[ind])

"""
$(SIGNATURES)

Return the eigenvalues of the ind-th continuation step. `verbose` is used to tell the number of unstable eigen elements.
"""
function eigenvals(br::AbstractBranchResult, ind::Int, verbose::Bool = false)
	@assert br.eig[ind+1].step == ind "Error in indexing eigenvalues. Please open an issue on the website."
	if verbose
		println("--> For ", getLensSymbol(br), " = ", br.branch[ind].param)
		println("--> There are ", br.branch[ind].n_unstable, " unstable eigenvalues")
		println("--> Eigenvalues for continuation step ", br.eig[ind+1].step)
	end
	~br.eig[ind+1].converged && @error "Eigen solver did not converged on the step!!"
	br.eig[ind+1].eigenvals
end

"""
$(SIGNATURES)

Return the eigenvalues of the ind-th bifurcation point.
"""
eigenvalsfrombif(br::AbstractBranchResult, ind::Int) = br.eig[br.specialpoint[ind].idx].eigenvals

"""
$(SIGNATURES)

Return the indev-th eigenvectors of the ind-th continuation step.
"""
function eigenvec(br::AbstractBranchResult, ind::Int, indev::Int)
	@assert br.eig[ind+1].step == ind "Error in indexing eigenvalues. Please open an issue on the website."
	return geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[ind+1].eigenvecs, indev)
end

@inline getLensSymbol(br::AbstractBranchResult) = getLensSymbol(getLens(br))

function Base.show(io::IO, br::ContResult; comment = "", prefix = " ")
	print(io, prefix * "┌─ Curve type: ")
	printstyled(io, typeof(br.kind).name.name, comment, "\n", color=:cyan, bold = true)
	println(io, prefix * "├─ Number of points: ", length(br.branch))
	print(io, prefix * "├─ Type of vectors: ")
	printstyled(io, getvectortype(br), color=:cyan, bold = true)
	print(io, "\n" * prefix * "├─ Parameter ")
	printstyled(io, getLensSymbol(br), color=:cyan, bold = true)
	println(io, " starts at ", br.branch[1].param, ", ends at ", br.branch[end].param,)
	print(io, prefix * "├─ Algo: ")
	printstyled(io, typeof(br.alg).name.name, "\n", color=:cyan, bold = true)
	if length(br.specialpoint) > 0
		println(io, prefix * "└─ Special points:\n\nIf `br` is the name of the branch,\nind_ev = index of the bifurcating eigenvalue e.g. `br.eig[idx].eigenvals[ind_ev]`\n")
		for ii in eachindex(br.specialpoint)
			_show(io, br.specialpoint[ii], ii, String(getLensSymbol(br)))
		end
	end
end

# this function is important in that it gives the eigenelements corresponding to bp and stored in br. We do not check that bp ∈ br for speed reasons
getEigenelements(br::ContResult{Tkind, Tbr, Teigvals, Teigvec, Biftype, Tsol, Tparc, Tprob, Talg}, bp::Biftype) where {Tkind, Tbr, Teigvals, Teigvec, Biftype, Tsol, Tparc, Tprob, Talg} = br.eig[bp.idx]

"""
$(SIGNATURES)
Function is used to initialize the composite type `ContResult` according to the options contained in `contParams`

# Arguments
- `br` result from `getStateSummary`
- `par`: parameters
- `lens`: lens to specify the continuation parameter
- `eiginfo`: eigen-elements (eigvals, eigvecs)
"""
 function _ContResult(prob, alg, printsol, br, x0, τ, eiginfo, contParams::ContinuationPar{T, S, E}, computeEigElements::Bool, kind::AbstractContinuationKind) where {T, S, E}
	# example of bifurcation point
	bif0 = SpecialPoint(x0, τ, T, namedprintsol(printsol))
	# shall we save full solution?
	sol = contParams.saveSolEveryStep > 0 ? [(x = x0, p = getParam(prob), step = 0)] : nothing
	n_unstable = 0; n_imag = 0; stability = true

	if computeEigElements
		evvectors = saveEigenvectors(contParams) ? eiginfo[2] : nothing
		_evvectors = (eigenvals = eiginfo[1], eigenvecs = evvectors, converged = true, step = 0)
	else
		_evvectors = (eigenvals = nothing, eigenvecs = nothing, converged = true, step = 0)
	end
	return ContResult(
		branch = StructArray([br]),
		specialpoint = Vector{typeof(bif0)}(undef, 0),
		eig = computeEigElements ? [_evvectors] : empty([_evvectors]),
		sol = sol,
		contparams =  contParams,
		alg = alg,
		kind = kind,
		prob = prob)
end
####################################################################################################
"""
$(TYPEDEF)

A Branch is a structure which encapsulates the result of the computation of a branch bifurcating from a bifurcation point.

$(TYPEDFIELDS)
"""
struct Branch{Tkind, Tprob, T <: Union{ContResult, Vector{ContResult}}, Tbp} <: AbstractResult{Tkind, Tprob}
	"Set of branches branching off the bifurcation point `bp`"
	γ::T
	"Bifurcation point. It is thought as the root of the branches in γ"
	bp::Tbp
	function Branch(γ::Tγ, bp) where {Tkind, Tprob, Tγ <: Union{ AbstractResult{Tkind, Tprob}, Vector{ AbstractResult{Tkind, Tprob}} }}
		return new{Tkind, Tprob, typeof(γ), typeof(bp)}(γ, bp)
	end
end

Base.length(br::Branch) = length(br.γ)
@inline kernelDim(br::Branch, ind) = kernelDim(br.γ, ind)

"""
$(SIGNATURES)

Return the bifurcation point of a `::Branch`.
"""
from(br::Branch) = br.bp
from(br::Vector{Branch}) = length(br) > 0 ? from(br[1]) : nothing
from(tree::ContResult) = nothing
getfirstusertype(br::Branch) = getfirstusertype(br.γ)
Base.show(io::IO, br::Branch{Tk, Tp, T, Tbp}; k...) where {Tk, Tp, T <: ContResult, Tbp} = show(io, br.γ; comment = " from $(type(br.bp)) bifurcation point.", k...)
Base.lastindex(br::Branch) = lastindex(br.γ)
@inline getParams(br::Branch) = getParams(br.γ)

# extend the getproperty for easy manipulation of a Branch
# for example, it allows to use the plot recipe for ContResult as is
Base.getproperty(br::Branch, s::Symbol) = s in (:γ, :bp) ? getfield(br, s) : getproperty(br.γ, s)
Base.getindex(br::Branch, k::Int) = getindex(br.γ, k)
####################################################################################################
_reverse!(x) = reverse!(x)
_reverse!(::Nothing) = nothing
function _reverse(br0::ContResult)
	br = deepcopy(br0)
	nb = length(br.branch)
	if ~isnothing(br.branch)
		@set! br.branch =
			StructArray([setproperties(pt; step = nb - pt.step - 1) for pt in Iterators.reverse(br.branch)])
	end

	if ~isnothing(br.specialpoint)
		@set! br.specialpoint =
			[setproperties(pt;
				step = nb - pt.step - 1,
				idx = nb - pt.idx + 1,
				δ = (-pt.δ[1], -pt.δ[2])) for pt in Iterators.reverse(br.specialpoint)]
	end

	if ~isnothing(br.eig)
		@set! br.eig =
			[setproperties(pt; step = nb - pt.step - 1) for pt in Iterators.reverse(br.eig)]
	end

	if ~isnothing(br.sol)
		@set! br.sol =
			[setproperties(pt; step = nb - pt.step - 1) for pt in Iterators.reverse(br.sol)]
	end
	return br
end

_append!(x, y) = append!(x, y)
_append!(x, ::Nothing) = nothing

"""
$(SIGNATURES)

Merge two `ContResult`s and put the result in `br`.
"""
function _cat!(br::ContResult, br2::ContResult)
	# br = deepcopy(br1)
	nb = length(br.branch)
	if ~isnothing(br.branch)
		append!(br.branch,
			[setproperties(pt; step = nb + pt.step) for pt in br2.branch])
	end
	if ~isnothing(br.specialpoint)
		append!(br.specialpoint,
			[setproperties(pt;
				step = nb + pt.step,
				idx = nb + pt.idx) for pt in br2.specialpoint])
	end

	if ~isnothing(br.eig)
		append!(br.eig,
			[setproperties(pt; step = nb + pt.step) for pt in br2.eig])
	end

	if ~isnothing(br.sol)
		append!(br.sol,
			[setproperties(pt; step = nb + pt.step) for pt in br2.sol])
	end
	return br
end

# _catrev(br1::ContResult, br2::ContResult) = _merge!(_reverse(br1), br2)
# _cat(br1::ContResult, br2::ContResult) = _merge!(deepcopy(br1), br2)

"""
Same as _cat! but determine the ordering so that the branches merge properly
"""
function _merge(br1::ContResult, br2::ContResult; tol = 1e-6)
	# find the intersection point
	dst(x1, p1, x2, p2) = max(abs(x1 - x2), abs(p1 - p2))
	dst(i, j) = dst(br1.branch[i][1], br1.branch[i].param, br2.branch[j][1], br2.branch[j].param)
	ind = (1, 1)
	for i in [1, length(br1)], j in [1, length(br2)]
		if dst(i, j) < tol
			ind = (i, j)
			break
		end
	end

	if ind[1] == 1
		if ind[2] == 1
			return _cat!(_reverse(br2), br1)
		else
			return _cat!((br2), br1)
		end
	else
		if ind[2] == 1
			return _cat!(br1, br2)
		else
			return _cat!(br1, _reverse(br2))
		end

	end

	if minimum(br1.branch.param) < minimum(br2.branch.param)
		@debug "b1-b2"
		# br1 is the first branch and then br2
		# we need to look at the indexing
		return _cat!(_reverse(br1), br2)
	else
		@debug "b2-b1"
		return _cat!(_reverse(br1), br2)
	end
end
