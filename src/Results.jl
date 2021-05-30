abstract type AbstractBranchResult end

####################################################################################################
namedprintsol(x) = (x = x,)
namedprintsol(x::Real) = (x = x,)
namedprintsol(x::NamedTuple) = x
namedprintsol(x::Tuple) = (;zip((Symbol("x$i") for i in 1:length(x)), x)...)
mergefromuser(x, a::NamedTuple) = merge(namedprintsol(x), a)
####################################################################################################
# Structure to hold continuation result
"""
$(TYPEDEF)

Structure which holds the results after a call to [`continuation`](@ref).

You can see the propertynames of a result by using `propertynames(::ContResult)` or by typing `br.` + TAB where `br::ContResult`.

# Fields

$(TYPEDFIELDS)

# Associated methods
- `length(br)` number of the continuation steps
- `eigenvals(br, ind)` returns the eigenvalues for the ind-th continuation step
- `eigenvec(br, ind, indev)` returns the indev-th eigenvector for the ind-th continuation step
- `br[k+1]` gives information about the k-th step
"""
@with_kw_noshow struct ContResult{Ta, Teigvals, Teigvec, Biftype, Ts, Tfunc, Tpar, Tl <: Lens} <: AbstractBranchResult
	"holds the low-dimensional information about the branch. More precisely, `branch[:, i+1]` contains the following information `(printSolution(u, param), param, itnewton, itlinear, ds, theta, n_unstable, n_imag, stable, step)` for each continuation step `i`.\n
  - `itnewton` number of Newton iterations
  - `itlinear` total number of linear iterations during corrector
  - `n_unstable` number of eigenvalues with positive real part for each continuation step (to detect stationary bifurcation)
  - `n_imag` number of eigenvalues with positive real part and non zero imaginary part for each continuation step (to detect Hopf bifurcation).
  - `stable`  stability of the computed solution for each continuation step. Hence, `stable` should match `eig[step]` which corresponds to `branch[k]` for a given `k`.
  - `step` continuation step (here equal `i`)"
	branch::StructArray{Ta}

	"A vector with eigen-elements at each continuation step."
	eig::Vector{NamedTuple{(:eigenvals, :eigenvec, :step), Tuple{Teigvals, Teigvec, Int64}}}

	"Vector of solutions sampled along the branch. This is set by the argument `saveSolEveryNsteps::Int64` (default 0) in [`ContinuationPar`](@ref)."
	sol::Ts

	"The parameters used for the call to `continuation` which produced this branch."
	contparams::ContinuationPar

	"Type of solutions computed in this branch."
	type::Symbol = :Equilibrium

	"Structure associated to the functional, useful for branch switching. For example, when computing periodic orbits, the functional `PeriodicOrbitTrapProblem`, `ShootingProblem`... will be saved here."
	functional::Tfunc = nothing

	"Parameters passed to continuation and used in the equation `F(x, par) = 0`."
	params::Tpar = nothing

	"Parameter axis used for computing the branch"
	lens::Tl

	"A vector holding the set of detected bifurcation points. See [`SpecialPoint`](@ref) for a description of the fields."
	specialpoint::Vector{Biftype}
end

# returns the number of steps in a branch
Base.length(br::AbstractBranchResult) = length(br.branch)

# check whether the eigenvalues are saved in the branch
# this is a good test bifucause we always fill br.eig with a dummy vector :(
@inline haseigenvalues(br::ContResult) = hasstability(br)
@inline haseigenvalues(br::AbstractBranchResult) = haseigenvalues(br.γ)

# check whether the eigenvectors are saved in the branch
@inline haseigenvector(br::ContResult{Ta, Teigvals, Teigvec, Biftype, Ts, Tfunc, Tpar, Tl} ) where {Ta, Teigvals, Teigvec, Biftype, Ts, Tfunc, Tpar, Tl } = Teigvec != Nothing
@inline haseigenvector(br::AbstractBranchResult) = haseigenvector(br.γ)

@inline hasstability(br::AbstractBranchResult) = typeof(br.branch).parameters[1].parameters[2].parameters[end-1] == Bool


getfirstusertype(br::AbstractBranchResult) = keys(br.branch[1])[1]
@inline getvectortype(br::AbstractBranchResult) = getVectorType(eltype(br.specialpoint))
@inline getvectoreltype(br::AbstractBranchResult) = eltype(getvectortype(br))
setParam(br::AbstractBranchResult, p0) = set(br.params, br.lens, p0)
Base.getindex(br::ContResult, k::Int) = (br.branch[k]..., eigenvals = haseigenvalues(br) ? br.eig[k].eigenvals : nothing, eigenvec = haseigenvector(br) ? br.eig[k].eigenvec : nothing)

function Base.getproperty(br::ContResult, s::Symbol)
	if s in (:specialpoint, :contparams, :lens, :sol, :type, :branch, :eig, :functional, :params)
		getfield(br, s)
	else
		getproperty(br.branch, s)
	end
end
Base.propertynames(br::ContResult) = (propertynames(br.branch)..., :specialpoint, :contparams, :lens, :sol, :type, :branch, :eig, :functional, :params)

"""
$(SIGNATURES)

Return the eigenvalues of the ind-th continuation step.
"""
function eigenvals(br::AbstractBranchResult, ind::Int, verbose::Bool = false)
	if verbose
		println("--> There are ", br.branch[ind].n_unstable, " unstable eigenvalues")
	end
	br.eig[ind].eigenvals
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
eigenvec(br::AbstractBranchResult, ind::Int, indev::Int) = geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[ind].eigenvec, indev)
@inline kernelDim(br::ContResult, ind) = kernelDim(br.specialpoint[ind])

function Base.show(io::IO, br::ContResult, comment = "")
	println(io, "Branch number of points: ", length(br.branch))
	println(io, "Branch of ", br.type, comment)
	println(io, "Parameters ", getLensParam(br.lens), " from ", br.branch[1].param, " to ", br.branch[end].param,)
	if length(br.specialpoint) > 0
		println(io, "Special points:\n (ind_ev = index of the bifurcating eigenvalue e.g. `br.eig[idx].eigenvals[ind_ev]`)")
		for ii in eachindex(br.specialpoint)
			_show(io, br.specialpoint[ii], ii, String(getLensParam(br.lens)))
		end
	end
end

# this function is important in that it gives the eigenelements corresponding to bp and stored in br. We do not check that bp ∈ br for speed reasons
getEigenelements(br::ContResult{T, Teigvals, Teigvec, Biftype, Ts, Tfunc, Tpar, Tl}, bp::Biftype) where {T, Teigvals, Teigvec, Biftype, Ts, Tfunc, Tpar, Tl} = br.eig[bp.idx]

"""
$(SIGNATURES)
Function is used to initialize the composite type `ContResult` according to the options contained in `contParams`

# Arguments
- `br` result from `getStateSummary`
- `par`: parameters
- `lens`: lens to specify the continuation parameter
- `eiginfo`: eigen-elements (eigvals, eigvecs)
"""
 function _ContResult(printsol, br, x0, par, lens::Lens, eiginfo, contParams::ContinuationPar{T, S, E}, computeEigElements::Bool) where {T, S, E}
	# example of bifurcation point
	bif0 = SpecialPoint(x0, T, namedprintsol(printsol))
	# shall we save full solution?
	sol = contParams.saveSolEveryStep > 0 ? [(x = copy(x0), p = get(par, lens), step = 0)] : nothing
	n_unstable = 0; n_imag = 0; stability = true

	if computeEigElements
		evvectors = contParams.saveEigenvectors ? eiginfo[2] : nothing
		_evvectors = (eigenvals = eiginfo[1], eigenvec = evvectors, step = 0)
	else
		_evvectors = (eigenvals = nothing, eigenvec = nothing, step = 0)
	end
	return ContResult(
		branch = StructArray([br]),
		specialpoint = Vector{typeof(bif0)}(undef, 0),
		eig = computeEigElements ? [_evvectors] : empty([_evvectors]),
		sol = sol,
		contparams =  contParams,
		params = par,
		lens = lens)
end

function _ContResult()
	ContResult(branch = StructArray([(a=1,b=2)]), eig = [(eigenvals = Any[], eigenvec=Any[],step=0)],sol=Any[],contparams = opts_br, lens=(@lens _[1]), bifpoint=Any[]);
end
####################################################################################################
"""
$(TYPEDEF)

A Branch is a structure which encapsulates the result of the computation of a branch bifurcating from a bifurcation point.

$(TYPEDFIELDS)
"""
struct Branch{T <: Union{ContResult, Vector{ContResult}}, Tbp} <: AbstractBranchResult
	γ::T
	bp::Tbp
end

Base.length(br::Branch) = length(br.γ)
from(br::Branch) = br.bp
from(br::Vector{Branch}) = length(br) > 0 ? from(br[1]) : nothing
getfirstusertype(br::Branch) = getfirstusertype(br.γ)
Base.show(io::IO, br::Branch{T, Tbp}) where {T <: ContResult, Tbp} = show(io, br.γ, " from $(type(br.bp)) bifurcation point.")

# extend the getproperty for easy manipulation of a Branch
# for example, it allows to use the plot recipe for ContResult as is
Base.getproperty(br::Branch, s::Symbol) = s in (:γ, :bp) ? getfield(br, s) : getproperty(br.γ, s)
Base.propertynames(br::Branch) = ((:γ, :bp)..., propertynames(br.γ)...)
Base.getindex(br::Branch, k::Int) = getindex(br.γ, k)
####################################################################################################
_reverse!(x) = reverse!(x)
_reverse!(::Nothing) = nothing
function _reverse(br0::ContResult)
	br = deepcopy(br0)
	nb = length(br.branch)
	if ~isnothing(br.branch)
		br = @set br.branch =
			StructArray([setproperties(pt; step = nb - pt.step - 1) for pt in Iterators.reverse(br.branch)])
	end

	if ~isnothing(br.specialpoint)
		br = @set br.specialpoint =
			[setproperties(pt;
				step = nb - pt.step - 1,
				idx = nb - pt.idx + 1,
				δ = (-pt.δ[1], -pt.δ[2])) for pt in Iterators.reverse(br.specialpoint)]
	end

	if ~isnothing(br.eig)
		br = @set br.eig =
			[setproperties(pt; step = nb - pt.step - 1) for pt in Iterators.reverse(br.eig)]
	end

	if ~isnothing(br.sol)
		br = @set br.sol =
			[setproperties(pt; step = nb - pt.step - 1) for pt in Iterators.reverse(br.sol)]
	end
	return br
end

_append!(x,y) = append!(x,y)
_append!(x,::Nothing) = nothing
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
	dst(x1,p1,x2,p2) = max(abs(x1-x2),abs(p1-p2))
	dst(i,j) = dst(br1.branch[i][1],br1.branch[i].param,br2.branch[j][1],br2.branch[j].param)
	ind = (1,1)
	for i in [1,length(br1)], j in [1,length(br2)]
		if dst(i,j) < tol
			ind = (i,j)
			break
		end
	end

	if ind[1] == 1
		if ind[2] == 1
			return _cat!(_reverse(br2),br1)
		else
			return _cat!((br2),br1)
		end
	else
		if ind[2] == 1
			return _cat!(br1,br2)
		else
			return _cat!(br1,_reverse(br2))
		end

	end

	if minimum(br1.branch.param) < minimum(br2.branch.param)
		@info "b1-b2"
		# br1 is the first branch and then br2
		# we need to look at the indexing
		return _cat!(_reverse(br1), br2)
	else
		@info "b2-b1"
		return _cat!(_reverse(br1), br2)
	end
end
