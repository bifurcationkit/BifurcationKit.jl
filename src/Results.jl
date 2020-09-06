abstract type BranchResult end

# Structure to hold result
"""
$(TYPEDEF)

Structure which holds the results after a call to [`continuation`](@ref).

$(TYPEDFIELDS)

          A vector holding the set of bifurcation points detected during the computation of the branch. Each entry of the vector contains a tuple with fields:
-
    - `type` bifurcation type, `:hopf, :bp...`,
    - `idx` is the index in `eig` (see above) for which the bifurcation occurs.
    - `param` parameter value at the bifurcation point
    - `norm` norm of the equilibrium at the bifurcation point
    - `printsol = printSolution(x, param)`
    - `x` equilibrium at the bifurcation point
    - `tau` tangent along the branch at the bifurcation point
    - `ind_ev` is the eigenvalue index responsible for the bifurcation (if applicable)
    - `step` is the continuation step at which the bifurcation occurs,
    - `status ∈ {:converged, :guess}` indicates if the bisection algorithm was successful in detecting the bifurcation point
    - `δ = (δr, δi)` where δr indicates the change in the number of unstable eigenvalues and δi indicates the change in the number of unstable eigenvalues with nonzero imaginary part. `abs(δr)` is thus an estimate of the dimension of the kernel of the Jacobian at the bifurcation point.
    - `precision` precision in location of the bifurcation point

# Associated methods
- `length(br)` number of the continuation steps
- `eigenvals(br, ind)` returns the eigenvalues for the ind-th continuation step
- `eigenvec(br, ind, indev)` returns the indev-th eigenvector for the ind-th continuation step
"""
@with_kw_noshow struct ContResult{T, Teigvals, Teigvec, Biftype, Ts, Tfunc, Tpar, Tl <: Lens} <: BranchResult
	"holds the low-dimensional information about the branch. More precisely, `branch[:,i]` contains the following information `(param, printSolution(u, param), Newton iterations, ds, theta, i)` for each continuation step `i`."
	branch::VectorOfArray{T, 2, Array{Vector{T}, 1}}

	"A vector with eigen-elements at each continuation step."
	eig::Vector{NamedTuple{(:eigenvals, :eigenvec, :step), Tuple{Teigvals, Teigvec, Int64}}}

	"A vector holding the set of fold points detected during the computation of the branch."
	foldpoint::Vector{Biftype}

	"A `Vector{Bool}` holding the stability of the computed solution for each continuation step. Hence, the stability `stability[k]` should match `eig[k]` which corresponds to `branch[k]` for a given `k`"
	stability::Vector{Bool}

	"A `Vector{Int64}` holding the number of eigenvalues with positive real part and non zero imaginary part for each continuation step (to detect Hopf bifurcation)"
	n_imag::Vector{Int64}

	"A `Vector{Int64}` holding the number of eigenvalues with positive real part for each continuation step (to detect stationary bifurcation)"
	n_unstable::Vector{Int64}

	"Vector of solutions sampled along the branch. This is set by the argument `saveSolEveryNsteps::Int64` (default 0) in [`ContinuationPar`](@ref)"
	sol::Ts

	"The parameters used for the call to `continuation` which produced this branch."
	contparams::ContinuationPar

	"Type of solutions computed in this branch."
	type::Symbol = :Equilibrium

	"Structure associated to the functional, useful for branch switching. For example, when computing periodic orbits, the functional `PeriodicOrbitTrapProblem`, `ShootingProblem`... will be saved here."
	functional::Tfunc = nothing

	"Parameters passed to continuation and used in the equation F(x, par) = 0"
	params::Tpar = nothing

	"Parameter axis used for computing the branch"
	param_lens::Tl

	# the explanation for this field is given after. Always put this field in last position
	""
	bifpoint::Vector{Biftype}

end

Base.length(br::ContResult) = length(br.branch[1, :])
haseigenvector(br::ContResult{T, Teigvals, Teigvec, Biftype, Ts, Tfunc, Tpar, Tl} ) where {T, Teigvals, Teigvec, Biftype, Ts, Tfunc, Tpar, Tl } = Teigvec != Nothing
@inline vectortype(br::BranchResult) = ((eltype(br.bifpoint)).parameters[2]).parameters[6]
eigenvals(br::BranchResult, ind) = br.eig[ind].eigenvals
eigenvec(br::BranchResult, ind, indev) = geteigenvector(br.contparams.newtonOptions.eigsolver, br.eig[ind].eigenvec, indev)

_show(io, bp, ii) = @printf(io, "- #%3i, %7s at p ≈ % 4.8f ± %1.0e, step = %3i, eigenelements in eig[%3i], ind_ev = %3i [%9s], δ = (%2i, %2i)\n", ii, bp.type, bp.param, bp.precision, bp.step, bp.idx, bp.ind_ev, bp.status, bp.δ[1], bp.δ[2])
_showFold(io, bp, ii) = @printf(io, "- #%3i, %7s at p ≈ % 4.8f, step = %3i, eigenelements in eig[%3i], ind_ev = %3i [%9s], δ = (%2i, %2i)\n", ii, bp.type, bp.param, bp.step, bp.idx, bp.ind_ev, bp.status, bp.δ[1], bp.δ[2])
@inline kerneldim(bp) = abs(bp.δ[1])
@inline kerneldim(br::ContResult, ind) = kerneldim(br.bifpoint[ind])

function Base.show(io::IO, br::ContResult, comment = "")
	println(io, "Branch number of points: ", length(br.branch))
	println(io, "Branch of ", br.type, comment)
	if length(br.bifpoint) > 0
		println(io, "Bifurcation points:\n (ind_ev = index of the bifurcating eigenvalue e.g. `br.eig[idx].eigenvals[ind_ev]`)")
		for ii in eachindex(br.bifpoint)
			_show(io, br.bifpoint[ii], ii)
		end
	end
	if length(br.foldpoint) > 0
		println(io, "Fold points:")
		for ii in eachindex(br.foldpoint)
			_showFold(io, br.foldpoint[ii], ii)
		end
	end
end

# this function is important in that it gives the eigenelements corresponding to bp and stored in br. We do not check that bp ∈ br for speed reasons
function getEigenelements(br::ContResult{T, Teigvals, Teigvec, Biftype, Ts, Tfunc, Tpar, Tl}, bp::Biftype) where {T, Teigvals, Teigvec, Biftype, Ts, Tfunc, Tpar, Tl}
	br.eig[bp.idx]
end

"""
This function is used to initialize the composite type `ContResult` according to the options contained in `contParams`
"""
 function ContResult(br, x0, par, lens::Lens, evsol, contParams::ContinuationPar{T, S, E}) where {T, S, E}
	bif0 = (type = :none, idx = 0, param = T(0), norm  = T(0), printsol = T(0), x = x0, tau = BorderedArray(x0, T(0)), ind_ev = 0, step = 0, status = :guess, δ = (0,0), precision = T(-1))
	sol = contParams.saveSolEveryStep > 0 ? [(x = copy(x0), p = br[1,1], step = 0)] : nothing
	n_unstable = 0
	n_imag = 0
	stability = true

	if computeEigenElements(contParams)
		evvectors = contParams.saveEigenvectors ? evsol[2] : nothing
		stability, n_unstable, n_imag = isstable(contParams, evsol[1])
		_evvectors = (eigenvals = evsol[1], eigenvec = evvectors, step = 0)
	else
		_evvectors = (eigenvals = evsol[1], eigenvec = nothing, step = 0)
	end

	return ContResult(
		branch = br,
		bifpoint = [bif0],
		foldpoint = [bif0],
		n_imag = [n_imag],
		n_unstable = [n_unstable],
		stability = [stability],
		eig = [_evvectors],
		sol = sol,
		contparams =  contParams,
		params = par,
		param_lens = lens)
end
####################################################################################################
"""
A Branch is a structure which encapsulates the result of the computation of a branch bifurcating from a bifurcation point.
"""
struct Branch{T <: Union{ContResult, Vector{ContResult}}, Tbp} <: BranchResult
	γ::T
	bp::Tbp
end

from(br::Branch) = br.bp
from(br::Vector{Branch}) = length(br) > 0 ? from(br[1]) : nothing
Base.show(io::IO, br::Branch{T, Tbp}) where {T <: ContResult, Tbp} = show(io, br.γ, " from $(type(br.bp)) bifurcation point.")

# extend the getproperty for easy manipulation of a Branch
# for example, it allows to use the plot recipe for ContResult as is
Base.getproperty(br::Branch, s::Symbol) = s in (:γ, :bp) ? getfield(br, s) : getproperty(br.γ, s)
Base.propertynames(br::Branch) = ((:γ, :bp)..., propertynames(br.γ)...)
####################################################################################################
