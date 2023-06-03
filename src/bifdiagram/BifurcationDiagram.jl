mutable struct BifDiagNode{Tγ, Tc}
	# current level of recursion
	level::Int64

	# code for finding this node in the tree, this is the index of the bifurcation point
	# from which γ branches off
	code::Int64

	# branch associated to the current node
	γ::Tγ

	# childs of current node. These are the different branches off the bifurcation points
	# in γ
	child::Tc
end

# getters
@inline level(node::BifDiagNode) = node.level
hasbranch(tree::BifDiagNode) = ~isnothing(tree.γ)
from(tree::BifDiagNode) = from(tree.γ)
add!(tree::BifDiagNode, γ::AbstractBranchResult, level::Int, code::Int) = push!(tree.child, BifDiagNode(level, code, γ, BifDiagNode[]))
add!(tree::BifDiagNode, γ::Vector{ <: AbstractBranchResult}, level::Int, code::Int) = map(x -> add!(tree, x, level, code), γ)
add!(tree::BifDiagNode, γ::Nothing, level::Int, code::Int) = nothing
getContResult(br::ContResult) = br
getContResult(br::Branch) = br.γ
getAlg(tree::BifDiagNode) = tree.γ.alg

function Base.show(io::IO, tree::BifDiagNode)
	println(io, "[Bifurcation diagram]")
	println(io, " ┌─ From $(tree.code)-th bifurcation point.")
	println(io, " ├─ Children number: $(length(tree.child))" );
	println(io, " └─ Root (recursion level $(tree.level))")
	show(io, tree.γ; prefix = "      ")
end

# total size of the tree
_size(tree::BifDiagNode) = length(tree.child) > 0 ? 1 + mapreduce(size, +, tree.child) : 1

"""
$(SIGNATURES)

Return the size of the bifurcation diagram. The argument `code` is the same as in `getBranch`.
"""
Base.size(tree::BifDiagNode, code = ()) = _size(getBranch(tree, code))

"""
$(SIGNATURES)

Return the part of the tree (bifurcation diagram) by recursively descending down the tree using the `Int` valued tuple `code`. For example `getBranch(tree, (1,2,3,))` returns `tree.child[1].child[2].child[3]`.
"""
function getBranch(tree::BifDiagNode, code)
	isempty(code) && return tree
	return getBranch(tree.child[code[1]], code[2:end])
end

"""
$(SIGNATURES)

Return the part of the tree corresponding to the indbif-th bifurcation point on the root branch.
"""
function getBranchesFromBP(tree::BifDiagNode, indbif::Int)
	# parameter value at the bp
	p = tree.γ.specialpoint[indbif].param
	return BifDiagNode[br for br in tree.child if from(br).p == p]
end

"""
$(SIGNATURES)

Compute the bifurcation diagram associated with the problem `F(x, p) = 0` recursively.

# Arguments
- `prob::AbstractBifurcationProblem` bifurcation problem
- `alg` continuation algorithm
- `level` maximum branching (or recursion) level for computing the bifurcation diagram
- `options = (x, p, level) -> contparams` this function allows to change the [`continuation`](@ref) options depending on the branching `level`. The argument `x, p` denotes the current solution to `F(x, p)=0`.
- `kwargs` optional arguments. Look at [`bifurcationdiagram!`](@ref) for more details.

# Simplified call:

We also provide the method

`bifurcationdiagram(prob, br::ContResult, level::Int, options; kwargs...)`

where `br` is a branch computed after a call to [`continuation`](@ref) from which we want to compute the bifurcating branches recursively.
"""
function bifurcationdiagram(prob::AbstractBifurcationProblem, alg::AbstractContinuationAlgorithm, level::Int, options; kwargs...)
	γ = continuation(prob, alg, options(prob.u0, prob.params, 1); kwargs...)
	bifurcationdiagram(prob, γ, level, options; code = "0", kwargs...)
end

# TODO, BifDiagNode[] makes it type unstable it seems
function bifurcationdiagram(prob::AbstractBifurcationProblem, br::AbstractBranchResult, maxlevel::Int, options; kwargs...)
	printstyled(color = :magenta, "#"^50 * "\n───> Automatic computation of bifurcation diagram\n\n")
	bifurcationdiagram!(prob, BifDiagNode(1, 0, br, BifDiagNode[]), maxlevel, options; code = "0", kwargs...)
end

"""
$(SIGNATURES)

Similar to [`bifurcationdiagram`](@ref) but you pass a previously computed `node` from which you want to further compute the bifurcated branches. It is usually used with `node = getBranch(diagram, code)` from a previously computed bifurcation `diagram`.

# Arguments
- `node::BifDiagNode` a node in the bifurcation diagram
- `maxlevel = 1` required maximal level of recursion.
- `options = (x, p, level) -> contparams` this function allows to change the [`continuation`](@ref) options depending on the branching `level`. The argument `x, p` denotes the current solution to `F(x, p)=0`.

# Optional arguments
- `code = "0"` code used to display iterations
- `usedeflation = false`
- `halfbranch = false` for Pitchfork/Transcritical bifurcations, compute only half of the branch. Can be useful when there are symmetries.
- `kwargs` optional arguments as for [`continuation`](@ref) but also for the different versions listed in [Continuation](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/library/#Continuation-1).
"""
function bifurcationdiagram!(prob::AbstractBifurcationProblem,
		node::BifDiagNode,
		maxlevel::Int,
		options;
		code = "0",
		usedeflation = false,
		halfbranch = false,
		kwargs...)
	if node.level >= maxlevel || isnothing(node.γ); return node; end

	# current level of recursion
	level = node.level

	# convenient function for branching
	function letsbranch(_id, _pt, _level; _dsfactor = 1, _ampfactor = 1)
		plotfunc = get(kwargs, :plotSolution, (x, p; kws...) -> plot!(x; kws...))
		optscont = options(_pt.x, _pt.param, _level + 1)
		@set! optscont.ds *= _dsfactor

		function plotSolBD(x, p; kws...)
			plotfunc(x, p; ylabel = code*"-$_id", xlabel = "level = $(_level+1), dim = $(kernelDim(_pt))", label="", kws...)
			plot!(node.γ; subplot = 1, legend=:topleft, putspecialptlegend = false, markersize = 2)
		end

		continuation(getContResult(node.γ), _id, optscont;
			nev = optscont.nev, kwargs...,
			ampfactor = _ampfactor,
			usedeflation = usedeflation,
			plotSolution = plotSolBD
		)
	end

	for (id, pt) in enumerate(node.γ.specialpoint)
		# we put this condition in case the specialpoint at step = 0 corresponds to the one we are branching from. If we remove this, we keep computing the same branch (possibly).
		if pt.step > 1 && pt.type in (:bp, :nd)
			try
				println("─"^80*"\n──> New branch, level = $(level+1), dim(Kernel) = ", kernelDim(pt), ", code = $code, from bp #",id," at p = ", pt.param, ", type = ", type(pt))
				γ = letsbranch(id, pt, level)
				add!(node, γ, level+1, id)
				 ~isnothing(γ) && printstyled(color = :green, "────> From ", type(from(γ)), "\n")

				# in the case of a Transcritical bifurcation, we compute the other branch
				if ~isnothing(γ) && ~(γ isa Vector)
					if ~halfbranch && from(γ) isa Transcritical
						γ = letsbranch(id, pt, level; _dsfactor = -1)
						add!(node, γ, level+1, id)
					end
					if ~halfbranch && from(γ) isa Pitchfork
						γ = letsbranch(id, pt, level; _ampfactor = -1)
						add!(node, γ, level+1, id)
					end
				end

			catch ex
				# @error ex
			# 	return node
			end
		end
	end
	for (ii, _node) in enumerate(node.child)
		bifurcationdiagram!(prob, _node, maxlevel, options; code = code*"-$ii", kwargs...)
	end
	return node
end
