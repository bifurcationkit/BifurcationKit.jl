mutable struct BifDiagNode{Tγ, Tc}
	level::Int64
	γ::Tγ
	child::Tc
end

hasbranch(tree::BifDiagNode) = ~isnothing(tree.γ)
from(tree::BifDiagNode) = from(tree.γ)
add!(tree::BifDiagNode, γ::AbstractBranchResult, l::Int) = push!(tree.child, BifDiagNode(l, γ, BifDiagNode[]))
add!(tree::BifDiagNode, γ::Vector{ <: AbstractBranchResult}, l::Int) = map(x->add!(tree,x,l),γ)
add!(tree::BifDiagNode, γ::Nothing, l::Int) = nothing
getContResult(br::ContResult) = br
getContResult(br::Branch) = br.γ
Base.show(io::IO, tree::BifDiagNode) = (println(io, "Bifurcation diagram. Root branch (level $(tree.level)) has $(length(tree.child)) children and is such that:"); show(io, tree.γ))

# total size of the tree
_size(tree::BifDiagNode) = length(tree.child) > 0 ? 1 + mapreduce(size, +, tree.child) : 1

"""
$(SIGNATURES)

Return the size of the bifurcation diagram. The arguement `code` is the same as in `getBranch`.
"""
Base.size(tree::BifDiagNode, code = ()) = _size(getBranch(tree, code))

"""
$(SIGNATURES)

Return the part of the tree (bifurcation diagram) by recursively descending the tree using the `Int` valued tuple `code`. For example `getBranch(tree, (1,2,3,))` returns `tree.child[1].child[2].child[3]`.
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
	p = tree.γ.bifpoint[indbif].param
	return BifDiagNode[br for br in tree.child if from(br).p == p]
end

"""
$(SIGNATURES)

Compute the bifurcation diagram associated with the problem `F(x, p) = 0` recursively.

# Arguments
- `F, dF, d2F, d3F` functional and its derivatives
- `x0` initial guess
- `par0` parameter values at `x0`
- `lens` lens to select the parameter axis
- `level` maximum branching (or recursion) level for computing the bifurcation diagram
- `options = (x, p, level) -> contparams` this function allows to change the [`continuation`](@ref) options depending on the branching `level`. The argument `x, p` denotes the current solution to `F(x,p)=0`.
- `kwargs` optional arguments as for [`continuation`](@ref) but also for the different versions listed in [Continuation](https://rveltz.github.io/BifurcationKit.jl/dev/library/#Continuation-1).

# Simplified call:

We also provide the call

`bifurcationdiagram(F, dF, d2F, d3F, br::ContResult, level::Int, options; usedeflation = false, kwargs...)`

where `br` is a branch computed after a call to [`continuation`](@ref) from which we want to compute the bifurcating branches recursively.
"""
function bifurcationdiagram(F, dF, d2F, d3F, x0, par0, lens::Lens, level::Int, options; usedeflation = false, kwargs...)
	γ, u = continuation(F, dF, x0, par0, lens, options(x0, par0, 1); kwargs...)
	bifurcationdiagram(F, dF, d2F, d3F, γ, level, options; code = "0", usedeflation = usedeflation, kwargs...)
end

# TODO, BifDiagNode[] makes it type unstable it seems
function bifurcationdiagram(F, dF, d2F, d3F, br::AbstractBranchResult, level::Int, options; usedeflation = false, kwargs...)
	printstyled(color = :magenta, "#"^50 * "\n---> Automatic computation of bifurcation diagram\n\n")
	bifurcationdiagram!(F, dF, d2F, d3F, BifDiagNode(1, br, BifDiagNode[]), (current = 1, maxlevel = level), options; code = "0", usedeflation = usedeflation, kwargs...)
end

"""
$(SIGNATURES)

Same as [`bifurcationdiagram`](@ref) but you pass a previously computed bifurcation diagram `node` from which you want to further compute the bifurcated branches. It is usually used with `node = getBranch(diagram, code)` from a previously computed bifurcation `diagram`.
"""
function bifurcationdiagram!(F, dF, d2F, d3F, node::BifDiagNode, level::NamedTuple{(:current, :maxlevel),Tuple{Int64,Int64}}, options; code = "0", usedeflation = false, kwargs...)
	if level[1] >= level[2] || isnothing(node.γ); return node; end

	# convenient function for branching
	function letsbranch(_id, _pt, _level, _dsfactor = 1)
		plotfunc = get(kwargs, :plotSolution, (x, p; kws...) -> plot!(x; kws...))
		optscont = options(_pt.x, _pt.param, _level.current + 1)
		optscont = @set optscont.ds *= _dsfactor

		continuation(F, dF, d2F, d3F, getContResult(node.γ), _id, optscont;
			nev = optscont.nev, kwargs...,
			usedeflation = usedeflation,
			plotSolution = (x, p; kws...) -> (plotfunc(x, p; ylabel = code*"-$_id", xlabel = "level = $(_level[1]+1), dim = $(kernelDim(_pt))", label="", kws...);plot!(node.γ; subplot = 1, legend=:topleft, putbifptlegend = false, markersize = 2)))
	end

	for (id, pt) in enumerate(node.γ.bifpoint)
		# we put this condition in case the bifpoint at step = 0 corresponds to the one we are branching from. If we remove this, we keep computing the same branch (possibly).
		if pt.step > 1
			try
				println("─"^80*"\n--> New branch, level = $(level[1]+1), dim(Kernel) = ", kernelDim(pt), ", code = $code, from bp #",id," at p = ", pt.param, ", type = ", type(pt))
				γ, = letsbranch(id, pt, level)
				add!(node, γ, level.current+1)
				 ~isnothing(γ) && printstyled(color = :green, "----> From ", type(from(γ)), "\n")

				# in the case of a Transcritical bifurcation, we compute the other branch
				if ~isnothing(γ) && ~(γ isa Vector) && (from(γ) isa Transcritical)
					γ, = letsbranch(id, pt, level, -1)
					add!(node, γ, level.current+1)
				end

			catch ex
			#	@error ex
			# 	return node
			end
		end
	end
	for (ii, _node) in enumerate(node.child)
		bifurcationdiagram!(F, dF, d2F, d3F, _node, (@set level.current += 1), options; code = code*"-$ii", kwargs...)
	end
	return node
end
