import Base: size, show
mutable struct BifDiagNode{Tγ, Tc}
	level::Int64
	γ::Tγ
	child::Tc
end

hasbranch(tree::BifDiagNode) = ~isnothing(tree.γ)
add!(tree::BifDiagNode, γ::BranchResult, l::Int) = push!(tree.child, BifDiagNode(l, γ, BifDiagNode[]))
add!(tree::BifDiagNode, γ::Vector{ <: BranchResult}, l::Int) = map(x->add!(tree,x,l),γ)
add!(tree::BifDiagNode, γ::Nothing, l::Int) = nothing
getContResult(br::ContResult) = br
getContResult(br::Branch) = br.γ

# total size of the tree
_size(tree::BifDiagNode) = length(tree.child) > 0 ? 1 + mapreduce(size, +, tree.child) : 1
size(tree::BifDiagNode, code = ()) = _size(getBranch(tree, code))

"""
$(TYPEDEF)

Return the part of the tree (bifurcation diagram) by recursively descending the tree using the `Int` valued tuple `code`. For example `getBranch(tree, (1,2,3,))` returns `tree.child[1].child[2].child[3]`.
"""
function getBranch(tree::BifDiagNode, code)
	isempty(code) && return tree
	return getBranch(tree.child[code[1]], code[2:end])
end

show(io::IO, tree::BifDiagNode) = (println(io, "Bifurcation diagram. Root branch (level $(tree.level)) has $(length(tree.child)) children and is such that:"); show(io, tree.γ))

"""
options = (x, p, level) -> contparams

kwargs... Should be  options for `continuation` with branch switching. See
"""
function bifurcationdiagram(F, dF, d2F, d3F, x0, par0, lens::Lens, level::Int, options; usedeflation = false, kwargs...)
	γ, u = continuation(F, dF, x0, par0, lens, options(x0, par0, 1); kwargs...)
	bifurcationdiagram(F, dF, d2F, d3F, γ, level, options; code = "0", usedeflation = usedeflation, kwargs...)
end

function bifurcationdiagram(F, dF, d2F, d3F, br::ContResult, level::Int, options; usedeflation = false, kwargs...)
	printstyled(color = :magenta, "#"^50 * "\n-- > Automatic computation of bifurcation diagram\n\n")
	bifurcationdiagram!(F, dF, d2F, d3F, BifDiagNode(1, br, BifDiagNode[]), (current = 1, maxlevel = level), options; code = "0", usedeflation = usedeflation, kwargs...)
end

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
			plotSolution = (x, p; kws...) -> (plotfunc(x, p; ylabel = code*"-$_id", xlabel = "level = $(_level[1]+1), dim = $(kerneldim(_pt))", label="", kws...);plot!(node.γ; subplot = 1, legend=:topleft, putbifptlegend = false, markersize =3)))
	end

	for (id, pt) in enumerate(node.γ.bifpoint)
		# we put this condition in case the bifpoint at step = 0 corresponds to the one where are branching from. If we remove this, we keep computing the same branch (possibly).
		if pt.step > 1
			try
				println("#"^50*"\n--> New branch level = $(level[1]+1), dim = $(kerneldim(pt)), code = $code")
				γ, = letsbranch(id, pt, level)
				add!(node, γ, level.current+1)
				 ~isnothing(γ) && printstyled(color = :green, "----> From ", type(from(γ)), "\n")

				# in the case of a Transcritical bifurcation, we compute the other branch
				if ~isnothing(γ) && ~(γ isa Vector) && (from(γ) isa Transcritical)
					γ, = letsbranch(id, pt, level, -1)
					add!(node, γ, level.current+1)
				end

			catch ex
				@error ex
			# 	return node
			end
		end
	end
	for (ii, _node) in enumerate(node.child)
		bifurcationdiagram!(F, dF, d2F, d3F, _node, (@set level.current += 1), options; code = code*"-$ii", kwargs...)
	end
	return node
end


####################################################################################################
# plot recipes for the bifurcation diagram

@recipe function f(bd::Vector{BifDiagNode}; code = (), level = (-Inf, Inf))
	for b in bd
		@series begin
			level --> level
			code --> code
			b
		end
	end
end

@recipe function f(bd::BifDiagNode; code = (), level = (-Inf, Inf))
	if ~hasbranch(bd); return; end
	_bd = getBranch(bd, code)
	@series begin
		level --> level
		code --> ()
		_bd.child
	end
	# !! plot root branch in last so the bifurcation points do not alias, for example a 2d BP would be plot as a 1d BP if the order were reversed
	if level[1] <= _bd.level <= level[2]
		@series begin
			_bd.γ
		end
	end
end

@recipe function f(bd::Nothing)
	nothing
end
