"""
$(SIGNATURES)

Structure to hold a connected component of a bifurcation diagram.

## Fields

$(TYPEDFIELDS)

## Methods

- `hasbranch(diagram)`
- `from(diagram)`
- `diagram[code]` For example `diagram[1,2,3]` returns `diagram.child[1].child[2].child[3]`
"""
mutable struct BifDiagNode{Tγ, Tc}
    "current recursion level"
    level::Int64

    "code for finding the current node in the tree, this is the index of the bifurcation point from which γ branches off"
    code::Int64

    "branch associated to the current node"
    γ::Tγ

    "children of current node. These are the different branches off the bifurcation point in γ"
    child::Tc
end

# getters
@inline level(node::BifDiagNode) = node.level
hasbranch(tree::BifDiagNode) = ~isnothing(tree.γ)
from(tree::BifDiagNode) = from(tree.γ)
add!(tree::BifDiagNode, γ::AbstractBranchResult, level::Int, code::Int) = push!(tree.child, BifDiagNode(level, code, γ, BifDiagNode[]))
add!(tree::BifDiagNode, γ::Vector{ <: AbstractBranchResult}, level::Int, code::Int) = map(x -> add!(tree, x, level, code), γ)
add!(tree::BifDiagNode, γ::Nothing, level::Int, code::Int) = nothing
get_contresult(br::ContResult) = br
get_contresult(br::Branch) = br.γ
getalg(tree::BifDiagNode) = tree.γ.alg
Base.getindex(tree::BifDiagNode, code...) = get_branch(tree, code)

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

Return the size of the bifurcation diagram. The argument `code` is the same as in `get_branch`.
"""
Base.size(tree::BifDiagNode, code = ()) = _size(get_branch(tree, code))

"""
$(SIGNATURES)

Return the part of the diagram (bifurcation diagram) by recursively descending down the diagram using the `Int` valued tuple `code`. For example `get_branch(diagram, (1,2,3,))` returns `diagram.child[1].child[2].child[3]`.
"""
function get_branch(diagram::BifDiagNode, code)
    isempty(code) && return diagram
    return get_branch(diagram.child[code[1]], code[2:end])
end

"""
$(SIGNATURES)

Return the part of the diagram corresponding to the indbif-th bifurcation point on the root branch.
"""
function get_branches_from_BP(diagram::BifDiagNode, indbif::Int)
    # parameter value at the bp
    p = diagram.γ.specialpoint[indbif].param
    return BifDiagNode[br for br in diagram.child if from(br).p == p]
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
function bifurcationdiagram(prob::AbstractBifurcationProblem,
                            alg::AbstractContinuationAlgorithm, 
                            level::Int, 
                            options;
                            linear_algo = nothing,
                            kwargs...)
    kwargs_cont = _keep_opts_cont(values(kwargs))
    γ = continuation(prob, alg, options(prob.u0, prob.params, 1); kwargs_cont..., linear_algo)
    bifurcationdiagram(prob, γ, level, options; 
                        code = (0,), 
                        kwargs...)
end

# simplified function where the user does not need to provide a function for the options
function bifurcationdiagram(prob::AbstractBifurcationProblem,
                            alg::AbstractContinuationAlgorithm, 
                            level::Int, 
                            options::ContinuationPar; 
                            kwargs...)
    bifurcationdiagram(prob, alg, level, (args...) -> options; kwargs...)
end

# TODO, BifDiagNode[] makes it type unstable it seems
function bifurcationdiagram(prob::AbstractBifurcationProblem,
                            br::AbstractBranchResult, 
                            maxlevel::Int, 
                            options; 
                            verbosediagram = false, 
                            kwargs...)
    verbose = (get(kwargs, :verbosity, 0) > 0) || verbosediagram
    verbose && printstyled(color = :magenta, "━"^50 * "\n───▶ Automatic computation of bifurcation diagram\n\n")
    bifurcationdiagram!(prob, BifDiagNode(1, 0, br, BifDiagNode[]), maxlevel, options; code = "0", verbosediagram, kwargs...)
end

"""
$(SIGNATURES)

Similar to [`bifurcationdiagram`](@ref) but you pass a previously computed `node` from which you want to further compute the bifurcated branches. It is usually used with `node = get_branch(diagram, code)` from a previously computed bifurcation `diagram`.

# Arguments
- `node::BifDiagNode` a node in the bifurcation diagram
- `maxlevel = 1` required maximal level of recursion.
- `options = (x, p, level; k...) -> contparams` this function allows to change the [`continuation`](@ref) options depending on the branching `level`. The argument `x, p` denotes the current solution to `F(x, p)=0`.

# Optional arguments
- `code = "0"` code used to display iterations
- `usedeflation = false`
- `halfbranch = false` for Pitchfork / Transcritical bifurcations, compute only half of the branch. Can be useful when there are symmetries.
- `verbosediagram` verbose specific to bifurcation diagram. Print information about the branches as they are being computed.
- `kwargs` optional arguments as for [`continuation`](@ref) but also for the different versions listed in [Continuation](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/library/#Continuation-1).
"""
function bifurcationdiagram!(prob::AbstractBifurcationProblem,
                            node::BifDiagNode,
                            maxlevel::Int,
                            options;
                            code = "0",
                            halfbranch = false,
                            verbosediagram = false,
                            kwargs...)
    if node.level >= maxlevel || isnothing(node.γ); return node; end
    verbose = (get(kwargs, :verbosity, 0) > 0) || verbosediagram

    # current level of recursion
    level = node.level

    # convenient function for branching
    function letsbranch(_id, _pt, _level; _dsfactor = 1, _ampfactor = 1)
        plotfunc = get(kwargs, :plot_solution, plot_default)
        optscont = options(_pt.x, _pt.param, _level + 1)
        @reset optscont.ds *= _dsfactor

        continuation(get_contresult(node.γ), _id, optscont;
            nev = optscont.nev, 
            kwargs...,
            ampfactor = _ampfactor,  
        )
    end

    for (id, pt) in enumerate(node.γ.specialpoint)
        # we put this condition in case the specialpoint at step = 0 corresponds to the one we are branching from. If we remove this, we keep computing the same branch (possibly).
        if pt.step > 1 && (pt.type in (:bp, :nd))
            try
                if verbose
                    println("─"^80*"\n──▶ New branch, level = $(level+1), dim(Kernel) = ", 
                                kernel_dimension(pt), 
                                    ", code = $code, from bp #",id,
                                    " at p = ", pt.param, 
                                    ", type = ", type(pt))
                end
                γ = letsbranch(id, pt, level)
                add!(node, γ, level+1, id)
                if ~isnothing(γ) && verbose 
                    printstyled(color = :green, "────▶ From ", type(from(γ)), "\n")
                end
                verbose && _show(stdout, node.γ.specialpoint[id], id)

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
                @error "Failed to compute new branch at p = $(pt.param)" exception=ex
            end
        end
    end
    for (ii, _node) in enumerate(node.child)
        bifurcationdiagram!(prob, 
                            _node, 
                            maxlevel, 
                            options; 
                            code = (code..., ii), 
                            verbosediagram, 
                            kwargs...)
    end
    return node
end
