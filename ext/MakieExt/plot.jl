using Makie: Point2

function Makie.convert_arguments(::PointBased, contres::AbstractBranchResult, vars = nothing, applytoY = identity, applytoX = identity)
    ind1, ind2 = get_plot_vars(contres, vars)
    return ([Point2{Float32}(i, j) for (i, j) in zip(map(applytoX, getproperty(contres.branch, ind1)), map(applytoY, getproperty(contres.branch, ind2)))],)
end

function isplit(x::AbstractVector{T}, indices::AbstractVector{<: Integer}, splitval::Bool = true) where {T <: Real}
    # Adapt behavior for CairoMakie only
    if !isempty(indices) && isdefined(Main, :CairoMakie) && Makie.current_backend() == Main.CairoMakie
        xx = similar(x, length(x) + 2 * (length(indices)))
        for (i, ind) in enumerate(indices)
            if ind == first(indices)
                xx[1:ind] .= @views x[1:ind]
            else
                xx[(2*(i-1)).+(indices[i-1]+1:ind)] .= @views x[(indices[i-1]+1:ind)]
            end
            # safeguard for boundary indices (e.g. sp.idx == 1)
            _left = ind > 1 ? x[ind-1] : x[ind]
            _right = ind < length(x) ? x[ind+1] : x[ind]
            if !splitval
                xx[2*(i-1)+ind] = _left
            end
            # Add a NaN is necessary, otherwise continue with same value as before (useful for linewidth)
            xx[2*(i-1)+ind+1] = splitval ? NaN : _left
            # Repeat last value before NaN, but adapt for linewidth
            xx[2*(i-1)+ind+2] = splitval ? x[ind] : _right
        end
        # Fill the rest of the extended array
        xx[last(indices)+2*length(indices)+1:end] .= @views x[last(indices)+1:end]
        return xx
    else
        return x
    end
end

function plot_stability_segments!(ax,
                                x,
                                y,
                                stable;
                                color = Makie.wong_colors()[1],
                                dash_unstable_style = true,
                                linewidthunstable = 1,
                                linewidthstable = 3linewidthunstable,
                                kw...
                            )
    @assert length(x) == length(y) == length(stable)
    i = 1
    n = length(x) - 1

    first_plot = true
    color = nothing

    while i < n
        current = stable[i]

        j = i + 1
        while j <= n && stable[j] == current
            j += 1
        end
        # first call: let Makie choose color
        if first_plot
            p = lines!(ax, x[i:j], y[i:j]; linestyle = (current ? :solid : :dash), kw...)
            color = p.color[]
            first_plot = false
        else
            lines!(ax, x[i:j], y[i:j]; color, linestyle = (current ? :solid : :dash), kw...)
        end
        i = j
    end
    return ax
end

const _PLOT_KWARGS_DOC = """
# Keyword arguments
- `vars = nothing`: `Tuple{Symbol, Symbol}` or `Tuple{Int, Int}` selecting the quantities plotted on the horizontal and vertical axes. A `Symbol` refers to a field of `contres.branch` (for example `(:param, :x1)` or `(:param, :period)`) while an `Int` refers to the index of the recorded solution. Defaults to `(:param, _getfirstusertype(contres))`, i.e. the continuation parameter against the first recorded variable; see `get_plot_vars`.
- `applytoX = identity`, `applytoY = identity`: functions applied to the horizontal and vertical coordinates before plotting, e.g. `applytoX = inv` to plot the inverse of the parameter.
- `branchlabel = nothing`: label of the branch in the legend.
- `branchcolor = nothing`: color of the branch line. It does not affect the colors of the special points.
- `plotstability = true`: if `true` and stability information is available, the stability along the branch is encoded in the line width, using `linewidthstable` for stable parts and `linewidthunstable` for unstable parts. For branches of periodic orbits, stability is deduced from the type of the bifurcation (e.g. supercriticality); for two-parameter continuations without stability information, the line width is constant.
- `linewidthunstable = 1`: line width used for the unstable parts of the branch.
- `linewidthstable = 3linewidthunstable`: line width used for the stable parts of the branch.
- `dash_unstable_style = false`: if `true`, the unstable parts are drawn with dashed lines (see `plot_stability_segments!`) rather than relying on line widths only.
- `plotspecialpoints = true`: whether to draw the special points (fold, Hopf, NS, PD, ...) found along the branch.
- `plotfold = false`: whether to draw fold points. They are hidden by default.
- `filterspecialpoints = false`: whether to filter out fold/branch points which are too close to each other to avoid overplotting, see `filter_bifurcations`.
- `plotcirclesbif = true`: whether special points are drawn as circles. If `false`, points whose location is only a guess (`status == :guess`) are drawn as squares, indicating that they have not been refined by bisection.
- `putspecialptlegend = true`: whether to add one legend entry per type of special point.
- `kw...`: additional attributes forwarded to the Makie plotting functions (`lines!`, `scatter!`, ...), for example `color`, `linewidth`, `linestyle` or `alpha`.
"""

"""
$(TYPEDSIGNATURES)

Add the branch of solutions `contres` to the existing Makie axis `ax`. The axis labels are set according to `vars`.

$(_PLOT_KWARGS_DOC)

# Returns
- `ax`: the Makie axis on which the branch was drawn.

See also [`plot`](@ref) to create a new figure and axis.
"""
function plot!(ax1, contres::AbstractResult{Tkind, Tprob}; 
                applytoY = identity, 
                applytoX = identity,
                branchlabel = nothing,
                branchcolor = nothing,
                dash_unstable_style = false,
                filterspecialpoints = false,
                linewidthunstable = 1,
                linewidthstable = 3linewidthunstable,
                plotcirclesbif = true,
                plotfold = false,
                plotstability = true,
                plotspecialpoints = true,
                putspecialptlegend = true,
                vars = nothing,
                kw...) where {Tkind, Tprob}

    # names for axis labels
    ind1, ind2 = get_plot_vars(contres, vars)
    xlab, ylab = get_axis_labels(ind1, ind2, contres)

    # stability linewidth
    linewidth = linewidthunstable
    indices = Int[sp.idx for sp in contres.specialpoint if sp.type !== :endpoint]
    # isplit required to work with CairoMakie due to change of linewidth for stability
    if _hasstability(contres) && plotstability
        if Tkind == NSPeriodicOrbitCont || Tkind == HopfCont || Tkind == PDPeriodicOrbitCont
            linewidth = isplit(map(ind -> is_supercritical(contres, ind) ? linewidthstable : linewidthunstable, eachindex(contres.stable)), indices, false)
        elseif Tkind <: AbstractTwoParamCont
            linewidth = isplit(fill(linewidthunstable, length(contres.stable)), indices, false)
        else
            linewidth = isplit(map(x -> x ? linewidthstable : linewidthunstable, contres.stable), indices, false)
        end
    end

    if dash_unstable_style == false
        xbranch = isplit(map(applytoX, getproperty(contres.branch, ind1)), indices)
        ybranch = isplit(map(applytoY, getproperty(contres.branch, ind2)), indices)
        if isnothing(branchcolor)
            lines!(ax1, xbranch, ybranch; linewidth, label = branchlabel, kw...)
        else
            lines!(ax1, xbranch, ybranch; linewidth, label = branchlabel, color = branchcolor, kw...)
        end
    else
        if Tkind <: AbstractTwoParamCont
            _stable = (Tkind == NSPeriodicOrbitCont || Tkind == HopfCont || Tkind == PDPeriodicOrbitCont) ? [is_supercritical(contres, ind) for ind in eachindex(contres.stable)] : fill(true, length(contres.stable))
        else
            _stable = contres.stable
        end
        plot_stability_segments!(ax1, 
                                map(applytoX, getproperty(contres.branch, ind1)),
                                map(applytoY, getproperty(contres.branch, ind2)),
                                _stable,
                                color = branchcolor,
                                label = branchlabel,
                                kw...
                                )
    end
    ax1.xlabel = xlab
    ax1.ylabel = ylab

    # display bifurcation points
    bifpt = filter(x -> (x.type != :none) && (x.type != :endpoint) && (plotfold || x.type != :fold) && (x.idx <= length(contres) - 1), contres.specialpoint)
    if length(bifpt) >= 1 && plotspecialpoints #&& (ind1 == :param)
        if filterspecialpoints
            bifpt = filterBifurcations(bifpt)
        end
        scatter!(ax1, 
                [applytoX(getproperty(contres[pt.idx], ind1)) for pt in bifpt],
                [applytoY(getproperty(contres[pt.idx], ind2)) for pt in bifpt];
                marker = map(x -> (x.status == :guess) && (plotcirclesbif == false) ? :rect : :circle, bifpt), markersize = 10,
                color = map(x -> get_color(x.type),
                bifpt))
    end

    # add legend for bifurcation points
    if putspecialptlegend && length(bifpt) >= 1
        bps = unique(x -> x.type, [pt for pt in bifpt if (pt.type != :none && (plotfold || pt.type != :fold))])
        (length(bps) == 0) && return
        for pt in bps
            scatter!(ax1, 
                    [applytoX(getproperty(contres[pt.idx], ind1))], 
                    [applytoY(getproperty(contres[pt.idx], ind2))]; 
                    color = get_color(pt.type),
                    markersize = 10,
                    label = "$(pt.type)")
        end
        Makie.axislegend(ax1, merge = true, unique = true)
    end
    ax1
end

function plot_branch_cont(::BK_Makie,
                          contres::ContResult,
                          state,
                          iter,
                          plotuserfunction; 
                          plotfold = false,
                          plotstability = true,
                          plotspecialpoints = true,
                          putspecialptlegend = true,
                          filterspecialpoints = false,
                          linewidthunstable = 1.0,
                          linewidthstable = 3.0linewidthunstable,
                          plotcirclesbif = true,
                          applytoY = identity,
                          applytoX = identity,
                          kw...)
    sol = getsolution(state)
    if length(contres) == 0
        return
    end

    # names for axis labels
    ind1, ind2 = get_plot_vars(contres, nothing)
    xlab, ylab = get_axis_labels(ind1, ind2, contres)

    # stability linewidth
    linewidth = linewidthunstable

    fig = Figure(size = (1200, 700))
    ax1 = fig[1:2, 1] = Axis(fig, xlabel = String(xlab), ylabel = String(ylab), tellheight = true)

    ax2 = fig[1, 2] = Axis(fig, xlabel = "step [$(state.step)]", ylabel = String(xlab))
    lines!(ax2, contres.step, contres.param)

    if compute_eigenelements(iter)
        eigvals = contres.eig[end].eigenvals
        ax_ev = fig[3, 1:2] = Axis(fig, xlabel = "ℜ", ylabel = "ℑ")
        scatter!(ax_ev, real.(eigvals), imag.(eigvals), strokewidth = 0, markersize = 10, color = :black)
        # add stability boundary
        maxIm = maximum(imag, eigvals)
        minIm = minimum(imag, eigvals)
        if maxIm - minIm < 1e-6
            maxIm, minIm = 1, -1
        end
        lines!(ax_ev, [0, 0], [maxIm, minIm]; color = :blue, linewidth = linewidthunstable)
    end

    # plot arrow to indicate the order of computation
    if length(contres) > 1
        x = contres.branch[end].param
        y = getproperty(contres.branch, 1)[end]
        u = contres.branch[end].param - contres.branch[end-1].param
        v = getproperty(contres.branch, 1)[end] - getproperty(contres.branch, 1)[end-1]
        Makie.arrows2d!(ax1, [x], [y], [u], [v], color = :green, tipwidth = 15, tiplength = 15, minshaftlength = 0)
    end

    plot!(ax1, contres; plotfold, plotstability, plotspecialpoints, putspecialptlegend, filterspecialpoints, linewidthunstable, linewidthstable, plotcirclesbif, applytoY, applytoX, kw...)

    if isnothing(plotuserfunction) == false
        ax_perso = fig[2, 2] = Axis(fig, tellheight = true)
        plotuserfunction(ax_perso, sol.u, sol.p; ax1, iter, state)
    end

    display(fig)
    fig
end

"""
$(TYPEDSIGNATURES)

Plot the branch(es) of solutions with the Makie backend. This method is only available after loading Makie (or one of its backends, e.g. `CairoMakie` or `GLMakie`); call it as `BifurcationKit.plot` to avoid a name clash with `Makie.plot`.

The following dispatch variants are supported:
- `plot(contres::AbstractBranchResult; kwargs...)` plots a single branch, e.g. a [`ContResult`](@ref) returned by [`continuation`](@ref) or a [`Branch`](@ref).
- `plot(brs::AbstractBranchResult...; branchlabel = [...], kwargs...)` plots several branches on the same axis and adds a legend.
- `plot(brdc::DCResult; kwargs...)` plots all the branches computed by deflated continuation.
- `plot(br::Branch{..., <: Vector{<: ContResult}}; kwargs...)` plots all the branches emanating from a multiple bifurcation point.
- `plot(bd::BifDiagNode; code = (), level = (-Inf, Inf), kwargs...)` plots the branches stored in a bifurcation diagram, see [`bifurcationdiagram`](@ref). `code` selects a subtree (see [`get_branch`](@ref)) and only branches whose level lies in `level` are drawn.

$(_PLOT_KWARGS_DOC)

# Returns
- `(fig, ax)`: the Makie `Figure` and `Axis` on which the branches were drawn.
- `nothing` if there is nothing to plot (empty branch or diagram without branch).

# Example
```julia
using BifurcationKit, CairoMakie
br = continuation(prob, PALC(), opts)
fig, ax = BifurcationKit.plot(br)
BifurcationKit.plot!(ax, br2; branchcolor = :red)
```
"""
function plot(contres::AbstractBranchResult; kP...)
    if length(contres) == 0
        return
    end

    ind1, ind2 = get_plot_vars(contres, nothing)
    xlab, ylab = get_axis_labels(ind1, ind2, contres)

    fig = Figure()
    ax = fig[1, 1] = Axis(fig, xlabel = String(xlab), ylabel = String(ylab), tellheight = true)

    plot!(ax, contres; kP...)
    fig, ax
end

plot(brdc::DCResult; kP...) = plot(brdc.branches...; kP...)

# allow to plot a Branch holding several branches (from an Nd bifurcation point)
"""
$(TYPEDSIGNATURES)

Add all the branches held in `br` (typically arising from an `Nd` bifurcation point) to the Makie axis `ax`. Each branch is drawn with [`plot!`](@ref) and receives the corresponding entry of `branchlabel` if it is provided.

# Returns
- `ax`: the Makie axis on which the branches were drawn.
"""
function plot!(ax1, br::Branch{Tkind, Tprob, <: Vector{ <: ContResult}}; kw...) where {Tkind, Tprob}
    for contres in getbranches(br)
        plot!(ax1, Branch(contres, getfield(br, :bp)); kw...)
    end
    ax1
end

plot(br::Branch{Tkind, Tprob, <: Vector{ <: ContResult}}; branchlabel = ["$i" for i = 1:length(br)], kP...) where {Tkind, Tprob} = plot([Branch(contres, getfield(br, :bp)) for contres in getbranches(br)]...; branchlabel, kP...)

function plot(brs::AbstractBranchResult...; branchlabel = ["$i" for i = 1:length(brs)], kP...)
    if length(brs) == 0
        return
    end
    fig = Figure()
    ax1 = fig[1, 1] = Axis(fig)

    for (id, contres) in pairs(brs)
        plot!(ax1, contres; branchlabel = branchlabel[id], kP...)
    end
    Makie.axislegend(ax1, merge = true, unique = true)
    display(fig)
    fig, ax1
end

####################################################################################################
# plotting function of the periodic orbits
"""
$(TYPEDSIGNATURES)

Plot a space-time representation of a periodic orbit as one or two heatmaps. This is mainly useful for periodic orbits of discretized PDEs.

# Arguments
- `outpof`: flattened vector containing the space-time solution; its last entry is discarded.
- `n`: number of spatial points.
- `M`: number of time steps.

# Keyword arguments
- `ratio = 2`: number of components stored in `outpof`. With `ratio == 1` a single heatmap is produced; with `ratio == 2` the two components are plotted side by side.
"""
function plot_periodic_potrap(outpof, n, M; ratio = 2)
    @assert ratio > 0 "You need at least one component"
    outpo = reshape(outpof[1:end-1], ratio * n, M)
    if ratio == 1
        heatmap(outpo[1:n, :]', colormap = :viridis)
    else
        fig = Makie.Figure()
        ax1 = Axis(fig[1, 1])
        ax2 = Axis(fig[1, 2])
        # Makie.heatmap!(ax1, rand(2,2))
        Makie.heatmap!(ax1, outpo[1:n, :]')
        Makie.heatmap!(ax2, outpo[n+2:end, :]')
        fig
    end
end
####################################################################################################
# plot recipes for the bifurcation diagram
"""
$(TYPEDSIGNATURES)

Plot the branches stored in the bifurcation diagram `bd` with the Makie backend. A new figure is created and displayed; see [`bifurcationdiagram`](@ref) for how to compute `bd`.

# Keyword arguments
- `code = ()`: tuple of indices selecting the subtree to draw, see [`get_branch`](@ref).
- `level = (-Inf, Inf)`: tuple `(minlevel, maxlevel)`; only branches whose recursion level lies in this range are drawn.
- `k...`: keyword arguments forwarded to [`plot!`](@ref) for each branch, e.g. any of the plotting keywords accepted by [`plot`](@ref).

# Returns
- `(fig, ax)`: the Makie `Figure` and `Axis` on which the diagram was drawn.
- `nothing` if the diagram contains no branch.
"""
function plot(bd::BifDiagNode; code = (), level = (-Inf, Inf), k...)
    if ~hasbranch(bd)
        return
    end

    fig = Figure()
    ax = fig[1, 1] = Axis(fig)

    plot!(ax, bd; code, level, k...)

    display(fig)
    fig, ax
end

"""
$(TYPEDSIGNATURES)

Add the branches stored in the bifurcation diagram `bd` (or in the vector of diagrams `bd`) to the Makie axis `ax`.

# Keyword arguments
- `code = ()`: tuple of indices selecting the subtree to draw, see [`get_branch`](@ref).
- `level = (-Inf, Inf)`: tuple `(minlevel, maxlevel)`; only branches whose recursion level lies in this range are drawn.
- `k...`: keyword arguments forwarded to [`plot!`](@ref) for each branch.

# Returns
- `ax`: the Makie axis on which the branches were drawn.
"""
function plot!(ax, bd::BifDiagNode; code = (), level = (-Inf, Inf), k...)
    _plot_bifdiag_makie!(ax, bd; code, level, k...)
    return ax
end

function plot!(ax, bd::Vector{<:BifDiagNode}; code = (), level = (-Inf, Inf), k...)
    _plot_bifdiag_makie!(ax, bd; code, level, k...)
    return ax
end

function _plot_bifdiag_makie!(ax, bd::BifDiagNode; code = (), level = (-Inf, Inf), k...)
    if ~hasbranch(bd)
        return
    end

    _bd = get_branch(bd, code)
    _plot_bifdiag_makie!(ax, _bd.child; code = (), level = level, k...)

    # !! plot root branch in last so the bifurcation points do not alias, for example a 2d BP would be plot as a 1d BP if the order were reversed
    if level[1] <= _bd.level <= level[2]
        plot!(ax, _bd.γ; k...)
    end
end

function _plot_bifdiag_makie!(ax, bd::Vector{<:BifDiagNode}; code = (), level = (-Inf, Inf), k...)
    for b in bd
        _plot_bifdiag_makie!(ax, b; code, level, k...)
    end
end
####################################################################################################
"""
$(TYPEDSIGNATURES)

Plot the eigenvalues stored along the branch `br` with the Makie backend.

# Arguments
- `with_param = true`: if `true`, the parameter selected by `var` is used as the horizontal coordinate; otherwise the index along the branch is used.

# Keyword arguments
- `var = :param`: field of `br.branch` used as the horizontal coordinate. Only used when `with_param = true`.
- `applyY = identity`: function applied to the eigenvalues before plotting, e.g. `applyY = imag` to plot their imaginary parts.
- `k...`: attributes forwarded to Makie `series`, e.g. `color` or `linewidth`.
"""
function plot_eigenvals(br::AbstractResult, with_param = true; var = :param, applyY = identity, k...)
    p = getproperty(br.branch, var)
    data = mapreduce(x -> applyY.(x.eigenvals), hcat, br.eig)
    if with_param
        series(p, real.(data); color = Makie.distinguishable_colors(size(data,1)), k...)
    else
        series(real.(data); color = Makie.distinguishable_colors(size(data,1)), k...)
    end
end
####################################################################################################
plotAllDCBranch(::BK_Makie, branches) = plot(branches...)

function plot_DCont_branch(::BK_Makie, 
                            branches, 
                            nbrs::Int, 
                            nactive::Int, 
                            nstep::Int)
    f, ax = plot(branches...)
    ax.title = "$nbrs branches, actives = $(nactive), step = $nstep"
    display(f)
    f, ax
end