using Makie: Point2

function Makie.convert_arguments(::PointBased, contres::AbstractBranchResult, vars = nothing, applytoY = identity, applytoX = identity)
    ind1, ind2 = get_plot_vars(contres, vars)
    return ([Point2{Float32}(i, j) for (i, j) in zip(map(applytoX, getproperty(contres.branch, ind1)), map(applytoY, getproperty(contres.branch, ind2)))],)
end

function isplit(x::AbstractVector{T}, indices::AbstractVector{<:Integer}, splitval::Bool = true) where {T<:Real}
    # Adapt behavior for CairoMakie only
    if !isempty(indices) && isdefined(Main, :CairoMakie) && Makie.current_backend() == Main.CairoMakie
        xx = similar(x, length(x) + 2 * (length(indices)))
        for (i, ind) in enumerate(indices)
            if ind == first(indices)
                xx[1:ind] .= @views x[1:ind]
            else
                xx[(2*(i-1)).+(indices[i-1]+1:ind)] .= @views x[(indices[i-1]+1:ind)]
            end
            if !splitval
                xx[2*(i-1)+ind] = x[ind-1]
            end
            # Add a NaN is necessary, otherwise continue with same value as before (useful for linewidth)
            xx[2*(i-1)+ind+1] = splitval ? NaN : x[ind-1]
            # Repeat last value before NaN, but adapt for linewidth
            xx[2*(i-1)+ind+2] = splitval ? x[ind] : x[ind+1]
        end
        # Fill the rest of the extended array
        xx[last(indices)+2*length(indices)+1:end] .= @views x[last(indices)+1:end]
        return xx
    else
        return x
    end
end

"""
$(SIGNATURES)

Plot the branch of solutions.

# Arguments
- `plotcirclesbif = true`, indicates the special points well determined by bisection (cirlce) versus not well determined (square).
- `branchlabel` example `["Fold", "Hopf"]` assigns label to branch.
- `branchcolor` assign color to a branch without affecting the color of the special points.
"""
function plot!(ax1, contres::AbstractResult{Tkind, Tprob}; 
                plotfold = false,
                plotstability = true,
                plotspecialpoints = true,
                putspecialptlegend = true,
                filterspecialpoints = false,
                vars = nothing,
                linewidthunstable = 1.0,
                linewidthstable = 3.0linewidthunstable,
                plotcirclesbif = true,
                branchlabel = nothing,
                branchcolor = nothing,
                applytoY = identity, 
                applytoX = identity) where {Tkind, Tprob}

    # names for axis labels
    ind1, ind2 = get_plot_vars(contres, vars)
    xlab, ylab = get_axis_labels(ind1, ind2, contres)

    # stability linewidth
    linewidth = linewidthunstable
    if Tkind <: TwoParamCont
        linewidthstable = 1.0
    end
    indices = Int[sp.idx for sp in contres.specialpoint if sp.type !== :endpoint]
    # isplit required to work with CairoMakie due to change of linewidth for stability
    if _hasstability(contres) && plotstability
        linewidth = isplit(map(x -> x ? linewidthstable : linewidthunstable, contres.stable), indices, false)
    end

    xbranch = isplit(map(applytoX, getproperty(contres.branch, ind1)), indices)
    ybranch = isplit(map(applytoY, getproperty(contres.branch, ind2)), indices)
    if isnothing(branchcolor)
        lines!(ax1, xbranch, ybranch; linewidth = linewidth, label = branchlabel)
    else
        lines!(ax1, xbranch, ybranch; linewidth = linewidth, label = branchlabel, color = branchcolor)
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

function plot_branch_cont(contres::ContResult,
                          state,
                          iter,
                          plotuserfunction; plotfold = false,
                          plotstability = true,
                          plotspecialpoints = true,
                          putspecialptlegend = true,
                          filterspecialpoints = false,
                          linewidthunstable = 1.0,
                          linewidthstable = 3.0linewidthunstable,
                          plotcirclesbif = true,
                          applytoY = identity,
                          applytoX = identity)
    sol = getsolution(state)
    if length(contres) == 0
        return
    end

    # names for axis labels
    ind1, ind2 = get_plot_vars(contres, nothing)
    xlab, ylab = get_axis_labels(ind1, ind2, contres)

    # stability linewidth
    linewidth = linewidthunstable
    if _hasstability(contres) && plotstability
        linewidth = map(x -> isodd(x) ? linewidthstable : linewidthunstable, contres.stable)
    end

    fig = Figure(size = (1200, 700))
    ax1 = fig[1:2, 1] = Axis(fig, xlabel = String(xlab), ylabel = String(ylab), tellheight = true)

    ax2 = fig[1, 2] = Axis(fig, xlabel = "step [$(state.step)]", ylabel = String(xlab))
    lines!(ax2, contres.step, contres.param, linewidth = linewidth)

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
        lines!(ax_ev, [0, 0], [maxIm, minIm], color = :blue, linewidth = linewidthunstable)
    end

    # plot arrow to indicate the order of computation
    if length(contres) > 1
        x = contres.branch[end].param
        y = getproperty(contres.branch, 1)[end]
        u = contres.branch[end].param - contres.branch[end-1].param
        v = getproperty(contres.branch, 1)[end] - getproperty(contres.branch, 1)[end-1]
        Makie.arrows!(ax1, [x], [y], [u], [v], color = :green, arrowsize = 20)
    end

    plot!(ax1, contres; plotfold, plotstability, plotspecialpoints, putspecialptlegend, filterspecialpoints, linewidthunstable, linewidthstable, plotcirclesbif, applytoY, applytoX)

    if isnothing(plotuserfunction) == false
        ax_perso = fig[2, 2] = Axis(fig, tellheight = true)
        plotuserfunction(ax_perso, sol.u, sol.p; ax1 = ax1)
    end

    display(fig)
    fig
end

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
function plot_periodic_potrap(outpof, n, M; ratio = 2)
    @assert ratio > 0 "You need at least one component"
    outpo = reshape(outpof[1:end-1], ratio * n, M)
    if ratio == 1
        heatmap(outpo[1:n, :]', ylabel = "Time", color = :viridis)
    else
        fig = Makie.Figure()
        ax1 = Axis(fig[1, 1], ylabel = "Time")
        ax2 = Axis(fig[1, 2], ylabel = "Time")
        # Makie.heatmap!(ax1, rand(2,2))
        Makie.heatmap!(ax1, outpo[1:n, :]')
        Makie.heatmap!(ax2, outpo[n+2:end, :]')
        fig
    end
end
####################################################################################################
# plot recipes for the bifurcation diagram
function plot(bd::BifDiagNode; code = (), level = (-Inf, Inf), k...)
    if ~hasbranch(bd)
        return
    end

    fig = Figure()
    ax = fig[1, 1] = Axis(fig)

    _plot_bifdiag_makie!(ax, bd; code, level, k...)

    display(fig)
    fig, ax
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

function _plot_bifdiag_makie!(ax, bd::Vector{BifDiagNode}; code = (), level = (-Inf, Inf), k...)
    for b in bd
        _plot_bifdiag_makie!(ax, b; code, level, k...)
    end
end
####################################################################################################
function plot_eigenvals(br::AbstractResult, with_param = true; var = :param, applyY = identity, k...)
    p = getproperty(br.branch, var)
    data = mapreduce(x -> applyY.(x.eigenvals), hcat, br.eig)
    if with_param
        series(p, real.(data); k...)
    else
        series(real.(data); k...)
    end
end
####################################################################################################
plotAllDCBranch(branches) = plot(branches...)

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