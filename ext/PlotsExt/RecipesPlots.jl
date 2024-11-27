using RecipesBase

# allow to plot a single branch
RecipesBase.@recipe function Plots(contres::AbstractBranchResult;
                            plotfold = false,
                            putspecialptlegend = true,
                            filterspecialpoints = false,
                            vars = nothing,
                            plotstability = true,
                            plotspecialpoints = true,
                            branchlabel = "",
                            linewidthunstable = 1.0,
                            linewidthstable = 2linewidthunstable,
                            plotcirclesbif = true,
                            applytoY = identity,
                            applytoX = identity)
    # Special case labels when vars = (:p,:y,:z) or (:x) or [:x,:y] ...
    ind1, ind2 = get_plot_vars(contres, vars)
    xlab, ylab = get_axis_labels(ind1, ind2, contres)
    @series begin
        if _hasstability(contres) && plotstability
            linewidth --> map(x -> isodd(x) ? linewidthstable : linewidthunstable, contres.stable)
        end
        xguide --> xlab
        yguide --> ylab
        label --> branchlabel
        map(applytoX, getproperty(contres.branch, ind1)), map(applytoY, getproperty(contres.branch, ind2))
    end

    # display bifurcation points
    bifpt = filter(x -> (x.type != :none) && (x.type != :endpoint) && (plotfold || x.type != :fold) && (x.idx <= length(contres)), contres.specialpoint)

    if length(bifpt) >= 1 && plotspecialpoints #&& (ind1 == :param)
        if filterspecialpoints == true
            bifpt = filter_bifurcations(bifpt)
        end
        @series begin
            seriestype := :scatter
            seriescolor --> map(x -> get_color(x.type), bifpt)
            markershape --> map(x -> (x.status != :converged) && (plotcirclesbif==false) ? :square : :circle, bifpt)
            markersize --> 3
            markerstrokewidth --> 0
            label --> ""
            xguide --> xlab
            yguide --> ylab
            [applytoX(getproperty(contres[pt.idx], ind1)) for pt in bifpt], [applytoY(getproperty(contres[pt.idx], ind2)) for pt in bifpt]
        end
        # add legend for bifurcation points
        if putspecialptlegend && length(bifpt) >= 1
            bps = unique(x -> x.type, [pt for pt in bifpt if (pt.type != :none && (plotfold || pt.type != :fold))])
            (length(bps) == 0) && return
            for pt in bps
                @series begin
                    seriestype := :scatter
                    seriescolor --> get_color(pt.type)
                    label --> "$(pt.type)"
                    markersize --> 3
                    markerstrokewidth --> 0
                    xguide --> xlab
                    yguide --> ylab
                    [applytoX(getproperty(contres[pt.idx], ind1))], [applytoY(getproperty(contres[pt.idx], ind2))]
                end
            end
        end
    end
end

# allow to plot branches specified by splatting
RecipesBase.@recipe function Plots(brs::AbstractBranchResult...;
                            plotfold = false,
                            putspecialptlegend = true,
                            filterspecialpoints = false,
                            vars = nothing,
                            plotstability = true,
                            plotspecialpoints = true,
                            branchlabel = fill("", length(brs)),
                            linewidthunstable = 1.0,
                            linewidthstable = 2linewidthunstable,
                            plotcirclesbif = true,
                            applytoY = identity,
                            applytoX = identity)
    ind1, ind2 = get_plot_vars(brs[1], vars)
    if length(brs) == 0; return; end
    # handle bifurcation points, the issue is to simplify the legend. So we collect all bifurcation points
    bptps = Any[(type = pt.type, param = getproperty(pt, ind1), printsol = getproperty(pt.printsol, ind2)) for pt in brs[1].specialpoint if ((pt.type != :none)&&(pt.type != :endpoint))]
    for ii=2:length(brs)
        _ind1, _ind2 = get_plot_vars(brs[ii], vars)
        for pt in brs[ii].specialpoint
            if (pt.type != :none) && (pt.type != :endpoint)
                push!(bptps, (type = pt.type, param = getproperty(pt, _ind1), printsol = getproperty(pt.printsol, _ind2)))
            end
        end
    end
    bp = unique(x -> x.type, bptps)

    # add legend for bifurcation points
    if putspecialptlegend && length(bp) > 0
        for pt in unique(x -> x.type, bp)
            @series begin
                seriestype := :scatter
                seriescolor --> get_color(pt.type)
                label --> "$(pt.type)"
                markersize --> 3
                markerstrokewidth --> 0
                [applytoX(pt.param)], [applytoY(pt.printsol)]
            end
        end
    end

    for (id, res) in pairs(brs)
        @series begin
            plotcirclesbif --> plotcirclesbif
            putspecialptlegend --> false
            plotfold --> plotfold
            plotspecialpoints --> plotspecialpoints
            plotstability --> plotstability
            branchlabel --> branchlabel[id]
            linewidthunstable --> linewidthunstable
            linewidthstable --> linewidthstable
            applytoX --> applytoX
            applytoY --> applytoY
            vars --> vars
            if ind1 == 1 || ind1 == :param
                xguide --> String(get_lens_symbol(brs[id]))
            elseif ind1 isa Symbol
                xguide --> String(ind1)
            end
            if ind2 isa Symbol
                yguide --> String(ind2)
            end
            # collect the values of the bifurcation points to be added in the legend
            ind1, ind2 = get_plot_vars(brs[id], vars)
            for pt in res.specialpoint
                # this does not work very well when changing optional argument vars
                pt.type != :none && push!(bp, (type = pt.type, param = getproperty(pt, :param), printsol = getproperty(pt.printsol, ind2)))
            end
            res
        end
    end
end

RecipesBase.@recipe function Plots(brs::DCResult;
                            plotfold = false,
                            putspecialptlegend = true,
                            filterspecialpoints = false,
                            vars = nothing,
                            plotstability = true,
                            plotspecialpoints = true,
                            branchlabel = "",
                            linewidthunstable = 1.0,
                            linewidthstable = 2linewidthunstable,
                            plotcirclesbif = false,
                            applytoY = identity,
                            applytoX = identity)
    for (id, res) in pairs(brs.branches)
        @series begin
            putspecialptlegend --> false
            plotfold --> plotfold
            plotspecialpoints --> plotspecialpoints
            plotstability --> plotstability
            branchlabel --> branchlabel
            linewidthunstable --> linewidthunstable
            linewidthstable --> linewidthstable
            applytoX --> applytoX
            applytoY --> applytoY
            vars --> vars
            res
        end
    end
end

####################################################################################################
RecipesBase.@recipe function Plots(sol::SolPeriodicOrbit;
                                    indx = nothing
                                    )
    @assert indx isa Int || indx isa Nothing
    @series begin
        if indx === nothing
            sol.t, sol.u'
        else
            sol.t, sol.u[indx, :]
        end
    end
end
####################################################################################################
# plot recipes for the bifurcation diagram
RecipesBase.@recipe function f(bd::Vector{BifDiagNode}; code = (), level = (-Inf, Inf), ind_br = code)
    for (id, b) in pairs(bd)
        @series begin
            level --> level
            ind_br --> (ind_br..., id)
            code --> code
            b
        end
    end
end

RecipesBase.@recipe function f(bd::BifDiagNode; code = (), level = (-Inf, Inf), ind_br = ())
    if ~hasbranch(bd); return; end
    _bd = get_branch(bd, code)
    @series begin
        level --> level
        ind_br --> (ind_br..., _bd.code)
        _bd.child
    end
    # !! plot root branch in last so the bifurcation points do not alias, for example a 2d BP would be plot as a 1d BP if the order were reversed
    if level[1] <= _bd.level <= level[2]
        @series begin
            branchlabel --> "$(ind_br[2:2:end])"
            _bd.γ
        end
    end
end

# this might well be type piracy
RecipesBase.@recipe function f(bd::Nothing)
    nothing
end
####################################################################################################
# plot recipe for codim 2 plot
# TODO Use dispatch for this
RecipesBase.@recipe function Plots(contres::AbstractResult{Tk, Tprob};
                                    plotfold = false,
                                    putspecialptlegend = true,
                                    filterspecialpoints = false,
                                    vars = nothing,
                                    plotstability = true,
                                    plotspecialpoints = true,
                                    branchlabel = "",
                                    linewidthunstable = 1.0,
                                    linewidthstable = 2linewidthunstable,
                                    plotcirclesbif = true,
                                    _basicplot = true,
                                    applytoY = identity,
                                    applytoX = identity) where {Tk <: TwoParamCont, Tprob}
    # Special case labels when vars = (:p,:y,:z) or (:x) or [:x,:y] ...
    ind1, ind2 = get_plot_vars(contres, vars)
    xlab, ylab = get_axis_labels(ind1, ind2, contres)
    @series begin
        if _hasstability(contres) && false
            linewidth --> map(x -> isodd(x) ? linewidthstable : linewidthunstable, contres.stable)
        end
        xguide --> xlab
        yguide --> ylab
        label --> branchlabel
        map(applytoX, getproperty(contres.branch, ind1)), map(applytoY, getproperty(contres.branch, ind2))
    end

    # display bifurcation points
    bifpt = filter(x -> (x.type != :none) && (x.type != :endpoint) && (plotfold || x.type != :fold), contres.specialpoint)

    if length(bifpt) >= 1 && plotspecialpoints #&& (ind1 == :param)
        if filterspecialpoints == true
            bifpt = filter_bifurcations(bifpt)
        end
        @series begin
            seriestype := :scatter
            seriescolor --> map(x -> get_color(x.type), bifpt)
            markershape --> map(x -> (x.status != :converged) && (plotcirclesbif==false) ? :square : :circle, bifpt)
            markersize --> 3
            markerstrokewidth --> 0
            label --> ""
            xguide --> xlab
            yguide --> ylab
            [applytoX(getproperty(contres[pt.idx], ind1)) for pt in bifpt], [applytoY(getproperty(contres[pt.idx], ind2)) for pt in bifpt]
        end
        # add legend for bifurcation points
        if putspecialptlegend && length(bifpt) >= 1
            bps = unique(x -> x.type, [pt for pt in bifpt if (pt.type != :none && (plotfold || pt.type != :fold))])
            (length(bps) == 0) && return
            for pt in bps
                @series begin
                    seriestype := :scatter
                    seriescolor --> get_color(pt.type)
                    label --> "$(pt.type)"
                    markersize --> 3
                    markerstrokewidth --> 0
                    xguide --> xlab
                    yguide --> ylab
                    [applytoX(getproperty(contres[pt.idx], ind1))], [applytoY(getproperty(contres[pt.idx], ind2))]
                end
            end
        end
    end
end
