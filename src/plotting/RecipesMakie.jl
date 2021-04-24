function AbstractPlotting.convert_arguments(::AbstractPlotting.PointBased, contres::AbstractBranchResult, vars = nothing)
	ind1, ind2 = getPlotVars(contres, vars)
	return ([AbstractPlotting.Point2f0(i, j) for (i, j) in zip(getproperty(contres.branch, ind1), getproperty(contres.branch, ind2))],)
end

function plotBranchCont(contres::ContResult, sol::BorderedArray, contparms, plotuserfunction; plotfold = false, plotbifpoints = true, filterbifpoints = false, plotcirclesbif = true)
	if length(contres) ==0; return ;end

	ind1, ind2 = getPlotVars(contres, nothing)
	xlab, ylab = getAxisLabels(ind1, ind2, contres)

	fig = AbstractPlotting.Figure(resolution = (1200, 700))
	ax1 = fig[1, 1] = AbstractPlotting.Axis(fig, xlabel = String(xlab), ylabel = String(ylab), tellheight = true)
	AbstractPlotting.lines!(ax1, contres, color = :blue)

	ax2 = fig[1, 2] = AbstractPlotting.Axis(fig, xlabel = "step", ylabel = String(xlab))
	AbstractPlotting.lines!(ax2, contres.step, contres.param, color = :blue)

	if isnothing(plotuserfunction) == false
		ax3, line3 = plotuserfunction(fig[2:3, 1:2], sol.u, sol.p)
	end
	#
	# put arrow to indicate the order of computation
	if length(contres) > 1
		# plot!([contres.branch[end-1:end].param], [getproperty(contres.branch,1)[end-1:end]], label = "", arrow = true, subplot = 1)
		x = contres.branch[end].param
		y = getproperty(contres.branch,1)[end]
		# u = contres.branch[end].param - contres.branch[end-1].param
		# v = getproperty(contres.branch,1)[end]-getproperty(contres.branch,1)[end-1]
		AbstractPlotting.scatter!(ax1, [x], [y], marker = :cross )
	end

	# display bifurcation points
	bifpt = filter(x -> (x.type != :none) && (plotfold || x.type != :fold), contres.bifpoint)
	if length(bifpt) >= 1 && plotbifpoints #&& (ind1 == :param)
		if filterbifpoints == true
			bifpt = filterBifurcations(bifpt)
		end
		AbstractPlotting.scatter!(ax1, map(x -> getproperty(x, ind1), bifpt), map(x -> getproperty(x.printsol, ind2), bifpt), marker = map(x -> (x.status == :guess) && (plotcirclesbif==false) ? :rect : :circle, bifpt), markersize = 7, color = map(x -> colorbif[x.type], bifpt))
	end

	AbstractPlotting.display(fig)
	fig
end

function plotBranch(contres::AbstractBranchResult; plotfold = false, plotbifpoints = true, filterbifpoints = false, plotcirclesbif = true, linewidthunstable = 1.0, linewidthstable = 2linewidthunstable)
	if length(contres) == 0; return ;end

	ind1, ind2 = getPlotVars(contres, nothing)
	xlab, ylab = getAxisLabels(ind1, ind2, contres)

	fig = AbstractPlotting.Figure(resolution = (1200, 700))
	ax1 = fig[1, 1] = AbstractPlotting.Axis(fig, xlabel = String(xlab), ylabel = String(ylab), tellheight = true)

	#TODO use band to have similar  result to Plots
	colors = hasstability(contres) ? map(x -> isodd(x) ? :green : :red, contres.stable) : :black
	AbstractPlotting.lines!(ax1, contres, color = colors)

	#
	# put arrow to indicate the order of computation
	if length(contres) > 1
		x = contres.branch[end].param
		y = getproperty(contres.branch,1)[end]
		AbstractPlotting.scatter!(ax1, [x], [y], marker = :cross )
	end

	# display bifurcation points
	bifpt = filter(x -> (x.type != :none) && (plotfold || x.type != :fold), contres.bifpoint)
	if length(bifpt) >= 1 && plotbifpoints #&& (ind1 == :param)
		if filterbifpoints == true
			bifpt = filterBifurcations(bifpt)
		end
		AbstractPlotting.scatter!(ax1, map(x -> getproperty(x, ind1), bifpt), map(x -> getproperty(x.printsol, ind2), bifpt), marker = map(x -> (x.status == :guess) && (plotcirclesbif==false) ? :rect : :circle, bifpt), markersize = 7, color = map(x -> colorbif[x.type], bifpt))
	end

	AbstractPlotting.display(fig)
	fig
end

function plotBranch(contresV::AbstractBranchResult...; plotfold = false, plotbifpoints = true, filterbifpoints = false, plotcirclesbif = true, linewidthunstable = 1.0, linewidthstable = 2linewidthunstable)
	if length(contresV) == 0; return ;end
	fig = AbstractPlotting.Figure(resolution = (1200, 700))
	ax1 = fig[1, 1] = AbstractPlotting.Axis(fig)

	for contres in contresV

		ind1, ind2 = getPlotVars(contres, nothing)
		xlab, ylab = getAxisLabels(ind1, ind2, contres)

		colors = hasstability(contres) ? map(x -> isodd(x) ? :green : :red, contres.stable) : :black
		# AbstractPlotting.lines!(ax1, contres, color = colors)

		h = AbstractPlotting.lines!(ax1, contres, color = colors)
		ax1.xlabel = xlab; ax1.ylabel = ylab

		# display bifurcation points
		bifpt = filter(x -> (x.type != :none) && (plotfold || x.type != :fold), contres.bifpoint)
		if length(bifpt) >= 1 && plotbifpoints #&& (ind1 == :param)
			if filterbifpoints == true
				bifpt = filterBifurcations(bifpt)
			end
			AbstractPlotting.scatter!(ax1, map(x -> getproperty(x, ind1), bifpt), map(x -> getproperty(x.printsol, ind2), bifpt), marker = map(x -> (x.status == :guess) && (plotcirclesbif==false) ? :rect : :circle, bifpt), markersize = 7, color = map(x -> colorbif[x.type], bifpt))
		end
	end

	AbstractPlotting.display(fig)
	fig
end
