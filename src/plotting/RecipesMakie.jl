function GLMakie.convert_arguments(::PointBased, contres::AbstractBranchResult, vars = nothing)
	ind1, ind2 = getPlotVars(contres, vars)
	return ([Point2f0(i, j) for (i, j) in zip(getproperty(contres.branch, ind1), getproperty(contres.branch, ind2))],)
end

function plotBranchCont(contres::ContResult,
		sol::BorderedArray,
		contparms,
		plotuserfunction;
		plotfold = false,
		plotspecialpoints = true,
		filterspecialpoints = false,
		plotcirclesbif = true)
	if length(contres) == 0; return ;end

	ind1, ind2 = getPlotVars(contres, nothing)
	xlab, ylab = getAxisLabels(ind1, ind2, contres)

	fig = Figure(resolution = (1200, 700))
	ax1 = fig[1, 1] = Axis(fig, xlabel = String(xlab), ylabel = String(ylab), tellheight = true)
	lines!(ax1, contres, color = :blue)

	ax2 = fig[1, 2] = Axis(fig, xlabel = "step", ylabel = String(xlab))
	lines!(ax2, contres.step, contres.param, color = :blue)

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
		scatter!(ax1, [x], [y], marker = :cross )
	end

	# display bifurcation points
	bifpt = filter(x -> (x.type != :none) && (plotfold || x.type != :fold), contres.specialpoint)
	if length(bifpt) >= 1 && plotspecialpoints #&& (ind1 == :param)
		if filterspecialpoints == true
			bifpt = filterBifurcations(bifpt)
		end
		scatter!(ax1, map(x -> getproperty(x, ind1), bifpt), map(x -> getproperty(x.printsol, ind2), bifpt), marker = map(x -> (x.status == :guess) && (plotcirclesbif==false) ? :rect : :circle, bifpt), markersize = 7, color = map(x -> colorbif[x.type], bifpt))
	end

	display(fig)
	fig
end

function plotBranch(contres::AbstractBranchResult; plotfold = false, plotspecialpoints = true, filterspecialpoints = false, plotcirclesbif = true, linewidthunstable = 1.0, linewidthstable = 2linewidthunstable, applytoX = identity, applytoY = identity)
	if length(contres) == 0; return ;end

	# Special case labels when vars = (:p,:y,:z) or (:x) or [:x,:y] ...
	ind1, ind2 = getPlotVars(contres, nothing)
	xlab, ylab = getAxisLabels(ind1, ind2, contres)

	fig = Figure(resolution = (1200, 700))
	ax1 = fig[1, 1] = Axis(fig, xlabel = String(xlab), ylabel = String(ylab), tellheight = true)

	#TODO use band to have similar  result to Plots
	colors = hasstability(contres) ? map(x -> isodd(x) ? :green : :red, contres.stable) : :black
	lines!(ax1, contres, color = colors)

	#
	# put arrow to indicate the order of computation
	if length(contres) > 1
		x = contres.branch[end].param
		y = getproperty(contres.branch,1)[end]
		scatter!(ax1, [x], [y], marker = :cross )
	end

	# display bifurcation points
	bifpt = filter(x -> (x.type != :none) && (x.type != :endpoint) && (plotfold || x.type != :fold) && (x.idx <= length(contres)-1), contres.specialpoint)

	if length(bifpt) >= 1 && plotspecialpoints #&& (ind1 == :param)
		if filterspecialpoints == true
			bifpt = filterBifurcations(bifpt)
		end
		scatter!(ax1, [applytoX(getproperty(contres[pt.idx], ind1)) for pt in bifpt],
					  [applytoY(getproperty(contres[pt.idx], ind2)) for pt in bifpt],
			 marker = map(x -> (x.status != :converged) && (plotcirclesbif==false) ? :rect : :circle, bifpt), markersize = 7, color = map(x -> colorbif[x.type], bifpt))
	end

	display(fig)
	fig
end

function plotBranch(contresV::AbstractBranchResult...; plotfold = false, plotspecialpoints = true, filterspecialpoints = false, plotcirclesbif = true, linewidthunstable = 1.0, linewidthstable = 2linewidthunstable)
	if length(contresV) == 0; return ;end
	fig = Figure(resolution = (1200, 700))
	ax1 = fig[1, 1] = Axis(fig)

	for contres in contresV

		ind1, ind2 = getPlotVars(contres, nothing)
		xlab, ylab = getAxisLabels(ind1, ind2, contres)

		colors = hasstability(contres) ? map(x -> isodd(x) ? :green : :red, contres.stable) : :black
		# lines!(ax1, contres, color = colors)

		h = lines!(ax1, contres, color = colors)
		ax1.xlabel = xlab; ax1.ylabel = ylab

		# display bifurcation points
		bifpt = filter(x -> (x.type != :none) && (x.type != :endpoint) && (plotfold || x.type != :fold), contres.specialpoint)
		if length(bifpt) >= 1 && plotspecialpoints #&& (ind1 == :param)
			if filterspecialpoints == true
				bifpt = filterBifurcations(bifpt)
			end
			scatter!(ax1, map(x -> getproperty(x, ind1), bifpt), map(x -> getproperty(x.printsol, ind2), bifpt), marker = map(x -> (x.status == :guess) && (plotcirclesbif==false) ? :rect : :circle, bifpt), markersize = 7, color = map(x -> colorbif[x.type], bifpt))
		end
	end

	display(fig)
	fig
end
