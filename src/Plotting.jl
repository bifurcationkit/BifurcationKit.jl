####################################################################################################
"""
Plot the branch of solutions during the continuation
"""
function plotBranchCont(contres::ContResult, sol::BorderedArray, contparms, plotuserfunction)
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow, :ns => :orange, :pd => :green)
	branch = contres.branch

	if contparms.computeEigenValues == false
		l =  Plots.@layout [a{0.5w} [b; c]]
	else
		l =  Plots.@layout [a{0.5w} [b; c]; e{0.2h}]
	end
	Plots.plot(layout = l)

	# plot the branch of solutions
	plotBranch!(contres; filterbifpoints = true, putbifptlegend = false, xlabel="p",  ylabel="||x||", label="", subplot=1)
	plot!(branch[1, :],	 xlabel="it", ylabel="p", label="", subplot=2)

	if contparms.computeEigenValues
		eigvals = contres.eig[end].eigenvals
		scatter!(real.(eigvals), imag.(eigvals), subplot=4, label="", markerstrokewidth=0, markersize = 3, color=:black)
	end

	plotuserfunction(sol.u, sol.p; subplot = 3)

	display(title!(""))
end

"""
	plotBranch(contres::ContResult, plot_fold = true; kwargs...)

Plot the branch of solutions from a `ContResult`. You can also pass parameters like `plotBranch(br, marker = :dot)`.
For the continuation diagram, the legend is as follows `(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow)`
"""
function plotBranch(contres, plot_fold = true; kwargs...)
	# we do not specify the type of contres, not convenient when using JLD2
	plot()
	plotBranch!(contres, plot_fold; kwargs...)
end

"""
	plotBranch(brs::Vector, plot_fold = true; putbifptlegend = true, filterbifpoints = false, kwargs...)

Plot all the branches contained in `brs` in a single figure. Convenient when many bifurcation diagram have been computed.
"""
function plotBranch(brs::Vector, plot_fold = true; putbifptlegend = true, filterbifpoints = false, kwargs...)
	# we do not specify the type of contres, not convenient when using JLD2
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow, :ns => :orange, :pd => :green)
	plotBranch(brs[1], plot_fold; putbifptlegend = false, filterbifpoints = filterbifpoints, kwargs...)
	bp = Set(unique([pt.type for pt in brs[1].bifpoint]))
	for ii=2:length(brs)
		plotBranch!(brs[ii], plot_fold; putbifptlegend = false, filterbifpoints = filterbifpoints, kwargs...) |> display
		for pt in brs[ii].bifpoint
			push!(bp, pt.type)
		end
	end
	# add legend for bifurcation points
	if putbifptlegend && length(bp) > 0
		for pt in bp
			scatter!([], [], color = colorbif[pt], label = "$pt", markerstrokewidth = 0)
		end
	end
end

function filterBifurcations(bifpt)
	# this function filters Fold points and Branch points which are located at the same/previous/next point
	res = [(type = :none, idx = 1, param = 1., printsol = 1.)]
	ii = 1
	while ii <= length(bifpt) - 1
		if (abs(bifpt[ii].idx - bifpt[ii+1].idx) <= 1) && bifpt[ii].type ∈ [:fold, :bp]
			if (bifpt[ii].type == :fold && bifpt[ii].type == :bp) ||
				(bifpt[ii].type == :bp && bifpt[ii].type == :fold)
				push!(res, (type = :fold, idx = bifpt[ii].idx, param = bifpt[ii].param, printsol = bifpt[ii].printsol) )
			else
				push!(res, (type = bifpt[ii].type, idx = bifpt[ii].idx, param = bifpt[ii].param, printsol = bifpt[ii].printsol) )
				push!(res, (type = bifpt[ii+1].type, idx = bifpt[ii+1].idx, param = bifpt[ii+1].param, printsol = bifpt[ii+1].printsol) )
			end
			ii += 2
		else
			push!(res, (type = bifpt[ii].type, idx = bifpt[ii].idx, param = bifpt[ii].param, printsol = bifpt[ii].printsol) )
			ii += 1
		end
	end
	0<ii<=length(bifpt) &&	push!(res, (type = bifpt[ii].type, idx = bifpt[ii].idx, param = bifpt[ii].param, printsol = bifpt[ii].printsol) )

	# for p in (res[2:end])
	# 	println(p)
	# end
	res[2:end]
end

"""
	plotBranch!(contres, plot_fold = true; putbifptlegend = true, filterbifpoints::Bool = false, kwargs...)

Append to the current plot, the plot of the branch of solutions from a `ContResult`. You can also pass parameters like `plotBranch!(br, marker = :dot)`. Options to filter the bifurcation points (which are mostly guesses) are provided. For example, `filterbifpoints = true` merges the nearby Fold and Branch points.
"""
function plotBranch!(contres, plot_fold = true; putbifptlegend = true, filterbifpoints::Bool = false, kwargs...)
	# we do not specify the type of contres, not convenient when using JLD2
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow, :ns => :orange, :pd => :green)
	branch = contres.branch
	if length(contres.stability) > 2
		# plot!(branch[1, :], branch[2, :], linestyle = map(x -> isodd(x) ? :solid : :dash, contres.stability); kwargs...)
		# plot!(branch[1, :], branch[2, :], color = map(x -> isodd(x) ? :green : :red, contres.stability); kwargs...)
		plot!(branch[1, :], branch[2, :], linewidth = map(x -> isodd(x) ? 3.0 : 1.0, contres.stability); kwargs...)
	else
		plot!(branch[1, :], branch[2, :]; kwargs...)
	end
	scatter!([branch[1, end]], [branch[2, end]], marker=:cross, subplot=1, label = "")
	# add the bifurcation points along the branch
	bifpoints = vcat(contres.bifpoint, filter(x->x.type != :none, contres.foldpoint))
	if length(bifpoints) >= 1
		id = 1
		bifpoints[1].type == :none ? id = 2 : id = 1
		if plot_fold
			bifpt = bifpoints[id:end]
		else
			bifpt = filter(x->x.type != :fold, bifpoints[id:end])
		end
		if filterbifpoints == true
			bifpt = filterBifurcations(bifpt)
		end
		scatter!(map(x -> x.param, bifpt), map(x -> x.printsol, bifpt), label="", color = map(x->colorbif[x.type], bifpt), markersize=3, markerstrokewidth=0 ; kwargs...)
	end
	# add legend for bifurcation points
	if putbifptlegend && length(bifpoints) >= 1
		bp = unique([pt.type for pt in bifpt if pt.type != :none])
		(length(bp) == 0) && return
		for pt in bp
			scatter!([], [], color=colorbif[pt], label="$pt", markerstrokewidth = 0)
		end
	end
end
