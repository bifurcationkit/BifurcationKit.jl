"""
Plot the branch of solutions during the continuation
"""
function plotBranchCont(contres::ContResult, sol::BorderedArray, contparms, plotuserfunction)
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow, :ns => :orange, :pd => :green)

	l = computeEigenElements(contparms) ? @layout([a{0.5w} [b; c]; e{0.2h}]) : @layout([a{0.5w} [b; c]])
	plot(layout = l )

	plot!(contres ; filterbifpoints = true, putbifptlegend = false,
		xlabel = getLensParam(contres.param_lens),
		ylabel = getfirstusertype(contres),
		label = "", plotfold = false, subplot = 1)

	plotuserfunction(sol.u, sol.p; subplot = 3)

	# put arrow to indicate the order of computation
	length(contres) > 1 &&	plot!([contres.branch[end-1:end].param], [getproperty(contres.branch,1)[end-1:end]], label = "", arrow = true, subplot = 1)

	if computeEigenElements(contparms)
		eigvals = contres.eig[end].eigenvals
		scatter!(real.(eigvals), imag.(eigvals), subplot=4, label = "", markerstrokewidth = 0, markersize = 3, color = :black)
	end

	plot!(contres; vars = (:step, :param), putbifptlegend = false, plotbifpoints = false, xlabel = "step", ylabel = getLensParam(contres.param_lens), label = "", subplot = 2) |> display

end
