"""
Plot the branch of solutions during the continuation
"""
function plotBranchCont(contres::ContResult, sol::BorderedArray, contparms, plotuserfunction)
	l = computeEigenElements(contparms) ? Plots.@layout([a{0.5w} [b; c]; e{0.2h}]) : Plots.@layout([a{0.5w} [b; c]])
	plot(layout = l )

	plot!(contres ; filterspecialpoints = true, putspecialptlegend = false,
		xlabel = getLensSymbol(contres),
		ylabel = getfirstusertype(contres),
		label = "", plotfold = false, subplot = 1)

	plotuserfunction(sol.u, sol.p; subplot = 3)

	# put arrow to indicate the order of computation
	length(contres) > 1 &&	plot!([contres.branch[end-1:end].param], [getproperty(contres.branch,1)[end-1:end]], label = "", arrow = true, subplot = 1)

	if computeEigenElements(contparms)
		eigvals = contres.eig[end].eigenvals
		scatter!(real.(eigvals), imag.(eigvals), subplot=4, label = "", markerstrokewidth = 0, markersize = 3, color = :black)
		# add stability boundary
		maxIm = maximum(imag, eigvals)
		minIm = minimum(imag, eigvals)
		plot!([0, 0], [maxIm, minIm], subplot=4, label = "", color = :blue)
	end

	plot!(contres; vars = (:step, :param), putspecialptlegend = false, plotspecialpoints = false, xlabel = "step", ylabel = getLensSymbol(contres), label = "", subplot = 2) |> display

end
