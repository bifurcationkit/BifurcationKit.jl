using RecipesBase
using Setfield
getLensParam(lens::Setfield.PropertyLens{F}) where F = F
getLensParam(::Setfield.IdentityLens) = :p
getLensParam(::Setfield.IndexLens{Tuple{Int64}}) = :p

@recipe function f(contres::BranchResult; plotfold = true, putbifptlegend = true, filterbifpoints = false, vars = nothing, plotstability = true, plotbifpoints = true, branchlabel = "")
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow, :ns => :orange, :pd => :green)
	axisDict = Dict(:p => 1, :sol => 2, :itnewton => 3, :ds => 4, :theta => 5, :step => 6)
	# Special case labels when vars = (:p,:y,:z) or (:x) or [:x,:y] ...
	if typeof(vars) <: Tuple && (typeof(vars[1]) == Symbol && typeof(vars[2]) == Symbol)
		ind1 = axisDict[vars[1]]
		ind2 = axisDict[vars[2]]
	elseif typeof(vars) <: Tuple && (typeof(vars[1]) <: Int && typeof(vars[2]) <: Int)
		ind1 = vars[1]
		ind2 = vars[2]
	else
		ind1 = 1
		ind2 = 2
	end

	@series begin
		if length(contres.stability) > 2 && plotstability
			linewidth --> map(x -> isodd(x) ? 2.0 : 1.0, contres.stability)
		end
		if ind1 == 1
			xguide --> getLensParam(contres.param_lens)
		end
		label --> branchlabel
		contres.branch[ind1, :], contres.branch[ind2, :]
	end

	# display bifurcation points
	bifpoints = vcat(contres.bifpoint, filter(x->x.type != :none, contres.foldpoint))
	if length(bifpoints) >= 1 && ind2 == 2 && plotbifpoints
		id = 1
		bifpoints[1].type == :none ? id = 2 : id = 1
		if plotfold
			bifpt = bifpoints[id:end]
		else
			bifpt = filter(x->x.type != :fold, bifpoints[id:end])
		end
		if filterbifpoints == true
			bifpt = filterBifurcations(bifpt)
		end
		@series begin
			seriestype := :scatter
			seriescolor --> map(x -> colorbif[x.type], bifpt)
			markershape --> map(x -> x.status == :guess ? :square : :circle, bifpt)
			markersize --> 2
			markerstrokewidth --> 0
			label --> ""
			map(x -> x.param, bifpt), map(x -> x.printsol, bifpt)
		end

		# add legend for bifurcation points
		if putbifptlegend && length(bifpoints) >= 1
			bp = unique([pt.type for pt in bifpt if pt.type != :none])
			(length(bp) == 0) && return
			for pt in bp
				@series begin
					seriestype := :scatter
					seriescolor --> colorbif[pt]
					label --> "$pt"
					markerstrokewidth --> 0
					[], []
				end
			end
		end

	end
end

@recipe function Plots(brs::AbstractVector{<:BranchResult}; plotfold = true, putbifptlegend = true, filterbifpoints = false, vars = nothing, pspan=nothing, plotstability = true, plotbifpoints = true, branchlabel = repeat([""],length(brs)))
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow, :ns => :orange, :pd => :green)
	if length(brs) == 0; return; end
	bp = Set(unique([pt.type for pt in brs[1].bifpoint]))
	for (id,res) in enumerate(brs)
		@series begin
			putbifptlegend --> false
			plotfold --> plotfold
			plotbifpoints --> plotbifpoints
			plotstability --> plotstability
			branchlabel --> branchlabel[id]
			for pt in res.bifpoint
				push!(bp, pt.type)
			end
			res
		end
	end
	# add legend for bifurcation points
	if putbifptlegend && length(bp) > 0
		for pt in bp
			@series begin
				seriestype := :scatter
				seriescolor --> colorbif[pt]
				label --> "$pt"
				markerstrokewidth --> 0
			# scatter!([], [], color = colorbif[pt], label = "$pt", markerstrokewidth = 0)
				[], []
			end
		end
	end
end


####################################################################################################
"""
Plot the branch of solutions during the continuation
"""
function plotBranchCont(contres::ContResult, sol::BorderedArray, contparms, plotuserfunction)
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow, :ns => :orange, :pd => :green)

	if computeEigenElements(contparms) == false
		l =  Plots.@layout [a{0.5w} [b; c]]
	else
		l =  Plots.@layout [a{0.5w} [b; c]; e{0.2h}]
	end
	Plots.plot(layout = l)

	plot!(contres ; filterbifpoints = true, putbifptlegend = false, xlabel = getLensParam(contres.param_lens),  ylabel = "||x||", label = "", plotfold = false, subplot = 1)
	if length(contres) >1
		# put arrow to indicate the order of computation
		plot!([contres.branch[1, end-1:end]], [contres.branch[2, end-1:end]], label = "", arrow = true, subplot = 1)
	end

	plot!(contres;	vars = (:step, :p), putbifptlegend = false, xlabel = "it", ylabel = getLensParam(contres.param_lens), label = "", subplot=2)

	if computeEigenElements(contparms)
		eigvals = contres.eig[end].eigenvals
		scatter!(real.(eigvals), imag.(eigvals), subplot=4, label = "", markerstrokewidth = 0, markersize = 3, color = :black)
	end

	plotuserfunction(sol.u, sol.p; subplot = 3)

	display(title!(""))
end

function filterBifurcations(bifpt)
	# this function filters Fold points and Branch points which are located at the same/previous/next point
	res = [(type = :none, idx = 1, param = 1., printsol = 1., status = :guess)]
	ii = 1
	while ii <= length(bifpt) - 1
		if (abs(bifpt[ii].idx - bifpt[ii+1].idx) <= 1) && bifpt[ii].type ∈ [:fold, :bp]
			if (bifpt[ii].type == :fold && bifpt[ii].type == :bp) ||
				(bifpt[ii].type == :bp && bifpt[ii].type == :fold)
				push!(res, (type = :fold, idx = bifpt[ii].idx, param = bifpt[ii].param, printsol = bifpt[ii].printsol, status = bifpt[ii].status) )
			else
				push!(res, (type = bifpt[ii].type, idx = bifpt[ii].idx, param = bifpt[ii].param, printsol = bifpt[ii].printsol, status = bifpt[ii].status) )
				push!(res, (type = bifpt[ii+1].type, idx = bifpt[ii+1].idx, param = bifpt[ii+1].param, printsol = bifpt[ii+1].printsol,status = bifpt[ii].status) )
			end
			ii += 2
		else
			push!(res, (type = bifpt[ii].type, idx = bifpt[ii].idx, param = bifpt[ii].param, printsol = bifpt[ii].printsol,status = bifpt[ii].status) )
			ii += 1
		end
	end
	0<ii<=length(bifpt) &&	push!(res, (type = bifpt[ii].type, idx = bifpt[ii].idx, param = bifpt[ii].param, printsol = bifpt[ii].printsol,status = bifpt[ii].status) )

	# for p in (res[2:end])
	# 	println(p)
	# end
	res[2:end]
end
