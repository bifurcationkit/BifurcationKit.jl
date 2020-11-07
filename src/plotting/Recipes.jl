using RecipesBase
using Setfield
getLensParam(lens::Setfield.PropertyLens{F}) where F = F
getLensParam(::Setfield.IdentityLens) = :p
getLensParam(::Setfield.IndexLens{Tuple{Int64}}) = :p

function getPlotVars(contres, vars)
	if vars isa Tuple{Symbol, Symbol} || typeof(vars) <: Tuple{Int64, Int64}
		return vars
	else
		return :param, getfirstusertype(contres)
	end
end

# allow to plot a single branch
@recipe function Plots(contres::BranchResult; plotfold = false, putbifptlegend = true, filterbifpoints = false, vars = nothing, plotstability = true, plotbifpoints = true, branchlabel = "")
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow, :ns => :orange, :pd => :green)
	# Special case labels when vars = (:p,:y,:z) or (:x) or [:x,:y] ...
	ind1, ind2 = getPlotVars(contres, vars)
	@series begin
		if computeEigenElements(contres.contparams) && plotstability
			linewidth --> map(x -> isodd(x) ? 2.0 : 1.0, contres.stable)
		end
		if ind1 == 1
			xguide --> getLensParam(contres.param_lens)
		end
		label --> branchlabel
		getproperty(contres.branch, ind1), getproperty(contres.branch, ind2)
	end

	# display bifurcation points
	bifpt = filter(x -> x.type != :none, contres.bifpoint)
	if plotfold
		bifpt = vcat(bifpt, filter(x -> x.type != :none, contres.foldpoint))
	end
	if length(bifpt) >= 1 && plotbifpoints && (ind1 == :param)
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
			map(x -> getproperty(x, ind1), bifpt), map(x -> getproperty(x.printsol, ind2), bifpt)
		end
		# add legend for bifurcation points
		if putbifptlegend && length(bifpt) >= 1
			bps = unique(x -> x.type, [pt for pt in bifpt if pt.type != :none])
			(length(bps) == 0) && return
			for pt in bps
				@series begin
					seriestype := :scatter
					seriescolor --> colorbif[pt.type]
					label --> "$(pt.type)"
					markersize --> 2
					markerstrokewidth --> 0
					[getproperty(pt, ind1)], [getproperty(pt.printsol, ind2)]
				end
			end
		end
	end
end

# allow to plot branches specified by splatting
@recipe function Plots(brs::BranchResult...; plotfold = false, putbifptlegend = true, filterbifpoints = false, vars = nothing, plotstability = true, plotbifpoints = true, branchlabel = repeat([""],length(brs)))
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow, :ns => :orange, :pd => :green)
	ind1, ind2 = getPlotVars(brs[1], vars)
	if length(brs) == 0; return; end
	bp = unique(x -> x.type, [(type = pt.type, param = getproperty(pt, ind1), printsol = getproperty(pt.printsol, ind2)) for pt in brs[1].bifpoint if pt.type != :none])
	for (id, res) in pairs(brs)
		@series begin
			putbifptlegend --> false
			plotfold --> plotfold
			plotbifpoints --> plotbifpoints
			plotstability --> plotstability
			branchlabel --> branchlabel[id]
			vars --> vars
			xguide --> getLensParam(res.param_lens)
			# collect the values of the bifurcation points to be added in the legend
			ind1, ind2 = getPlotVars(brs[id], vars)
			for pt in res.bifpoint
				pt.type != :none && push!(bp, (type = pt.type, param = getproperty(pt, ind1), printsol = getproperty(pt.printsol, ind2)))
			end
			res
		end
	end
	# add legend for bifurcation points
	if putbifptlegend && length(bp) > 0
		for pt in unique(x -> x.type, bp)
			@series begin
				seriestype := :scatter
				seriescolor --> colorbif[pt.type]
				label --> "$(pt.type)"
				markersize --> 2
				markerstrokewidth --> 0
				[pt.param], [pt.printsol]
			end
		end
	end
end
####################################################################################################

function filterBifurcations(bifpt)
	# this function filters Fold points and Branch points which are located at the same/previous/next point
	length(bifpt) == 0 && return bifpt
	res = [(type = :none, idx = 1, param = 1., printsol = bifpt[1].printsol, status = :guess)]
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
			push!(res, (type = bifpt[ii].type, idx = bifpt[ii].idx, param = bifpt[ii].param, printsol = bifpt[ii].printsol, status = bifpt[ii].status) )
			ii += 1
		end
	end
	0 < ii <= length(bifpt) &&	push!(res, (type = bifpt[ii].type, idx = bifpt[ii].idx, param = bifpt[ii].param, printsol = bifpt[ii].printsol, status = bifpt[ii].status) )

	return res[2:end]
end
