using RecipesBase
using Setfield
getLensSymbol(lens) = :p
getLensSymbol(lens::Setfield.PropertyLens{F}) where F = F
getLensSymbol(lens::Setfield.ComposedLens) = getLensSymbol(lens.inner)
getLensSymbol(::Setfield.IdentityLens) = :p
getLensSymbol(::Setfield.IndexLens{Tuple{Int64}}) = :p
getLensSymbol(br::AbstractBranchResult) = getLensSymbol(br.lens)

function getLensSymbol(lens1::Lens, lens2::Lens)
	p1 = getLensSymbol(lens1)
	p2 = getLensSymbol(lens2)
	out = p1 == p2 ? (Symbol(String(p1)*"1"), Symbol(String(p2)*"2")) : (p1, p2)
end

function getPlotVars(contres, vars)
	if vars isa Tuple{Symbol, Symbol} || typeof(vars) <: Tuple{Int64, Int64}
		return vars
	else
		return :param, getfirstusertype(contres)
	end
end

# https://github.com/JuliaGraphics/Colors.jl/blob/master/src/names_data.jl
const colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow, :ns => :orange, :pd => :green, :bt => :gray, :cusp => :sienna1, :gh => :brown, :zh => :pink, :hh => :gray, :user => :darkgoldenrod)

function getColor(sp)
	if sp in keys(colorbif)
		return colorbif[sp]
	else
		return :darkgoldenrod
	end
end

function getAxisLabels(ind1, ind2, br)
	xguide = ""
	yguide = ""
	if ind1 == 1 || ind1 == :param
		xguide = String(getLensSymbol(br))
	elseif ind1 isa Symbol
		xguide = String(ind1)
	end
	if ind2 isa Symbol
		yguide = String(ind2)
	end
	return xguide, yguide
end

# allow to plot a single branch
RecipesBase.@recipe function Plots(contres::AbstractBranchResult; plotfold = false, putspecialptlegend = true, filterspecialpoints = false, vars = nothing, plotstability = true, plotspecialpoints = true, branchlabel = "", linewidthunstable = 1.0, linewidthstable = 2linewidthunstable, plotcirclesbif = false, applytoY = identity)
	# Special case labels when vars = (:p,:y,:z) or (:x) or [:x,:y] ...
	ind1, ind2 = getPlotVars(contres, vars)
	xlab, ylab = getAxisLabels(ind1, ind2, contres)
	@series begin
		if hasstability(contres) && plotstability
			linewidth --> map(x -> isodd(x) ? linewidthstable : linewidthunstable, contres.stable)
		end
		xguide --> xlab
		yguide --> ylab
		label --> branchlabel
		getproperty(contres.branch, ind1), map(applytoY, getproperty(contres.branch, ind2))
	end

	# display bifurcation points
	bifpt = filter(x -> (x.type != :none) && (plotfold || x.type != :fold) && (x.idx <= length(contres)-1), contres.specialpoint)

	if length(bifpt) >= 1 && plotspecialpoints #&& (ind1 == :param)
		if filterspecialpoints == true
			bifpt = filterBifurcations(bifpt)
		end
		@series begin
			seriestype := :scatter
			seriescolor --> map(x -> getColor(x.type), bifpt)
			markershape --> map(x -> (x.status != :converged) && (plotcirclesbif==false) ? :square : :circle, bifpt)
			markersize --> 3
			markerstrokewidth --> 0
			label --> ""
			xguide --> xlab
			yguide --> ylab
			[getproperty(contres[pt.idx], ind1) for pt in bifpt], [applytoY(getproperty(contres[pt.idx], ind2)) for pt in bifpt]
		end
		# add legend for bifurcation points
		if putspecialptlegend && length(bifpt) >= 1
			bps = unique(x -> x.type, [pt for pt in bifpt if (pt.type != :none && (plotfold || pt.type != :fold))])
			(length(bps) == 0) && return
			for pt in bps
				@series begin
					seriestype := :scatter
					seriescolor --> getColor(pt.type)
					label --> "$(pt.type)"
					markersize --> 3
					markerstrokewidth --> 0
					xguide --> xlab
					yguide --> ylab
					[getproperty(contres[pt.idx], ind1)], [applytoY(getproperty(contres[pt.idx], ind2))]
				end
			end
		end
	end
end

# allow to plot branches specified by splatting
RecipesBase.@recipe function Plots(brs::AbstractBranchResult...; plotfold = false, putspecialptlegend = true, filterspecialpoints = false, vars = nothing, plotstability = true, plotspecialpoints = true, branchlabel = fill("",length(brs)), linewidthunstable = 1.0, linewidthstable = 2linewidthunstable, applytoY = identity)
	ind1, ind2 = getPlotVars(brs[1], vars)
	if length(brs) == 0; return; end
	bp = unique(x -> x.type, [(type = pt.type, param = getproperty(pt, ind1), printsol = getproperty(pt.printsol, ind2)) for pt in brs[1].specialpoint if pt.type != :none])
	# add legend for bifurcation points
	if putspecialptlegend && length(bp) > 0
		for pt in unique(x -> x.type, bp)
			@series begin
				seriestype := :scatter
				seriescolor --> getColor(pt.type)
				label --> "$(pt.type)"
				markersize --> 3
				markerstrokewidth --> 0
				[pt.param], [applytoY(pt.printsol)]
			end
		end
	end

	for (id, res) in pairs(brs)
		@series begin
			putspecialptlegend --> putspecialptlegend
			plotfold --> plotfold
			plotspecialpoints --> plotspecialpoints
			plotstability --> plotstability
			branchlabel --> branchlabel[id]
			linewidthunstable --> linewidthunstable
			linewidthstable --> linewidthstable
			applytoY --> applytoY
			vars --> vars
			if ind1 == 1 || ind1 == :param
				xguide --> String(getLensSymbol(brs[id].lens))
			elseif ind1 isa Symbol
				xguide --> String(ind1)
			end
			if ind2 isa Symbol
				yguide --> String(ind2)
			end
			# collect the values of the bifurcation points to be added in the legend
			ind1, ind2 = getPlotVars(brs[id], vars)
			for pt in res.specialpoint
				# this does not work very well when changing optional argument vars
				pt.type != :none && push!(bp, (type = pt.type, param = getproperty(pt, :param), printsol = getproperty(pt.printsol, ind2)))
			end
			res
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
####################################################################################################
# plot recipes for the bifurcation diagram
RecipesBase.@recipe function f(bd::Vector{BifDiagNode}; code = (), level = (-Inf, Inf))
	for b in bd
		@series begin
			level --> level
			code --> code
			b
		end
	end
end

RecipesBase.@recipe function f(bd::BifDiagNode; code = (), level = (-Inf, Inf))
	if ~hasbranch(bd); return; end
	_bd = getBranch(bd, code)
	@series begin
		level --> level
		code --> ()
		_bd.child
	end
	# !! plot root branch in last so the bifurcation points do not alias, for example a 2d BP would be plot as a 1d BP if the order were reversed
	if level[1] <= _bd.level <= level[2]
		@series begin
			_bd.γ
		end
	end
end

# this might well be type piracy
# TODO try to remove it
RecipesBase.@recipe function f(bd::Nothing)
	nothing
end

####################################################################################################
# plotting function of the periodic orbits
function plotPeriodicPOTrap(x, M, Nx, Ny; ratio = 2, kwargs...)
	@assert ratio > 0 "You need at least one component"
	n = Nx*Ny
	outpo = reshape(x[1:end-1], ratio * n, M)
	po = reshape(x[1:n,1], Nx, Ny)
	rg = 2:6:M
	for ii in rg
		po = hcat(po, reshape(outpo[1:n,ii], Nx, Ny))
	end
	heatmap!(po; color = :viridis, fill=true, xlabel = "space", ylabel = "space", kwargs...)
	for ii in 1:length(rg)
		plot!([ii*Ny, ii*Ny], [1, Nx]; color = :red, width = 3, label = "", kwargs...)
	end
end

function plotPeriodicPOTrap(outpof, n, M; ratio = 2)
	@assert ratio > 0 "You need at least one component"
	outpo = reshape(outpof[1:end-1], ratio * n, M)
	if ratio == 1
		heatmap(outpo[1:n,:]', ylabel="Time", color=:viridis)
	else
		plot(heatmap(outpo[1:n,:]', ylabel="Time", color=:viridis),
			heatmap(outpo[n+2:end,:]', color=:viridis))
	end
end
####################################################################################################
function plotPeriodicShooting!(x, M; kwargs...)
	N = div(length(x), M);	plot!(x; label = "", kwargs...)
	for ii in 1:M
		plot!([ii*N, ii*N], [minimum(x), maximum(x)] ;color = :red, label = "", kwargs...)
	end
end

function plotPeriodicShooting(x, M; kwargs...)
	plot();plotPeriodicShooting!(x, M; kwargs...)
end
####################################################################################################
# plot recipe for codim 2 plot
# TODO Use dispatch for this
RecipesBase.@recipe function Plots(contres::ContResult{Ta, Teigvals, Teigvec, Biftype, Ts, Tfunc, Tpar, Tl}; plotfold = false, putspecialptlegend = true, filterspecialpoints = false, vars = nothing, plotstability = true, plotspecialpoints = true, branchlabel = "", linewidthunstable = 1.0, linewidthstable = 2linewidthunstable, plotcirclesbif = false, _basicplot = true, applytoY = identity) where {Ta, Teigvals, Teigvec, Biftype, Ts, Tfunc <: Union{FoldProblemMinimallyAugmented, HopfProblemMinimallyAugmented}, Tpar, Tl}
	# Special case labels when vars = (:p,:y,:z) or (:x) or [:x,:y] ...
	ind1, ind2 = getPlotVars(contres, vars)
	xlab, ylab = getAxisLabels(ind1, ind2, contres)
	@series begin
		if hasstability(contres) && false
			linewidth --> map(x -> isodd(x) ? linewidthstable : linewidthunstable, contres.stable)
		end
		xguide --> xlab
		yguide --> ylab
		label --> branchlabel
		getproperty(contres.branch, ind1), map(applytoY, getproperty(contres.branch, ind2))
	end

	# display bifurcation points
	bifpt = filter(x -> (x.type != :none) && (plotfold || x.type != :fold), contres.specialpoint)

	if length(bifpt) >= 1 && plotspecialpoints #&& (ind1 == :param)
		if filterspecialpoints == true
			bifpt = filterBifurcations(bifpt)
		end
		@series begin
			seriestype := :scatter
			seriescolor --> map(x -> getColor(x.type), bifpt)
			markershape --> map(x -> (x.status != :converged) && (plotcirclesbif==false) ? :square : :circle, bifpt)
			markersize --> 3
			markerstrokewidth --> 0
			label --> ""
			xguide --> xlab
			yguide --> ylab
			[getproperty(contres[pt.idx], ind1) for pt in bifpt], [applytoY(getproperty(contres[pt.idx], ind2)) for pt in bifpt]
		end
		# add legend for bifurcation points
		if putspecialptlegend && length(bifpt) >= 1
			bps = unique(x -> x.type, [pt for pt in bifpt if (pt.type != :none && (plotfold || pt.type != :fold))])
			(length(bps) == 0) && return
			for pt in bps
				@series begin
					seriestype := :scatter
					seriescolor --> getColor(pt.type)
					label --> "$(pt.type)"
					markersize --> 3
					markerstrokewidth --> 0
					xguide --> xlab
					yguide --> ylab
					[getproperty(contres[pt.idx], ind1)], [applytoY(getproperty(contres[pt.idx], ind2))]
				end
			end
		end
	end
end
