####################################################################################################

function get_plot_backend()
    !isnothing(Base.get_extension(BifurcationKit,:GLMakieExt)) && return BK_Makie()
    !isnothing(Base.get_extension(BifurcationKit, :PlotsExt)) && return BK_Plots()
    return BK_NoPlot()
end

const pb = get_plot_backend()

function get_plot_vars(contres, vars)
    if vars isa Tuple{Symbol, Symbol} || typeof(vars) <: Tuple{Int64, Int64}
        return vars
    else
        return :param, _getfirstusertype(contres)
    end
end

# https://github.com/JuliaGraphics/Colors.jl/blob/master/src/names_data.jl
# we don't need to have different colors. Indeed, some bifurcations never occurs together codim1 ∩ codim2 = ∅
const colorbif = Dict(:fold => :black,
                        :hopf => :red,
                        :bp => :blue,
                        :nd => :magenta,
                        :none => :yellow,
                        :ns => :orange,
                        :pd => :green,
                        :bt => :red,
                        :cusp => :sienna1,
                        :gh => :brown,
                        :zh => :burlywood2,
                        :hh => :green,
                        :R => :chartreuse4,
                        :R1 => :chartreuse4,
                        :R2 => :chartreuse3,
                        :R3 => :chartreuse2,
                        :R4 => :blue,
                        :foldFlip => :blue4,
                        :ch => :red3,
                        :foldNS => :cyan3,
                        :flipNS => :darkgoldenrod,
                        :pdNS => :maroon,
                        :nsns => :darkorchid,
                        :gpd => :darksalmon,
                        :user => :darkgoldenrod)

function get_color(sp)
    if sp in keys(colorbif)
        return colorbif[sp]
    else
        return :darkgoldenrod
    end
end

function get_axis_labels(ind1, ind2, br)
    xguide = ""
    yguide = ""
    if ind1 == 1 || ind1 == :param
        xguide = String(get_lens_symbol(br))
    elseif ind1 isa Symbol
        xguide = String(ind1)
    end
    if ind2 isa Symbol
        yguide = String(ind2)
    end
    return xguide, yguide
end

####################################################################################################
function filter_bifurcations(bifpt)
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
    0 < ii <= length(bifpt) && push!(res, (type = bifpt[ii].type, idx = bifpt[ii].idx, param = bifpt[ii].param, printsol = bifpt[ii].printsol, status = bifpt[ii].status) )

    return res[2:end]
end

function plot end
function plot! end
function plot_branch_cont end
function plot_periodic_potrap end
function plot_periodic_shooting! end
function plot_periodic_shooting end
function plot_eigenvals end