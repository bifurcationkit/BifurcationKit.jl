"""
Plot the branch of solutions during the continuation
"""
function plot_branch_cont(contres::ContResult, 
                        state, 
                        iter, 
                        plotuserfunction)
    sol = getsolution(state)
    l = compute_eigenelements(iter) ? Plots.@layout([a{0.5w} [b; c]; e{0.2h}]) : Plots.@layout([a{0.5w} [b; c]])
    Plots.plot(layout = l )

    Plots.plot!(contres ; 
                filterspecialpoints = true,
                putspecialptlegend = false,
                xlabel = get_lens_symbol(contres),
                ylabel = _getfirstusertype(contres),
                label = "",
                plotfold = false,
                subplot = 1)

    plotuserfunction(sol.u, sol.p; subplot = 3)

    # put arrow to indicate the order of computation
    length(contres) > 1 && plot!([contres.branch[end-1:end].param], [getproperty(contres.branch,1)[end-1:end]], label = "", arrow = true, subplot = 1)

    if compute_eigenelements(iter)
        eigvals = contres.eig[end].eigenvals
        Plots.scatter!(real.(eigvals), imag.(eigvals), subplot=4, label = "", markerstrokewidth = 0, markersize = 3, color = :black, xlabel = "ℜ", ylabel = "ℑ")
        # add stability boundary
        maxIm = maximum(imag, eigvals)
        minIm = minimum(imag, eigvals)
        Plots.plot!([0, 0], [maxIm, minIm], subplot=4, label = "", color = :blue)
    end

    Plots.plot!(contres; 
                vars = (:step, :param),
                putspecialptlegend = false,
                plotspecialpoints = false,
                xlabel = "step [$(state.step)]",
                ylabel = get_lens_symbol(contres),
                label = "",
                subplot = 2) |> display
end
####################################################################################################
plotAllDCBranch(branches) = display(plot(branches..., label = ""))

function plot_DCont_branch(::Union{BK_Plots, BK_NoPlot},
                            branches, 
                            nbrs::Int, 
                            nactive::Int,
                            nstep::Int)
    plot(branches...; label = "", title  = "$nbrs branches, actives = $(nactive), step = $nstep")
    for br in branches
    (length(br) > 1) && plot!([br.branch[end-1:end].param], 
            [getproperty(br.branch,1)[end-1:end]], 
            label = "", arrow = true, color = :red)
    end
    scatter!([br.branch[1].param for br in branches], 
             [br.branch[1][1] for br in branches], 
              marker = :cross, color=:green, label = "") |> display
end
####################################################################################################
# plotting function of the periodic orbits
function plot_periodic_potrap(x, M, Nx, Ny; ratio = 2, kwargs...)
    @assert ratio > 0 "You need at least one component"
    n = Nx*Ny
    outpo = reshape(x[begin:end-1], ratio * n, M)
    po = reshape(x[1:n,1], Nx, Ny)
    rg = 2:6:M
    for ii in rg
        po = hcat(po, reshape(outpo[1:n,ii], Nx, Ny))
    end
    heatmap!(po; color = :viridis, fill=true, xlabel = "space", ylabel = "space", kwargs...)
    for ii in eachindex(rg)
        Plots.plot!([ii*Ny, ii*Ny], [1, Nx]; color = :red, width = 3, label = "", kwargs...)
    end
end

function plot_periodic_potrap(outpof, n, M; ratio = 2)
    @assert ratio > 0 "You need at least one component"
    outpo = reshape(outpof[begin:end-1], ratio * n, M)
    if ratio == 1
        Plots.heatmap(outpo[1:n,:]', ylabel="Time", color=:viridis)
    else
        Plots.plot(
            Plots.heatmap(outpo[1:n,:]', ylabel="Time", color=:viridis),
            Plots.heatmap(outpo[n+2:end,:]', color=:viridis))
    end
end
####################################################################################################
function plot_periodic_shooting!(x, M; kwargs...)
    N = div(length(x), M); plot!(x; label = "", kwargs...)
    for ii in 1:M
        Plots.plot!([ii*N, ii*N], [minimum(x), maximum(x)] ;color = :red, label = "", kwargs...)
    end
end

function plot_periodic_shooting(x, M; kwargs...)
    Plots.plot()
    plot_periodic_shooting!(x, M; kwargs...)
end
####################################################################################################
function plot_eigenvals(br::AbstractResult, with_param = true; plot_imag = false, var = :param, k...)
    p = getproperty(br.branch, var)
    data = mapreduce(x -> x.eigenvals, hcat, br.eig)
    
    values_to_plot = plot_imag ? imag.(data') : real.(data')
    
    if with_param
        plot(p, values_to_plot; k...)
    else
        plot(values_to_plot; k...)
    end
end
