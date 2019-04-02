using RecursiveArrayTools # for bifurcation point handling
import Base: show
###############################################################################################
# Structure to hold result
@with_kw struct ContResult{T, vectype, eigenvectype}
	# this vector is used to hold (param, printsolution(u), Newton iterations, ds)
	branch::VectorOfArray{T, 2, Array{Vector{T}, 1}} = VectorOfArray([zeros(T, 4)])

	# the following holds the eigen elements at the index of the point along the curve. This index is the last element of eig[1] (for example). Recording this index is useful for storing only some eigenelements and not all of them along the curve
	eig::Vector{Tuple{Array{Complex{T}, 1}, eigenvectype, Int64}}

	# the following holds information about the detected bifurcation points like
	# [(:none, step, printsolution(u), u, tau, eigenvalue_index)] where tau is the tangent along the curve and eigenvalue_index is the index of the eigenvalue in eig (see above) which change stability
	bifpoint::Vector{Tuple{Symbol, Int64, T, T, vectype, vectype, Int64}}

	# whether the associated point is linearly stable
	stability::Vector{Bool} = [false]

	# number of eigenvalues with positive real part and non zero imaginary part
	n_imag::Vector{Int64}

	# number of eigenvalues with positive real part
	n_unstable::Vector{Int64}
end

function show(io::IO, br::PseudoArcLengthContinuation.ContResult)
	println(io, "Branch number of points: ", length(br.branch))
	if length(br.bifpoint) >0
		println(io, "Bifurcation points:")
		for ii in eachindex(br.bifpoint)
			bp  = br.bifpoint[ii]
			println(io, "- $ii, ", bp[1], " point, at p = ", bp[3])
		end
	end
end
###############################################################################################
function displayIteration(i, funceval, residual, itlinear = 0)
(i == 0) && println("\n Newton Iterations \n   Iterations      Func-count      f(x)      Linear-Iterations\n")
	if length(itlinear)==1
		@printf("%9d %16d %14.4e %9d\n", i, funceval, residual, itlinear);
	else
		@printf("%9d %16d %14.4e (%6d, %6d)\n", i, funceval, residual, itlinear[1], itlinear[2]);
	end
end
###############################################################################################
"""
Plot the continued branch of solutions
"""
function plotBranchCont(contres::ContResult{T}, sol::M, contparms, plotuserfunction) where {T, vectype, M<:BorderedVector{vectype, T}}
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta)
	branch = contres.branch

	if contparms.computeEigenValues == false
		l =  Plots.@layout [a{0.5w} b{0.5w}; c d]
	else
		l =  Plots.@layout [a{0.5w} b{0.5w}; c d;e]
	end
	Plots.plot(layout=l)

	# plot the branch of solutions
	plotBranch!(contres,  xlabel="p", ylabel="|x|", label="", subplot=1 )

	plot!(branch[1, :], xlabel="s", ylabel="p", label="", subplot=2)
	plot!(branch[2, :], xlabel="it", ylabel="|x|",	label="", subplot=3)

	# add the bifurcation points along the branch
	if length(contres.bifpoint)>1
		scatter!(map(x->x[2],contres.bifpoint[2:end]), map(x->x[4],contres.bifpoint[2:end]), label="", color = map(x->colorbif[x[1]],contres.bifpoint[2:end]), markersize=3, markerstrokewidth=0, subplot=3) |> display
	end

	if contparms.computeEigenValues
		for ii=1:length(contres.eig)
			scatter!(real.(contres.eig[ii][1]), imag.(contres.eig[ii][1]), subplot=5, label="", markersize = 1, color=:black)
		end
	end
	plotuserfunction(sol.u)

display(title!(""))
end

"""
Plot the branch of solutions from a `ContResult`. You can also pass parameters like `plotBranch(br, marker = :dot)`.
For the continuation diagram, the legend is as follows `(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow)`
"""
function plotBranch(contres::ContResult; kwargs...)
	plot([],[],label="")
	plotBranch!(contres; kwargs...)
end

"""
Plot all the branches contained in `brs` in a single figure. Convenient when many bifurcation diagram have been computed.
"""
function plotBranch(brs::Vector; kwargs...)
	plotBranch(brs[1]; kwargs...)
	for ii=2:length(brs)
		plotBranch!(brs[ii]; kwargs...) |> display
	end
end

"""
Append to the current plot the plot of the branch of solutions from a `ContResult`. You can also pass parameters like `plotBranch!(br, marker = :dot)`
"""
function plotBranch!(contres::ContResult; kwargs...)
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow)
	branch = contres.branch
	if length(contres.stability) > 2
		plot!(branch[1, :], branch[2, :], linewidth = 1 .+ 3contres.stability ; kwargs...)
	else
		plot!(branch[1, :], branch[2, :]; kwargs...)
	end
	# add the bifurcation points along the branch
	if length(contres.bifpoint)>=1
		id = 1
		contres.bifpoint[1][1] == :none ? id = 2 : id = 1
		scatter!(map(x->x[3],contres.bifpoint[id:end]), map(x->x[4],contres.bifpoint[id:end]), label="", color = map(x->colorbif[x[1]],contres.bifpoint[id:end]), markersize=3, markerstrokewidth=0 ; kwargs...) |> display
	end
end
###############################################################################################
function detectBifucation(contparams, contResult, z, tau, printsolution, verbosity)
	branch = contResult.branch

	# Fold point detection based on continuation parameter monotony
	if contparams.detect_fold && size(branch)[2] > 2 && (branch[1, end] - branch[1, end-1]) * (branch[1, end-1] - branch[1, end-2]) < 0
		(verbosity > 1) && printstyled(color=:red, "Fold bifurcation point!! between $(branch[1, end-1]) and  $(branch[1, end]) \n")
		push!(contResult.bifpoint, (:fold,
							length(branch)-1,
							branch[1, end-1],
							printsolution(z.u), copy(z.u), normalize(tau.u), 0))
	end
	if contparams.detect_bifurcation == false
		return
	end

	# update number of unstable eigenvalues
	n_unstable = mapreduce(x -> round(real(x), digits=6) > 0, +, contResult.eig[end][1])
	push!(contResult.n_unstable, n_unstable)

	# update number of unstable eigenvalues with nonzero imaginary part
	n_imag = mapreduce(x -> (abs(round(imag(x), digits = 6)) > 0) * (round(real(x), digits = 6) > 0), +, contResult.eig[end][1])
	push!(contResult.n_imag, n_imag)

	# computation of the index of the bifurcating eigenvalue
	ind_bif = n_unstable
	if n_unstable < contResult.n_unstable[end-1]
		ind_bif += 1
	end

	# Hopf / BP bifurcation point detection based on eigenvalue distribution
	if size(branch)[2] > 1
		if abs(contResult.n_unstable[end] - contResult.n_unstable[end-1]) == 1
			@show  branch
			push!(contResult.bifpoint, (:bp,
								length(branch)-1,
								branch[1, end-1],
								printsolution(z.u),
								copy(z.u),
								normalize(tau.u), ind_bif))
		elseif abs(contResult.n_unstable[end] - contResult.n_unstable[end-1]) == 2
			if abs(contResult.n_imag[end] - contResult.n_imag[end-1]) == 2
				push!(contResult.bifpoint, (:hopf,
									length(branch)-1,
									branch[1, end-1],
									printsolution(z.u),
									copy(z.u), zero(tau.u), ind_bif))
			else
				push!(contResult.bifpoint, (:bp,
									length(branch)-1,
									branch[1, end-1],
									printsolution(z.u),
									copy(z.u),
									normalize(tau.u), n_unstable))
			end
		elseif abs(contResult.n_unstable[end] - contResult.n_unstable[end-1]) >2
			push!(contResult.bifpoint, (:nd,
							length(branch)-1,
							branch[1, end-1],
							printsolution(z.u),
							copy(z.u),
							normalize(tau.u), ind_bif))
		end
	end
end
###############################################################################################
"""
Compute Jacobian by Finite Differences
"""
function finiteDifferences(F, x::AbstractVector)
	f = F(x)
	epsi = 1e-9
	N = length(x)
	J = zeros(N, N)
	x1 = copy(x)
	for i=1:N
		x1[i] += epsi
		J[:, i] .= (F(x1) .- F(x)) / epsi
		x1[i] -= epsi
	end
	return J
end
###############################################################################################
"""
Save solution / data in JLD2 file
- `filename` is for example "example.jld2"
- `sol` is the solution
- `p` is the parameter
- `i` is the index of the solution to be saved
"""
function saveSolution(filename, sol, p, i::Int64, br::ContResult, contParam)
	# create a group in the JLD format
	jldopen(filename*".jld2", "a+") do file
	    mygroup = JLD2.Group(file, "sol-$i")
	    mygroup["sol"] = sol
		mygroup["param"] = p
	end

	jldopen(filename*"-branch.jld2", "w") do file
		file["branch"] = br
		file["contParam"] = contParam
	end
end
###############################################################################################
# this trick is extracted from KrylovKit. It allows for the Jacobian to be specified as a matrix (sparse / dense) or as a function.
apply(A::AbstractMatrix, x::AbstractVector) = A * x
apply(f, x) = f(x)

apply!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = mul!(y, A, x)
apply!(y, f, x) = copyto!(y, f(x))
