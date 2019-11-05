using RecursiveArrayTools # for bifurcation point handling
import Base: show, length
####################################################################################################
# Structure to hold result
@with_kw struct ContResult{T, Vectype, Eigenvectype, Biftype}
	# this vector is used to hold (param, normC(u), Newton iterations, ds)
	branch::VectorOfArray{T, 2, Array{Vector{T}, 1}} = VectorOfArray([zeros(T, 4)])

	# the following holds the eigen elements at the index of the point along the curve. This index is the last element of eig[1] (for example). Recording this index is useful for storing only some eigenelements and not all of them along the curve
	eig::Vector{Tuple{Array{Complex{T}, 1}, Eigenvectype, Int64}}

	# the following holds information about the detected bifurcation points like
	# [(:none, step, normC(u), u, tau, eigenvalue_index)] where tau is the tangent along the curve and eigenvalue_index is the index of the eigenvalue in eig (see above) which change stability
	bifpoint::Vector{Biftype}

	# whether the associated point is linearly stable
	stability::Vector{Bool} = [false]

	# number of eigenvalues with positive real part and non zero imaginary part
	n_imag::Vector{Int64}

	# number of eigenvalues with positive real part
	n_unstable::Vector{Int64}
end

length(br::ContResult) = length(br.branch[4, :])

function show(io::IO, br::ContResult)
	println(io, "Branch number of points: ", length(br.branch))
	if length(br.bifpoint) >0
		println(io, "Bifurcation points:")
		for ii in eachindex(br.bifpoint)
			bp  = br.bifpoint[ii]
			# println(io, "- $ii, ", bp.type, " point, at p = ", bp.param, ", index = ", bp.idx)
			@printf(io, "- %3i, %7s point at p = %4.8f, index = %3i\n", ii, bp.type, bp.param, bp.idx)
		end
	end
end
####################################################################################################
function displayIteration(i, funceval, residual, itlinear = 0)
	(i == 0) && println("\n Newton Iterations \n   Iterations      Func-count      f(x)      Linear-Iterations\n")
	if length(itlinear)==1
		@printf("%9d %16d %14.4e %9d\n", i, funceval, residual, itlinear);
	else
		@printf("%9d %16d %14.4e (%6d, %6d)\n", i, funceval, residual, itlinear[1], itlinear[2]);
	end
end
####################################################################################################
"""
Plot the continued branch of solutions
"""
function plotBranchCont(contres::ContResult, sol::M, contparms, plotuserfunction) where {T, vectype, M<:BorderedArray{vectype, T}}
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta)
	branch = contres.branch

	if contparms.computeEigenValues == false
		l =  Plots.@layout [a{0.5w} b{0.5w}; c d]
	else
		l =  Plots.@layout [a{0.5w} b{0.5w}; c d;e]
	end
	Plots.plot(layout = l)

	# plot the branch of solutions
	plotBranch!(contres, xlabel="p", ylabel="|x|", label="", subplot=1)
	plot!(branch[1, :],	 xlabel="s", ylabel="p",   label="", subplot=2)
	plot!(branch[2, :], xlabel="it", ylabel="|x|", label="", subplot=3)

	# add the bifurcation points along the branch
	if length(contres.bifpoint)>1
		scatter!(map(x -> x.idx, contres.bifpoint[2:end]),
				 map(x -> x.printsol ,contres.bifpoint[2:end]),
				label="", color = map(x -> colorbif[x.type], contres.bifpoint[2:end]), markersize=3, markerstrokewidth=0, subplot=3)
	end

	if contparms.computeEigenValues
		ii = length(contres.eig)
		scatter!(real.(contres.eig[ii][1]), imag.(contres.eig[ii][1]), subplot=5, label="", markersize = 1, color=:black)

	end
	plotuserfunction(sol.u, subplot = 4)

	display(title!(""))
end

"""
Plot the branch of solutions from a `ContResult`. You can also pass parameters like `plotBranch(br, marker = :dot)`.
For the continuation diagram, the legend is as follows `(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow)`
"""
function plotBranch(contres, plot_fold = true; kwargs...)
	# we do not specify the type of contres, not convenient when using JLD2
	plot()
	plotBranch!(contres, plot_fold; kwargs...)
end

"""
Plot all the branches contained in `brs` in a single figure. Convenient when many bifurcation diagram have been computed.
"""
function plotBranch(brs::Vector, plot_fold = true; kwargs...)
	# we do not specify the type of contres, not convenient when using JLD2
	plotBranch(brs[1], plot_fold; kwargs...)
	for ii=2:length(brs)
		plotBranch!(brs[ii], plot_fold; kwargs...) |> display
	end
end

"""
Append to the current plot the plot of the branch of solutions from a `ContResult`. You can also pass parameters like `plotBranch!(br, marker = :dot)`
"""
function plotBranch!(contres, plot_fold = true; kwargs...)
	# we do not specify the type of contres, not convenient when using JLD2
	colorbif = Dict(:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow)
	branch = contres.branch
	if length(contres.stability) > 2
		plot!(branch[1, :], branch[2, :], linestyle = map(x->isodd(x) ? :solid : :dot, contres.stability) ; kwargs...)
	else
		plot!(branch[1, :], branch[2, :]; kwargs...)
	end
	# add the bifurcation points along the branch
	if length(contres.bifpoint) >= 1
		id = 1
		contres.bifpoint[1].type == :none ? id = 2 : id = 1
		if plot_fold
			bifpt = contres.bifpoint[id:end]
		else
			bifpt = [pt for pt in contres.bifpoint[id:end] if pt.type != :fold]
		end
		scatter!(map(x -> x.param, bifpt), map(x -> x.printsol, bifpt), label="", color = map(x->colorbif[x.type], bifpt), markersize=3, markerstrokewidth=0 ; kwargs...)
	end
end
####################################################################################################
function computeEigenvalues(contparams, contres, J, step)
	# this line is to ensure we compute enough eigenvalues to probe stability
	nev_ = max(sum( real.(contres.eig[end][1]) .> 0) + 2, contparams.nev)
	eig_elements = contparams.newtonOptions.eigsolver(J, nev_)
	if mod(step, contparams.save_eig_every_n_steps) == 0
		if contparams.save_eigenvectors
			push!(contres.eig, (eig_elements[1], eig_elements[2], step + 1))
		else
			push!(contres.eig, (eig_elements[1], empty(eig_elements[2]), step + 1))
		end
	end
	eig_elements
end

################################################################################################
function normalize(x)
	out = copy(x)
	rmul!(out, norm(x))
	return out
end

function detectBifucation(contparams, contres, z, tau, normC, printsolution, verbosity)
	branch = contres.branch

	# Fold point detection based on continuation parameter monotony
	if contparams.detect_fold && size(branch)[2] > 2 && (branch[1, end] - branch[1, end-1]) * (branch[1, end-1] - branch[1, end-2]) < 0
		(verbosity > 1) && printstyled(color=:red, "Fold bifurcation point!! between $(branch[1, end-1]) and  $(branch[1, end]) \n")
		push!(contres.bifpoint, (type = :fold,
							idx = length(branch)-1,
							param = branch[1, end-1],
							norm = normC(z.u),
							printsol = printsolution(z.u),
							u = copy(z.u), tau = normalize(tau.u), ind_bif = 0))
	end
	if contparams.detect_bifurcation == false
		return
	end

	# update number of unstable eigenvalues
	n_unstable = mapreduce(x -> round(real(x), digits = 15) > 0, +, contres.eig[end][1])
	push!(contres.n_unstable, n_unstable)

	# update number of unstable eigenvalues with nonzero imaginary part
	n_imag = mapreduce(x -> (abs(round(imag(x), digits = 15)) > 0) * (round(real(x), digits = 15) > 0), +, contres.eig[end][1])
	push!(contres.n_imag, n_imag)

	# computation of the index of the bifurcating eigenvalue
	ind_bif = n_unstable
	if n_unstable < contres.n_unstable[end-1]
		ind_bif += 1
	end

	idxmoststable = contres.n_unstable[end] < contres.n_unstable[end-1] ? length(contres) : length(contres) - 1

	# Hopf / BP bifurcation point detection based on eigenvalue distribution
	if size(branch)[2] > 1
		if abs(contres.n_unstable[end] - contres.n_unstable[end-1]) == 1
			# in order to match the Fold biurcation, we chose the most stable point
			push!(contres.bifpoint, (type = :bp,
					idx = idxmoststable,
					param = branch[1, idxmoststable],
					norm = normC(z.u),
					printsol = branch[2, idxmoststable],
					u = copy(z.u),
					tau = normalize(tau.u), ind_bif = ind_bif))
		elseif abs(contres.n_unstable[end] - contres.n_unstable[end-1]) == 2
			if abs(contres.n_imag[end] - contres.n_imag[end-1]) == 2
				push!(contres.bifpoint, (type = :hopf,
					idx = length(branch)-1,
					param = branch[1, end-1],
					norm = normC(z.u),
					printsol = printsolution(z.u),
					u = copy(z.u),
					tau = zero(tau.u), ind_bif = ind_bif))
			else
				push!(contres.bifpoint, (type = :bp,
					idx = idxmoststable,
					param = branch[1, idxmoststable],
					norm = normC(z.u),
					printsol = branch[2, idxmoststable],
					u = copy(z.u),
					tau = normalize(tau.u), ind_bif = n_unstable))
			end
		elseif abs(contres.n_unstable[end] - contres.n_unstable[end-1]) >2
			push!(contres.bifpoint, (type = :nd,
					idx = length(branch)-1,
					param = branch[1, end-1],
					norm = normC(z.u),
					printsol = printsolution(z.u),
					u = copy(z.u),
					tau = normalize(tau.u), ind_bif = ind_bif))
		end
	end
end
####################################################################################################
"""
This function is used to initialize the struct `br` according to the options passed by `contParams`
"""
function initContRes(br, u0, evsol, contParams::ContinuationPar)
	T = eltype(br)
	bif0 = (type = :none, idx = 0, param = T(0.), norm  = T(0.), printsol = T(0.), u = u0, tau = u0, ind_bif = 0)
	if contParams.computeEigenValues
		contRes = ContResult{T, typeof(u0), typeof(evsol[2]), typeof(bif0)}(
			branch = br,
			bifpoint = [bif0],
			n_imag = [0],
			n_unstable = [0],
			eig = [(evsol[1], evsol[2], 0)] )
		if !contParams.save_eigenvectors
			empty!(contRes.eig[1][2])
		end
		# whether the current solution is stable
		contRes.stability[1]  = mapreduce(x -> real(x) < 0, *, evsol[1])
		contRes.n_unstable[1] = mapreduce(x -> round(real(x), digits=14) > 0, +, evsol[1])
		if length(evsol[1][1:contRes.n_unstable[1]])>0
			contRes.n_imag[1] = mapreduce(x -> round(imag(x), digits=14) > 0, +, evsol[1][1:contRes.n_unstable[1]])
		else
			contRes.n_imag[1] = 0
		end
		return contRes
	else
		return ContResult{T, typeof(u0), Array{Complex{T}}, typeof(bif0)}(
				branch = br,
				bifpoint = [bif0],
				n_imag = [0],
				n_unstable = [0],
				eig = [([Complex{T}(1)], zeros(Complex{T}, 2, 2), 0)])
	end
end
####################################################################################################
"""
Compute Jacobian by Finite Differences
"""
function finiteDifferences(F, x::AbstractVector; δ = 1e-9)
	f = F(x)
	epsi = 1e-9
	N = length(x)
	J = zeros(eltype(f), N, N)
	x1 = copy(x)
	for i=1:N
		x1[i] += δ
		J[:, i] .= (F(x1) .- F(x)) / δ
		x1[i] -= δ
	end
	return J
end
####################################################################################################
"""
Save solution / data in JLD2 file
- `filename` is for example "example.jld2"
- `sol` is the solution
- `p` is the parameter
- `i` is the index of the solution to be saved
"""
function saveSolution(filename, sol, p, i::Int64, br::ContResult, contParam::ContinuationPar)
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
####################################################################################################
# this trick is extracted from KrylovKit. It allows for the Jacobian to be specified as a matrix (sparse / dense) or as a function.
apply(A::AbstractMatrix, x::AbstractVector) = A * x
apply(f, x) = f(x)

# apply!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = mul!(y, A, x)
# apply!(y, f, x) = copyto!(y, f(x))
