using Revise
	using DiffEqOperators, ForwardDiff
	using BifurcationKit, LinearAlgebra, Plots, SparseArrays, Parameters, Setfield
	const BK = BifurcationKit

norminf = x -> norm(x, Inf)
normbratu = x -> norm(x .* w) / sqrt(length(x))
##########################################################################################
# plotting function
plotsol!(x, nx = Nx, ny = Ny; kwargs...) = heatmap!(LinRange(0,1,nx), LinRange(0,1,ny), reshape(x, nx, ny)'; color = :viridis, xlabel = "x", ylabel = "y", kwargs...)
plotsol(x, nx = Nx, ny = Ny; kwargs...) = (plot();plotsol!(x, nx, ny; kwargs...))

function Laplacian2D(Nx, Ny, lx, ly, bc = :Neumann)
	hx = 2lx/Nx
	hy = 2ly/Ny
	D2x = CenteredDifference(2, 2, hx, Nx)
	D2y = CenteredDifference(2, 2, hy, Ny)

	if bc == :Neumann
		Qx = Neumann0BC(hx)
		Qy = Neumann0BC(hy)
	elseif bc == :Dirichlet
		Qx = Dirichlet0BC(typeof(hx))
		Qy = Dirichlet0BC(typeof(hy))
	end

	D2xsp = sparse(D2x * Qx)[1]
	D2ysp = sparse(D2y * Qy)[1]
	A = kron(sparse(I, Ny, Ny), D2xsp) + kron(D2ysp, sparse(I, Nx, Nx))
	return A, D2x
end

ϕ(u, λ)  = -10(u-λ*exp(u))
dϕ(u, λ) = -10(1-λ*exp(u))

function NL!(dest, u, p)
	@unpack λ = p
	dest .= ϕ.(u, λ)
	return dest
end

NL(u, p) = NL!(similar(u), u, p)

function Fmit!(f, u, p)
	mul!(f, p.Δ, u)
	f .= f .+ NL(u, p)
	return f
end

Fmit(u, p) = Fmit!(similar(u), u, p)

function dFmit(x, p, dx)
	f = similar(dx)
	mul!(f, p.Δ, dx)
	nl = d1NL(x, p, dx)
	f .= f .+ nl
end

function JFmit(x,p)
	J = p.Δ
	dg = dϕ.(x, p.λ)
	return J + spdiagm(0 => dg)
end

# computation of the derivatives
d1NL(x, p, dx) = ForwardDiff.derivative(t -> NL(x .+ t .* dx, p), 0.)

# compute 3-Jet
jet = BK.get3Jet(Fmit, JFmit)
####################################################################################################
Nx = 30
	Ny = 30
	lx = 0.5
	ly = 0.5

	Δ, = Laplacian2D(Nx, Ny, lx, ly)
	par_mit = (λ = .01, Δ = Δ)
	sol0 = 0*ones(Nx, Ny) |> vec
	const w = (lx .+ LinRange(-lx,lx,Nx)) * (LinRange(-ly,ly,Ny))' |> vec
	w .-= minimum(w)
####################################################################################################
eigls = EigArpack(20.5, :LM)
# eigls = EigKrylovKit(dim = 70)
# eigls = EigArpack()
	opt_newton = NewtonPar(tol = 1e-8, verbose = true, eigsolver = eigls, maxIter = 20)
	out, hist, flag = newton(Fmit, JFmit, sol0, par_mit, opt_newton, normN = norminf)

plotsol(out)
####################################################################################################
function finSol(z, tau, step, br; k...)
	if length(br.specialpoint)>0
		if br.specialpoint[end].step == step
			BK._show(stdout, br.specialpoint[end], step)
		end
	end
	return true
end

function cb(x,f,J,res,it,itl,optN; kwargs...)
	_x = get(kwargs, :z0, nothing)
	fromNewton = get(kwargs, :fromNewton, false)
	if ~fromNewton
		return (norm(_x.u - x) < 20.5 && abs(_x.p - kwargs[:p]) < 0.05)
	end
	true
end

# optional parameters for continuation
kwargsC = (verbosity = 3,
	plot = true,
	printSolution = (x, p) -> (x = normbratu(x), n2 = norm(x), n∞ = norminf(x)),
	plotSolution = (x, p; k...) -> plotsol!(x ; k...),
	callbackN = cb,
	finaliseSolution = finSol,
	normC = norminf
	)

opts_br = ContinuationPar(dsmin = 0.0001, dsmax = 0.04, ds = 0.005, pMax = 3.5, pMin = 0.01, detectBifurcation = 3, nev = 50, plotEveryStep = 10, newtonOptions = (@set opt_newton.verbose = false), maxSteps = 251, precisionStability = 1e-6, nInversion = 6, maxBisectionSteps = 25)

	br, = @time continuation(
		Fmit, JFmit,
		sol0, par_mit, (@lens _.λ), opts_br; kwargsC...)

plot(br)
####################################################################################################
# automatic branch switching

br1, = continuation(jet..., br, 3; kwargsC...)

plot(br,br1,plotfold=false)

br2, = continuation(jet..., br1, 1; kwargsC...)

plot(br, br1, br2, plotfold=false)
####################################################################################################
# bifurcation diagram
function optionsCont(x,p,l; opt = opts_br)
	if l <= 1
		return opt
	elseif l==2
		return setproperties(opt ;detectBifurcation = 3,ds = 0.001, a = 0.75)
	else
		return setproperties(opt ;detectBifurcation = 3,ds = 0.00051, dsmax = 0.01)
	end
end

diagram = @time bifurcationdiagram(jet..., sol0, par_mit, (@lens _.λ), 3, optionsCont; kwargsC...,
	usedeflation = true,
	halfbranch = true,
	)

bifurcationdiagram!(jet..., getBranch(diagram, (14,)), (current = 3, maxlevel = 6), optionsCont;
	kwargsC..., usedeflation = true, halfbranch = true,)

code = ()
	plot(diagram; code = code,  plotfold = false, putspecialptlegend=false, markersize=2, vars = (:param, :x))
	# plot!(br)
	# xlims!(0.01, 0.4)
	title!("#branches = $(size(getBranch(diagram, code)))")
	# xlims!(0.01, 0.065, ylims=(2.5,6.5))

plot(getBranchesFromBP(diagram, 4); plotfold = false, legend = false, vars = (:param, :x))

BK.getBranch(diagram, (10,1,)) |> plot

####################################################################################################
# analyse 2d bifurcation point
bp2d = @time BK.computeNormalForm(jet..., br, 4,  verbose=true, nev=30)

BK.nf(bp2d)[2] |> println

using ProgressMeter
Nd = 100
	L = 3.9
	X = LinRange(-L,L, Nd)
	Y = LinRange(-L,L, Nd)
	P = LinRange(-0.001,0.001, Nd+1)

V1a = @showprogress [bp2d(Val(:reducedForm),[x1,y1], p1)[1] for p1 in P, x1 in X, y1 in Y]

Ind1 = findall( abs.(V1a) .<= 9e-4 * maximum(abs.(V1a)))

	V2a = @showprogress [bp2d(Val(:reducedForm),[X[ii[2]],Y[ii[3]]], P[ii[1]])[2] for ii in Ind1]

	Ind2 = findall( abs.(V2a) .<= 3e-3 * maximum(abs.(V2a)))
	@show length(Ind2)



resp = Float64[]
	resx = Vector{Float64}[]
	resnrm = Float64[]
	@showprogress for k in Ind2
		ii = Ind1[k]
		push!(resp, P[ii[1]])
		# push!(resx, max(X[ii[2]],Y[ii[3]]))
		push!(resnrm, sqrt(X[ii[2]]^2+Y[ii[3]]^2))
		push!(resx, [X[ii[2]], Y[ii[3]]])
	end


using LaTeXStrings

plot(
	scatter(1e4resp, map(x->x[1], resx), map(x->x[2], resx); label = "", markerstrokewidth=0, xlabel = L"10^4 \cdot \lambda", ylabel = L"x_1", zlabel = L"x_2", zcolor = resnrm, color = :viridis,colorbar=false),
	scatter(1e4resp, resnrm; label = "", markersize =2, markerstrokewidth=0, xlabel = L"10^4 \cdot \lambda", ylabel = L"\|x\|"))

####################################################################################################
bp2d = @time computeNormalForm(jet..., br, 2, nev = 30; issymmetric = true)

res, = BK.continuation(jet..., br, 2,
	setproperties(opts_br; detectBifurcation = 3, ds = 0.001, pMin = 0.01, maxSteps = 32 ) ;
	nev = 30, verbosity = 3,
	kwargsC...,
	)

plot(res..., br ;plotfold= false)


δp = 0.005
	deflationOp = DeflationOperator(2, dot, 1.0, [zeros(2)])
		success = [0]
while sum(success) < 10
	outdef1, _, flag, _ = newton((x, p) -> bp2d(Val(:reducedForm), x, p[1]), rand(2), [δp], NewtonPar(maxIter = 50), deflationOp)
	@show flag
	flag && push!(deflationOp, outdef1)
	(flag==false) && push!(success, 1)
end
	println("--> found $(length(deflationOp)) solutions")

plotsol(bp(deflationOp[3], δp))
solbif, flag, _ = newton(Fmit, JFmit, bp2d.x0, bp2d(deflationOp[3], δp), (@set par_mit.λ = bp2d.p + δp), opts_br.newtonOptions)[1]

plotsol(solbif-0*bp2d(deflationOp[2], δp))

brnf1, = continuation(Fmit, JFmit, solbif, (@set par_mit.λ = bp2d.p + δp), (@lens _.λ), setproperties(opts_br; ds = 0.005);
	printSolution = (x, p) -> norm(x),
	plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...),
	plot = true, verbosity = 3, normC = norminf)

branches2 = (br,br1,br2,brnf1)
push!(branches2, brnf1)
# plot([br,br1,br2])
# plot!(brnf1)

brnf2, = continuation(Fmit, JFmit, solbif, (@set par_mit.λ = bp2d.p + δp), (@lens _.λ), setproperties(opts_br; ds = -0.005);
	printSolution = (x, p) -> norm(x),
	plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...),
	plot = true, verbosity = 3, normC = norminf)

# plot([br,br1,br2]);plot!(brnf1);plot!(brnf2)
plot(branches2...)
plot!(brnf2)
####################################################################################################
# find isolated branch, see Farrell et al.
deflationOp = DeflationOperator(2, 1.0, [out])
optdef = setproperties(opt_newton; tol = 1e-8, maxIter = 150)

# eigen-elements close to the second bifurcation point on the branch
# of homogenous solutions
vp, ve, _, _= eigls(JFmit(out, @set par_mit.λ = br.specialpoint[2].param), 5)

for ii=1:size(ve, 1)
		outdef1, _, flag, _ = @time newton(
			Fmit, JFmit,
			# initial guess for newton
			real.(br.specialpoint[2].x .+ 0.01 .* ve[ii] .* (1 .+ 0.01 .* rand(Nx*Ny))),
			(@set par_mit.λ = br.specialpoint[2].param + 0.005),
			optdef, deflationOp)
			flag && push!(deflationOp, outdef1)
	end
	length(deflationOp)



l = @layout grid(3,2)
	plot(layout = l)
	for ii=1:length(deflationOp)
		plotsol!(deflationOp[ii], title="$ii", subplot = ii, label = "", xlabel="$ii", colorbar=true)
	end
	title!("")

brdef1, _ = @time BK.continuation(
	Fmit, JFmit,
	deflationOp[3], (@set par_mit.λ = br.specialpoint[2].param + 0.005), (@lens _.λ),
	# bp2d([0.6,0.6], -0.01), br.specialpoint[2].param - 0.005,
	setproperties(opts_br;ds = 0.001, detectBifurcation = 0, dsmax = 0.01, maxSteps = 500);
	verbosity = 3, plot = true,
	printSolution = (x, p) ->  normbratu(x),
	plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...), normC = norminf)

plot(br,br1,br2, brdef1,plotfold=false)


brdef2, _ = @time BK.continuation(
	Fmit, JFmit,
	deflationOp[5], (@set par_mit.λ = br.specialpoint[2].param + 0.005), (@lens _.λ),
	setproperties(opts_br;ds = -0.001, detectBifurcation = 0, dsmax = 0.02);
	verbosity = 3, plot = true,
	printSolution = (x, p) ->  normbratu(x),
	plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...), normC = norminf)

plot(br,br1,br2, brdef1, brdef2,plotfold=false, putspecialptlegend = false)

plot(brdef1, brdef2,plotfold = false, putspecialptlegend = false)
####################################################################################################
# deflated continuation
brdef2, _ = @time BK.continuation(
	Fmit, JFmit, (@set par_mit.λ = 0.367), (@lens _.λ),
	ContinuationPar(opts_br; ds = -0.0001, maxSteps = 800000, plotEveryStep = 10, detectBifurcation = 0),
	DeflationOperator(2.0, dot, 1., ([sol0]));
	showplot=true, verbosity = 2,
	perturbSolution = (x,p,id) -> (x .+ 0.1 .* rand(length(x))),
	printSolution = (x, p) ->  normbratu(x),
	plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...),
	normN = norminf)

plot(brdef2..., color=:red)
