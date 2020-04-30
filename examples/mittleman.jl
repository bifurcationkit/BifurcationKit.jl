using Revise
	using DiffEqOperators, ForwardDiff
	using PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays, Parameters, Setfield
	const PALC = PseudoArcLengthContinuation

norminf = x -> norm(x, Inf)
plotsol!(x, nx = Nx, ny = Ny; kwargs...) = heatmap!(reshape(x, nx, ny); color = :viridis, kwargs...)
plotsol(x, nx = Nx, ny = Ny; kwargs...) = (plot();plotsol!(x, nx, ny; kwargs...))

function Laplacian2D(Nx, Ny, lx, ly, bc = :Neumann)
	hx = 2lx/Nx
	hy = 2ly/Ny
	D2x = CenteredDifference(2, 2, hx, Nx)
	D2y = CenteredDifference(2, 2, hy, Ny)

	Qx = Neumann0BC(hx)
	Qy = Neumann0BC(hy)

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
d1Fmit(x, p, dx) = ForwardDiff.derivative(t -> Fmit(x .+ t .* dx, p), 0.)
d2Fmit(x, p, dx1, dx2) = ForwardDiff.derivative(t2 -> ForwardDiff.derivative( t1 -> Fmit(x .+ t1 .* dx1 .+ t2 .* dx2, p), 0.), 0.)
####################################################################################################
Nx = 200
	Ny = 100
	lx = 0.5
	ly = 0.5

	Δ = Laplacian2D(Nx, Ny, lx, ly)[1]
	par_mit = (λ = .05, Δ = Δ)
	sol0 = 0*ones(Nx, Ny) |> vec
####################################################################################################
eigls = EigArpack(0.5, :LM)
	opt_newton = PALC.NewtonPar(tol = 1e-8, verbose = true, eigsolver = eigls, maxIter = 20)
	out, hist, flag = @time PALC.newton(
		x ->  Fmit(x, par_mit),
		x -> JFmit(x, par_mit),
		sol0, opt_newton, normN = norminf)

plotsol(out)

####################################################################################################
opts_br = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds = 0.01, pMax = 3.5, pMin = 0.025, detectBifurcation = 2, nev = 30, plotEveryNsteps = 10, newtonOptions = (@set opt_newton.verbose = true), maxSteps = 100, precisionStability = 1e-6, nInversion = 4, dsminBisection = 1e-7, maxBisectionSteps = 25)

	br, _ = @time PALC.continuation(
		(x, p) -> Fmit(x, @set par_mit.λ = p),
		(x, p) -> JFmit(x, @set par_mit.λ = p),
		sol0, 0.05,
		printSolution = (x, p) -> norm(x),
		plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...),
		opts_br; plot = true, verbosity = 3, normC = norminf)

####################################################################################################
# branch switching

D(f, x, p, dx) = ForwardDiff.derivative(t->f(x .+ t .* dx, p), 0.)

d1Fmit(x,p,dx1) = D((z, p0) -> Fmit(z, p0), x, p, dx1)
d2Fmit(x,p,dx1,dx2) = D((z, p0) -> d1Fmit(z, p0, dx1), x, p, dx2)
d3Fmit(x,p,dx1,dx2,dx3) = D((z, p0) -> d2Fmit(z, p0, dx1, dx2), x, p, dx3)

br1, _ = continuation((x, p) -> Fmit(x, @set par_mit.λ = p),
 		(x, p) -> JFmit(x, @set par_mit.λ = p),
		(x, p, dx1, dx2) -> d2Fmit(x, (@set par_mit.λ = p), dx1, dx2),
		(x, p, dx1, dx2, dx3) -> d3Fmit(x, (@set par_mit.λ = p), dx1, dx2, dx3),
		br, 3, setproperties(opts_br;ds = 0.001, maxSteps = 40); verbose=true,
		verbosity = 3, plot = true,
		printSolution = (x, p) -> norm(x),
		plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...),
		normC = norminf)

plot([br,br1],plotfold=false)

br2, _ = continuation((x, p) -> Fmit(x, @set par_mit.λ = p),
 		(x, p) -> JFmit(x, @set par_mit.λ = p),
		(x, p, dx1, dx2) -> d2Fmit(x, (@set par_mit.λ = p), dx1, dx2),
		(x, p, dx1, dx2, dx3) -> d3Fmit(x, (@set par_mit.λ = p), dx1, dx2, dx3),
		br1, 1, setproperties(opts_br;ds = 0.001, maxSteps = 40); verbose=true, nev = 15,
		verbosity = 3, plot = true,
		printSolution = (x, p) -> norm(x),
		plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...), normC = norminf)

plot([br,br1,br2],plotfold=false)

br3, _ = continuation((x, p) -> Fmit(x, @set par_mit.λ = p),
 		(x, p) -> JFmit(x, @set par_mit.λ = p),
		(x, p, dx1, dx2) -> d2Fmit(x, (@set par_mit.λ = p), dx1, dx2),
		(x, p, dx1, dx2, dx3) -> d3Fmit(x, (@set par_mit.λ = p), dx1, dx2, dx3),
		br, 4, setproperties(opts_br;ds = 0.001, maxSteps = 40); verbose=true, nev = 15,
		verbosity = 3, plot = true,
		printSolution = (x, p) -> norm(x),
		plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...), normC = norminf)

plot([br,br1,br2,br3],plotfold=false)


####################################################################################################
# find isolated branch, see Farrell et al.
deflationOp = DeflationOperator(2.0, (x, y) -> dot(x, y), 1.0, [out])
optdef = setproperties(opt_newton; tol = 1e-8, maxIter = 100)

# we find an initial guess. It has to satisfy Neumann BC
vp, ve, _, _= eigls(JFmit(out, @set par_mit.λ = br.bifpoint[2].param + 0.005), 25)

for ii=1:size(ve,2)
	outdef1, _, flag, _ = @time newton(
		x ->  Fmit(x, @set par_mit.λ = br.bifpoint[2].param + 0.005),
		x -> JFmit(x, @set par_mit.λ = br.bifpoint[2].param + 0.005),
		br.bifpoint[2].x .+ 0.01.*ve[:,ii] .* (1 .+ 0.1 .* rand(Nx*Ny)),
		# out,
		optdef, deflationOp)
		flag && push!(deflationOp, outdef1)
	end
	length(deflationOp)

plotsol(deflationOp[6])

brdef1, _ = @time PALC.continuation(
	(x, p) -> Fmit(x, @set par_mit.λ = p),
	(x, p) -> JFmit(x, @set par_mit.λ = p),
	deflationOp[6], br.bifpoint[2].param + 0.005, setproperties(opts_br;ds = 0.001, detectBifurcation =2, dsmax = 0.01, maxSteps = 500);
	verbosity = 3, plot = true,
	printSolution = (x, p) -> norm(x),
	plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...), normC = norminf)

plot([br,br1,br2,br3, brdef1],plotfold=false)


brdef2, _ = @time PALC.continuation(
	(x, p) -> Fmit(x, @set par_mit.λ = p),
	(x, p) -> JFmit(x, @set par_mit.λ = p),
	deflationOp[6], br.bifpoint[2].param + 0.005, setproperties(opts_br;ds = -0.001, detectBifurcation = 2, dsmax = 0.01);
	verbosity = 3, plot = true,
	printSolution = (x, p) -> norm(x),
	plotSolution = (x, p; kwargs...) -> plotsol!(x ; kwargs...), normC = norminf)

plot([br,br1,br2,br3, brdef1, brdef2],plotfold=false, putbifptlegend = false)
