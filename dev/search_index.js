var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#PseudoArcLengthContinuation.jl-1",
    "page": "Home",
    "title": "PseudoArcLengthContinuation.jl",
    "category": "section",
    "text": "This package aims at solving equations F(ulambda)=0 where lambda inmathbb R starting from an initial guess (u_0lambda_0). It relies on the pseudo arclength continuation algorithm which provides a predictor (u_1lambda_1) from (u_0lambda_0). A Newton method is then used to correct this predictor.The current package focuses on large scale problem and multiple hardware. Hence, the goal is to use Matrix Free / Sparse methods on GPU or a cluster in order to solve non linear equations (PDE for example).Finally, we leave it to the user to take advantage of automatic differentiation."
},

{
    "location": "#Other-softwares-1",
    "page": "Home",
    "title": "Other softwares",
    "category": "section",
    "text": "We were inspired by pde2path. One can also mention the venerable AUTO, or also, XPPAUT, MATCONT and COCO or Trilinos. Most continuation softwares are listed on DSWeb. There is also this MATLAB continuation code by D. Avitabile.In Julia, we have for now a wrapper to PyDSTools, and also Bifurcations.jl.One design choice is that we try not to require u to be a subtype of an AbstractArray as this would forbid the use of spectral methods (like the one from ApproxFun.jl) or some GPU package, e.g. ArrayFire.jl. So far, our implementation does not allow this for Fold / Hopf continuation and computation of periodic orbits. It will be improved later."
},

{
    "location": "#A-word-on-performance-1",
    "page": "Home",
    "title": "A word on performance",
    "category": "section",
    "text": "The examples which follow have not been written with the goal of performance but rather simplicity. One could surely turn them into more efficient codes. The intricacies of PDEs make the writing of efficient code highly problem dependent and one should take advantage of every particularity of the problem under study.For example, in the first example below, one could use BandedMatrices.jl for the jacobian and an inplace modification when the jacobian is called ; using a composite type would be favored. Porting them to GPU would be another option."
},

{
    "location": "#Example-1:-nonlinear-pendulum-1",
    "page": "Home",
    "title": "Example 1: nonlinear pendulum",
    "category": "section",
    "text": "This is a simple example in which we aim at solving Deltatheta+alpha f(thetabeta)=0 with boundary conditions theta(0) = theta(1)=0. This example is coded in examples/chan.jl ; it is a basic example from the Trilinos library. We start with some imports:using PseudoArcLengthContinuation, LinearAlgebra, Plots\nconst Cont = PseudoArcLengthContinuation\n\nsource_term(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)\ndsource_term(x; a = 0.5, b = 0.01) = (1-b*x^2+2*a*x)/(1+b*x^2)^2We then write our functional:function F_chan(x, α, β = 0.)\n	f = similar(x)\n	n = length(x)\n	f[1] = x[1] - β\n	f[n] = x[n] - β\n	for i=2:n-1\n		f[i] = (x[i-1] - 2 * x[i] + x[i+1]) * (n-1)^2 + α * source_term(x[i], b = β)\n	end\n	return f\nendWe want to call a Newton solver. We first need an initial guess:n = 101\nsol = [(i-1)*(n-i)/n^2+0.1 for i=1:n]and parametersa = 3.3Finally, we need to define some parameters for the Newton iterations. This is done by callingopt_newton = Cont.NewtonPar(tol = 1e-11, verbose = true)We call the Newton solver:out, hist, flag = @time Cont.newton(\n		x -> F_chan(x, a, 0.01),\n		sol,\n		opt_newton)and you should seeNewton Iterations\n   Iterations      Func-count      f(x)      Linear-Iterations\n\n        0                1     2.3440e+01         0\n        1                2     1.3774e+00         1\n        2                3     1.6267e-02         1\n        3                4     2.4521e-06         1\n        4                5     5.9356e-11         1\n        5                6     5.8117e-12         1\n  0.102035 seconds (119.04 k allocations: 7.815 MiB)Note that, in this case, we did not give the Jacobian. It was computed internally using Finite Differences. We can now perform numerical continuation wrt the parameter a. Again, we need to define some parameters for the continuation:opts_br0 = Cont.ContinuationPar(dsmin = 0.01, dsmax = 0.15, ds= 0.01, pMax = 4.1)\n	# options for the newton solver\n	opts_br0.newtonOptions.maxIter = 20\n	opts_br0.newtonOptions.tol = 1e-8\n	opts_br0.detect_fold = true\n	opts_br0.maxSteps = 150Then, we can call the continuation routinebr, u1 = @time Cont.continuation((x,p) -> F_chan(x,p, 0.01),\n	out, a,\n	opts_br0,\n	printsolution = x -> norm(x,Inf64),\n	plot = true,\n	plotsolution = (x;kwargs...) -> (plot!(x,subplot=4,ylabel=\"solution\",label=\"\")))and you should see (Image: )The top left figure is the norm of the solution as function of the parameter a. The bottom left figure is the norm of the solution as function of iteration number. The bottom right is the solution for the current value of the parameter.note: Bif. point detection\nKrylov Two Fold points were detected. This can be seen by looking at br.bifpoint or by the black 	dots on the continuation plots."
},

{
    "location": "#Continuation-of-Fold-points-1",
    "page": "Home",
    "title": "Continuation of Fold points",
    "category": "section",
    "text": "We can for example take the first Fold point and create an initial guess to locate it precisely. However, this only works when the jacobian is computed precisely:function Jac_mat(u, α, β = 0.)\n	n = length(u)\n	J = zeros(n, n)\n	J[1, 1] = 1.0\n	J[n, n] = 1.0\n	for i = 2:n-1\n		J[i, i-1] = (n-1)^2\n		J[i, i+1] = (n-1)^2\n    	J[i, i] = -2 * (n-1)^2 + α * dsource_term(u[i], b = β)\n	end\n	return J\nend\n\nindfold = 2\n\noutfold, hist, flag = @time Cont.newtonFold((x,α) -> F_chan(x, α, 0.01),\n				(x, α) -> Jac_mat(x, α, 0.01),\n				br, indfold, #index of the fold point\n				opts_br0.newtonOptions)\n		flag && printstyled(color=:red, \"--> We found a Fold Point at α = \", outfold.p, \", β = 0.01, from \", br.bifpoint[indfold][3],\"\\n\")which gives  0.085458 seconds (98.05 k allocations: 40.414 MiB, 21.55% gc time)\n--> We found a Fold Point at α = 3.1556507316107947, β = 0.01, from 3.155651011218501We can also continue this fold point in the plane (ab) performing a Fold Point Continuation. In the present case, we find a Cusp point.optcontfold = ContinuationPar(dsmin = 0.001, dsmax = 0.05,ds= 0.01, pMax = 4.1, pMin = 0., a = 2., theta = 0.3)\n	optcontfold.newtonOptions.tol = 1e-8\n	outfoldco, hist, flag = @time Cont.continuationFold(\n		(x, α, β) ->  F_chan(x, α, β),\n		(x, α, β) -> Jac_mat(x, α, β),\n		br, indfold,\n		0.01,\n		optcontfold)\n\nCont.plotBranch(outfoldco;xlabel=\"b\",ylabel=\"a\")which produces(Image: )"
},

{
    "location": "#Using-GMRES-or-another-linear-solver-1",
    "page": "Home",
    "title": "Using GMRES or another linear solver",
    "category": "section",
    "text": "We continue the previous example but now using Matrix Free methods. The user can pass its own solver by implementing a version of LinearSolver. Some basic linear solvers have been implemented from KrylovKit.jl and IterativeSolvers.jl, we can use them here. Note that we can implement preconditioners with this. The same functionality is present for the eigensolver.# very easy to write since we have F_chan. Could use Automatic Differentiation as well\nfunction dF_chan(x, dx, α, β = 0.)\n	out = similar(x)\n	n = length(x)\n	out[1] = dx[1]\n	out[n] = dx[n]\n	for i=2:n-1\n		out[i] = (dx[i-1] - 2 * dx[i] + dx[i+1]) * (n-1)^2 + α * dsource_term(x[i], b = β) * dx[i]\n	end\n	return out\nend\n\nls = Cont.GMRES_KrylovKit{Float64}(dim = 100)\n	opt_newton_mf = Cont.NewtonPar(tol = 1e-11, verbose = true, linsolve = ls, eigsolve = Default_eig())\n	out_mf, hist, flag = @time Cont.newton(\n		x -> F_chan(x, a, 0.01),\n		x -> (dx -> dF_chan(x, dx, a, 0.01)),\n		sol,\n		opt_newton_mf)which gives:Newton Iterations\n   Iterations      Func-count      f(x)      Linear-Iterations\n\n        0                1     2.3440e+01         0\n        1                2     1.3774e+00        68\n        2                3     1.6267e-02        98\n        3                4     2.4336e-06        73\n        4                5     6.2617e-12        73\n  0.336398 seconds (1.15 M allocations: 54.539 MiB, 7.93% gc time)"
},

{
    "location": "#Example-2:-Snaking-with-2d-Swift-Hohenberg-equation-1",
    "page": "Home",
    "title": "Example 2: Snaking with 2d Swift-Hohenberg equation",
    "category": "section",
    "text": "We look at the following PDE 0=-(I+Delta)^2 u+lcdot u +nu u^2-u^3with periodic boundary conditions. This example is in the file example/SH2d-fronts.jl. It is extracted from pde2path. We use a Sparse Matrix to express the operator L_1=(I+Delta)^2using DiffEqOperators\nusing PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays\nconst Cont = PseudoArcLengthContinuation\n\nheatmapsol(x) = heatmap(reshape(x,Nx,Ny)\',color=:viridis)\n\nNx = 151\nNy = 100\nlx = 4*2pi\nly = 2*2pi/sqrt(3)\n\nfunction Laplacian2D(Nx,Ny,lx,ly,bc = :Neumann0)\n    hx = 2lx/Nx\n    hy = 2ly/Ny\n    D2x = sparse(DerivativeOperator{Float64}(2,2,hx,Nx,bc,bc))\n    D2y = sparse(DerivativeOperator{Float64}(2,2,hy,Ny,bc,bc))\n    A = kron(sparse(I,Ny,Ny),D2x) + kron(D2y,sparse(I,Nx,Nx))\n    return A\nend\n\nΔ = Laplacian2D(Nx,Ny,lx,ly)\nconst L1 = (I + Δ)^2We also write the functional and its differential which is a Sparse Matrixfunction F_sh(u, l=-0.15, ν=1.3)\n	return -L1 * u .+ (l .* u .+ ν .* u.^2 .- u.^3)\nend\n\nfunction dF_sh(u, l=-0.15, ν=1.3)\n	return -L1 + spdiagm(0 => l .+ 2ν .* u .- 3u.^2)\nendWe first look for hexagonal patterns. This done withX = -lx .+ 2lx/(Nx) * collect(0:Nx-1)\nY = -ly .+ 2ly/(Ny) * collect(0:Ny-1)\n\nsol0 = [(cos(x) + cos(x/2) * cos(sqrt(3) * y/2) ) for x in X, y in Y]\n		sol0 .= sol0 .- minimum(vec(sol0))\n		sol0 ./= maximum(vec(sol0))\n		sol0 = sol0 .- 0.25\n		sol0 .*= 1.7\n		heatmap(sol0\',color=:viridis)\n\nopt_new = Cont.NewtonPar(verbose = true, tol = 1e-9, maxIter = 100)\n	sol_hexa, hist, flag = @time Cont.newton(\n				x -> F_sh(x,-.1,1.3),\n				u -> dF_sh(u,-.1,1.3),\n				vec(sol0),\n				opt_new)\n	println(\"--> norm(sol) = \",norm(sol_hexa,Inf64))\n	heatmapsol(sol_hexa)which produces the resultsNewton Iterations \n   Iterations      Func-count      f(x)      Linear-Iterations\n        0                1     7.7310e+02         0\n        1                2     7.3084e+03         1\n        2                3     2.1595e+03         1\n        3                4     1.4173e+04         1\n        4                5     4.1951e+03         1\n        5                6     1.2394e+03         1\n        6                7     3.6414e+02         1\n        7                8     1.0659e+02         1\n        8                9     3.1291e+01         1\n        9               10     1.3202e+01         1\n       10               11     2.6793e+00         1\n       11               12     3.2728e-01         1\n       12               13     1.2491e-02         1\n       13               14     2.7447e-05         1\n       14               15     2.2626e-10         1\n  1.413734 seconds (45.47 k allocations: 749.631 MiB, 3.60% gc time)with(Image: )We can now continue this solutionopts_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.015,ds= -0.0051, pMax = 0.2, pMin = -1.0, save = false, theta = 0.1, plot_every_n_steps = 3, newtonOptions = opt_new)\n	opts_cont.detect_fold = true\n	opts_cont.maxSteps = 340\n\n	br, u1 = @time Cont.continuation(\n		(x,p) -> F_sh(x,p,1.3),\n		(x,p) -> dF_sh(x,p,1.3),\n		sol_hexa,0.099,opts_cont,plot = true,\n		plotsolution = (x;kwargs...)->(heatmap!(X, Y, reshape(x, Nx, Ny)\', color=:viridis, subplot=4, label=\"\")),\n		printsolution = x -> norm(x,Inf64))with result:(Image: )"
},

{
    "location": "#Snaking-computed-with-deflation-1",
    "page": "Home",
    "title": "Snaking computed with deflation",
    "category": "section",
    "text": "We know that there is snaking near the left fold. Let us look for other solutions like fronts. The problem is that if the guess is not precise enough, the newton iterations will converge to the solution with hexagons sol_hexa. We appeal to the technique initiated by P. Farrell and use a deflated problem. More precisely, we apply the newton iterations to the following functional utoleft(frac1u-sol_hexa^2+sigmaright)F_sh(u) which penalizes sol_hexa.deflationOp = DeflationOperator(2.0,(x,y) -> dot(x,y),1.0,[sol_hexa])\nopt_new.maxIter = 250\noutdef, _,flag,_ = @time Cont.newtonDeflated(\n				x -> F_sh(x,-.1,1.3),\n				u -> dF_sh(u,-.1,1.3),\n				0.2vec(sol_hexa) .* vec([exp.(-(x+lx)^2/25) for x in X, y in Y]),\n				opt_new,deflationOp, normN = x -> norm(x,Inf64))\n		heatmapsol(outdef) |> display\n		flag && push!(deflationOp, outdef)which gives:(Image: )Note that push!(deflationOp, outdef) deflates the newly found solution so that by repeating the process we find another one:outdef, _,flag,_ = @time Cont.newtonDeflated(\n				x -> F_sh(x,-.1,1.3),\n				u -> dF_sh(u,-.1,1.3),\n				0.2vec(sol_hexa) .* vec([exp.(-(x)^2/25) for x in X, y in Y]),\n				opt_new,deflationOp, normN = x -> norm(x,Inf64))\n		heatmapsol(outdef) |> display\n		flag && push!(deflationOp, outdef)(Image: )Again, repeating this from random guesses, we find several more solutions, like for example(Image: )(Image: )We can now continue the solutions located in deflationOp.rootsopts_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.005,ds= -0.0015, pMax = -0.01, pMin = -1.0, theta = 0.5, plot_every_n_steps = 3, newtonOptions = opt_new, a = 0.5, detect_fold = true)\n	opts_cont.newtonOptions.tol = 1e-9\n	opts_cont.newtonOptions.maxIter = 50\n	opts_cont.maxSteps = 450\n\n	br, u1 = @time Cont.continuation(\n		(x,p) -> F_sh(x,p,1.3), (x,p) -> dF_sh(x,p,1.3),\n		deflationOp.roots[5],\n		-0.1,\n		opts_cont,plot = true,\n		plotsolution = (x;kwargs...)->(heatmap!(X,Y,reshape(x,Nx,Ny)\',color=:viridis,subplot=4,label=\"\")),\n		printsolution = x->norm(x))and get using Cont.plotBranch(br)(Image: )"
},

{
    "location": "#Example-3:-Brusselator-in-1d-1",
    "page": "Home",
    "title": "Example 3: Brusselator in 1d",
    "category": "section",
    "text": "We look at the Brusselator in 1d. The equations are as followsbeginaligned frac  partial X   partial t   = frac  D _  1    l ^  2   frac  partial ^  2  X   partial z ^  2   + X ^  2  Y - ( β + 1 ) X + α  frac  partial Y   partial t   = frac  D _  2    l ^  2   frac  partial ^  2  Y   partial z ^  2   + β X - X ^  2  Y endalignedwith Dirichlet boundary conditionsbeginarray  l   X ( t  z = 0 ) = X ( t  z = 1 ) = α    Y ( t  z = 0 ) = Y ( t  z = 1 ) = β  α  endarrayThese equations have been derived to reproduce an oscillating chemical reaction. There is an obvious equilibrium (α β  α). Here, we consider bifurcation with respect to the parameter l.We start by writing the functionalusing PseudoArcLengthContinuation, LinearAlgebra, Plots, SparseArrays\nconst Cont = PseudoArcLengthContinuation\n\nf1(u, v) = u^2*v\n\nfunction F_bru(x, α, β; D1 = 0.008, D2 = 0.004, l = 1.0)\n	n = div(length(x), 2)\n	h = 1.0 / (n+1); h2 = h*h\n\n	u = @view x[1:n]\n	v = @view x[n+1:2n]\n\n	# output\n	f = similar(x)\n\n	f[1] = u[1] - α\n	f[n] = u[n] - α\n	for i=2:n-1\n		f[i] = D1/l^2 * (u[i-1] - 2u[i] + u[i+1]) / h2 - (β + 1) * u[i] + α + f1(u[i], v[i])\n	end\n\n\n	f[n+1] = v[1] - β / α\n	f[end] = v[n] - β / α;\n	for i=2:n-1\n		f[n+i] = D2/l^2 * (v[i-1] - 2v[i] + v[i+1]) / h2 + β * u[i] - f1(u[i], v[i])\n	end\n\n	return f\nendFor computing periodic orbits, we will need a Sparse representation of the Jacobian:function Jac_sp(x, α, β; D1 = 0.008, D2 = 0.004, l = 1.0)\n	# compute the Jacobian using a sparse representation\n	n = div(length(x), 2)\n	h = 1.0 / (n+1); hh = h*h\n\n	diag  = zeros(2n)\n	diagp1 = zeros(2n-1)\n	diagm1 = zeros(2n-1)\n\n	diagpn = zeros(n)\n	diagmn = zeros(n)\n\n	diag[1] = 1.0\n	diag[n] = 1.0\n	diag[n + 1] = 1.0\n	diag[end] = 1.0\n\n	for i=2:n-1\n		diagm1[i-1] = D1 / hh/l^2\n		diag[i]   = -2D1 / hh/l^2 - (β + 1) + 2x[i] * x[i+n]\n		diagp1[i] = D1 / hh/l^2\n		diagpn[i] = x[i]^2\n	end\n\n	for i=n+2:2n-1\n		diagmn[i-n] = β - 2x[i-n] * x[i]\n		diagm1[i-1] = D2 / hh/l^2\n		diag[i]   = -2D2 / hh/l^2 - x[i-n]^2\n		diagp1[i] = D2 / hh/l^2\n	end\n	return spdiagm(0 => diag, 1 => diagp1, -1 => diagm1, n => diagpn, -n => diagmn)\nendFinally, to monitor if the solution is constant in space, we will use the following callbackfunction finalise_solution(z, tau, step, contResult)\n	n = div(length(z), 2)\n	printstyled(color=:red, \"--> Solution constant = \", norm(diff(z[1:n])), \" - \", norm(diff(z[n+1:2n])), \"\\n\")\n	return true\nendWe can now compute to equilibrium and its stabilityn = 301\n\na = 2.\nb = 5.45\n\nsol0 = vcat(a * ones(n), b/a * ones(n))\n\nopt_newton = Cont.NewtonPar(tol = 1e-11, verbose = true, eigsolve = eig_KrylovKit(tol=1e-6, dim = 60))\n	out, hist, flag = @time Cont.newton(\n		x -> F_bru(x, a, b),\n		x -> Jac_sp(x, a, b),\n		sol0,\n		opt_newton)\n		\nopts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.0061, ds= 0.0051, pMax = 1.8, save = false, theta = 0.01, detect_fold = true, detect_bifurcation = true, nev = 41, plot_every_n_steps = 50, newtonOptions = opt_newton)\n	opts_br0.newtonOptions.maxIter = 20\n	opts_br0.newtonOptions.tol = 1e-8\n	opts_br0.maxSteps = 280\n\n	br, u1 = @time Cont.continuation(\n		(x, p) ->   F_bru(x, a, b, l = p),\n		(x, p) -> Jac_sp(x, a, b, l = p),\n		out,\n		0.3,\n		opts_br0,\n		plot = true,\n		plotsolution = (x;kwargs...)->(N = div(length(x), 2);plot!(x[1:N], subplot=4, label=\"\");plot!(x[N+1:2N], subplot=4, label=\"\")),\n		finaliseSolution = finalise_solution,\n		printsolution = x -> norm(x, Inf64))		We obtain the following bifurcation diagram with 3 Hopf bifurcation points(Image: )"
},

{
    "location": "#Continuation-of-Hopf-points-1",
    "page": "Home",
    "title": "Continuation of Hopf points",
    "category": "section",
    "text": "We use the bifurcation points guesses located in br.bifpoint to turn them into precise bifurcation points. For the first one, we haveind_hopf = 1\nhopfpt = Cont.HopfPoint(br, ind_hopf)\n\nouthopf, hist, flag = @time Cont.newtonHopf((x, p) ->  F_bru(x, a, b, l = p),\n				(x, p) -> Jac_sp(x, a, b, l = p),\n				br, ind_hopf,\n				opt_newton)\nflag && printstyled(color=:red, \"--> We found a Hopf Point at l = \", outhopf[end-1], \", ω = \", outhopf[end], \", from l = \",hopfpt[end-1],\"\\n\")which produces--> We found a Hopf Point at l = 0.5164377051987692, ω = 2.13950928953342, from l = 0.5197012664156633We can also perform a Hopf continuation with respect to parameters l, βbr_hopf, u1_hopf = @time Cont.continuationHopf(\n	(x, p, β) ->  F_bru(x, a, β, l = p),\n	(x, p, β) -> Jac_sp(x, a, β, l = p),\n	br, ind_hopf,\n	b,\n	ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.01, pMax = 6.5, pMin = 0.0, a = 2., theta = 0.4, newtonOptions = opt_newton))which gives using Cont.plotBranch(br_hopf, xlabel=\"beta\", ylabel = \"l\")(Image: )"
},

{
    "location": "#Continuation-of-periodic-orbits-1",
    "page": "Home",
    "title": "Continuation of periodic orbits",
    "category": "section",
    "text": "Finally, we can perform continuation of periodic orbits branching from the Hopf bifurcation points. Note that we did not compute the Hopf normal form, so we need an educated guess for the periodic orbit. We first create the initial guess for the periodic orbit:function plotPeriodic(outpof,n,M)\n	outpo = reshape(outpof[1:end-1], 2n, M)\n	plot(heatmap(outpo[1:n,:], xlabel=\"Time\"), heatmap(outpo[n+2:end,:]))\nend\n\n# index of the Hopf point we want to branch from\nind_hopf = 2\nhopfpt = Cont.HopfPoint(br, ind_hopf)\n\n# bifurcation parameter\nl_hopf = hopfpt[end-1]\n\n# Hopf frequency\nωH     = hopfpt[end] |> abs\n\n# number of time slices for the periodic orbit\nM = 100\n\norbitguess = zeros(2n, M)\nphase = []; scalphase = []\nvec_hopf = getEigenVector(opt_newton.eigsolve ,br.eig[br.bifpoint[ind_hopf][2]][2] ,br.bifpoint[ind_hopf][end]-1)\nfor ii=1:M\n	t = (ii-1)/(M-1)\n	orbitguess[:, ii] .= real.(hopfpt[1:2n] +\n		26*0.1 * vec_hopf * exp(2pi * complex(0, 1) * (t - 0.235)))\n	push!(phase, t);push!(scalphase, dot(orbitguess[:, ii]- hopfpt[1:2n], real.(vec_hopf)))\nendWe want to make two remarks. The first is that an initial guess is composed of a space time solution and of the guess for the period of the solution:orbitguess_f = vcat(vec(orbitguess), 2pi/ωH) |> vecThe second remark concerns the phase 0.235 written above. To account for the additional parameter, periodic orbit localisation using Shooting methods or Finite Differences methods add an additional constraint. In the present case, this constraint is u(0) - u_hopf phi = 0where u_{hopf} = hopfpt[1:2n] and phi is real.(vec_hopf). This is akin to a Poincare section.This constraint fixes the phase of the periodic orbit. By plotting plot(phase, scalphase), one can find the phase 0.235. We can now use Newton iterations to find a periodic orbit.We first create a functional which holds the problempoTrap = l-> PeriodicOrbitTrap(\n			x-> F_bru(x, a, b, l = l),\n			x-> Jac_sp(x, a, b, l = l),\n			real.(vec_hopf),\n			hopfpt[1:2n],\n			M,\n			opt_newton.linsolve)The functional is x -> poTrap(l_hopf + 0.01)(x) at parameter l_hopf + 0.01. For this problem, it is more efficient to use a Sparse Matrix representation of the jacobian rather than a Matrix Free one (with GMRES). The matrix at (x,p) is computed like thispoTrap(p)(x, :jacsparse)while the Matrix Free version isdx -> poTrap(p)(x, dx)We use Newton solve:opt_po = Cont.NewtonPar(tol = 1e-8, verbose = true, maxIter = 50)\n	outpo_f, hist, flag = @time Cont.newton(\n			x ->  poTrap(l_hopf + 0.01)(x),\n			x ->  poTrap(l_hopf + 0.01)(x, :jacsparse),\n			orbitguess_f,\n			opt_po)\n	println(\"--> T = \", outpo_f[end], \", amplitude = \", maximum(outpo_f[1:n,:])-minimum(outpo_f[1:n,:]))\n	plotPeriodic(outpo_f,n,M)and obtainNewton Iterations \n   Iterations      Func-count      f(x)      Linear-Iterations\n\n        0                1     9.7352e-02         0\n        1                2     2.2367e-02         1\n        2                3     5.1125e-04         1\n        3                4     6.4370e-06         1\n        4                5     5.8870e-10         1\n 25.460922 seconds (5.09 M allocations: 22.759 GiB, 31.76% gc time)\n--> T = 2.978950450406386, amplitude = 0.35069253154451707and(Image: )Finally, we can perform continuation of this periodic orbitopts_po_cont = ContinuationPar(dsmin = 0.0001, dsmax = 0.05, ds= 0.001, pMax = 4.5, maxSteps = 400, secant = true, theta=0.1, plot_every_n_steps = 3, newtonOptions = NewtonPar(verbose = true))\n	br_pok1, _ , _= @time Cont.continuation(\n		(x, p) ->  poTrap(p)(x),\n		(x, p) ->  poTrap(p)(x, :jacsparse),\n		outpo_f, l_hopf + 0.01,\n		opts_po_cont,\n		plot = true,\n		plotsolution = (x;kwargs...)->heatmap!(reshape(x[1:end-1], 2*n, M)\', subplot=4, ylabel=\"time\"),\n		printsolution = u -> u[end])to obtain the period of the orbit as function of l(Image: )It is likely that the kink in the branch is caused by a spurious branch switching. This can be probably resolved using larger dsmax.A more complete diagram is the following where we computed the 3 branches of periodic orbits off the Hopf points.(Image: )"
},

{
    "location": "#Example-4:-nonlinear-pendulum-with-ApproxFun-1",
    "page": "Home",
    "title": "Example 4: nonlinear pendulum with ApproxFun",
    "category": "section",
    "text": "We reconsider the first example using the package ApproxFun.jl which allows very precise function approximation. We start with some imports:using ApproxFun, LinearAlgebra\n\nusing PseudoArcLengthContinuation, Plots\nconst Cont = PseudoArcLengthContinuationusing PseudoArcLengthContinuation, Plots\nconst Cont = PseudoArcLengthContinuationWe then need to overwrite some functions of ApproxFun:# specific methods for ApproxFun\nimport Base: length, eltype, copyto!\nimport LinearAlgebra: norm, dot, axpy!, rmul!\n\neltype(x::ApproxFun.Fun) = eltype(x.coefficients)\nlength(x::ApproxFun.Fun) = length(x.coefficients)\n\nnorm(x::ApproxFun.Fun, p::Real) = (@show p;norm(x.coefficients, p))\nnorm(x::Array{Fun, 1}, p::Real)  = (@show p;norm(x[3].coefficients, p))\nnorm(x::Array{Fun{Chebyshev{Segment{Float64}, Float64}, Float64, Array{Float64, 1}}, 1}, p::Real) = (@show p;norm(x[3].coefficients, p))\n\ndot(x::ApproxFun.Fun, y::ApproxFun.Fun) = sum(x * y)\ndot(x::Array{Fun{Chebyshev{Segment{Float64}, Float64}, Float64, Array{Float64, 1}}, 1}, y::Array{Fun{Chebyshev{Segment{Float64}, Float64}, Float64, Array{Float64, 1}}, 1}) = sum(x[3] * y[3])\n\naxpy!(a::Float64, x::ApproxFun.Fun, y::ApproxFun.Fun) = (y .= a .* x .+ y)\nrmul!(y::ApproxFun.Fun, b::Float64) = (y .= b .* y)\n\ncopyto!(x::ApproxFun.Fun, y::ApproxFun.Fun) = (x.coefficients = copy(y.coefficients))We can easily write our functional with boundary conditions in a convenient manner using ApproxFun:source_term(x; a = 0.5, b = 0.01) = 1 + (x + a*x^2)/(1 + b*x^2)\ndsource_term(x; a = 0.5, b = 0.01) = (1-b*x^2+2*a*x)/(1+b*x^2)^2\n\nfunction F_chan(u, alpha::Float64, beta = 0.01)\n	return [Fun(u(0.), domain(sol)) - beta,\n		Fun(u(1.), domain(sol)) - beta,\n		Δ * u + alpha * source_term(u, b = beta)]\nend\n\nfunction Jac_chan(u, alpha, beta = 0.01)\n	return [Evaluation(u.space, 0.),\n		Evaluation(u.space, 1.),\n		Δ + alpha * dsource_term(u, b = beta)]\nendWe want to call a Newton solver. We first need an initial guess and the Laplacian operator:sol = Fun(x -> x * (1-x), Interval(0.0, 1.0))\nconst Δ = Derivative(sol.space, 2)Finally, we need to define some parameters for the Newton iterations. This is done by callingopt_newton = Cont.NewtonPar(tol = 1e-12, verbose = true)We call the Newton solver:opt_new = Cont.NewtonPar(tol = 1e-12, verbose = true)\n	out, hist, flag = @time Cont.newton(\n				x -> F_chan(x, 3.0, 0.01),\n				u -> Jac_chan(u, 3.0, 0.01),\n				sol, opt_new, normN = x -> norm(x, Inf64))and you should seeNewton Iterations \n   Iterations      Func-count      f(x)      Linear-Iterations\n\n        0                1     1.5707e+00         0\n        1                2     1.1546e-01         1\n        2                3     8.0149e-04         1\n        3                4     3.9038e-08         1\n        4                5     4.6975e-13         1\n  0.079482 seconds (344.44 k allocations: 13.856 MiB)We can now perform numerical continuation wrt the parameter a. Again, we need to define some parameters for the continuation:opts_br0 = ContinuationPar(dsmin = 0.001, dsmax = 0.05, ds= 0.005, a = 0.1, pMax = 4.1, theta = 0.91, secant = true, plot_every_n_steps = 3, newtonOptions = NewtonPar(tol = 1e-8, maxIter = 50, verbose = true), doArcLengthScaling = false)\n	opts_br0.newtonOptions.linesearch  = false\n	opts_br0.detect_fold = true\n	opts_br0.maxSteps = 143We also provide a function to check how the ApproxFun solution vector grows:function finalise_solution(z, tau, step, contResult)\n	printstyled(color=:red,\"--> AF length = \", (z, tau) .|> length ,\"\\n\")\n	# chop!(z, 1e-14);chop!(tau, 1e-14)\n	true\nendThen, we can call the continuation routinebr, u1 = @time Cont.continuation(\n		(x, p) -> F_chan(x, p, 0.01),\n		(x, p) -> Jac_chan(x, p, 0.01),\n		out, 3.0, opts_br0,\n		plot = true,\n		finaliseSolution = finalise_solution,\n		plotsolution = (x; kwargs...) -> plot!(x, subplot = 4, label = \"l = $(length(x))\"),\n		normC = x -> norm(x, Inf64))and you should see (Image: )"
},

{
    "location": "#Example-5:-the-Swift-Hohenberg-equation-on-the-GPU-1",
    "page": "Home",
    "title": "Example 5: the Swift-Hohenberg equation on the GPU",
    "category": "section",
    "text": "Here we give an example where the continuation can be done entirely on the GPU. We choose the 2d Swift-Hohenberg as an example and consider a larger grid. Solving the sparse linear problem in v-(I+Delta)^2 v+lcdot v +2nu uv-3u^2v = rhswith a direct solver becomes prohibitive. Looking for an iterative method, the conditioning of the jacobian is not good enough to have fast convergence, mainly because of the Laplacian. However, this problem reads-v + L * (d * v) = L*rhswhere L = ((I+Delta)^2 + I)^-1is very well conditioned and d = l+1+2nu v-3v^2Hence, to solve the previous equation, only a few GMRES iterations are required. "
},

{
    "location": "#Computing-the-inverse-of-the-differential-operator-1",
    "page": "Home",
    "title": "Computing the inverse of the differential operator",
    "category": "section",
    "text": "The issue now is to compute L but this is easy using Fourier transforms.Hence, that\'s why we slightly modify the above Example 2. by considering periodic boundary conditions. Let us now show how to compute L. Although the code looks quite technical, it is based on two facts. First, the Fourier transform symbol associated to L isl_1 = 1+(1-k_x^2-k_y^2)^2which is pre-computed in the structure SHLinearOp. Then the effect of L on u is as simple as real.(ifft( l1 .* fft(u) )) and the inverse L\\u is real.(ifft( fft(u) ./ l1 )). However, in order to save memory on the GPU, we use inplace FFTs to reduce temporaries which explains the following code.using AbstractFFTs, FFTW, KrylovKit\n\n# Making the linear operator a subtype of Cont.LinearSolver is handy as we will use it \n# in the Newton iterations.\nstruct SHLinearOp <: Cont.LinearSolver\n	tmp_real         # temporary\n	tmp_complex      # temporary\n	l1\n	fftplan\n	ifftplan\nend\n\nfunction SHLinearOp(Nx, lx, Ny, ly; AF = Array{TY})\n	# AF is a type, it could be CuArray{TY} to run the following on GPU\n	k1 = vcat(collect(0:Nx/2), collect(Nx/2+1:Nx-1) .- Nx)\n	k2 = vcat(collect(0:Ny/2), collect(Ny/2+1:Ny-1) .- Ny)\n	d2 = [(1-(pi/lx * kx)^2 - (pi/ly * ky)^2)^2 + 1. for kx in k1, ky in k2]\n	tmpc = Complex.(AF(zeros(Nx,Ny)))\n	return SHLinearOp(AF(zeros(Nx,Ny)),tmpc,AF(d2),plan_fft!(tmpc),plan_ifft!(tmpc))\nend\n\nimport Base: *, \\\n\nfunction *(c::SHLinearOp, u)\n	c.tmp_complex .= Complex.(u)\n	c.fftplan * c.tmp_complex\n	c.tmp_complex .= c.l1 .* c.tmp_complex\n	c.ifftplan * c.tmp_complex\n	c.tmp_real .= real.(c.tmp_complex)\n	return copy(c.tmp_real)\nend\n\nfunction \\(c::SHLinearOp, u)\n	c.tmp_complex .= Complex.(u)\n	c.fftplan * c.tmp_complex\n	c.tmp_complex .=  c.tmp_complex ./ c.l1\n	c.ifftplan * c.tmp_complex\n	c.tmp_real .= real.(c.tmp_complex)\n	return copy(c.tmp_real)\nendNow that we have our operator L, we can give our functional:function F_shfft(u, l = -0.15, ν = 1.3; shlop::SHLinearOp)\n	return -(shlop * u) .+ ((l+1) .* u .+ ν .* u.^2 .- u.^3)\nend"
},

{
    "location": "#Functions-for-LinearAlgebra-on-the-GPU-1",
    "page": "Home",
    "title": "Functions for LinearAlgebra on the GPU",
    "category": "section",
    "text": "We plan to use KrylovKit on the GPU. For this to work, we need to overload some functions for CuArray.jl. Note that this will be removed in the future when CuArrays improves.using CuArrays\nCuArrays.allowscalar(false)\nimport LinearAlgebra: mul!, axpby!\nmul!(x::CuArray, y::CuArray, α::T) where {T <: Number} = (x .= α .* y)\nmul!(x::CuArray, y::T, α::CuArray) where {T <: Number} = (x .= α .* y)\naxpby!(a::T, X::CuArray, b::T, Y::CuArray) where {T <: Number} = (Y .= a .* X .+ b .* Y)\n\nAF = CuArray{TY}\nTY = Float64We can now define our operator L and an initial guess sol0.using PseudoArcLengthContinuation, LinearAlgebra, Plots\nconst Cont = PseudoArcLengthContinuation\n\n# to simplify plotting of the solution\nheatmapsol(x) = heatmap(reshape(Array(x)), Nx, Ny)\', color=:viridis)\n\nNx = 2^10\nNy = 2^10\nlx = 8pi * 2\nly = 2*2pi/sqrt(3) * 2\n\nX = -lx .+ 2lx/(Nx) * collect(0:Nx-1)\nY = -ly .+ 2ly/(Ny) * collect(0:Ny-1)\n\nsol0 = [(cos(x) .+ cos(x/2) * cos(sqrt(3) * y/2) ) for x in X, y in Y]\n		sol0 .= sol0 .- minimum(vec(sol0))\n		sol0 ./= maximum(vec(sol0))\n		sol0 = sol0 .- 0.25\n		sol0 .*= 1.7\n		heatmap(sol0, color=:viridis)\n		\nL = SHLinearOp(Nx, lx, Ny, ly, AF = AF)	Before applying a Newton solver, we need to show how to solve the linear equation arising in the Newton Algorithm.function (sh::SHLinearOp)(J, rhs)\n	u, l, ν = J\n	udiag = l .+ 1 .+ 2ν .* u .- 3 .* u.^2\n	res, info = res, info = KrylovKit.linsolve( u -> -u .+ sh \\ (udiag .* u), sh \\ rhs, tol = 1e-9, maxiter = 6) \n	return res, true, info.numops\nend"
},

{
    "location": "#Newton-iterations-and-deflation-1",
    "page": "Home",
    "title": "Newton iterations and deflation",
    "category": "section",
    "text": "We are now ready to perform Newton iterations:pt_new = Cont.NewtonPar(verbose = true, tol = 1e-6, maxIter = 100, linsolve = L)\n	sol_hexa, hist, flag = @time Cont.newton(\n				x -> F_shfft(x, -.1, 1.3, shlop = L),\n				u -> (u, -0.1, 1.3),\n				AF(sol0),\n				opt_new, normN = x->maximum(abs.(x)))\n	println(\"--> norm(sol) = \", maximum(abs.(sol_hexa)))\n	heatmapsol(sol_hexa)You should see this: Newton Iterations \n   Iterations      Func-count      f(x)      Linear-Iterations\n\n        0                1     2.7383e-01         0\n        1                2     1.2891e+02        14\n        2                3     3.8139e+01        70\n        3                4     1.0740e+01        37\n        4                5     2.8787e+00        22\n        5                6     7.7522e-01        17\n        6                7     1.9542e-01        13\n        7                8     3.0292e-02        13\n        8                9     1.1594e-03        12\n        9               10     1.8842e-06        11\n       10               11     4.2642e-08        10\n  2.261527 seconds (555.45 k allocations: 44.849 MiB, 1.61% gc time)\n--> norm(sol) = 1.26017611779702Note that this is about the same computation time as in Example 2 but for a problem almost 100x larger!The solution is:(Image: )We can also use the deflation technique on the GPU as followsdeflationOp = DeflationOperator(2.0, (x, y)->dot(x, y), 1.0, [sol_hexa])\n\nopt_new.maxIter = 250\noutdef, _, flag, _ = @time Cont.newtonDeflated(\n				x -> F_shfft(x, -.1, 1.3, shlop = L),\n				u -> (u, -0.1, 1.3),\n				0.4 .* sol_hexa .* AF([exp(-1(x+0lx)^2/25) for x in X, y in Y]),\n				opt_new, deflationOp, normN = x->maximum(abs.(x)))\n		println(\"--> norm(sol) = \", norm(outdef))\n		heatmapsol(outdef) |> display\n		flag && push!(deflationOp, outdef)and get:(Image: )"
},

{
    "location": "#Computation-of-the-branches-1",
    "page": "Home",
    "title": "Computation of the branches",
    "category": "section",
    "text": "Finally, we can perform continuation of the branches on the GPU:opts_cont = ContinuationPar(dsmin = 0.001, dsmax = 0.005, ds= -0.0015, pMax = -0.0, pMin = -1.0, theta = 0.5, plot_every_n_steps = 5, newtonOptions = opt_new, a = 0.5, detect_fold = true, detect_bifurcation = false)\n	opts_cont.newtonOptions.tol = 1e-6\n	opts_cont.newtonOptions.maxIter = 50\n	opts_cont.maxSteps = 80\n\n	br, u1 = @time Cont.continuation(\n		(u, p) -> F_shfft(u, p, 1.3, shlop = L),\n		(u, p) -> (u, p, 1.3),\n		deflationOp.roots[1],\n		-0.1,\n		opts_cont, plot = true,\n		plotsolution = (x;kwargs...)->heatmap!(reshape(Array(x), Nx, Ny)\', color=:viridis, subplot=4),\n		printsolution = x->maximum(abs.(x)), normC = x->maximum(abs.(x)))(Image: )"
},

{
    "location": "linearsolver/#",
    "page": "Linear Solvers",
    "title": "Linear Solvers",
    "category": "page",
    "text": ""
},

{
    "location": "linearsolver/#Linear-solvers-1",
    "page": "Linear Solvers",
    "title": "Linear solvers",
    "category": "section",
    "text": "The linear solvers are subtypes of LinearSolver. Basically, one must provide a way of inverting the Jacobian J or computing J \\ x.Here is an example of the simplest of them (see src/LinearSolver.jl) to give you an idea, the backslash operator:struct Default <: LinearSolver end\n\n# solves the equation J * out = x\nfunction (l::Default)(J, x)\n    return J \\ x, true, 1\nendNote that for newton to work, you must return 3 arguments. The first one is the result, the second one is whether the computation was successful and the third is the number of iterations required to perform the computation.You can call it like (and it will be called like this in newton)ls = Default()\nls(rand(2,2), rand(2))You can instead define struct myLinearSolver <: LinearSolver end and write (l::myLinearSolver)(J, x) where this function would implement GMRES or whatever you prefer."
},

{
    "location": "linearsolver/#Eigen-solvers-1",
    "page": "Linear Solvers",
    "title": "Eigen solvers",
    "category": "section",
    "text": "The eigen solvers are subtypes of EigenSolver. Basically, one must provide a way of computing the eigen elements of the Jacobian J.Here is an example of the simplest of them (see src/EigSolver.jl) to give you an idea:struct Default_eig <: EigenSolver end\n\nfunction (l::Default_eig)(J, nev::Int64)\n	# I put Array so we can call it on small sparse matrices\n    F = eigen(Array(J))\n    I = sortperm(F.values, by = x-> real(x), rev = true)\n    return F.values[I[1:nev]], F.vectors[:, I[1:nev]]\nendwarning: Eigenvalues\nThe eigenvalues must be ordered by increasing real part for the detection of bifurcations to work properly.note: Eigenvectors\nThe eigenvectors must be a 2d array for the simplified calls newtonHopf and newtonFold to work properly."
},

{
    "location": "detectionBifurcation/#",
    "page": "Bifurcations",
    "title": "Bifurcations",
    "category": "page",
    "text": ""
},

{
    "location": "detectionBifurcation/#Detection-of-bifurcation-points-1",
    "page": "Bifurcations",
    "title": "Detection of bifurcation points",
    "category": "section",
    "text": "Depending on the bifurcation type, detection is ensured during a call to br, _ = continuation(...) by turning on a flag."
},

{
    "location": "detectionBifurcation/#Eigensolver-1",
    "page": "Bifurcations",
    "title": "Eigensolver",
    "category": "section",
    "text": "The user must provide an eigensolver by setting NewtonOptions.eigsolve where NewtonOptions is located in the parameter ::ContinuationPar passed to continuation. See src/Newton.jl for more information on the structure of the options passed to newton and continuation.The eigensolver is highly problem dependent and this is why the user should implement / parametrize its own eigensolver through the abstract type EigenSolver or select one among the provided like Default_eig(), eig_IterativeSolvers(), eig_KrylovKit. See src/EigSolver.jl."
},

{
    "location": "detectionBifurcation/#Fold-bifurcation-1",
    "page": "Bifurcations",
    "title": "Fold bifurcation",
    "category": "section",
    "text": "The detection of Fold point is done by monitoring  the monotonicity of the parameter.The detection is triggered by setting detect_fold = true in the parameter ::ContinuationPar passed to continuation. When a Fold is detected, a flag is added to br.bifpoint allowing for later refinement."
},

{
    "location": "detectionBifurcation/#Generic-bifurcation-1",
    "page": "Bifurcations",
    "title": "Generic bifurcation",
    "category": "section",
    "text": "By this we mean a change in the dimension of the Jacobian kernel. The detection of Branch point is done by analysis of the spectrum of the Jacobian.The detection is triggered by setting detect_bifurcation = true in the parameter ::ContinuationPar passed to continuation. The user must also provide a hint of the number of eigenvalues to be computed nev = 10 in the parameter ::ContinuationPar passed to continuation. Note that nev is incremented whenever a bifurcation point is detected. When a Branch point is detected, a flag is added to br.bifpoint allowing for later refinement."
},

{
    "location": "detectionBifurcation/#Hopf-bifurcation-1",
    "page": "Bifurcations",
    "title": "Hopf bifurcation",
    "category": "section",
    "text": "The detection of Branch point is done by analysis of the spectrum of the Jacobian.The detection is triggered by setting detect_bifurcation = true in the parameter ::ContinuationPar passed to continuation. The user must also provide a hint of the number of eigenvalues to be computed nev = 10 in the parameter ::ContinuationPar passed to continuation. Note that nev is incremented whenever a bifurcation point is detected. When a Hopf point is detected, a flag is added to br.bifpoint allowing for later refinement."
},

{
    "location": "codim2Continuation/#",
    "page": "Fold / Hopf Continuation",
    "title": "Fold / Hopf Continuation",
    "category": "page",
    "text": ""
},

{
    "location": "codim2Continuation/#Fold-/-Hopf-Continuation-1",
    "page": "Fold / Hopf Continuation",
    "title": "Fold / Hopf Continuation",
    "category": "section",
    "text": "For this to work, it is important to have an analytical expression for the jacobian. See the example Chan for more details."
},

{
    "location": "codim2Continuation/#The-case-of-the-Fold-point-1",
    "page": "Fold / Hopf Continuation",
    "title": "The case of the Fold point",
    "category": "section",
    "text": "Once a Fold point have been detected after a call to br, _ = continuation(...), it can be refined with the use of newton iterations. We have implemented a Minimally Augmented formulation. A simplified interface is provided."
},

{
    "location": "codim2Continuation/#Newton-refinement-1",
    "page": "Fold / Hopf Continuation",
    "title": "Newton refinement",
    "category": "section",
    "text": "Let us say that ind_fold is the index in br.bifpoint of a Fold point. This guess can be refined by calling the simplified interface. More precisions are provided below for an advanced usage.outfold, hist, flag = @time Cont.newtonFold((x,p) -> F(x, p),\n							(x, p) -> Jac(x, p),\n							br, ind_fold,\n							opt_newton)It is important to note that for improved performance, a function implementing the expression of the hessian should be provided. This is by far the fastest for the computations. Reader interested in this advanced usage should look at the example example/chan.jl. Although it is a simple problem, many different use case are shown in a simple setting."
},

{
    "location": "codim2Continuation/#The-case-of-the-Hopf-point-1",
    "page": "Fold / Hopf Continuation",
    "title": "The case of the Hopf point",
    "category": "section",
    "text": "One a Hopf point have been detected after a call to br, _ = continuation(...), it can be refined with the use of newton iterations. We have implemented a Minimally Augmented formulation. A simplified interface is provided as for the Fold case."
},

{
    "location": "codim2Continuation/#Newton-refinement-2",
    "page": "Fold / Hopf Continuation",
    "title": "Newton refinement",
    "category": "section",
    "text": "Let us say that ind_hopf is the index in br.bifpoint of a Hopf point. This guess can be refined by calling the simplified interface. More precisions are provided below for an advanced usage.outfold, hist, flag = @time Cont.newtonHopf((x,p) -> F(x, p),\n							(x, p) -> Jac(x, p),\n							br, ind_hopf,\n							opt_newton)"
},

{
    "location": "codim2Continuation/#PseudoArcLengthContinuation.newtonFold",
    "page": "Fold / Hopf Continuation",
    "title": "PseudoArcLengthContinuation.newtonFold",
    "category": "function",
    "text": "This function turns an initial guess for a Fold point into a solution to the Fold problem based on a Minimally Augmented formulation. The arguments are as follows\n\nF   = (x, p) -> F(x, p) where p is the parameter associated to the Fold point\ndF  = (x, p) -> d_xF(x, p) associated jacobian\ndFt = (x, p) -> transpose(d_xF(x, p)) associated jacobian, it should be implemented in an efficient manner. For matrix-free methods, tranpose is not readily available.\nd2F = (x, p, v1, v2) ->  d2F(x, p, v1, v2) a bilinear operator representing the hessian of F. It has to provide an expression for d2F(x,p)[v1,v2].\nfoldpointguess initial guess (x0, p0) for the Fold point. It should be a AbstractArray or a BorderedVector\neigenvec guess for the 0 eigenvector\noptions::NewtonPar\n\n\n\n\n\ncall when hessian is unknown, finite differences are then used\n\n\n\n\n\nSimplified call to refine an initial guess for a Fold point. More precisely, the call is as follows\n\n`newtonFold(F, J, Jt, br::ContResult, index::Int64, options)`\n\nor\n\n`newtonFold(F, J, Jt, d2F, br::ContResult, index::Int64, options)`\n\nwhen the Hessian d2F is known. The parameters are as usual except that you have to pass the branch br from the result of a call to continuation with detection of bifurcations enabled and index is the index of bifurcation point in br you want to refine.\n\n\n\n\n\n"
},

{
    "location": "codim2Continuation/#PseudoArcLengthContinuation.newtonHopf",
    "page": "Fold / Hopf Continuation",
    "title": "PseudoArcLengthContinuation.newtonHopf",
    "category": "function",
    "text": "This function turns an initial guess for a Hopf point into a solution to the Hopf problem based on a Minimally Augmented formulation. The arguments are as follows\n\nF  = (x, p) -> F(x, p) where p is the parameter associated to the Hopf point\nJ  = (x, p) -> d_xF(x, p) associated jacobian\nJt = (x, p) -> transpose(d_xF(x, p)) associated jacobian\nhopfpointguess initial guess (x0, p0) for the Hopf point. It should be a AbstractVector or a BorderedVector.\neigenvec guess for the  iω eigenvector\neigenvec_ad guess for the -iω eigenvector\noptions::NewtonPar\n\n\n\n\n\nSimplified call to refine an initial guess for a Hopf point. More precisely, the call is as follows newtonHopf(F, J, Jt, br::ContResult, index::Int64, options) where the parameters are as usual except that you have to pass the branch br from the result of a call to continuation with detection of bifurcations enabled and index is the index of bifurcation point in br you want to refine.\n\nwarning: Eigenvectors`\nThis simplified call has been written when the eigenvectors are organised in a 2d Array evec where evec[:,2] is the second eigenvector in the list.\n\n\n\n\n\n"
},

{
    "location": "codim2Continuation/#PseudoArcLengthContinuation.continuationFold",
    "page": "Fold / Hopf Continuation",
    "title": "PseudoArcLengthContinuation.continuationFold",
    "category": "function",
    "text": "codim 2 continuation of Fold points. This function turns an initial guess for a Fold point into a curve of Fold points based on a Minimally Augmented formulation. The arguments are as follows\n\nF = (x, p1, p2) ->    F(x, p1, p2) where p is the parameter associated to the Fold point\nJ = (x, p1, p2) -> d_xF(x, p1, p2) associated jacobian\nJt = (x, p1, p2) -> transpose(d_xF(x, p1, p2)) associated jacobian\nd2F = (x, p1, p2, v1, v2) -> d2F(x, p1, p2, v1, v2) this is the hessian of F computed at (x, p1, p2) and evaluated at (v1, v2).\nfoldpointguess initial guess (x0, p10) for the Fold point. It should be a Vector\np2 parameter p2 for which foldpointguess is a good guess\neigenvec guess for the 0 eigenvector at p1_0\noptions::NewtonPar\n\n\n\n\n\ncodim 2 continuation of Fold points. This function turns an initial guess for a Fold point into a curve of Fold points based on a Minimally Augmented formulation. The arguments are as follows\n\nF = (x, p1, p2) -> F(x, p1, p2) where p is the parameter associated to the Fold point\nJ = (x, p1, p2) -> d_xF(x, p1, p2) associated jacobian\nfoldpointguess initial guess (x0, p10) for the Fold point. It should be a Vector\np2 parameter p2 for which foldpointguess is a good guess\neigenvec guess for the 0 eigenvector at p1_0\noptions::NewtonPar\n\nwarning: Hessian\nThe hessian of F in this case is computed with Finite differences. This can be slow for many variables, e.g. ~1e6\n\n\n\n\n\nSimplified call for continuation of Fold point. More precisely, the call is as follows continuationFold(F, J, Jt, br::ContResult, index::Int64, options) where the parameters are as for continuationFold except that you have to pass the branch br from the result of a call to continuation with detection of bifurcations enabled and index is the index of bifurcation point in br you want to refine.\n\n\n\n\n\n"
},

{
    "location": "codim2Continuation/#PseudoArcLengthContinuation.continuationHopf",
    "page": "Fold / Hopf Continuation",
    "title": "PseudoArcLengthContinuation.continuationHopf",
    "category": "function",
    "text": "codim 2 continuation of Hopf points. This function turns an initial guess for a Hopf point into a curve of Hopf points based on a Minimally Augmented formulation. The arguments are as follows\n\n(x, p1, p2)-> F(x, p1, p2) where p is the parameter associated to the hopf point\nJ = (x, p1, p2)-> d_xF(x, p1, p2) associated jacobian\nhopfpointguess initial guess (x0, p10) for the Hopf point. It should be a Vector or a BorderedVector\np2 parameter p2 for which hopfpointguess is a good guess\neigenvec guess for the iω eigenvector at p1_0\neigenvec_ad guess for the -iω eigenvector at p1_0\noptions::NewtonPar\n\n\n\n\n\nSimplified call for continuation of Hopf point. More precisely, the call is as follows continuationHopf(F, J, Jt, br::ContResult, index::Int64, options) where the parameters are as for continuationHopf except that you have to pass the branch br from the result of a call to continuation with detection of bifurcations enabled and index is the index of bifurcation point in br you want to refine.\n\nwarning: Eigenvectors`\nThis simplified call has been written when the eigenvectors are organised in a 2d Array evec where evec[:,2] is the second eigenvector in the list.\n\n\n\n\n\n"
},

{
    "location": "codim2Continuation/#Functions-1",
    "page": "Fold / Hopf Continuation",
    "title": "Functions",
    "category": "section",
    "text": "newtonFold(F, J, Jt, d2F, foldpointguess::Union{Vector, BorderedVector{vectype, T}}, eigenvec, options::NewtonPar; normN = norm) where {T,vectype}newtonFoldnewtonHopfcontinuationFoldcontinuationHopf"
},

{
    "location": "periodicOrbitCont/#",
    "page": "Periodic Orbits",
    "title": "Periodic Orbits",
    "category": "page",
    "text": ""
},

{
    "location": "periodicOrbitCont/#Periodic-orbits-1",
    "page": "Periodic Orbits",
    "title": "Periodic orbits",
    "category": "section",
    "text": "Several ways for finding periodic orbits are provided. A simple shooting algorithm is provided for different schemes. For example, we have ShootingProblemMid for the implicit Mid Point (order 2 in time), ShootingProblemBE for Backward Euler and ShootingProblemTrap for the trapezoidal rule.warning: Shooting methods\nWe do not recommend using the above methods, this is still work in progress. For now, you can use newton with Finite Differences jacobian (ie you do not specify the jocobian option in newton). Indeed, the implementations of the inverse of the jacobian is unstable because one needs to multiply M matrices.Instead, we have another method were we compute M slices of the periodic orbit. This requires more memory than the previous methods. This is implemented by PeriodicOrbitTrap for which the problem of finding periodic orbit is discretized using Finite Differences based on a trapezoidal rule. See Structs."
},

{
    "location": "periodicOrbitCont/#Computation-with-newton-1",
    "page": "Periodic Orbits",
    "title": "Computation with newton",
    "category": "section",
    "text": "Have a look at the Continuation of periodic orbits example for the Brusselator."
},

{
    "location": "periodicOrbitCont/#Continuation-1",
    "page": "Periodic Orbits",
    "title": "Continuation",
    "category": "section",
    "text": "Have a look at the Continuation of periodic orbits example for the Brusselator."
},

{
    "location": "faq/#",
    "page": "Frequently Asked Questions",
    "title": "Frequently Asked Questions",
    "category": "page",
    "text": ""
},

{
    "location": "faq/#FAQ-1",
    "page": "Frequently Asked Questions",
    "title": "FAQ",
    "category": "section",
    "text": ""
},

{
    "location": "faq/#How-can-I-save-a-solution-every-n-steps,-or-at-specific-parameter-values?-1",
    "page": "Frequently Asked Questions",
    "title": "How can I save a solution every n steps, or at specific parameter values?",
    "category": "section",
    "text": "You can use the callback finaliseSolution in the function call continuation. For example, you can use something like this to save all stepsmySave(u, tau, step, contResult, personaldata)\n	push!(personaldata, u)\nendand pass it like continuation(F,J,u,p0, finaliseSolution = (z, tau, step, contResult) -> mySave(z, tau, step, contResult, myData))"
},

{
    "location": "faq/#The-Fold-/-Hopf-Continuation-does-not-work,-why?-1",
    "page": "Frequently Asked Questions",
    "title": "The Fold / Hopf Continuation does not work, why?",
    "category": "section",
    "text": "This requires some precise computations. Have you tried passing the expression of the Jacobian instead of relying on finite differences."
},

{
    "location": "faq/#What-is-the-parameter-theta-about-in-ContinuationPar?-1",
    "page": "Frequently Asked Questions",
    "title": "What is the parameter theta about in ContinuationPar?",
    "category": "section",
    "text": "See the description of continuation on the page Library."
},

{
    "location": "library/#",
    "page": "Library",
    "title": "Library",
    "category": "page",
    "text": ""
},

{
    "location": "library/#Library-1",
    "page": "Library",
    "title": "Library",
    "category": "section",
    "text": ""
},

{
    "location": "library/#PseudoArcLengthContinuation.PeriodicOrbitTrap",
    "page": "Library",
    "title": "PseudoArcLengthContinuation.PeriodicOrbitTrap",
    "category": "type",
    "text": "pb = PeriodicOrbitTrap(F, J, ϕ, xπ, M::Int, linsolve)\n\nThis structure implements Finite Differences based on Trapezoidal rule to locate periodic orbits. The arguements are as follows\n\nF vector field\nJ jacobian of the vector field\nϕ used for the Poincare section\nxπ used for the Poincare section\nM::Int number of slices in [0,2π]\nlinsolve <: LinearSolver  linear solver\n\nYou can then call pb(orbitguess) to apply the functional to a guess. Note that orbitguess must be of size M * N + 1 where N is the number of unknowns in the state space and orbitguess[M*N+1] is an estimate of the period of the limit cycle.\n\nThe scheme is as follows, one look for T = x[end] and  x_i+1 - x_i - frach2 left(F(x_i+1) + F(x_i)right) = 0\n\nwhere h = T/M. Finally, the phase of the periodic orbit is constraint by\n\nlangle x1 - xpi phirangle\n\n\n\n\n\n"
},

{
    "location": "library/#Structs-1",
    "page": "Library",
    "title": "Structs",
    "category": "section",
    "text": "PeriodicOrbitTrap"
},

{
    "location": "library/#PseudoArcLengthContinuation.newton",
    "page": "Library",
    "title": "PseudoArcLengthContinuation.newton",
    "category": "function",
    "text": "	newton(F, J, x0, options, normN = norm)\n\nThis is the Newton Solver for F(x) = 0 with Jacobian J and initial guess x0. The function normN allows to specify a norm for the convergence criteria. It is important to set the linear solver options.linsolve properly depending on your problem. This solver is used to solve J(x)u = -F(x) in the Newton step. You can for example use Default() which is the operator backslash which works well for Sparse / Dense matrices. Iterative solver (GMRES) are also implemented. You should implement your own solver for maximal efficiency. This is quite easy to do, have a look at src/LinearSolver.jl. The functions or callable provided are as follows:\n\nx -> F(x) functional whose zeros are looked for. In particular, it is not inplace,\ndF(x) = x -> J(x) compute the jacobian of F at x. It is then passed to options.linsolve\n\nOutput:\n\nsolution:\nhistory of residuals\nflag of convergence\nnumber of iterations\n\n\n\n\n\n"
},

{
    "location": "library/#PseudoArcLengthContinuation.newtonDeflated",
    "page": "Library",
    "title": "PseudoArcLengthContinuation.newtonDeflated",
    "category": "function",
    "text": "newtonDeflated(Fhandle::Function, Jhandle, x0, options:: NewtonPar{T}, defOp::DeflationOperator{T, vectype})\n\nThis is the deflated version of the Newton Solver. It penalises the roots saved in defOp.roots\n\n\n\n\n\n"
},

{
    "location": "library/#Newton-1",
    "page": "Library",
    "title": "Newton",
    "category": "section",
    "text": "newtonnewtonDeflated"
},

{
    "location": "library/#PseudoArcLengthContinuation.newtonHopf-Tuple{Function,Any,Any,AbstractArray{T,1} where T,Any,Any,NewtonPar}",
    "page": "Library",
    "title": "PseudoArcLengthContinuation.newtonHopf",
    "category": "method",
    "text": "This function turns an initial guess for a Hopf point into a solution to the Hopf problem based on a Minimally Augmented formulation. The arguments are as follows\n\nF  = (x, p) -> F(x, p) where p is the parameter associated to the Hopf point\nJ  = (x, p) -> d_xF(x, p) associated jacobian\nJt = (x, p) -> transpose(d_xF(x, p)) associated jacobian\nhopfpointguess initial guess (x0, p0) for the Hopf point. It should be a AbstractVector or a BorderedVector.\neigenvec guess for the  iω eigenvector\neigenvec_ad guess for the -iω eigenvector\noptions::NewtonPar\n\n\n\n\n\n"
},

{
    "location": "library/#PseudoArcLengthContinuation.newtonHopf-Tuple{Any,Any,Any,ContResult,Int64,NewtonPar}",
    "page": "Library",
    "title": "PseudoArcLengthContinuation.newtonHopf",
    "category": "method",
    "text": "Simplified call to refine an initial guess for a Hopf point. More precisely, the call is as follows newtonHopf(F, J, Jt, br::ContResult, index::Int64, options) where the parameters are as usual except that you have to pass the branch br from the result of a call to continuation with detection of bifurcations enabled and index is the index of bifurcation point in br you want to refine.\n\nwarning: Eigenvectors`\nThis simplified call has been written when the eigenvectors are organised in a 2d Array evec where evec[:,2] is the second eigenvector in the list.\n\n\n\n\n\n"
},

{
    "location": "library/#Newton-for-Fold-/-Hopf-1",
    "page": "Library",
    "title": "Newton for Fold / Hopf",
    "category": "section",
    "text": "newtonFold(F, J, Jt, d2F, foldpointguess::Union{Vector, BorderedVector{vectype, T}}, eigenvec, options::NewtonPar; normN = norm) where {T,vectype}newtonFold(F::Function, J, Jt, foldpointguess::AbstractVector, eigenvec::AbstractVector, options::NewtonPar)newtonFold(F::Function, J, Jt, br::ContResult, ind_fold::Int64, options::NewtonPar)newtonHopf(F::Function, J, Jt, hopfpointguess::AbstractVector, eigenvec, eigenvec_ad, options::NewtonPar)newtonHopf(F, J, Jt, br::ContResult, ind_hopf::Int64, options::NewtonPar)"
},

{
    "location": "library/#PseudoArcLengthContinuation.continuation",
    "page": "Library",
    "title": "PseudoArcLengthContinuation.continuation",
    "category": "function",
    "text": "continuation(F::Function, J, u0, p0::Real, contParams::ContinuationPar; plot = false, normC = norm, printsolution = norm, plotsolution::Function = (x;kwargs...)->nothing, finaliseSolution::Function = (x, y)-> nothing, linearalgo   = :bordering, verbosity = 2)\n\nCompute the continuation curve associated to the functional F and its jacobian J. The parameters are as follows\n\nF = (x, p) -> F(x, p) where p is the parameter for the continuation\nJ = (x, p) -> d_xF(x, p) its associated jacobian\nu0 initial guess\ncontParams parameters for continuation, with type ContinuationPar\nplot = false whether to plot the solution while computing\nprintsolution = norm function used to plot in the continuation curve, e.g. norm or x -> x[1]\nplotsolution::Function = (x; kwargs...)->nothing function implementing the plotting of the solution.\nfinaliseSolution::Function = (z, tau, step, contResult) -> true Function called at the end of each continuation step. Can be used to alter the continuation step (stop it by returning false) or saving personal data...\nlinearalgo   = :bordering. Must belong to [:bordering, :full]\nverbosity controls the amount of information printed during the continuation process.\n\'normC = norm\' norm to be used in the different Newton solves\n\nThe function outputs\n\ncontres::ContResult structure which contains the computed branch\nu::BorderedVector the last solution computed on the branch\n\nMethod\n\nBordered system of equations\n\nThe pseudo arclength continuation method solves the equation F(xp) = 0 (or dimension N) together with the pseudo-arclength constraint N(x p) = fracthetalength(u) langle x - x_0 tau_0rangle + (1 - theta)cdot(p - p_0)cdot dp_0 - ds = 0. In practice, the curve is parametrised by s so that (x(s)p(s)) is a curve of solutions to F(xp). This formulation allows to pass turning points (where the implicit theorem fails). In the previous formula, (x_0 p_0) is a solution for a given s_0, (tau_0 dp_0) is the tangent to the curve at s_0. Hence, to compute the curve of solutions, we need solve an equation of dimension N+1 which is called a Bordered system.\n\nwarning: Parameter `theta`\nThe parameter theta in the type ContinuationParis very important. It should be tuned for the continuation to work properly especially in the case of large problems where the langle x - x_0 tau_0rangle component in the constraint might be favoured too much.\n\nThe parameter ds is adjusted internally depending on the number of Newton iterations and other factors. See the function stepSizeControl for more information. An important parameter to adjust the magnitude of this adaptation is the parameter a in the type ContinuationPar.\n\nAlgorithm\n\nThe algorithm works as follows:\n\nStart from a known solution (x_0p_0tau_0dp_0)\nPredictor set (x_1p_1) = (x_0p_0) + dscdot (tau_0dp_0)\nCorrector solve F(xp)=0 N(xp)=0 with a (Bordered) Newton Solver.\nNew tangent Compute (tau_1dp_1), set (x_0p_0tau_0dp_0)=(x_1p_1tau_1dp_1) and return to step 2\n\nNatural continuation\n\nWe speak of natural continuation when we do not consider the constraint N(xp)=0. Knowing (x_0p_0), we use x_0 as a guess for solving F(xp_1)=0 with p_1 close to p_0. Again, this will fail at Turning points but it can be faster to compute than the constrained case. This is set by the field natural in the type ContinuationPar`\n\nTangent computation (step 4)\n\nThere are various ways to compute (tau_1p_1). The first one is called secant and is parametrised by the field secant in the type ContinuationPar. It is computed by (tau_1p_1) = (z_1p_1) - (z_0p_0) and normalised by the norm up^2_theta = fracthetalength(u) langle uurangle + (1 - theta)cdot p^2. If secant is set to false, another method is use computing (tau_1p_1) by solving a bordered linear system, see the function getTangentBordered for more information.\n\nBordered linear solver\n\nWhen solving the Bordered system F(xp) = 0 N(x p)=0, one faces the issue of solving the Bordered linear system beginbmatrix J  a     b^T  cendbmatrixbeginbmatrixX   yendbmatrix =beginbmatrixR  nendbmatrix. This can be solved in many ways via bordering (which requires two Jacobian inverses) or by forming the bordered matrix (which works well for sparse matrices). The choice of method is set by the argument linearalgo. Have a look at the function linearBorderedSolver for more information.\n\n\n\n\n\n"
},

{
    "location": "library/#Continuation-1",
    "page": "Library",
    "title": "Continuation",
    "category": "section",
    "text": "continuation"
},

{
    "location": "library/#PseudoArcLengthContinuation.continuationHopf-Tuple{Function,Any,Any,AbstractArray{T,1} where T,Any,Any,Any,ContinuationPar}",
    "page": "Library",
    "title": "PseudoArcLengthContinuation.continuationHopf",
    "category": "method",
    "text": "codim 2 continuation of Hopf points. This function turns an initial guess for a Hopf point into a curve of Hopf points based on a Minimally Augmented formulation. The arguments are as follows\n\n(x, p1, p2)-> F(x, p1, p2) where p is the parameter associated to the hopf point\nJ = (x, p1, p2)-> d_xF(x, p1, p2) associated jacobian\nhopfpointguess initial guess (x0, p10) for the Hopf point. It should be a Vector or a BorderedVector\np2 parameter p2 for which hopfpointguess is a good guess\neigenvec guess for the iω eigenvector at p1_0\neigenvec_ad guess for the -iω eigenvector at p1_0\noptions::NewtonPar\n\n\n\n\n\n"
},

{
    "location": "library/#PseudoArcLengthContinuation.continuationHopf-Tuple{Function,Any,Any,ContResult,Int64,Real,ContinuationPar}",
    "page": "Library",
    "title": "PseudoArcLengthContinuation.continuationHopf",
    "category": "method",
    "text": "Simplified call for continuation of Hopf point. More precisely, the call is as follows continuationHopf(F, J, Jt, br::ContResult, index::Int64, options) where the parameters are as for continuationHopf except that you have to pass the branch br from the result of a call to continuation with detection of bifurcations enabled and index is the index of bifurcation point in br you want to refine.\n\nwarning: Eigenvectors`\nThis simplified call has been written when the eigenvectors are organised in a 2d Array evec where evec[:,2] is the second eigenvector in the list.\n\n\n\n\n\n"
},

{
    "location": "library/#Continuation-for-Fold-/-Hopf-1",
    "page": "Library",
    "title": "Continuation for Fold / Hopf",
    "category": "section",
    "text": "continuationFold(F::Function, J, Jt, foldpointguess::AbstractVector, p2_0::Real, eigenvec::AbstractVector, options_cont::ContinuationPar ; kwargs...)continuationFold(F::Function, J, Jt, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)continuationHopf(F::Function, J, Jt, hopfpointguess::AbstractVector, p2_0, eigenvec, eigenvec_ad, options_cont::ContinuationPar ; kwargs...)continuationHopf(F::Function, J, Jt, br::ContResult, ind_hopf::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)"
},

{
    "location": "library/#PseudoArcLengthContinuation.plotBranch-Tuple{ContResult}",
    "page": "Library",
    "title": "PseudoArcLengthContinuation.plotBranch",
    "category": "method",
    "text": "Plot the branch of solutions from a ContResult. You can also pass parameters like plotBranch(br, marker = :dot). For the continuation diagram, the legend is as follows (:fold => :black, :hopf => :red, :bp => :blue, :nd => :magenta, :none => :yellow)\n\n\n\n\n\n"
},

{
    "location": "library/#PseudoArcLengthContinuation.plotBranch!-Tuple{ContResult}",
    "page": "Library",
    "title": "PseudoArcLengthContinuation.plotBranch!",
    "category": "method",
    "text": "Append to the current plot the plot of the branch of solutions from a ContResult. You can also pass parameters like plotBranch!(br, marker = :dot)\n\n\n\n\n\n"
},

{
    "location": "library/#PseudoArcLengthContinuation.plotBranch-Tuple{Array{T,1} where T}",
    "page": "Library",
    "title": "PseudoArcLengthContinuation.plotBranch",
    "category": "method",
    "text": "Plot all the branches contained in brs in a single figure. Convenient when many bifurcation diagram have been computed.\n\n\n\n\n\n"
},

{
    "location": "library/#Plotting-1",
    "page": "Library",
    "title": "Plotting",
    "category": "section",
    "text": "plotBranch(contres::ContResult; kwargs...)plotBranch!(contres::ContResult; kwargs...)plotBranch(brs::Vector; kwargs...)"
},

]}
