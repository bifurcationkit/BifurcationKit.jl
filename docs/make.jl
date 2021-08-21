using Documenter, BifurcationKit, Setfield
# using DocThemeIndigo
ENV["GKSwstype"] = "100"

makedocs(doctest = false,
	sitename = "Bifurcation Analysis in Julia",
	format = Documenter.HTML(collapselevel = 1,assets = ["assets/indigo.css"]),
	# format = DocumenterLaTeX.LaTeX(),
	authors = "Romain Veltz",
	pages = Any[
		"Home" => "index.md",
		"Overview" => "guidelines.md",
		"Tutorials" => "tutorials.md",
		"Functionalities" => [
			"Plotting" => "plotting.md",
			"Periodic Orbits" => [
				"Introduction" => "periodicOrbit.md",
				"Finite Differences" => "periodicOrbitTrapeze.md",
				"Shooting" => "periodicOrbitShooting.md",
				],
			"Continuation methods" => [
					"Introduction" => "IntroContinuation.md",
			"Predictors / correctors" => "Predictors.md",
					"PALC" => "PALC.md",
					"Moore Spence Continuation" => "MooreSpence.md",
					"Deflated Continuation" => "DeflatedContinuation.md",
				],
			"Event Handling and Callback" => "EventCallback.md",
			"Bifurcations" => [
				"Bifurcation detection (codim 1)" => "detectionBifurcation.md",
				"Fold / Hopf Continuation (codim 2)" => "codim2Continuation.md",
				],
			"Normal form" =>[
				"Simple branch point" => "simplebp.md",
				"Non-simple branch point" => "nonsimplebp.md",
				"Simple Hopf point" => "simplehopf.md",
			],
			"Branch switching" => "branchswitching.md",
			"Bifurcation diagram" => "BifurcationDiagram.md",
			"Deflated problems" => "deflatedproblem.md",
			"Constrained problem" => "constrainedproblem.md",
			"DiffEq wrapper" => "diffeq.md",
			"Iterator Interface" => "iterator.md",
		],
		"Options" => [
			"Linear Solvers" => "linearsolver.md",
			"Bordered linear solvers" => "borderedlinearsolver.md",
			"Eigen Solvers" => "eigensolver.md",
			"Bordered arrays" => "Borderedarrays.md",
		],
		"Frequently Asked Questions" => "faq.md",
		"Library" => "library.md"
	]
	)

deploydocs(
	repo = "github.com/rveltz/BifurcationKit.jl.git",
)
