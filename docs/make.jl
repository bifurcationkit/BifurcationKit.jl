using Documenter, PseudoArcLengthContinuation

makedocs(doctest = false,
	sitename = "Pseudo Arc Length Continuation in Julia",
	format = Documenter.HTML(collapselevel = 1),
	# format = DocumenterLaTeX.LaTeX(),
	authors = "Romain Veltz",
	pages = Any[
		"Home" => "index.md",
		"Tutorials" => [
			"1/ Temperature model" => "tutorials1.md",
			"2/ Temperature model with Spectral Collocation" => "tutorials1b.md",
			"3/ Swift-Hohenberg" => "tutorials2.md",
			"4/ The Swift-Hohenberg on the GPU (non-local)" => "tutorials2b.md",
			"5/ Brusselator 1d" => "tutorials3.md",
			"6/ Period Doubling in BVAM model" => "tutorialsPD.md",
			"7/ Ginzburg-Landau 2d (GPU)" => "tutorialsCGL.md",
			"8/ Ginzburg-Landau 2d (Shooting)" => "tutorialsCGLShoot.md",
		],
		"Advanced Usage" => [
			"Linear / Eigen Solvers" => "linearsolver.md",
			"Bordered linear solvers" => "borderedlinearsolver.md",
			"Bifurcations" => "detectionBifurcation.md",
			"Fold / Hopf Continuation" => "codim2Continuation.md",
			"Deflated problem" => "deflatedproblem.md",
			"Constrained problem" => "constrainedproblem.md",
			"Periodic Orbits" => [
				"Introduction" => "periodicOrbit.md",
				"Finite Differences" => "periodicOrbitFD.md",
				"Shooting" => "periodicOrbitShooting.md",
				],
			"DiffEq wrapper" => "diffeq.md",
			"Bordered arrays" => "Borderedarrays.md",
			"Iterator Interface" => "iterator.md",
		],
		"Frequently Asked Questions" => "faq.md",
		"Library" => "library.md"
	]
	)

deploydocs(
	repo = "github.com/rveltz/PseudoArcLengthContinuation.jl.git",
)
