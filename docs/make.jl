using Documenter, PseudoArcLengthContinuation

makedocs(doctest = false,
	clean = true,
	format = :html,
	sitename = "Pseudo Arc Length Continuation in Julia",
	pages = Any[
						"Home" => "index.md",
						"Advanced Usage" => [
							"Linear Solvers" => "linearsolver.md",
							"Bifurcations" => "detectionBifurcation.md",
							"Fold / Hopf Continuation" => "codim2Continuation.md",
							"Periodic Orbits" => "periodicOrbitCont.md"
						],
						"Frequently Asked Questions" => "faq.md",
						"Library" => "library.md"
					]
	)

deploydocs(
	# deps   = Deps.pip("mkdocs", "python-markdown-math", "mkdocs-cinder", "pygments"),
	repo   = "github.com/rveltz/PseudoArcLengthContinuation.jl.git",
	osname = "linux",
	# target = "build",
	julia = "1.0.3"
)
