using Documenter, PseudoArcLengthContinuation

makedocs(doctest = false,
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
	repo   = "github.com/rveltz/PseudoArcLengthContinuation.jl.git",
)
