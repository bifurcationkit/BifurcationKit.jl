using Documenter, PseudoArcLengthContinuation

makedocs(doctest = false,
	clean = true,
	format = Documenter.HTML(),
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


# ENV["DOCUMENTER_DEBUG"] = true
# ENV["TRAVIS_REPO_SLUG"] = "github.com/rveltz/PseudoArcLengthContinuation.jl.git"

deploydocs( 
	# deps   = Deps.pip("mkdocs", "python-markdown-math", "mkdocs-cinder", "pygments"),
	repo   = "github.com/rveltz/PseudoArcLengthContinuation.jl.git",
	osname = "linux",
	target = "build"
)
