# Detection of bifurcation points

Depending on the bifurcation type, detection is ensured during a call to `br, _ = continuation(...)` by turning on a flag.

## Eigensolver

The user must provide an eigensolver by setting `NewtonOptions.eigsolve` where `NewtonOptions` is located in the parameter `::ContinuationPar` passed to continuation. See `src/Newton.jl` for more information on the structure of the options passed to `newton` and `continuation`.

The eigensolver is highly problem dependent and this is why the user should implement / parametrize its own eigensolver through the abstract type `AbstractEigenSolver` or select one among the provided like `DefaultEig(), eig_IterativeSolvers(), eig_KrylovKit`. See `src/EigSolver.jl`.

## Fold bifurcation
The detection of **Fold** point is done by monitoring  the monotonicity of the parameter.

The detection is triggered by setting `detect_fold = true` in the parameter `::ContinuationPar` passed to continuation. When a **Fold** is detected, a flag is added to `br.bifpoint` allowing for later refinement.

## Generic bifurcation

By this we mean a change in the dimension of the Jacobian kernel. The detection of Branch point is done by analysis of the spectrum of the Jacobian.

The detection is triggered by setting `detect_bifurcation = true` in the parameter `::ContinuationPar` passed to continuation. The user must also provide a hint of the number of eigenvalues to be computed `nev = 10` in the parameter `::ContinuationPar` passed to continuation. Note that `nev` is incremented whenever a bifurcation point is detected. When a **Branch point** is detected, a flag is added to `br.bifpoint` allowing for later refinement.

## Hopf bifurcation

The detection of Branch point is done by analysis of the spectrum of the Jacobian.

The detection is triggered by setting `detect_bifurcation = true` in the parameter `::ContinuationPar` passed to continuation. The user must also provide a hint of the number of eigenvalues to be computed `nev = 10` in the parameter `::ContinuationPar` passed to continuation. Note that `nev` is incremented whenever a bifurcation point is detected. When a **Hopf point** is detected, a flag is added to `br.bifpoint` allowing for later refinement.