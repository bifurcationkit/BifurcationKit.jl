# Detection of bifurcation points

The bifurcations are detected during a call to `br, _ = continuation(F, J, u0, p0::Real, contParams::ContinuationPar;kwrags...)` by turning on one of the following flags:

- `contParams.computeEigenValues = true` to trigger the computation of the eigenvalues of the jacobian. It automatically turn on the detection of bifurcation points.
- ``contParams.detect_bifurcation = true` which also turns on `contParams.computeEigenValues = true`

The located bifurcation points are then returned in `br.bifpoint`. **Note that this points are only approximate bifurcation points.** They need to be refined with the methods described here after.

## Eigensolver

The user must provide an eigensolver by setting `NewtonOptions.eigsolve` where `NewtonOptions` is located in the parameter `::ContinuationPar` passed to continuation. See `src/Newton.jl` for more information on the structure of the options passed to `newton` and `continuation`.

The eigensolver is highly problem dependent and this is why the user should implement / parametrize its own eigensolver through the abstract type `AbstractEigenSolver` or select one among the provided like `DefaultEig(), eig_IterativeSolvers(), eig_KrylovKit`. See `src/EigSolver.jl`.

## Fold bifurcation
The detection of **Fold** point is done by monitoring  the monotonicity of the parameter.

The detection is triggered by setting `detect_fold = true` in the parameter `::ContinuationPar` passed to continuation. When a **Fold** is detected, a point is added to `br.bifpoint` allowing for later refinement using the function `newtonFold`.

## Generic bifurcation

By this we mean a change in the dimension of the Jacobian kernel. The detection of Branch point is done by analysis of the spectrum of the Jacobian.

The detection is triggered by setting `detect_bifurcation = true` in the parameter `::ContinuationPar` passed to continuation. The user must also provide a hint of the number of eigenvalues to be computed `nev = 10` in the parameter `::ContinuationPar` passed to continuation. Note that `nev` is incremented whenever a bifurcation point is detected. When a **Branch point** is detected, a point is added to `br.bifpoint` allowing for later refinement.

## Hopf bifurcation

The detection of Branch point is done by analysis of the spectrum of the Jacobian.

The detection is triggered by setting `detect_bifurcation = true` in the parameter `::ContinuationPar` passed to continuation. The user must also provide a hint of the number of eigenvalues to be computed `nev = 10` in the parameter `::ContinuationPar` passed to continuation. Note that `nev` is incremented whenever a bifurcation point is detected. When a **Hopf point** is detected, a point is added to `br.bifpoint` allowing for later refinement using the function `newtonHopf`.