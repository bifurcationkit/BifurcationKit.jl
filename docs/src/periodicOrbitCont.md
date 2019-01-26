# Periodic orbits

Several ways for finding periodic orbits. A simple shooting algorithm (quite unstable) is provided for different scheme. For example, we have `ShootingProblemMid` for the implicit Mid Point (order 2 in time), `ShootingProblemBE` for Backward Euler and `ShootingProblemTrap` for the trapezoidal rule.

Finally, we also have a method were we compute `M` slices of the periodic orbit. This requires more memory than the previous method but seems more general. This is implemented by `PeriodicOrbitTrap` for which the problem of finding periodic orbit is discretized using Finite Differences based on a trapezoidal rule.

## Computation with `newton`

Have a look at the Brusselator example.

## Continuation

Have a look at the Brusselator example.