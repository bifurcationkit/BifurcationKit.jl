# Periodic orbits

Several ways for finding periodic orbits are provided. A simple shooting algorithm is provided for different schemes. For example, we have `ShootingProblemMid` for the implicit Mid Point (order 2 in time), `ShootingProblemBE` for Backward Euler and `ShootingProblemTrap` for the trapezoidal rule.

!!! warning "Shooting methods"
    We do not recommend using the above methods, this is still work in progress. For now, you can use `newton` with Finite Differences jacobian (ie you do not specify the jocobian option in `newton`). Indeed, the implementations of the inverse of the jacobian is unstable because one needs to multiply `M` matrices.

Instead, we have another method were we compute `M` slices of the periodic orbit. This requires more memory than the previous methods. This is implemented by `PeriodicOrbitTrap` for which the problem of finding periodic orbit is discretized using Finite Differences based on a trapezoidal rule. See [Structs](@ref).

## Computation with `newton`

Have a look at the [Continuation of periodic orbits](@ref) example for the Brusselator.

## Continuation

Have a look at the [Continuation of periodic orbits](@ref) example for the Brusselator.

