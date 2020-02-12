# Wrapper to the package `DifferentialEquations.jl`

> Note that this is still work in progress.

The current package provides basic methods to wrap some of the functionality of `DifferentialEquations.jl`. 

Basically, the ultimate idea is that you provide a `prob::ODEProblem` to our `newton, continuation...` and they will use the expression of the jacobians, the linear solvers... that you already provided for the construction of `prob`.