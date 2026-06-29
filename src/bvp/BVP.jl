# BVP Module - Main entry point
#
# This module provides a clean interface for solving Boundary Value Problems
# using various discretization methods (Shooting, Trapezoid, Collocation).

"""
BVP - Boundary Value Problem Interface for BifurcationKit

This module provides:
- `BVPModel`: Mathematical formulation of a BVP
- `PeriodicOrbitModel`: Convenience constructor for periodic orbits
- `Shooting`, `Trapeze`, `Collocation`: Discretization methods
- `discretize`: Combine model + method into a solvable problem
- `BVPBifProblem`: Bifurcation problem for periodic orbits

## Basic Usage

```julia
using BifurcationKit
const BK = BifurcationKit

# 1. Define the vector field
F(u, p) = [u[2], -p.ω² * u[1]]

# 2. Create a BVP model
model = BK.PeriodicOrbitModel(F; n=2)

# 3. Choose discretization
disc = BK.Trapeze(M=100)

# 4. Discretize
bvp = BK.discretize(model, disc)

# 5. Generate initial guess
x0 = BK.generate_solution(bvp, t -> [cos(t), sin(t)], 2π)

# 6. Create BVP bifurcation problem
prob = BK.BVPBifProblem(bvp, x0, (ω=1.0,), (@optic _.ω))

# 7. Continue
br = BK.continuation(prob, PALC(), ContinuationPar())
```

See also: [`BVPModel`](@ref), [`discretize`](@ref), [`BVPBifProblem`](@ref), [`Shooting`](@ref), [`Trapeze`](@ref), [`Collocation`](@ref)
"""
module BVP

using DocStringExtensions
import LinearAlgebra as LA
import ForwardDiff as FD
# using Accessors
import BifurcationKit as BK

abstract type AbstractBVPModel end
abstract type AbstractDiscretizer end
abstract type AbstractDiscretizedBVP end
abstract type AbstractDiscretizedPO <: AbstractDiscretizedBVP end
abstract type AbstractBVPBifProblem <: BK.AbstractBifurcationProblem end

# Core types
include("BVPModel.jl")
include("Discretizers.jl")
include("DiscretizedBVP.jl")
include("discretize.jl")

# Residual/Jacobian implementations for each discretizer
include("shooting/residual.jl")
include("shooting/jacobian.jl")  # Shooting has specialized analytical jacobian
include("trapeze/residual.jl")
include("trapeze/jacobian.jl")
include("collocation/residual.jl")
include("collocation/jacobian.jl")

# Integration with BifurcationKit
include("BVPBifProblem.jl")
include("integration.jl")
include("Continuation.jl")
include("Bifurcations.jl")

# Exports - Core types
export BVPModel, PeriodicOrbitModel
# export AbstractDiscretizer, Shooting, Trapeze, Collocation
export DiscretizedBVP
export discretize, generate_solution
export bvp_residual, bvp_jacobian, jvp
export state_dimension, getperiod

# Exports - BVP Bifurcation Problem
export BVPBifProblem
export get_periodic_orbit, get_bvp

# Internal exports for extensions
export integrate_shooting, integrate_with_sensitivity

end # module BVP
