# Educational introduction to bifurcation analysis

This page is an educational introduction to bifurcation analysis and creation of bifurcation diagrams. It will show a completely self-contained approach, and its goal is to serve as an introduction to users not yet fully familiar with bifurcation analysis. 

In this example we will focus only on the most simple parts: calculating the bifurcation diagram of a 1-dimensional system, which can only have fixed points and can only undergo the most standard bifurcations like a saddle-node or a pitchfork one.

While what we will discuss here does not reflect the full features and power of BifurcationKit.jl, it follows the same guiding principles.

## A simple 1-dimensional system

For this example we will focus on creating the bifurcation diagram of a simple, 1-dimensional continuous dynamical system given by
$$
\frac{dT}{dt} = 0.5 + 0.2\tanh\left(\frac{T-263}{4}\right)-  \epsilon  \cdot 10 \cdot (0.002T)^4
$$
which is an extreme simplification of the energy balance of the Earth, including the ice-albedo feedback, as was studied by e.g. [Budyko, 1969](https://onlinelibrary.wiley.com/doi/10.1111/j.2153-3490.1969.tb00466.x).

Before going into the bifurcations aspect, let's simply define this system in Julia:

```@example Education
using ForwardDiff # to calculate derivative (yes, it is possible analytically)
αtan(T) = 0.5 - 0.2*tanh((T-263)/4)
dTdt(T, ε = 0.65, α=αtan, s= 1.0) = s*(1 - α(T)) - 1.6e-10 * ε * T^4
dTdt(T; ε = 0.65, α=αtan, s = 1.0) = dTdt(T, ε, α, s)
d²Tdt²(T, ε) = ForwardDiff.derivative(T -> dTdt(T, ε), T)
```
Now the two functions `dTdt, d²Tdt²` describe the rule of the system and its Jacobian. Because we have a 1-dimensional system, the Jacobian is just a single number, the derivative of the rule `dTdt`.

The bifurcation diagram we will produce for this system looks like this:
**insert figure here.**

## Newton's method for finding fixed points
A (simplified) bifurcation diagram is by definition a plot that shows how the fixed points of a system, and their stability, changes when changing a parameter $p$. To compute such a plot we need to find the fixed points.

While sometimes it is possible to compute bifurcation diagrams analytically, because one can extract the fixed points of the system and their stability analytically, this becomes harder in higher dimensional systems.
Therefore, in general we need a numerical method that can detect and track not only stable but also unstable fixed points. 
For locating stable or unstable fixed points we can consider the fixed point problem $f(\vec{x}) = 0$ as a root finding problem and use some numerical algorithm to locate the root(s) $\vec{x}^*$ with $f(\vec{x}^*) = 0$.

This can be computed iteratively using _Newton's method_
$$
\vec{x}_{j+1} = \vec{x}_{j} - \delta_j J_f^{-1}(\vec{x}_{j}) f(\vec{x}_{j})
$$
where $J_F$ stands for the Jacobian of $f$ (at state $\vec{x}$ and parameter $p$). Since in this tutorial we will only focus on 1-dimensional systems, the Jacobian reduces to a number, and the state is not a vector $\vec{x}$ but a number $x$.

The index $j$ counts the number of iterations. Starting from an initial guess $x_0$ the method will converge to the fixed point $x^*$ once $|x_{j+1}-x_j|$ is smaller than some threshold value like `1e-3`. The factor $0 \le \delta_j \le 1$ is a is chosen `< 1` so that Newton's method can also find unstable fixed points of $f$.

With the power of Newton's method, we can now make a bifurcation diagram with a brute force approach: starting with an initial state and parameter $(x_0, p_0)$, we iterate Newton's method until we converge to the fixed point $x^*(p_0)$. Checking the derivative of $f$ (i.e., the "Jacobian"), we also label this point as stable or unstable. We then slightly decrease or increase $p_1 = p_0 + dp$, and use the previously found $x^*(p_0)$ as the new starting guess, and start the Newton iteration again. This will create a bifurcation curve. Given that we scan the fixed point stability along the curve, once we found that it changes, we have identified a bifurcation point.

Repeating this process for many starting guesses $x_0$ will hopefully fill the bifurcation diagram with all bifurcation curves.

Here is a simple code that implements the Newton's method:
```@example Education
"""
    newton(p, x0, dxdt, d²xdt², δ; max_steps = 200) → (x*, success)
The Newton algorithm to converge to the fixed point for given parameter `p`,
initial state guess `x0`, function `dxdt`, derivative `d²xdt²` and stepping factor `δ`.
If iteration occurs for more than `max_steps`, then `success` will be `false`.
"""
function newton(p, x0, dxdt, d²xdt², δ; max_steps = 200)
    i = 0
    xⱼ = x0
    xⱼ₊₁ = newton_step!(p, xⱼ, dxdt, d²xdt², δ)
    while abs(xⱼ₊₁ - xⱼ) > 1e-3
        xⱼ = xⱼ₊₁
        xⱼ₊₁ = newton_step!(p, xⱼ, dxdt, d²xdt², δ)
        i += 1
        i > max_steps && return (xⱼ₊₁, false)
    end
    return xⱼ₊₁, true
end
function newton_step!(p, xⱼ, dxdt, d²xdt², δ)
    g = dxdt(xⱼ, p)
    d = d²xdt²(xⱼ, p)
    return xⱼ₊₁ = xⱼ - δ*(1/d)*g
end
```

## Continuation
This brute force approach is not only inefficient, but will also fail when we are very close to bifurcations. E.g., imagine that at $p=p_c$ a saddle-node bifurcation happens. If we set as new guess $p_0 = p_c - dp$, no fixed points will exist, and no matter from which $x_0$ we start from, Newton's method will iterate forever.

We want a way to _continue_ the already found bifurcation curves.

**TODO: The polynomial fit I do below is _bad_. It doesn't behave well in higher dimensions. Gotta change it.**


Here is the code for continuation:
```@example Education
import Polynomials

function predict_next_initial_guess(xs, ps, dx, dp; ℓ = 6, order = 3)
    if length(ps) < (max(ℓ, order+1))
        return (xs[end], ps[end] + dp)
    else
        # Fit polynomial (parameter = y-axis)
        poly = Polynomials.fit(@view(xs[end-ℓ+1:end]), @view(ps[end-ℓ+1:end]), order)
        next_x = xs[end] + dx
        return (next_x, poly(next_x))
    end
end

function continuation!(xs, ps, dxdt, d²xdt²; δ, dp, dx)
    # The `success` while loop is used in case we over-shooted the bifurcation curve
    # and have gone beyond a turning point. We then retry with halfed prediction
    success = false; j = 1
    xˣ_next = x_next = xs[end]
    p_next = ps[end]
    while !success
        x_next, p_next = predict_next_initial_guess(xs, ps, dx/(2^j), dp/(2^j))
        xˣ_next, success = newton(p_next, x_next, dxdt, d²xdt², δ)
        j += 1
        j > 3 && error("j exceed normal amount, something is wrong!")
    end
    push!(xs, xˣ_next); push!(ps, p_next)
end

function continuation(dxdt, d²xdt², x0, p0;
        pmin, pmax, dp, dx, N, δ    
    )

    ps = [p0]
    xs = [x0]
    stability = Bool[]
    for _ in 1:N
        continuation!(xs, ps, dxdt, d²xdt²; δ, dp, dx)
        # Detect stability of found fixed point
        isstable = d²xdt²(xs[end], ps[end]) < 0
        push!(stability, isstable)
        # Stop iteration if we exceed given parameter margins
        (pmin ≤ ps[end] ≤ pmax) || break
    end
    popfirst!(xs); popfirst!(ps) # remove initial guess
    return xs, ps
end
```

## Produce and plot the bifurcation diagram
```@example Education
# Arguments for algorithm
p0 = 0.9
x0 = 250.0
dxdt = dTdt
d²xdt² = d²Tdt²

# Keywords for algorithm
dp = -0.001
dx = 0.1
pmin = 0
pmax = 1
N = 2000 # how many times to continue the process.
δ = 0.9

xs, ps = continuation(dxdt, d²xdt², x0, p0;
    pmin, pmax, dp, dx, N, δ
)

using Plots
colors = [s ? "C0" : "C1" for s in stability]
scatter(ps, xs; c = colors)
```
