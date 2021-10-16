# Tutorials

There are three levels of tutorials:

1. fully **automatic bifurcation diagram** (**aBD**) computation (only for equilibria): one uses the function `bifurcationdiagram` and let it compute the diagram fully automatically. Another possibility is to use **deflated continuation**.
2. semi-automatic bifurcation diagram computation: one uses **automatic branch switching** (**aBS**) to compute branches at specified bifurcation points
3. manual bifurcation diagram computation: one does not uses automatic branch switching. This has only educational purposes or for complex problems where aBS fails.
## ODE examples

We present examples of the use of the package in the case of ODEs. Although `BifurcationKit.jl` is not geared towards them, we provide some specific methods which allow to study the bifurcations of ODE in a relatively efficient way.

```@contents
Pages = ["ode/tutorialsODE.md", "ode/tutorialCO.md", "ode/tutorialPP2.md","ode/tutorialsODE-PD.md"]
Depth = 1
```

## DAE examples

```@contents
Pages = ["ode/Colpitts.md"]
Depth = 1
```

## Bifurcation of Equilibria
```@contents
Pages = ["tutorials1.md", "tutorials2.md", "mittelmann.md", "tutorials1b.md", "tutorials2b.md", "tutorialsSH3d.md"]
Depth = 1
```

### Automatic bifurcation diagram
```@contents
Pages = ["Swift-Hohenberg1d.md", "mittelmannAuto.md", "tutorialCarrier.md", "ks1d.md", "ks2d.md"]
Depth = 1
```

### Solving PDEs using Finite elements with [Gridap.jl](https://github.com/gridap/Gridap.jl)
```@contents
Pages = ["mittelmannGridap.md"]
Depth = 1
```

## Bifurcation diagrams with periodic orbits
```@contents
Pages = ["tutorials3.md","tutorials3b.md", "BrusselatorFF.md", "tutorialsPD.md", "tutorialsCGL.md", "tutorialsCGLShoot.md","Langmuir.md"]
Depth = 1
```

## Symmetries, freezing, waves, fronts

```@contents
Pages = ["autocatalytic.md"]
Depth = 1
```
