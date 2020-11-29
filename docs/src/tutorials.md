# Tutorials

There are three levels of tutorials:

1. fully automatic bifurcation diagram (**aBD**) computation (only for equilibria): one uses the function `bifurcationdiagram` and let it compute the diagram fully automatically. Note that you may have to tune the options before hand. Another possibility is to use **deflated continuation**.
2. semi-automatic bifurcation diagram computation: one uses automatic branch switching (**aBS**) to compute branches at specified bifurcation points
3. manual bifurcation diagram computation: one does not uses automatic branch switching. This has only educational purposes and for complex problems where aBS fails.

## Bifurcation of Equilibria
```@contents
Pages = ["tutorials1.md", "tutorials2.md", "mittelmann.md", "tutorials1b.md","tutorials2b.md"]
Depth = 1
```

### Automatic bifurcation diagram
```@contents
Pages = ["Swift-Hohenberg1d.md", "mittelmannAuto.md", "tutorialCarrier.md"]
Depth = 1
```

### Solving PDEs using Finite elements with [Gridap.jl](https://github.com/gridap/Gridap.jl)
```@contents
Pages = ["mittelmannGridap.md"]
Depth = 1
```

## Bifurcation diagrams with periodic orbits
```@contents
Pages = ["tutorials3.md","tutorials3b.md", "BrusselatorFF.md", "tutorialsPD.md", "tutorialsCGL.md", "tutorialsCGLShoot.md"]
Depth = 1
```