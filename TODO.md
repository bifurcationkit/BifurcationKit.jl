# TODO List - BifurcationKit.jl

> Généré automatiquement à partir des commentaires `TODO` dans `src/`

| # | Fichier | Ligne | TODO |
|---|---------|-------|------|
| 1 | `src/ContKind.jl` | 6 | `rename abstract` |
| 2 | `src/NormalForms.jl` | 306 | `this line makes it type unstable` |
| 3 | `src/NormalForms.jl` | 572 | `start at 2 or begin+1 ??` |
| 4 | `src/Problems.jl` | 15 | `🚧 change name for jvp 🚧` |
| 5 | `src/Problems.jl` | 159 | `🚧 call it jvp ! 🚧` |
| 6 | `src/bifdiagram/BranchSwitching.jl` | 253 | `makes the function type unstable` |
| 7 | `src/bifdiagram/BranchSwitching.jl` | 372 | `makes it type unstable` |
| 8 | `src/bifdiagram/BranchSwitching.jl` | 428 | `should be a Branch of vectors not a vector of Branches` |
| 9 | `src/bifdiagram/BifurcationDiagram.jl` | 128 | `BifDiagNode[] makes it type unstable` |
| 10 | `src/bvp/BVPBifProblem.jl` | 147 | `remove this hack` |
| 11 | `src/bvp/BVPBifProblem.jl` | 162 | `not sure!!` |
| 12 | `src/bvp/BVPBifProblem.jl` | 187 | `Remove?` |
| 13 | `src/bvp/BVPModel.jl` | 8 | `we allow F(u,p,t)?` |
| 14 | `src/bvp/BVPModel.jl` | 115 | `use static test` |
| 15 | `src/bvp/BVPModel.jl` | 126 | `should remove` |
| 16 | `src/bvp/discretize.jl` | 143 | `je pense jamais appele` |
| 17 | `src/bvp/discretize.jl` | 169 | `je pense jamais appele` |
| 18 | `src/bvp/discretize.jl` | 182 | `je pense jamais appele` |
| 19 | `src/codim2/MinAugFold.jl` | 110 | `we know u2!!` |
| 20 | `src/codim2/MinAugFold.jl` | 140 | `we already know u2!!` |
| 21 | `src/codim2/MinAugFold.jl` | 410 | `change the name RecordForFold` |
| 22 | `src/codim2/MinAugFold.jl` | 434 | `TYPE UNSTABLE` |
| 23 | `src/codim2/MinAugFold.jl` | 439 | `remove this hack` |
| 24 | `src/codim2/MinAugFold.jl` | 506 | `use _compute_bordered_vectors !!!!` |
| 25 | `src/codim2/MinAugHopf.jl` | 97 | `This is only finite differences` |
| 26 | `src/codim2/MinAugHopf.jl` | 123 | `this is R20` |
| 27 | `src/codim2/MinAugHopf.jl` | 202 | `This seems TU` |
| 28 | `src/codim2/MinAugHopf.jl` | 627 | `WE NEED A KWARGS here` |
| 29 | `src/codim2/NormalForms.jl` | 186 | `THIS MAKES IT TYPE UNSTABLE` |
| 30 | `src/codim2/NormalForms.jl` | 1008 | `IMPROVE THIS` |
| 31 | `src/codim2/NormalForms.jl` | 1291 | `type unstable` |
| 32 | `src/codim2/NormalForms.jl` | 1356 | `IMPROVE THIS` |
| 33 | `src/codim2/codim2.jl` | 169 | `is it a copy or else?` |
| 34 | `src/continuation/MoorePenrose.jl` | 70 | `type unstable` |
| 35 | `src/periodicorbit/Floquet.jl` | 65 | `must not be computed, cf TRAP` |
| 36 | `src/periodicorbit/NormalForms.jl` | 604 | `use R01` |
| 37 | `src/periodicorbit/NormalForms.jl` | 647 | `extract from continuation_pd` |
| 38 | `src/periodicorbit/PeriodicOrbitCollocation.jl` | 79 | `this is strongly type unstable` |
| 39 | `src/periodicorbit/PeriodicOrbitCollocation.jl` | 1040 | `this is an abomination! coll is a discretization method, not a problem` |
| 40 | `src/periodicorbit/PeriodicOrbitCollocation.jl` | 1050 | `this is an abomination! coll is a discretization method, not a problem` |
| 41 | `src/periodicorbit/PeriodicOrbitCollocation.jl` | 1070 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| 42 | `src/periodicorbit/PeriodicOrbitTrapeze.jl` | 248 | `this line does not almost seem to be type stable` |
| 43 | `src/periodicorbit/PeriodicOrbitTrapeze.jl` | 784 | `remove this or improve!!` |
| 44 | `src/periodicorbit/PeriodicOrbitTrapeze.jl` | 785 | `REMOVE vcat!!` |
| 45 | `src/periodicorbit/PeriodicOrbitTrapeze.jl` | 916 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| 46 | `src/periodicorbit/PeriodicOrbitTrapeze.jl` | 926 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| 47 | `src/periodicorbit/PeriodicOrbitTrapeze.jl` | 1041 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| 48 | `src/periodicorbit/PeriodicOrbitUtils.jl` | 50 | `Make small function for this and merge with the one in MinAugFold.jl` |
| 49 | `src/periodicorbit/PeriodicOrbits.jl` | 60 | `REMOVE?` |
| 50 | `src/periodicorbit/PeriodicOrbits.jl` | 248 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| 51 | `src/periodicorbit/PeriodicOrbits.jl` | 301 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| 52 | `src/periodicorbit/PeriodicOrbits.jl` | 401 | `improve type stability` |
| 53 | `src/periodicorbit/PeriodicOrbits.jl` | 443 | `Not good: we cannot change lens` |
| 54 | `src/periodicorbit/PeriodicOrbits.jl` | 472 | `should only update guess here, cf Poincaré` |
| 55 | `src/periodicorbit/PeriodicOrbits.jl` | 549 | `Use Minimally Augmented system for this instead of re-computing all eigenvalues` |
| 56 | `src/periodicorbit/PeriodicOrbits.jl` | 672 | `This should not be! Call it po_from_disc_newton ?` |
| 57 | `src/periodicorbit/PoincareRM.jl` | 122 | `needs a jacobian` |
| 58 | `src/periodicorbit/PoincareShooting.jl` | 224 | `declaration of xc allocates. It would be better to make it inplace` |
| 59 | `src/periodicorbit/PoincareShooting.jl` | 231 | `create the projections on the fly` |
| 60 | `src/periodicorbit/PoincareShooting.jl` | 333 | `declaration of xc allocates. It would be better to make it inplace` |
| 61 | `src/periodicorbit/PoincareShooting.jl` | 337 | `create the projections on the fly` |
| 62 | `src/periodicorbit/StandardShooting.jl` | 234 | `user defined scaleζ` |
| 63 | `src/periodicorbit/StandardShooting.jl` | 353 | `it breaks for VoA! use getindex?` |
| 64 | `src/periodicorbit/codim2/MinAugNS.jl` | 101 | `This is only finite differences` |
| 65 | `src/periodicorbit/codim2/MinAugNS.jl` | 370 | `fix this` |
| 66 | `src/periodicorbit/codim2/MinAugPD.jl` | 17 | `THIS CASE IS NOT REALLY USED` |
| 67 | `src/periodicorbit/codim2/MinAugPD.jl` | 128 | `This is only finite differences` |
| 68 | `src/periodicorbit/codim2/MinAugPD.jl` | 133 | `a bit of a hack` |
| 69 | `src/periodicorbit/codim2/MinAugPD.jl` | 422 | `what is this hack??` |
| 70 | `src/periodicorbit/codim2/PeriodicOrbitCollocation.jl` | 79 | `this is strongly type unstable` |
| 71 | `src/periodicorbit/codim2/codim2.jl` | 297 | `this does not work for matrix free or Shooting?` |
| 72 | `src/periodicorbit/codim2/codim2.jl` | 358 | `improve the following` |
| 73 | `src/periodicorbit/cop.jl` | 224 | `REMOVE` |

## Résumé par catégorie

### Type instabilité (16 TODOs)
- `src/codim2/MinAugFold.jl:434`
- `src/codim2/NormalForms.jl:186`
- `src/codim2/NormalForms.jl:1291`
- `src/continuation/MoorePenrose.jl:70`
- `src/periodicorbit/NormalForms.jl:604`
- `src/periodicorbit/PeriodicOrbitCollocation.jl:79`
- `src/periodicorbit/PeriodicOrbitCollocation.jl:1040`
- `src/periodicorbit/PeriodicOrbitCollocation.jl:1050`
- `src/periodicorbit/PeriodicOrbitTrapeze.jl:248`
- `src/periodicorbit/PeriodicOrbits.jl:401`
- `src/bifdiagram/BranchSwitching.jl:253`
- `src/bifdiagram/BranchSwitching.jl:372`
- `src/bifdiagram/BifurcationDiagram.jl:128`
- `src/NormalForms.jl:306`
- `src/periodicorbit/StandardShooting.jl:353`

### Hack / à supprimer (14 TODOs)
- `src/bvp/BVPBifProblem.jl:147`
- `src/bvp/BVPBifProblem.jl:187`
- `src/bvp/BVPModel.jl:126`
- `src/codim2/MinAugFold.jl:439`
- `src/periodicorbit/PeriodicOrbitTrapeze.jl:785`
- `src/periodicorbit/PeriodicOrbits.jl:60`
- `src/periodicorbit/PeriodicOrbits.jl:248`
- `src/periodicorbit/PeriodicOrbits.jl:301`
- `src/periodicorbit/PeriodicOrbitTrapeze.jl:916`
- `src/periodicorbit/PeriodicOrbitTrapeze.jl:926`
- `src/periodicorbit/PeriodicOrbitTrapeze.jl:1041`
- `src/periodicorbit/cop.jl:224`
- `src/bvp/discretize.jl:143`
- `src/bvp/discretize.jl:169`
- `src/bvp/discretize.jl:182`

### Refactoring / amélioration (13 TODOs)
- `src/codim2/MinAugFold.jl:410`
- `src/codim2/MinAugFold.jl:506`
- `src/periodicorbit/PeriodicOrbitUtils.jl:50`
- `src/periodicorbit/PeriodicOrbits.jl:472`
- `src/periodicorbit/PeriodicOrbits.jl:549`
- `src/periodicorbit/PeriodicOrbits.jl:672`
- `src/periodicorbit/PeriodicOrbitTrapeze.jl:784`
- `src/periodicorbit/codim2/codim2.jl:358`
- `src/periodicorbit/codim2/MinAugPD.jl:133`
- `src/periodicorbit/codim2/MinAugPD.jl:422`
- `src/NormalForms.jl:572`
- `src/codim2/codim2.jl:169`
- `src/bvp/BVPModel.jl:8`

### Finite differences (4 TODOs)
- `src/codim2/MinAugHopf.jl:97`
- `src/periodicorbit/codim2/MinAugNS.jl:101`
- `src/periodicorbit/codim2/MinAugPD.jl:128`

### Hack Functional vs Collocation (6 TODOs)
- `src/periodicorbit/PeriodicOrbits.jl:248`
- `src/periodicorbit/PeriodicOrbits.jl:301`
- `src/periodicorbit/PeriodicOrbitTrapeze.jl:916`
- `src/periodicorbit/PeriodicOrbitTrapeze.jl:926`
- `src/periodicorbit/PeriodicOrbitTrapeze.jl:1041`
- `src/periodicorbit/PeriodicOrbitCollocation.jl:1070`

### Allocation / performance (3 TODOs)
- `src/periodicorbit/PoincareShooting.jl:224`
- `src/periodicorbit/PoincareShooting.jl:333`

### À vérifier / question ouverte (14 TODOs)
- `src/codim2/MinAugHopf.jl:123`
- `src/codim2/MinAugHopf.jl:202`
- `src/codim2/MinAugHopf.jl:627`
- `src/codim2/MinAugFold.jl:110`
- `src/codim2/MinAugFold.jl:140`
- `src/periodicorbit/Floquet.jl:65`
- `src/periodicorbit/PeriodicOrbits.jl:443`
- `src/periodicorbit/PoincareRM.jl:122`
- `src/periodicorbit/PoincareShooting.jl:231`
- `src/periodicorbit/PoincareShooting.jl:337`
- `src/periodicorbit/StandardShooting.jl:234`
- `src/periodicorbit/codim2/MinAugNS.jl:370`
- `src/periodicorbit/codim2/MinAugPD.jl:17`
- `src/periodicorbit/codim2/codim2.jl:297`
- `src/bifdiagram/BranchSwitching.jl:428`
- `src/bvp/BVPBifProblem.jl:162`
- `src/ContKind.jl:6`

---
**Total: 73 TODOs**
