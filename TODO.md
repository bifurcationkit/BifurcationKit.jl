# TODO List - BifurcationKit.jl

> Généré automatiquement à partir des commentaires `TODO` dans `src/`

| Fichier | Ligne | TODO |
|---------|-------|------|
| `src/ContKind.jl` | 6 | `rename abstract` |
| `src/NormalForms.jl` | 306 | `this line makes it type unstable` |
| `src/NormalForms.jl` | 572 | `start at 2 or begin+1 ??` |
| `src/Problems.jl` | 15 | `change name for jvp` |
| `src/Problems.jl` | 159 | `call it jvp !` |
| `src/bifdiagram/BranchSwitching.jl` | 253 | `makes the function type unstable` |
| `src/bifdiagram/BranchSwitching.jl` | 372 | `makes it type unstable` |
| `src/bifdiagram/BranchSwitching.jl` | 428 | `should be a Branch of vectors not a vector of Branches` |
| `src/bifdiagram/BifurcationDiagram.jl` | 128 | `BifDiagNode[] makes it type unstable` |
| `src/bvp/BVPBifProblem.jl` | 147 | `remove this hack` |
| `src/bvp/BVPBifProblem.jl` | 162 | `not sure!!` |
| `src/bvp/BVPBifProblem.jl` | 187 | `Remove?` |
| `src/bvp/BVPModel.jl` | 8 | `we allow F(u,p,t)?` |
| `src/bvp/BVPModel.jl` | 115 | `use static test` |
| `src/bvp/BVPModel.jl` | 126 | `should remove` |
| `src/codim2/MinAugFold.jl` | 110 | `we know u2!!` |
| `src/codim2/MinAugFold.jl` | 140 | `we already know u2!!` |
| `src/codim2/MinAugFold.jl` | 410 | `change the name RecordForFold` |
| `src/codim2/MinAugFold.jl` | 434 | `TYPE UNSTABLE` |
| `src/codim2/MinAugFold.jl` | 439 | `remove this hack` |
| `src/codim2/MinAugFold.jl` | 506 | `use _compute_bordered_vectors !!!!` |
| `src/codim2/MinAugHopf.jl` | 97 | `This is only finite differences` |
| `src/codim2/MinAugHopf.jl` | 123 | `this is R20` |
| `src/codim2/MinAugHopf.jl` | 202 | `This seems TU` |
| `src/codim2/MinAugHopf.jl` | 627 | `WE NEED A KWARGS here` |
| `src/codim2/NormalForms.jl` | 186 | `THIS MAKES IT TYPE UNSTABLE` |
| `src/codim2/NormalForms.jl` | 1008 | `IMPROVE THIS` |
| `src/codim2/NormalForms.jl` | 1291 | `type unstable` |
| `src/codim2/NormalForms.jl` | 1356 | `IMPROVE THIS` |
| `src/codim2/codim2.jl` | 169 | `is it a copy or else?` |
| `src/continuation/MoorePenrose.jl` | 70 | `type unstable` |
| `src/periodicorbit/Floquet.jl` | 65 | `must not be computed, cf TRAP` |
| `src/periodicorbit/NormalForms.jl` | 604 | `use R01` |
| `src/periodicorbit/NormalForms.jl` | 647 | `extract from continuation_pd` |
| `src/periodicorbit/PeriodicOrbitCollocation.jl` | 79 | `this is strongly type unstable` |
| `src/periodicorbit/PeriodicOrbitCollocation.jl` | 1040 | `this is an abomination! coll is a discretization method, not a problem` |
| `src/periodicorbit/PeriodicOrbitCollocation.jl` | 1050 | `this is an abomination! coll is a discretization method, not a problem` |
| `src/periodicorbit/PeriodicOrbitCollocation.jl` | 1070 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| `src/periodicorbit/PeriodicOrbitTrapeze.jl` | 784 | `remove this or improve!!` |
| `src/periodicorbit/PeriodicOrbitTrapeze.jl` | 785 | `REMOVE vcat!!` |
| `src/periodicorbit/PeriodicOrbitTrapeze.jl` | 916 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| `src/periodicorbit/PeriodicOrbitTrapeze.jl` | 926 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| `src/periodicorbit/PeriodicOrbitTrapeze.jl` | 1041 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| `src/periodicorbit/PeriodicOrbitUtils.jl` | 50 | `Make small function for this and merge with the one in MinAugFold.jl` |
| `src/periodicorbit/PeriodicOrbits.jl` | 60 | `REMOVE?` |
| `src/periodicorbit/PeriodicOrbits.jl` | 248 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| `src/periodicorbit/PeriodicOrbits.jl` | 301 | `This is a bit of a hack. It should be a Functional not a discretization like Collocation` |
| `src/periodicorbit/PeriodicOrbits.jl` | 401 | `improve type stability` |
| `src/periodicorbit/PeriodicOrbits.jl` | 443 | `Not good: we cannot change lens` |
| `src/periodicorbit/PeriodicOrbits.jl` | 472 | `should only update guess here, cf Poincaré` |
| `src/periodicorbit/PeriodicOrbits.jl` | 549 | `Use Minimally Augmented system for this instead of re-computing all eigenvalues` |
| `src/periodicorbit/PeriodicOrbits.jl` | 672 | `This should not be! Call it po_from_disc_newton ?` |
| `src/periodicorbit/PoincareRM.jl` | 122 | `needs a jacobian` |
| `src/periodicorbit/PoincareShooting.jl` | 224 | `declaration of xc allocates. It would be better to make it inplace` |
| `src/periodicorbit/PoincareShooting.jl` | 231 | `create the projections on the fly` |
| `src/periodicorbit/PoincareShooting.jl` | 333 | `declaration of xc allocates. It would be better to make it inplace` |
| `src/periodicorbit/PoincareShooting.jl` | 337 | `create the projections on the fly` |
| `src/periodicorbit/StandardShooting.jl` | 234 | `user defined scaleζ` |
| `src/periodicorbit/StandardShooting.jl` | 353 | `it breaks for VoA! use getindex?` |
| `src/periodicorbit/codim2/MinAugNS.jl` | 101 | `This is only finite differences` |
| `src/periodicorbit/codim2/MinAugNS.jl` | 370 | `fix this` |
| `src/periodicorbit/codim2/MinAugPD.jl` | 17 | `THIS CASE IS NOT REALLY USED` |
| `src/periodicorbit/codim2/MinAugPD.jl` | 128 | `This is only finite differences` |
| `src/periodicorbit/codim2/MinAugPD.jl` | 133 | `a bit of a hack` |
| `src/periodicorbit/codim2/MinAugPD.jl` | 422 | `what is this hack??` |
| `src/periodicorbit/codim2/PeriodicOrbitCollocation.jl` | 79 | `this is strongly type unstable` |
| `src/periodicorbit/codim2/codim2.jl` | 297 | `this does not work for matrix free or Shooting?` |
| `src/periodicorbit/codim2/codim2.jl` | 358 | `improve the following` |
| `src/periodicorbit/cop.jl` | 224 | `REMOVE` |

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
