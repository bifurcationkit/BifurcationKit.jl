# Periodic orbits computation

We provide two methods for computing periodic orbits (PO):

* one based on finite differences to discretize a Cauchy problem
* one based on the flow associated to a Cauchy problem

It is important to understand the pro and cons of each method to compute PO in large dimensions.

The methods based on finite differences are usually faster than the ones based on Shooting but they require more memory as they save the whole orbit. However the main drawback of this method is that the associated linear solver is not "nice", being composed of a cyclic matrix for which no generic Matrix-free preconditioner is known. Hence, this method is **often used with an ILU preconditioner** which is severely constrained by memory. Also, when the period of the cycle is large, finer time discretization (or mesh adaptation which is not yet implemented) must be employed which is also a limiting factor both in term of memory and preconditioning.

The methods based on Shooting do not share the same drawbacks because the associated linear system is usually well conditioned, at least in the simple shooting case. There are thus often used **without preconditioner at all**. Even in the case of multiple shooting, this can be alleviated by a simple generic preconditioner based on deflation of eigenvalues (see [Linear solvers (LS)](@ref)). Also, the time stepper will automatically adapt to the stiffness of the problem, putting more time points where needed unlike the method based on finite differences which requires an adaptive (time) meshing to provide a similar property. The main drawback of the method is to find a fast time stepper, at least to compete with the method based on finite differences.
