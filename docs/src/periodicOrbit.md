# Periodic orbits computation

We provide two methods of computing periodic orbits:

* one based on finite differences to discretize a Cauchy problem
* one based on the flow associated to a Cauchy problem

It is important to understand the pro and cons of each method to compute periodic orbits in large dimensions.

The method based on finite differences are usually faster than the one based on Shooting but they require more memory as they save the whole orbit in memory. However the main drawback of this method is that the associated linear solver is not "nice", being composed of a cyclic matrix for which no generic Matrix-free preconditioner is known. Hence, this method is **often used with an ILU preconditioner** which is severely constrained by memory. Also, when the period of the cycle is large, finer time discretization must be employed which is limited by memory.

The method based on Shooting does not share the same drawbacks because the associated linear system is usually well conditioned, at least in the simple shooting case. It is thus often **used without preconditioner at all**. Even in the case of multiple shooting, this can be alleviated by a simple generic preconditioner based on deflation of eigenvalues (see [Linear solvers](@ref)). The main drawback of the method is to find a fast time stepper, at least to compete with the method based on finite differences. Finally, the time stepper will automatically adapt to the stiffness of the problem, putting more time points where needed unlike the method based on finite differences which requires an adaptive (time) meshing to provide a similar property. 
