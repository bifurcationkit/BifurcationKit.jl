# Langmuir–Blodgett transfer model (really advanced)

!!! warning "Advanced"
    This is by far the most advanced example in the package. It uses all functionalities to their best. For example, the computation of periodic orbits with Finite Differences uses an inplace modification of the jacobian which allow to have quite a fine time discretization. The Shooting methods rely on parallel shooting with preconditioner and highly tuned ODE time stepper.
    
!!! warn "Reference"
    The tutorial is inspired by Köpf and Thiele, **Emergence of the Bifurcation Structure of a Langmuir–Blodgett Transfer Model.**    
    
In this tutorial, we try to replicate some of the results of the above quoted paper. This example is quite a marvel in the realm of bifurcation analysis, featuring a harp-like bifurcation diagram.

$$\partial_{t} c=-\partial_{x}^{2}\left[\partial_{x}^{2} c-c^{3}+c-\mu \zeta(x)\right]-V \partial_{x} c$$

with boundary condition

$$c(0)=c_{0}, \quad \partial_{x x} c(0)=\partial_{x} c(L)=\partial_{x x} c(L)=0$$

and 

$$\zeta(x)=-\frac{1}{2}\left[1+\tanh \left(\frac{x-x_{s}}{l_{s}}\right)\right]$$

As can be seen in the reference above, the bifurcation diagram is significantly more involved as $L$ increases. So we set up for the "simple" case $L=50$.