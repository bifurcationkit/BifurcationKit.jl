# Deflated Continuation

!!! unknown "References"
    Farrell, Patrick E., Casper H. L. Beentjes, and Ásgeir Birkisson. **The Computation of Disconnected Bifurcation Diagrams.** ArXiv:1603.00809 [Math], March 2, 2016. http://arxiv.org/abs/1603.00809.

Deflated continuation allows to compute branches of solutions to the equation $F(x,p)=0$. It is based on the Deflated Newton (see [Deflated problems](@docs)).

However, unlike the regular continuation method, deflated continuation allows to compute **disconnected** bifurcation diagrams, something that is impossible to our Automatic Bifurcation diagram computation method.

You can find an example of use in the [Deflated Continuation in the Carrier Problem](@ref). We reproduce below the result of the computation:

![](carrier.png)

## Algorithm

```
Input: Initial parameter value λmin.
Input: Final parameter value λmax > λmin. Input: Step size ∆λ > 0.
Input: Nonlinear residual f(u,λ).
Input: Deflation operator M(u; u∗).
Input: Initial solutions S(λmin) to f(·,λmin).
λ ← λmin
while λ < λmax do
	F(·) ← f(·,λ+∆λ) ◃ Fix the value of λ to solve for.
	S(λ+∆λ) ← ∅
	for u0 ∈ S(λ) do ◃ Continue known branches.
	apply Newton’s method to F from initial guess u0.
	if solution u∗ found then
		S(λ + ∆λ) ← S(λ + ∆λ) ∪ {u∗} ◃ Record success
		F(·) ← M(·;u∗)F(·)		◃ Deflate solution
		
	for u0 ∈ S(λ) do 	◃ Seek new branches.
		success ← true 
		while success do
			apply Newton’s method to F from initial guess u0.
			if solution u∗ found then
				S(λ + ∆λ) ← S(λ + ∆λ) ∪ {u∗} ◃ Record success
				F(·) ← M(·;u∗)F(·)		◃ Deflate solution
		else
			success ← false 
	λ←λ+∆λ
return S
```