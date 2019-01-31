# Library

## Structs

```@docs
PeriodicOrbitTrap
```

## Newton

```@docs
newton
```


```@docs
newtonDeflated
```

## Newton for Fold / Hopf
```@docs
newtonFold(F, J, Jt, d2F, foldpointguess::Union{Vector, BorderedVector{vectype, T}}, eigenvec, options::NewtonPar; normN = norm) where {T,vectype}
```


```@docs
newtonFold(F::Function, J, Jt, foldpointguess::AbstractVector, eigenvec::AbstractVector, options::NewtonPar)
```

```@docs
newtonFold(F::Function, J, Jt, br::ContResult, ind_fold::Int64, options::NewtonPar)
```

```@docs
newtonHopf(F::Function, J, Jt, hopfpointguess::AbstractVector, eigenvec, eigenvec_ad, options::NewtonPar)
```

```@docs
newtonHopf(F, J, Jt, br::ContResult, ind_hopf::Int64, options::NewtonPar)
```

## Continuation

```@docs
continuation
```

## Continuation for Fold / Hopf

```@docs
continuationFold(F::Function, J, Jt, foldpointguess::AbstractVector, p2_0::Real, eigenvec::AbstractVector, options_cont::ContinuationPar ; kwargs...)
```

```@docs
continuationFold(F::Function, J, Jt, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)
```

```@docs
continuationHopf(F::Function, J, Jt, hopfpointguess::AbstractVector, p2_0, eigenvec, eigenvec_ad, options_cont::ContinuationPar ; kwargs...)
```

```@docs
continuationHopf(F::Function, J, Jt, br::ContResult, ind_hopf::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)
```

## Plotting

```@docs
plotBranch(contres::ContResult; kwargs...)
```


```@docs
plotBranch!(contres::ContResult; kwargs...)
```

```@docs
plotBranch(brs::Vector; kwargs...)
```

