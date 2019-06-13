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
newtonFold(F, J, Jt, d2F, foldpointguess::BorderedArray{vectype, T}, eigenvec, options::NewtonPar; normN = norm, d2F_is_known = true ) where {T,vectype}
```

```@docs
newtonFold(F, J, Jt, d2F, br::ContResult, ind_fold::Int64, options::NewtonPar;kwargs...)
```

## Continuation

```@docs
continuation
```

## Continuation for Fold / Hopf

```@docs
continuationFold(F, J, Jt, d2F, foldpointguess::BorderedArray{vectype, T}, p2_0::T, eigenvec, options_cont::ContinuationPar ; d2F_is_known = true, kwargs...) where {T,vectype}
```

```@docs
continuationFold(F, J, Jt, foldpointguess::BorderedArray{vectype, T}, p2_0::T, eigenvec, options_cont::ContinuationPar ;kwargs...) where {T,vectype}
```

```@docs
continuationFold(F, J, Jt, d2F, br::ContResult, ind_fold::Int64, p2_0::Real, options_cont::ContinuationPar ; kwargs...)
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

