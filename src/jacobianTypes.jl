abstract type AbstractJacobianType end
abstract type AbstractJacobianFree <: AbstractJacobianType end
abstract type AbstractJacobianMatrix <: AbstractJacobianType end
abstract type AbstractJacobianSparseMatrix <: AbstractJacobianMatrix end

# :FiniteDifferencesDense

"""
Singleton type to trigger the computation of the jacobian Matrix using ForwardDiff.jl. It can be used for example in newton or in deflated newton.
"""
struct AutoDiff <: AbstractJacobianType end

"""
Singleton type to trigger the computation of the jacobian vector product (jvp) using ForwardDiff.jl. It can be used for example in newton, in deflated newton or in continuation.
"""
struct AutoDiffMF <: AbstractJacobianFree end

"""
Singleton type to trigger the computation of the jacobian vector product (jvp) using finite differences. It can be used for example in newton or in deflated newton.
"""
struct FiniteDifferencesMF <: AbstractJacobianFree end

"""
Singleton type to trigger the computation of the jacobian using finite differences. It can be used for example in newton or in deflated newton.
"""
struct FiniteDifferences <: AbstractJacobianType end

"""
For periodic orbits. The jacobian is a sparse matrix which is expressed with a custom analytical formula.
"""
struct FullSparse <: AbstractJacobianType end

"""
Same as FullSparse but the Jacobian is allocated only once and updated inplace. This is much faster than FullSparse but the sparsity pattern of the vector field must be constant.
"""
struct FullSparseInplace <: AbstractJacobianType end
