module SymEGRSSMatrices

# Must be imported here, as it is relevant before "syntax.jl"
import LinearAlgebra: inv!, tr, mul!, Adjoint, dot, Factorization
import Base: inv, size, eltype, getindex

#export SymEGRSSMatrix, SymEGRSSCholesky

# Properties
#include("adjointoperator.jl")

# Matrices
include("SymEGRSSMatrix.jl")
include("chol_SymEGRSSMatrix.jl")


#include("syntax.jl")



end # module
