module SymEGRSSMatrices

import LinearAlgebra: cholesky,inv!, tr, mul!, Adjoint, det, logdet, dot,
                      Factorization, tril, triu, ldiv!,
                      LowerTriangular, UpperTriangular
import Base: \, inv, size, eltype, getindex, getproperty

export SymEGRSSMatrix, SymEGRSSCholesky,
       SymEGRQSMatrix, SymEGRQSCholesky

# Matrices
include("SymEGRSSMatrix.jl")
include("SymEGRSSCholesky.jl")
include("SymEGRQSMatrix.jl")
include("SymEGRQSCholesky.jl")

# Extra functions
include("spline_kernel.jl")



end # module
