
using LinearAlgebra
import LinearAlgebra: cholesky,inv!, tr, mul!, Adjoint, det, logdet, dot,
                  Factorization,tril,triu, ldiv!
import Base: inv, size, eltype, getindex, getproperty


# Matrices
include("SymEGRSSMatrix.jl")
include("SymEGRSSCholesky.jl")
include("SymEGRQSMatrix.jl")
include("SymEGRQSCholesky.jl")





N = 100; p = 2;
M = SymEGRSSMatrix(rand(N,p),randn(N,p));
M*ones(M.n,2)
M'*ones(M.n,2)





include("spline_kernel.jl")
t = Vector(0.1:0.1:1); p = 2;
xt = ones(length(t))
# Creating generators U,V that result in a positive-definite matrix Σ
U, V = spline_kernel(t, p);
Σ    = spline_kernel_matrix(U, V);
chol = cholesky(Σ)

# Creating a symmetric extended generator representable semiseperable matrix
K = SymEGRSSMatrix(U,V)
# Calculating its Cholesky factorization
L = cholesky(K)



Kq = SymEGRQSMatrix(U,V,ones(length(t)));
Kq*xt
Kq'*xt
Lq = cholesky(Kq);

Kq*(Lq\xt)
