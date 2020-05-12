using SymEGRSSMatrices
using LinearAlgebra





N = 100; p = 2;
M = SymEGRSSMatrix(rand(N,p),randn(N,p));
M*ones(M.n,2)
M'*ones(M.n,2)


t = Vector(0.1:0.1:1); p = 2;
xt = ones(length(t))
# Creating generators U,V that result in a positive-definite matrix Σ
U, V = SymEGRSSMatrices.spline_kernel(t, p);
# Creating a symmetric extended generator representable semiseperable matrix
K = SymEGRSSMatrix(U,V)
# Creating dense replicas
Σ    = Matrix(K);
chol = cholesky(Σ);

# Calculating its Cholesky factorization
L = cholesky(K)



Kq = SymEGRQSMatrix(U,V,ones(length(t)));
Σq = Matrix(Kq)


Kq*xt
Kq'*xt
Lq = cholesky(Kq);

Kq*(Lq\xt)
