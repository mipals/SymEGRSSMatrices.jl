# Removing t = 0, such that Σ is invertible
t = Vector(0.1:0.1:1); p = 2;

# Creating generators U,V that result in a positive-definite matrix Σ
U, V = SymEGRSSMatrices.spline_kernel(t, p)

K = SymEGRQSMatrix(U,V,ones(length(t)))
x = randn(length(t))
Kfull = Matrix(K)

# Testing multiplication
@test isapprox(K*x, Kfull*x, atol = 1e-6)

#  Testing multiplication with the adjoint operator
@test isapprox(K'*x, Kfull'*x, atol = 1e-6)

# Testing linear solve
@test isapprox(K\x, Kfull\x, atol=1e-6)

# Testing show
@test isapprox(Matrix(K), tril(K.U*K.V') + triu(K.V*K.U',1) + Diagonal(K.d))
