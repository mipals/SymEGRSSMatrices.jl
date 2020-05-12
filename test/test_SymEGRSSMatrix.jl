t = Vector(0.1:0.1:1)
n = length(t); p = 2;
U, V = SymEGRSSMatrices.spline_kernel(t, p);

K = SymEGRSSMatrix(U,V)
x = randn(K.n);
Kfull = Matrix(K);

# Testing multiplication
@test isapprox(K*x, Kfull*x, atol = 1e-6)

# Testing multiplication with the adjoint operator
@test isapprox(K'*x, Kfull'*x, atol = 1e-6)

# Testing linear solves
@test isapprox(K\x, Kfull\x,atol=1e-6)

# Testing show
@test isapprox(Matrix(K), tril(K.U*K.V') + triu(K.V*K.U',1))
