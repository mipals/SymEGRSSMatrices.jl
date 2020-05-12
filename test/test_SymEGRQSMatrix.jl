n = 100;
p = 5;
U = randn(n,p);
V = randn(n,p);


K = SymEGRQSMatrix(U,V,ones(n));
x = randn(n);
Kfull = Matrix(K);

# Testing multiplication
@test isapprox(K*x, Kfull*x, atol = 1e-6)

#  Testing multiplication with the adjoint operator
@test isapprox(K'*x, Kfull'*x, atol = 1e-6)
