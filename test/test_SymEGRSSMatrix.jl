t = Vector(0.1:0.1:1)
n = length(t); p = 2;
U, V = SymEGRSSMatrices.spline_kernel(t, p);

K = SymEGRSSMatrix(U,V)
x = randn(size(K,1));
Kfull = Matrix(K);

# Testing multiplication
@test isapprox(K*x, Kfull*x, atol = 1e-6)
@test isapprox(K'*x, Kfull'*x, atol = 1e-6)

# Testing linear solves
@test isapprox(K\x, Kfull\x,atol=1e-6)

# Testing (log)determinant
@test isapprox(logdet(K), logdet(Kfull), atol=1e-6)
@test isapprox(det(K), det(Kfull), atol=1e-6)

# Testing show
@test isapprox(Matrix(K), tril(K.U*K.V') + triu(K.V*K.U',1))
@test isapprox(Kfull[3,1], K[3,1])
@test isapprox(Kfull[2,2], K[2,2])
@test isapprox(Kfull[1,3], K[1,3])
