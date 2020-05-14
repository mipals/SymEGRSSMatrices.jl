t = Vector(0.1:0.1:10)
n = length(t); p = 2;
U, V = spline_kernel(t, p);

K = SymEGRSSMatrix(U,V)
x = randn(size(K,1));
Kfull = Matrix(K);

# Testing multiplication
@test K*x ≈ Kfull*x
@test K'*x ≈ Kfull'*x

# Testing linear solves
@test K\x ≈ Kfull\x

# Testing (log)determinant
@test logdet(K) ≈ logdet(Kfull)
@test det(K) ≈ det(Kfull)

# Testing show
@test Matrix(K) ≈ tril(K.U*K.V') + triu(K.V*K.U',1)
@test Kfull[3,1] ≈ K[3,1]
@test Kfull[2,2] ≈ K[2,2]
@test Kfull[1,3] ≈ K[1,3]
