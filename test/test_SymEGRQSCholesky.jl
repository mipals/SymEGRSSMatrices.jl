# Removing t = 0, such that Σ is invertible
n = 500.0;
t = Vector(0.1:1/n:1)

# Creating a test matrix Σ = tril(UV') + triu(VU',1) that is PSD
p = 2;
Ut, Vt = spline_kernel(t', p)
K  = SymEGRQSMatrix(Ut,Vt,ones(size(Ut,2)))
Σ    = Matrix(K)
chol = cholesky(Σ)
L = cholesky(K)
x = randn(size(K,1))

# Testing inverses (Using Cholesky factorizations)
B = randn(length(t),10);
@test L\B  ≈ chol.L\B
@test L'\B ≈ chol.U\B
@test L*B  ≈ chol.L*B
@test L'*B ≈ chol.U*B

# Testing logdet
@test logdet(L) ≈ logdet(chol.L)
@test det(L) ≈ det(chol.L)

# Testing traces and norm
M = SymEGRSSMatrix(Ut,Vt);
@test isapprox(tr(L,M), tr(chol\Matrix(M)), atol=1e-6)
@test trinv(L) ≈ tr(chol\Diagonal(ones(size(L,1))))
@test SymEGRSSMatrices.fro_norm_L(L) ≈ norm(chol.L[:])^2

# Testing show
@test L.L ≈ tril(Ut'*L.Wt,-1) + Diagonal(L.d)
@test L.U ≈ triu(L.Wt'*Ut,1) + Diagonal(L.d)
@test Matrix(L) ≈ tril(L.Ut'*L.Wt,-1) + Diagonal(L.d)
@test L[3,1] ≈ chol.L[3,1]
@test L[2,2] ≈ chol.L[2,2]
@test L[1,3] ≈ chol.L[1,3]

# Testing explicit-implicit-inverse
Yt, Zt = SymEGRSSMatrices.dss_create_yz(L.Ut,L.Wt,L.d)
@test tril(Yt'*Zt,-1) + Diagonal(L.d.^(-1)) ≈ inv(chol.L)
@test L*(tril(Yt'*Zt,-1) + Diagonal(L.d.^(-1))) ≈ I
@test L'*(triu(Zt'*Yt,1) + Diagonal(L.d.^(-1))) ≈ I
@test SymEGRSSMatrices.squared_norm_cols(Yt,Zt,L.d.^(-1)) ≈ sum(inv(chol.L).^2,dims=1)'
