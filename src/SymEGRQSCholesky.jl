struct SymEGRQSCholesky{T,UT<:AbstractMatrix,WT<:AbstractMatrix,dT<:AbstractArray} <: Factorization{T}
    U::UT
    W::WT
	d::dT
    n::Int
    p::Int
    function SymEGRQSCholesky{T,UT,WT,dT}(U,W,d,n,p) where
		{T,UT<:AbstractMatrix,WT<:AbstractMatrix,dT<:AbstractArray}
		Un, Um = size(U)
		Wn, Wm = size(W)
		(Un == Wn && Um == Wm && Un == length(d)) || throw(DimensionMismatch())
        new(U,W,d,n,p)
    end
end

#### Creating W and dbar s.t. L = tril(UW',-1) + diag(dbar) ####
function dss_create_wdbar(U::AbstractArray, V::AbstractArray, d::AbstractArray)
    n, m = size(U);
    P  = zeros(m, m);
    W  = zeros(n, m);
    dbar = zeros(n);
    for i = 1:n
        tmpU  = U[i,:]
        tmpW  = V[i,:] - P*tmpU;
		tmpds = sqrt(abs(tmpU'*tmpW + d[i]));
        #tmpds = sqrt(tmpU'*tmpW + d[i]);
        tmpW  = tmpW/tmpds
        W[i,:] = tmpW;
        dbar[i] = tmpds;
        P += tmpW*tmpW';
    end
    return W, dbar
end

function cholesky(K::SymEGRQSMatrix{T,UT,VT,dT}) where
	{T,UT<:AbstractMatrix,VT<:AbstractMatrix,dT<:AbstractArray}
	W, dbar = dss_create_wdbar(K.U,K.V,K.d)
	SymEGRQSCholesky{T,typeof(K.U),typeof(W),typeof(dbar)}(K.U,W,dbar,K.n,K.p)
end


########################################################################
#### Helpful properties. Not nessecarily computionally efficient    ####
########################################################################
Matrix(K::SymEGRQSCholesky) = getproperty(K,:L)
size(K::SymEGRQSCholesky) = (K.n, K.n)
size(K::SymEGRQSCholesky,d::Int) = (1 <= d && d <=2) ? size(K)[d] : throw(DimensionMismatch())

function getindex(K::SymEGRQSCholesky, i::Int, j::Int)
	U = getfield(K,:U);
	W = getfield(K,:W);
	i > j && return dot(U[i,:], W[j,:])
	i == j && return K.d[i]
	return 0
end

function getproperty(K::SymEGRQSCholesky, d::Symbol)
    U = getfield(K, :U)
    W = getfield(K, :W)
	c = getfield(K, :d)
    if d === :U
        return UpperTriangular(triu(W*U',1) + Diagonal(c))
    elseif d === :L
        return LowerTriangular(tril(U*W',-1) + Diagonal(c))
    else
        return getfield(K, d)
    end
end

Base.propertynames(F::SymEGRQSCholesky, private::Bool=false) =
    (:U, :L, (private ? fieldnames(typeof(F)) : ())...)


function Base.show(io::IO, mime::MIME{Symbol("text/plain")},
		 K::SymEGRQSCholesky{<:Any,<:AbstractArray,<:AbstractArray,<:AbstractArray})
    summary(io, K); println(io)
    show(io, mime, K.L)
end

########################################################################
#### Cholesky factorization of Higher-order quasiseparable matrices ####
########################################################################

#### Forward substitution ####
function dss_forward!(X::AbstractArray,U::AbstractArray,W::AbstractArray,
                     ds::AbstractArray,B::AbstractArray)
    n, m = size(U)
    mx = size(B,2)
    Wbar = zeros(m,mx);
    @inbounds for i = 1:n
        tmpU = U[i,:];
        tmpW = W[i,:];
        X[i:i,:] = (B[i:i,:] - tmpU'*Wbar)/ds[i];
        Wbar += tmpW .* X[i:i,:];
    end
end
#### Backward substitution ####
function dssa_backward!(X::AbstractArray,U::AbstractArray,W::AbstractArray,
                       ds::AbstractArray,B::AbstractArray)
    n, m = size(U)
    mx = size(B,2)
    Ubar = zeros(m,mx);
    @inbounds for i = n:-1:1
        tmpU = U[i,:];
        tmpW = W[i,:];
        X[i:i,:] = (B[i:i,:] - tmpW'*Ubar)/ds[i];
        Ubar += tmpU .* X[i:i,:];
    end
end

function ldiv!(F::SymEGRQSCholesky, B::AbstractVecOrMat)
	X = similar(B)
	Y = similar(B)
	U = getfield(F,:U)
	W = getfield(F,:W)
	d = getfield(F,:d)
	dss_forward!(X,U,W,d,B)
	dssa_backward!(Y,U,W,d,X)
	return Y
end

#### Squared norm of columns of L = tril(UW',-1) + diag(dbar) ####
function squared_norm_cols(U::AbstractArray,W::AbstractArray,
                        dbar::AbstractArray)
    n, m = size(U)
    P = zeros(m, m)
    c = zeros(n)
    @inbounds for i = n:-1:1
        tmpW = W[i,:]
        tmpU = U[i,:]
        c[i]  = dbar[i]^2 + tmpW'*P*tmpW
        P += tmpU*tmpU'
    end
    return c
end
#### Implicit inverse of  L = tril(UW',-1) + diag(dbar) ####
function dss_create_yz(U::AbstractArray, W::AbstractArray,
                    dbar::AbstractArray)
    n, m = size(U)
    Y = zeros(n,m)
    Z = zeros(n,m)
    dss_forward!(Y, U, W, dbar, U)
    dssa_backward!(Z, U, W, dbar, W)
    # Probably best not to use inv
    return Y, Z*inv(U'*Z - Diagonal(ones(m)))
	#return Y, Z*((U'*Z - Diagonal(ones(m)))\Diagonal(ones(m)))
end

function fro_norm_L(L::SymEGRQSCholesky)
	return sum(squared_norm_cols(getfield(L,:U), getfield(L,:W), getfield(L,:d)))
end

function trinv(L::SymEGRQSCholesky)
	d = getfield(L,:d);
	Y, Z = dss_create_yz(getfield(L,:U), getfield(L,:W), d)
	return sum(squared_norm_cols(Y, Z, d.^(-1)))
end


function tr(Ky::SymEGRQSCholesky, K::SymEGRSSMatrix)
	n = Ky.n;
	p = Ky.p;
	c = Ky.d;
	U = K.U;
	V = K.V;
	Y, Z = dss_create_yz(getfield(Ky,:U), getfield(Ky,:W), getfield(Ky,:d));
	b = 0;
	P = zeros(p,p);
	R = zeros(p,p);
	@inbounds for k = 1:Ky.n
		yk = Y[k,:];
		zk = Z[k,:];
		uk = U[k,:];
		vk = V[k,:];
		cki = c[k]^(-1);
		b += yk'*P*yk + 2*yk'*R*uk*cki + uk'*vk*(cki^2);
		P += ((uk'*vk)*zk)*zk' + zk*(R*uk)' + (R*uk)*zk';
		R += zk*vk';
	end
	return b
end


#### Log-determinant ####
function logdet(K::SymEGRQSCholesky)
	dd = sum(log.(K.d))
    return dd + dd
end

#### Determinant ####
function det(K::SymEGRQSCholesky)
    return exp(logdet(K))
end
