struct SymEGRQSCholesky{T,UT<:AbstractMatrix,WT<:AbstractMatrix,dT<:AbstractArray} <: AbstractMatrix{T}
	Ut::UT
    Wt::WT
	d::dT
    n::Int
    p::Int
    function SymEGRQSCholesky{T,UT,WT,dT}(Ut,Wt,d,n,p) where
		{T,UT<:AbstractMatrix,WT<:AbstractMatrix,dT<:AbstractArray}
		Up, Un = size(Ut)
		Wp, Wn = size(Wt)
		(Up == Wp && Un == Wn && Un == length(d)) || throw(DimensionMismatch())
        new(Ut,Wt,d,n,p)
    end
end

#### Creating W and dbar s.t. L = tril(UW',-1) + diag(dbar) ####
function dss_create_wdbar(Ut::AbstractArray, Vt::AbstractArray, d::AbstractArray)
    p, n = size(Ut);
    P  = zeros(p, p);
    Wt  = similar(Vt);
    dbar = zeros(n);
    @inbounds for i = 1:n
        tmpU  = Ut[:,i]
        tmpW  = Vt[:,i] - P*tmpU;
		tmpds = sqrt(abs(tmpU'*tmpW + d[i]));
        #tmpds = sqrt(tmpU'*tmpW + d[i]);
        tmpW  = tmpW/tmpds
        Wt[:,i] = tmpW;
        dbar[i] = tmpds;
        P += tmpW*tmpW';
    end
    return Wt, dbar
end

function cholesky(K::SymEGRQSMatrix{T,UT,VT,dT}) where
	{T,UT<:AbstractMatrix,VT<:AbstractMatrix,dT<:AbstractArray}
	Wt, dbar = dss_create_wdbar(K.Ut,K.Vt,K.d)
	SymEGRQSCholesky{T,typeof(K.Ut),typeof(Wt),typeof(dbar)}(K.Ut,Wt,dbar,K.n,K.p)
end

function cholesky(K::SymEGRSSMatrix{T,UT,VT}, σ::Number) where
	{T,UT<:AbstractMatrix,VT<:AbstractMatrix}
	Wt, dbar = dss_create_wdbar(K.Ut,K.Vt,ones(K.n)*σ)
	SymEGRQSCholesky{T,typeof(K.Ut),typeof(Wt),typeof(dbar)}(K.Ut,Wt,dbar,K.n,K.p)
end

function SymEGRQSCholesky(K::SymEGRSSMatrix{T,UT,VT}, σ::Number) where
	{T,UT<:AbstractMatrix,VT<:AbstractMatrix}
	Wt, dbar = dss_create_wdbar(K.V,K.V,ones(K.n)*σ)
	SymEGRQSCholesky{T,typeof(K.Ut),typeof(Wt),typeof(dbar)}(K.Ut,Wt,dbar,K.n,K.p)
end

function SymEGRQSCholesky(Ut::UT, Wt::WT, d::AbstractArray) where
	{T,UT<:AbstractMatrix,WT<:AbstractMatrix}
	p, n = size(Ut);
	SymEGRQSCholesky{eltype(Ut),typeof(Ut),typeof(Wt),typeof(d)}(Ut,Wt,d,n,p)
end


########################################################################
#### Helpful properties. Not nessecarily computionally efficient    ####
########################################################################
Matrix(K::SymEGRQSCholesky) = getproperty(K,:L)
size(K::SymEGRQSCholesky) = (K.n, K.n)
size(K::SymEGRQSCholesky,d::Int) = (1 <= d && d <=2) ? size(K)[d] : throw(ArgumentError("Invalid dimension $d"))

function getindex(K::SymEGRQSCholesky, i::Int, j::Int)
	i > j && return dot(K.Ut[:,i], K.Wt[:,j])
	i == j && return K.d[i]
	return 0
end

function getproperty(K::SymEGRQSCholesky, d::Symbol)
    if d === :U
        return UpperTriangular(triu(K.Wt'*K.Ut,1) + Diagonal(K.d))
    elseif d === :L
        return LowerTriangular(tril(K.Ut'*K.Wt,-1) + Diagonal(K.d))
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
function dss_forward!(X::AbstractArray,Ut::AbstractArray,Wt::AbstractArray,
                     ds::AbstractArray,B::AbstractArray)
    p, n = size(Ut)
    mx = size(B,2)
    Wbar = zeros(p,mx);
    @inbounds for i = 1:n
        tmpU = Ut[:,i];
        tmpW = Wt[:,i];
        X[i:i,:] = (B[i:i,:] - tmpU'*Wbar)/ds[i];
        Wbar += tmpW .* X[i:i,:];
    end
end
#### Backward substitution ####
function dssa_backward!(X::AbstractArray,Ut::AbstractArray,Wt::AbstractArray,
                       ds::AbstractArray,B::AbstractArray)
    p, n = size(Ut)
    mx = size(B,2)
    Ubar = zeros(p,mx);
    @inbounds for i = n:-1:1
        tmpU = Ut[:,i];
        tmpW = Wt[:,i];
        X[i:i,:] = (B[i:i,:] - tmpW'*Ubar)/ds[i];
        Ubar += tmpU .* X[i:i,:];
    end
end

#### Multiplying with L ####
function dss_tri_mul!(Y::AbstractArray,K::SymEGRQSCholesky,X::AbstractArray)
     p, n = size(K.Ut)
     mx = size(X, 2)
     Wbar = zeros(p, mx)
     @inbounds for i = 1:n
         tmpW = K.Wt[:,i]
         tmpU = K.Ut[:,i]
         tmpX = X[i:i,:];
         Y[i,:] = tmpU'*Wbar + K.d[i]*tmpX;
         Wbar  +=  tmpW*tmpX;
     end
	 return Y
end

#### Multiplying with L' ####
function dssa_tri_mul!(Y::AbstractArray,K::SymEGRQSCholesky,X::AbstractArray)
     p, n = size(K.Ut)
     mx = size(X, 2)
     Ubar = zeros(p,mx)
     @inbounds for i = n:-1:1
         tmpW = K.Wt[:,i]
         tmpU = K.Ut[:,i]
         tmpX = X[i:i,:]
         Y[i,:] = tmpW'*Ubar + K.d[i]*tmpX;
         Ubar = Ubar + tmpU*tmpX;
     end
	 return Y
end


#### Squared norm of columns of L = tril(UW',-1) + diag(dbar) ####
function squared_norm_cols(Ut::AbstractArray,Wt::AbstractArray,
                         dbar::AbstractArray)
    p, n = size(Ut)
    P = zeros(p, p)
    c = zeros(n)
    @inbounds for i = n:-1:1
        tmpW = Wt[:,i]
        tmpU = Ut[:,i]
        c[i] = dbar[i]^2 + tmpW'*P*tmpW
        P += tmpU*tmpU'
    end
    return c
end
#### Implicit inverse of  L = tril(UW',-1) + diag(dbar) ####
function dss_create_yz(Ut::AbstractArray, Wt::AbstractArray,
                     dbar::AbstractArray)
    p, n = size(Ut)
    Y = zeros(n,p)
    Z = zeros(n,p)
    dss_forward!(Y, Ut, Wt, dbar, Ut')
    dssa_backward!(Z, Ut, Wt, dbar, Wt')
    # Probably best not to use inv
    return copy(Y'), copy((Z*inv(Ut*Z - Diagonal(ones(p))))')
	#return Y, Z*((U'*Z - Diagonal(ones(m)))\Diagonal(ones(m)))
end

function fro_norm_L(L::SymEGRQSCholesky)
	return sum(squared_norm_cols(L.Ut, L.Wt, L.d))
end

function trinv(L::SymEGRQSCholesky)
	Yt, Zt = dss_create_yz(L.Ut, L.Wt, L.d)
	return sum(squared_norm_cols(Yt, Zt, L.d.^(-1)))
end


function tr(Ky::SymEGRQSCholesky, K::SymEGRSSMatrix)
	n = Ky.n;
	p = Ky.p;
	c = Ky.d;
	Ut = K.Ut;
	Vt = K.Vt;
	Yt, Zt = dss_create_yz(Ky.Ut, Ky.Wt, Ky.d);
	b = 0;
	P = zeros(p,p);
	R = zeros(p,p);
	@inbounds for k = 1:Ky.n
		yk = Yt[:,k];
		zk = Zt[:,k];
		uk = Ut[:,k];
		vk = Vt[:,k];
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
    return dd
end

#### Determinant ####
function det(K::SymEGRQSCholesky)
    return exp(logdet(K))
end


mul!(y::AbstractVecOrMat, K::SymEGRQSCholesky, x::AbstractVecOrMat) =
	dss_tri_mul!(y,K,x)
mul!(y::AbstractVecOrMat, K::Adjoint{<:Any,<:SymEGRQSCholesky}, x::AbstractVecOrMat) =
	dssa_tri_mul!(y,K.parent,x)
function (\)(F::SymEGRQSCholesky, B::AbstractVecOrMat)
	X = similar(B)
	dss_forward!(X,F.Ut,F.Wt,F.d,B)
	return X
end

function (\)(F::Adjoint{<:Any,<:SymEGRQSCholesky}, B::AbstractVecOrMat)
	Y = similar(B)
	dssa_backward!(Y,F.parent.Ut,F.parent.Wt,F.parent.d,B)
	return Y
end
