struct SymEGRSSCholesky{T,UT<:AbstractMatrix,WT<:AbstractMatrix} <: Factorization{T}
    U::UT
    W::WT
    n::Int
    p::Int
    function SymEGRSSCholesky{T,UT,WT}(U,W,n,p) where
			 {T,UT<:AbstractMatrix,WT<:AbstractMatrix}
        Un, Um = size(U)
		Wn, Wm = size(W)
		(Un == Wn && Um == Wm) || throw(DimensionMismatch())
        new(U,W,n,p)
    end
end

# Creating generator such that L = tril(U*W')
function ss_create_w(U,V)
    n,p = size(U);
    wTw = zeros(p,p);
    W = similar(V);
    @inbounds for j = 1:n
        tmpu = U[j,:];
        tmp  = V[j,:] - wTw*tmpu;
        w = tmp/sqrt(abs(tmpu'*tmp))
        W[j,:] = w;
        wTw = wTw + w*w';
    end
    return W
end

cholesky(K::SymEGRSSMatrix{T,UT,VT}) where {T,UT<:AbstractMatrix,VT<:AbstractMatrix} =
        SymEGRSSCholesky{T,typeof(K.U),typeof(K.V)}(K.U,ss_create_w(K.U,K.V),K.n,K.p)


########################################################################
#### Helpful properties. Not nessecarily computionally efficient    ####
########################################################################
Matrix(K::SymEGRSSCholesky) = getproperty(K,:L)

size(K::SymEGRSSCholesky) = (K.n, K.n)

function getindex(K::SymEGRSSCholesky, i::Int, j::Int)
	U = getfield(K,:U);
	W = getfield(K,:W);
	i >= j && return dot(U[i,:], W[j,:])
	return 0
end

function getproperty(K::SymEGRSSCholesky, d::Symbol)
    U = getfield(K, :U)
    W = getfield(K, :W)
    if d === :U
        return UpperTriangular(W*U')
    elseif d === :L
        return LowerTriangular(U*W')
    else
        return getfield(K, d)
    end
end

Base.propertynames(F::SymEGRSSCholesky, private::Bool=false) =
    (:U, :L, (private ? fieldnames(typeof(F)) : ())...)


function Base.show(io::IO, mime::MIME{Symbol("text/plain")},
		 K::SymEGRSSCholesky{<:Any,<:AbstractMatrix,<:AbstractMatrix})
    summary(io, K); println(io)
    show(io, mime, K.L)
end


########################################################################
#### Solving systems using the Cholesky factorization    			####
########################################################################

#### Backward substitution (solve Lx = b) ####
function ss_forward!(X::AbstractArray, U::AbstractArray,
                     W::AbstractArray, B::AbstractArray)
    n, m = size(U)
    mx = size(B,2)
    Wbar = zeros(m, mx);
    for i = 1:n
        tmpU = U[i,:]
        tmpW = W[i,:]
        X[i,:] = (B[i:i,:] - tmpU'*Wbar)./(tmpU'*tmpW);
        Wbar += tmpW .* X[i:i,:];
    end
end

#### Backward substitution (solve L'x = b) ####
function ssa_backward!(X::AbstractArray, U::AbstractArray,
                       W::AbstractArray, B::AbstractArray)
    n, m = size(U)
    mx = size(B,2)
    Ubar = zeros(m,mx);
    for i = n:-1:1
        tmpU = U[i,:];
        tmpW = W[i,:];
        X[i,:] = (B[i:i,:] - tmpW'*Ubar)/(tmpU'*tmpW);
        Ubar += tmpU .* X[i:i,:];
    end
end

function ldiv!(F::SymEGRSSCholesky, B::AbstractVecOrMat)
	X = similar(B)
	Y = similar(B)
	U = getfield(F,:U)
	W = getfield(F,:W)
	ss_forward!(X,U,W,B)
	ssa_backward!(Y,U,W,X)
	return Y
end

########################################################################
#### Linear Algebra routines 							   			####
########################################################################
function det(L::SymEGRSSCholesky)
    dd = one(eltype(L))
	U = getfield(L, :U)
    W = getfield(L, :W)
    @inbounds for i in 1:L.n
        dd *= dot(U[i,:],W[i,:])^2
    end
    return dd
end

function logdet(L::SymEGRSSCholesky)
    dd = zero(eltype(L))
	U = getfield(L, :U)
    W = getfield(L, :W)
    @inbounds for i in 1:L.n
        dd += log(dot(U[i,:],W[i,:]))
    end
    dd + dd # instead of 2.0dd which can change the type
end
