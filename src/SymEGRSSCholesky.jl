struct SymEGRSSCholesky{T,UT<:AbstractMatrix,WT<:AbstractMatrix} <: AbstractMatrix{T}
    Ut::UT
    Wt::WT
    n::Int
    p::Int
    function SymEGRSSCholesky{T,UT,WT}(Ut,Wt,n,p) where
			 {T,UT<:AbstractMatrix,WT<:AbstractMatrix}
        Un, Um = size(Ut)
		Wn, Wm = size(Wt)
		(Un == Wn && Um == Wm) || throw(DimensionMismatch())
        new(Ut,Wt,n,p)
    end
end

# Creating generator such that L = tril(U*W')
function ss_create_w(Ut,Vt)
    p,n = size(Ut);
    wTw = zeros(p,p);
    Wt = similar(Vt);
    @inbounds for j = 1:n
        tmpu = Ut[:,j];
        tmp  = Vt[:,j] - wTw*tmpu;
        w = tmp/sqrt(abs(tmpu'*tmp))
        Wt[:,j] = w;
        wTw = wTw + w*w';
    end
    return Wt
end

cholesky(K::SymEGRSSMatrix{T,UT,VT}) where {T,UT<:AbstractMatrix,VT<:AbstractMatrix} =
        SymEGRSSCholesky{T,typeof(K.Ut),typeof(K.Vt)}(K.Ut,ss_create_w(K.Ut,K.Vt),K.n,K.p)


########################################################################
#### Helpful properties. Not nessecarily computionally efficient    ####
########################################################################
Matrix(K::SymEGRSSCholesky) = getproperty(K,:L)

size(K::SymEGRSSCholesky) = (K.n, K.n)
size(K::SymEGRSSCholesky, d::Int) = (1 <= d && d <=2) ? size(K)[d] : throw(ArgumentError("Invalid dimension $d"))

function getindex(K::SymEGRSSCholesky, i::Int, j::Int)
	i >= j && return dot(K.Ut[:,i], K.Wt[:,j])
	return 0
end

function getproperty(K::SymEGRSSCholesky, d::Symbol)
    if d === :U
        return UpperTriangular(K.Wt'*K.Ut)
    elseif d === :L
        return LowerTriangular(K.Ut'*K.Wt)
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
function ss_forward!(X::AbstractArray, Ut::AbstractArray,
                     Wt::AbstractArray, B::AbstractArray)
    p, n = size(Ut)
    mx = size(B,2)
    Wbar = zeros(p, mx);
    #@inbounds for i = 1:n
	for i = 1:n
        tmpU = Ut[:,i]
        tmpW = Wt[:,i]
        X[i,:] = (B[i:i,:] - tmpU'*Wbar)./(tmpU'*tmpW);
        Wbar += tmpW .* X[i:i,:];
    end
end

#### Backward substitution (solve L'x = b) ####
function ssa_backward!(X::AbstractArray, Ut::AbstractArray,
                       Wt::AbstractArray, B::AbstractArray)
    p, n = size(Ut)
    mx = size(B,2)
    Ubar = zeros(p,mx);
    #@inbounds for i = n:-1:1
	for i = n:-1:1
        tmpU = Ut[:,i];
        tmpW = Wt[:,i];
        X[i,:] = (B[i:i,:] - tmpW'*Ubar)/(tmpU'*tmpW);
        Ubar += tmpU .* X[i:i,:];
    end
end

function (\)(F::SymEGRSSCholesky, B::AbstractVecOrMat)
	X = similar(B)
	ss_forward!(X,F.Ut,F.Wt,B)
	return X
end
function (\)(F::Adjoint{<:Any,<:SymEGRSSCholesky}, B::AbstractVecOrMat)
	Y = similar(B)
	ssa_backward!(Y,F.parent.Ut,F.parent.Wt,B)
	return Y
end

########################################################################
#### Linear Algebra routines 							   			####
########################################################################
function det(L::SymEGRSSCholesky)
    dd = one(eltype(L))
    @inbounds for i in 1:L.n
        dd *= dot(L.Ut[:,i],L.Wt[:,i])^2
    end
    return dd
end

function logdet(L::SymEGRSSCholesky)
    dd = zero(eltype(L))
    @inbounds for i in 1:L.n
        dd += log(dot(L.Ut[:,i],L.Wt[:,i]))
    end
    dd + dd # instead of 2.0dd which can change the type
end
