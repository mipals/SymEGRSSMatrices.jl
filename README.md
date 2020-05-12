# SymEGRSSMatrices.jl

[![Build Status](https://travis-ci.com/mipals/SymEGRSSMatrices.jl.svg?branch=master)](https://travis-ci.com/mipals/SymEGRSSMatrices.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/mipals/SymEGRSSMatrices.jl?svg=true)](https://ci.appveyor.com/project/mipals/SymEGRSSMatrices-jl)
[![Codecov](https://codecov.io/gh/mipals/SymEGRSSMatrices.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mipals/SymEGRSSMatrices.jl)
[![Coveralls](https://coveralls.io/repos/github/mipals/SymEGRSSMatrices.jl/badge.svg?branch=master)](https://coveralls.io/github/mipals/SymEGRSSMatrices.jl?branch=master)

## Description
A package for efficiently computing with symmetric extended generator representable semiseparable matrices and a variant thereof. In short this means matrices of the form

<img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;K=\text{\textbf{tril}}(UV^T)&space;&plus;&space;\text{\textbf{triu}}(V^TU,1),&\quad%20U,V\in\mathbb{R}^{n\times%20p}" title="K=\text{\textbf{tril}}(UV^T) + \text{\textbf{triu}}(VU^T,1)" />

as well as

<img src="https://latex.codecogs.com/svg.latex?\dpi{100}&space;K=\text{\textbf{tril}}(UV^T)&space;&plus;&space;\text{\textbf{tril}}(VU^T,1)&space;&plus;&space;\text{\textbf{diag}}(d),&\quad%20U,V\in\mathbb{R}^{n\times%20p},&space;d\in\mathbb{R}^n" title="K=\text{\textbf{tril}}(UV^T) + \text{\textbf{tril}}(VU^T,1) + \text{\textbf{diag}}(d)" />

All implemented algorithms (multiplication, Cholesky factorization, forward/backward substitution as well as various traces and determinants) run linear in time and memory w.r.t. to the number of data points ```n```.

A more in-depth descriptions of the algorithms can be found in [1] or [here](https://github.com/mipals/SmoothingSplines.jl/blob/master/mt_mikkel_paltorp.pdf).

## Usage
First we need to create generators U and V that represent the symmetric matrix, ```K = tril(UV') + triu(VU',1)``` as well a test vector ```x```.
```julia
julia> using SymEGRSSMatrices
julia> U, V = SymEGRSSMatrices.spline_kernel(Vector(0.1:0.01:1), 2); # Creating input such that K is positive definite
julia> K = SymEGRSSMatrix(U,V); # Symmetric generator representable semiseparable matrix
julia> x = ones(size(K,1)); # Test vector
```
We can now compute products with ```K``` and ```K'```. The result are the same as ```K``` is symmetric.
```julia
julia> K*x
91×1 Array{Float64,2}:
  0.23508333333333334
  0.28261583333333334
  0.3341535          
  0.3896073333333333 
  0.44888933333333336
  0.5119124999999999 
  ⋮                  
 11.977057499999997  
 12.146079333333331  
 12.31510733333333   
 12.484138499999995  
 12.65317083333333 

julia> K'*x
91×1 Array{Float64,2}:
  0.23508333333333334
  0.28261583333333334
  0.3341535          
  0.3896073333333333 
  0.44888933333333336
  0.5119124999999999 
  ⋮                  
 11.977057499999997  
 12.146079333333331  
 12.31510733333333   
 12.484138499999995  
 12.65317083333333  

julia> K*(K\x)
91×1 Array{Float64,2}:
 1.0000000000000036
 0.9999999999999982
 0.9999999999999956
 0.9999999999999944
 0.9999999999999951
 0.999999999999995 
 ⋮                 
 0.9999999999996279
 0.9999999999996153
 0.9999999999996028
 0.9999999999995898
 0.9999999999995764
```

Furthermore from the ```SymEGRSSMatrix``` structure we can efficiently compute the Cholesky factorization as
```julia 
julia> L = cholesky(K); # Computing the Cholesky factorization of K
```
Now ```L``` represents a Cholesky factorization with of form ```L = tril(UW')```, requiring only ```O(np)``` storage. 

A struct for the dealing with symmetric matrices of the form, ```K = tril(UV') + triu(VU',1) + diag(d)``` called ```SymEGRQSMatrix``` is also implemented. The usage is similar to that of ```SymEGRSSMatrix``` and can be created as follows
```julia
julia> U, V = SymEGRSSMatrices.spline_kernel(Vector(0.1:0.01:1), 2); # Creating input such that K is positive definite
julia> K = SymEGRQSMatrix(U,V,rand(size(U,1)); # Symmetric generator representable semiseparable matrix + diagonal
```
The Cholesky factorization of this matrix can be computed using ```cholesky```. Note however here that ```L``` represents a matrix of the form ```L = tril(UW',-1) + diag(c)```

## Benchmarks
### Computing Cholesky factorization of ```K = tril(UV') + triu(VU',1)```
![Scaling of the Cholesky factorization of an EGRSS matrix](https://i.imgur.com/NFqfreO.png)
### Computing Cholesky factorization of ```K = tril(UV') + triu(VU',1) + diag(d)```
![Scaling of the Cholesky factorization of an EGRQS matrix](https://i.imgur.com/IuupJSP.png)
### Solving linear systems using a Cholesky factorization with the form ```L = tril(UW')```
![Solving a system using the implicit Cholesky factorization](https://i.imgur.com/mYBNTSr.png)

## References
[1] M. S. Andersen and T. Chen, “Smoothing Splines and Rank Structured Matrices: Revisiting the Spline Kernel,” SIAM Journal on Matrix Analysis and Applications, 2020.

[2] J. Keiner. "Fast Polynomial Transforms." Logos Verlag Berlin, 2011.
