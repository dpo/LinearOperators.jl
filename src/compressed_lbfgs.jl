#=
Compressed LBFGS implementation from:
    REPRESENTATIONS OF QUASI-NEWTON MATRICES AND THEIR USE IN LIMITED MEMORY METHODS
    Richard H. Byrd, Jorge Nocedal and Robert B. Schnabel (1994)
    DOI: 10.1007/BF01582063

Implemented by Paul Raynaud (supervised by Dominique Orban)
=#

using LinearAlgebra, LinearAlgebra.BLAS
using CUDA

export CompressedLBFGSOperator

"""
    CompressedLBFGSOperator{T, M<:AbstractMatrix{T}, V<:AbstractVector{T}}

A LBFGS limited-memory operator.
It represents a linear application Rⁿˣⁿ, considering at most `mem` BFGS updates.
This implementation considers the bloc matrices reoresentation of the BFGS (forward) update.
It follows the algorithm described in [REPRESENTATIONS OF QUASI-NEWTON MATRICES AND THEIR USE IN LIMITED MEMORY METHODS](https://link.springer.com/article/10.1007/BF01582063) from Richard H. Byrd, Jorge Nocedal and Robert B. Schnabel (1994).
This operator considers several fields directly related to the bloc representation of the operator:
- `mem`: the maximal memory of the operator;
- `n`: the dimension of the linear application;
- `k`: the current memory's size of the operator;
- `α`: scalar for `B₀ = α I`;
- `Sₖ`: retain the `k`-th last vectors `s` from the updates parametrized by `(s,y)`;
- `Yₖ`: retain the `k`-th last vectors `y` from the updates parametrized by `(s,y)`;;
- `Dₖ`: a diagonal matrix mandatory to perform the linear application and to form the matrix;
- `Lₖ`: a lower diagonal mandatory to perform the linear application and to form the matrix.
In addition to this structures which are circurlarly update when `k` reaches `mem`, we consider other intermediate data structures renew at each update:
- `chol_matrix`: a matrix required to store a Cholesky factorization of a Rᵏˣᵏ matrix;
- `intermediate_1`: a R²ᵏˣ²ᵏ matrix;
- `intermediate_2`: a R²ᵏˣ²ᵏ matrix;
- `inverse_intermediate_1`: a R²ᵏˣ²ᵏ matrix;
- `inverse_intermediate_2`: a R²ᵏˣ²ᵏ matrix;
- `intermediary_vector`: a vector ∈ Rᵏ to store intermediate solutions;
- `sol`: a vector ∈ Rᵏ to store intermediate solutions;
This implementation is designed to work either on CPU or GPU.
"""
mutable struct CompressedLBFGSOperator{T, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
  mem::Int # memory of the operator
  n::Int # vector size
  k::Int # k ≤ mem, active memory of the operator
  α::T # B₀ = αI
  Sₖ::M # gather all sₖ₋ₘ 
  Yₖ::M # gather all yₖ₋ₘ 
  Dₖ::Diagonal{T,V} # mem * mem
  Lₖ::LowerTriangular{T,M} # mem * mem

  chol_matrix::M # 2m * 2m
  intermediate_diagonal::Diagonal{T,V} # mem * mem
  intermediate_1::UpperTriangular{T,M} # 2m * 2m
  intermediate_2::LowerTriangular{T,M} # 2m * 2m
  inverse_intermediate_1::UpperTriangular{T,M} # 2m * 2m
  inverse_intermediate_2::LowerTriangular{T,M} # 2m * 2m
  intermediary_vector::V # 2m
  sol::V # mem
end

default_gpu() = CUDA.functional() ? true : false
default_matrix_type(gpu::Bool; T::DataType=Float64) = gpu ? CuMatrix{T} : Matrix{T}
default_vector_type(gpu::Bool; T::DataType=Float64) = gpu ? CuVector{T} : Vector{T}

function columnshift!(A::AbstractMatrix{T}; direction::Int=-1, indicemax::Int=size(A)[1]) where T
  map(i-> view(A,:,i+direction) .= view(A,:,i), 1-direction:indicemax)
  return A
end

function vectorshift!(v::AbstractVector{T}; direction::Int=-1, indicemax::Int=length(v)) where T
  view(v, 1:indicemax+direction) .= view(v,1-direction:indicemax)
  return v
end

"""
    CompressedLBFGSOperator(n::Int; [T=Float64, mem=5], gpu:Bool)

A implementation of a LBFGS operator (forward), representing a `nxn` linear application.
It considers at most `k` BFGS iterates, and fit the architecture depending if it is launched on a CPU or a GPU.
"""
function CompressedLBFGSOperator(n::Int; mem::Int=5, T=Float64, gpu=default_gpu(), M=default_matrix_type(gpu; T), V=default_vector_type(gpu; T))
  α = (T)(1)
  k = 0  
  Sₖ = M(undef, n, mem)
  Yₖ = M(undef, n, mem)
  Dₖ = Diagonal(V(undef, mem))
  Lₖ = LowerTriangular(M(undef, mem, mem))
  Lₖ .= (T)(0)

  chol_matrix = M(undef, mem, mem)
  intermediate_diagonal = Diagonal(V(undef, mem))
  intermediate_1 = UpperTriangular(M(undef, 2*mem, 2*mem))
  intermediate_2 = LowerTriangular(M(undef, 2*mem, 2*mem))
  inverse_intermediate_1 = UpperTriangular(M(undef, 2*mem, 2*mem))
  inverse_intermediate_2 = LowerTriangular(M(undef, 2*mem, 2*mem))
  intermediary_vector = V(undef, 2*mem)
  sol = V(undef, 2*mem)
  return CompressedLBFGSOperator{T,M,V}(mem, n, k, α, Sₖ, Yₖ, Dₖ, Lₖ, chol_matrix, intermediate_diagonal, intermediate_1, intermediate_2, inverse_intermediate_1, inverse_intermediate_2, intermediary_vector, sol)
end

function Base.push!(op::CompressedLBFGSOperator{T,M,V}, s::V, y::V) where {T,M,V<:AbstractVector{T}}
  if op.k < op.mem # still some place in the structures
    op.k += 1
    view(op.Sₖ, :, op.k) .= s
    view(op.Yₖ, :, op.k) .= y
    view(op.Dₖ.diag, op.k) .= dot(s, y)
    mul!(view(op.Lₖ.data, op.k, 1:op.k-1), transpose(view(op.Yₖ, :, 1:op.k-1)), view(op.Sₖ, :, op.k) )
  else # k == mem update circurlarly the intermediary structures
    columnshift!(op.Sₖ; indicemax=op.k)
    columnshift!(op.Yₖ; indicemax=op.k)
    # op.Dₖ .= circshift(op.Dₖ, (-1, -1))
    vectorshift!(op.Dₖ.diag; indicemax=op.k)
    view(op.Sₖ, :, op.k) .= s
    view(op.Yₖ, :, op.k) .= y
    view(op.Dₖ.diag, op.k) .= dot(s, y)

    map(i-> view(op.Lₖ, i:op.mem-1, i-1) .= view(op.Lₖ, i+1:op.mem, i), 2:op.mem)
    mul!(view(op.Lₖ.data, op.k, 1:op.k-1), transpose(view(op.Yₖ, :, 1:op.k-1)), view(op.Sₖ, :, op.k) )
  end

  # step 4 and 6
  precompile_iterated_structure!(op)

  # secant equation fails if uncommented
  # op.α = dot(y,s)/dot(s,s)
  return op
end

# Algorithm 3.2 (p15)
# Theorem 2.3 (p6)
function Base.Matrix(op::CompressedLBFGSOperator{T,M,V}) where {T,M,V}
  B₀ = M(zeros(T, op.n, op.n))
  map(i -> B₀[i, i] = op.α, 1:op.n)

  BSY = M(undef, op.n, 2*op.k)
  (op.k > 0) && (BSY[:, 1:op.k] = B₀ * op.Sₖ[:, 1:op.k])
  (op.k > 0) && (BSY[:, op.k+1:2*op.k] = op.Yₖ[:, 1:op.k])
  _C = M(undef, 2*op.k, 2*op.k)
  (op.k > 0) && (_C[1:op.k, 1:op.k] .= transpose(op.Sₖ[:, 1:op.k]) * op.Sₖ[:, 1:op.k])
  (op.k > 0) && (_C[1:op.k, op.k+1:2*op.k] .= op.Lₖ[1:op.k, 1:op.k])
  (op.k > 0) && (_C[op.k+1:2*op.k, 1:op.k] .= transpose(op.Lₖ[1:op.k, 1:op.k]))
  (op.k > 0) && (_C[op.k+1:2*op.k, op.k+1:2*op.k] .= .- op.Dₖ[1:op.k, 1:op.k])
  C = inv(_C)

  Bₖ = B₀ .- BSY * C * transpose(BSY)
  return Bₖ
end

# Algorithm 3.2 (p15)
# step 4, Jₖ is computed only if needed
function inverse_cholesky(op::CompressedLBFGSOperator{T,M,V}) where {T,M,V}
  view(op.intermediate_diagonal.diag, 1:op.k) .= inv.(view(op.Dₖ.diag, 1:op.k))
  
  mul!(view(op.inverse_intermediate_1, 1:op.k, 1:op.k), view(op.intermediate_diagonal, 1:op.k, 1:op.k), transpose(view(op.Lₖ, 1:op.k, 1:op.k)))
  mul!(view(op.chol_matrix, 1:op.k, 1:op.k), view(op.Lₖ, 1:op.k, 1:op.k), view(op.inverse_intermediate_1, 1:op.k, 1:op.k))

  mul!(view(op.chol_matrix, 1:op.k, 1:op.k), transpose(view(op.Sₖ, :, 1:op.k)), view(op.Sₖ, :, 1:op.k), op.α, (T)(1))

  cholesky!(Symmetric(view(op.chol_matrix, 1:op.k, 1:op.k)))
  Jₖ = transpose(UpperTriangular(view(op.chol_matrix, 1:op.k, 1:op.k)))
  return Jₖ
end

# step 6, must be improve
function precompile_iterated_structure!(op::CompressedLBFGSOperator)
  Jₖ = inverse_cholesky(op)

  # constant update
  view(op.intermediate_1, op.k+1:2*op.k, 1:op.k) .= 0
  view(op.intermediate_2, 1:op.k, op.k+1:2*op.k) .= 0
  view(op.intermediate_1, op.k+1:2*op.k, op.k+1:2*op.k) .= transpose(Jₖ)
  view(op.intermediate_2, op.k+1:2*op.k, op.k+1:2*op.k) .= Jₖ

  # updates related to D^(1/2)
  view(op.intermediate_diagonal.diag, 1:op.k) .= sqrt.(view(op.Dₖ.diag, 1:op.k))
  view(op.intermediate_1, 1:op.k,1:op.k) .= .- view(op.intermediate_diagonal, 1:op.k, 1:op.k)
  view(op.intermediate_2, 1:op.k, 1:op.k) .= view(op.intermediate_diagonal, 1:op.k, 1:op.k)

  # updates related to D^(-1/2)
  view(op.intermediate_diagonal.diag, 1:op.k) .= (x -> 1/sqrt(x)).(view(op.Dₖ.diag, 1:op.k))
  mul!(view(op.intermediate_1, 1:op.k,op.k+1:2*op.k), view(op.intermediate_diagonal, 1:op.k, 1:op.k), transpose(view(op.Lₖ, 1:op.k, 1:op.k)))
  mul!(view(op.intermediate_2, op.k+1:2*op.k, 1:op.k), view(op.Lₖ, 1:op.k, 1:op.k), view(op.intermediate_diagonal, 1:op.k, 1:op.k))
  view(op.intermediate_2, op.k+1:2*op.k, 1:op.k) .= view(op.intermediate_2, op.k+1:2*op.k, 1:op.k) .* -1
  
  view(op.inverse_intermediate_1, 1:2*op.k, 1:2*op.k) .= inv(op.intermediate_1[1:2*op.k, 1:2*op.k])
  view(op.inverse_intermediate_2, 1:2*op.k, 1:2*op.k) .= inv(op.intermediate_2[1:2*op.k, 1:2*op.k])
end

# Algorithm 3.2 (p15)
function LinearAlgebra.mul!(Bv::V, op::CompressedLBFGSOperator{T,M,V}, v::V) where {T,M,V<:AbstractVector{T}}
  # step 1-4 and 6 mainly done by Base.push!
  # step 5
  mul!(view(op.sol, 1:op.k), transpose(view(op.Yₖ, :, 1:op.k)), v)
  mul!(view(op.sol, op.k+1:2*op.k), transpose(view(op.Sₖ, :, 1:op.k)), v)
  # scal!(op.α, view(op.sol, op.k+1:2*op.k)) # more allocation, slower
  view(op.sol, op.k+1:2*op.k) .*= op.α

  mul!(view(op.intermediary_vector, 1:2*op.k), view(op.inverse_intermediate_2, 1:2*op.k, 1:2*op.k), view(op.sol, 1:2*op.k))
  mul!(view(op.sol, 1:2*op.k), view(op.inverse_intermediate_1, 1:2*op.k, 1:2*op.k), view(op.intermediary_vector, 1:2*op.k))
  
  # step 7 
  mul!(Bv, view(op.Yₖ, :, 1:op.k),  view(op.sol, 1:op.k))
  mul!(Bv, view(op.Sₖ, :, 1:op.k), view(op.sol, op.k+1:2*op.k), - op.α, (T)(-1))
  Bv .+= op.α .* v 
  return Bv
end