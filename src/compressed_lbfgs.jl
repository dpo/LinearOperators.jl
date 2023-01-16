#=
Compressed LBFGS implementation from:
    REPRESENTATIONS OF QUASI-NEWTON MATRICES AND THEIR USE IN LIMITED MEMORY METHODS
    Richard H. Byrd, Jorge Nocedal and Robert B. Schnabel (1994)
    DOI: 10.1007/BF01582063

Implemented by Paul Raynaud (supervised by Dominique Orban)
=#

using LinearAlgebra, LinearAlgebra.BLAS
using CUDA

export CompressedLBFGS

"""
    CompressedLBFGS{T, M<:AbstractMatrix{T}, V<:AbstractVector{T}}

A LBFGS limited-memory operator.
It represents a linear application Rⁿˣⁿ, considering at most `m` BFGS updates.
This implementation considers the bloc matrices reoresentation of the BFGS (forward) update.
It follows the algorithm described in [REPRESENTATIONS OF QUASI-NEWTON MATRICES AND THEIR USE IN LIMITED MEMORY METHODS](https://link.springer.com/article/10.1007/BF01582063) from Richard H. Byrd, Jorge Nocedal and Robert B. Schnabel (1994).
This operator considers several fields directly related to the bloc representation of the operator:
- `m`: the maximal memory of the operator;
- `n`: the dimension of the linear application;
- `k`: the current memory's size of the operator;
- `α`: scalar for `B₀ = α I`;
- `Sₖ`: retain the `k`-th last vectors `s` from the updates parametrized by `(s,y)`;
- `Yₖ`: retain the `k`-th last vectors `y` from the updates parametrized by `(s,y)`;;
- `Dₖ`: a diagonal matrix mandatory to perform the linear application and to form the matrix;
- `Lₖ`: a lower diagonal mandatory to perform the linear application and to form the matrix.
In addition to this structures which are circurlarly update when `k` reaches `m`, we consider other intermediate data structures renew at each update:
- `chol_matrix`: a matrix required to store a Cholesky factorization of a Rᵏˣᵏ matrix;
- `intermediate_1`: a R²ᵏˣ²ᵏ matrix;
- `intermediate_2`: a R²ᵏˣ²ᵏ matrix;
- `inverse_intermediate_1`: a R²ᵏˣ²ᵏ matrix;
- `inverse_intermediate_2`: a R²ᵏˣ²ᵏ matrix;
- `intermediary_vector`: a vector ∈ Rᵏ to store intermediate solutions;
- `sol`: a vector ∈ Rᵏ to store intermediate solutions;
- `intermediate_structure_updated`: inform if the intermediate structures are up-to-date or not.
This implementation is designed to work either on CPU or GPU.
"""
mutable struct CompressedLBFGS{T, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
  m::Int # memory of the operator
  n::Int # vector size
  k::Int # k ≤ m, active memory of the operator
  α::T # B₀ = αI
  Sₖ::M # gather all sₖ₋ₘ 
  Yₖ::M # gather all yₖ₋ₘ 
  Dₖ::Diagonal{T,V} # m * m
  Lₖ::LowerTriangular{T,M} # m * m

  chol_matrix::M # 2m * 2m
  intermediate_1::UpperTriangular{T,M} # 2m * 2m
  intermediate_2::LowerTriangular{T,M} # 2m * 2m
  inverse_intermediate_1::UpperTriangular{T,M} # 2m * 2m
  inverse_intermediate_2::LowerTriangular{T,M} # 2m * 2m
  intermediary_vector::V # 2m
  sol::V # m
  intermediate_structure_updated::Bool
end

default_gpu() = CUDA.functional() ? true : false
default_matrix_type(gpu::Bool, T::DataType) = gpu ? CuMatrix{T} : Matrix{T}
default_vector_type(gpu::Bool, T::DataType) = gpu ? CuVector{T} : Vector{T}

"""
    CompressedLBFGS(n::Int; [T=Float64, m=5], gpu:Bool)

A implementation of a LBFGS operator (forward), representing a `nxn` linear application.
It considers at most `k` BFGS iterates, and fit the architecture depending if it is launched on a CPU or a GPU.
"""
function CompressedLBFGS(n::Int; m::Int=5, T=Float64, gpu=default_gpu(), M=default_matrix_type(gpu, T), V=default_vector_type(gpu, T))
  α = (T)(1)
  k = 0  
  Sₖ = M(undef, n, m)
  Yₖ = M(undef, n, m)
  Dₖ = Diagonal(V(undef, m))
  Lₖ = LowerTriangular(M(undef, m, m))

  chol_matrix = M(undef, m, m)
  intermediate_1 = UpperTriangular(M(undef, 2*m, 2*m))
  intermediate_2 = LowerTriangular(M(undef, 2*m, 2*m))
  inverse_intermediate_1 = UpperTriangular(M(undef, 2*m, 2*m))
  inverse_intermediate_2 = LowerTriangular(M(undef, 2*m, 2*m))
  intermediary_vector = V(undef, 2*m)
  sol = V(undef, 2*m)
  intermediate_structure_updated = false
  return CompressedLBFGS{T,M,V}(m, n, k, α, Sₖ, Yₖ, Dₖ, Lₖ, chol_matrix, intermediate_1, intermediate_2, inverse_intermediate_1, inverse_intermediate_2, intermediary_vector, sol, intermediate_structure_updated)
end

function Base.push!(op::CompressedLBFGS{T,M,V}, s::V, y::V) where {T,M,V<:AbstractVector{T}}
  if op.k < op.m # still some place in the structures
    op.k += 1
    op.Sₖ[:, op.k] .= s
    op.Yₖ[:, op.k] .= y
    op.Dₖ.diag[op.k] = dot(s, y)
    op.Lₖ.data[op.k, op.k] = 0
    for i in 1:op.k-1
      op.Lₖ.data[op.k, i] = dot(op.Sₖ[:, op.k], op.Yₖ[:, i])
    end
  else # k == m update circurlarly the intermediary structures
    op.Sₖ .= circshift(op.Sₖ, (0, -1))
    op.Yₖ .= circshift(op.Yₖ, (0, -1))
    op.Dₖ .= circshift(op.Dₖ, (-1, -1))
    op.Sₖ[:, op.k] .= s
    op.Yₖ[:, op.k] .= y
    op.Dₖ.diag[op.k] = dot(s, y)
    # circshift doesn't work for a LowerTriangular matrix
    # for the time being, reinstantiate completely the Lₖ matrix
    for j in 1:op.k 
      for i in 1:j-1
        op.Lₖ.data[j, i] = dot(op.Sₖ[:, j], op.Yₖ[:, i])
      end
    end
  end
  # secant equation fails if uncommented
  # op.α = dot(y,s)/dot(s,s)
  op.intermediate_structure_updated = false
  return op
end

# Algorithm 3.2 (p15)
# Theorem 2.3 (p6)
function Base.Matrix(op::CompressedLBFGS{T,M,V}) where {T,M,V}
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
function inverse_cholesky(op::CompressedLBFGS)
  view(op.chol_matrix, 1:op.k, 1:op.k) .= op.α .* (transpose(view(op.Sₖ, :, 1:op.k)) * view(op.Sₖ, :, 1:op.k)) .+ view(op.Lₖ, 1:op.k, 1:op.k) * inv(op.Dₖ[1:op.k, 1:op.k]) * transpose(view(op.Lₖ, 1:op.k, 1:op.k))
  cholesky!(Symmetric(view(op.chol_matrix, 1:op.k, 1:op.k)))
  Jₖ = transpose(UpperTriangular(view(op.chol_matrix, 1:op.k, 1:op.k)))
  return Jₖ
end

# step 6, must be improve
function precompile_iterated_structure!(op::CompressedLBFGS)
  Jₖ = inverse_cholesky(op)

  view(op.intermediate_1, 1:op.k,1:op.k) .= .- view(op.Dₖ, 1:op.k, 1:op.k)^(1/2)
  view(op.intermediate_1, 1:op.k,op.k+1:2*op.k) .= view(op.Dₖ, 1:op.k, 1:op.k)^(-1/2) * transpose(view(op.Lₖ, 1:op.k, 1:op.k))
  view(op.intermediate_1, op.k+1:2*op.k, 1:op.k) .= 0
  view(op.intermediate_1, op.k+1:2*op.k, op.k+1:2*op.k) .= transpose(Jₖ)

  view(op.intermediate_2, 1:op.k, 1:op.k) .= view(op.Dₖ, 1:op.k, 1:op.k)^(1/2)
  view(op.intermediate_2, 1:op.k, op.k+1:2*op.k) .= 0
  view(op.intermediate_2, op.k+1:2*op.k, 1:op.k) .= .- view(op.Lₖ, 1:op.k, 1:op.k) * view(op.Dₖ, 1:op.k, 1:op.k)^(-1/2)
  view(op.intermediate_2, op.k+1:2*op.k, op.k+1:2*op.k) .= Jₖ

  view(op.inverse_intermediate_1, 1:2*op.k, 1:2*op.k) .= inv(op.intermediate_1[1:2*op.k, 1:2*op.k])
  view(op.inverse_intermediate_2, 1:2*op.k, 1:2*op.k) .= inv(op.intermediate_2[1:2*op.k, 1:2*op.k])
  
  op.intermediate_structure_updated = true
end

# Algorithm 3.2 (p15)
function LinearAlgebra.mul!(Bv::V, op::CompressedLBFGS{T,M,V}, v::V) where {T,M,V<:AbstractVector{T}}
  # step 1-3 mainly done by Base.push!

  # steps 4 and 6, in case the intermediary structures required are not up to date
  (!op.intermediate_structure_updated) && (precompile_iterated_structure!(op))

  # step 5
  mul!(view(op.sol, 1:op.k), transpose(view(op.Yₖ, :, 1:op.k)), v)
  mul!(view(op.sol, op.k+1:2*op.k), transpose(view(op.Sₖ, :, 1:op.k)), v)
  # scal!(op.α, view(op.sol, op.k+1:2*op.k)) # more allocation, slower
  view(op.sol, op.k+1:2*op.k) .*= op.α

  # view(op.sol, 1:2*op.k) .= view(op.inverse_intermediate_1, 1:2*op.k, 1:2*op.k) * (view(op.inverse_intermediate_2, 1:2*op.k, 1:2*op.k) * view(op.sol, 1:2*op.k))
  mul!(view(op.intermediary_vector, 1:2*op.k), view(op.inverse_intermediate_2, 1:2*op.k, 1:2*op.k), view(op.sol, 1:2*op.k))
  mul!(view(op.sol, 1:2*op.k), view(op.inverse_intermediate_1, 1:2*op.k, 1:2*op.k), view(op.intermediary_vector, 1:2*op.k))
  
  # step 7 
  # Bv .= op.α .* v .- (view(op.Yₖ, :,1:op.k) * view(op.sol, 1:op.k) .+ op.α .* view(op.Sₖ, :, 1:op.k) * view(op.sol, op.k+1:2*op.k))
  mul!(Bv, view(op.Yₖ, :, 1:op.k),  view(op.sol, 1:op.k))
  mul!(Bv, view(op.Sₖ, :, 1:op.k), view(op.sol, op.k+1:2*op.k), - op.α, (T)(-1))
  Bv .+= op.α .* v 
  return Bv
end