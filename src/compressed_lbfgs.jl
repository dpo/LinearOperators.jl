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

function CompressedLBFGS(m::Int, n::Int; T=Float64, gpu=default_gpu(), M=default_matrix_type(gpu, T), V=default_vector_type(gpu, T))
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
  if op.k < op.m # still some place in structures
    op.k += 1
    op.Sₖ[:, op.k] .= s
    op.Yₖ[:, op.k] .= y
    op.Dₖ.diag[op.k] = dot(s, y)
    op.Lₖ.data[op.k, op.k] = 0
    for i in 1:op.k-1
      # op.Lₖ.data[op.k, i] = dot(s, op.Yₖ[:, i])
      op.Lₖ.data[op.k, i] = dot(op.Sₖ[:, op.k], op.Yₖ[:, i])
    end
    # the secan equation fails if this line is uncommented
  else # update matrix with circular shift
    println("else")
    # must be tested
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
  @show op.Lₖ  
  @show op.Sₖ
  @show op.Yₖ
  @show op.Dₖ
  # op.α = dot(y,s)/dot(s,s)
  op.intermediate_structure_updated = false
  return op
end

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

  # steps 4 and 6, in case the intermediary required structure are not up to date
  (!op.intermediate_structure_updated) && (precompile_iterated_structure!(op))

  # step 5, try views for mul!
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