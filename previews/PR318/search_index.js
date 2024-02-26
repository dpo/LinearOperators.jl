var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [LinearOperators]","category":"page"},{"location":"reference/#Base.Matrix-Union{Tuple{AbstractLinearOperator{T}}, Tuple{T}} where T","page":"Reference","title":"Base.Matrix","text":"A = Matrix(op)\n\nMaterialize an operator as a dense array using op.ncol products.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearAlgebra.Hermitian","page":"Reference","title":"LinearAlgebra.Hermitian","text":"Hermitian(op, uplo=:U)\n\n\n\n\n\n","category":"type"},{"location":"reference/#LinearAlgebra.Symmetric","page":"Reference","title":"LinearAlgebra.Symmetric","text":"Symmetric(op, uplo=:U)\n\n\n\n\n\n","category":"type"},{"location":"reference/#LinearOperators.DiagonalAndrei","page":"Reference","title":"LinearOperators.DiagonalAndrei","text":"DiagonalAndrei(d)\n\nConstruct a linear operator that represents a diagonal quasi-Newton approximation as described in\n\nAndrei, N. A diagonal quasi-Newton updating method for unconstrained optimization. https://doi.org/10.1007/s11075-018-0562-7\n\nThe approximation satisfies the weak secant equation and is not guaranteed to be positive definite.\n\nArguments\n\nd::AbstractVector: initial diagonal approximation.\n\n\n\n\n\n","category":"type"},{"location":"reference/#LinearOperators.DiagonalAndrei-Union{Tuple{AbstractVector{T}}, Tuple{T}} where T<:Real","page":"Reference","title":"LinearOperators.DiagonalAndrei","text":"DiagonalAndrei(d)\n\nConstruct a linear operator that represents a diagonal quasi-Newton approximation as described in\n\nAndrei, N. A diagonal quasi-Newton updating method for unconstrained optimization. https://doi.org/10.1007/s11075-018-0562-7\n\nThe approximation satisfies the weak secant equation and is not guaranteed to be positive definite.\n\nArguments\n\nd::AbstractVector: initial diagonal approximation.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.DiagonalPSB","page":"Reference","title":"LinearOperators.DiagonalPSB","text":"DiagonalPSB(d)\n\nConstruct a linear operator that represents a diagonal PSB quasi-Newton approximation as described in\n\nM. Zhu, J. L. Nazareth and H. Wolkowicz The Quasi-Cauchy Relation and Diagonal Updating. SIAM Journal on Optimization, vol. 9, number 4, pp. 1192-1204, 1999. https://doi.org/10.1137/S1052623498331793.\n\nThe approximation satisfies the weak secant equation and is not guaranteed to be positive definite.\n\nArguments\n\nd::AbstractVector: initial diagonal approximation.\n\n\n\n\n\n","category":"type"},{"location":"reference/#LinearOperators.DiagonalPSB-Union{Tuple{AbstractVector{T}}, Tuple{T}} where T<:Real","page":"Reference","title":"LinearOperators.DiagonalPSB","text":"DiagonalPSB(d)\n\nConstruct a linear operator that represents a diagonal PSB quasi-Newton approximation as described in\n\nM. Zhu, J. L. Nazareth and H. Wolkowicz The Quasi-Cauchy Relation and Diagonal Updating. SIAM Journal on Optimization, vol. 9, number 4, pp. 1192-1204, 1999. https://doi.org/10.1137/S1052623498331793.\n\nThe approximation satisfies the weak secant equation and is not guaranteed to be positive definite.\n\nArguments\n\nd::AbstractVector: initial diagonal approximation.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.LBFGSData","page":"Reference","title":"LinearOperators.LBFGSData","text":"A data type to hold information relative to LBFGS operators.\n\n\n\n\n\n","category":"type"},{"location":"reference/#LinearOperators.LBFGSOperator","page":"Reference","title":"LinearOperators.LBFGSOperator","text":"A type for limited-memory BFGS approximations.\n\n\n\n\n\n","category":"type"},{"location":"reference/#LinearOperators.LBFGSOperator-Union{Tuple{I}, Tuple{DataType, I}} where I<:Integer","page":"Reference","title":"LinearOperators.LBFGSOperator","text":"LBFGSOperator(T, n; [mem=5, scaling=true])\nLBFGSOperator(n; [mem=5, scaling=true])\n\nConstruct a limited-memory BFGS approximation in forward form. If the type T is omitted, then Float64 is used.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.LSR1Data","page":"Reference","title":"LinearOperators.LSR1Data","text":"A data type to hold information relative to LSR1 operators.\n\n\n\n\n\n","category":"type"},{"location":"reference/#LinearOperators.LSR1Operator","page":"Reference","title":"LinearOperators.LSR1Operator","text":"A type for limited-memory SR1 approximations.\n\n\n\n\n\n","category":"type"},{"location":"reference/#LinearOperators.LSR1Operator-Union{Tuple{I}, Tuple{DataType, I}} where I<:Integer","page":"Reference","title":"LinearOperators.LSR1Operator","text":"LSR1Operator(T, n; [mem=5, scaling=false)\nLSR1Operator(n; [mem=5, scaling=false)\n\nConstruct a limited-memory SR1 approximation in forward form. If the type T is omitted, then Float64 is used.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.LinearOperator","page":"Reference","title":"LinearOperators.LinearOperator","text":"Base type to represent a linear operator. The usual arithmetic operations may be applied to operators to combine or otherwise alter them. They can be combined with other operators, with matrices and with scalars. Operators may be transposed and conjugate-transposed using the usual Julia syntax.\n\n\n\n\n\n","category":"type"},{"location":"reference/#LinearOperators.LinearOperator-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T","page":"Reference","title":"LinearOperators.LinearOperator","text":"LinearOperator(M::AbstractMatrix{T}; symmetric=false, hermitian=false, S = Vector{T}) where {T}\n\nConstruct a linear operator from a dense or sparse matrix. Use the optional keyword arguments to indicate whether the operator is symmetric and/or hermitian. Change S to use LinearOperators on GPU.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.LinearOperator-Union{Tuple{I}, Tuple{T}, Tuple{Type{T}, I, I, Bool, Bool, Any}, Tuple{Type{T}, I, I, Bool, Bool, Any, Any}, Tuple{Type{T}, I, I, Bool, Bool, Any, Any, Any}} where {T, I<:Integer}","page":"Reference","title":"LinearOperators.LinearOperator","text":"LinearOperator(type::Type{T}, nrow, ncol, symmetric, hermitian, prod!,\n                [tprod!=nothing, ctprod!=nothing],\n                S = Vector{T}) where {T}\n\nConstruct a linear operator from functions where the type is specified as the first argument. Change S to use LinearOperators on GPU. Notice that the linear operator does not enforce the type, so using a wrong type can result in errors. For instance,\n\nA = [im 1.0; 0.0 1.0] # Complex matrix\nfunction mulOp!(res, M, v, α, β)\n  mul!(res, M, v, α, β)\nend\nop = LinearOperator(Float64, 2, 2, false, false, \n                    (res, v, α, β) -> mulOp!(res, A, v, α, β), \n                    (res, u, α, β) -> mulOp!(res, transpose(A), u, α, β), \n                    (res, w, α, β) -> mulOp!(res, A', w, α, β))\nMatrix(op) # InexactError\n\nThe error is caused because Matrix(op) tries to create a Float64 matrix with the contents of the complex matrix A.\n\nUsing * may generate a vector that contains NaN values. This can also happen if you use the 3-args mul! function with a preallocated vector such as  Vector{Float64}(undef, n). To fix this issue you will have to deal with the cases β == 0 and β != 0 separately:\n\nd1 = [2.0; 3.0]\nfunction mulSquareOpDiagonal!(res, d, v, α, β::T) where T\n  if β == zero(T)\n    res .= α .* d .* v\n  else \n    res .= α .* d .* v .+ β .* res\n  end\nend\nop = LinearOperator(Float64, 2, 2, true, true, \n                    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β))\n\nIt is possible to create an operator with the 3-args mul!. In this case, using the 5-args mul! will generate storage vectors.\n\nA = rand(2, 2)\nop = LinearOperator(Float64, 2, 2, false, false, \n                    (res, v) -> mul!(res, A, v),\n                    (res, w) -> mul!(res, A', w))\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.LinearOperator-Union{Tuple{LinearAlgebra.Hermitian{T, S} where S<:(AbstractMatrix{<:T})}, Tuple{T}, Tuple{LinearAlgebra.Hermitian{T, S} where S<:(AbstractMatrix{<:T}), Any}} where T","page":"Reference","title":"LinearOperators.LinearOperator","text":"LinearOperator(M::Hermitian{T}, S = Vector{T}) where {T}\n\nConstructs a linear operator from a Hermitian matrix. If its elements are real, it is also symmetric. Change S to use LinearOperators on GPU.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.LinearOperator-Union{Tuple{LinearAlgebra.SymTridiagonal{T, V} where V<:AbstractVector{T}}, Tuple{T}, Tuple{LinearAlgebra.SymTridiagonal{T, V} where V<:AbstractVector{T}, Any}} where T","page":"Reference","title":"LinearOperators.LinearOperator","text":"LinearOperator(M::SymTridiagonal{T}, S = Vector{T}) where {T}\n\nConstructs a linear operator from a symmetric tridiagonal matrix. If its elements are real, it is also Hermitian, otherwise complex symmetric. Change S to use LinearOperators on GPU.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.LinearOperator-Union{Tuple{LinearAlgebra.Symmetric{T, S} where S<:(AbstractMatrix{<:T})}, Tuple{T}, Tuple{LinearAlgebra.Symmetric{T, S} where S<:(AbstractMatrix{<:T}), Any}} where T<:Real","page":"Reference","title":"LinearOperators.LinearOperator","text":"LinearOperator(M::Symmetric{T}, S = Vector{T}) where {T <: Real} =\n\nConstruct a linear operator from a symmetric real square matrix M. Change S to use LinearOperators on GPU.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.SpectralGradient","page":"Reference","title":"LinearOperators.SpectralGradient","text":"Implementation of a spectral gradient quasi-Newton approximation described in\n\nBirgin, E. G., Martínez, J. M., & Raydan, M. Spectral Projected Gradient Methods: Review and Perspectives. https://doi.org/10.18637/jss.v060.i03\n\n\n\n\n\n","category":"type"},{"location":"reference/#LinearOperators.SpectralGradient-Union{Tuple{I}, Tuple{T}, Tuple{T, I}} where {T<:Real, I<:Integer}","page":"Reference","title":"LinearOperators.SpectralGradient","text":"    SpectralGradient(σ, n)\n\nConstruct a spectral gradient Hessian approximation. The approximation is defined as σI.\n\nArguments\n\nσ::Real: initial positive multiple of the identity;\nn::Int: operator size.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.TimedLinearOperator-Union{Tuple{AbstractLinearOperator{T}}, Tuple{T}} where T","page":"Reference","title":"LinearOperators.TimedLinearOperator","text":"TimedLinearOperator(op)\n\nCreates a linear operator instrumented with timers from TimerOutputs.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opEye","page":"Reference","title":"LinearOperators.opEye","text":"opEye()\n\nIdentity operator.\n\nopI = opEye()\nv = rand(5)\n@assert opI * v === v\n\n\n\n\n\n","category":"type"},{"location":"reference/#LinearOperators.opEye-Tuple{DataType, Int64}","page":"Reference","title":"LinearOperators.opEye","text":"opEye(T, n; S = Vector{T})\nopEye(n)\n\nIdentity operator of order n and of data type T (defaults to Float64). Change S to use LinearOperators on GPU.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opEye-Union{Tuple{I}, Tuple{DataType, I, I}} where I<:Integer","page":"Reference","title":"LinearOperators.opEye","text":"opEye(T, nrow, ncol; S = Vector{T})\nopEye(nrow, ncol)\n\nRectangular identity operator of size nrowxncol and of data type T (defaults to Float64). Change S to use LinearOperators on GPU.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.kron-Tuple{AbstractLinearOperator, AbstractLinearOperator}","page":"Reference","title":"Base.kron","text":"kron(A, B)\n\nKronecker tensor product of A and B in linear operator form, if either or both are linear operators. If both A and B are matrices, then Base.kron is used.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.push!-Tuple{LSR1Operator, AbstractVector, AbstractVector}","page":"Reference","title":"Base.push!","text":"push!(op, s, y)\n\nPush a new {s,y} pair into a L-SR1 operator.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.push!-Union{Tuple{F3}, Tuple{F2}, Tuple{F1}, Tuple{I}, Tuple{T}, Tuple{LBFGSOperator{T, I, F1, F2, F3}, Vector{T}, Vector{T}}} where {T, I, F1, F2, F3}","page":"Reference","title":"Base.push!","text":"push!(op, s, y)\npush!(op, s, y, Bs)\npush!(op, s, y, α, g)\npush!(op, s, y, α, g, Bs)\n\nPush a new {s,y} pair into a L-BFGS operator. The second calling sequence is used for forward updating damping, using the preallocated vector Bs. If the operator is damped, the first call will create Bs and call the second call. The third and fourth calling sequences are used in inverse LBFGS updating in conjunction with damping, where α is the most recent steplength and g the gradient used when solving d=-Hg.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.show-Tuple{IO, AbstractLinearOperator}","page":"Reference","title":"Base.show","text":"show(io, op)\n\nDisplay basic information about a linear operator.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.size-Tuple{AbstractLinearOperator, Integer}","page":"Reference","title":"Base.size","text":"m = size(op, d)\n\nReturn the size of a linear operator along dimension d.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.size-Tuple{AbstractLinearOperator}","page":"Reference","title":"Base.size","text":"m, n = size(op)\n\nReturn the size of a linear operator as a tuple.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearAlgebra.diag-Union{Tuple{LBFGSOperator{T}}, Tuple{T}} where T","page":"Reference","title":"LinearAlgebra.diag","text":"diag(op)\ndiag!(op, d)\n\nExtract the diagonal of a L-BFGS operator in forward mode.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearAlgebra.diag-Union{Tuple{LSR1Operator{T}}, Tuple{T}} where T","page":"Reference","title":"LinearAlgebra.diag","text":"diag(op)\ndiag!(op, d)\n\nExtract the diagonal of a L-SR1 operator in forward mode.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearAlgebra.ishermitian-Tuple{AbstractLinearOperator}","page":"Reference","title":"LinearAlgebra.ishermitian","text":"ishermitian(op)\n\nDetermine whether the operator is Hermitian.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearAlgebra.issymmetric-Tuple{AbstractLinearOperator}","page":"Reference","title":"LinearAlgebra.issymmetric","text":"issymmetric(op)\n\nDetermine whether the operator is symmetric.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.BlockDiagonalOperator-Tuple","page":"Reference","title":"LinearOperators.BlockDiagonalOperator","text":"BlockDiagonalOperator(M1, M2, ..., Mn; S = promote_type(storage_type.(M1, M2, ..., Mn)))\n\nCreates a block-diagonal linear operator:\n\n[ M1           ]\n[    M2        ]\n[       ...    ]\n[           Mn ]\n\nChange S to use LinearOperators on GPU.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.InverseLBFGSOperator-Union{Tuple{I}, Tuple{DataType, I}} where I<:Integer","page":"Reference","title":"LinearOperators.InverseLBFGSOperator","text":"InverseLBFGSOperator(T, n, [mem=5; scaling=true])\nInverseLBFGSOperator(n, [mem=5; scaling=true])\n\nConstruct a limited-memory BFGS approximation in inverse form. If the type T is omitted, then Float64 is used.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.check_ctranspose-Union{Tuple{AbstractLinearOperator{T}}, Tuple{T}} where T<:Union{AbstractFloat, Complex}","page":"Reference","title":"LinearOperators.check_ctranspose","text":"check_ctranspose(op)\n\nCheap check that the operator and its conjugate transposed are related.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.check_hermitian-Union{Tuple{AbstractLinearOperator{T}}, Tuple{T}} where T<:Union{AbstractFloat, Complex}","page":"Reference","title":"LinearOperators.check_hermitian","text":"check_hermitian(op)\n\nCheap check that the operator is Hermitian.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.check_positive_definite-Union{Tuple{AbstractLinearOperator{T}}, Tuple{T}} where T<:Union{AbstractFloat, Complex}","page":"Reference","title":"LinearOperators.check_positive_definite","text":"check_positive_definite(op; semi=false)\n\nCheap check that the operator is positive (semi-)definite.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.has_args5-Tuple{AbstractLinearOperator}","page":"Reference","title":"LinearOperators.has_args5","text":"has_args5(op)\n\nDetermine whether the operator can work with the 5-args mul!. If false, storage vectors will be generated at the first call of the 5-args mul!. No additional vectors are generated when using the 3-args mul!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.normest","page":"Reference","title":"LinearOperators.normest","text":"normest(S) estimates the matrix 2-norm of S. This function is an adaptation of Matlab's built-in NORMEST. This method allocates.\n\n\n\nInputs:   S –- Matrix or LinearOperator type,    tol –-  relative error tol, default(or -1) Machine eps   maxiter –- maximum iteration, default 100\n\nReturns:   e –- the estimated norm   cnt –- the number of iterations used\n\n\n\n\n\n","category":"function"},{"location":"reference/#LinearOperators.opCholesky-Tuple{AbstractMatrix}","page":"Reference","title":"LinearOperators.opCholesky","text":"opCholesky(M, [check=false])\n\nInverse of a Hermitian and positive definite matrix as a linear operator using its Cholesky factorization.  The factorization is computed only once. The optional check argument will perform cheap hermicity and definiteness checks. This Operator is not in-place when using mul!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opDiagonal-Union{Tuple{AbstractVector{T}}, Tuple{T}} where T","page":"Reference","title":"LinearOperators.opDiagonal","text":"opDiagonal(d)\n\nDiagonal operator with the vector d on its main diagonal.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opDiagonal-Union{Tuple{I}, Tuple{T}, Tuple{I, I, AbstractVector{T}}} where {T, I<:Integer}","page":"Reference","title":"LinearOperators.opDiagonal","text":"opDiagonal(nrow, ncol, d)\n\nRectangular diagonal operator of size nrow-by-ncol with the vector d on its main diagonal.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opExtension-Union{Tuple{I}, Tuple{AbstractVector{I}, I}} where I<:Integer","page":"Reference","title":"LinearOperators.opExtension","text":"Z = opExtension(I, ncol)\nZ = opExtension(:, ncol)\n\nCreates a LinearOperator extending a vector of size length(I) to size ncol, where the position of the elements on the new vector are given by the indices I. The operation w = Z * v is equivalent to w = zeros(ncol); w[I] = v.\n\nZ = opExtension(k, ncol)\n\nAlias for opExtension([k], ncol).\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opHermitian-Tuple{AbstractMatrix}","page":"Reference","title":"LinearOperators.opHermitian","text":"opHermitian(A)\n\nA symmetric/hermitian operator based on a matrix.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opHermitian-Union{Tuple{T}, Tuple{S}, Tuple{AbstractVector{S}, AbstractMatrix{T}}} where {S, T}","page":"Reference","title":"LinearOperators.opHermitian","text":"opHermitian(d, A)\n\nA symmetric/hermitian operator based on the diagonal d and lower triangle of A.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opHouseholder-Union{Tuple{AbstractVector{T}}, Tuple{T}} where T","page":"Reference","title":"LinearOperators.opHouseholder","text":"opHouseholder(h)\n\nApply a Householder transformation defined by the vector h. The result is x -> (I - 2 h hᵀ) x.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opInverse-Union{Tuple{AbstractMatrix{T}}, Tuple{T}} where T","page":"Reference","title":"LinearOperators.opInverse","text":"opInverse(M; symm=false, herm=false)\n\nInverse of a matrix as a linear operator using \\. Useful for triangular matrices. Note that each application of this operator applies \\. This Operator is not in-place when using mul!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opLDL-Tuple{AbstractMatrix}","page":"Reference","title":"LinearOperators.opLDL","text":"opLDL(M, [check=false])\n\nInverse of a symmetric matrix as a linear operator using its LDLᵀ factorization if it exists. The factorization is computed only once. The optional check argument will perform a cheap hermicity check.\n\nIf M is sparse and real, then only the upper triangle should be stored in order to use  LDLFactorizations.jl:\n\ntriu!(M)\nopLDL(Symmetric(M, :U))\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opOnes-Union{Tuple{I}, Tuple{DataType, I, I}} where I<:Integer","page":"Reference","title":"LinearOperators.opOnes","text":"opOnes(T, nrow, ncol; S = Vector{T})\nopOnes(nrow, ncol)\n\nOperator of all ones of size nrow-by-ncol of data type T (defaults to Float64). Change S to use LinearOperators on GPU.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opRestriction-Union{Tuple{I}, Tuple{AbstractVector{I}, I}} where I<:Integer","page":"Reference","title":"LinearOperators.opRestriction","text":"Z = opRestriction(I, ncol)\nZ = opRestriction(:, ncol)\n\nCreates a LinearOperator restricting a ncol-sized vector to indices I. The operation Z * v is equivalent to v[I]. I can be :.\n\nZ = opRestriction(k, ncol)\n\nAlias for opRestriction([k], ncol).\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.opZeros-Union{Tuple{I}, Tuple{DataType, I, I}} where I<:Integer","page":"Reference","title":"LinearOperators.opZeros","text":"opZeros(T, nrow, ncol; S = Vector{T})\nopZeros(nrow, ncol)\n\nZero operator of size nrow-by-ncol, of data type T (defaults to Float64). Change S to use LinearOperators on GPU.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.reset!-Tuple{AbstractLinearOperator}","page":"Reference","title":"LinearOperators.reset!","text":"reset!(op)\n\nReset the product counters of a linear operator.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.reset!-Tuple{LBFGSOperator}","page":"Reference","title":"LinearOperators.reset!","text":"reset!(op)\n\nResets the LBFGS data of the given operator.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.reset!-Tuple{LSR1Operator}","page":"Reference","title":"LinearOperators.reset!","text":"reset!(op)\n\nResets the LSR1 data of the given operator.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.reset!-Union{Tuple{AbstractDiagonalQuasiNewtonOperator{T}}, Tuple{T}} where T<:Real","page":"Reference","title":"LinearOperators.reset!","text":"reset!(op::AbstractDiagonalQuasiNewtonOperator)\n\nReset the diagonal data of the given operator.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.reset!-Union{Tuple{I}, Tuple{T}, Tuple{LinearOperators.LBFGSData{T, I}, Bool}} where {T, I<:Integer}","page":"Reference","title":"LinearOperators.reset!","text":"reset!(data)\n\nResets the given LBFGS data.\n\n\n\n\n\n","category":"method"},{"location":"reference/#LinearOperators.reset!-Union{Tuple{LinearOperators.LSR1Data{T, I}}, Tuple{I}, Tuple{T}} where {T, I<:Integer}","page":"Reference","title":"LinearOperators.reset!","text":"reset!(data)\n\nReset the given LSR1 data.\n\n\n\n\n\n","category":"method"},{"location":"#A-Julia-Linear-Operator-Package","page":"Home","title":"A Julia Linear Operator Package","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Operators behave like matrices (with exceptions) but are defined by their effect when applied to a vector. They can be transposed, conjugated, or combined with other operators cheaply. The costly operation is deferred until multiplied with a vector.","category":"page"},{"location":"#Compatibility","page":"Home","title":"Compatibility","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Julia 1.6 and up.","category":"page"},{"location":"#How-to-Install","page":"Home","title":"How to Install","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"pkg> add LinearOperators\npkg> test LinearOperators","category":"page"},{"location":"#Operators-Available","page":"Home","title":"Operators Available","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Operator Description\nLinearOperator Base class. Useful to define operators from functions\nTimedLinearOperator Linear operator instrumented with timers from TimerOutputs\nBlockDiagonalOperator Block-diagonal linear operator\nopEye Identity operator\nopOnes All ones operator\nopZeros All zeros operator\nopDiagonal Square (equivalent to diagm()) or rectangular diagonal operator\nopInverse Equivalent to \\\nopCholesky More efficient than opInverse for symmetric positive definite matrices\nopLDL Similar to opCholesky, for general sparse symmetric matrices\nopHouseholder Apply a Householder transformation I-2hh'\nopHermitian Represent a symmetric/hermitian operator based on the diagonal and strict lower triangle\nopRestriction Represent a selection of \"rows\" when composed on the left with an existing operator\nopExtension Represent a selection of \"columns\" when composed on the right with an existing operator\nLBFGSOperator Limited-memory BFGS approximation in operator form (damped or not)\nInverseLBFGSOperator Inverse of a limited-memory BFGS approximation in operator form (damped or not)\nLSR1Operator Limited-memory SR1 approximation in operator form\nkron Kronecker tensor product in linear operator form","category":"page"},{"location":"#Utility-Functions","page":"Home","title":"Utility Functions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Function Description\ncheck_ctranspose Cheap check that A' is correctly implemented\ncheck_hermitian Cheap check that A = A'\ncheck_positive_definite Cheap check that an operator is positive (semi-)definite\ndiag Extract the diagonal of an operator\nMatrix Convert an abstract operator to a dense array\nhermitian Determine whether the operator is Hermitian\npush! For L-BFGS or L-SR1 operators, push a new pair {s,y}\nreset! For L-BFGS or L-SR1 operators, reset the data\nshow Display basic information about an operator\nsize Return the size of a linear operator\nsymmetric Determine whether the operator is symmetric\nnormest Estimate the 2-norm","category":"page"},{"location":"#Other-Operations-on-Operators","page":"Home","title":"Other Operations on Operators","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Operators can be transposed (A.'), conjugated (conj(A)) and conjugate-transposed (A'). Operators can be sliced (A[:,3], A[2:4,1:5], A[1,1]), but unlike matrices, slices always return operators (see differences).","category":"page"},{"location":"#differences","page":"Home","title":"Differences","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Unlike matrices, an operator never reduces to a vector or a number.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using LinearOperators\nA = rand(5,5)\nopA = LinearOperator(A)\nA[:,1] * 3 isa Vector","category":"page"},{"location":"","page":"Home","title":"Home","text":"opA[:,1] * 3 isa LinearOperator","category":"page"},{"location":"","page":"Home","title":"Home","text":"opA[:,1] * [3] isa Vector","category":"page"},{"location":"","page":"Home","title":"Home","text":"However, the following returns an error","category":"page"},{"location":"","page":"Home","title":"Home","text":"A[:,1] * [3]","category":"page"},{"location":"","page":"Home","title":"Home","text":"This is also true for A[i,:], which would return a vector and for the scalar A[i,j]. Similarly, opA[1,1] is an operator of size (1,1):\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"(opA[1,1] * [3])[1] - A[1,1] * 3","category":"page"},{"location":"","page":"Home","title":"Home","text":"In the same spirit, the operator Matrix always returns a matrix.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Matrix(opA[:,1])","category":"page"},{"location":"#Other-Operators","page":"Home","title":"Other Operators","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"LLDL features a limited-memory LDLᵀ factorization operator that may be used as preconditioner in iterative methods\nMUMPS.jl features a full distributed-memory factorization operator that may be used to represent the preconditioner in, e.g., constraint-preconditioned Krylov methods.","category":"page"},{"location":"#Testing","page":"Home","title":"Testing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"julia> Pkg.test(\"LinearOperators\")","category":"page"},{"location":"#Bug-reports-and-discussions","page":"Home","title":"Bug reports and discussions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you think you found a bug, feel free to open an issue. Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.","category":"page"},{"location":"","page":"Home","title":"Home","text":"If you want to ask a question not suited for a bug report, feel free to start a discussion here. This forum is for general discussion about this repository and the JuliaSmoothOptimizers organization, so questions about any of our packages are welcome.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"You can check an Introduction to LinearOperators.jl on our site, jso.dev.","category":"page"}]
}
