@testset "CompressedLBFGS operator" begin
  iter=50
  n=100
  n=5
  lbfgs = CompressedLBFGS(n) # m=5
  V = LinearOperators.default_vector_type(LinearOperators.default_gpu())
  Bv = V(rand(n))
  s = V(rand(n))
  mul!(Bv, lbfgs, s) # warm-up
  for i in 1:iter
    s = V(rand(n))
    y = V(rand(n))
    push!(lbfgs, s, y)
    # warmp-up computing the mandatory intermediate structures
    allocs = @allocated mul!(Bv, lbfgs, s)
    @test allocs == 0
    @test Bv ≈ y
  end  
end
