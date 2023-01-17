@testset "CompressedLBFGS operator" begin
  iter=50
  n=100
  n=5
  lbfgs = CompressedLBFGS(n) # m=5
  Bv = rand(n)
  for i in 1:iter
    s = rand(n)
    y = rand(n)
    push!(lbfgs, s, y)
    # warmp-up computing the mandatory intermediate structures
    mul!(Bv, lbfgs, s)
    allocs = @allocated mul!(Bv, lbfgs, s)
    @test allocs == 0
    @test Bv ≈ y
  end  
end
