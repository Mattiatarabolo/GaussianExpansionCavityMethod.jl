using GaussianExpansionCavityMethod
using Test
using Aqua

@testset "GaussianExpansionCavityMethod.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(GaussianExpansionCavityMethod)
    end
    # Write your tests here.
end
