using GaussianExpansionCavityMethod
using Test
using Aqua

@testset "GaussianExpansionCavityMethod.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(GaussianExpansionCavityMethod, deps_compat=(check_extras=false, check_weakdeps=false))    
    end
    # Write your tests here.
end
