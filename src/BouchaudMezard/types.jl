"""
    BMModel

A type representing the Bouchaud-Mezard model.

# Fields
- `N::Integer`: The number of sites.
- `K::Union{Integer, Float64}`: The average number of neighbors.
- `J::SparseMatrixCSC{Float64, Integer}`: The sparse coupling matrix.
- `sigma::Float64`: The multiplicative noise strength.
"""
struct BMModel
    N::Integer
    K::Union{Int, Float64}
    J::SparseMatrixCSC{Float64, Int}
    sigma::Float64
    """
        BMModel(K, J, sigma)

    Construct a Bouchaud-Mezard model.

    # Arguments
    - `K::Union{Int, Float64}`: The average number of neighbors.
    - `J::SparseMatrixCSC{Float64, Int}`: The sparse coupling matrix.
    - `sigma::Float64`: The multiplicative noise strength.

    # Returns
    - `BMModel`: The Bouchaud-Mezard model.
    """
    function BMModel(K::Union{Int, Float64}, J::SparseMatrixCSC{Float64, Int}, sigma::Float64)
        N = size(J, 1)
        new(N, K, J, sigma)
    end
end



"""
    BMModelEnsemble

A type representing the ensemble of disordered Bouchaud-Mezard model.

# Fields
- `N::Integer`: The number of sites.
- `K::Union{Integer, Float64}`: The average number of neighbors.
- `gen_J::Function`: The function to generate the sparse coupling matrix.
- `J_params::Vector{Float64}`: The parameters for the coupling matrix generator.
- `sigma::Float64`: The multiplicative noise strength.
"""
struct BMModelEnsemble
    N::Integer
    K::Union{Int, Float64}
    gen_J::Function
    J_params::Vector{Float64}
    sigma::Float64
    """
        BMModelEnsemble(N, K, gen_J, J_params, sigma)
    
    Construct a Bouchaud-Mezard  model ensemble.

    # Arguments
    - `N::Integer`: The number of sites.
    - `K::Union{Int, Float64}`: The average number of neighbors.
    - `gen_J::Function`: The function to generate the sparse coupling matrix.
    - `J_params::Vector{Float64}`: The parameters for the coupling matrix generator.
    - `sigma::Float64`: The multiplicative noise strength.

    # Returns
    - `BMModelEnsemble`: The Bouchaud-Mezard model ensemble.
    """
    function BMModelEnsemble(N::Int, K::Union{Int, Float64}, gen_J::Function, J_params::Vector{Float64}, sigma::Float64)
        new(N, K, gen_J, J_params, sigma)
    end
end

"""
    BMModelRRG(N, K, J, lambda, sigma)

Construct a Bouchaud-Mezard model with a random regular graph coupling matrix with ferromagnetic interactions.

# Arguments
- `N::Integer`: The number of sites.
- `K::Union{Int, Float64}`: The average number of neighbors.
- `J::Float64`: The coupling strength.
- `sigma::Float64`: The multiplicative noise strength.

# Returns
- `BMModelEnsemble`: The Bouchaud-Mezard model.
"""
function BMModelRRG(N::Int, K::Union{Int, Float64}, J::Float64, sigma::Float64)
    J_params = [J]
    gen_J = (N, K, J_params; rng=Xoshiro(1234)) -> adjacency_matrix(random_regular_graph(N, K; rng=rng)) .* J_params[1]
    BMModelEnsemble(N, K, gen_J, J_params, sigma)
end

