"""
    Phi4Model

A type representing the Phi^4 model.

# Fields
- `N::Integer`: The number of sites.
- `K::Union{Integer, Float64}`: The average number of neighbors.
- `J::SparseMatrixCSC{Float64, Integer}`: The sparse coupling matrix.
- `lambdas::Vector{Float64}`: The on-site linear terms.
- `D::Float64`: The noise strength.
- `u::Float64`: The cubic perturbative constant.
"""
struct Phi4Model
    N::Integer
    K::Union{Int, Float64}
    J::SparseMatrixCSC{Float64, Int}
    lambdas::Vector{Float64}
    D::Float64
    u::Float64
    """
        Phi4Model(K, J, lambdas, D, u)

    Construct a Phi^4 model.

    # Arguments
    - `K::Union{Int, Float64}`: The average number of neighbors.
    - `J::SparseMatrixCSC{Float64, Int}`: The sparse coupling matrix.
    - `lambdas::Vector{Float64}`: The on-site linear terms.
    - `D::Float64`: The noise strength.
    - `u::Float64`: The cubic perturbative constant.

    # Returns
    - `Phi4Model`: The Phi^4 model.
    """
    function Phi4Model(K::Union{Int, Float64}, J::SparseMatrixCSC{Float64, Int}, lambdas::Vector{Float64}, D::Float64, u::Float64)
        N = size(J, 1)
        new(N, K, J, lambdas, D, u)
    end
    """
        Phi4Model(K, J, lambda, D, u)

    Construct a Phi^4 model.

    # Arguments
    - `K::Union{Int, Float64}`: The average number of neighbors.
    - `J::SparseMatrixCSC{Float64, Int}`: The sparse coupling matrix.
    - `lambda::Float64`: The uniform on-site linear term.
    - `D::Float64`: The noise strength.
    - `u::Float64`: The cubic perturbative constant.

    # Returns
    - `Phi4Model`: The Phi^4 model.
    """
    function Phi4Model(K::Union{Int, Float64}, J::SparseMatrixCSC{Float64, Int}, lambda::Float64, D::Float64, u::Float64)
        N = size(J, 1)
        new(N, K, J, fill(lambda, N), D, u)
    end
end



"""
    Phi4ModelEnsemble

A type representing the ensemble of disordered Phi^4 model.

# Fields
- `N::Integer`: The number of sites.
- `K::Union{Integer, Float64}`: The average number of neighbors.
- `gen_J::Function`: The function to generate the sparse coupling matrix.
- `J_params::Vector{Float64}`: The parameters for the coupling matrix generator.
- `lambdas::Vector{Float64}`: The on-site linear terms.
- `D::Float64`: The noise strength.
- `u::Float64`: The cubic perturbative constant.
"""
struct Phi4ModelEnsemble
    N::Integer
    K::Union{Int, Float64}
    gen_J::Function
    J_params::Vector{Float64}
    lambdas::Vector{Float64}
    D::Float64
    u::Float64
    """
        Phi4ModelEnsemble(N, K, gen_J, J_params, lambdas, D, u)
    
    Construct a Phi^4 model ensemble.

    # Arguments
    - `N::Integer`: The number of sites.
    - `K::Union{Int, Float64}`: The average number of neighbors.
    - `gen_J::Function`: The function to generate the sparse coupling matrix.
    - `J_params::Vector{Float64}`: The parameters for the coupling matrix generator.
    - `lambdas::Vector{Float64}`: The on-site linear terms.
    - `D::Float64`: The noise strength.
    - `u::Float64`: The cubic perturbative constant.

    # Returns
    - `Phi4ModelEnsemble`: The Phi^4 model ensemble.
    """
    function Phi4ModelEnsemble(N::Int, K::Union{Int, Float64}, gen_J::Function, J_params::Vector{Float64}, lambdas::Vector{Float64}, D::Float64, u::Float64)
        new(N, K, gen_J, J_params, lambdas, D, u)
    end
end

"""
    Phi4ModelRRG(N, K, J, lambdas, D, u)

Construct a Phi^4 model with a random regular graph coupling matrix with ferromagnetic interactions.

# Arguments
- `N::Integer`: The number of sites.
- `K::Union{Int, Float64}`: The average number of neighbors.
- `J::Float64`: The coupling strength.
- `lambdas::Vector{Float64}`: The on-site linear terms.
- `D::Float64`: The noise strength.
- `u::Float64`: The cubic perturbative constant.

# Returns
- `Phi4ModelEnsemble`: The Phi^4 model.
"""
function Phi4ModelRRG(N::Int, K::Union{Int, Float64}, J::Float64, lambdas::Vector{Float64}, D::Float64, u::Float64)
    J_params = [J]
    gen_J = (N, K, J_params; rng=Xoshiro(1234)) -> adjacency_matrix(random_regular_graph(N, K; rng=rng)) .* J_params[1]
    Phi4ModelEnsemble(N, K, gen_J, J_params, lambdas, D, u)
end
"""
    Phi4ModelRRG(N, K, J, lambda, D, u)

Construct a Phi^4 model with a random regular graph coupling matrix with ferromagnetic interactions.

# Arguments
- `N::Integer`: The number of sites.
- `K::Union{Int, Float64}`: The average number of neighbors.
- `J::Float64`: The coupling strength.
- `lambda::Float64`: The on-site linear term.
- `D::Float64`: The noise strength.
- `u::Float64`: The cubic perturbative constant.

# Returns
- `Phi4ModelEnsemble`: The Phi^4 model.
"""
function Phi4ModelRRG(N::Int, K::Union{Int, Float64}, J::Float64, lambda::Float64, D::Float64, u::Float64)
    J_params = [J]
    gen_J = (N, K, J_params; rng=Xoshiro(1234)) -> adjacency_matrix(random_regular_graph(N, K; rng=rng)) .* J_params[1]
    Phi4ModelEnsemble(N, K, gen_J, J_params, fill(lambda, N), D, u)
end

