"""
    TwoSpinModel

A type representing the spherical 2-Spin model.

# Fields
- `N::Integer`: The number of sites.
- `K::Union{Integer, Float64}`: The average number of neighbors.
- `J::SparseMatrixCSC{Float64, Integer}`: The sparse coupling matrix.
- `D::Float64`: The noise strength.
"""
struct TwoSpinModel
    N::Integer
    K::Union{Int, Float64}
    J::SparseMatrixCSC{Float64, Int}
    D::Float64
    """
        TwoSpinModel(K, J, sigma)

    Construct a spherical 2-Spin  model.

    # Arguments
    - `K::Union{Int, Float64}`: The average number of neighbors.
    - `J::SparseMatrixCSC{Float64, Int}`: The sparse coupling matrix.
    - `D::Float64`: The noise strength.

    # Returns
    - `TwoSpinModel`: The spherical 2-Spin  model.
    """
    function TwoSpinModel(K::Union{Int, Float64}, J::SparseMatrixCSC{Float64, Int}, D::Float64)
        N = size(J, 1)
        new(N, K, J, D)
    end
end



"""
    TwoSpinModelEnsemble

A type representing the ensemble of disordered spherical 2-Spin model.

# Fields
- `N::Integer`: The number of sites.
- `K::Union{Integer, Float64}`: The average number of neighbors.
- `gen_J::Function`: The function to generate the sparse coupling matrix.
- `J_params::Vector{Float64}`: The parameters for the coupling matrix generator.
- `D::Float64`: The noise strength.
"""
struct TwoSpinModelEnsemble
    N::Integer
    K::Union{Int, Float64}
    gen_J::Function
    J_params::Vector{Float64}
    D::Float64
    """
        TwoSpinModelEnsemble(N, K, gen_J, J_params, D)
    
    Construct a spherical 2-Spin model ensemble.

    # Arguments
    - `N::Integer`: The number of sites.
    - `K::Union{Int, Float64}`: The average number of neighbors.
    - `gen_J::Function`: The function to generate the sparse coupling matrix.
    - `J_params::Vector{Float64}`: The parameters for the coupling matrix generator.
    - `D::Float64`: The noise strength.

    # Returns
    - `TwoSpinModelEnsemble`: The spherical 2-Spin  model ensemble.
    """
    function TwoSpinModelEnsemble(N::Int, K::Union{Int, Float64}, gen_J::Function, J_params::Vector{Float64}, D::Float64)
        new(N, K, gen_J, J_params, D)
    end
end

"""
    TwoSpinModelRRG(N, K, J, lambda, D)

Construct a spherical 2-Spin model with a random regular graph coupling matrix with iid symmetric couplings Jᵢⱼ = Jⱼᵢ ~ p(J).

# Arguments
- `N::Integer`: The number of sites.
- `K::Union{Int, Float64}`: The average number of neighbors.
- `pJ::Function`: The coupling distribution.
- `J_params::Vector{Float64}`: The parameters for the coupling distribution.
- `D::Float64`: The noise strength.

# Returns
- `TwoSpinModelEnsemble`: The spherical 2-Spin  model.
"""
function TwoSpinModelRRG(N::Int, K::Union{Int, Float64}, pJ::Function, J_params::Vector{Float64}, D::Float64)
    function gen_J(N, K, J_params; rng=Xoshiro(1234))
        J = adjacency_matrix(random_regular_graph(N, K; rng=rng), Float64)
        @inbounds @fastmath for i in 1:N
            @inbounds @fastmath for j in i+1:N
                if J[i, j] ≠ 0
                    Jval = pJ(J_params; rng=rng)
                    J[i, j] = Jval
                    J[j, i] = Jval
                end
            end
        end
        dropzeros!(J)
        return J
    end
    TwoSpinModelEnsemble(N, K, gen_J, J_params, D)
end

"""
    TwoSpinModelRRG_Bim(N, K, J, lambda, D)

Construct a spherical 2-Spin model with a random regular graph coupling matrix with bimodal interactions.

# Arguments
- `N::Integer`: The number of sites.
- `K::Union{Int, Float64}`: The average number of neighbors.
- `J::Float64`: The coupling strength.
- `D::Float64`: The noise strength.

# Returns
- `TwoSpinModelEnsemble`: The spherical 2-Spin  model.
"""
function TwoSpinModelRRG_Bim(N::Int, K::Union{Int, Float64}, J::Float64, D::Float64)
    J_params = [J]
    pJ(J_params; rng=Xoshiro(1234)) = J_params[1]/sqrt(K) * (rand(rng) < 0.5 ? 1 : -1)
    TwoSpinModelRRG(N, K, pJ, J_params, D)
end

"""
    TwoSpinModelRRG_Ferro(N, K, J, lambda, D)

Construct a spherical 2-Spin model with a random regular graph coupling matrix with ferromagnetic interactions.

# Arguments
- `N::Integer`: The number of sites.
- `K::Union{Int, Float64}`: The average number of neighbors.
- `J::Float64`: The coupling strength.
- `D::Float64`: The noise strength.

# Returns
- `TwoSpinModelEnsemble`: The spherical 2-Spin  model.
"""
function TwoSpinModelRRG_Ferro(N::Int, K::Union{Int, Float64}, J::Float64, D::Float64)
    J_params = [J]
    pJ(J_params; rng=Xoshiro(1234)) = J_params[1]/K
    TwoSpinModelRRG(N, K, pJ, J_params, D)
end