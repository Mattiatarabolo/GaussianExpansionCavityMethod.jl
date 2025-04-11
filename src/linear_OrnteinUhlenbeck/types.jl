"""
    OUModel

A structure representing a linearly-coupled Ornstein-Uhlenbeck model with its associated parameters.

# Fields
- `N::Int`: The number of nodes in the model.
- `K::Union{Int, Float64}`: The average number of neighbors.
- `J::SparseMatrixCSC{Float64, Int}`: The coupling matrix.
- `lambdas::Vector{Float64}`: The local decay rates.
- `D::Float64`: The diffusion coefficient.

# Description
The `OUModel` structure represents a linearly-coupled Ornstein-Uhlenbeck model, which is a stochastic process used to model the dynamics of a system with local decay rates and coupling between nodes. Each model has a number of nodes `N`, local decay rates `lambdas`, a coupling matrix `J`, and a diffusion coefficient `D`.

# Methods
"""
struct OUModel
    N::Int
    K::Union{Int, Float64}
    J::SparseMatrixCSC{Float64, Int}
    lambdas::Vector{Float64}
    D::Float64
    """
        OUModel(K, J, lambdas, D)

    Construct an linearly-coupled Ornstein-Uhlenbeck model with the given parameters.

    # Arguments
    - `K::Union{Int, Float64}`: The average number of neighbors.
    - `J::SparseMatrixCSC{Float64, Int}`: The coupling matrix.
    - `lambdas::Vector{Float64}`: The local decay rates.
    - `D::Float64`: The diffusion coefficient.

    # Returns
    - `OUModel`: The constructed OU model.
    """
    function OUModel(K::Union{Int, Float64}, J::SparseMatrixCSC{Float64, Int}, lambdas::Vector{Float64}, D::Float64)
        @assert length(lambdas) == size(J, 1) "The length of lambdas must match the number of rows in J."
        @assert size(J, 1) == size(J, 2) "The coupling matrix J must be square."
        N = length(lambdas)
        new(N, K, J, lambdas, D)
    end
    """
        OUModel(K, J, lambda, D)

    Construct an linearly-coupled Ornstein-Uhlenbeck model with the given parameters.

    # Arguments
    - `K::Union{Int, Float64}`: The average number of neighbors.
    - `J::SparseMatrixCSC{Float64, Int}`: The coupling matrix.
    - `lambda::Float64`: The uniform local decay rate.
    - `D::Float64`: The diffusion coefficient.

    # Returns
    - `OUModel`: The constructed OU model.
    """
    function OUModel(K::Union{Int, Float64}, J::SparseMatrixCSC{Float64, Int}, lambda::Float64, D::Float64)
        @assert size(J, 1) == size(J, 2) "The coupling matrix J must be square."
        N = size(J, 1)
        lambdas = fill(lambda, N)
        new(N, K, J, lambdas, D)
    end
end

"""
    OUModelEnsemble

A type representing the ensemble of disordered Ornstein-Uhlenbeck model.

# Fields
- `N::Integer`: The number of sites.
- `K::Union{Integer, Float64}`: The average number of neighbors.
- `gen_J::Function`: The function to generate the sparse coupling matrix.
- `J_params::Vector{Float64}`: The parameters for the coupling matrix generator.
- `lambdas::Vector{Float64}`: The local decay rates.
- `D::Float64`: The diffusion coefficient.

"""
struct OUModelEnsemble
    N::Integer
    K::Union{Int, Float64}
    gen_J::Function
    J_params::Vector{Float64}
    lambdas::Vector{Float64}
    D::Float64
    gen_x0::Function
    x0_params::Vector{Float64}
    """
        OUModelEnsemble(N, K, gen_J, J_params, lambdas, D, gen_x0, x0_params)
    
    Construct a Ornstein-Uhlenbeck model ensemble.

    # Arguments
    - `N::Integer`: The number of sites.
    - `K::Union{Integer, Float64}`: The average number of neighbors.
    - `gen_J::Function`: The function to generate the sparse coupling matrix.
    - `J_params::Vector{Float64}`: The parameters for the coupling matrix generator.
    - `lambdas::Vector{Float64}`: The local decay rates.
    - `D::Float64`: The diffusion coefficient.
    - `gen_x0::Function`: The function to generate the initial conditions.
    - `x0_params::Vector{Float64}`: The parameters for the initial condition generator.

    # Returns
    - `OUModelEnsemble`: The Ornstein-Uhlenbeck model ensemble.
    """
    function OUModelEnsemble(N::Int, K::Union{Int, Float64}, gen_J::Function, J_params::Vector{Float64}, lambdas::Vector{Float64}, D::Float64, gen_x0::Function, x0_params::Vector{Float64})
        new(N, K, gen_J, J_params, lambdas, D, gen_x0, x0_params)
    end
end

"""
    OUModelRRG(N, K, J, lambdas, D, x0_min, x0_max)

Construct a Ornstein-Uhlenbeck model with a random regular graph coupling matrix with ferromagnetic interactions.

# Arguments
- `N::Integer`: The number of sites.
- `K::Union{Int, Float64}`: The average number of neighbors.
- `J::Float64`: The coupling strength.
- `lambdas::Vector{Float64}`: The local decay rates.
- `D::Float64`: The diffusion coefficient.
- `x0_min::Float64`: The minimum initial condition.
- `x0_max::Float64`: The maximum initial condition.

# Returns
- `OUModelEnsemble`: The Ornstein-Uhlenbeck model.
"""
function OUModelRRG(N::Int, K::Union{Int, Float64}, J::Float64, lambdas::Vector{Float64}, D::Float64, x0_min::Float64, x0_max::Float64)
    J_params = [J]
    gen_J = (N, K, J_params; rng=Xoshiro(1234)) -> adjacency_matrix(random_regular_graph(N, K; rng=rng)) .* J_params[1]
    x0_params = [x0_min, x0_max]
    gen_x0 = (N, x0_params; rng=Xoshiro(1234)) -> rand(rng, N) .* (x0_params[2] - x0_params[1]) .+ x0_params[1]
    OUModelEnsemble(N, K, gen_J, J_params, lambdas, D, gen_x0, x0_params)
end
"""
    OUModelRRG(N, K, J, lambda, D, x0_min, x0_max)

Construct a Ornstein-Uhlenbeck model with a random regular graph coupling matrix with ferromagnetic interactions.

# Arguments
- `N::Integer`: The number of sites.
- `K::Union{Int, Float64}`: The average number of neighbors.
- `J::Float64`: The coupling strength.
- `lambda::Float64`: The uniform local decay rates.
- `D::Float64`: The diffusion coefficient.
- `x0_min::Float64`: The minimum initial condition.
- `x0_max::Float64`: The maximum initial condition.

# Returns
- `OUModelEnsemble`: The Ornstein-Uhlenbeck model.
"""
function OUModelRRG(N::Int, K::Union{Int, Float64}, J::Float64, lambda::Float64, D::Float64, x0_min::Float64, x0_max::Float64)
    J_params = [J]
    gen_J = (N, K, J_params; rng=Xoshiro(1234)) -> adjacency_matrix(random_regular_graph(N, K; rng=rng)) .* J_params[1]
    lambdas = fill(lambda, N)
    x0_params = [x0_min, x0_max]
    gen_x0 = (N, x0_params; rng=Xoshiro(1234)) -> rand(rng, N) .* (x0_params[2] - x0_params[1]) .+ x0_params[1]
    OUModelEnsemble(N, K, gen_J, J_params, lambdas, D, gen_x0, x0_params)
end




###############################################################################################################################################
########################################################### CAVITY METHOD TYPES  ##############################################################
###############################################################################################################################################

"""
    Cavity

A structure representing a cavity in a graph with its associated parameters.

# Fields
- `i::Int`: The index of the node.
- `j::Int`: The index of the neighbor.
- `mu::OffsetVector{Float64, Vector{Float64}}`: The mean vector.
- `C::OffsetMatrix{Float64, Matrix{Float64}}`: The autocorellation matrix.
- `R::OffsetMatrix{Float64, Matrix{Float64}}`: The response matrix.

# Description
The `Cavity` structure represents a cavity in a graph, which is a subgraph formed by removing a node and its associated edges. Each cavity has an index `i`, a neighbor index `j`, and associated cavity fields such as the mean vector `mu`, the autocorrelation matrix `C`, and the response matrix `R`.

# Methods
- `Cavity(i::Int, j::Int, T::Int)`: Constructs a cavity with index `i`, neighbor index `j`, and `T` timesteps.
"""
struct Cavity
    i::Int
    j::Int
    mu::OffsetVector{Float64, Vector{Float64}}
    C::OffsetMatrix{Float64, Matrix{Float64}}
    R::OffsetMatrix{Float64, Matrix{Float64}}
    """
        Cavity(i, j, T)

    Construct a cavity with index `i`, neighbor index `j`, and `T` timesteps.

    # Arguments
    - `i::Int`: The index of the node.
    - `j::Int`: The index of the neighbor.
    - `T::Int`: The number of timesteps.

    # Returns
    - `Cavity`: The constructed cavity.
    """
    function Cavity(i::Int, j::Int, T::Int)
        mu = OffsetVector(zeros(T+1), 0:T)
        C = OffsetMatrix(zeros(T+1, T+1), 0:T, 0:T)
        R = OffsetMatrix(zeros(T+1, T+1), 0:T, 0:T)
        new(i, j, mu, C, R)
    end
end

"""
    Marginal

A structure representing a marginal in a graph with its associated parameters.

# Fields
- `i::Int`: The index of the node.
- `mu::OffsetVector{Float64, Vector{Float64}}`: The mean vector.
- `C::OffsetMatrix{Float64, Matrix{Float64}}`: The autocorellation matrix.
- `R::OffsetMatrix{Float64, Matrix{Float64}}`: The response matrix.

# Description
The `Marginal` structure represents a marginal in a graph, which is a subgraph formed by removing a node and its associated edges. Each marginal has an index `i`, and associated marginal fields such as the mean vector `mu`, the autocorrelation matrix `C`, and the response matrix `R`.

# Methods
- `Marginal(i::Int, T::Int)`: Constructs a marginal with index `i` and `T` timesteps.
"""
struct Marginal
    i::Int
    mu::OffsetVector{Float64, Vector{Float64}}
    C::OffsetMatrix{Float64, Matrix{Float64}}
    R::OffsetMatrix{Float64, Matrix{Float64}}
    """
        Marginal(i, T)

    Construct a marginal with index `i` and `T` timesteps.
    
    # Arguments
    - `i::Int`: The index of the node.
    - `T::Int`: The number of timesteps.

    # Returns
    - `Marginal`: The constructed marginal.
    """
    function Marginal(i::Int, T::Int)
        mu = OffsetVector(zeros(T+1), 0:T)
        C = OffsetMatrix(zeros(T+1, T+1), 0:T, 0:T)
        R = OffsetMatrix(zeros(T+1, T+1), 0:T, 0:T)
        new(i, mu, C, R)
    end
end

"""
    Node

A structure representing a node in a graph with its neighbors and associated cavities and marginal.

# Fields
- `i::Int`: The index of the node.
- `neighs::Vector{Int}`: The indices of the neighbors.
- `cavs::Vector{Cavity}`: The cavities associated with the node.
- `marg::Marginal`: The marginal associated with the node.

# Description
The `Node` structure represents a node in a graph, along with its neighbors and associated cavities and marginal. Each node has an index `i`, a vector of neighbor indices `neighs`, a vector of cavities `cavs` and a marginal `marg`.

# Methods
- `Node(i::Int, neighs::Vector{Int}, T::Int)`: Constructs a node with index `i`, neighbors `neighs`, and `T` timesteps.
"""
struct Node
    i::Int
    neighs::Vector{Int}
    cavs::Vector{Cavity}
    marg::Marginal
    """
        Node(i, neighs, T)

    Construct a node with index `i`, neighbors `neighs`, and `T` timesteps.

    # Arguments
    - `i::Int`: The index of the node.
    - `neighs::Vector{Int}`: The indices of the neighbors.
    - `T::Int`: The number of timesteps.
    
    # Returns
    - `Node`: The constructed node.
    """
    function Node(i::Int, neighs::Vector{Int}, T::Int)
        cavs = [Cavity(i, neighs[j], T) for j in 1:length(neighs)]
        margs = Marginal(i, T)
        new(i, neighs, cavs, margs)
    end
end

#################################### Equilibrium (Eq) case (t, t' -> ∞ with t - t' = tau) ######################################################

"""
    CavityEQ

A structure representing a equilibrium cavity in a graph with its associated parameters.

# Fields
- `i::Int`: The index of the node.
- `j::Int`: The index of the neighbor.
- `C::OffsetVector{Float64, Vector{Float64}}`: The autocorellation vector.

# Description
The `CavityEQ` structure represents a equilibrium cavity in a graph, which is a subgraph formed by removing a node and its associated edges. Each cavity has an index `i`, a neighbor index `j`, and associated cavity field such as the autocorrelation vector `C`.

# Methods
- `CavityEQ(i::Int, j::Int, T::Int)`: Constructs a equilibrium cavity with index `i`, neighbor index `j`, and `T` timesteps.
"""
struct CavityEQ
    i::Int
    j::Int
    C::OffsetVector{Float64, Vector{Float64}}
    """
        CavityEQ(i, j, T)

    Construct a equilibrium cavity with index `i`, neighbor index `j`, and `T` timesteps.

    # Arguments
    - `i::Int`: The index of the node.
    - `j::Int`: The index of the neighbor.
    - `T::Int`: The number of timesteps.

    # Returns
    - `CavityEQ`: The constructed equilibrium cavity.
    """
    function CavityEQ(i::Int, j::Int, T::Int)
        C = OffsetVector(zeros(T+1), 0:T)
        new(i, j, C)
    end
end

"""
    MarginalEQ

A structure representing a equilibrium marginal in a graph with its associated parameters.

# Fields
- `i::Int`: The index of the node.
- `mu::Float64`: The mean at infinity.
- `C::OffsetVector{Float64, Vector{Float64}}`: The autocorellation vector.
- `R::OffsetVector{Float64, Vector{Float64}}`: The response vector.

# Description
The `MarginalEQ` structure represents a equilibrium marginal in a graph, which is a subgraph formed by removing a node and its associated edges. Each marginal has an index `i`, and associated marginal fields such as the mean at infinity `mu`, the autocorrelation vector `C`, and the response vector `R`.

# Methods
- `MarginalEQ(i::Int, T::Int)`: Constructs an equilibrium marginal with index `i` and `T` timesteps.
"""
struct MarginalEQ
    i::Int
    mu::Float64
    C::OffsetVector{Float64, Vector{Float64}}
    R::OffsetVector{Float64, Vector{Float64}}
    """
        MarginalEQ(i, T)

    Construct a equilibrium marginal with index `i` and `T` timesteps.
    
    # Arguments
    - `i::Int`: The index of the node.
    - `T::Int`: The number of timesteps.

    # Returns
    - `MarginalEQ`: The constructed equilibrium marginal.
    """
    function MarginalEQ(i::Int, T::Int)
        C = OffsetVector(zeros(T+1), 0:T)
        R = OffsetVector(zeros(T+1), 0:T)
        new(i, 0.0, C, R)
    end
end

"""    
    NodeEQ

A structure representing a equilibrium node in a graph with its neighbors and associated cavities and marginal.

# Fields
- `i::Int`: The index of the node.
- `neighs::Vector{Int}`: The indices of the neighbors.
- `neighs_idxs::Dict{Int, Int}`: A dictionary mapping neighbor indices to their positions in the `neighs` vector.
- `cavs::Vector{CavityEQ}`: The equilibrium cavities associated with the node.
- `marg::MarginalEQ`: The equilibrium marginal associated with the node.
- `sumC::OffsetVector{Float64, Vector{Float64}}`: Internal variable.
- `sumdiffC::OffsetVector{Float64, Vector{Float64}}`: Internal variable.

# Description
The `NodeEQ` structure represents a equilibrium node in a graph, along with its neighbors and associated equilibrium cavities and marginal. Each node has an index `i`, a vector of neighbor indices `neighs`, a vector of equilibrium cavities `cavs` and a equilibrium marginal `marg`.

# Methods
- `NodeEQ(i::Int, neighs::Vector{Int}, T::Int)`: Constructs a node with index `i`, neighbors `neighs`, and `T` timesteps.
"""
mutable struct NodeEQ
    i::Int
    neighs::Vector{Int}
    neighs_idxs::Dict{Int, Int}
    cavs::Vector{CavityEQ}
    marg::MarginalEQ
    sumC::OffsetVector{Float64, Vector{Float64}}
    sumdiffC::OffsetVector{Float64, Vector{Float64}}
    """
        NodeEQ(i, neighs, T)

    Construct a EQ node with index `i`, neighbors `neighs`, and `T` timesteps.

    # Arguments
    - `i::Int`: The index of the node.
    - `neighs::Vector{Int}`: The indices of the neighbors.
    - `T::Int`: The number of timesteps.
    
    # Returns
    - `NodeEQ`: The constructed equilibrium node.
    """
    function NodeEQ(i::Int, neighs::Vector{Int}, T::Int)
        neighs_idxs = Dict{Int, Int}(neighs[j] => j for j in 1:length(neighs))
        cavs = [CavityEQ(i, neighs[j], T) for j in 1:length(neighs)]
        margs = MarginalEQ(i, T)
        sumC = OffsetVector(zeros(T), 0:T-1)
        sumdiffC = OffsetVector(zeros(T-1), 0:T-2)
        new(i, neighs, neighs_idxs, cavs, margs, sumC, sumdiffC)
    end
end