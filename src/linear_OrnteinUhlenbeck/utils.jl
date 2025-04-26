function init_nodes_EQ(model::OUModel, T::Int)
    # Initialize nodes
    nodes = Vector{NodeEQ}(undef, model.N)
    @inbounds for i in 1:model.N
        neighs = findall(x -> x ≠ 0, view(model.J, i, :))
        nodes[i] = NodeEQ(i, neighs, T)
    end
    return nodes
end

"""
    compute_averages(nodes, model, T)

Compute the equilibrium averages over nodes of the corrlation and response functions.

# Arguments
- `nodes::Vector{NodeEQ}`: The nodes containing the cavity autocorrelations and the marginal fields.
- `model::OUModel`: The Ornstein-Uhlenbeck model.
- `T::Int`: The number of time steps.

# Returns
- `C_avg::OffsetVector{Float64, Vector{Float64}}`: The average correlation function over the nodes.
- `R_avg::OffsetVector{Float64, Vector{Float64}}`: The average response function over the nodes.
"""
function compute_averages(nodes::Vector{NodeEQ}, model::OUModel, T::Int)
    # Initialize averages vectors
    C_avg = OffsetVector(zeros(T+1), 0:T)
    R_avg = OffsetVector(zeros(T+1), 0:T)
    # Compute the averages of the fields
    @inbounds @fastmath for node in nodes
        C_avg .+= node.marg.C
        R_avg .+= node.marg.R
    end
    # Normalize the averages
    C_avg ./= model.N
    R_avg ./= model.N
    # Return the averages
    return C_avg, R_avg
end


function init_nodes(model::OUModel, T::Int)
    # Initialize nodes
    nodes = Vector{Node}(undef, model.N)
    @inbounds for i in 1:model.N
        neighs = findall(x -> x ≠ 0, view(model.J, i, :))
        nodes[i] = Node(i, neighs, T)
    end
    return nodes
end

"""
    compute_averages(nodes, model, T)

Compute the averages over nodes of the corrlation and response functions.

# Arguments
- `nodes::Vector{Node}`: The nodes containing the cavity autocorrelations and the marginal fields.
- `model::OUModel`: The Ornstein-Uhlenbeck model.
- `T::Int`: The number of time steps.

# Returns
- `C_avg::OffsetMatrix{Float64, Matrix{Float64}}`: The average correlation function over the nodes.
- `R_avg::OffsetMatrix{Float64, Matrix{Float64}}`: The average response function over the nodes.
"""
function compute_averages(nodes::Vector{Node}, model::OUModel, T::Int)
    # Initialize averages vectors
    C_avg = OffsetMatrix(zeros(T+1, T+1), 0:T, 0:T)
    R_avg = OffsetMatrix(zeros(T+1, T+1), 0:T, 0:T)
    # Compute the averages of the fields
    @inbounds @fastmath for node in nodes
        C_avg .+= node.marg.C
        R_avg .+= node.marg.R
    end
    # Normalize the averages
    C_avg ./= model.N
    R_avg ./= model.N
    # Return the averages
    return C_avg, R_avg
end

"""
    compute_mean(nodes, model, T)

Compute the mean over nodes of the average degree of freedom.

# Arguments
- `nodes::Vector{Node}`: The nodes containing the cavity autocorrelations and the marginal fields.
- `model::OUModel`: The Ornstein-Uhlenbeck model.
- `T::Int`: The number of time steps.

# Returns
- `mu::Float64`: The mean of the average degree of freedom over the nodes.
"""
function compute_mean(nodes::Vector{Node}, model::OUModel, T::Int)
    # Initialize mean
    mu = OffsetVector(zeros(T+1), 0:T)
    # Compute the mean of the fields
    @inbounds @fastmath for node in nodes
        mu .+= node.marg.mu
    end
    # Normalize the mean
    mu ./= model.N
    # Return the mean
    return mu
end