##############################################################################################################################
########################### Time Translational Invariant (EQ) case (t, t' -> ∞ with t - t' = tau) ###########################
##############################################################################################################################

function update_sumC!(sumC::OffsetVector{Float64, Vector{Float64}}, sumdiffC::OffsetVector{Float64, Vector{Float64}}, inode::NodeEQ, model::OUModel, n::Int)
    i = inode.i
    @inbounds @fastmath for (kindex, k) in enumerate(inode.neighs)
        # Compute the sum of Cavity fields
        sumC[n] += inode.cavs[kindex].C[n] * model.J[i, k]
        sumdiffC[n] += (inode.cavs[kindex].C[n+1] - inode.cavs[kindex].C[n]) * model.J[i, k] ^ 2
    end
end

function compute_integral(sumdiffC::OffsetVector{Float64, Vector{Float64}}, Cij::OffsetVector{Float64, Vector{Float64}}, Cji::OffsetVector{Float64, Vector{Float64}}, Jij, n)
    int = 0.0
    @inbounds @fastmath @simd for m in 0:n-1
        int += Cij[n-m] * (sumdiffC[m] - Jij ^ 2 * (Cji[m+1] - Cji[m]))
    end
    return int
end

function update_Ceq_cav!(inode::NodeEQ, nodes::Vector{NodeEQ}, model::OUModel, dt::Float64, sumC::OffsetVector{Float64, Vector{Float64}}, sumdiffC::OffsetVector{Float64, Vector{Float64}}, n::Int)
    # Unpack parameters
    i = inode.i
    lambda = model.lambdas[i]
    D = model.D
    # Update sumC and sumdiffC at time n
    update_sumC!(sumC, sumdiffC, inode, model, n)
    # Iterate over neighbors
    @inbounds @fastmath for (jindex, j) in enumerate(inode.neighs)
        iindex = nodes[j].neighs_idxs[i]
        int = compute_integral(sumdiffC, nodes[j].cavs[iindex].C, inode.cavs[jindex].C, model.J[i, j], n)
        nodes[j].cavs[iindex].C[n+1] = (1 - lambda * dt) * nodes[j].cavs[iindex].C[n] - dt / D * (nodes[j].cavs[iindex].C[0] * (sumC[n] - inode.cavs[jindex].C[n] * model.J[i, j]) + int)
    end
end

function cavity_update!(nodes::Vector{NodeEQ}, model::OUModel, dt::Float64, T::Int, sumC::OffsetVector{Float64, Vector{Float64}}, sumdiffC::OffsetVector{Float64, Vector{Float64}}, p)
    #Iterate over time steps
    @inbounds @fastmath for n in 0:T-1
        # Iterate over nodes
        @inbounds @fastmath for inode in nodes
            # Update the cavity fields
            update_Ceq_cav!(inode, nodes, model, dt, sumC, sumdiffC, n)
            # Upddate progress bar
            next!(p)
        end
    end
    # Close the progress bar
    finish!(p)
end

function compute_integral(sumdiffC::OffsetVector{Float64, Vector{Float64}}, Ci::OffsetVector{Float64, Vector{Float64}}, n)
    int = 0.0
    @inbounds @fastmath @simd for m in 0:n-1
        int += Ci[n-m] * sumdiffC[m]
    end
    return int
end

function update_Ceq_marg!(inode::NodeEQ, model::OUModel, dt::Float64, sumC::OffsetVector{Float64, Vector{Float64}}, sumdiffC::OffsetVector{Float64, Vector{Float64}}, n::Int)
    # Unpack parameters
    i = inode.i
    lambda = model.lambdas[i]
    D = model.D
    # Update sumC and sumdiffC at time n
    update_sumC!(sumC, sumdiffC, inode, model, n)
    # Compute the marginal field at time n+1
    int = compute_integral(sumdiffC, inode.marg.C, n)
    inode.marg.C[n+1] = (1 - lambda * dt) * inode.marg.C[n] - dt / D * (inode.marg.C[0] * sumC[n] + int)
end

function update_Req_marg!(inode::NodeEQ, model::OUModel, dt::Float64, n::Int)
    # Unpack parameters
    D = model.D
    # Update the marginal field at time n+1
    inode.marg.R[n+1] =  1 / (dt * D) * (inode.marg.C[n] - inode.marg.C[n+1])
end

function marginal_update!(nodes::Vector{NodeEQ}, model::OUModel, dt::Float64, T::Int, sumC::OffsetVector{Float64, Vector{Float64}}, sumdiffC::OffsetVector{Float64, Vector{Float64}})
    #Iterate over time steps
    @inbounds @fastmath for n in 0:T-1
        # Iterate over nodes
        @inbounds @fastmath for inode in nodes
            # Update the marginal fields
            update_Ceq_marg!(inode, model, dt, sumC, sumdiffC, n)
            update_Req_marg!(inode, model, dt, n)
        end
    end
end

"""
    run_cavity_EQ(model::OUModel, dt::Float64, T::Int)

Run the equilibrium cavity method for the equilibrium Ornstein-Uhlenbeck model.

# Arguments
- `model::OUModel`: The Ornstein-Uhlenbeck model.
- `dt::Float64`: The time step size.
- `T::Int`: The number of time steps.

# Returns
- `nodes::Vector{NodeEQ}`: The updated nodes after running the cavity method, containing the cavity autocorrelations and the marginal fields.
"""
function run_cavity_EQ(model::OUModel, dt::Float64, T::Int; show_progress::Bool = false)
    # Initialize nodes
    nodes = init_nodes_EQ(model, T)
    @inbounds @fastmath for inode in nodes
        @inbounds @fastmath for cav in inode.cavs
            cav.C[0] = 1.0
        end
        inode.marg.C[0] = 1.0
    end
    # Initialize sumC and sumdiffC
    sumC = OffsetVector(zeros(T+1), 0:T)
    sumdiffC = OffsetVector(zeros(T+1), 0:T)
    # Initialize progress bar
    p = Progress(T; enabled=show_progress, dt=0.3, showspeed=true)# Run the cavity method
    cavity_update!(nodes, model, dt, T, sumC, sumdiffC, p)
    # Update the marginal fields
    marginal_update!(nodes, model, dt, T, sumC, sumdiffC)
    return nodes
end