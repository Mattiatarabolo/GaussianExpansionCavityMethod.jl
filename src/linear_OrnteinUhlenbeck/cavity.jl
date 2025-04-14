##############################################################################################################################
########################### Time Translational Invariant (EQ) case (t, t' -> ∞ with t - t' = tau) ###########################
##############################################################################################################################

function update_sumC0!(inode::NodeEQ, model::OUModel, p)
    i = inode.i
    @inbounds @fastmath for (kindex, k) in enumerate(inode.neighs)
        # Compute the sum of Cavity fields
        inode.sumC[0] += inode.cavs[kindex].C[0] * model.J[i, k] ^ 2
        # Update progress bar
        next!(p)
    end
end

function update_sumC!(inode::NodeEQ, model::OUModel, n::Int, p)
    @assert n > 0 "n must be greater than 0"
    i = inode.i
    @inbounds @fastmath for (kindex, k) in enumerate(inode.neighs)
        # Compute the sum of Cavity fields
        inode.sumC[n] += inode.cavs[kindex].C[n] * model.J[i, k] ^ 2
        inode.sumdiffC[n-1] += (inode.cavs[kindex].C[n] - inode.cavs[kindex].C[n-1]) * model.J[i, k] ^ 2
        # Update progress bar
        next!(p)
    end
end

function compute_integral(sumdiffC::OffsetVector{Float64, Vector{Float64}}, Cij::OffsetVector{Float64, Vector{Float64}}, Cji::OffsetVector{Float64, Vector{Float64}}, Jij::Float64, n::Int, p)
    int = 0.0
    @inbounds @fastmath @simd for m in 0:n-1
        int += Cij[n-m] * (sumdiffC[m] - Jij ^ 2 * (Cji[m+1] - Cji[m]))
        # Update progress bar
        next!(p)
    end
    return int
end

function update_Ceq_cav0!(inode::NodeEQ, nodes::Vector{NodeEQ}, model::OUModel, dt::Float64, p)
    # Unpack parameters
    i = inode.i
    lambda = model.lambdas[i]
    D = model.D
    # Update the sumC at time n=0
    update_sumC0!(inode, model, p)
    # Iterate over neighbors
    @inbounds @fastmath for (jindex, j) in enumerate(inode.neighs)
        iindex = nodes[j].neighs_idxs[i]
        # Update the cavity field at time n=1 
        nodes[j].cavs[iindex].C[1] = (1 - lambda * dt) * nodes[j].cavs[iindex].C[0] + dt / D * nodes[j].cavs[iindex].C[0] * (inode.sumC[0] - inode.cavs[jindex].C[0] * model.J[i, j] ^ 2)
        # Update progress bar
        next!(p)
    end
end

function update_Ceq_cav!(inode::NodeEQ, nodes::Vector{NodeEQ}, model::OUModel, dt::Float64, n::Int, p)
    @assert n > 0 "n must be greater than 0"
    # Unpack parameters
    i = inode.i
    lambda = model.lambdas[i]
    D = model.D
    # Update sumC and sumdiffC at time n
    update_sumC!(inode, model, n, p)
    # Iterate over neighbors
    @inbounds @fastmath for (jindex, j) in enumerate(inode.neighs)
        iindex = nodes[j].neighs_idxs[i]
        # Compute integral
        int = compute_integral(inode.sumdiffC, nodes[j].cavs[iindex].C, inode.cavs[jindex].C, model.J[i, j], n, p)
        # Update the cavity field at time n+1 
        nodes[j].cavs[iindex].C[n+1] = (1 - lambda * dt) * nodes[j].cavs[iindex].C[n] + dt / D * (nodes[j].cavs[iindex].C[0] * (inode.sumC[n] - inode.cavs[jindex].C[n] * model.J[i, j] ^ 2) - int)
    end
end

function cavity_update!(nodes::Vector{NodeEQ}, model::OUModel, dt::Float64, T::Int, p)
    # Update cavity correlations at n=1
    @inbounds @fastmath for inode in nodes
        update_Ceq_cav0!(inode, nodes, model, dt, p)
    end
    #Iterate over time steps
    @inbounds @fastmath for n in 1:T-1
        # Iterate over nodes
        @inbounds @fastmath for inode in nodes
            # Update the cavity fields
            update_Ceq_cav!(inode, nodes, model, dt, n, p)
        end
    end
    finish!(p)
end

function compute_integral(sumdiffC::OffsetVector{Float64, Vector{Float64}}, Ci::OffsetVector{Float64, Vector{Float64}}, n::Int, p)
    int = 0.0
    @inbounds @fastmath @simd for m in 0:n-1
        int += Ci[n-m] * sumdiffC[m]
        # Update progress bar
        next!(p)
    end
    return int
end

function update_Ceq_marg0!(inode::NodeEQ, model::OUModel, dt::Float64)
    # Unpack parameters
    i = inode.i
    lambda = model.lambdas[i]
    D = model.D
    # Update the marginal field at time n=1
    inode.marg.C[1] = (1 - lambda * dt) * inode.marg.C[0] + dt / D * inode.marg.C[0] * inode.sumC[0]
end

function update_Ceq_marg!(inode::NodeEQ, model::OUModel, dt::Float64, n::Int, p)
    # Unpack parameters
    i = inode.i
    lambda = model.lambdas[i]
    D = model.D
    # Compute the integral
    int = compute_integral(inode.sumdiffC, inode.marg.C, n, p)
    # Update the marginal field at time n+1
    inode.marg.C[n+1] = (1 - lambda * dt) * inode.marg.C[n] + dt / D * (inode.marg.C[0] * inode.sumC[n] - int)
end

function update_Req_marg!(inode::NodeEQ, model::OUModel, dt::Float64, n::Int)
    # Unpack parameters
    D = model.D
    # Update the marginal field at time n+1
    inode.marg.R[n+1] =  1 / (dt * D) * (inode.marg.C[n] - inode.marg.C[n+1])
end

function marginal_update!(nodes::Vector{NodeEQ}, model::OUModel, dt::Float64, T::Int, p)
    # Update marginal correlations at n=1
    @inbounds @fastmath for inode in nodes
        update_Ceq_marg0!(inode, model, dt)
        next!(p)
    end
    #Iterate over time steps
    @inbounds @fastmath for n in 1:T-1
        # Iterate over nodes
        @inbounds @fastmath for inode in nodes
            # Update the marginal fields
            update_Ceq_marg!(inode, model, dt, n, p)
            update_Req_marg!(inode, model, dt, n)
        end
    end
    finish!(p)
end

"""
    run_cavity_EQ(model::OUModel, dt::Float64, T::Int)

Run the equilibrium cavity method for the equilibrium Ornstein-Uhlenbeck model.

# Arguments
- `model::OUModel`: The Ornstein-Uhlenbeck model.
- `dt::Float64`: The time step size.
- `T::Int`: The number of time steps.

# Keyword Arguments
- `C0::Float64`: The initial value of the cavity autocorrelation (default is 1.0).
- `showprogress::Bool`: Whether to show progress bars (default is false).

# Returns
- `nodes::Vector{NodeEQ}`: The updated nodes after running the cavity method, containing the cavity autocorrelations and the marginal fields.
"""
function run_cavity_EQ(model::OUModel, dt::Float64, T::Int; C0=1.0, showprogress=false)
    # Initialize nodes
    nodes = init_nodes_EQ(model, T)
    @inbounds @fastmath for inode in nodes
        @inbounds @fastmath for cav in inode.cavs
            cav.C[0] = C0
        end
        inode.marg.C[0] = C0
    end
    # Initialize progress bar for the cavity updates
    nedges = sum(model.J .!= 0)
    pc_tot_iterations = nedges * T + nedges * Int((T - 1) * T / 2) # Total number of iterations for the cavity updates: for each time step (t=0,...,T-1) the sums of C are updated for each edge (contribution of nedges * T timesteps) and then the integrals from 0 to t-1 are computed for each edge (contribution of nedges * (T - 1) * T / 2, where (T - 1) * T / 2 is the total number of iterations for the integrals of one edge: ∑_{t=0}^{T-1} t = (T - 1) * T / 2)
    pc = Progress(pc_tot_iterations; enabled=showprogress, dt=0.3, showspeed=true, desc="Cavity update: ")
    # Run the cavity method
    cavity_update!(nodes, model, dt, T, pc)
    # Initialize progress bar for the marginal updates
    pm_tot_iterations = model.N + model.N * Int((T - 1) * T / 2) # Total number of iterations for the marginal updates: for each node (contribution of model.N) the initial step at t=0 is computed and then the integrals from 0 to t-1 are computed for each node and each time from t=1 to t=T-1 (contribution of model.N * (T - 1) * T / 2, where (T - 1) * T / 2 is the total number of iterations for the integrals of one node: ∑_{t=0}^{T-1} t = (T - 1) * T / 2)
    pm = Progress(pm_tot_iterations; enabled=showprogress, dt=0.3, showspeed=true, desc="Marginal update: ")
    # Update the marginal fields
    marginal_update!(nodes, model, dt, T, pm)
    return nodes
end