##############################################################################################################################
################################ Equilibrium  (EQ) case (t, t' -> ∞ with t - t' = tau) #######################################
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



##############################################################################################################################
################################################ Transient case ##############################################################
##############################################################################################################################

function compute_C_R_integrals(inode::Node, jnode::Node, model::OUModel, n::Int, l::Int, p)
    # Unpack parameters
    i, j = inode.i, jnode.i
    iidx, jidx = jnode.neighs_idxs[i], inode.neighs_idxs[j]
    Jij, Jji = model.J[i, j], model.J[j, i]
    # Iterate over integral indices, but split in ordere to avoid double counting. Since l = 0:n, we can split the loops from 0 to l-1 (int3 completed), then from l+1 to n-1 (int1 completed, int2 completed)
    # m = 0:l-1 (add terms only to int2 and int3)
    int1, int2, int3 = 0.0, 0.0, 0.0
    # m = 0:l (add terms only to int2 and int3)
    @inounds @fastmath @simd for m in 0:l
        int2 += jnode.cavs[iidx].C[l,m] * (inode.sumR[n,m] - Jij * Jji * inode.cavs[jidx].R[n,m])
        int3 += jnode.cavs[iidx].R[l,m] * (inode.sumC[n,m] - Jij ^ 2 * inode.cavs[jidx].C[n,m])
        next!(p)
    end
    # m = l+1:n-1 (add terms only to int1 and int2)
    @inounds @fastmath @simd for m in l+1:n-1
        int1 += jnode.cavs[iidx].R[m,l] * (inode.sumR[n,m] - Jij * Jji * inode.cavs[jidx].R[n,m])
        int2 += jnode.cavs[iidx].C[m,l] * (inode.sumR[n,m] - Jij * Jji * inode.cavs[jidx].R[n,m])
        next!(p)
    end
    return int1, int2, int3
end

function update_C_R_cav!(inode::Node, nodes::Vector{Node}, model::OUModel, dt::Float64, n::Int, l::Int, p)
    # Unpack parameters
    i = inode.i
    lambda = model.lambdas[i]
    # Iterate over neighbors
    @inbounds @fastmath for j in inode.neighs
        iindex = nodes[j].neighs_idxs[i]
        # Compute integrals
        int1, int2, int3 = compute_C_R_integrals(inode, nodes[j], model, n, l, p)
        # Update the cavity response at times n+1, l 
        nodes[j].cavs[iindex].R[n+1,l] = (1 - lambda * dt) * nodes[j].cavs[iindex].R[n,l] + dt * (n == l) + dt ^ 2 * int1
        # Update the cavity correlation at times n+1, l 
        nodes[j].cavs[iindex].C[n+1,l] = (1 - lambda * dt) * nodes[j].cavs[iindex].C[n,l] + dt ^ 2 * int2 + dt ^ 2 * int3
    end
end

function update_C_R_sums!(inode::Node, model::OUModel, n, l, p)
    # Unpack parameters
    i = inode.i
    # Iterate over neighbors
    @inbounds @fastmath for (jidx, cav) in enumerate(inode.cavs)
        inode.sumR[n,l] += model.J[i, inode.neighs[jidx]] * model.J[inode.neighs[jidx], i] * cav.R[n,l]
        inode.sumC[n,l] += model.J[i, inode.neighs[jidx]] ^ 2 * cav.C[n,l]
        next!(p)
    end
end

function compute_C_mu_integrals(inode::Node, jnode::Node, model::OUModel, n::Int, p)
    # Unpack parameters
    i, j = inode.i, jnode.i
    iidx, jidx = jnode.neighs_idxs[i], inode.neighs_idxs[j]
    Jij, Jji = model.J[i, j], model.J[j, i]
    # Iterate over integral indices, but split in ordere to avoid double counting. We can split the loops from 0 to n-1 (int1 and int2 completed), then add the term corresponding to m = n (only to int3, int3 completed)
    # m = 0:n-1
    @inounds @fastmath @simd for m in 0:n-1
        int1 += jnode.cavs[iidx].mu[m] * (inode.sumR[n,m] - Jij * Jji * inode.cavs[jidx].R[n,m])    
        int2 += jnode.cavs[iidx].C[n+1,m] * (inode.sumR[n,m] - Jij * Jji * inode.cavs[jidx].R[n,m])
        int3 += inode.cavs[jidx].R[n+1,m] * (inode.sumC[n,m] - Jij ^ 2 * inode.cavs[jidx].C[n,m])
        next!(p)
    end
    # m = n (add terms only to int3)
    int3 += inode.cavs[jidx].R[n+1,n] * (inode.sumC[n,n] - Jij ^ 2 * inode.cavs[jidx].C[n,n])
    return int1, int2, int3
end

function update_C_mu_cav!(inode::Node, nodes::Vector{Node}, model::OUModel, dt::Float64, n::Int, p)
    # Unpack parameters
    i = inode.i
    lambda = model.lambdas[i]
    D = model.D
    # Iterate over neighbors
    @inbounds @fastmath for (jindex, j) in enumerate(inode.neighs)
        iindex = nodes[j].neighs_idxs[i]
        # Compute integrals
        int1, int2, int3 = compute_C_mu_integrals(inode, nodes[j], model, n, p)
        # Update the cavity correlation at times n+1, n+1 
        nodes[j].cavs[iindex].C[n+1,n+1] = (1 - lambda * dt) * nodes[j].cavs[iindex].C[n+1,n] + 2 * dt * D * nodes[j].cavs[iindex].R[n+1,n] + dt ^ 2 * int2 + dt ^ 2 * int3
        # Update the cavity marginal at time n+1
        nodes[j].cavs[iindex].mu[n+1] = (1 - lambda * dt) * nodes[j].cavs[iindex].mu[n] + dt * (inode.summu[n] - model.J[i, j] * inode.cavs[jindex].mu[n]) + dt ^ 2 * int1
    end
end

function update_C_mu_sums!(inode::Node, model::OUModel, n::Int, p)
    # Unpack parameters
    i = inode.i
    # Iterate over neighbors
    @inbounds @fastmath for (jidx, cav) in enumerate(inode.cavs)
        inode.summu[n+1] += model.J[i, inode.neighs[jidx]] * cav.mu[n+1]
        inode.sumC[n+1,n+1] += model.J[i, inode.neighs[jidx]] ^ 2 * cav.C[n+1,n+1]
        next!(p)
    end
end

function cavity_update!(nodes::Vector{Node}, model::OUModel, dt::Float64, T::Int, p)
    # Iterate over time steps
    @inbounds @fastmath for n in 0:T-1
        @inbounds @fastmath for l in 0:n
            # Iterate over nodes
            @inbounds @fastmath for inode in nodes  
                # Update the cavity correlations and responses
                update_C_R_cav!(inode, nodes, model, dt, n, l, p)
                # Update the sums of correlations and responses
                update_C_R_sums!(inode, model, n+1, l, p)
            end
        end
        # Update the cavity correlations at times n+1, n+1 and the marginal at time n+1
        @inbounds @fastmath for inode in nodes
            update_C_mu_cav!(inode, nodes, model, dt, n, p)
            # Updated the sums of correlations at times n+1, n+1 and of marhinals at time n+1
            update_C_mu_sums!(inode, model, n+1, p)
        end
    end
    finish!(p)
end

function compute_C_R_integrals(inode::Node, n::Int, l::Int, p)
    # Iterate over integral indices, but split in ordere to avoid double counting. Since l = 0:n, we can split the loops from 0 to l-1 (int3 completed), then add the term corresponding to m = l (only to int2), then from l+1 to n-1 (int1 completed, int2 completed)
    # m = 0:l-1 (add terms only to int2 and int3)
    int1, int2, int3 = 0.0, 0.0, 0.0
    # m = 0:l (add terms only to int2 and int3)
    @inounds @fastmath @simd for m in 0:l
        int2 += inode.marg.C[l,m] * inode.sumR[n,m]
        int3 += inode.marg.R[l,m] * inode.sumC[n,m]
        next!(p)
    end
    # m = l+1:n-1 (add terms only to int1 and int2)
    @inounds @fastmath @simd for m in l+1:n-1
        int1 += inode.marg.R[m,l] * inode.sumR[n,m]
        int2 += inode.marg.C[m,l] * inode.sumR[n,m]
        next!(p)
    end
    return int1, int2, int3
end

function update_C_R_marg!(inode::Node, model::OUModel, dt::Float64, n::Int, l::Int, p)
    # Unpack parameters
    lambda = model.lambdas[i]
    # Compute integrals
    int1, int2, int3 = compute_C_R_integrals(inode, n, l, p)
    # Update the marginal response at times n+1, l 
    inode.marg.R[n+1,l] = (1 - lambda * dt) * inode.marg.R[n,l] + dt * (n == l) + dt ^ 2 * int1
    # Update the marginal correlation at times n+1, l 
    inode.marg.C[n+1,l] = (1 - lambda * dt) * inode.marg.C[n,l] + dt ^ 2 * int2 + dt ^ 2 * int3
end

function compute_C_mu_integrals(inode::Node, n::Int, p)
    # Iterate over integral indices, but split in ordere to avoid double counting. We can split the loops from 0 to n-1 (int1 and int2 completed), then add the term corresponding to m = n (only to int3, int3 completed)
    # m = 0:n-1
    @inounds @fastmath @simd for m in 0:n-1
        int1 += inode.marg.mu[m] * inode.sumR[n,m]
        int2 += inode.marg.C[n+1,m] * inode.sumR[n,m]
        int3 += inode.marg.R[n+1,m] * inode.sumC[n,m]
        next!(p)
    end
    # m = n (add terms only to int3)
    int3 += inode.marg..R[n+1,n] * inode.sumC[n,n]
    return int1, int2, int3
end

function update_C_mu_marg!(inode, model, dt, n, p)
    # Unpack parameters
    i = inode.i
    lambda = model.lambdas[i]
    D = model.D
    # Compute the integral
    int1, int2, int3 = compute_C_mu_integrals(inode, n, p)
    # Update the marginal field at time n+1
    inode.marg.C[n+1,n+1] = (1 - lambda * dt) * inode.marg.C[n+1,n] + 2 * dt * D * inode.marg.R[n+1,n] + dt ^ 2 * int2 + dt ^ 2 * int3
    # Update the marginal field at time n+1
    inode.marg.mu[n+1] = (1 - lambda * dt) * inode.marg.mu[n] + dt * inode.summu[n] + dt ^ 2 * int1
end

function marginal_update!(nodes::Vector{Node}, model::OUModel, dt::Float64, T::Int, p)
    # Iterate over time steps
    @inbounds @fastmath for n in 0:T-1
        @inbounds @fastmath for l in 0:n
            # Iterate over nodes
            @inbounds @fastmath for inode in nodes  
                # Update the marginal correlations and responses
                update_C_R_marg!(inode, model, dt, n, l, p)
            end
        end
        # Update the cavity correlations at times n+1, n+1 and the marginal at time n+1
        @inbounds @fastmath for inode in nodes
            update_C_mu_marg!(inode, model, dt, n, p)
        end
    end
    finish!(p)
end
function run_cavity(model::OUModel, dt::Float64, T::Int; C0=1.0, mu0=0.0, showprogress=false)
    # Initialize nodes
    nodes = init_nodes(model, T)
    @inbounds @fastmath for inode in nodes
        @inbounds @fastmath for (jidx, cav) in enumerate(inode.cavs)
            cav.C[0,0] = C0
            cav.mu[0] = mu0
            inode.summu[0] += model.J[inode.i, inode.neighs[jidx]] * mu0
            inode.sumC[0,0] += model.J[inode.i, inode.neighs[jidx]] ^ 2 * C0
        end
        inode.marg.C[0,0] = C0
        inode.marg.mu[0] = mu0
    end
    # Initialize progress bar for the cavity updates
    nedges = sum(model.J .!= 0)
    pc_tot_iterations = nedges * Int(T * (T - 1) * (2 * T -1) / 6 + 3 * T * (T - 1) / 2 + 3 * T)
    pc = Progress(pc_tot_iterations; enabled=showprogress, dt=0.3, showspeed=true, desc="Cavity update: ")
    # Run the cavity method
    cavity_update!(nodes, model, dt, T, pc)
    # Initialize progress bar for the marginal updates
    pm_tot_iterations = model.N * Int(T * (T - 1) * (2 * T - 1) / 6 + T * (T - 1) + T)
    pm = Progress(pm_tot_iterations; enabled=showprogress, dt=0.3, showspeed=true, desc="Marginal update: ")
    # Update the marginal fields
    marginal_update!(nodes, model, dt, T, pm)
    return nodes
end


