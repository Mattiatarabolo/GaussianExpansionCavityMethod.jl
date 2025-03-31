"""
    compute_meanvar(trajs)

Compute the mean and standard deviation of the trajectories.

# arguments
- `trajs::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories at discretized time `l``.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories at discretized time `l`.
"""
function compute_meanstd(trajs::Matrix{Float64})
    N, timesteps = size(trajs)
    # Initialize mean and standard deviation arrays
    mean_traj = zeros(timesteps)
    std_traj = zeros(timesteps)
    @inbounds @fastmath for t in 1:timesteps
        sum_traj, sum2_traj = 0.0, 0.0
        @inbounds @fastmath for i in 1:N
            sum_traj += trajs[i, t]
            sum2_traj += trajs[i, t] ^ 2
        end
        mean_traj[t] = sum_traj / N
        std_traj[t] = sqrt(sum2_traj / (N-1) - sum_traj ^ 2 / (N * (N-1)))
    end
    return mean_traj, std_traj
end

"""
    compute_stats(trajs)

Compute the mean, standard deviation and average autocorrelation of the trajectories.

# Arguments
- `trajs::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories at discretized time `l`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories at discretized time `l`.
- `autocorr::Matrix{Float64}`: The average autocorrelation. The element at index `(l, k)` is the average autocorrelation of the trajectories at discretized times `l` and `k`.
"""
function compute_stats(trajs::Matrix{Float64})
    N, timesteps = size(trajs)
    # Initialize mean and standard deviation arrays and autocorrelation matrix
    mean_traj = zeros(timesteps)
    std_traj = zeros(timesteps)
    autocorr = zeros(timesteps, timesteps)
    @inbounds @fastmath for t in 1:timesteps
        sum_traj, sum2_traj = 0.0, 0.0
        @inbounds @fastmath for i in 1:N
            sum_traj += trajs[i, t]
            sum2_traj += trajs[i, t] ^ 2
        end
        mean_traj[t] = sum_traj / N
        std_traj[t] = sqrt(sum2_traj / (N-1) - sum_traj ^ 2 / (N * (N-1)))
        @inbounds @fastmath for t2 in t:timesteps
            sum_autocorr = 0.0
            @inbounds @fastmath for i in 1:N
                sum_autocorr += trajs[i, t] * trajs[i, t2]
            end
            autocorr[t, t2] = sum_autocorr / N
        end
    end
    return mean_traj, std_traj, autocorr
end

"""
    compute_meanstd(sim)

Compute the mean and standard deviation of the trajectories in the ensemble solution object.

# Arguments
- `sim::Vector{Matrix{Float64}}`: The ensemble solution object. Each element is a matrix where each column corresponds to a time point.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories over nodes and ensemble realizations at discretized time `l`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories over nodes and ensemble realizations at discretized time `l`.
"""
function compute_meanstd(sim::Vector{Matrix{Float64}})
    nsim = length(sim)
    N, timesteps = size(sim[1])
    meanvar_sim = [compute_meanstd(sim[s]) for s in 1:nsim]

    # Compute overall mean
    mean_traj = zeros(timesteps)
    @inbounds @fastmath for t in 1:timesteps
        sum_traj = 0.0
        @inbounds @fastmath for s in 1:nsim
            sum_traj += meanvar_sim[s][1][t]
        end
        mean_traj[t] = sum_traj / nsim
    end

    # Compute overall variance using the law of total variance
    std_traj = zeros(timesteps)
    @inbounds @fastmath for t in 1:timesteps
        sum2_traj = 0.0
        @inbounds @fastmath for s in 1:nsim
            sum2_traj += meanvar_sim[s][2][t]^2 + (meanvar_sim[s][1][t] - mean_traj[t])^2
        end
        std_traj[t] = sqrt(sum2_traj / (nsim - 1))
    end
    return mean_traj, std_traj
end

"""
    compute_stats(sim)

Compute the mean, standard deviation and average autocorrelation of the trajectories in the ensemble solution object.

# Arguments
- `sim::Vector{Matrix{Float64}}`: The ensemble solution object. Each element is a matrix where each column corresponds to a time point.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories over nodes and ensemble realizations at discretized time `l`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories over nodes and ensemble realizations at discretized time `l`.
- `autocorr::Matrix{Float64}`: The average autocorrelation. The element at index `(l, k)` is the average autocorrelation of the trajectories over nodes and ensemble realizations at discretized times `l` and `k`.
"""
function compute_stats(sim::Vector{Matrix{Float64}})
    nsim = length(sim)
    N, timesteps = size(sim[1])
    stats_sim = [compute_stats(sim[s]) for s in 1:nsim]

    # Compute overall mean and autocorrelation
    mean_traj = zeros(timesteps)
    autocorr = zeros(timesteps, timesteps)
    @inbounds @fastmath for t in 1:timesteps
        sum_traj = 0.0
        @inbounds @fastmath for s in 1:nsim
            sum_traj += stats_sim[s][1][t]
        end
        mean_traj[t] = sum_traj / nsim
        @inbounds @fastmath for t2 in t:timesteps
            sum_autocorr = 0.0
            @inbounds @fastmath for s in 1:nsim
                sum_autocorr += stats_sim[s][3][t, t2]
            end
            autocorr[t, t2] = sum_autocorr / nsim
        end
    end

    # Compute overall variance using the law of total variance
    std_traj = zeros(timesteps)
    @inbounds @fastmath for t in 1:timesteps
        sum2_traj = 0.0
        @inbounds @fastmath for s in 1:nsim
            sum2_traj += stats_sim[s][2][t]^2 + (stats_sim[s][1][t] - mean_traj[t])^2
        end
        std_traj[t] = sqrt(sum2_traj / (nsim - 1))
    end
    return mean_traj, std_traj, autocorr
end

