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
    T = size(trajs, 2)
    mean_traj = reshape(mean(trajs; dims=1), T)
    std_traj = reshape(std(trajs; dims=1), T)
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
    T = size(trajs, 2)
    # Compute mean and standard deviation for each time
    mean_traj = mean(trajs; dims=1)
    std_traj = stdm(trajs, mean_traj; dims=1)
    # Reshape mean_traj and std_traj to be vectors
    mean_traj = reshape(mean_traj, T)
    std_traj = reshape(std_traj, T)
    # Compute the average autocorrelation
    autocorr = cov(trajs; dims=1) # Covariance matrix
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
    T = size(sim[1], 2)
    # Compute mean and variance for each simulation
    meanvar_sim = [compute_meanstd(sim[s]) for s in 1:nsim]
    # Compute overall mean
    mean_traj = zeros(T)
    @inbounds @fastmath for s in 1:nsim
        mean_traj .+= meanvar_sim[s][1]
    end
    mean_traj ./= nsim
    # Compute overall variance using the law of total variance
    std_traj = zeros(T)
    @inbounds @fastmath for s in 1:nsim
        std_traj .+= meanvar_sim[s][2] .^ 2 .+ (meanvar_sim[s][1] .- mean_traj) .^ 2
    end
    std_traj = sqrt.(std_traj ./ (nsim - 1))
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
    T = size(sim[1], 2)
    # Compute mean, variance  and autocorrelation for each simulation
    stats_sim = [compute_stats(sim[s]) for s in 1:nsim]
    # Compute overall mean and autocorrelation
    mean_traj = zeros(T)
    autocorr = zeros(T, T)
    @inbounds @fastmath for s in 1:nsim
        mean_traj .+= stats_sim[s][1]
        autocorr .+= stats_sim[s][3]
    end
    mean_traj ./= nsim
    autocorr ./= nsim
    # Compute overall variance using the law of total variance
    std_traj = zeros(T)
    @inbounds @fastmath for s in 1:nsim
        std_traj .+= stats_sim[s][2] .^ 2 .+ (stats_sim[s][1] .- mean_traj) .^ 2
    end
    std_traj = sqrt.(std_traj ./ (nsim - 1))
    return mean_traj, std_traj, autocorr
end

