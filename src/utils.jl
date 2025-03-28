"""
    compute_meanvar(sol)

Compute the mean and standard deviation of the trajectories in the solution object.

# Arguments
- `sol::RODESolution`: The solution object.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories at time `sol.t[l]`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories at time `sol.t[l]`.
"""
function compute_meanstd(sol::RODESolution)
    timesteps, N = length(sol.t), length(sol.u)
    mean_traj = zeros(timesteps)
    std_traj = zeros(timesteps)
    @inbounds @fastmath for t in 1:timesteps
        sum_traj, sum2_traj = 0.0, 0.0
        @inbounds @fastmath for i in 1:N
            sum_traj += sol.u[i][t]
            sum2_traj += sol.u[i][t]^2
        end
        mean_traj[t] = sum_traj / N
        std_traj[t] = sqrt(sum2_traj / (N-1) - sum_traj ^ 2 / (N * (N-1)))
    end
    return mean_traj, std_traj
end

"""
    compute_stats(sol)

Compute the mean, standard deviation and average autocorrelation of the trajectories in the solution object.

# Arguments
- `sol::RODESolution`: The solution object.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories at time `sol.t[l]`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories at time `sol.t[l]`.
- `autocorr::Matrix{Float64}`: The average autocorrelation. The element at index `(l, k)` is the average autocorrelation of the trajectories at times `sol.t[l]` and `sol.t[l]`.
"""
function compute_stats(sol::RODESolution)
    timesteps, N = length(sol.t), length(sol.u)
    mean_traj = zeros(timesteps)
    std_traj = zeros(timesteps)
    autocorr = zeros(timesteps, timesteps)
    @inbounds @fastmath for t in 1:timesteps
        sum_traj, sum2_traj = 0.0, 0.0
        @inbounds @fastmath for i in 1:N
            sum_traj += sol.u[i][t]
            sum2_traj += sol.u[i][t]^2
        end
        mean_traj[t] = sum_traj / N
        std_traj[t] = sqrt(sum2_traj / (N-1) - sum_traj ^ 2 / (N * (N-1)))
        @inbounds @fastmath for t2 in t:timesteps
            sum_autocorr = 0.0
            @inbounds @fastmath for i in 1:N
                sum_autocorr += sol.u[i][t] * sol.u[i][t2]
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
- `sim::EnsembleSolution`: The ensemble solution object.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories at time `sim[1].t[l]`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories at time `sim[1].t[l]`.
"""
function compute_meanstd(sim::EnsembleSolution)
    nsim, timesteps = length(sim), length(sim[1].t)
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
- `sim::EnsembleSolution`: The ensemble solution object.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories at time `sim[1].t[l]`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories at time `sim[1].t[l]`.
- `autocorr::Matrix{Float64}`: The average autocorrelation. The element at index `(l, k)` is the average autocorrelation of the trajectories at times `sim[1].t[l]` and `sim[1].t[k]`.
"""
function compute_stats(sim::EnsembleSolution)
    nsim, timesteps = length(sim), length(sim[1].t)
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

"""
    compute_meanstd(sim)

Compute the mean and standard deviation of the trajectories in the ensemble solution object.

# Arguments
- `sim::Vector{RODESolution}`: The ensemble solution object, which is a vector of `RODESolution` objects.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories at time `sim[1].t[l]`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories at time `sim[1].t[l]`.
"""
function compute_meanstd(sim::Vector{RODESolution})
    nsim, timesteps = length(sim), length(sim[1].t)
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
- `sim::Vector{RODESolution}`: The ensemble solution object, which is a vector of `RODESolution` objects.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories at time `sim[1].t[l]`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories at time `sim[1].t[l]`.
- `autocorr::Matrix{Float64}`: The average autocorrelation. The element at index `(l, k)` is the average autocorrelation of the trajectories at times `sim[1].t[l]` and `sim[1].t[k]`.
"""
function compute_stats(sim::Vector{RODESolution})
    nsim, timesteps = length(sim), length(sim[1].t)
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