lagmax(T) = min(T-1, round(Int,T * 0.99))

function demean_ts!(z, X, i, mean_traj)
    z .= view(X, i, :) .- mean_traj
end
_autodot(x, lx, l) = dot(x, 1:lx-l, x, 1+l:lx)
function autocorr_TTI(X, lags)
    Tx = eltype(X)
    N, T = size(X) # number of time series and length of each time series
    m = length(lags) # number of lags
    r = zeros(Tx, N, m) # autocorrelation matrix
    z = zeros(Tx, T) # temporary vector to store the demeaned time series
    mean_traj= reshape(mean(X; dims=1), T) # mean over time series at each time
    for i = 1 : N # loop over each time series, i.e. over rows
        demean_ts!(z, X, i, mean_traj) # demean the time series
        r[i, 1] = _autodot(z, T, lags[1])
        for j = 2 : m # loop over each lag
            r[i, j] = _autodot(z, T, lags[j]) / r[i, 1]
        end
    end
    return r
end

function _autocorr(X; dims=1, p=nothing)
    Tx = eltype(X)
    if dims == 1
        N, T = size(X) # number of time series and length of each time series
        C = zeros(Tx, T, T) # autocorrelation matrix
        @inbounds @fastmath for t1 in 1:T # loop over each time series, i.e. over rows
            @inbounds @fastmath for t2 in 1:t1 # loop over each lag
                @inbounds @fastmath @simd for n in 1:N
                    C[t1, t2] += X[n, t1] * X[n, t2]
                    # update progress bar
                    if p !== nothing
                        next!(p)
                    end
                end
                C[t1, t2] /= N
            end
        end
        return C
    elseif dims == 2
        T, N = size(X) # number of time series and length of each time series
        C = zeros(Tx, T, T) # autocorrelation matrix
        @inbounds @fastmath for t1 in 1:T # loop over each time series, i.e. over rows
            @inbounds @fastmath for t2 in 1:t1 # loop over each lag
                @inbounds @fastmath @simd for n in 1:N
                    C[t1, t2] += X[t1, n] * X[t2, n]
                    # update progress bar
                    if p !== nothing
                        next!(p)
                    end
                end
                C[t1, t2] /= N
            end
        end
        return C
    else
        throw(ArgumentError("Invalid dimension: $dims"))
    end
end

function _autocorr_tws(X, tws_idxs, ts_idxs; dims=1, p=nothing)
    Tx = eltype(X)
    if dims == 1
        N, T = size(X) # number of time series and length of each time series
        Cs = [zeros(Tx, length(t_idx)) for t_idx in ts_idxs] # autocorrelations
        @inbounds @fastmath for (i, tw_idx) in enumerate(tws_idxs)
            @inbounds @fastmath for (j, t_idx) in enumerate(ts_idxs[i]) # loop over each time series, i.e. over rows
                @inbounds @fastmath @simd for n in 1:N
                    Cs[i][j] += X[n, tw_idx] * X[n, t_idx]
                    # update progress bar
                    if p !== nothing
                        next!(p)
                    end
                end
                Cs[i][j] /= N
            end
        end
        return Cs
    elseif dims == 2
        T, N = size(X) # number of time series and length of each time series
        Cs = [zeros(Tx, length(t_idx)) for t_idx in ts_idxs] # autocorrelations
        @inbounds @fastmath for (i, tw_idx) in enumerate(tws_idxs)
            @inbounds @fastmath for (j, t_idx) in enumerate(ts_idxs[i]) # loop over each time series, i.e. over rows
                @inbounds @fastmath @simd for n in 1:N
                    Cs[i][j] += X[tw_idx, n] * X[t_idx, n]
                    # update progress bar
                    if p !== nothing
                        next!(p)
                    end
                end
                Cs[i][j] /= N
            end
        end
        return Cs
    else
        throw(ArgumentError("Invalid dimension: $dims"))
    end
end


"""
    compute_meanstd(trajs; time_indices=nothing)

Compute the mean and standard deviation of the trajectories.

# arguments
- `trajs::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.

# Keyword Arguments
- `time_indices::Union{Nothing, AbstractVector{Int}}`: The time indices to compute the mean and standard deviation. If `nothing`, all time indices are used.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories at discretized time `l``.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories at discretized time `l`.
- `t_idx::Vector{Int}`: The time indices used to compute the mean and standard deviation.
"""
function compute_meanstd(trajs::Matrix{Float64}; time_indices::Union{Nothing, AbstractVector{Int}}=nothing)
    T = size(trajs, 2)
    # Filter time indices
    if time_indices === nothing
        t_idx = 1:T
    else
        t_idx = filter(t -> 1 ≤ t ≤ T, time_indices)
    end
    T_eff = length(t_idx)
    # Compute mean and standard deviation for each time
    mean_traj = mean(view(trajs, :, t_idx); dims=1)
    std_traj = std(view(trajs, :, t_idx); dims=1, mean=mean_traj, corrected=true)
    # Reshape mean_traj and std_traj to be vectors
    mean_traj = reshape(mean_traj, T_eff)
    std_traj = reshape(std_traj, T_eff)
    return mean_traj, std_traj, t_idx
end


"""
    compute_autocorr(trajs; time_indices=nothing, showprogress=false)

Compute the average autocorrelation of the trajectories.

# Arguments
- `trajs::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.

# Keyword Arguments
- `time_indices::Union{Nothing, AbstractVector{Int}}`: The time indices to compute the autocorrelation. If `nothing`, all time indices are used.
- `showprogress::Bool`: Whether to show a progress bar. Default is `false`.

# Returns
- `autocorr::Matrix{Float64}`: The average autocorrelation. The element at index `(l, k)` is the average autocorrelation of the trajectories at discretized times `l` and `k`.
"""
function compute_autocorr(trajs::Matrix{Float64}; time_indices::Union{Nothing, AbstractVector{Int}}=nothing, showprogress=false)
    N, T = size(trajs)
    # Filter time indices
    if time_indices === nothing
        t_idx = 1:T
    else
        t_idx = filter(t -> 1 ≤ t ≤ T, time_indices)
    end   
    T_eff = length(t_idx)
    # Create a progress bar
    p = Progress(Int(N * T * (T+1) / 2); enabled=showprogress, dt=0.3, showspeed=true, desc="Computing autocorrelation: ")
    # Compute the autocorrelation at different times
    autocorr = _autocorr(view(trajs, :, t_idx); dims=1, p=p) # Covariance matrix
    # Reshape autocorr to be a matrix
    autocorr = reshape(autocorr, T_eff, T_eff)
    return autocorr, t_idx
end

"""
    compute_autocorr(trajs::Matrix{Float64}, tws_idxs::Vector{Int}; time_indices=nothing, showprogress=false)

Compute the average autocorrelation of the trajectories. It computes the autocorrelation for each waiting time specified by the vector `tws_idxs`. For each waiting time `tw`, it computes the autocorrelation `C(t + tw, tw)` for all `t` in the `time_indices`. If `time_indices` is `nothing`, it computes the autocorrelation for all time indices after `tw`.

# Arguments
- `trajs::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.
- `tws_idxs::Vector{Int}`: The waiting times to compute the autocorrelation.

# Keyword Arguments
- `time_indices::Union{Nothing, AbstractVector{Int}}`: The time indices to compute the autocorrelation. If `nothing`, all time indices are used.
- `showprogress::Bool`: Whether to show a progress bar. Default is `false`.

# Returns
- `autocorrs::Vector{Matrix{Float64}}`: The average autocorrelation for each waiting time. The element at index `(i, l, k)` is the average autocorrelation of the trajectories at discretized times `l` and `k` for the waiting time `tws_idxs[i]`.
- `time_indices::Vector{Vector{Int}}`: The time indices used to compute the autocorrelation for each waiting time.
"""
function compute_autocorr(trajs::Matrix{Float64}, tws_idxs::Vector{Int}; time_indices=nothing, showprogress=false)
    N, T = size(trajs)
    if time_indices === nothing
        time_indices = [tw_idx:T for tw_idx in tws_idxs]
    else
        time_indices = [filter(t -> tw_idx ≤ t ≤ T, time_indices[i]) for (i, tw_idx) in enumerate(tws_idxs)]  
    end
    # Create a progress bar
    p = Progress(N * sum(length(t_idxs) for t_idxs in time_indices); enabled=showprogress, dt=1.0, showspeed=true, desc="Computing autocorrelation: ")
    # Compute the autocorrelation at different times
    autocorrs = _autocorr_tws(trajs, tws_idxs, time_indices; dims=1, p=p) # Covariances
    return autocorrs, time_indices
end

"""
    compute_stats(trajs; time_indices=nothing, showprogress=false)

Compute the mean, standard deviation and average autocorrelation of the trajectories.

# Arguments
- `trajs::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.

# Keyword Arguments
- `time_indices::Union{Nothing, AbstractVector{Int}}`: The time indices to compute the mean and standard deviation. If `nothing`, all time indices are used.
- `showprogress::Bool`: Whether to show a progress bar. Default is `false`.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories at discretized time `l`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories at discretized time `l`.
- `autocorr::Matrix{Float64}`: The average autocorrelation. The element at index `(l, k)` is the average autocorrelation of the trajectories at discretized times `l` and `k`.
- `t_idx::Vector{Int}`: The time indices used to compute the mean and standard deviation.
"""
function compute_stats(trajs::Matrix{Float64}; time_indices=nothing, showprogress=false)
    # Compute mean, standard deviation and autocorrelation
    mean_traj, std_traj, t_idxm = compute_meanstd(trajs; time_indices=time_indices)
    autocorr, t_idxC = compute_autocorr(trajs; time_indices=time_indices, showprogress=showprogress)
    @assert t_idxC == t_idxm "Time indices for mean/std and autocorrelation do not match"
    return mean_traj, std_traj, autocorr, t_idxC
end

"""
    compute_meanstd(sim; time_indices=nothing)

Compute the mean and standard deviation of the trajectories in the ensemble solution object.

# Arguments
- `sim::Vector{Matrix{Float64}}`: The ensemble solution object. Each element is a matrix where each column corresponds to a time point.

# Keyword Arguments
- `time_indices::Union{Nothing, AbstractVector{Int}}`: The time indices to compute the mean and standard deviation. If `nothing`, all time indices are used.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories over nodes and ensemble realizations at discretized time `l`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories over nodes and ensemble realizations at discretized time `l`.
- `t_idx::Vector{Int}`: The time indices used to compute the mean and standard deviation.
"""
function compute_meanstd(sim::Vector{Matrix{Float64}}; time_indices::Union{Nothing, AbstractVector{Int}}=nothing)
    nsim = length(sim)
    N, T = size(sim[1])
    # Filter time indices
    if time_indices === nothing
        t_idx = 1:T
    else
        t_idx = filter(t -> 1 ≤ t ≤ T, time_indices)
    end
    T_eff = length(t_idx)
    # Flatten data into an (N*nsim) × T_eff matrix
    sim_flattened = zeros(N * nsim, T_eff)
    @inbounds @simd for s in 1:nsim
        sim_flattened[(s-1)*N+1:s*N, :] .= view(sim[s], :, t_idx)
    end
    # Compute mean and standard deviation for each time
    mean_traj = mean(sim_flattened; dims=1)
    std_traj = std(sim_flattened; dims=1, mean=mean_traj, corrected=true)
    # Reshape mean_traj and std_traj to be vectors
    mean_traj = reshape(mean_traj, T_eff)
    std_traj = reshape(std_traj, T_eff)
    return mean_traj, std_traj, t_idx
end

"""
    compute_autocorr(sim; time_indices=nothing, showprogress=false)

Compute the average autocorrelation of the trajectories in the ensemble solution object.

# Arguments
- `sim::Vector{Matrix{Float64}}`: The ensemble solution object. Each element is a matrix where each column corresponds to a time point.

# Keyword Arguments
- `time_indices::Union{Nothing, AbstractVector{Int}}`: The time indices to compute the autocorrelation. If `nothing`, all time indices are used.
- `showprogress::Bool`: Whether to show a progress bar. Default is `false`.

# Returns
- `autocorr::Matrix{Float64}`: The average autocorrelation. The element at index `(l, k)` is the average autocorrelation of the trajectories over nodes and ensemble realizations at discretized times `l` and `k`.
- `t_idx::Vector{Int}`: The time indices used to compute the autocorrelation.
"""
function compute_autocorr(sim::Vector{Matrix{Float64}}; time_indices=nothing, showprogress=false)
    nsim = length(sim)
    N, T = size(sim[1])
    # Filter time indices
    if time_indices === nothing
        t_idx = 1:T
    else
        t_idx = filter(t -> 1 ≤ t ≤ T, time_indices)
    end
    T_eff = length(t_idx)
    # Create a progress bar
    p = Progress(Int(N * nsim * T_eff * (T_eff+1) / 2); enabled=showprogress, dt=0.3, showspeed=true, desc="Computing autocorrelation: ")
    # Flatten data into an (N*nsim) × T_eff matrix
    sim_flattened = zeros(N * nsim, T_eff)
    @inbounds @simd for s in 1:nsim
        sim_flattened[(s-1)*N+1:s*N, :] .= view(sim[s], :, t_idx)
    end
    # Compute the autocorrelation at different times
    autocorr = _autocorr(sim_flattened; dims=1, p=p) # Covariance matrix
    # Reshape autocorr to be a matrix
    autocorr = reshape(autocorr, T_eff, T_eff)
    return autocorr, t_idx
end

"""
    compute_autocorr(sim::Vector{Matrix{Float64}}, tws_idxs::Vector{Int}; time_indices=nothing, showprogress=false)

Compute the average autocorrelation of the trajectories in the ensemble solution object.
It computes the autocorrelation for each waiting time specified by the vector `tws_idxs`. For each waiting time `tw`, it computes the autocorrelation `C(t + tw, tw)` for all `t` in the `time_indices`. If `time_indices` is `nothing`, it computes the autocorrelation for all time indices after `tw`.

# Arguments
- `sim::Vector{Matrix{Float64}}`: The ensemble solution object. Each element is a matrix where each column corresponds to a time point.
- `tws_idxs::Vector{Int}`: The waiting times to compute the autocorrelation.

# Keyword Arguments
- `time_indices::Union{Nothing, AbstractVector{Int}}`: The time indices to compute the autocorrelation. If `nothing`, all time indices are used.
- `showprogress::Bool`: Whether to show a progress bar. Default is `false`.

# Returns
- `autocorrs::Vector{Matrix{Float64}}`: The average autocorrelation for each waiting time. The element at index `(i, l, k)` is the average autocorrelation of the trajectories over nodes and ensemble realizations at discretized times `l` and `k` for the waiting time `tws_idxs[i]`.
- `time_indices::Vector{Vector{Int}}`: The time indices used to compute the autocorrelation for each waiting time.
"""
function compute_autocorr(sim::Vector{Matrix{Float64}}, tws_idxs::Vector{Int}; time_indices=nothing, showprogress=false)
    nsim = length(sim)
    N, T = size(sim[1])
    if time_indices === nothing
        time_indices = [tw_idx:T for tw_idx in tws_idxs]
    else
        time_indices = [filter(t -> tw_idx ≤ t ≤ T, time_indices[i]) for (i, tw_idx) in enumerate(tws_idxs)]  
    end
    # Create a progress bar
    p = Progress(Int(N * nsim * sum(length(t_idxs) for t_idxs in time_indices)); enabled=showprogress, dt=1.0, showspeed=true, desc="Computing autocorrelation: ")
    # Initialize autocorrs vector
    autocorrs = [zeros(length(t_idx)) for t_idx in time_indices] # autocorrelations
    # Iterate over simulations and sum into autocorrs
    @inbounds @fastmath for isim in 1:nsim
        autocorrs .+= _autocorr_tws(sim[isim], tws_idxs, time_indices; dims=1, p=p) # Covariances
    end
    # Average over simulations
    autocorrs ./= nsim
    return autocorrs, time_indices
end

""" 
    compute_stats(sim; time_indices=nothing, showprogress=false)

Compute the mean, standard deviation and average autocorrelation of the trajectories in the ensemble solution object.

# Arguments
- `sim::Vector{Matrix{Float64}}`: The ensemble solution object. Each element is a matrix where each column corresponds to a time point.

# Keyword Arguments
- `time_indices::Union{Nothing, AbstractVector{Int}}`: The time indices to compute the mean and standard deviation. If `nothing`, all time indices are used.
- `showprogress::Bool`: Whether to show a progress bar. Default is `false`.

# Returns
- `mean_traj::Vector{Float64}`: The mean trajectory. The element at index `l` is the mean of the trajectories over nodes and ensemble realizations at discretized time `l`.
- `std_traj::Vector{Float64}`: The standard deviation trajectory. The element at index `l` is the standard deviation of the trajectories over nodes and ensemble realizations at discretized time `l`.
- `autocorr::Matrix{Float64}`: The average autocorrelation. The element at index `(l, k)` is the average autocorrelation of the trajectories over nodes and ensemble realizations at discretized times `l` and `k`.
- `t_idx::Vector{Int}`: The time indices used to compute the mean and standard deviation.
"""
function compute_stats(sim::Vector{Matrix{Float64}}; time_indices=nothing, showprogress=false)
    # Compute mean, standard deviation and autocorrelation
    mean_traj, std_traj, t_idxm = compute_meanstd(sim; time_indices=time_indices)
    autocorr, t_idxC = compute_autocorr(sim; time_indices=time_indices, showprogress=showprogress)
    @assert t_idxC == t_idxm "Time indices for mean/std and autocorrelation do not match"
    return mean_traj, std_traj, autocorr, t_idxC
end


"""
    compute_autocorr_TTI(trajs, teq; lag_indices=nothing)

Compute the average autocorrelation of the trajectories in the stationary phase, i.e. after the transient time `teq`. It assumes that after the transient time, the trajectories are stationary, therefore the autocorrelation is time tranlational invariant (TTI), i.e. it only depends on the time differences `C(t,t') = C(t-t')`.

# Arguments
- `trajs::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.
- `teq::Int`: The transient time.

# Keyword Arguments
- `lag_indices::Union{Nothing, AbstractVector{Int}}`: The lag indices to compute the autocorrelation. If `nothing`, all lag indices are used.

# Returns
- `autocorr::Vector{Float64}`: The average TTI autocorrelation. The element at index `l` is the average autocorrelation of the trajectories at discretized time difference `l`.
- `err_autocorr::Vector{Float64}`: The error associated to the TTI autocorrelation. The element at index `l` is error of the average autocorrelation of the trajectories at discretized time difference `l`.
- `l_idx::Vector{Int}`: The lag indices used to compute the autocorrelation.
"""
function compute_autocorr_TTI(trajs::Matrix{Float64}, teq::Int; lag_indices=nothing)
    N, T = size(trajs)    
    
    # Filter lags
    if lag_indices === nothing
        l_idx = 0:lagmax(T-teq+1)
    else
        l_idx = filter(l -> 0 ≤ l ≤ lagmax(T-teq+1), lag_indices)
    end
    L_eff = length(l_idx)

    # Compute the autocorrelation at different lags for each node and simulation
    autocorr_all = autocorr_TTI(trajs[:,teq:T], l_idx)
    # Average over nodes and simulations and estimate error
    autocorr = mean(autocorr_all; dims=1)
    std_autocorr = std(autocorr_all; dims=1, mean=autocorr, corrected=true)
    err_autocorr = std_autocorr ./ sqrt(N)
    # Reshape autocorr and err_autocorr to be vectors
    autocorr = reshape(autocorr, L_eff)
    err_autocorr = reshape(err_autocorr, L_eff)
    return autocorr, err_autocorr, l_idx
end

"""
    compute_stats_TTI(trajs, teq; time_indices=nothing, lag_indices=nothing)

Compute the mean, standard deviation and average autocorrelation of the trajectories in the stationary phase, i.e. after the transient time `teq`. It assumes that after the transient time, the trajectories are stationary, therefore the autocorrelation is time tranlational invariant (TTI), i.e. it only depends on the time differences `C(t,t') = C(t-t')`.

# Arguments
- `trajs::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.
- `teq::Int`: The transient time.

# Keyword Arguments
- `time_indices::Union{Nothing, AbstractVector{Int}}`: The time indices to compute the mean and standard deviation. If `nothing`, all time indices are used.
- `lag_indices::Union{Nothing, AbstractVector{Int}}`: The lag indices to compute the autocorrelation. If `nothing`, all lag indices are used.

# Returns
- `mean_traj::Vector{Float64}`: The mean TTI trajectory. The element at index `l` is the mean of the trajectories at discretized time `l`.
- `std_traj::Vector{Float64}`: The standard deviation of the TTI trajectory. The element at index `l` is the standard deviation of the trajectories at discretized time `l`.
- `t_idx::Vector{Int}`: The time indices used to compute the mean and standard deviation.
- `autocorr::Vector{Float64}`: The average TTI autocorrelation. The element at index `l` is the average autocorrelation of the trajectories at discretized time difference `l`.
- `err_autocorr::Vector{Float64}`: The error associated to the TTI autocorrelation. The element at index `l` is error of the average autocorrelation of the trajectories at discretized time difference `l`.
- `l_idx::Vector{Int}`: The lag indices used to compute the autocorrelation.
"""
function compute_stats_TTI(trajs::Matrix{Float64}, teq::Int; time_indices=nothing, lag_indices=nothing)
    # Compute mean and standard deviation for each time
    mean_traj, std_traj, t_idx = compute_meanstd(trajs; time_indices=time_indices)
    # Compute the autocorrelation
    autocorr, err_autocorr, l_idx = compute_autocorr_TTI(trajs, teq; lag_indices=lag_indices)
    return mean_traj, std_traj, t_idx, autocorr, err_autocorr, l_idx
end

"""
    compute_autocorr_TTI(sim, teq; lag_indices=nothing)

Compute the average autocorrelation of the trajectories in the ensemble solution object in the stationary phase, i.e. after the transient time `teq`. It assumes that after the transient time, the trajectories are stationary, therefore the autocorrelation is time tranlational invariant (TTI), i.e. it only depends on the time differences `C(t,t') = C(t-t')`.

# Arguments
- `sim::Vector{Matrix{Float64}}`: The ensemble solution object. Each element is a matrix where each column corresponds to a time point.
- `teq::Int`: The transient time.

# Keyword Arguments
- `lag_indices::Union{Nothing, AbstractVector{Int}}`: The lag indices to compute the autocorrelation. If `nothing`, all lag indices are used.

# Returns
- `autocorr::Vector{Float64}`: The average TTI autocorrelation. The element at index `l` is the average autocorrelation of the trajectories over nodes and ensemble realizations at discretized time difference `l`.
- `err_autocorr::Vector{Float64}`: The error associated to the TTI autocorrelation. The element at index `l` is error of the average autocorrelation of the trajectories over nodes and ensemble realizations at discretized time difference `l`.
- `l_idx::Vector{Int}`: The lag indices used to compute the autocorrelation.
"""
function compute_autocorr_TTI(sim::Vector{Matrix{Float64}}, teq::Int; lag_indices::Union{Nothing, AbstractVector{Int}}=nothing)
    nsim = length(sim)
    N, T = size(sim[1])    
    
    # Filter lags
    if lag_indices === nothing
        l_idx = 0:lagmax(T-teq+1)
    else
        l_idx = filter(l -> 0 ≤ l ≤ lagmax(T-teq+1), lag_indices)
    end
    L_eff = length(l_idx)

    # Flatten data into an (N+nsim) × T matrix
    sim_flattened = zeros(N * nsim, T-teq+1)
    @inbounds @simd for s in 1:nsim
        sim_flattened[(s-1)*N+1:s*N, :] .= view(sim[s], :, teq:T)
    end

    # Compute the autocorrelation at different lags for each node and simulation
    autocorr_all = autocorr_TTI(sim_flattened, l_idx)
    # Average over nodes and simulations and estimate error
    autocorr = mean(autocorr_all; dims=1)
    std_autocorr = std(autocorr_all; dims=1, mean=autocorr, corrected=true)
    err_autocorr = std_autocorr ./ sqrt(nsim * N)
    # Reshape autocorr and err_autocorr to be vectors
    autocorr = reshape(autocorr, L_eff)
        err_autocorr = reshape(err_autocorr, L_eff)
    return autocorr, err_autocorr, l_idx
end

"""
    compute_stats_TTI(sim, teq; time_indices=nothing, lag_indices=nothing)

Compute the mean, standard deviation and average autocorrelation of the trajectories in the ensemble solution object in the stationary phase, i.e. after the transient time `teq`. It assumes that after the transient time, the trajectories are stationary, therefore the autocorrelation is time tranlational invariant (TTI), i.e. it only depends on the time differences `C(t,t') = C(t-t')`.

# Arguments
- `sim::Vector{Matrix{Float64}}`: The ensemble solution object. Each element is a matrix where each column corresponds to a time point.
- `teq::Int`: The transient time.

# Keyword Arguments
- `time_indices::Union{Nothing, AbstractVector{Int}}`: The time indices to compute the mean and standard deviation. If `nothing`, all time indices are used.
- `lag_indices::Union{Nothing, AbstractVector{Int}}`: The lag indices to compute the autocorrelation. If `nothing`, all lag indices are used.

# Returns
- `mean_traj::Vector{Float64}`: The mean TTI trajectory. The element at index `l` is the mean of the trajectories over nodes and ensemble realizations at discretized time `l`.
- `std_traj::Vector{Float64}`: The standard deviation of the TTI trajectory. The element at index `l` is the standard deviation of the trajectories over nodes and ensemble realizations at discretized time `l`.
- `t_idx::Vector{Int}`: The time indices used to compute the mean and standard deviation.
- `autocorr::Vector{Float64}`: The average TTI autocorrelation. The element at index `l` is the average autocorrelation of the trajectories over nodes and ensemble realizations at discretized time difference `l`.
- err_autocorr::Vector{Float64}: The error associated to the TTI autocorrelation. The element at index `l` is error of the average autocorrelation of the trajectories over nodes and ensemble realizations at discretized time difference `l`.
- `l_idx::Vector{Int}`: The lag indices used to compute the autocorrelation.
"""
function compute_stats_TTI(sim::Vector{Matrix{Float64}}, teq::Int; time_indices::Union{Nothing, AbstractVector{Int}}=nothing, lag_indices::Union{Nothing, AbstractVector{Int}}=nothing)
    # Compute mean and standard deviation for each time
    mean_traj, std_traj, t_idx = compute_meanstd(sim; time_indices=time_indices)
    # Compute the autocorrelation
    autocorr, err_autocorr, l_idx = compute_autocorr_TTI(sim, teq; lag_indices=lag_indices)
    return mean_traj, std_traj, t_idx, autocorr, err_autocorr, l_idx
end