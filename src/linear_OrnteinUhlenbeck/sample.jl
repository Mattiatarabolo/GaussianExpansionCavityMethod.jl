
# Define update functions for the SDE system (dx = f(x, t) dt + g(x, t) dW)
function f_OU!(du, u, dt, model) # deterministic part
    mul!(du, model.J, u)
    du .-= u .* model.lambdas
    du .*= dt
end

"""
    sample_OU(model, x0, tmax, tsave; rng=Xoshiro(1234), diverging_threshold=1e6, dt=1e-3)

Sample the Ornstein-Uhlenbeck model using the Euler-Maruyama solver.

# Arguments
- `model::OUModel`: The Ornstein-Uhlenbeck model to sample.
- `x0::Vector{Float64}`: The initial condition.
- `tmax::Float64`: The maximum time to integrate to.
- `tsave::Vector{Float64}`: The time points to save the trajectory at.

# Optional arguments
- `rng::AbstractRNG`: The random number generator to use (default: `Xoshiro(1234)`).
- `diverging_threshold::Float64`: The threshold for detecting diverging solutions (default: `1e6`).
- `dt::Float64`: The time step size (default: `1e-3`).

# Returns
- `t_vals::Vector{Float64}`: The time points.
- `trajectories::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.
- `sol::RODESolution`: The solution object.
"""
function sample_OU(model::OUModel, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64}; rng=Xoshiro(1234), diverging_threshold=1e6, dt=1e-3)
    @assert length(x0) == model.N "x0 must have the same length as the number of spins in the model"
    @assert tmax > 0 "tmax must be positive"
    @assert dt > 0 "dt must be positive"
    @assert length(tsave) > 0 "tsave must have at least one time point"
    # Define isoutofdomain function
    isoutofdomain(u) = any(x -> x > diverging_threshold, u)
    # Unpack model parameters
    N, D = model.N, model.D
    # Compute total number of time steps in the EM solver
    T = ceil(Int, tmax / dt) # Number of iterations
    # Initialize time vector and trajectory array
    tvec = zeros(length(tsave))
    traj = zeros(N, length(tsave))
    # Initialize state vectors (used internally in the solver)
    x = copy(x0)
    dx = Vector{Float64}(undef, N)
    dW = Vector{Float64}(undef, N)

    # First step at t=0
    t = 0.0 # Initial time
    # Generate random white noise
    randn!(rng, dW)
    dW .*= sqrt(dt * 2 * D) 
    # Check if the first time point is within the range of tsave
    save_idx = 1
    if tsave[save_idx] == 0.0 || tsave[save_idx] <= (t + dt/2)
        tvec[save_idx] = t
        copy!(view(traj, :, save_idx), x)
        save_idx += 1
    end
    # Iterate over time steps
    @inbounds @fastmath for iter in 1:T
        # Update time
        t += dt #(t = iter * dt)
        # Update state vector using the Euler-Maruyama method
        f_OU!(dx, x, dt, model)
        x .+= dx
        x .+= dW
        # Generate random white noise
        randn!(rng, dW)
        dW .*= sqrt(dt * 2 * D) 
        # Update time vector and trajectory array
        if save_idx <= length(tsave) && (t + dt/2) >= tsave[save_idx]
            tvec[save_idx] = t
            copy!(view(traj, :, save_idx), x)
            save_idx += 1
        end
        # Check for divergence
        if isoutofdomain(x)
            println("Diverging solution at t = $t")
            # Set the remaining time points to the threshold value
            copy!(view(tvec,save_idx:length(tsave)), view(tsave,save_idx:length(tsave)))
            fill!(view(traj,:,save_idx:length(tsave)), diverging_threshold)
            break
        end
        # Stop early if all desired times are recorded
        if save_idx > length(tsave)
            break
        end
    end
    return tvec, traj
end

# Set the remaining time points to the threshold value

"""
    sample_ensemble_OU(model_ensemble, tmax, tsave, nsample; rng=Xoshiro(1234), diverging_threshold=1e6, showprogress=false, dt=1e-3)

Sample an ensemble of Ornstein-Uhlenbeck models using solvers from DifferentialEquations.jl.

# Arguments
- `model_ensemble::OUModelEnsemble`: The ensemble of Ornstein-Uhlenbeck models to sample.
- `tmax::Float64`: The maximum time to integrate to.
- `tsave::Vector{Float64}`: The time points to save the trajectory at.
- `nsample::Int`: The number of samples to generate.

# Optional arguments
- `rng::AbstractRNG`: The random number generator to use (default: `Xoshiro(1234)`).
- `diverging_threshold::Float64`: The threshold for detecting diverging solutions (default: `1e6`).
- `showprogress::Bool`: Whether to show a progress bar (default: `false`).
- `dt::Float64`: The time step size (default: `1e-3`).

# Returns
- `tvals_alls::Vector{Vector{Float64}}`: The time points for each sample.
- `traj_alls::Vector{Matrix{Float64}}`: The trajectories for each sample. Each column corresponds to a time point.
- `sim::Vector{RODESolution}`: The solution objects for each sample.
"""
function sample_ensemble_OU(ensemble_model::OUModelEnsemble, tmax::Float64, tsave::Vector{Float64}, nsample::Int; rng=Xoshiro(1234), diverging_threshold=1e6, showprogress=false, dt=1e-3)
    # Define threadsafe variables
    local_ensemble_models = [deepcopy(ensemble_model) for _ in 1:Threads.nthreads()]
    local_tsaves = [tsave for _ in 1:Threads.nthreads()]
    local_tmaxs = [tmax for _ in 1:Threads.nthreads()]
    local_rngs = [Xoshiro() for _ in 1:Threads.nthreads()]
    # Initialize vector to store results
    traj_alls = [Matrix{Float64}(undef, ensemble_model.N, length(tsave)) for _ in 1:nsample]
    tvals_alls = [Vector{Float64}(undef, length(tsave)) for _ in 1:nsample]
    # Generate random seeds for each thread
    seeds = rand(rng, UInt32, nsample)
    # Create a lock for thread-safe access to variables
    lk = ReentrantLock()
    # Create a progress bar
    progbar = Progress(nsample; enabled=showprogress, dt=1.0, showspeed=true)
    # Sample the ensemble in parallel
    Threads.@threads for i in 1:nsample
        # Initialize local variables
        #lock(lk) do
            local_ensemble_model = local_ensemble_models[Threads.threadid()]
            local_tmax = local_tmaxs[Threads.threadid()]
            local_tsave = local_tsaves[Threads.threadid()]
            local_rng = local_rngs[Threads.threadid()]
            Random.seed!(local_rng, seeds[i])
        #end
        # Generate a random adjacency matrix
        J = local_ensemble_model.gen_J(local_ensemble_model.N, local_ensemble_model.K, local_ensemble_model.J_params; rng=local_rng)
        local_model = OUModel(local_ensemble_model.K, J, local_ensemble_model.lambdas, local_ensemble_model.D)
        # Update the model with the new adjacency matrix
        local_model.J .= J
        # Sample the initial condition
        x0 = local_ensemble_model.gen_x0(local_ensemble_model.N, local_ensemble_model.x0_params; rng=local_rng)
        tvals, trajectories = sample_OU(local_model, x0, local_tmax, local_tsave; rng=local_rng, diverging_threshold=diverging_threshold, dt=dt)
        # Store the results using threadsafe access
        lock(lk) do
            traj_alls[i] .= trajectories
            tvals_alls[i] .= tvals
        end
        # Update the progress bar
        next!(progbar)
    end
    # Close the progress bar
    finish!(progbar)
    return tvals_alls, traj_alls
end