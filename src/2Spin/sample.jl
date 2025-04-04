# Define the function to update the Lagrange multiplier (spherical constraint)
function update_lambda(u,dW, dt, model)
    lambda_new = u' * model.J * u + dot(u, dW) / dt
    return lambda_new / model.N
end
# Define update functions for the SDE system (dx = f(x, t) dt + g(x, t) dW)
function f_2Spin!(du, u,lambda, dt, model) # deterministic part
    mul!(du, model.J, u)
    du .-= u .* lambda
    du .*= dt
end


"""
    sample_2Spin(model, x0, tmax, tsave; rng=Xoshiro(1234), dt=1e-4)

Sample the spherical 2-Spin model using the Euler-Maruyama solver.

# Arguments
- `model::TwoSpinModel`: The spherical 2-Spin model to sample.
- `x0::Vector{Float64}`: The initial condition.
- `tmax::Float64`: The maximum time to integrate to.
- `tsave::Vector{Float64}`: The time points to save the trajectory at.

# Optional arguments
- `rng::AbstractRNG`: The random number generator to use (default: `Xoshiro(1234)`).
- `dt::Float64`: The time step size (default: `1e-4`).

# Returns
- `tvec::Vector{Float64}`: The time points.
- `traj::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.
- `lambda_traj::Vector{Float64}`: The Lagrange multipliers at each time point.
"""
function sample_2Spin(model::TwoSpinModel, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64}; rng=Xoshiro(1234), dt=1e-4)
    @assert length(x0) == model.N "x0 must have the same length as the number of spins in the model"
    @assert tmax > 0 "tmax must be positive"
    @assert dt > 0 "dt must be positive"
    @assert length(tsave) > 0 "tsave must have at least one time point"
    @assert sum(x0 .^ 2) ≈ model.N "x0 must be spherical"
    # Unpack model parameters
    N, D = model.N, model.D
    # Compute total number of time steps in the EM solver
    T = ceil(Int, tmax / dt) # Number of iterations
    # Initialize time vector and trajectory array
    tvec = zeros(length(tsave))
    traj = zeros(N, length(tsave))
    lambda_traj = zeros(length(tsave))
    # Initialize state vectors (used internally in the solver)
    x = copy(x0)
    dx = Vector{Float64}(undef, N)
    dW = Vector{Float64}(undef, N)

    # First step at t=0
    t = 0.0 # Initial time
    # Generate random white noise
    randn!(rng, dW)
    dW .*= sqrt(dt * 2 * D) 
    # Update Lagrange multiplier
    lambda = update_lambda(x, dW, dt, model)
    # Check if the first time point is within the range of tsave
    save_idx = 1
    if tsave[save_idx] == 0.0 || tsave[save_idx] <= (t + dt/2)
        tvec[save_idx] = t
        copy!(view(traj, :, save_idx), x)
        lambda_traj[save_idx] = lambda
        save_idx += 1
    end
    # Iterate over time steps
    @inbounds @fastmath for iter in 1:T
        # Update time
        t += dt #(t = iter * dt)
        # Update state vector using the Euler-Maruyama method
        f_2Spin!(dx, x, lambda, dt, model)
        x .+= dx
        x .+= dW
        # Generate random white noise
        randn!(rng, dW)
        dW .*= sqrt(dt * 2 * D) 
        # Update Lagrange multiplier
        lambda = update_lambda(x, dW, dt, model)
        # Update time vector and trajectory array
        if save_idx <= length(tsave) && (t + dt/2) >= tsave[save_idx]
            tvec[save_idx] = t
            copy!(view(traj, :, save_idx), x)
            lambda_traj[save_idx] = lambda
            save_idx += 1
        end
        # Stop early if all desired times are recorded
        if save_idx > length(tsave)
            break
        end
    end
    return tvec, traj, lambda_traj
end

"""
    sample_ensemble_2Spin(model_ensemble, x0_min, x0_max, tmax, tsave, nsample; rng=Xoshiro(1234), showprogress=false, dt=1e-4)

Sample an ensemble of spherical 2-Spin models using solvers from DifferentialEquations.jl.

# Arguments
- `model_ensemble::TwoSpinModelEnsemble`: The ensemble of Phi^4 models to sample.
- `x0_min::Float64`: The minimum initial condition.
- `x0_max::Float64`: The maximum initial condition.
- `tmax::Float64`: The maximum time to integrate to.
- `tsave::Vector{Float64}`: The time points to save the trajectory at.
- `nsample::Int`: The number of samples to generate.

# Optional arguments
- `rng::AbstractRNG`: The random number generator to use (default: `Xoshiro(1234)`).
- `showprogress::Bool`: Whether to show a progress bar (default: `false`).
- `dt::Float64`: The time step size (default: `1e-4`).

# Returns
- `tvals_alls::Vector{Vector{Float64}}`: The time points for each sample.
- `traj_alls::Vector{Matrix{Float64}}`: The trajectories for each sample. Each column corresponds to a time point.
- `lambda_traj_alls::Vector{Vector{Float64}}`: The Lagrange multipliers for each sample.
"""
function sample_ensemble_2Spin(ensemble_model::TwoSpinModelEnsemble, x0_min::Float64, x0_max::Float64, tmax::Float64, tsave::Vector{Float64}, nsample::Int; rng=Xoshiro(1234), showprogress=false, dt=1e-4)
    # Define threadsafe variables
    local_ensemble_models = [deepcopy(ensemble_model) for _ in 1:Threads.nthreads()]
    local_x0_lims = [(x0_min, x0_max) for _ in 1:Threads.nthreads()]
    local_tsaves = [tsave for _ in 1:Threads.nthreads()]
    local_tmaxs = [tmax for _ in 1:Threads.nthreads()]
    local_rngs = [Xoshiro() for _ in 1:Threads.nthreads()]
    # Initialize vector to store results
    traj_alls = [Matrix{Float64}(undef, ensemble_model.N, length(tsave)) for _ in 1:nsample]
    lambda_traj_alls = [Vector{Float64}(undef, length(tsave)) for _ in 1:nsample]
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
        local_ensemble_model = local_ensemble_models[Threads.threadid()]
        local_x0_min, local_x0_max = local_x0_lims[Threads.threadid()]
        local_tmax = local_tmaxs[Threads.threadid()]
        local_tsave = local_tsaves[Threads.threadid()]
        local_rng = local_rngs[Threads.threadid()]
        Random.seed!(local_rng, seeds[i])
        # Generate a random adjacency matrix
        J = local_ensemble_model.gen_J(local_ensemble_model.N, local_ensemble_model.K, local_ensemble_model.J_params; rng=local_rng)
        local_model = TwoSpinModel(local_ensemble_model.K, J, local_ensemble_model.D)
        # Update the model with the new adjacency matrix
        local_model.J .= J
        # Sample the initial condition
        x0 = rand(local_rng, local_model.N) * (local_x0_max - local_x0_min) .+ local_x0_min
        x0 .*= sqrt(local_model.N / sum(x0 .^ 2))
        tvals, trajectories, lambda_traj = sample_2Spin(local_model, x0, local_tmax, local_tsave; rng=local_rng, dt=dt)
        # Store the results using threadsafe access
        lock(lk) do
            copy!(traj_alls[i], trajectories)
            copy!(tvals_alls[i], tvals)
            copy!(lambda_traj_alls[i], lambda_traj)
        end
        # Update the progress bar
        next!(progbar)
    end
    # Close the progress bar
    finish!(progbar)
    return tvals_alls, traj_alls, lambda_traj_alls
end
