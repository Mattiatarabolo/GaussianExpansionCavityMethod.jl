
# Define update functions for the SDE system (dx = f(x, t) dt + g(x, t) dW)
function f_BM!(du, u, p, t) # deterministic part
    mul!(du, p[1], u) # p[1] = J - lambdas
end
g_BM!(du, u, p, t) = mul!(du, p[2], u) # stochastic part, p[2] = sigma
jacobian_BM!(J, u, p, t) = copyto!(J, p[1]) # Jacobian of f
function milstein_derivative_BM!(J, u, p, t) # Milstein derivative of g: dg/dx * g = 0
    fill!(J, 0.0)
    @inbounds @fastmath for i in 1:length(u)
        J[i, i] += p[2]^2 * u[i]
    end
end

# Define an algorithm switching solver for both stiff and non-stiff problems
choice_function_BM(integrator) = (Int(integrator.dt < 0.001) + 1)
alg_switch_BM = StochasticCompositeAlgorithm((ImplicitEM(), ISSEM()), choice_function_BM)

# Define a isoutofdomain function for the Bouchaud-Mezard model
isoutofdomain_BM(u,p,t) = any(x -> x < 0, u)


"""
    sample_BM(model, x0, tmax, tsave; rng=Xoshiro(1234), diverging_threshold=1e6)

Sample the Bouchaud-Mezard model using solvers from DifferentialEquations.jl.

# Arguments
- `model::BMModel`: The Bouchaud-Mezard model to sample.
- `x0::Vector{Float64}`: The initial condition.
- `tmax::Float64`: The maximum time to integrate to.
- `tsave::Vector{Float64}`: The time points to save the trajectory at.

# Optional arguments
- `rng::AbstractRNG`: The random number generator to use (default: `Xoshiro(1234)`).
- `diverging_threshold::Float64`: The threshold for detecting diverging solutions (default: `1e6`).

# Returns
- `t_vals::Vector{Float64}`: The time points.
- `trajectories::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.
"""
function sample_BM(model::BMModel, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64}; rng=Xoshiro(1234), diverging_threshold=1e6)
    # Define the SDE problem
    p = (model.J, model.sigma)
    func = SDEFunction(f_BM!, g_BM!, jac=jacobian_BM!, ggprime=milstein_derivative_BM!, jac_prototype=deepcopy(model.J))
    sde = SDEProblem(func, x0, (0.0, tmax), p)
    # Solver options
    isunstable(dt,u,p,t) = any(x->x>diverging_threshold, u)
    # Solve the SDE problem
    sol = solve(sde, alg_switch_BM; reltol=1e-3, abstol=1e-4, dt=0.01, saveat=tsave, seed=rand(rng, UInt32), unstable_check=isunstable)#, isoutofdomain=isoutofdomain_BM)
    if sol.retcode == ReturnCode.Unstable
        throw(error("Diverging solution"))
    end
    # Extract the time points and trajectories
    t_vals = sol.t
    trajectories = hcat(sol.u...) # Convert solution vectors to a matrix
    return t_vals, trajectories
end

"""
    sample_ensemble_BM(model_ensemble, x0_min, x0_max, tmax, tsave, nsample; rng=Xoshiro(1234), diverging_threshold=1e6, showprogress=false)

Sample an ensemble of Bouchaud-Mezard models using solvers from DifferentialEquations.jl.

# Arguments
- `model_ensemble::BMModelEnsemble`: The ensemble of Phi^4 models to sample.
- `x0_min::Float64`: The minimum initial condition.
- `x0_max::Float64`: The maximum initial condition.
- `tmax::Float64`: The maximum time to integrate to.
- `tsave::Vector{Float64}`: The time points to save the trajectory at.
- `nsample::Int`: The number of samples to generate.

# Optional arguments
- `rng::AbstractRNG`: The random number generator to use (default: `Xoshiro(1234)`).
- `diverging_threshold::Float64`: The threshold for detecting diverging solutions (default: `1e6`).
- `showprogress::Bool`: Whether to show a progress bar (default: `false`).

# Returns
- `tvals_alls::Vector{Vector{Float64}}`: The time points for each sample.
- `traj_alls::Vector{Matrix{Float64}}`: The trajectories for each sample. Each column corresponds to a time point.
"""
function sample_ensemble_BM(ensemble_model::BMModelEnsemble, x0_min::Float64, x0_max::Float64, tmax::Float64, tsave::Vector{Float64}, nsample::Int; rng=Xoshiro(1234), diverging_threshold=1e6, showprogress=false)
    # Define threadsafe variables
    local_ensemble_models = [deepcopy(ensemble_model) for _ in 1:Threads.nthreads()]
    local_x0_lims = [(x0_min, x0_max) for _ in 1:Threads.nthreads()]
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
            local_x0_min, local_x0_max = local_x0_lims[Threads.threadid()]
            local_tmax = local_tmaxs[Threads.threadid()]
            local_tsave = local_tsaves[Threads.threadid()]
            local_rng = local_rngs[Threads.threadid()]
            Random.seed!(local_rng, seeds[i])
        #end
        # Generate a random adjacency matrix
        J = local_ensemble_model.gen_J(local_ensemble_model.N, local_ensemble_model.K, local_ensemble_model.J_params; rng=local_rng)
        local_model = BMModel(local_ensemble_model.K, J, local_ensemble_model.sigma)
        # Update the model with the new adjacency matrix
        local_model.J .= J
        # Sample the initial condition
        x0 = rand(local_rng, local_model.N) .* (local_x0_max - local_x0_min) .+ local_x0_min
        tvals, trajectories = sample_BM(local_model, x0, local_tmax, local_tsave; rng=local_rng, diverging_threshold=diverging_threshold)
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