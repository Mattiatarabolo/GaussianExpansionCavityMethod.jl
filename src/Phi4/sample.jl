
# Define update functions for the SDE system (dx = f(x, t) dt + g(x, t) dW)
function f!(du, u, p, t) # deterministic part
    mul!(du, p[1], u) # p[1] = J - lambdas
    du .-= p[2] * u .^ 3 # p[2] = u
end
g!(du, u, p, t) = mul!(du, sqrt(2 * p[3]), u) # stochastic part, p[3] = sqrt(D)
function jacobian!(J, u, p, t) # Jacobian of f
    copyto!(J, p[1])
    @inbounds @fastmath for i in 1:length(u)
        J[i, i] -= 3 * p[2] * u[i] ^ 2
    end
end
milstein_derivative!(J, u, p, t) = fill!(J, 0.0) # Milstein derivative of g: dg/dx * g = 0

"""
    sample_phi4(model, x0, tmax, tsave; rng=Xoshiro(1234), diverging_threshold=1e6)

Sample the Phi^4 model using the SOSRI solver from DifferentialEquations.jl.

# Arguments
- `model::Phi4Model`: The Phi^4 model to sample.
- `x0::Vector{Float64}`: The initial condition.
- `tmax::Float64`: The maximum time to integrate to.
- `tsave::Vector{Float64}`: The time points to save the trajectory at.

# Optional arguments
- `rng::AbstractRNG`: The random number generator to use (default: `Xoshiro(1234)`).
- `diverging_threshold::Float64`: The threshold for detecting diverging solutions (default: `1e6`).

# Returns
- `t_vals::Vector{Float64}`: The time points.
- `trajectories::Matrix{Float64}`: The trajectories. Each column corresponds to a time point.
- `sol::RODESolution`: The solution object.
"""
function sample_phi4(model::Phi4Model, x0::Vector{Float64}, tmax::Float64, tsave::Vector{Float64}; rng=Xoshiro(1234), diverging_threshold=1e6)
    # Define the SDE problem
    p = (model.J .- Diagonal(model.lambdas), model.u, model.D)
    func = SDEFunction(f!, g!, jac=jacobian!, ggprime=milstein_derivative!, jac_prototype=deepcopy(model.J))
    sde = SDEProblem(func, x0, (0.0, tmax), p)
    # Solver options
    isunstable(dt,u,p,t) = any(x->x>diverging_threshold, u)
    # Solve the SDE problem
    sol = solve(sde, SOSRI(); saveat=tsave, seed=rand(rng, UInt32), unstable_check=isunstable)
    if sol.retcode == ReturnCode.Unstable
        throw(error("Diverging solution"))
    end
    # Extract the time points and trajectories
    t_vals = sol.t
    trajectories = hcat(sol.u...) # Convert solution vectors to a matrix
    return t_vals, trajectories, sol
end

"""
    sample_ensemble_phi4(model, x0_min, x0_max, tmax, tsave, nsamples; rng=Xoshiro(1234), diverging_threshold=1e6)

Sample an ensemble of Phi^4 models using the SOSRI solver from DifferentialEquations.jl.

# Arguments
- `model::Phi4Model`: The Phi^4 model to sample.
- `x0_min::Float64`: The minimum initial condition.
- `x0_max::Float64`: The maximum initial condition.
- `tmax::Float64`: The maximum time to integrate to.
- `tsave::Vector{Float64}`: The time points to save the trajectory at.
- `nsamples::Int`: The number of samples to generate.

# Optional arguments
- `rng::AbstractRNG`: The random number generator to use (default: `Xoshiro(1234)`).
- `diverging_threshold::Float64`: The threshold for detecting diverging solutions (default: `1e6`).

# Returns
- `sim::EnsembleSolution`: The ensemble solution object.
"""
function sample_ensemble_phi4(model::Phi4Model, x0_min::Float64, x0_max::Float64, tmax::Float64, tsave::Vector{Float64}, nsamples::Int; rng=Xoshiro(1234), diverging_threshold=1e6)
    # Define the SDE problem
    p = (model.J .- Diagonal(model.lambdas), model.u, model.D)
    func = SDEFunction(GaussianExpansionCavityMethod.f!, GaussianExpansionCavityMethod.g!, jac=GaussianExpansionCavityMethod.jacobian!, ggprime=GaussianExpansionCavityMethod.milstein_derivative!, jac_prototype=deepcopy(model.J))
    x0 = rand() * (x0_max - x0_min) + x0_min
    sde = SDEProblem(func, x0, (0.0, tmax), p)
    # Define the ensemble problem function, which changes the initial condition and the adjacency matrix at each simulation
    prob_func = let p = p
        (prob, i, repeat) -> begin
        J = adjacency_matrix(random_regular_graph(model.N, model.K)) .* 1.2     
        remake(prob, u0 = rand(model.N) .* ((x0_max - x0_min) + x0_min), p = (J, p[2], p[3]))
        end
    end
    # Define the ensemble problem
    ensemble_prob = EnsembleProblem(sde, prob_func=prob_func)
    # Solver options
    isunstable(dt,u,p,t) = any(x->x>diverging_threshold, u)
    # Solve the ensemble problem
    sim = solve(ensemble_prob, SOSRI(); saveat=tsave, seed=rand(rng, UInt32), unstable_check=isunstable, trajectories = nsamples)
    return sim
end

"""
    sample_ensemble_phi4(model_ensemble, x0_min, x0_max, tmax, tsave, nsample; rng=Xoshiro(1234), diverging_threshold=1e6, showprogress=false)

Sample an ensemble of Phi^4 models using the SOSRI solver from DifferentialEquations.jl.

# Arguments
- `model_ensemble::Phi4ModelEnsemble`: The ensemble of Phi^4 models to sample.
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
- `sim::Vector{RODESolution}`: The solution objects for each sample.
"""
function sample_ensemble_phi4(ensemble_model::Phi4ModelEnsemble, x0_min::Float64, x0_max::Float64, tmax::Float64, tsave::Vector{Float64}, nsample::Int; rng=Xoshiro(1234), diverging_threshold=1e6, showprogress=false)
    # Define threadsafe variables
    local_ensemble_models = [deepcopy(ensemble_model) for _ in 1:Threads.nthreads()]
    local_x0_lims = [(x0_min, x0_max) for _ in 1:Threads.nthreads()]
    local_tsaves = [tsave for _ in 1:Threads.nthreads()]
    local_tmaxs = [tmax for _ in 1:Threads.nthreads()]
    local_rngs = [Xoshiro() for _ in 1:Threads.nthreads()]
    # Initialize vector to store results
    sim = Vector{RODESolution}(undef, nsample)
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
        local_model = Phi4Model(local_ensemble_model.K, J, local_ensemble_model.lambdas, local_ensemble_model.D, local_ensemble_model.u)
        # Update the model with the new adjacency matrix
        local_model.J .= J
        # Sample the initial condition
        x0 = rand(local_rng, local_model.N) .* ((local_x0_max - local_x0_min) + local_x0_min)
        tvals, trajectories, sol = sample_phi4(local_model, x0, local_tmax, local_tsave; rng=local_rng, diverging_threshold=diverging_threshold)
        # Store the results using threadsafe access
        lock(lk) do
            sim[i] = sol
            traj_alls[i] .= trajectories
            tvals_alls[i] .= tvals
        end
        # Update the progress bar
        next!(progbar)
    end
    # Close the progress bar
    finish!(progbar)
    return tvals_alls, traj_alls, sim
end