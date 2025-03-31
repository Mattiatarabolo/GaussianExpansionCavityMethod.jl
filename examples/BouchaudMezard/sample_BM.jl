using GaussianExpansionCavityMethod
using Graphs, Random, JLD2, ProgressMeter

# Check existence of data path
isdir("data") || mkdir("data")
isdir("data/RRG") || mkdir("data/RRG")

# Define ensemble parameters
N = 5000
Ks = [3, 10, 500]
J = 1.0
sigmas_lims = [(1.5, 1.8), (1.8, 2.0), (1.9, 2.1)]

nsigmas = 20
sigmass = [collect(range(sigma_min, stop=sigma_max, length=nsigmas)) for (sigma_min, sigma_max) in sigmas_lims]

# Define simulation parameters
tmax = 1500.0
naverage = 50
tsave = collect(range(tmax-500.0, stop=tmax, length=naverage))
nsim = 30
x0_min, x0_max = 1.0, 2.0

# Initialize thread-safe variables
local_x0_lims = [(x0_min, x0_max) for _ in 1:Threads.nthreads()]
local_tsaves = [tsave for _ in 1:Threads.nthreads()]
local_tmaxs = [tmax for _ in 1:Threads.nthreads()]
local_rngs = [Xoshiro() for _ in 1:Threads.nthreads()]

# Generate random seeds for each sample
rng = Xoshiro(1234)
seeds = rand(rng, UInt32, nsim)
# Generate random adjacencey matrices for each sample (will use same matrix for each value of (D, lam) with same seed)
function gen_Jmat(N, K, J; rng=Xoshiro(1234))
    Jmat = adjacency_matrix(random_regular_graph(N, K; rng=rng)) .* J
    @inbounds @fastmath @simd for i in 1:N
        Jmat[i,i] = - J * K
    end
    return Jmat
end
J_matrices = Dict((K, isim) => gen_Jmat(N, K, J; rng=rng) for (K, isim) in Iterators.product(Ks, 1:nsim))

# Create a lock for thread-safe access to variables
lk = ReentrantLock()
# Create a progress bar
progbar = Progress(nsim * nsigmas * length(Ks); dt=1.0, showspeed=true)

# Create vector iterator to loop over K values, sigma values, and samples of the disordered ensemble
iter_vec = Vector{Tuple{Int, Float64, Int}}(undef, length(Ks) * nsigmas * nsim)
global index_iterator = 1
for (iK, K) in enumerate(Ks)
    for (sigma, isim) in Iterators.product(sigmass[iK], 1:nsim)
        iter_vec[index_iterator] = (sigma, isim)
        global index_iterator += 1
    end
end

# Create specific directory for saving results
savedirs = Dict()
for (iK, K) in enumerate(Ks)
    savedir = "data/RRG/N-$(N)_K-$(K)_J-$(J)_u-$(u)_sigmalims-$((sigmas_lims[iK][1],sigmas_lims[iK][2]))_nsigmas-$(nsigmas)_tmax-$(tmax)_naverage-$(naverage)_nsim-$(nsim)/"
    savedirs[K] = savedir
    isdir(savedir) || mkdir(savedir)
end

# Loop over K values, sigma values and samples of the disordered ensemble
Threads.@threads for (K, sigma, isim) in iter_vec
    local_J = J_matrices[(K, isim)]
    local_x0_min, local_x0_max = local_x0_lims[Threads.threadid()]
    local_tmax = local_tmaxs[Threads.threadid()]
    local_tsave = local_tsaves[Threads.threadid()]
    local_rng = local_rngs[Threads.threadid()]
    Random.seed!(local_rng, seeds[isim])
    # Generate a local model with the current parameters
    local_model = BMModel(K, local_J, sigma)
    # Sample the initial condition
    x0 = rand(local_rng, local_model.N) .* ((local_x0_max - local_x0_min) + local_x0_min)
    tvals, trajectories = sample_BM(local_model, x0, local_tmax, local_tsave; rng=local_rng, diverging_threshold=1e10)
    # Save the results using threadsafe access
    lock(lk) do 
        filename = savedirs[K] * "sol_sigma-$(sigma)_isim-$(isim).jld2"
        jldsave(filename; tvals, trajectories, seed=seeds[isim])
    end
    # Update the progress bar
    next!(progbar)
end
# Close the progress bar
finish!(progbar)