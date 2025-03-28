using GaussianExpansionCavityMethod
using Graphs, Random, JLD2, ProgressMeter

# Check existence of data path
isdir("data") || mkdir("data")
isdir("data/RRG") || mkdir("data/RRG")

# Define ensemble parameters
N = 200
Ks = [3, 10]
J, u = 1.0, 0.01
D_min, D_max = 1e-3, 1.0
lamdas_lims = [(2.96, 3.03), (9.98, 10.02)]

nmesh = 50
Ds = collect(10 .^ range(log10(D_min), stop=log10(D_max), length=nmesh))
lambdass = [collect(range(lims[1], stop=lims[2], length=nmesh)) for lims in lamdas_lims]

# Define simulation parameters
tmax = 1000.0
naverage = 50
tsave = collect(range(tmax-10.0, stop=tmax, length=naverage))
nsim = 30
x0_min, x0_max = 0.1, 2.0

# Initialize thread-safe variables
local_x0_lims = [(x0_min, x0_max) for _ in 1:Threads.nthreads()]
local_tsaves = [tsave for _ in 1:Threads.nthreads()]
local_tmaxs = [tmax for _ in 1:Threads.nthreads()]
local_rngs = [Xoshiro() for _ in 1:Threads.nthreads()]
local_us = [u for _ in 1:Threads.nthreads()]

# Generate random seeds for each sample
rng = Xoshiro(1234)
seeds = rand(rng, UInt32, nsim)
# Generate random adjacencey matrices for each sample (will use same matrix for each value of (D, lam) with same seed)
J_matrices = Dict((K, isim) => adjacency_matrix(random_regular_graph(N, K; rng=rng)) .* J for (K, isim) in Iterators.product(Ks, 1:nsim))

# Create a lock for thread-safe access to variables
lk = ReentrantLock()
# Create a progress bar
progbar = Progress(nsim * nmesh^2 * length(Ks); dt=1.0, showspeed=true)

# Create vector iterator to loop over K values, D values, lambda values and samples of the disordered ensemble
iter_vec = Vector{Tuple{Int, Float64, Float64, Int}}(undef, length(Ks) * nmesh^2 * nsim)
global index_iterator = 1
for (iK, K) in enumerate(Ks)
    for (lam, D, isim) in Iterators.product(lambdass[iK], Ds, 1:nsim)
        iter_vec[index_iterator] = (K, D, lam, isim)
        global index_iterator += 1
    end
end
    
# Create specific directory for saving results
savedirs = Dict()
for (iK, K) in enumerate(Ks)
    savedir = "data/RRG/N-$(N)_K-$(K)_J-$(J)_u-$(u)_Dlims-$((D_min,D_max))_lamlims-$((lamdas_lims[iK][1],lamdas_lims[iK][2]))_nmesh-$(nmesh)_tmax-$(tmax)_naverage-$(naverage)_nsim-$(nsim)/"
    savedirs[K] = savedir
    isdir(savedir) || mkdir(savedir)
end

# Loop over K values, D values, lambda values and samples of the disordered ensemble
Threads.@threads for (K, D, lam, isim) in iter_vec
    local_J = J_matrices[(K, isim)]
    local_x0_min, local_x0_max = local_x0_lims[Threads.threadid()]
    local_tmax = local_tmaxs[Threads.threadid()]
    local_tsave = local_tsaves[Threads.threadid()]
    local_u = local_us[Threads.threadid()]
    local_rng = local_rngs[Threads.threadid()]
    Random.seed!(local_rng, seeds[isim])
    # Generate a local model with the current parameters
    local_model = Phi4Model(K, local_J, lam, D, local_u)
    # Sample the initial condition
    x0 = rand(local_rng, local_model.N) .* ((local_x0_max - local_x0_min) + local_x0_min)
    tvals, trajectories, sol = sample_phi4(local_model, x0, local_tmax, local_tsave; rng=local_rng, diverging_threshold=1e6)
    # Save the results using threadsafe access
    lock(lk) do 
        filename = savedirs[K] * "sol_D-$(D)_lambda-$(lam)_isim-$(isim).jld2"
        jldsave(filename; tvals, trajectories, sol, seed=seeds[isim])
    end
    # Update the progress bar
    next!(progbar)
end
# Close the progress bar
finish!(progbar)