using Random, GaussianExpansionCavityMethod, ProgressMeter, JLD2
using PyCall, Conda
mpl_toolkits = pyimport("mpl_toolkits.axes_grid1")
Line2D = pyimport("matplotlib.lines").Line2D
import PyPlot as plt
using LaTeXStrings

rcParams = PyDict(plt.matplotlib["rcParams"])
rcParams["font.size"] = 8

function run_Bin_RRG_single_nodes(N, K, J, D, tmax, nsim, dt, dt_save, ilist)
    rng = Xoshiro(1234)

    x0_min, x0_max = -3.0, 3.0
    
    tsave = collect(range(0, tmax; step=dt_save))

    # Define ensemble model
    model_ensemble = TwoSpinModelRRG_Bim(N, K, J, D)

    # Sample ensemble
    _, traj_alls, _ = sample_ensemble_2Spin(model_ensemble, x0_min, x0_max, tmax, tsave, nsim; rng=rng, showprogress=true, dt=dt)
    
    
    # Compute autocorrelation function for 2 nodes only
    trajs = [zeros(nsim, length(tsave)) for _ in 1:length(ilist)]
    autocorrs = Vector{Matrix{Float64}}(undef, length(ilist))
    tidxs = Vector{Vector{Int}}(undef, length(ilist))
    for (idx, i) in enumerate(ilist)
        @inbounds for isim in 1:nsim
            trajs[idx][isim, :] .= traj_alls[isim][i, :]
        end
        autocorrs[idx], tidxs[idx] = compute_autocorr(trajs[idx]; showprogress=true)
    end
    
    # Save results
    isdir("data") || mkdir("data")
    isdir("data/RRG") || mkdir("data/RRG")

    filename = "data/RRG/Corr_Bim_N-$(N)_K-$(K)_J-$(J)_D-$(D)_tmax-$(tmax)_dtMC-$(dt)_dtsave-$(dt_save)_nsim-$(nsim)_x0min-$(x0_min)_x0max-$(x0_max).jld2"
    jldsave(filename; tsave, tidxs, autocorrs)
end

function run_Bim_RRG(N, K, J, D, tmax, nsim, dt, dt_save)
    rng = Xoshiro(1234)

    x0_min, x0_max = -3.0, 3.0
    
    tsave = collect(range(0, tmax; step=dt_save))

    # Define ensemble model
    model_ensemble = TwoSpinModelRRG_Bim(N, K, J, D)

    # Sample ensemble
    _, traj_alls, _ = sample_ensemble_2Spin(model_ensemble, x0_min, x0_max, tmax, tsave, nsim; rng=rng, showprogress=true, dt=dt)

    # Compute correlation function
    autocorr_traj, tidxs = compute_autocorr(traj_alls; showprogress=true)
    tvec = tsave[tidxs]
    
    # Save results
    isdir("data") || mkdir("data")
    isdir("data/RRG") || mkdir("data/RRG")

    filename = "data/RRG/Corr_Bim_N-$(N)_K-$(K)_J-$(J)_D-$(D)_tmax-$(tmax)_dtMC-$(dt)_dtsave-$(dt_save)_nsim-$(nsim)_x0min-$(x0_min)_x0max-$(x0_max).jld2"
    jldsave(filename; autocorr_traj, tsave, tidxs)
end

function run_Bim_RRG(N, K, J, D, tmax, nsim, dt, dt_save, tws, nts)
    rng = Xoshiro(1234)

    x0_min, x0_max = -3.0, 3.0

    tsave = collect(range(0, tmax; step=dt_save))

    # Define ensemble model
    model_ensemble = TwoSpinModelRRG_Bim(N, K, J, D)

    # Sample ensemble
    _, traj_alls, _ = sample_ensemble_2Spin(model_ensemble, x0_min, x0_max, tmax, tsave, nsim; rng=rng, showprogress=true, dt=dt)

    # Create tws_idxs vector
    tws_idxs = [argmin(abs.(tsave .- tw)) for tw in tws]
    # Create vector of time indices at which to compute the correlation function
    time_indices = Vector{Vector{Int}}(undef, length(tws_idxs))
    for (i, tw_idx) in enumerate(tws_idxs)
        imin, imax = tw_idx, tw_idx + 100
        logmin, logmax = log10(imin), log10(imax)
        indices = round.(Int, exp10.(range(logmin, logmax, length=nts)))
        time_indices[i] = unique(indices[indices .>= imin .&& indices .<= imax])
    end

    # Compute correlation function
    autocorr_traj, tidxs = compute_autocorr(traj_alls, tws_idxs; time_indices=time_indices, showprogress=true)
    
    # Save results
    isdir("data") || mkdir("data")
    isdir("data/RRG") || mkdir("data/RRG")

    filename = "data/RRG/Corr_Bim_N-$(N)_K-$(K)_J-$(J)_D-$(D)_tmax-$(tmax)_dtMC-$(dt)_dtsave-$(dt_save)_nsim-$(nsim)_x0min-$(x0_min)_x0max-$(x0_max).jld2"
    jldsave(filename; autocorr_traj, tsave, tidxs)
end


function run_Ferro_RRG(N, K, J, D, tmax, nsim, dt, dt_save)
    rng = Xoshiro(1234)

    x0_min, x0_max = -1.0, 1.0

    tsave = collect(range(0, tmax; step=dt_save))

    # Define ensemble model
    model_ensemble = TwoSpinModelRRG_Ferro(N, K, J, D)

    # Sample ensemble
    _, traj_alls, _ = sample_ensemble_2Spin(model_ensemble, x0_min, x0_max, tmax, tsave, nsim; rng=rng, showprogress=true, dt=dt)

    # Compute correlation function
    autocorr_traj, tidxs = compute_autocorr(traj_alls; showprogress=true)
    tvec = tsave[tidxs]
    
    # Save results
    isdir("data") || mkdir("data")
    isdir("data/RRG") || mkdir("data/RRG")

    filename = "data/RRG/Corr_Ferro_N-$(N)_K-$(K)_J-$(J)_D-$(D)_tmax-$(tmax)_dtMC-$(dt)_dtsave-$(dt_save)_nsim-$(nsim)_x0min-$(x0_min)_x0max-$(x0_max).jld2"
    jldsave(filename; autocorr_traj, tsave, tidxs)
end

function run_GECaM_Bim_RRG(K, J, D, dt, T)
    # Create tvec
    tvec_G = [dt * n for n in 0:T]
    tmax = T * dt
    
    # Compute GECaM
    C_G, R_G, Ch_G, Rh_G, mu_G = integrate_2spin_Bim_RRG(K, J, D, dt, T; showprogress=true)
    
    # Save results
    filename = "data/RRG/Corr_GECaM_K-$(K)_J-$(J)_D-$(D)_tmax-$(tmax)_dt-$(dt).jld2"
    jldsave(filename; tvec_G, C_G, R_G, Ch_G, Rh_G, mu_G)
end


N_dt_s = [(1000, 1e-1), (1000, 1e-2)]
K = 4
J, D = 1.0, 0.3
tmax = 2e2
nsim = 500
dt_save = 1e-1
tws = [1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2]
nts = 500

for (N, dt) in N_dt_s
    println("Running Bim RRG with N = $N, K = $K, dt=$dt")
    run_Bim_RRG(N, K, J, D, tmax, nsim, dt, dt_save, tws, nts)
end

#=
N = 1000
Ks = [500]
J, D = 1.0, 0.3
tmaxs = [2e2]
nsim = 1000
dt = 1e-2
dt_saves = [1e-1]

for (K, tmax, dt_save) in Iterators.zip(Ks, tmaxs, dt_saves)
    println("Running Bim RRG with N=$N, K = $K")
    run_Bim_RRG(N, K, J, D, tmax, nsim, dt, dt_save)
end


N = 500
K = N - 1
J, D = 1.0, 0.3
tmax = 2e0
nsim = 500
dt = 1e-2
dt_save = 1e-2

println("Running Bim FC with N=$N")
run_Bim_RRG(N, K, J, D, tmax, nsim, dt, dt_save)


N, K = 1000, 5
J, D = 1.0, 0.3
tmax = 2e1
nsim = 10000
dt = 1e-2
dt_save = 1e-1
ilist = [1, 2, 1000]

run_Bin_RRG_single_nodes(N, K, J, D, tmax, nsim, dt, dt_save, ilist)


N = 1000
Ks = [10]
J, D = 1.0, 0.3
tmaxs = [2e1]
nsim = 500
dt = 1e-2
dt_saves = [1e-1]

for (K, tmax, dt_save) in Iterators.zip(Ks, tmaxs, dt_saves)
    println("Running Ferro RRG with N=$N, K = $K")
    run_Ferro_RRG(N, K, J, D, tmax, nsim, dt, dt_save)
end

N = 500
K = N - 1
J, D = 1.0, 0.3
tmax = 2e1
nsim = 500
dt = 1e-2
dt_save = 1e-2

println("Running Ferro FC with N=$N")
run_Ferro_RRG(N, K, J, D, tmax, nsim, dt, dt_save)
=#


Ks = [4]
J, D = 1.0, 0.3
tmax = 2e2
dt = 2e-2
T = ceil(Int, tmax / dt)

for K in Ks
    println("Running GECaM for K=$K")
    run_GECaM_Bim_RRG(K, J, D, dt, T)
end
