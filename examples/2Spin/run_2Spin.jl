using Random, GaussianExpansionCavityMethod, ProgressMeter, JLD2
using PyCall, Conda
mpl_toolkits = pyimport("mpl_toolkits.axes_grid1")
Line2D = pyimport("matplotlib.lines").Line2D
import PyPlot as plt
using LaTeXStrings

rcParams = PyDict(plt.matplotlib["rcParams"])
rcParams["font.size"] = 8

function run_Bim_RRG(N, K, J, D, tmax, nsim)
    rng = Xoshiro(1234)

    x0_min, x0_max = -3.0, 3.0
    dt = 1e-3

    tsave = collect(range(0, tmax; step=1e-1))

    # Define ensemble model
    model_ensemble = TwoSpinModelRRG_Bim(N, K, J, D)

    # Sample ensemble
    _, traj_alls, _ = sample_ensemble_2Spin(model_ensemble, x0_min, x0_max, tmax, tsave, nsim; rng=rng, showprogress=true, dt=dt)

    # Compute correlation function
    autocorr_traj, tidxs = compute_autocorr(traj_alls)
    tvec = tsave[tidxs]

    # Compute autocorrelation function for 2 nodes only
    traj_1 = zeros(nsim, length(tsave))
    traj_2 = zeros(nsim, length(tsave))
    @inbounds for i in 1:nsim
        traj_1[i, :] = view(traj_alls[i], 1, :)
        traj_2[i, :] = view(traj_alls[i], 2, :)
    end
    autocorr_traj_1, tidxs_1 = compute_autocorr(traj_1)
    autocorr_traj_2, tidxs_2 = compute_autocorr(traj_2)

    # Save results
    isdir("data") || mkdir("data")
    isdir("data/RRG") || mkdir("data/RRG")

    filename = "data/RRG/Corr_N$(N)_K$(K)_J$(J)_D$(D)_tmax$(tmax)_dtMC-$(dt)_dtsave-1e-1_nsim$(nsim)_x0min-$(x0_min)_x0max-$(x0_max).jld2"
    jldsave(filename; autocorr_traj, tsave, tidxs, autocorr_traj_1, tidxs_1, autocorr_traj_2, tidxs_2)

    # Plot correlations
    tws = collect(10 .^ range(log10(1.0), stop=log10(1000.0), length=10))[1:end-1]
    tw_idxs = [argmin(abs.(tvec .- tw)) for tw in tws]
    tws_real = tvec[tw_idxs]

    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=true)

    # Initialize color map
    my_cmap = plt.matplotlib[:cm][:hot]
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.matplotlib[:colors][:LogNorm](vmin=1, vmax=1000))

    step = 220/10
    for (i, (tw, tw_idx)) in enumerate(Iterators.zip(tws_real, tw_idxs))
        ax.plot(tvec[tw_idx:end] .- tw, autocorr_traj[tw_idx:end, tw_idx] ./ autocorr_traj[tw_idx, tw_idx], color=my_cmap(floor(Int, (i-1)*step)))
    end
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1e-1,1e2)
    ax.set_ylim(autocorr_traj[end,tw_idxs[1]], 1.001)
    ax.set_xlabel(L"\tau")
    ax.set_ylabel(L"C(t_w + \tau, t_w)")
    ax.grid(alpha=.5)
    clb = plt.colorbar(sm, ax=ax)
    clb.ax.set_title(L"t_w")

    # Save figure
    fig.savefig("Corr_N$(N)_K$(K)_J$(J)_D$(D)_tmax$(tmax)_dtMC-$(dt)_dtsave-1e-1_nsim$(nsim)_x0min-$(x0_min)_x0max-$(x0_max).pdf", bbox_inches="tight")
    fig.savefig("Corr_N$(N)_K$(K)_J$(J)_D$(D)_tmax$(tmax)_dtMC-$(dt)_dtsave-1e-1_nsim$(nsim)_x0min-$(x0_min)_x0max-$(x0_max).png", dpi=300, bbox_inches="tight")
    plt.close(fig)
end

N = 1000
Ks = [4, 10, 100, 999]
J, D = 1.0, 0.3
tmax = 6e2
nsim = 100
for K in Ks
    println("Running Bim RRG with K = $K")
    run_Bim_RRG(N, K, J, D, tmax, nsim)
end


function run_Ferro_RRG(N, K, J, D, tmax, nsim)
    rng = Xoshiro(1234)

    x0_min, x0_max = -3.0, 3.0
    dt = 1e-3

    tsave = collect(range(0, tmax; step=1e-1))

    # Define ensemble model
    model_ensemble = TwoSpinModelRRG_Ferro(N, K, J, D)

    # Sample ensemble
    _, traj_alls, _ = sample_ensemble_2Spin(model_ensemble, x0_min, x0_max, tmax, tsave, nsim; rng=rng, showprogress=true, dt=dt)

    # Compute correlation function
    autocorr_traj, tidxs = compute_autocorr(traj_alls)
    tvec = tsave[tidxs]

    # Save results
    isdir("data") || mkdir("data")
    isdir("data/RRG") || mkdir("data/RRG")

    filename = "data/RRG/Corr_N$(N)_K$(K)_J$(J)_D$(D)_tmax$(tmax)_dtMC-$(dt)_dtsave-1e-1_nsim$(nsim)_x0min-$(x0_min)_x0max-$(x0_max).jld2"
    jldsave(filename; autocorr_traj, tsave, tidxs)

    # Plot correlations
    tws = collect(10 .^ range(log10(1.0), stop=log10(1000.0), length=10))[1:end-1]
    tw_idxs = [argmin(abs.(tvec .- tw)) for tw in tws]
    tws_real = tvec[tw_idxs]

    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=true)

    # Initialize color map
    my_cmap = plt.matplotlib[:cm][:hot]
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.matplotlib[:colors][:LogNorm](vmin=1, vmax=1000))

    step = 220/10
    for (i, (tw, tw_idx)) in enumerate(Iterators.zip(tws_real, tw_idxs))
        ax.plot(tvec[tw_idx:end] .- tw, autocorr_traj[tw_idx:end, tw_idx] ./ autocorr_traj[tw_idx, tw_idx], color=my_cmap(floor(Int, (i-1)*step)))
    end
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1e-1,1e2)
    ax.set_ylim(autocorr_traj[end,tw_idxs[1]], 1.001)
    ax.set_xlabel(L"\tau")
    ax.set_ylabel(L"C(t_w + \tau, t_w)")
    ax.grid(alpha=.5)
    clb = plt.colorbar(sm, ax=ax)
    clb.ax.set_title(L"t_w")

    # Save figure
    fig.savefig("Corr_N$(N)_K$(K)_J$(J)_D$(D)_tmax$(tmax)_dtMC-$(dt)_dtsave-1e-1_nsim$(nsim)_x0min-$(x0_min)_x0max-$(x0_max).pdf", bbox_inches="tight")
    fig.savefig("Corr_N$(N)_K$(K)_J$(J)_D$(D)_tmax$(tmax)_dtMC-$(dt)_dtsave-1e-1_nsim$(nsim)_x0min-$(x0_min)_x0max-$(x0_max).png", dpi=300, bbox_inches="tight")
    plt.close(fig)
end

N = 1000
Ks = [4, 10, 100, 999]
J, D = 1.0, 0.3
tmax = 6e2
nsim = 100

for K in Ks
    println("Running Ferro RRG with K = $K")
    run_Ferro_RRG(N, K, J, D, tmax, nsim)
end