using Random, Graphs, GaussianExpansionCavityMethod, SparseArrays, LinearAlgebra, Statistics, ProgressMeter, QuadGK, JLD2
using PowerLawSamplers: doubling_up_sampler
using PyCall, Conda
mpl_toolkits = pyimport("mpl_toolkits.axes_grid1")
Line2D = pyimport("matplotlib.lines").Line2D
import PyPlot as plt
using LaTeXStrings

rcParams = PyDict(plt.matplotlib["rcParams"])
rcParams["font.size"] = 8


function run_RCM_Ferro(N, Ktarget, alpha, lam, J, D, tmax, nsim, teq1, teq2)
    rng = Xoshiro(1234)
    G = static_scale_free(N, Int(Ktarget*N/2), alpha; rng=rng, finite_size_correction=true)
    degs = degree(G)
    K = floor(Int, mean(degs))

    Jmat = adjacency_matrix(G, Float64) .* J
    lambdas = fill(lam, N)

    x0_min, x0_max = -1.0, 1.0
    gen_J(N, K, J_params; rng=Xoshiro(1234)) = deepcopy(Jmat)
    gen_x0(N, x0_params; rng=Xoshiro(1234)) = rand(rng, N) .* (x0_params[2] - x0_params[1]) .+ x0_params[1]
    J_params = [J]
    x0_params = [x0_min, x0_max]

    ensemble_model = OUModelEnsemble(N, K, gen_J, J_params, lambdas, D, gen_x0, x0_params)

    dt_mc = 1e-3
    tsave_mc = collect(range(teq2, tmax; step=1e-2))
    T_mc = length(tsave_mc)

    # Sample ensemble
    _, traj_alls = sample_ensemble_OU(ensemble_model, tmax, tsave_mc, nsim; rng=rng, showprogress=true, dt=dt_mc)

    model = OUModel(K, Jmat, lambdas, D)

    dt_G = 1e-2

    T_G = Int((tmax - teq1) / dt_G)
    nodes = run_cavity_EQ(model, dt_G, T_G; showprogress=true)

    C, _ = compute_averages(nodes, model, T_G)

    lagvec_mc = vcat(0.0, exp10.(collect(range(log10(3e-2), log10((3e1)); length=25))))
    lagidxs_mc = [argmin(abs.(tsave_mc .- teq2 .- t)) for t in lagvec_mc] 
    # Remvoe duplicates
    lagidxs_mc = unique(lagidxs_mc)

    tvec_G = collect(range(0, T_G*dt_G, length=T_G+1))

    autocorr_traj, err_autocorr_traj, l_idx = compute_autocorr_TTI(traj_alls, 1; lag_indices=lagidxs_mc)
    lagvec_mc = tsave_mc[l_idx] .- teq2

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=true)

    # Plot correlations
    ax.plot(tvec_G, C, color="blue", label="GECaM")
    ax.scatter(lagvec_mc, autocorr_traj, color="black", label="MC")
    ax.fill_between(lagvec_mc, autocorr_traj .- err_autocorr_traj, autocorr_traj .+ err_autocorr_traj, color="black", alpha=0.1)
    ax.legend(loc="upper right")
    ax.set_xlabel("τ")
    ax.set_ylabel("Ceq(τ)")
    ax.set_title("Equilibrium correlation")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlim(1e-2, 1e1)
    ax.grid(alpha=.5)

    degmax = maximum(degs)
    degmin = max(1, minimum(degs))
    degs_plot = [degmin, K, degmax]
    idxs_plot = [argmin(abs.(degs .- k)) for k in degs_plot]
    mts = ["d", "o", "*"]
    lts = ["-", "--", "-."]
    colors = ["C0", "C1", "C2"]

    lagvecs_mcdegs = [zeros(length(lagvec_mc)) for _ in 1:length(degs_plot)]
    Cs_degs = [zeros(length(C)) for _ in 1:length(degs_plot)]
    autocorrs_degs = [zeros(length(autocorr_traj)) for _ in 1:length(degs_plot)]
    err_autocorrs_degs = [zeros(length(err_autocorr_traj)) for _ in 1:length(degs_plot)]
    
    inset_ax = ax.inset_axes([0.02, 0.03, 0.6, 0.6]) 
    custom_handles = PyObject[]  # Will store combined handles
    custom_labels = String[]  # Will store single labels like "k = $k"

    for (idx_plot, k) in enumerate(degs_plot)
        idxk = idxs_plot[idx_plot]
        trajk = zeros(nsim, T_mc)
        @inbounds for (isim, traj) in enumerate(traj_alls)
            trajk[isim, :] .= traj[idxk, :]
        end
        autocorrk, err_autocorrk, l_idxk = compute_autocorr_TTI(trajk, 1; lag_indices=lagidxs_mc)
        lagvec_mck = tsave_mc[l_idxk] .- teq2
    
        # Plot both datasets
        inset_ax.plot(tvec_G, nodes[idxk].marg.C, color=colors[idx_plot], ls=lts[idx_plot])
        inset_ax.scatter(lagvec_mck[1:3:end], autocorrk[1:3:end], marker=mts[idx_plot], color=colors[idx_plot])
        inset_ax.fill_between(lagvec_mck, autocorrk .- err_autocorrk, autocorrk .+ err_autocorrk, color=colors[idx_plot], alpha=0.1)
    
        # Create combined handle
        handle = Line2D([], [], color=colors[idx_plot], marker=mts[idx_plot], linestyle=lts[idx_plot])
        push!(custom_handles, handle)
        push!(custom_labels, "k = $k")
    
        # Store results if needed
        lagvecs_mcdegs[idx_plot] .= deepcopy(lagvec_mck)
        Cs_degs[idx_plot] .= deepcopy(nodes[idxk].marg.C.parent)
        autocorrs_degs[idx_plot] .= deepcopy(autocorrk)
        err_autocorrs_degs[idx_plot] .= deepcopy(err_autocorrk)
    end
    
    # Use custom legend
    inset_ax.legend(custom_handles, custom_labels, loc="lower left", fontsize=8)
    
    # Set inset axis properties
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    inset_ax.tick_params(labelbottom=false, labelleft=false)
    inset_ax.set_yscale("log")
    inset_ax.set_xscale("log")
    inset_ax.set_ylim(1e-3, 1.5)
    inset_ax.set_xlim(1e-2, 3e1)
    inset_ax.grid(alpha=.5)

    fig.savefig("Ceq_RCM_Ferro_N-$(N)_K-$(K)_alpha-$(alpha)_lam-$(lam)_D-$(D)_J-$(round(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).png", dpi=300, bbox_inches="tight")

    # save data
    isdir("data") || mkdir("data")
    isdir("data/RCM_Ferro") || mkdir("data/RCM_Ferro")

    jldsave("data/RCM_Ferro/N-$(N)_K_alpha-$(alpha)_$(K)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).jld2"; degs, tsave_mc, T_mc, traj_alls, nodes, autocorr_traj, err_autocorr_traj, lagvec_mc, lagidxs_mc, tvec_G, C)

    # save plot data
    isdir("data/RCM_Ferro/plot") || mkdir("data/RCM_Ferro/plot")

    jldsave("data/RCM_Ferro/plot/N-$(N)_K-$(K)_alpha-$(alpha)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).jld2"; tvec_G, C, lagvec_mc, autocorr_traj, err_autocorr_traj, lagvecs_mcdegs, Cs_degs, autocorrs_degs, err_autocorrs_degs, degs_plot)
end
#=
N, Ktarget = 200, 7
alpha = 2.5
lam, J, D = 3.0, 1.0/Ktarget, 1.0
tmax = 500.0
nsim = 1000
teq1, teq2 = 470, 150

run_RCM_Ferro(N, Ktarget, alpha, lam, J, D, tmax, nsim, teq1, teq2)
=#

function run_ER_Ferro(N, K, lam, J, D, tmax, nsim, teq1, teq2)
    rng = Xoshiro(1234)
    G = erdos_renyi(N, K/N; rng=rng)
    degs = degree(G)

    Jmat = adjacency_matrix(G, Float64) .* J
    lambdas = fill(lam, N)

    x0_min, x0_max = -1.0, 1.0
    gen_J(N, K, J_params; rng=Xoshiro(1234)) = deepcopy(Jmat)
    gen_x0(N, x0_params; rng=Xoshiro(1234)) = rand(rng, N) .* (x0_params[2] - x0_params[1]) .+ x0_params[1]
    J_params = [J]
    x0_params = [x0_min, x0_max]

    ensemble_model = OUModelEnsemble(N, K, gen_J, J_params, lambdas, D, gen_x0, x0_params)

    dt_mc = 1e-3
    tsave_mc = collect(range(teq2, tmax; step=1e-2))
    T_mc = length(tsave_mc)

    # Sample ensemble
    _, traj_alls = sample_ensemble_OU(ensemble_model, tmax, tsave_mc, nsim; rng=rng, showprogress=true, dt=dt_mc)

    model = OUModel(K, Jmat, lambdas, D)

    dt_G = 1e-2

    T_G = Int((tmax - teq1) / dt_G)
    nodes = run_cavity_EQ(model, dt_G, T_G; showprogress=true)

    C, _ = compute_averages(nodes, model, T_G)

    lagvec_mc = vcat(0.0, exp10.(collect(range(log10(3e-2), log10((3e1)); length=25))))
    lagidxs_mc = [argmin(abs.(tsave_mc .- teq2 .- t)) for t in lagvec_mc] 
    # Remvoe duplicates
    lagidxs_mc = unique(lagidxs_mc)

    tvec_G = collect(range(0, T_G*dt_G, length=T_G+1))

    autocorr_traj, err_autocorr_traj, l_idx = compute_autocorr_TTI(traj_alls, 1; lag_indices=lagidxs_mc)
    lagvec_mc = tsave_mc[l_idx] .- teq2

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=true)

    # Plot correlations
    ax.plot(tvec_G, C, color="blue", label="GECaM")
    ax.scatter(lagvec_mc, autocorr_traj, color="black", label="MC")
    ax.fill_between(lagvec_mc, autocorr_traj .- err_autocorr_traj, autocorr_traj .+ err_autocorr_traj, color="black", alpha=0.1)
    ax.legend(loc="upper right")
    ax.set_xlabel("τ")
    ax.set_ylabel("Ceq(τ)")
    ax.set_title("Equilibrium correlation")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlim(1e-2, 1e1)
    ax.grid(alpha=.5)

    degmax = maximum(degs)
    degmin = max(1, minimum(degs))
    degs_plot = [degmin, K, degmax]
    idxs_plot = [argmin(abs.(degs .- k)) for k in degs_plot]
    mts = ["d", "o", "*"]
    lts = ["-", "--", "-."]
    colors = ["C0", "C1", "C2"]

    lagvecs_mcdegs = [zeros(length(lagvec_mc)) for _ in 1:length(degs_plot)]
    Cs_degs = [zeros(length(C)) for _ in 1:length(degs_plot)]
    autocorrs_degs = [zeros(length(autocorr_traj)) for _ in 1:length(degs_plot)]
    err_autocorrs_degs = [zeros(length(err_autocorr_traj)) for _ in 1:length(degs_plot)]
    
    inset_ax = ax.inset_axes([0.02, 0.03, 0.6, 0.6]) 
    custom_handles = PyObject[]  # Will store combined handles
    custom_labels = String[]  # Will store single labels like "k = $k"

    for (idx_plot, k) in enumerate(degs_plot)
        idxk = idxs_plot[idx_plot]
        trajk = zeros(nsim, T_mc)
        @inbounds for (isim, traj) in enumerate(traj_alls)
            trajk[isim, :] .= traj[idxk, :]
        end
        autocorrk, err_autocorrk, l_idxk = compute_autocorr_TTI(trajk, 1; lag_indices=lagidxs_mc)
        lagvec_mck = tsave_mc[l_idxk] .- teq2
    
        # Plot both datasets
        inset_ax.plot(tvec_G, nodes[idxk].marg.C, color=colors[idx_plot], ls=lts[idx_plot])
        inset_ax.scatter(lagvec_mck[1:3:end], autocorrk[1:3:end], marker=mts[idx_plot], color=colors[idx_plot])
        inset_ax.fill_between(lagvec_mck, autocorrk .- err_autocorrk, autocorrk .+ err_autocorrk, color=colors[idx_plot], alpha=0.1)
    
        # Create combined handle
        handle = Line2D([], [], color=colors[idx_plot], marker=mts[idx_plot], linestyle=lts[idx_plot])
        push!(custom_handles, handle)
        push!(custom_labels, "k = $k")
    
        # Store results if needed
        lagvecs_mcdegs[idx_plot] .= deepcopy(lagvec_mck)
        Cs_degs[idx_plot] .= deepcopy(nodes[idxk].marg.C.parent)
        autocorrs_degs[idx_plot] .= deepcopy(autocorrk)
        err_autocorrs_degs[idx_plot] .= deepcopy(err_autocorrk)
    end
    
    # Use custom legend
    inset_ax.legend(custom_handles, custom_labels, loc="lower left", fontsize=8)
    
    # Set inset axis properties
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    inset_ax.tick_params(labelbottom=false, labelleft=false)
    inset_ax.set_yscale("log")
    inset_ax.set_xscale("log")
    inset_ax.set_ylim(1e-3, 1.5)
    inset_ax.set_xlim(1e-2, 3e1)
    inset_ax.grid(alpha=.5)

    fig.savefig("Ceq_ER_Ferro_N-$(N)_K-$(K)_lam-$(lam)_D-$(D)_J-$(round(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).png", dpi=300, bbox_inches="tight")

    # save data
    isdir("data") || mkdir("data")
    isdir("data/ER_Ferro") || mkdir("data/ER_Ferro")

    jldsave("data/ER_Ferro/N-$(N)_K-$(K)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).jld2"; degs, tsave_mc, T_mc, traj_alls, nodes, autocorr_traj, err_autocorr_traj, lagvec_mc, lagidxs_mc, tvec_G, C)

    # save plot data
    isdir("data/ER_Ferro/plot") || mkdir("data/ER_Ferro/plot")

    jldsave("data/ER_Ferro/plot/N-$(N)_K-$(K)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).jld2"; tvec_G, C, lagvec_mc, autocorr_traj, err_autocorr_traj, lagvecs_mcdegs, Cs_degs, autocorrs_degs, err_autocorrs_degs, degs_plot)
end

#=
N, K = 100, 4
lam, J, D = 1.7, 1.0/K, 1.0
tmax = 500.0
nsim = 500
teq1, teq2 = 470, 150

run_ER_Ferro(N, K, lam, J, D, tmax, nsim, teq1, teq2)
=#




function run_RCM_Bim(N, Ktarget, alpha, lam, J, D, tmax, nsim, teq1, teq2)
    rng = Xoshiro(1234)
    G = static_scale_free(N, Int(Ktarget*N/2), alpha; rng=rng, finite_size_correction=true)
    degs = degree(G)
    K = floor(Int, mean(degs))

    Jmat = adjacency_matrix(G, Float64)
    @inbounds @fastmath for i in 1:N
        @inbounds @fastmath @simd for j in i+1:N
            if Jmat[i, j] != 0
                Jmat[i, j] = rand(rng) > 0.5 ? J : -J
                Jmat[j, i] = Jmat[i, j]
            end
        end
    end
    lambdas = fill(lam, N)

    x0_min, x0_max = -1.0, 1.0
    gen_J(N, K, J_params; rng=Xoshiro(1234)) = deepcopy(Jmat)
    gen_x0(N, x0_params; rng=Xoshiro(1234)) = rand(rng, N) .* (x0_params[2] - x0_params[1]) .+ x0_params[1]
    J_params = [J]
    x0_params = [x0_min, x0_max]

    ensemble_model = OUModelEnsemble(N, K, gen_J, J_params, lambdas, D, gen_x0, x0_params)

    dt_mc = 1e-3
    tsave_mc = collect(range(teq2, tmax; step=1e-2))
    T_mc = length(tsave_mc)

    # Sample ensemble
    _, traj_alls = sample_ensemble_OU(ensemble_model, tmax, tsave_mc, nsim; rng=rng, showprogress=true, dt=dt_mc)

    model = OUModel(K, Jmat, lambdas, D)

    dt_G = 1e-2

    T_G = Int((tmax - teq1) / dt_G)
    nodes = run_cavity_EQ(model, dt_G, T_G; showprogress=true)

    C, _ = compute_averages(nodes, model, T_G)

    lagvec_mc = vcat(0.0, exp10.(collect(range(log10(3e-2), log10((3e1)); length=25))))
    lagidxs_mc = [argmin(abs.(tsave_mc .- teq2 .- t)) for t in lagvec_mc] 
    # Remvoe duplicates
    lagidxs_mc = unique(lagidxs_mc)

    tvec_G = collect(range(0, T_G*dt_G, length=T_G+1))

    autocorr_traj, err_autocorr_traj, l_idx = compute_autocorr_TTI(traj_alls, 1; lag_indices=lagidxs_mc)
    lagvec_mc = tsave_mc[l_idx] .- teq2

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=true)

    # Plot correlations
    ax.plot(tvec_G, C, color="blue", label="GECaM")
    ax.scatter(lagvec_mc, autocorr_traj, color="black", label="MC")
    ax.fill_between(lagvec_mc, autocorr_traj .- err_autocorr_traj, autocorr_traj .+ err_autocorr_traj, color="black", alpha=0.1)
    ax.legend(loc="upper right")
    ax.set_xlabel("τ")
    ax.set_ylabel("Ceq(τ)")
    ax.set_title("Equilibrium correlation")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlim(1e-2, 1e1)
    ax.grid(alpha=.5)

    degmax = maximum(degs)
    degmin = max(1, minimum(degs))
    degs_plot = [degmin, K, degmax]
    idxs_plot = [argmin(abs.(degs .- k)) for k in degs_plot]
    mts = ["d", "o", "*"]
    lts = ["-", "--", "-."]
    colors = ["C0", "C1", "C2"]

    lagvecs_mcdegs = [zeros(length(lagvec_mc)) for _ in 1:length(degs_plot)]
    Cs_degs = [zeros(length(C)) for _ in 1:length(degs_plot)]
    autocorrs_degs = [zeros(length(autocorr_traj)) for _ in 1:length(degs_plot)]
    err_autocorrs_degs = [zeros(length(err_autocorr_traj)) for _ in 1:length(degs_plot)]
    
    inset_ax = ax.inset_axes([0.02, 0.03, 0.6, 0.6]) 
    custom_handles = PyObject[]  # Will store combined handles
    custom_labels = String[]  # Will store single labels like "k = $k"

    for (idx_plot, k) in enumerate(degs_plot)
        idxk = idxs_plot[idx_plot]
        trajk = zeros(nsim, T_mc)
        @inbounds for (isim, traj) in enumerate(traj_alls)
            trajk[isim, :] .= traj[idxk, :]
        end
        autocorrk, err_autocorrk, l_idxk = compute_autocorr_TTI(trajk, 1; lag_indices=lagidxs_mc)
        lagvec_mck = tsave_mc[l_idxk] .- teq2
    
        # Plot both datasets
        inset_ax.plot(tvec_G, nodes[idxk].marg.C, color=colors[idx_plot], ls=lts[idx_plot])
        inset_ax.scatter(lagvec_mck[1:3:end], autocorrk[1:3:end], marker=mts[idx_plot], color=colors[idx_plot])
        inset_ax.fill_between(lagvec_mck, autocorrk .- err_autocorrk, autocorrk .+ err_autocorrk, color=colors[idx_plot], alpha=0.1)
    
        # Create combined handle
        handle = Line2D([], [], color=colors[idx_plot], marker=mts[idx_plot], linestyle=lts[idx_plot])
        push!(custom_handles, handle)
        push!(custom_labels, "k = $k")
    
        # Store results if needed
        lagvecs_mcdegs[idx_plot] .= deepcopy(lagvec_mck)
        Cs_degs[idx_plot] .= deepcopy(nodes[idxk].marg.C.parent)
        autocorrs_degs[idx_plot] .= deepcopy(autocorrk)
        err_autocorrs_degs[idx_plot] .= deepcopy(err_autocorrk)
    end
    
    # Use custom legend
    inset_ax.legend(custom_handles, custom_labels, loc="lower left", fontsize=8)
    
    # Set inset axis properties
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    inset_ax.tick_params(labelbottom=false, labelleft=false)
    inset_ax.set_yscale("log")
    inset_ax.set_xscale("log")
    inset_ax.set_ylim(1e-3, 1.5)
    inset_ax.set_xlim(1e-2, 3e1)
    inset_ax.grid(alpha=.5)

    fig.savefig("Ceq_RCM_Dis_Bim_N-$(N)_K-$(K)_alpha-$(alpha)_lam-$(lam)_D-$(D)_J-$(round(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).png", dpi=300, bbox_inches="tight")

    # save data
    isdir("data") || mkdir("data")
    isdir("data/RCM_Dis_Bim") || mkdir("data/RCM_Dis_Bim")

    jldsave("data/RCM_Dis_Bim/N-$(N)_K_alpha-$(alpha)_$(K)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).jld2"; degs, tsave_mc, T_mc, traj_alls, nodes, autocorr_traj, err_autocorr_traj, lagvec_mc, lagidxs_mc, tvec_G, C)

    # save plot data
    isdir("data/RCM_Dis_Bim/plot") || mkdir("data/RCM_Dis_Bim/plot")

    jldsave("data/RCM_Dis_Bim/plot/N-$(N)_K-$(K)_alpha-$(alpha)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).jld2"; tvec_G, C, lagvec_mc, autocorr_traj, err_autocorr_traj, lagvecs_mcdegs, Cs_degs, autocorrs_degs, err_autocorrs_degs, degs_plot)
end
#=
N, Ktarget = 100, 7
alpha = 2.5
lam, J, D = 3.0, 1.0/Ktarget, 1.0
tmax = 500.0
nsim = 1000
teq1, teq2 = 470, 150

run_RCM_Bim(N, Ktarget, alpha, lam, J, D, tmax, nsim, teq1, teq2)
=#

function run_ER_Bim(N, K, lam, J, D, tmax, nsim, teq1, teq2)
    rng = Xoshiro(1234)
    G = erdos_renyi(N, K/N; rng=rng)
    degs = degree(G)

    Jmat = adjacency_matrix(G, Float64)
    @inbounds @fastmath for i in 1:N
        @inbounds @fastmath @simd for j in i+1:N
            if Jmat[i, j] != 0
                Jmat[i, j] = rand(rng) > 0.5 ? J : -J
                Jmat[j, i] = Jmat[i, j]
            end
        end
    end
    lambdas = fill(lam, N)

    x0_min, x0_max = -1.0, 1.0
    gen_J(N, K, J_params; rng=Xoshiro(1234)) = deepcopy(Jmat)
    gen_x0(N, x0_params; rng=Xoshiro(1234)) = rand(rng, N) .* (x0_params[2] - x0_params[1]) .+ x0_params[1]
    J_params = [J]
    x0_params = [x0_min, x0_max]

    ensemble_model = OUModelEnsemble(N, K, gen_J, J_params, lambdas, D, gen_x0, x0_params)

    dt_mc = 1e-3
    tsave_mc = collect(range(teq2, tmax; step=1e-2))
    T_mc = length(tsave_mc)

    # Sample ensemble
    _, traj_alls = sample_ensemble_OU(ensemble_model, tmax, tsave_mc, nsim; rng=rng, showprogress=true, dt=dt_mc)

    model = OUModel(K, Jmat, lambdas, D)

    dt_G = 1e-2

    T_G = Int((tmax - teq1) / dt_G)
    nodes = run_cavity_EQ(model, dt_G, T_G; showprogress=true)

    C, _ = compute_averages(nodes, model, T_G)

    lagvec_mc = vcat(0.0, exp10.(collect(range(log10(3e-2), log10((3e1)); length=25))))
    lagidxs_mc = [argmin(abs.(tsave_mc .- teq2 .- t)) for t in lagvec_mc] 
    # Remvoe duplicates
    lagidxs_mc = unique(lagidxs_mc)

    tvec_G = collect(range(0, T_G*dt_G, length=T_G+1))

    autocorr_traj, err_autocorr_traj, l_idx = compute_autocorr_TTI(traj_alls, 1; lag_indices=lagidxs_mc)
    lagvec_mc = tsave_mc[l_idx] .- teq2

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=true)

    # Plot correlations
    ax.plot(tvec_G, C, color="blue", label="GECaM")
    ax.scatter(lagvec_mc, autocorr_traj, color="black", label="MC")
    ax.fill_between(lagvec_mc, autocorr_traj .- err_autocorr_traj, autocorr_traj .+ err_autocorr_traj, color="black", alpha=0.1)
    ax.legend(loc="upper right")
    ax.set_xlabel("τ")
    ax.set_ylabel("Ceq(τ)")
    ax.set_title("Equilibrium correlation")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlim(1e-2, 1e1)
    ax.grid(alpha=.5)

    degmax = maximum(degs)
    degmin = max(1, minimum(degs))
    degs_plot = [degmin, K, degmax]
    idxs_plot = [argmin(abs.(degs .- k)) for k in degs_plot]
    mts = ["d", "o", "*"]
    lts = ["-", "--", "-."]
    colors = ["C0", "C1", "C2"]

    lagvecs_mcdegs = [zeros(length(lagvec_mc)) for _ in 1:length(degs_plot)]
    Cs_degs = [zeros(length(C)) for _ in 1:length(degs_plot)]
    autocorrs_degs = [zeros(length(autocorr_traj)) for _ in 1:length(degs_plot)]
    err_autocorrs_degs = [zeros(length(err_autocorr_traj)) for _ in 1:length(degs_plot)]
    
    inset_ax = ax.inset_axes([0.02, 0.03, 0.6, 0.6]) 
    custom_handles = PyObject[]  # Will store combined handles
    custom_labels = String[]  # Will store single labels like "k = $k"

    for (idx_plot, k) in enumerate(degs_plot)
        idxk = idxs_plot[idx_plot]
        trajk = zeros(nsim, T_mc)
        @inbounds for (isim, traj) in enumerate(traj_alls)
            trajk[isim, :] .= traj[idxk, :]
        end
        autocorrk, err_autocorrk, l_idxk = compute_autocorr_TTI(trajk, 1; lag_indices=lagidxs_mc)
        lagvec_mck = tsave_mc[l_idxk] .- teq2
    
        # Plot both datasets
        inset_ax.plot(tvec_G, nodes[idxk].marg.C, color=colors[idx_plot], ls=lts[idx_plot])
        inset_ax.scatter(lagvec_mck[1:3:end], autocorrk[1:3:end], marker=mts[idx_plot], color=colors[idx_plot])
        inset_ax.fill_between(lagvec_mck, autocorrk .- err_autocorrk, autocorrk .+ err_autocorrk, color=colors[idx_plot], alpha=0.1)
    
        # Create combined handle
        handle = Line2D([], [], color=colors[idx_plot], marker=mts[idx_plot], linestyle=lts[idx_plot])
        push!(custom_handles, handle)
        push!(custom_labels, "k = $k")
    
        # Store results if needed
        lagvecs_mcdegs[idx_plot] .= deepcopy(lagvec_mck)
        Cs_degs[idx_plot] .= deepcopy(nodes[idxk].marg.C.parent)
        autocorrs_degs[idx_plot] .= deepcopy(autocorrk)
        err_autocorrs_degs[idx_plot] .= deepcopy(err_autocorrk)
    end
    
    # Use custom legend
    inset_ax.legend(custom_handles, custom_labels, loc="lower left", fontsize=8)
    
    # Set inset axis properties
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    inset_ax.tick_params(labelbottom=false, labelleft=false)
    inset_ax.set_yscale("log")
    inset_ax.set_xscale("log")
    inset_ax.set_ylim(1e-3, 1.5)
    inset_ax.set_xlim(1e-2, 3e1)
    inset_ax.grid(alpha=.5)

    fig.savefig("Ceq_ER_Dis_Bim_N-$(N)_K-$(K)_lam-$(lam)_D-$(D)_J-$(round(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).png", dpi=300, bbox_inches="tight")

    # save data
    isdir("data") || mkdir("data")
    isdir("data/ER_Dis_Bim") || mkdir("data/ER_Dis_Bim")

    jldsave("data/ER_Dis_Bim/N-$(N)_K-$(K)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).jld2"; degs, tsave_mc, T_mc, traj_alls, nodes, autocorr_traj, err_autocorr_traj, lagvec_mc, lagidxs_mc, tvec_G, C)

    # save plot data
    isdir("data/ER_Dis_Bim/plot") || mkdir("data/ER_Dis_Bim/plot")

    jldsave("data/ER_Dis_Bim/plot/N-$(N)_K-$(K)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).jld2"; tvec_G, C, lagvec_mc, autocorr_traj, err_autocorr_traj, lagvecs_mcdegs, Cs_degs, autocorrs_degs, err_autocorrs_degs, degs_plot)
end

#=
N, K = 100, 4
lam, J, D = 1.7, 1.0/K, 1.0
tmax = 500.0
nsim = 500
teq1, teq2 = 470, 150

run_ER_Bim(N, K, lam, J, D, tmax, nsim, teq1, teq2)
=#


N, K = 10, 4
alpha = 2.5
lam, J, D = 1.7, 1.0/K, 1.0
tmax = 10.0
nsim = 2
teq1, teq2 = 5, 7

run_RCM_Ferro(N, K, alpha, lam, J, D, tmax, nsim, teq1, teq2)

N, K = 10, 4
lam, J, D = 1.7, 1.0/K, 1.0
tmax = 10.0
nsim = 2
teq1, teq2 = 5, 7

run_ER_Ferro(N, K, lam, J, D, tmax, nsim, teq1, teq2)

N, K = 10, 4
alpha = 2.5
lam, J, D = 1.7, 1.0/K, 1.0
tmax = 10.0
nsim = 2
teq1, teq2 = 5, 7

run_RCM_Bim(N, K, alpha, lam, J, D, tmax, nsim, teq1, teq2)

N, K = 10, 4
lam, J, D = 1.7, 1.0/K, 1.0
tmax = 10.0
nsim = 2
teq1, teq2 = 5, 7

run_ER_Bim(N, K, lam, J, D, tmax, nsim, teq1, teq2)