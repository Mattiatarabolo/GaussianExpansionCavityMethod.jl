using Random, Graphs, GaussianExpansionCavityMethod, SparseArrays, LinearAlgebra, Statistics, ProgressMeter, QuadGK, JLD2
using PowerLawSamplers: doubling_up_sampler
using PyCall, Conda
import PyPlot as plt
using LaTeXStrings

rcParams = PyDict(plt.matplotlib["rcParams"])
rcParams["font.size"] = 8

nx = pyimport("networkx")
function to_networkx(g::Graphs.AbstractGraph)
    g_nx = nx.Graph()
    for v in vertices(g)
        g_nx.add_node(Int(v))
    end
    for e in edges(g)
        g_nx.add_edge(Int(src(e)), Int(dst(e)))
    end
    return g_nx
end


function run_RCM_Ferro(N, Ktarget, alpha, lam, J, D, tmax, nsim, teq1, teq2)
    rng = Xoshiro(1234)
    G = static_scale_free(N, Int(Ktarget*N/2), alpha; rng=rng, finite_size_correction=true)
    degs = degree(G)
    K = mean(degs)

    Jmat = adjacency_matrix(G) .* J
    lambdas = fill(lam, N)

    x0_min, x0_max = 1.0, 1.0
    gen_J(N, K, J_params; rng=Xoshiro(1234)) = deepcopy(Jmat)
    gen_x0(N, x0_params; rng=Xoshiro(1234)) = rand(rng, N) .* (x0_params[2] - x0_params[1]) .+ x0_params[1]
    J_params = [J]
    x0_params = [x0_min, x0_max]

    ensemble_model = OUModelEnsemble(N, K, gen_J, J_params, lambdas, D, gen_x0, x0_params)

    dt_mc = 1e-2
    tsave_mc = collect(range(0.0, tmax; step=dt_mc))
    T_mc = length(tsave_mc)

    # Sample ensemble
    tvals_alls, traj_alls = sample_ensemble_OU(ensemble_model, tmax, tsave_mc, nsim; rng=rng, showprogress=true, dt=dt_mc)

    tvec_mc = collect(range(0.0, tmax; length=200))
    tidxs_mc = [argmin(abs.(tsave_mc .- t)) for t in tvec_mc]
    mean_traj, std_traj = compute_meanstd(traj_alls; time_indices=tidxs_mc)

    model = OUModel(K, Jmat, lambdas, D)

    dt_G = 1e-2

    T_G = Int((tmax - teq1) / dt_G)
    nodes = run_cavity_EQ(model, dt_G, T_G; showprogress=true)

    C, _ = compute_averages(nodes, model, T_G)

    teq_idx = argmin(abs.(tsave_mc .- teq2))

    lagvec_mc = vcat(0.0, exp10.(collect(range(log10(3e-2), log10((tmax-teq2)); length=100))))
    lagidxs_mc = [argmin(abs.(view(tsave_mc,teq_idx:T_mc) .- teq .- t)) for t in lagvec_mc] 
    # Remvoe duplicates
    lagidxs_mc = unique(lagidxs_mc)
    lagvec_mc = tsave_mc[lagidxs_mc]

    tvec_G = collect(range(0, T_G*dt_G, length=T_G+1))

    autocorr_traj, err_autocorr_traj, l_idx = compute_autocorr_TTI(traj_alls, teq_idx; lag_indices=lagidxs_mc)
    lagvec_mc = tsave_mc[l_idx]

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4), tight_layout=true)

    # Plot correlations
    ax1.plot(tvec_G, C, color="blue", label="GECaM")
    ax1.scatter(lagvec_mc, autocorr_traj, color="black", label="MC")
    ax1.fill_between(lagvec_mc, autocorr_traj .- err_autocorr_traj, autocorr_traj .+ err_autocorr_traj, color="black", alpha=0.1)
    ax1.legend()
    ax1.set_xlabel("τ")
    ax1.set_ylabel("Ceq(τ)")
    ax1.set_title("Equilibrium correlation")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_ylim(1e-4, 1.5)
    ax1.set_xlim(1e-2, 1e1)
    ax1.grid(alpha=.5)
  
    fig1.savefig("Ceq_RCM_Ferro_N-$(N)_K-$(K)_alpha-$(alpha)_lam-$(lam)_D-$(D)_J-$(round(J*K)).png", dpi=300, bbox_inches="tight")

    degmax = maximum(degs)
    degmin = minimum(degs)

    idxmax = findfirst(x -> x == degmax, degs)
    idxmin = findfirst(x -> x == degmin, degs)
    idxavg = findfirst(x -> abs(x - K) < 1e-3, degs)

    traj_max = zeros(nsim, T_mc)
    traj_min = zeros(nsim, T_mc)
    traj_avg = zeros(nsim, T_mc)
    @inbounds for (isim, traj) in enumerate(traj_alls)
        traj_max[isim, :] .= traj[idxmax, :]
        traj_min[isim, :] .= traj[idxmin, :]
        traj_avg[isim, :] .= traj[idxavg, :]
    end
    autocorr_max, err_autocorr_max, l_idx_max = compute_autocorr_TTI(traj_max, teq_idx; lag_indices=lagidxs_mc)
    lagvec_mc_max = tsave_mc[l_idx_max]
    autocorr_min, err_autocorr_min, l_idx_min = compute_autocorr_TTI(traj_min, teq_idx; lag_indices=lagidxs_mc)
    lagvec_mc_min = tsave_mc[l_idx_min]
    autocorr_avg, err_autocorr_avg, l_idx_avg = compute_autocorr_TTI(traj_avg, teq_idx; lag_indices=lagidxs_mc)
    lagvec_mc_avg = tsave_mc[l_idx_avg]

    fig2, axs2 = plt.subplots(1, 3, figsize=(6*2, 4), tight_layout=true, sharey=true)

    # Plot correlations
    ax = axs2[1]
    ax.plot(tvec_G, nodes[idxmax].marg.C, color="blue", label="GECaM")
    ax.scatter(lagvec_mc_max, autocorr_max, color="black", label="MC")
    ax.fill_between(lagvec_mc_max, autocorr_max .- err_autocorr_max, autocorr_max .+ err_autocorr_max, color="black", alpha=0.1)
    ax.legend()
    ax.set_xlabel("τ")
    ax.set_ylabel("Ceq(τ)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlim(1e-2, 1e1)
    ax.grid(alpha=.5)
    ax.set_title("Node $idxmax: degree $(degs[idxmax])")

    ax = axs2[2]
    ax.plot(tvec_G, nodes[idxmin].marg.C, color="blue", label="GECaM")
    ax.scatter(lagvec_mc_min, autocorr_min, color="black", label="MC")
    ax.fill_between(lagvec_mc_min, autocorr_min .- err_autocorr_min, autocorr_min .+ err_autocorr_min, color="black", alpha=0.1)
    ax.legend()
    ax.set_xlabel("τ")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e1)
    ax.grid(alpha=.5)
    ax.set_title("Node $idxmin: degree $(degs[idxmin])")  

    ax = axs2[3]
    ax.plot(tvec_G, nodes[idxavg].marg.C, color="blue", label="GECaM")
    ax.scatter(lagvec_mc_avg, autocorr_avg, color="black", label="MC")
    ax.fill_between(lagvec_mc_avg, autocorr_avg .- err_autocorr_avg, autocorr_avg .+ err_autocorr_avg, color="black", alpha=0.1)
    ax.legend()
    ax.set_xlabel("τ")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e1)
    ax.grid(alpha=.5)
    ax.set_title("Node $idxavg: degree $(degs[idxavg])")

    fig2.savefig("Ceq_degs_RCM_Ferro_N-$(N)_K-$(K)_alpha-$(alpha)_lam-$(lam)_D-$(D)_J-$(round(J*K)).png", dpi=300, bbox_inches="tight")

    # save data
    isdir("data") || mkdir("data")
    isdir("data/RCM_Ferro") || mkdir("data/RCM_Ferro")

    jldsave("data/RCM_Ferro/N-$(N)_K-$(K)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq-$(teq).jld2"; tvals_alls, traj_alls, tvec_mc, mean_traj, std_traj, autocorr_traj, err_autocorr_traj, lagvec_mc, lagidxs_mc, tvec_G, C)

end

