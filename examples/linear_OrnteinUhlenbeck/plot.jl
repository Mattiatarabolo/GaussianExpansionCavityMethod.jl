using JLD2, Random, Graphs, GaussianExpansionCavityMethod, SparseArrays, LinearAlgebra, Statistics, ProgressMeter, QuadGK
ProgressMeter.ijulia_behavior(:append)
using PyCall, Conda
import PyPlot as plt
using LaTeXStrings

rcParams = PyDict(plt.matplotlib["rcParams"])
rcParams["font.size"] = 8

fig, axs = plt.subplots(2, 3, figsize=(7, 5), tight_layout=true, sharex=true, sharey=true)

# RRG Ferro
ax = axs[1, 1]
# Parameters
N, K = 200, 3
lam, J, D = 1.2, 1.0/K, 1.0
tmax = 300.0
nsim = 500
teq = 200.0
# Load data
jldfile = "data/RRG_Ferro/plot/N-$(N)_K-$(K)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq-$(teq).jld2"
autocorr_traj, err_autocorr_traj, lagvec_mc, tvec_G, C, Ceq_analytic = load(jldfile, "autocorr_traj", "err_autocorr_traj", "lagvec_mc", "tvec_G", "C", "Ceq_analytic")
# Plot correlations
ax.plot(tvec_G, C, color="blue", label="GECaM")
ax.scatter(lagvec_mc[1:3:end], autocorr_traj[1:3:end], color="black", marker="d", s=20, label="MC")
ax.fill_between(lagvec_mc, autocorr_traj .- err_autocorr_traj, autocorr_traj .+ err_autocorr_traj, color="black", alpha=0.3)
ax.plot(tvec_G, Ceq_analytic ./ Ceq_analytic[1], color="red", label="Analytic")
ax.legend()
ax.set_ylabel(L"C^{\rm eq}(\tau)")
ax.set_title("RRG ferro")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylim(1e-3, 1.5)
ax.set_xlim(1e-2, 3e1)
ax.grid(alpha=.5)

# ER ferro
ax = axs[1, 2]
# Parameters
N, K = 200, 4
lam, J, D = 10.0, 1.0/K, 1.0
tmax = 500.0
nsim = 100
teq = 400.0
# Load data
jldfile = "data/ER_Ferro/plot/N-$(N)_K-$(K)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq-$(teq).jld2"
autocorr_traj, err_autocorr_traj, lagvec_mc, tvec_G, C = load(jldfile, "autocorr_traj", "err_autocorr_traj", "lagvec_mc", "tvec_G", "C")
# Plot correlations
ax.plot(tvec_G, C, color="blue", label="GECaM")
ax.scatter(lagvec_mc[1:3:end], autocorr_traj[1:3:end], color="black", marker="d", s=20, label="MC")
ax.fill_between(lagvec_mc, autocorr_traj .- err_autocorr_traj, autocorr_traj .+ err_autocorr_traj, color="black", alpha=0.3)
ax.legend()
ax.set_title("ER ferro")
ax.set_yscale("log")
#ax.set_xscale("log")
ax.grid(alpha=.5)


# RCM ferro
ax = axs[1, 3]
# Parameters
N, Ktarget = 500, 7
alpha = 2.5
lam, J, D = 3.0, 1.0/Ktarget, 1.0
tmax = 500.0
nsim = 1000
teq1, teq2 = 470, 200
# Load data
jldfile = "data/RCM_Ferro/N-$(N)_K-$(Float64(Ktarget))_lam-$(lam)_D-$(D)_J-$(Int(J*Ktarget))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).jld2"
autocorr_traj, err_autocorr_traj, lagvec_mc, tvec_G, C = load(jldfile, "autocorr_traj", "err_autocorr_traj", "lagvec_mc", "tvec_G", "C")
# Plot correlations
ax.plot(tvec_G, C, color="blue", label="GECaM")
ax.scatter(lagvec_mc[1:3:end], autocorr_traj[1:3:end], color="black", marker="d", s=20, label="MC")
ax.fill_between(lagvec_mc, autocorr_traj .- err_autocorr_traj, autocorr_traj .+ err_autocorr_traj, color="black", alpha=0.3)
ax.legend()
ax.set_title("RCM ferro")
ax.set_yscale("log")
#ax.set_xscale("log")
ax.grid(alpha=.5)


# RRG disordered
ax = axs[2, 1]
# Parameters
N, K = 200, 3
lam, J, D = 1.2, 1.0/K, 1.0
tmax = 300.0
nsim = 500
teq = 200.0
# Load data
jldfile = "data/RRG_Dis/plot/N-$(N)_K-$(K)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq-$(teq).jld2"
autocorr_traj, err_autocorr_traj, lagvec_mc, tvec_G, C = load(jldfile, "autocorr_traj", "err_autocorr_traj", "lagvec_mc", "tvec_G", "C")
# Plot correlations
ax.plot(tvec_G, C, color="blue", label="GECaM")
ax.scatter(lagvec_mc[1:3:end], autocorr_traj[1:3:end], color="black", marker="d", s=20, label="MC")
ax.fill_between(lagvec_mc, autocorr_traj .- err_autocorr_traj, autocorr_traj .+ err_autocorr_traj, color="black", alpha=0.3)
ax.legend()
ax.set_ylabel(L"C^{\rm eq}(\tau)")
ax.set_xlabel(L"\tau")
ax.set_title("RRG disordered")
ax.set_yscale("log")
#ax.set_xscale("log")
ax.grid(alpha=.5)

# ER disordered
ax = axs[2, 2]
# Parameters
N, K = 100, 4
lam, J, D = 1.3, 1.0/K, 1.0
tmax = 300.0
nsim = 500
teq = 200.0
# Load data
jldfile = "data/ER_Dis/plot/N-$(N)_K-$(K)_lam-$(lam)_D-$(D)_J-$(Int(J*K))_tmax-$(tmax)_nsim-$(nsim)_teq-$(teq).jld2"
autocorr_traj, err_autocorr_traj, lagvec_mc, tvec_G, C = load(jldfile, "autocorr_traj", "err_autocorr_traj", "lagvec_mc", "tvec_G", "C")
# Plot correlations
ax.plot(tvec_G, C, color="blue", label="GECaM")
ax.scatter(lagvec_mc[1:3:end], autocorr_traj[1:3:end], color="black", marker="d", s=20, label="MC")
ax.fill_between(lagvec_mc, autocorr_traj .- err_autocorr_traj, autocorr_traj .+ err_autocorr_traj, color="black", alpha=0.3)
ax.legend()
ax.set_title("ER disordered")
ax.set_xlabel(L"\tau")
ax.set_yscale("log")
#ax.set_xscale("log")
ax.grid(alpha=.5)

# RCM Ferro, degrees
ax = axs[2, 3]
# Parameters
N, Ktarget = 500, 7
alpha = 2.5
lam, J, D = 3.0, 1.0/Ktarget, 1.0
tmax = 500.0
nsim = 1000
teq1, teq2 = 470, 200
# Load data 1
jldfile = "data/RCM_Ferro/N-$(N)_K-$(Ktarget).0_lam-$(lam)_D-$(D)_J-$(Int(J*Ktarget))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).jld2"
tvec_G, traj_alls, lagidxs_mc = load(jldfile, "tvec_G", "traj_alls", "lagidxs_mc")
dt_mc = 1e-4
tsave_mc = collect(range(teq2, tmax; step=1e-2))
T_mc = length(tsave_mc)
# Load data 2
jldfile = "data/RCM_Ferro/GECaM_nodes_N-$(N)_K-$(Float64(Ktarget))_lam-$(lam)_D-$(D)_J-$(Int(J*Ktarget))_tmax-$(tmax)_nsim-$(nsim)_teq1-$(teq1)_teq2-$(teq2).jld2"
nodes, degs = load(jldfile, "nodes", "degs")
# Create vector of degrees to plot
degmin = max(1, minimum(degs))
degmax = maximum(degs)
degs_plot = [degmin, Ktarget, degmax]
idxs_plot = [argmin(abs.(degs .- k)) for k in degs_plot]
mts = ["d", "o", "*"]
lts = ["-", "--", "-."]
colors = ["blue", "red", "black"]
for (idx_plot, k) in enumerate(degs_plot)
    idxk = idxs_plot[idx_plot]
    trajk = zeros(nsim, T_mc)
    @inbounds for (isim, traj) in enumerate(traj_alls)
        trajk[isim, :] .= traj[idxk, :]
    end
    autocorrk, err_autocorrk, l_idxk = compute_autocorr_TTI(trajk, 1; lag_indices=lagidxs_mc)
    lagvec_mck = tsave_mc[l_idxk] .- teq2
    # plot
    ax.plot(tvec_G, nodes[idxk].marg.C, color=colors[idx_plot], ls=lts[idx_plot], label="GECaM: k=$k")
    ax.scatter(lagvec_mck[1:3:end], autocorrk[1:3:end], marker=mts[idx_plot], color=colors[idx_plot], label="MC: k=$k")
    ax.fill_between(lagvec_mck, autocorrk .- err_autocorrk, autocorrk .+ err_autocorrk,  color=colors[idx_plot], alpha=0.1)
end
ax.legend()
ax.set_title("RCM ferro")
ax.set_xlabel(L"\tau")
ax.set_yscale("log")
#ax.set_xscale("log")
ax.grid(alpha=.5)


fig.savefig("Ceq_all.png", dpi=300, bbox_inches="tight")
fig.savefig("Ceq_all.pdf", bbox_inches="tight")