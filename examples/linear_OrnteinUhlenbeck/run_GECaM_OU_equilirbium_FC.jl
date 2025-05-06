using Random, Graphs, GaussianExpansionCavityMethod, SparseArrays, LinearAlgebra, Statistics, ProgressMeter, QuadGK, JLD2
using PowerLawSamplers: doubling_up_sampler
using PyCall, Conda
mpl_toolkits = pyimport("mpl_toolkits.axes_grid1")
Line2D = pyimport("matplotlib.lines").Line2D
import PyPlot as plt
using LaTeXStrings

rcParams = PyDict(plt.matplotlib["rcParams"])
rcParams["font.size"] = 8

function run_RRG_FC_Ferro(N, Ks, lam, D, tmax)
    lambdas = fill(lam, N)

    dt_G = 1e-2
    T_G = Int(tmax/dt_G)
    tvec_G = collect(range(0, T_G*dt_G, length=T_G + 1))

    Cs = [zeros(T_G+1) for _ in 1:length(Ks)]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=true)

    colors = ["C0", "C1", "C2", "C4"]
    lts = ["-", "--", "-.", ":"]
    # Iterate over different K values
    for (iK, K) in enumerate(Ks)
        rng = Xoshiro(1234)

        J = 1.0/K
        Jmat = adjacency_matrix(random_regular_graph(N, K; rng=rng), Float64) .* J 
        
        model = OUModel(K, Jmat, lambdas, D)

        nodes = run_cavity_EQ(model, dt_G, T_G; showprogress=true)

        C, _ = compute_averages(nodes, model, T_G)
        Cs[iK] .= C.parent

        # Plot correlations
        if K < N-1
            ax.plot(tvec_G, C, color=colors[iK], label="K = $K", ls=lts[iK])
        else
            ax.plot(tvec_G, C, color=colors[iK], label="FC", ls=lts[iK])
        end
    end

    ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel("C(t)")
    ax.set_title("Equilibrium correlation")
    ax.set_yscale("log")

    ax.set_xlim(0, tmax)
    #ax.set_ylim(, 1.5)
    ax.grid(alpha=.5)

    fig.savefig("Compare_FC_Ceq_RRG_N-$(N)_lam-$(lam)_D-$(D)_tmax-$(tmax).png", dpi=300, bbox_inches="tight")

    # Save data
    isdir("data/RRG_Ferro/plot/compare_FC") || mkdir("data/RRG_Ferro/plot/compare_FC")

    jldsave("data/RRG_Ferro/plot/compare_FC/N-$(N)_lam-$(lam)_D-$(D)_tmax-$(tmax).jld2"; Cs, tvec_G, Ks)
end

N = 1000
Ks = [3, 10, 100, N-1]

lam, D = 1.3, 1.0
tmax = 10.0

run_RRG_FC_Ferro(N, Ks, lam, D, tmax)