module GaussianExpansionCavityMethod

using Random, SparseArrays, LinearAlgebra, Statistics, Graphs, ProgressMeter, OffsetArrays

export OUModel, sample_OU, OUModelEnsemble, OUModelRRG, sample_ensemble_OU, Cavity, Marginal, Node, CavityEQ, MarginalEQ, NodeEQ, run_cavity_EQ, run_cavity, compute_averages, compute_mean, Phi4Model, sample_phi4, Phi4ModelEnsemble, Phi4ModelRRG, sample_ensemble_phi4, BMModel, sample_BM, BMModelEnsemble, BMModelRRG, sample_ensemble_BM, TwoSpinModel, TwoSpinModelEnsemble, TwoSpinModelRRG_Bim, TwoSpinModelRRG_Ferro, sample_2Spin, sample_ensemble_2Spin, integrate_2spin_Bim_RRG, compute_meanstd, compute_autocorr, compute_stats, compute_autocorr_TTI, compute_stats_TTI

include("linear_OrnteinUhlenbeck/types.jl")
include("linear_OrnteinUhlenbeck/utils.jl")
include("linear_OrnteinUhlenbeck/sample.jl")
include("linear_OrnteinUhlenbeck/cavity.jl")
include("Phi4/types.jl")
include("Phi4/sample.jl")
include("BouchaudMezard/types.jl")
include("BouchaudMezard/sample.jl")
include("2Spin/types.jl")
include("2Spin/sample.jl")
include("2Spin/integrate.jl")
include("utils.jl")

end
