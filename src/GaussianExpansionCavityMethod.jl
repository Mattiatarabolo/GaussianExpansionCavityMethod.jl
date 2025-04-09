module GaussianExpansionCavityMethod

using Random, SparseArrays, LinearAlgebra, Statistics, Graphs, ProgressMeter, OffsetArrays

export OUModel, sample_OU, OUModelEnsemble, OUModelRRG, sample_ensemble_OU, Cavity, Marginal, Node, CavityEQ, MarginalEQ, NodeEQ, run_cavity_EQ, compute_averages, Phi4Model, sample_phi4, Phi4ModelEnsemble, Phi4ModelRRG, sample_ensemble_phi4, BMModel, sample_BM, BMModelEnsemble, BMModelRRG, sample_ensemble_BM, TwoSpinModel, TwoSpinModelEnsemble, TwoSpinModelRRG, sample_2Spin, sample_ensemble_2Spin, integrate_2spin_RRG, compute_meanstd, compute_stats

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
