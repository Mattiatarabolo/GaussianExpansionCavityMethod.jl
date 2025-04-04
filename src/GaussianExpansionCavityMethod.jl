module GaussianExpansionCavityMethod

using Random, SparseArrays, LinearAlgebra, StatsBase, Graphs, ProgressMeter

export Phi4Model, sample_phi4, Phi4ModelEnsemble, Phi4ModelRRG, sample_ensemble_phi4, BMModel, sample_BM, BMModelEnsemble, BMModelRRG, sample_ensemble_BM, TwoSpinModel, TwoSpinModelEnsemble, TwoSpinModelRRG, sample_2Spin, sample_ensemble_2Spin, integrate_2spin_RRG, compute_meanstd, compute_stats

include("Phi4/types.jl")
include("Phi4/sample.jl")
include("BouchaudMezard/types.jl")
include("BouchaudMezard/sample.jl")
include("2Spin/types.jl")
include("2Spin/sample.jl")
include("2Spin/integrate.jl")
include("utils.jl")

end
