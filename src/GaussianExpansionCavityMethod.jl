module GaussianExpansionCavityMethod

using Random, SparseArrays, LinearAlgebra, StatsBase, Graphs, ProgressMeter, DifferentialEquations#, OffsetArrays,  CurveFit,  LoopVectorization, 

export Phi4Model, sample_phi4, Phi4ModelEnsemble, Phi4ModelRRG, sample_ensemble_phi4, compute_meanstd, compute_stats

include("Phi4/types.jl")
include("Phi4/sample.jl")
include("utils.jl")

end
