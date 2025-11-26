using GaussianExpansionCavityMethod
using Test, Random, SparseArrays, OffsetArrays, ProgressMeter, LinearAlgebra, Statistics
using Aqua

const BASE_SEED = UInt32(0x12345678)
const SEED_RNG = Xoshiro(BASE_SEED)

# simple symmetric adjacency for N nodes
simple_adj(N) = sparse(diagm(-1 => ones(N-1), 1 => ones(N-1)))

@testset "GaussianExpansionCavityMethod" begin
    @testset "OU types and sampling" begin
        N = 3; K = 2; lambda = 0.4; D = 0.0
        J = spzeros(N,N)
        lambdas = fill(lambda, N)

        model = OUModel(K, J, lambdas, D)
        @test model.N == N && model.K == K
        @test_throws AssertionError OUModel(K, spzeros(2,3), lambdas, D)
        @test OUModel(K, J, lambda, D).lambdas == lambdas

        ensemble = OUModelRRG(N, K, 0.0, lambda, D, -1.0, 1.0)
        @test ensemble.N == N && ensemble.K == K
        @test ensemble.x0_params == [-1.0, 1.0]

        tmax = 0.1; tsave = [0.0, 0.05, 0.1]; x0 = ones(N)
        tvec, traj = sample_OU(model, x0, tmax, tsave; rng=Xoshiro(BASE_SEED))
        @test all(tvec .≈ tsave)
        @test all((traj[:,2] .- exp(-lambda*tsave[2]) .* x0) .<= 1e-5)

        tvals_all, traj_all = sample_ensemble_OU(ensemble, tmax, tsave, 2; rng=Xoshiro(BASE_SEED+1))
        @test length(traj_all) == 2
        @test size(traj_all[1]) == (N, length(tsave))
    end

    @testset "OU cavity structs" begin
        T = 3
        cav = Cavity(1, 2, T)
        @test cav.i == 1 && cav.j == 2
        @test length(cav.mu) == T+1 && size(cav.C) == (T+1, T+1)

        marg = Marginal(1, T)
        @test marg.i == 1 && length(marg.mu) == T+1

        node = Node(1, [2,3], T)
        @test node.i == 1 && node.neighs == [2,3]
        @test length(node.cavs) == 2
        @test size(node.sumC) == (T+1, T+1)

        cav_eq = CavityEQ(1, 2, T)
        @test cav_eq.C[0] == 0.0

        marg_eq = MarginalEQ(1, T)
        @test marg_eq.C[0] == 0.0 && marg_eq.R[0] == 0.0

        node_eq = NodeEQ(1, [2,3], T)
        @test length(node_eq.cavs) == 2 && node_eq.neighs_idxs[3] == 2
    end

    @testset "OU cavity updates" begin
        N = 2; K = 1; lambda = 0.2; D = 0.0; dt = 0.01; T = 3
        J = spzeros(N,N); J[1,2] = J[2,1] = 0.1
        model = OUModel(K, J, lambda, D)

        p = Progress(1; enabled=false)

        nodes_eq = GaussianExpansionCavityMethod.init_nodes_EQ(model, T)
        # seed initial values to exercise update_sumC0!, update_sumC!
        GaussianExpansionCavityMethod.update_sumC0!(nodes_eq[1], model, p)
        GaussianExpansionCavityMethod.update_sumC!(nodes_eq[1], model, 1, p)
        @test all(isfinite, nodes_eq[1].sumC[:])

        cav_eq_nodes = run_cavity_EQ(model, dt, T; showprogress=false)
        @test length(cav_eq_nodes) == N
        @test cav_eq_nodes[1].marg.C[0] == 1.0

        nodes = GaussianExpansionCavityMethod.init_nodes(model, T)
        GaussianExpansionCavityMethod.cavity_update!(nodes, model, dt, T, p)
        GaussianExpansionCavityMethod.marginal_update!(nodes, model, dt, T, p)
        @test all(isfinite, nodes[1].marg.C[:])
        @test size(nodes[1].marg.mu, 1) == T+1

        cav_nodes = run_cavity(model, dt, T; showprogress=false)
        @test length(cav_nodes) == N
        @test size(cav_nodes[1].marg.R, 1) == T+1
    end

    @testset "OU utilities" begin
        N = 3; K = 1; lambda = 0.1; D = 0.0; dt = 0.01; T = 2
        J = simple_adj(N)
        model = OUModel(K, J, lambda, D)
        nodes_eq = run_cavity_EQ(model, dt, T; showprogress=false)
        C_avg, R_avg = compute_averages(nodes_eq, model, T)
        @test length(C_avg) == T+1
        @test length(R_avg) == T+1

        nodes = run_cavity(model, dt, T; showprogress=false)
        C_mat, R_mat = compute_averages(nodes, model, T)
        mu = compute_mean(nodes, model, T)
        @test size(C_mat) == (T+1, T+1)
        @test length(mu) == T+1
    end

    @testset "Phi4 sampling and ensemble" begin
        N = 2; K = 1; lambda = 0.2; D = 0.0; u = 0.5
        J = spzeros(N,N)
        model = Phi4Model(K, J, lambda, D, u)
        x0 = fill(0.5, N); tsave = [0.0, 0.05]; tmax = 0.05

        tvec, traj = sample_phi4(model, x0, tmax, tsave; rng=Xoshiro(BASE_SEED))
        @test all(tvec .≈ tsave)
        @test all(traj[:,end] .< x0) # decay

        ensemble = Phi4ModelRRG(N, K, 0.0, lambda, D, u)
        tvals_all, traj_all = sample_ensemble_phi4(ensemble, -0.1, 0.1, tmax, tsave, 2; rng=Xoshiro(BASE_SEED+2))
        @test length(traj_all) == 2
        @test size(traj_all[1]) == (N, length(tsave))
    end

    @testset "Bouchaud-Mezard sampling and ensemble" begin
        N = 2; K = 1; sigma = 0.0; J = spzeros(N,N)
        model = BMModel(K, J, sigma)
        x0 = ones(N); tsave = [0.0, 0.05]; tmax = 0.05

        tvec, traj = sample_BM(model, x0, tmax, tsave; rng=Xoshiro(BASE_SEED), dt=1e-3)
        @test all(tvec .≈ tsave) && all((traj[:,end] .- x0) .< 1e-5)

        ensemble = BMModelRRG(N, K, 0.0, sigma)
        tvals_all, traj_all = sample_ensemble_BM(ensemble, 0.0, 0.0, tmax, tsave, 2; rng=Xoshiro(BASE_SEED+3))
        @test length(traj_all) == 2 && size(traj_all[1]) == (N, length(tsave))
    end

    @testset "Two-Spin sampling and integration" begin
        N = 3; K = 2; D = 0.0; J = spzeros(N,N)
        model = TwoSpinModel(K, J, D)
        x0 = ones(N); tsave = [0.0, 0.02]; tmax = 0.02

        tvec, traj = sample_2Spin(model, x0, tmax, tsave; rng=Xoshiro(BASE_SEED), dt=1e-4)
        @test all(tvec .≈ tsave)
        @test all(sum(traj[:, n] .^ 2) ≈ N for n in 1:length(tsave))

        ensemble_bim = TwoSpinModelRRG_Bim(N, K, 0.0, D)
        tvals_all, traj_all = sample_ensemble_2Spin(ensemble_bim, -0.1, 0.1, tmax, tsave, 2; rng=Xoshiro(BASE_SEED+4), dt=1e-4)
        @test size(traj_all[1]) == (N, length(tsave))

        # Integration helpers on minimal grid
        Kint, Jint, Dint, dt = 1, 0.1, 0.1, 0.01; T = 1
        C, R, Ch, Rh, mu = integrate_2spin_Bim_RRG(Kint, Jint, Dint, dt, T; showprogress=false)
        @test size(C) == (T+1, T+1) && mu[0] == Dint
        # internal integrals
        p = Progress(1; enabled=false)
        mu_vec = OffsetArray(zeros(T+1), 0:T)
        GaussianExpansionCavityMethod.compute_mu_Bim!(mu_vec, Jint, Dint, Ch, Rh, C, R, 1, dt, p)
        @test isfinite(mu_vec[1])
    end

    @testset "Internal 2-Spin integrals" begin
        T = 2
        Ch = OffsetArray(fill(1.0, T+1, T+1), 0:T, 0:T)
        Rh = OffsetArray(fill(0.5, T+1, T+1), 0:T, 0:T)
        C = OffsetArray(fill(1.0, T+1, T+1), 0:T, 0:T)
        R = OffsetArray(fill(0.5, T+1, T+1), 0:T, 0:T)
        mu = OffsetArray(zeros(T+1), 0:T)
        p = Progress(10; enabled=false)

        GaussianExpansionCavityMethod.compute_corrs_mu_Bim!(mu, Ch, C, 2, 0.1, 0.1, Rh, R, 0, 0.01, p)
        @test isfinite(mu[1])

        ints = GaussianExpansionCavityMethod.compute_integrals_Bim(Ch, Rh, C, R, 1, 0, p)
        @test length(ints) == 6

        GaussianExpansionCavityMethod.compute_functions_Bim!(Ch, Rh, C, R, 2, 0.1, mu, 1, 0, 0.01, p)
        @test isfinite(Ch[2,0]) && isfinite(R[2,0])
    end

    @testset "Utils: stats helpers" begin
        trajs = [1.0 2.0 3.0; 2.0 4.0 6.0] # N=2, T=3
        p = Progress(10; enabled=false)
        z = zeros(3)
        GaussianExpansionCavityMethod.demean_ts!(z, trajs, 1, mean(trajs; dims=1)[:], p)
        @test sum(z) ≈ -3.0

        C = GaussianExpansionCavityMethod._autocorr(trajs; dims=1)
        @test C[1,1] == mean(trajs[:,1] .^ 2)

        Cs = GaussianExpansionCavityMethod._autocorr_tws(trajs, [1,2], [[1,2],[2,3]]; dims=1)
        @test length(Cs) == 2 && length(Cs[1]) == 2

        mu, q, _ = compute_meanstd(trajs)
        @test mu ≈ [1.5, 3.0, 4.5] && q ≈ sqrt.([0.5, 2.0, 2 * (1.5)^2])

        trajs1 = copy(trajs)
        trajs2 = copy(trajs)

        sim = [trajs1, trajs2]
        @test compute_meanstd(sim)[1] ≈ [1.5, 3.0, 4.5]
    end
end

@testset "Aqua" begin
    Aqua.test_all(GaussianExpansionCavityMethod, deps_compat=(check_extras=false, check_weakdeps=false))
end
