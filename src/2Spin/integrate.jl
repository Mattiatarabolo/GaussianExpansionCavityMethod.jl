# Function to compute mu(n*dt + dt) (\lambda/(t) in the paper, Eq (112))
function compute_mu_Bim!(mu, J, D, Ch, Rh, C, R, n, dt, p)
	# Initialize integrals
	int1, int2 = 0.0, 0.0
	# Iterate from 0 to n-1
	@fastmath @inbounds @simd for m in 0:n-1  # should be up to n, but R[n,n]=Rh[n,n]=0
		int1 += Rh[n, m] * C[n, m]
		int2 += R[n, m] * Ch[n, m]
		# update progress bar
		next!(p)
	end
	mu[n] = D + dt * J^2 * (int1 + int2)
end


# Function to compute mu(n*dt + dt) (\lambda/(t) in the paper, Eq (112))
function compute_corrs_mu_Bim!(mu, Ch, C, K, J, D, Rh, R, n, dt, p)
	# Initialize integrals
	intmu, int1h, int2h, int1, int2 = 0.0, 0.0, 0.0, 0.0, 0.0
	# Iterate from 0 to n-1
	@fastmath @inbounds @simd for m in 0:n-1  # should be up to n, but R[n,n]=Rh[n,n]=0
		intmu += Rh[n+1, m] * C[n+1, m] + R[n+1, m] * Ch[n+1, m]
		int1h += Rh[n, m] * Ch[n+1, m]
		int2h += Rh[n+1, m] * Ch[n, m]
		int1 += Rh[n, m] * C[n+1, m]
		int2 += R[n+1, m] * Ch[n, m]
		# update progress bar
		next!(p)
	end
	# Add term for m = n
	intmu += Rh[n+1, n] * C[n+1, n] + R[n+1, n] * Ch[n+1, n]
	int2h += Rh[n+1, n] * Ch[n+1, n]
	int2 += R[n+1, n] * Ch[n+1, n]
	# Compute mu(n*dt + dt) (\lambda/(t) in the paper, Eq (112))
	mu[n+1] = D + dt * J^2 * intmu
	# Compute Ch(n*dt + dt, n*dt + dt) (Eq. (109)), C(n*dt + dt, n*dt + dt) (Eq. (111))
	Ch[n+1, n+1] = (1 - mu[n] * dt) * Ch[n+1, n] + dt^2 * (K - 1) / K * J^2 * (int1h + int2h) + 2 * dt * D * Rh[n+1, n]
	C[n+1, n+1] = (1 - mu[n] * dt) * C[n+1, n] + dt^2 * J^2 * (int1 + int2) + 2 * dt * D * R[n+1, n]
end

function compute_integrals_Bim(Ch, Rh, C, R, n, l, p)
	# Initialize integrals
	int1h, int2h, int3h, int1, int2, int3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	# Iterate from 0 to l-1
	@inbounds @fastmath @simd for m in 0:l-1
		int2h += Rh[n, m] * Ch[l, m]
		int3h += Rh[l, m] * Ch[n, m]
		int2 += Rh[n, m] * C[l, m]
		int3 += R[l, m] * Ch[n, m]
		# update progress bar
		next!(p)
	end
	# Add term for m = l
	int2h += Rh[n, l] * Ch[l, l]
	int2 += Rh[n, l] * C[l, l]
	# update progress bar
	next!(p)
	# Iterate from l+1 to n-1
	@fastmath @inbounds @simd for m in l+1:n-1
		int1h += Rh[n, m] * Rh[m, l]
		int2h += Rh[n, m] * Ch[m, l]
		int1 += Rh[n, m] * R[m, l]
		int2 += R[n, m] * Ch[m, l]
		# update progress bar
		next!(p)
	end
	return int1h, int2h, int3h, int1, int2, int3
end

# Function to compute Ch(n*dt + dt, l*dt) (Eq. (109)), Rh(n*dt + dt, l*dt) (Eq. (108)), C(n*dt + dt, l*dt) (Eq. (111)), R(n*dt + dt, l*dt) (Eq. (110))
function compute_functions_Bim!(Ch, Rh, C, R, K, J, mu, n, l, dt, p)
	# Compute integrals
	int1h, int2h, int3h, int1, int2, int3 = compute_integrals_Bim(Ch, Rh, C, R, n, l, p)
	# Compute Rh(n*dt + dt, l*dt) (Eq. (108))
	Rh[n+1, l] = (1 - mu[n] * dt) * Rh[n, l] + dt^2 * (K - 1) / K * J^2 * int1h + (n == l)
	# Compute Ch(n*dt + dt, l*dt) (Eq. (109))
	Ch[n+1, l] = (1 - mu[n] * dt) * Ch[n, l] + dt^2 * (K - 1) / K * J^2 * (int2h + int3h)
	# Compute R(n*dt + dt, l*dt) (Eq. (110))
	R[n+1, l] = (1 - mu[n] * dt) * R[n, l] + dt^2 * J^2 * int1 + (n == l)
	# Compute C(n*dt + dt, l*dt) (Eq. (111))
	C[n+1, l] = (1 - mu[n] * dt) * C[n, l] + dt^2 * J^2 * (int2 + int3)
end


"""
    integrate_2spin_Bim_RRG(K, J, D, dt, T; backup=false, backupfile="data/RRG/backup_matrices.jld2", backupevery=1000, showprogress=false)

Integrate the disorder averaged cavity equations for a 2-spin model on a random regular graph with bimodal interactions.

# Srguments
- `K::Int`: The average number of neighbors.
- `J::Float64`: The coupling strength.
- `D::Float64`: The noise strength.
- `dt::Float64`: The time step for the integration.
- `T::Int`: The number of iterations to run the integration.

# Keyword Arguments
- `backup::Bool`: Whether to save a backup of the matrices every `backupevery` iterations. Default is `false`.
- `backupfile::String`: The filename for the backup file. Default is `"data/RRG/backup_matrices.jld2"`.
- `backupevery::Int`: The number of iterations to save a backup every. Default is `1000`.
- `showprogress::Bool`: Whether to show a progress bar. Default is `false`.

# Returns
- `C::OffsetArray`: The disorder averaged autocorrelation C matrix.
- `R::OffsetArray`: The disorder averaged response R matrix.
- `Ch::OffsetArray`: The disorder averaged cavity autocorrelation Ch matrix.
- `Rh::OffsetArray`: The disorder averaged cavity response Rh matrix.
- `mu::OffsetArray`: The Lagrange multiplier mu array.
"""
function integrate_2spin_Bim_RRG(K::Int, J::Float64, D::Float64, dt::Float64, T::Int; backup=false, backupfile="data/RRG/backup_matrices.jld2", backupevery=1000, showprogress=false)
    # Check if parameters are valid
    @assert K > 0 "K must be positive"
    @assert J > 0 "J must be positive"
    @assert D > 0 "D must be positive"

	# Check if data directory exists (only if backup is true)
	if backup
		isdir("data") || mkdir("data")
		isdir("data/RRG") || mkdir("data/RRG")
	end
    
	# Initiliaze matrices
    C = OffsetArray(zeros(T + 1, T + 1), 0:T, 0:T)
	R = OffsetArray(zeros(T + 1, T + 1), 0:T, 0:T)
	Ch = OffsetArray(zeros(T + 1, T + 1), 0:T, 0:T)
	Rh = OffsetArray(zeros(T + 1, T + 1), 0:T, 0:T)
	mu = OffsetArray(zeros(T + 1), 0:T)
	
	# Initialize Ch and C to 1.0 at t=0, t'=0 (needed for spherical symmetry of the system)
	Ch[0, 0] = 1.0
	C[0, 0] = 1.0

	# Initialize mu to D at t=0
	mu[0] = D

	# Initialize progress bar for the cavity updates
    p_tot_iterations = Int(T + T * (T - 1) * (2 * T - 1) / 6 + T * (T - 1))
    p = Progress(p_tot_iterations; enabled=showprogress, dt=0.3, showspeed=true, desc="Progress: ")

	# Integration loop
	@fastmath @inbounds for n in 0:T-1
        # Compute Ch(n*dt + dt, l*dt) (Eq. (109)), Rh(n*dt + dt, l*dt) (Eq. (108)), C(n*dt + dt, l*dt) (Eq. (111)), R(n*dt + dt, l*dt) (Eq. (110))
        # Note: Ch[n+1, l] = Ch(n*dt + dt, l*dt), Rh[n+1, l] = Rh(n*dt + dt, l*dt), C[n+1, l] = C(n*dt + dt, l*dt), R[n+1, l] = R(n*dt + dt, l*dt)
		@fastmath @inbounds for l in 0:n
			compute_functions_Bim!(Ch, Rh, C, R, K, J, mu, n, l, dt, p)
		end
		#=
        # Ensure spherical constraint, Ch(t, t) = 1.0 = C(t, t)
		Ch[n+1, n+1] = 1.0
		C[n+1, n+1] = 1.0
		# Compute mu(n*dt + dt) (\lambda/(t) in the paper, Eq (112))
		compute_mu_Bim!(mu, J, D, Ch, Rh, C, R, n+1, dt, p)
		=#
		# Compute Ch(n*dt + dt, n*dt + dt) (Eq. (109)), C(n*dt + dt, n*dt + dt) (Eq. (111)) and mu(n*dt + dt) (\lambda/(t) in the paper, Eq (112))
		compute_corrs_mu_Bim!(mu, Ch, C, K, J, D, Rh, R, n, dt, p)
        
		# Save backup every backupevery iterations
		if backup && n % backupevery == 0
            jldsave(backupfile; C, R, Ch, Rh, mu, n)
        end
	end
    return C, R, Ch, Rh, mu
end