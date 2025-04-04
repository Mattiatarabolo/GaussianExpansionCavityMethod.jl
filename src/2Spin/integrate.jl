# Function to compute mu(n*dt + dt) (\lambda/(t) in the paper, Eq (112))
function compute_mu(K, J, D, Ch, Rh, C, R, n, dt)
	auxp = 0.0
	@fastmath @inbounds @simd for m in 0:n-1  # should be up to n, but R[n,n]=Rh[n,n]=0
		auxp += Rh[n, m] * C[n, m] + R[n, m] * Ch[n, m]
		#auxp += Rh[n, l] * C[l, n] + R[n, l] * Ch[n, l]
	end
	auxp *= dt * K * J^2
	return D + auxp
end

# Function to compute Ch(n*dt + dt, l*dt) (Eq. (109))
function compute_Ch(K, J, D, Ch, Rh, mu, n, l, dt)
	if n >= l
		auxp1 = 0.0
		auxp2 = 0.0
		@fastmath @inbounds @simd for m in 0:l
			auxp1 += Rh[n, m] * Ch[l, m]
			#auxp1 += Rh[n, m] * Ch[m, l]
			auxp2 += Rh[l, m] * Ch[n, m]
		end
		@fastmath @inbounds @simd for m in l+1:n-1
			auxp1 += Rh[n, m] * Ch[m, l]
		end
		auxp1 *= dt^2 * (K - 1) * J^2
		auxp2 *= dt^2 * (K - 1) * J^2
		return Ch[n, l] * (1 - mu[n] * dt) + auxp1 + auxp2
	else
		auxp1 = 0.0
		auxp2 = 0.0
		@fastmath @inbounds @simd for m in 0:n
			auxp1 += Rh[n, m] * Ch[l, m]
			#auxp1 += Rh[n, m] * Ch[m, l]
			auxp2 += Rh[l, m] *Ch[n, m]
		end
		auxp1 *= dt^2 * (K - 1) * J^2 
		@fastmath @inbounds @simd for m in n+1:l-1  # should be up to l, but R[l,l]=Rh[l,l]=0
		auxp2 += Rh[l, m] * Ch[m, n]
		end
		auxp2 *= dt^2 * (K - 1) * J^2
		return Ch[l, n] * (1 - mu[n] * dt) + auxp1 + auxp2 + 2 * D * Rh[l, n] * dt
	end
end

# Function to compute Rh(n*dt + dt, l*dt) (Eq. (108))
function compute_Rh(K, J, D, Rh, mu, n, l, dt)
	auxp = 0.0
	@fastmath @inbounds @simd for m in l:n-1 # should be up to n, but R[n,n]=Rh[n,n]=0
		auxp += Rh[n, m] * Rh[m, l]
	end
	auxp *= dt^2 * (K - 1) * J^2
	return Rh[n, l] * (1 - mu[n] * dt) + auxp + Float64(n == l)
end

# Function to compute C(n*dt + dt, l*dt) (Eq. (111))
function compute_C(K, J, D, Ch, Rh, C, R, mu, n, l, dt)
	if n >= l
		auxp1 = 0.0
		auxp2 = 0.0
		@fastmath @inbounds @simd for m in 0:l
			auxp1 += Rh[n, m] * C[l, m]
			auxp2 += R[l, m] * Ch[n, m]
		end
		@fastmath @inbounds @simd for m in l+1:n-1
			auxp1 += Rh[n, m] * C[m, l]
		end
		auxp1 *= dt^2 * K * J^2
		auxp2 *= dt^2 * K * J^2
    	return C[n, l] * (1 - mu[n] * dt) + auxp1 + auxp2
	else
		auxp1 = 0.0
		auxp2 = 0.0
		@fastmath @inbounds @simd for m in 0:n
			auxp1 += Rh[n, m] * C[l, m]
			auxp2 += R[l, m] * Ch[n, m]
		end
		auxp1 *= dt^2 * J^2
		@fastmath @inbounds @simd for m in n+1:l-1  # should be up to l, but R[l,l]=Rh[l,l]=0
			auxp2 += R[l, m] * Ch[m, n]
		end
		auxp2 *= dt^2 * K * J^2
		return C[l, n] * (1 - mu[n] * dt) + auxp1 + auxp2 + 2 * D * R[l, n] * dt
	end
end

# Function to compute R(n*dt + dt, l*dt) (Eq. (110))
function compute_R(K, J, D, Rh, R, mu, n, l, dt)
	auxp = 0.0
	@fastmath @inbounds @simd for m in l:n-1 # should be up to n, but R[n,n]=Rh[n,n]=0
		auxp += Rh[n, m] * R[m, l]
	end
	auxp *= dt^2 * K * J^2
	return R[n, l] * (1 - mu[n] * dt) + auxp + Float64(n == l)
end

"""
    integrate_2spin_RRG(K, J, D, Nmax, tmax; backup=false, backupfile="backup_matrices.jld2", backupevery=1000)

Integrate the disorder averaged cavity equations for a 2-spin model on a random regular graph with bimodal interactions.

# Srguments
- `K::Int`: The average number of neighbors.
- `J::Float64`: The coupling strength.
- `D::Float64`: The noise strength.
- `Nmax::Int`: The maximum number of timesteps.
- `tmax::Float64`: The maximum time to integrate to.

# Keyword Arguments
- `backup::Bool`: Whether to save a backup of the matrices every `backupevery` iterations. Default is `false`.
- `backupfile::String`: The filename for the backup file. Default is `"backup_matrices.jld2"`.
- `backupevery::Int`: The number of iterations to save a backup every. Default is `1000`.

# Returns
- `C::OffsetArray`: The disorder averaged autocorrelation C matrix.
- `R::OffsetArray`: The disorder averaged response R matrix.
- `Ch::OffsetArray`: The disorder averaged cavity autocorrelation Ch matrix.
- `Rh::OffsetArray`: The disorder averaged cavity response Rh matrix.
- `mu::OffsetArray`: The Lagrange multiplier mu array.
"""
function integrate_2spin_RRG(K::Int, J::Float64, D::Float64, Nmax::Int, tmax::Float64; backup=false, backupfile="backup_matrices.jld2", backupevery=1000)
    # Check if parameters are valid
    @assert K > 0 "K must be positive"
    @assert J > 0 "J must be positive"
    @assert D > 0 "D must be positive"
    # Define timestep
	dt = tmax / Nmax
    # Initiliaze matrices
    C = OffsetArray(zeros(Nmax + 1, Nmax + 1), 0:Nmax, 0:Nmax)
	R = OffsetArray(zeros(Nmax + 1, Nmax + 1), 0:Nmax, 0:Nmax)
	Ch = OffsetArray(zeros(Nmax + 1, Nmax + 1), 0:Nmax, 0:Nmax)
	Rh = OffsetArray(zeros(Nmax + 1, Nmax + 1), 0:Nmax, 0:Nmax)
	mu = OffsetArray(zeros(Nmax + 1), 0:Nmax)
	# Initialize Ch and C to 1.0 at t=0, t'=0 (needed for spherical symmetry of the system)
	Ch[0, 0] = 1.0
	C[0, 0] = 1.0
	# Integration loop
	@fastmath @inbounds for n in 0:Nmax-1
        # Compute mu(n*dt + dt) (Eq. (112))
		mu[n] = compute_mu(K, J, D, Ch, Rh, C, R, n, dt)
        # Compute Ch(n*dt + dt, l*dt) (Eq. (109)), Rh(n*dt + dt, l*dt) (Eq. (108)), C(n*dt + dt, l*dt) (Eq. (111)), R(n*dt + dt, l*dt) (Eq. (110))
        # Note: Ch[n+1, l] = Ch(n*dt + dt, l*dt), Rh[n+1, l] = Rh(n*dt + dt, l*dt), C[n+1, l] = C(n*dt + dt, l*dt), R[n+1, l] = R(n*dt + dt, l*dt)
		@fastmath @inbounds for l in 0:n
			Ch[n+1, l] = compute_Ch(K, J, D, Ch, Rh, mu, n, l, dt)
			Rh[n+1, l] = compute_Rh(K, J, D, Rh, mu, n, l, dt)
			C[n+1, l] = compute_C(K, J, D, Ch, Rh, C, R, mu, n, l, dt)
			R[n+1, l] = compute_R(K, J, D, Rh, R, mu, n, l, dt)
		end
        # Ensure spherical constraint, Ch(t, t) = 1.0 = C(t, t)
		Ch[n+1, n+1] = 1.0
		C[n+1, n+1] = 1.0
        # 
		if n % 200 == 0
			if !isfinite(mu[n])
                write_error(K, Nmax, tmax)
                error("mu[$n] is not finite")
			end
			println("Iteration $n of $Nmax")
			println("time $(round(n*dt, digits=2)) mu $(mu[n])")
            write_log(K, Nmax, tmax, n*dt, mu[n])
		end
        # Save backup every backupevery iterations
		if backup && n % backupevery == 0
            jldsave(backupfile; C, R, Ch, Rh, mu, n)
        end
	end
    return C, R, Ch, Rh, mu
end