using DrWatson
using HDF5

"""
    W_II(α::Float64, J::Float64, Bx::Float64, Bz::Float64, dt::Float64, Ni::Int, Nj::Int)

Type for ``W_{II}`` representation of time-evolution operator.
Loads the coefficients of an order `Ni` approximation of the power-law Ising couplings.
Creates the ``A``, ``B``, ``C`` and ``D`` operator valued matrices of the MPO representation of ``H = J\\sum_{i,j}|r_i-r_j|^{-α}ℤ_i ℤ_j + \\sum_i B_x 𝕏_i + B_z ℤ_i``

### Arguments
- `α` : power-law exponent
- `J` : Ising strength
- `Bx, Bz` : field strengths
- `dt` : time-step
- `Ni` : MPO bond dimension
    
"""
mutable struct WII
    J::Float64;
    Bx::Float64;
    Bz::Float64;
    dt::Float64;
    
    alpha::Float64;
    L::Int64;
    betas::Array{Float64};
    lambdas::Array{Float64};
    Ni::Int64;
    Nj::Int64;
    
    A::Array{Array{ComplexF64,2},2}
    B::Array{Array{ComplexF64,2},2}
    C::Array{Array{ComplexF64,2},2}
    D::Array{ComplexF64, 2};

    WA::Array{ComplexF64, 2};
    WB::Array{ComplexF64, 2};
    WC::Array{ComplexF64, 2};
    WD::Array{ComplexF64, 2};
    
    Wi::Array{ComplexF64, 4};
    W1::Array{ComplexF64, 4};
    WN::Array{ComplexF64, 4};

    
    function WII(α::Float64, L::Int64, J::Float64, Bx::Float64, Bz::Float64, dt::Float64, Ni::Int, Nj::Int; kac_norm::Bool = true) #! the actual calculation of WII can be splitted to an outer constructor
        A = fill(zeros(2,2), (Ni,Nj));
        B = fill(zeros(2,2), (Ni,1));
        C = fill(zeros(2,2), (1,Nj));
        D = zeros(2, 2);
        
        if kac_norm == false
            pl_MPO = h5open(projectdir("input/pl_Ham_MPO_Mmax=$(Ni)_L=$(L).h5"), "r"); # load fit data
            # kac = sum([abs(i-j)^(-α) for i ∈ 1:L for j ∈ i+1:L])/(L-1); #! commented out on 08/09/2022
            kac = 1;
        else
            pl_MPO = h5open(projectdir("input/pl_Ham_MPO_Mmax=$(Ni)_L=$(L)_kac_norm.h5"), "r"); # load fit data
            kac = 1;
        end
        
        βs = read(pl_MPO["alpha=$(α)/betas"]);
        λs = read(pl_MPO["alpha=$(α)/lambdas"]);
        
        # Fills A,B,C,D operator valued matrices of Hamiltonian
        𝟙 = [1 0; 0 1]; ℤ = [1 0; 0 -1]; 𝕏 = [0 1; 1 0]; 𝟘 = [0 0; 0 0];
        for m in 1:Ni
            A[m, m] = λs[m]*𝟙;
            B[m] = J*λs[m]*ℤ;
            C[m] = βs[m]*ℤ;
        end
        D = kac*(Bz*ℤ + Bx*𝕏); 
        
        # Creates Wi matrices of U(dt)
        dτ = -im*dt;        

        WA = zeros(ComplexF64, 2*Ni, 2*Nj);
        WB = zeros(ComplexF64, 2*Ni, 2);
        WC = zeros(ComplexF64, 2, 2*Nj);
        WD = exp(dτ*D);

        @inbounds for aj ∈ 1:Ni, aj_c ∈ 1:Nj
            F = [
                dτ*D          𝟘         𝟘          𝟘;
                √dτ*C[aj_c]  dτ*D       𝟘          𝟘;
                √dτ*B[aj]     𝟘        dτ*D        𝟘;
                A[aj,aj_c]  √dτ*B[aj]  √dτ*C[aj_c] dτ*D;
                ]; #! A[i,j] = 0 if i ≠ j?
            
            Φ = exp(F); # Calc exp of matrix elements of F
            
            # Fill in matrix elements of WII
            WA[2aj-1:2aj, 2aj_c-1:2aj_c] = Φ[7:8,1:2];
            
            if aj_c == 1
                WB[2aj-1:2aj,:] = Φ[5:6,1:2];
            end
            
            if aj == 1
                WC[:,2aj_c-1:2aj_c] = Φ[3:4,1:2];
            end

        end

        Wi = reshape(vcat(hcat(WD,WC), hcat(WB, WA)), 2, Ni+1, 2, Nj+1);
        W1 = reshape(hcat(WD, WC), 2, 1, 2, Ni+1);
        WN = reshape(vcat(WD, WB), 2, Nj+1, 2, 1); #! Study Fig. 3b PRB 91 (165112), 2015

        new(J, Bx, Bz, dt, α, L, βs, λs, Ni, Ni, A, B, C, D, WA, WB, WC, WD, Wi, W1, WN); # set fields as order inside struct
    end
end


"""
    function calc_Ut(oWII::WII, t::Float64)

Calculates the time-evolution unitary ``U(t)``  by contracting MPO layers of the ``W_{II}`` representation of ``U(dt)``

### Arguments
- `oWII` : an instance of the type [`WII`](@ref)
- `t` : final time

### Returns 
- ``U(t)`` as an instance of the `MPO` type

"""
function calc_Ut(oWII::WII, t::Float64)
    dt = oWII.dt;
    M = Int(round(t/dt));
    Ni = oWII.Ni;

    Idt = zeros(2,1,2,1);    
    Idt[:,1,:,1] = [1 0; 0 1];
    W1m = Idt; 
    Wim = Idt; 
    WNm = Idt;

    for m ∈ 1:M
        @tensor begin # phys up, aux left N-1, aux left N, phys down, aux rigt N-1, aux right N
            W1m[u, l1, l2, d, r1, r2] := W1m[u, l1, x, r1]*oWII.W1[x, l2, d, r2]; 
            Wim[u, l1, l2, d, r1, r2] := Wim[u, l1, x, r1]*oWII.Wi[x, l2, d, r2]; 
            WNm[u, l1, l2, d, r1, r2] := WNm[u, l1, x, r1]*oWII.WN[x, l2, d, r2]; 
        end
        W1m = reshape(W1m, (2, 1, 2, (Ni+1)^m));
        Wim = reshape(Wim, (2, (Ni+1)^m, 2, (Ni+1)^m));
        WNm = reshape(WNm, (2, (Ni+1)^m, 2, 1));
    end

    return MPO([W1m, Wim, WNm])
end

function calc_Wt(U_t::MPO{T}, ss_op, loc::Int) where {T}
    L = U_t.L;
    W_i = Vector{Array{T, 4}}();

    for i ∈ 1:L
        i != loc && (@tensor UdagWU[u, l1, l2, d, r1, r2] := conj(U_t.Wi[i][u, l1, x, r1])*U_t.Wi[i][x, l2, d, r2]);
        i == loc && (@tensor UdagWU[u, l1, l2, d, r1, r2] := conj(U_t.Wi[i][u, l1, x, r1])*ss_op[x, y]*U_t.Wi[i][y, l2, d, r2]);
        i != 1 && i != L && (push!(W_i, reshape(UdagWU, (2, U_t.D[i-1]^2, 2, U_t.D[i]^2))));
        i == 1 && (push!(W_i, reshape(UdagWU, (2, 1, 2, U_t.D[i]^2))));
        i == L && (push!(W_i, reshape(UdagWU, (2, U_t.D[i-1]^2, 2, 1))));
    end

    return MPO(W_i)
end

#==========================================================================================#
########## OLD: keep for debugging

#= 
"""
Calculates the matrix element ``F_{j;a_j,\\bar{a}_j}`` where ``a_j = {1,...,N_j}`` labels the bosonic auxiliary fields (``\\bar{a}_j`` the complex conjugate of the fields) and ``N_j`` is the (non-trivial) bond dimension of the MPO tensor at each site ``j``


# Arguments
- `i,j` : indices corresponding to ``a_j`` and ``\\bar{a}_j``
- `A,B,C,D` : operator valued matrices of Hamiltonian

# Returns
- A ``4d`` matrix (where ``d`` is the physical site dimension)

"""
function F_ij(i, j, A, B, C, D, dt) #! if i ≠ j, then Aᵢⱼ = 0??
    @variables M
    M = [
        dt*D       0         0     0;
        √dt*C[j]  dt*D       0     0;
        √dt*B[i]   0        dt*D   0;
        A[i,j]  √dt*B[i]  √dt*C[j] dt*D;
        ]; 
    
    return M
end


mF = F_ij(1, 1, A, B, C, D, 0.05)

########## WII matrix elements


function exp_F(F; order=20) #! largerst factorial supported without bignumber
    @variables Φ
    Φ = sum([F^n/factorial(n) for n ∈ 1:order])
    return Φ
end


function exp_F(i, j, A, B, C, D, dt; order=20) #! largest factorial supported without bignumber
    @variables F Φ
    F = [
        dt*D       0         0     0;
        √dt*C[j]  dt*D       0     0;
        √dt*B[i]   0        dt*D   0;
        A[i,j]  √dt*B[i]  √dt*C[j] dt*D;
        ]; 
    Φ = sum([F^n/factorial(n) for n ∈ 1:order])
    return Φ
end

 =#