################################################
# Directional CTM (10.1103/PhysRevB.80.094403) #
################################################

"""
    function update_environment!(unitcell::UnitCell, projectors::Projectors, simulation::Simulation)

    Performs CTM iterations till convergence of the environment. Convergence is verified by
    comparing eigenvalues of all environment tensors (`Full`) or just the corner tensors (`OnlyCorners`).

"""
function update_environment!(unitcell::UnitCell, projectors::Projectors, simulation::Simulation)

    ϵ = 1.0;
    i = 0

    do_ctmrg_iteration!(unitcell, projectors, simulation.ctm_Χ);
    while ϵ > simulation.tol_ctm
        i += 1;

        ref_unitcell = deepcopy(unitcell)
        do_ctmrg_iteration!(unitcell, projectors, simulation.ctm_Χ);
        ϵ = calculate_error_ctm(ref_unitcell, unitcell);

        @info "CTM iteration $i, convergence error = $(round(abs(ϵ), sigdigits = 4))";

        if i > simulation.max_ctm_steps
            @warn "!!! Maximum number of iterations reached !!!"
            simulation.conv_ctm_steps = i;
            simulation.ctm_error = ϵ;
            return ϵ
        end
    end

    simulation.conv_ctm_steps = i;
    simulation.ctm_error = ϵ;
end

function do_ctmrg_iteration!(
    unitcell::UnitCell,
    projectors::Projectors{EachMove}, Χ::Int64)

    Nx = unitcell.dims[1];
    Ny = unitcell.dims[2];


    #= Left move for every unit-cell tensor. Sweeps from left to right, column by column =#
    for i ∈ 1:Nx
        # 1) Calculate all projectors along the column, i.e. all P_(i, j+n) for fixed j+n
        for j ∈ 1:Ny
            calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), LEFT, Χ); # P_(i,j)
            #@info "Size T4 : ", size(unitcell.E[i,j].T[1])
            #@info "Size T4 alt: ", size(unitcell(Environment, CartesianIndex(i+1, j-1)).T[4])

        end

        # 2) Absorb column j+n tensors in environment tensors of all tensors (i,j) with fixed j and renormalize
        for j ∈ 1:Ny
            do_ctm_move!(unitcell, projectors, LEFT, CartesianIndex(i,j));
        end
    end


    #= Right move for every unit-cell tensor. Sweeps from left to right, column by column =#
    for i ∈ 1:Nx
        # 1) Calculate all projectors for the column, i.e. all P_(i, j+n) for fixed j+n
        for j ∈ 1:Ny
            calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), RIGHT, Χ); # P_(i,j)
        end

        # 2) Absorb column j-n tensors in environment tensors of all tensors (i,j) with fixed j and renormalize
        for j ∈ 1:Ny
            do_ctm_move!(unitcell, projectors, RIGHT, CartesianIndex(i,j));
        end
    end

   #= Up move for every unit-cell tensor. Sweeps from top to bottom, row by row =#
    for j ∈ 1:Ny
        # 1) Calculate all projectors for the row, i.e. all P_(i+n, j) for fixed i+n
        for i ∈ 1:Nx
            calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), UP, Χ); # P_(i,j)
        end

        # 2) Absorb row i+n tensors in environment tensors of all tensors (i,j) with fixed i and renormalize
        for i ∈ 1:Nx
            do_ctm_move!(unitcell, projectors, UP, CartesianIndex(i,j));
        end
    end


   #= Down move for every unit-cell tensor. Sweeps from top to bottom, row by row =#
    for j ∈ 1:Ny
        # 1) Calculate all projectors for the row, i.e. all P_(i+n, j) for fixed i+n
        for i ∈ 1:Nx
            calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), DOWN, Χ); # P_(i,j)
        end

        # 2) Absorb row i-n tensors in environment tensors of all tensors (i,j) with fixed i and renormalize
        for i ∈ 1:Nx
            do_ctm_move!(unitcell, projectors, DOWN, CartesianIndex(i,j));
        end
    end


end


function do_ctm_move!(unitcell::UnitCell, projectors::Projectors, direction::Direction, loc::CartesianIndex)
    E_loc = unitcell(Environment, loc);
    R_loc = unitcell(ReducedTensor, loc);


    if direction == LEFT

        """
        C4(x-1,y)  --  T1(x,y) --
            |             |


            |             |
        T4(x-1,y)  --   R(x,y) --
            |             |


            |             |
        C3(x-1,y)  --  T3(x,y) --
        """

        E_add = unitcell(Environment, loc + CartesianIndex(-1, 0));

        # Grow environment
        @tensor C4T1[re, de, dc] := E_add.C[4][α, de] * E_loc.T[1][re, α, dc]
        @tensor T4A[ue, uc, de, dc, rc] := E_add.T[4][ue, de, α] * R_loc.R[uc, rc, dc, α]
        @tensor C3T3[ue, uc, re] := E_add.C[3][ue, α] * E_loc.T[3][re, α, uc]

        T4A = reshape(T4A, (prod(size(T4A)[1:2]), prod(size(T4A)[3:4]), size(T4A, 5)));
        C4T1 = reshape(C4T1, (size(C4T1, 1), :));
        C3T3 = reshape(C3T3, (:, size(C3T3, 3)));

        # Renormalize

        P̃ = projectors(LEFT, loc)[1];
        P = projectors(LEFT, loc + CartesianIndex(0, -1))[2];

        C̃4 = C4T1 * P̃; #(r, d)
        C̃3 = P * C3T3; #(u, r)
        @tensor T̃4[ue, de, rc] := P[ue, α] * T4A[α, β, rc] * P̃[β, de];


        # Update tensors environment
        update_tensors!(unitcell, [C̃3, T̃4, C̃4], LEFT, loc);

    elseif direction == RIGHT

        """
        -- T1(x,y)  --    C1(x+1,y)
            |                 |

            |                 |
        -- R(x,y)   --    T2(x+1,y)
            |                 |

            |                 |
        -- T3(x,y)  --   C2(x+1,y)

        """

        E_add = unitcell(Environment, loc + CartesianIndex(1, 0));

        # Grow environment
        @tensor C1T1[de, dc, le] := E_add.C[1][de, α] * E_loc.T[1][α, le, dc];
        @tensor T2A[ue, uc, de, dc, lc] := E_add.T[2][ue, de, α] * R_loc.R[uc, α, dc, lc];
        @tensor C2T3[ue, uc, le] := E_add.C[2][ue, α] * E_loc.T[3][α, le, uc]; #! indices permuted

        C1T1 = transpose(reshape(C1T1, (:, size(C1T1, 3))));
        C2T3 = reshape(C2T3, (:, size(C2T3, 3)));
        T2A = reshape(T2A, (prod(size(T2A)[1:2]), prod(size(T2A)[3:4]), size(T2A, 5)));

        # Renormalize

        P̃ = projectors(RIGHT, loc)[1];
        P = projectors(RIGHT, loc + CartesianIndex(0, -1))[2];

        C̃1 = transpose(C1T1 * P̃); # (l, d) -> (d, l)
        C̃2 = P * C2T3; # (u, l)
        @tensor T̃2[ue, de, lc] := P[ue, α] * T2A[α, β, lc] * P̃[β, de];


        # Update tensors environment
        update_tensors!(unitcell, [C̃1, T̃2, C̃2], RIGHT, loc);

    elseif direction == UP

        """
         C4(x,y-1)--      --T1(x,y-1)--     --C1(x,y-1)
            |                   |                 |
            |                   |                 |
        T4(x,y)--          --R(x,y)--        --T2(x,y)
            |                   |                 |

        """

        E_add = unitcell(Environment, loc + CartesianIndex(0, -1));

        # Grow environment
        @tensor C4T4[re, rc, de] := E_add.C[4][re, α] * E_loc.T[4][α, de, rc];
        @tensor T1A[le, lc, re, rc, dc] := E_add.T[1][re, le, α] * R_loc.R[α, rc, dc, lc];
        @tensor C1T2[de, le, lc] := E_add.C[1][α, le] * E_loc.T[2][α, de, lc]; #! indices permuted

        C4T4 = transpose(reshape(C4T4, (:, size(C4T4, 3))));
        C1T2 = transpose(reshape(C1T2, (size(C1T2, 1), :)));
        T1A = reshape(T1A, (prod(size(T1A)[1:2]), prod(size(T1A)[3:4]), size(T1A, 5)));

        # Renormalize

        P̃ = projectors(UP, loc)[1];
        P = projectors(UP, loc + CartesianIndex(-1, 0))[2];

        C̃4 = transpose(C4T4 * P̃); #(d,r) -> (r,d)
        C̃1 = transpose(P * C1T2); #(l,d) -> (d,l)
        @tensor T̃1[re, le, dc] := P[le, α] * T1A[α, β, dc] * P̃[β, re];

        # Update tensors environment
        update_tensors!(unitcell, [C̃4, T̃1, C̃1], UP, loc);

    elseif direction == DOWN

        """
             |                 |                  |
          T4(x,y)--       --R(x,y)--        --T2(x,y)
             |                 |                  |
             |                 |                  |
          C3(x,y+1)--    --T3(x,y+1)--      --C2(x,y+1)

        """

        E_add = unitcell(Environment, loc + CartesianIndex(0, 1));

        # Grow environment
        @tensor C3T4[ue, re, rc] := E_add.C[3][α, re] * E_loc.T[4][ue, α, rc]; #! indices permuted
        @tensor T3A[le, lc, re, rc, uc] := E_add.T[3][re, le, α] * R_loc.R[uc, rc, α, lc];
        @tensor C2T2[ue, le, lc] := E_add.C[2][α, le] * E_loc.T[2][ue, α, lc];

        C3T4 = reshape(C3T4, (size(C3T4, 1), :));
        C2T2 = transpose(reshape(C2T2, (size(C2T2, 1), :)));
        T3A = reshape(T3A, (prod(size(T3A)[1:2]), prod(size(T3A)[3:4]), size(T3A, 5)));

        # Renormalize

        P̃ = projectors(DOWN, loc)[1];
        P = projectors(DOWN, loc + CartesianIndex(-1, 0))[2];

        C̃3 = C3T4 * P̃; #(u,r)
        C̃2 = transpose(P * C2T2); #(l,u) -> (u,l)
        @tensor T̃3[re, le, uc] := P[le, α] * T3A[α, β, uc] * P̃[β, re];

        # Update tensors environment
        update_tensors!(unitcell, [C̃2, T̃3, C̃3], DOWN, loc);
    end
end


function calc_projectors_ctmrg!(
    uc::UnitCell,
    projectors::Projectors,
    loc::CartesianIndex,
    direction::Direction,
    Χ::Int64)

    if direction == LEFT

        """
        C4(x-1,y-1) -- T1(x,y-1) --
            |              |
            |              |
        T4(x-1,y)   --  R(x,y) --
            |              |


            |              |
        T4(x-1,y+1)  --  R(x,y+1) --
            |              |
            |              |
        C3(x-1,y+2)  --  T3(x,y+2)--

        """

        # Build expanded corners
        C4 = uc(Environment, loc + CartesianIndex(-1, -1)).C[4];
        T1 = uc(Environment, loc  + CartesianIndex(0, -1)).T[1];
        T4 = uc(Environment, loc + CartesianIndex(-1, 0)).T[4];
        R = uc(ReducedTensor, loc).R;

        @tensor Q4[ure, urc, de, dc] := C4[α, δ] * T4[δ, de, γ] * T1[ure, α, β] * R[β, urc, dc, γ];

        C3 = uc(Environment, loc + CartesianIndex(-1, 2)).C[3];
        T3 = uc(Environment, loc + CartesianIndex(0, 2)).T[3];
        T4 = uc(Environment, loc + CartesianIndex(-1, 1)).T[4];
        R = uc(ReducedTensor, loc + CartesianIndex(0, 1)).R;

        @tensor Q3[ue, uc, dre, drc] := C3[α, δ] * T4[ue, α, β] * T3[dre, δ, γ] * R[uc, drc, γ, β];

        # Calculate half-system density matrix
        Q4 = reshape(Q4, (size(Q4, 1) * size(Q4, 2), :));
        Q3 = reshape(Q3, (size(Q3, 1) * size(Q3, 2), :));
        HL = Q4 * Q3;

        # Calculate projectors
        Χ = size(uc(Environment, loc).T[4], 2);

        U, Sinvsqrt, V, S = factorize_rho(HL, Χ)
        P̃ = Q3 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q4;
        projectors.Pl[loc] = [P̃, P];

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S;
        uc.E[loc].spectra[4] = S_f;

        @debug "Left renormalization" norm(Q4 * Q3 - Q4 * P̃ * P * Q3);



    elseif direction == RIGHT

        """
        -- T1(x,y-1)--   C1(x+1,y-1)
              |             |
              |             |
        -- R(x,y)   --  T2(x+1,y)
              |             |


              |             |
        -- R(x,y+1)  --  T2(x+1,y+1)
              |             |
              |             |
        -- T3(x,y+2) --  C2(x+1,y+2)

        """

        # Build expanded corners
        C1 = uc(Environment, loc + CartesianIndex(1, -1)).C[1];
        T1 = uc(Environment, loc + CartesianIndex(0, -1)).T[1];
        T2 = uc(Environment, loc + CartesianIndex(1, 0)).T[2];
        R = uc(ReducedTensor, loc).R;

        @tensor Q1[ule, ulc, de, dc] := C1[α, δ] * T2[α, de, β] * T1[δ, ule, γ] * R[γ, β, dc, ulc];

        C2 = uc(Environment, loc + CartesianIndex(1, 2)).C[2];
        T3 = uc(Environment, loc + CartesianIndex(0, 2)).T[3];
        T2 = uc(Environment, loc + CartesianIndex(1, 1)).T[2];

        R = uc(ReducedTensor, loc + CartesianIndex(1, 0)).R;

        @tensor Q2[ue, uc, dle, dlc] := C2[α, β] * T3[β, dle, γ] * T2[ue, α, δ] * R[uc, δ, γ, dlc];

        Q1 = reshape(Q1, (size(Q1, 1) * size(Q1, 2), :));
        Q2 = reshape(Q2, (size(Q2, 1) * size(Q2, 2), :));
        HR = Q1 * Q2;

        # Calculate projectors
        Χ = size(uc(Environment, loc).T[2], 2);

        U, Sinvsqrt, V, S = factorize_rho(HR, Χ)
        P̃ = Q2 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q1;

        projectors.Pr[loc] = [P̃, P];

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S;
        uc.E[loc].spectra[2] = S_f;

    elseif direction == UP

        """
         C4(x-1,y-1) --  T1(x,y-1)--     -- T1(x+1,y-1) --  C1(x+2,y-1)
            |                |                 |                |
            |                |                 |                |
         T4(x-1,y)   --    R(x,y) --     --  R(x+1,y)   --   T2(x+1,y)
             |               |                 |                |

        """

        # Build expanded corners
        C4 = uc(Environment, loc + CartesianIndex(-1, -1)).C[4]; #!
        T1 = uc(Environment, loc + CartesianIndex(0, -1)).T[1];
        T4 = uc(Environment, loc + CartesianIndex(-1, 0)).T[4];
        R = uc(ReducedTensor, loc).R;

        @tensor Q4[lde, ldc, re, rc] := C4[α, δ] * T1[re, α, β] * T4[δ, lde, γ] * R[β, rc, ldc, γ];


        C1 = uc(Environment, loc + CartesianIndex(2, -1)).C[1];
        T1 = uc(Environment, loc + CartesianIndex(1, -1)).T[1];
        T2 = uc(Environment, loc + CartesianIndex(2, 0)).T[2];
        R = uc(ReducedTensor, loc + CartesianIndex(0, 1)).R;

        @tensor Q1[le, lc, rde, rdc] := C1[α, δ] * T2[α, rde, β] * T1[δ, le, γ] * R[γ, β, rdc, lc];

        Q4 = reshape(Q4, (size(Q4, 1) * size(Q4, 2), :));
        Q1 = reshape(Q1, (size(Q1, 1) * size(Q1, 2), :));
        HU = Q4 * Q1;


        # Calculate projectors
        Χ = size(uc(Environment, loc).T[1], 1);

        U, Sinvsqrt, V, S = factorize_rho(HU, Χ);
        P̃ = Q1 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q4;

        projectors.Pu[loc] = [P̃, P];

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S;
        uc.E[loc].spectra[1] = S_f;

    elseif direction == DOWN

        """
            |              |                    |               |
        T4(x-1,y)   --  R(x,y)   --     --  R(x+1,y)   --    T2(x+2,y)
            |              |                    |               |
            |              |                    |               |
        C3(x-1,y+1) -- T3(x,y+1) --     -- T3(x+1,y+1) --  C2(x+2,y+1)

        """


        C3 = uc(Environment, loc + CartesianIndex(-1, 1)).C[3];
        T3 = uc(Environment, loc + CartesianIndex(0, 1)).T[3];
        T4 = uc(Environment, loc + CartesianIndex(-1, 0)).T[4];
        R = uc(ReducedTensor, loc).R;


        @tensor Q3[lue, luc, re, rc] := C3[α, δ] * T4[lue, α, β] * T3[re, δ, γ] * R[luc, rc, γ, β]

        C2 = uc(Environment, loc + CartesianIndex(2, 1)).C[2];
        T3 = uc(Environment, loc + CartesianIndex(1, 1)).T[3];
        T2 = uc(Environment, loc + CartesianIndex(2, 0)).T[2];

        R = uc(ReducedTensor, loc + CartesianIndex(0, 1)).R;

        @tensor Q2[le, lc, rue, ruc] := C2[α, β] * T3[β, le, γ] * T2[rue, α, δ] * R[ruc, δ, γ, lc];

        Q2 = reshape(Q2, (size(Q2, 1) * size(Q2, 2), :));
        Q3 = reshape(Q3, (size(Q3, 1) * size(Q3, 2), :));
        HD = Q3 * Q2;


        Χ = size(uc(Environment, loc).T[3], 1);
        U, Sinvsqrt, V, S = factorize_rho(HD, Χ)
        P̃ = Q2 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q3;

        projectors.Pd[loc] = [P̃, P];

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S;
        uc.E[loc].spectra[3] = S_f;

    end


end


function update_tensors!(uc::UnitCell, tensors::Vector{T}, direction::Direction, loc::CartesianIndex; normalize::Bool = true) where {T<:AbstractArray}

    #@info "Updating tensors move $direction and loc $loc"

    if direction == UP
        pos = [4, 1, 1];
    elseif direction == RIGHT
        pos = [1, 2, 2];
    elseif direction == DOWN
        pos = [2, 3, 3];
    elseif direction == LEFT
        pos = [3, 4, 4];
    end

    if normalize == true
        uc.E[loc].C[pos[1]] = tensors[1]/opnorm(tensors[1]);
        uc.E[loc].T[pos[2]] = tensors[2]/norm(tensors[2]);
        uc.E[loc].C[pos[3]] = tensors[3]/opnorm(tensors[3]);

    else
        uc.E[loc].C[pos[1]] = collect(tensors[1]);
        uc.E[loc].T[pos[2]] = collect(tensors[2]);
        uc.E[loc].C[pos[3]] = collect(tensors[3]);
    end

end






####################
# Helper functions #
####################

function factorize_rho_sym(rho::Array{T,2}, Χ::Int, symmetry::LatticeSymmetry) where {T}
    if symmetry == XY || symmetry == C4
        #@info "Is hermitian $(rho ≈ rho')"
        Λ, U = eigen(rho, sortby= x -> -1 * real(x))
        Winvsqrt = diagm(Λ[1:Χ].^(-1/2));
        return U[:, 1:Χ], Winvsqrt, U[:, 1:Χ]
    else
        U, S, V = svd(rho);
        Winvsqrt = diagm(S[1:Χ].^(-1/2));
        return U[:, 1:Χ], Winvsqrt, V[:, 1:Χ]
    end
end

function factorize_rho(rho::Array{T,2}, Χ::Int) where {T}

    U, S, V = svd(rho);
    Χ > length(S) && (Χ = length(S);)
    Winvsqrt = diagm(S[1:Χ].^(-1/2));

    return U[:, 1:Χ], Winvsqrt, V[:, 1:Χ], S[1:Χ]/maximum(S[1:Χ])
    #return U[:, 1:Χ], Winvsqrt, V[:, 1:Χ]
end


function calculate_error_ctm(rc::UnitCell, uc::UnitCell)
    Ni = rc.dims[1];
    Nj = rc.dims[2];

    ϵ = 0.0;
    for i ∈ Ni, j ∈ Nj
        ϵ += sum([sum(abs.(uc.E[i, j].spectra[n] - rc.E[i, j].spectra[n])) for n ∈ 1:4])
    end

    return ϵ
end

function cutoff(S::Vector{Float64}, χmax::Int, ϵ::Float64)
    Χ = length(S);
    sum_disc = 0.0;
    n = 0;
    if ϵ != 0.0
        while sum_disc < ϵ && n < Χ
            sum_disc += S[end - n]^2
            n += 1;
        end
        n += -1; # to cancel the last step
    end
    Χkeep = Χ - n;
    return min(Χkeep, χmax)
end
