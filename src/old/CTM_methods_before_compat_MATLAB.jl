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

    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];


    #= Left move for every unit-cell tensor. Sweeps from left to right, column by column =#
    for i ∈ 1:Ni
        # 1) Calculate all projectors along the column, i.e. all P_(i, j+n) for fixed j+n
        for j ∈ 1:Nj
            calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), LEFT, Χ); # P_(i,j)
            #@info "Size T4 : ", size(unitcell.E[i,j].T[1])
            #@info "Size T4 alt: ", size(unitcell(Environment, CartesianIndex(i+1, j-1)).T[4])

        end

        # 2) Absorb column j+n tensors in environment tensors of all tensors (i,j) with fixed j and renormalize
        for j ∈ 1:Nj
            do_ctm_move!(unitcell, projectors, LEFT, CartesianIndex(i,j));
        end
    end


    #= Right move for every unit-cell tensor. Sweeps from left to right, column by column =#
    for i ∈ 1:Ni
        # 1) Calculate all projectors for the column, i.e. all P_(i, j+n) for fixed j+n
        for j ∈ 1:Nj
            calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), RIGHT, Χ); # P_(i,j)
        end

        # 2) Absorb column j-n tensors in environment tensors of all tensors (i,j) with fixed j and renormalize
        for j ∈ 1:Nj
            do_ctm_move!(unitcell, projectors, RIGHT, CartesianIndex(i,j));
        end
    end

   #= Up move for every unit-cell tensor. Sweeps from top to bottom, row by row =#
    for j ∈ 1:Nj
        # 1) Calculate all projectors for the row, i.e. all P_(i+n, j) for fixed i+n
        for i ∈ 1:Ni
            calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), UP, Χ); # P_(i,j)
        end

        # 2) Absorb row i+n tensors in environment tensors of all tensors (i,j) with fixed i and renormalize
        for i ∈ 1:Ni
            do_ctm_move!(unitcell, projectors, UP, CartesianIndex(i,j));
        end
    end


   #= Down move for every unit-cell tensor. Sweeps from top to bottom, row by row =#
    for j ∈ 1:Nj
        # 1) Calculate all projectors for the row, i.e. all P_(i+n, j) for fixed i+n
        for i ∈ 1:Ni
            calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), DOWN, Χ); # P_(i,j)
        end

        # 2) Absorb row i-n tensors in environment tensors of all tensors (i,j) with fixed i and renormalize
        for i ∈ 1:Ni
            do_ctm_move!(unitcell, projectors, DOWN, CartesianIndex(i,j));
        end
    end


end


function do_ctm_move!(unitcell::UnitCell, projectors::Projectors, direction::Direction, ij::CartesianIndex)
    E_loc = unitcell(Environment, ij);
    R_loc = unitcell(ReducedTensor, ij);

    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];

    if direction == LEFT

        """
        C̃4 = C4(i,j-1)  --  T1(i,j) --
                 |             |


                 |             |
        T̃4 = T4(i,j-1)  --   R(i,j) --
                 |             |


                 |             |
        C̃3 = C3(i,j-1)  --  T3(i,j) --
        """

        E_add = unitcell(Environment, ij + CartesianIndex(-1, 0));

        # Grow environment
        @tensor C4T1[re, de, dc] := E_add.C[4][α, de] * E_loc.T[1][re, α, dc]
        @tensor T4A[ue, uc, de, dc, rc] := E_add.T[4][ue, de, α] * R_loc.R[uc, rc, dc, α]
        @tensor C3T3[ue, uc, re] := E_add.C[3][ue, α] * E_loc.T[3][re, α, uc]

        T4A = reshape(T4A, (prod(size(T4A)[1:2]), prod(size(T4A)[3:4]), size(T4A, 5)));
        C4T1 = reshape(C4T1, (size(C4T1, 1), :));
        C3T3 = reshape(C3T3, (:, size(C3T3, 3)));

        # Renormalize
        ij[2] == 1 ? i_1j = CartesianIndex(ij[1], Nj) : i_1j = CartesianIndex(ij[1], ij[2] - 1);
        #ij[1] == 1 ? i_1j = CartesianIndex(Ni, ij[2]) : i_1j = CartesianIndex(ij[1] - 1, ij[2]);

        #= @info "Loc, $ij"
        @info "Locprev , $i_1j" =#

        C̃4 = C4T1 * projectors.Pl[ij][1]; #(r, d)
        C̃3 = projectors.Pl[i_1j][2] * C3T3; #(u, r)
        @tensor T̃4[ue, de, rc] := projectors.Pl[i_1j][2][ue, α] * T4A[α, β, rc] * projectors.Pl[ij][1][β, de];

        _, sP, _ = svd(projectors.Pl[i_1j][2]);
        _, sPt, _ = svd(projectors.Pl[ij][1]);

        #= @info "##########Left move############"
        @info "Size T4 before", size(E_loc.T[4])
        @info "Size Pt", size(projectors.Pl[ij][1])
        @info "Size P", size(projectors.Pl[i_1j][2])
        @info "Size T4 after", size(T̃4) =#

        #=@info "SVS Pt", round.(sPt[1:5]/maximum(sPt), sigdigits=5)
        @info "SVS P", round.(sP[1:5]/maximum(sP), sigdigits=5) =#

        # Update tensors environment
        update_tensors!(unitcell, [C̃3, T̃4, C̃4], LEFT, ij);

    elseif direction == RIGHT

        """
        -- T1(i,j)  --    C1(i,j+1)
            |                 |

            |                 |
        -- R(i,j)   --    T2(i,j+1)
            |                 |

            |                 |
        -- T3(i,j)  --   C2(i,j+1)

        """

        E_add = unitcell(Environment, ij + CartesianIndex(1, 0));

        # Grow environment
        @tensor C1T1[de, dc, le] := E_add.C[1][de, α] * E_loc.T[1][α, le, dc];
        @tensor T2A[ue, uc, de, dc, lc] := E_add.T[2][ue, de, α] * R_loc.R[uc, α, dc, lc];
        @tensor C2T3[ue, uc, le] := E_add.C[2][ue, α] * E_loc.T[3][α, le, uc]; #! indices permuted

        C1T1 = transpose(reshape(C1T1, (:, size(C1T1, 3))));
        C2T3 = reshape(C2T3, (:, size(C2T3, 3)));
        T2A = reshape(T2A, (prod(size(T2A)[1:2]), prod(size(T2A)[3:4]), size(T2A, 5)));

        # Renormalize
        #ij[1] == 1 ? i_1j = CartesianIndex(Ni, ij[2]) : i_1j = CartesianIndex(ij[1] - 1, ij[2]);
        ij[2] == 1 ? i_1j = CartesianIndex(ij[1], Nj) : i_1j = CartesianIndex(ij[1], ij[2]-1);

        C̃1 = transpose(C1T1 * projectors.Pr[ij][1]); # (l, d) -> (d, l)
        C̃2 = projectors.Pr[i_1j][2] * C2T3; # (u, l)
        @tensor T̃2[ue, de, lc] := projectors.Pr[i_1j][2][ue, α] * T2A[α, β, lc] * projectors.Pr[ij][1][β, de];


        #= @info "##########Right move############"
        @info "Loc right", ij
        @info "Size T2 before", size(E_loc.T[2])
        @info "Size Pt", size(projectors.Pr[ij][1])
        @info "Size P", size(projectors.Pr[i_1j][2])
        @info "Size T2 after", size(T̃2) =#


        # Update tensors environment
        update_tensors!(unitcell, [C̃1, T̃2, C̃2], RIGHT, ij);

    elseif direction == UP

        """
         C4(i-1,j)--      --T1(i-1,j)--     --C1(i-1,j)
            |                   |                 |
            |                   |                 |
        T4(i,j)--          --R(i,j)--        --T2(i,j)
            |                   |                 |

        """

        E_add = unitcell(Environment, ij + CartesianIndex(0, -1));

        # Grow environment
        @tensor C4T4[re, rc, de] := E_add.C[4][re, α] * E_loc.T[4][α, de, rc];
        @tensor T1A[le, lc, re, rc, dc] := E_add.T[1][re, le, α] * R_loc.R[α, rc, dc, lc];
        @tensor C1T2[de, le, lc] := E_add.C[1][α, le] * E_loc.T[2][α, de, lc]; #! indices permuted

        C4T4 = transpose(reshape(C4T4, (:, size(C4T4, 3))));
        C1T2 = transpose(reshape(C1T2, (size(C1T2, 1), :)));
        T1A = reshape(T1A, (prod(size(T1A)[1:2]), prod(size(T1A)[3:4]), size(T1A, 5)));

        # Renormalize
        #ij[2] == 1 ? ij_1 = CartesianIndex(ij[1], Nj) : ij_1 = CartesianIndex(ij[1], ij[2]-1);
        ij[1] == 1 ? ij_1 = CartesianIndex(Ni, ij[2]) : ij_1 = CartesianIndex(ij[1]-1, ij[2]);

        C̃4 = transpose(C4T4 * projectors.Pu[ij][1]); #(d,r) -> (r,d)
        C̃1 = transpose(projectors.Pu[ij_1][2] * C1T2); #(l,d) -> (d,l)
        @tensor T̃1[re, le, dc] := projectors.Pu[ij_1][2][le, α] * T1A[α, β, dc] * projectors.Pu[ij][1][β, re];


        #= @info "##########up move############"
        @info "Loc up", ij
        @info "Size T1 before", size(E_loc.T[1])
        @info "Size Pt", size(projectors.Pu[ij][1])
        @info "Size P", size(projectors.Pu[ij_1][2])
        @info "Size T1 after", size(T̃1) =#

        # Update tensors environment
        update_tensors!(unitcell, [C̃4, T̃1, C̃1], UP, ij);

    elseif direction == DOWN

        """
             |                 |                  |
          T4(i,j)--       --R(i,j)--        --T2(i,j)
             |                 |                  |
             |                 |                  |
          C3(i+1,j)--    --T3(i+1,j)--      --C2(i+1,j)

        """

        E_add = unitcell(Environment, ij + CartesianIndex(0, 1));

        # Grow environment
        @tensor C3T4[ue, re, rc] := E_add.C[3][α, re] * E_loc.T[4][ue, α, rc]; #! indices permuted
        @tensor T3A[le, lc, re, rc, uc] := E_add.T[3][re, le, α] * R_loc.R[uc, rc, α, lc];
        @tensor C2T2[ue, le, lc] := E_add.C[2][α, le] * E_loc.T[2][ue, α, lc];

        C3T4 = reshape(C3T4, (size(C3T4, 1), :));
        C2T2 = transpose(reshape(C2T2, (size(C2T2, 1), :)));
        T3A = reshape(T3A, (prod(size(T3A)[1:2]), prod(size(T3A)[3:4]), size(T3A, 5)));

        # Renormalize
        #ij[2] == 1 ? ij_1 = CartesianIndex(ij[1], Nj) : ij_1 = CartesianIndex(ij[1], ij[2]-1);
        ij[1] == 1 ? ij_1 = CartesianIndex(Ni, ij[2]) : ij_1 = CartesianIndex(ij[1]-1, ij[2]);

        C̃3 = C3T4 * projectors.Pd[ij][1]; #(u,r)
        C̃2 = transpose(projectors.Pd[ij_1][2] * C2T2); #(l,u) -> (u,l)
        @tensor T̃3[re, le, uc] := projectors.Pd[ij_1][2][le, α] * T3A[α, β, uc] * projectors.Pd[ij][1][β, re];

        #= @info "##########down move############"
        @info "Loc down", ij
        @info "Size T3 before", size(E_loc.T[3])
        @info "Size Pt", size(projectors.Pd[ij][1])
        @info "Size P", size(projectors.Pd[ij_1][2])
        @info "Size T3 after", size(T̃3) =#

        # Update tensors environment
        update_tensors!(unitcell, [C̃2, T̃3, C̃3], DOWN, ij);
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
        C4(i-1,j-1) -- T1(i-1,j) --
            |              |
            |              |
        T4(i,j-1)   --  R(i,j) --
            |              |


            |              |
        T4(i+1,j-1)  --  R(i+1,j) --
            |              |
            |              |
        C3(i+2,j-1)  --  T3(i+2,j)--

        """

        # Build expanded corners

        #C4 = uc(Environment, loc + CartesianIndex(-1, -1)).C[4];
        #T1 = uc(Environment, loc  + CartesianIndex(-1, 0)).T[1];
        #T4 = uc(Environment, loc + CartesianIndex(0, -1)).T[4];

        C4 = uc(Environment, loc + CartesianIndex(-1, -1)).C[4];
        T1 = uc(Environment, loc  + CartesianIndex(0, -1)).T[1];
        T4 = uc(Environment, loc + CartesianIndex(-1, 0)).T[4];
        R = uc(ReducedTensor, loc).R;

        Χ = size(uc(Environment, loc).T[4], 2);

        @tensor Q4[ure, urc, de, dc] := C4[α, δ] * T4[δ, de, γ] * T1[ure, α, β] * R[β, urc, dc, γ];

        #C3 = uc(Environment, loc + CartesianIndex(2, -1)).C[3];
        #T3 = uc(Environment, loc + CartesianIndex(2, 0)).T[3];
        #T4 = uc(Environment, loc + CartesianIndex(1, -1)).T[4];
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
        #U, Sinvsqrt, V = factorize_rho(HL, Χ)

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
        -- T1(i-1,j)--   C1(i-1,j+1)
              |             |
              |             |
        -- R(i,j)   --  T2(i,j+1)
              |             |


              |             |
        -- R(i+1,j)  --  T2(i+1,j+1)
              |             |
              |             |
        -- T3(i+2,j) --  C2(i+2,j+1)

        """

        #= C1 = uc(Environment, loc + CartesianIndex(-1, 1)).C[1];
        T1 = uc(Environment, loc + CartesianIndex(-1, 0)).T[1];
        T2 = uc(Environment, loc + CartesianIndex(0, 1)).T[2]; =#

        C1 = uc(Environment, loc + CartesianIndex(1, -1)).C[1];
        T1 = uc(Environment, loc + CartesianIndex(0, -1)).T[1];
        T2 = uc(Environment, loc + CartesianIndex(1, 0)).T[2];
        R = uc(ReducedTensor, loc).R;

        Χ = size(uc(Environment, loc).T[2], 2);

        @tensor Q1[ule, ulc, de, dc] := C1[α, δ] * T2[α, de, β] * T1[δ, ule, γ] * R[γ, β, dc, ulc];

        #= C2 = uc(Environment, loc + CartesianIndex(2, 1)).C[2];
        T3 = uc(Environment, loc + CartesianIndex(2, 0)).T[3];
        T2 = uc(Environment, loc + CartesianIndex(1, 1)).T[2]; =#

        C2 = uc(Environment, loc + CartesianIndex(1, 2)).C[2];
        T3 = uc(Environment, loc + CartesianIndex(0, 2)).T[3];
        T2 = uc(Environment, loc + CartesianIndex(1, 1)).T[2];

        R = uc(ReducedTensor, loc + CartesianIndex(1, 0)).R;

        @tensor Q2[ue, uc, dle, dlc] := C2[α, β] * T3[β, dle, γ] * T2[ue, α, δ] * R[uc, δ, γ, dlc];

        Q1 = reshape(Q1, (size(Q1, 1) * size(Q1, 2), :));
        Q2 = reshape(Q2, (size(Q2, 1) * size(Q2, 2), :));
        HR = Q1 * Q2;

        U, Sinvsqrt, V, S = factorize_rho(HR, Χ)
        P̃ = Q2 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q1;

        projectors.Pr[loc] = [P̃, P];

        #@info "Loc proj right", loc
        #@info "Size Pt", size(P̃)

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S;
        uc.E[loc].spectra[2] = S_f;

    elseif direction == UP

        """
         C4(i-1,j-1) --  T1(i-1,j) --
             |                |
             |                |
         T4(i,j-1)   --    R(i,j)  --
             |                |


        -- T1(i-1,j+1) --  C1(i-1,j+2)
              |              |
              |              |
        -- R(i,j+1)    --  T2(i,j+2)
              |              |

        """

        #= C4 = uc(Environment, loc + CartesianIndex(-1, -1)).C[4]; #!
        T1 = uc(Environment, loc + CartesianIndex(-1, 0)).T[1];
        T4 = uc(Environment, loc + CartesianIndex(0, -1)).T[4]; =#

        C4 = uc(Environment, loc + CartesianIndex(-1, -1)).C[4]; #!
        T1 = uc(Environment, loc + CartesianIndex(0, -1)).T[1];
        T4 = uc(Environment, loc + CartesianIndex(-1, 0)).T[4];

        R = uc(ReducedTensor, loc).R;

        Χ = size(uc(Environment, loc).T[1], 1);


        @tensor Q4[lde, ldc, re, rc] := C4[α, δ] * T1[re, α, β] * T4[δ, lde, γ] * R[β, rc, ldc, γ];

        #= C1 = uc(Environment, loc + CartesianIndex(-1, 2)).C[1];
        T1 = uc(Environment, loc + CartesianIndex(-1, 1)).T[1];
        T2 = uc(Environment, loc + CartesianIndex(0, 2)).T[2]; =#

        C1 = uc(Environment, loc + CartesianIndex(2, -1)).C[1];
        T1 = uc(Environment, loc + CartesianIndex(1, -1)).T[1];
        T2 = uc(Environment, loc + CartesianIndex(2, 0)).T[2];

        R = uc(ReducedTensor, loc + CartesianIndex(0, 1)).R;

        @tensor Q1[le, lc, rde, rdc] := C1[α, δ] * T2[α, rde, β] * T1[δ, le, γ] * R[γ, β, rdc, lc];

        Q4 = reshape(Q4, (size(Q4, 1) * size(Q4, 2), :));
        Q1 = reshape(Q1, (size(Q1, 1) * size(Q1, 2), :));
        HU = Q4 * Q1;

        U, Sinvsqrt, V, S = factorize_rho(HU, Χ)
        P̃ = Q1 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q4;

        projectors.Pu[loc] = [P̃, P];

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S;
        uc.E[loc].spectra[1] = S_f;

    elseif direction == DOWN

        """
            |              |
        T4(i,j-1)   --  R(i,j)   --
            |              |
            |              |
        C3(i+1,j-1) -- T3(i+1,j) --


                |               |
        --  R(i,j+1)   --    T2(i,j+2)
                |               |
                |               |
        -- T3(i+1,j+1) --  C2(i+1,j+2)

        """

        #= C3 = uc(Environment, loc + CartesianIndex(1, -1)).C[3];
        T3 = uc(Environment, loc + CartesianIndex(1, 0)).T[3];
        T4 = uc(Environment, loc + CartesianIndex(0, -1)).T[4]; =#

        C3 = uc(Environment, loc + CartesianIndex(-1, 1)).C[3];
        T3 = uc(Environment, loc + CartesianIndex(0, 1)).T[3];
        T4 = uc(Environment, loc + CartesianIndex(-1, 0)).T[4];
        R = uc(ReducedTensor, loc).R;

        Χ = size(uc(Environment, loc).T[3], 1);

        @tensor Q3[lue, luc, re, rc] := C3[α, δ] * T4[lue, α, β] * T3[re, δ, γ] * R[luc, rc, γ, β]

        #= C2 = uc(Environment, loc + CartesianIndex(1, 2)).C[2];
        T3 = uc(Environment, loc + CartesianIndex(1, 1)).T[3];
        T2 = uc(Environment, loc + CartesianIndex(0, 2)).T[2]; =#

        C2 = uc(Environment, loc + CartesianIndex(2, 1)).C[2];
        T3 = uc(Environment, loc + CartesianIndex(1, 1)).T[3];
        T2 = uc(Environment, loc + CartesianIndex(2, 0)).T[2];

        R = uc(ReducedTensor, loc + CartesianIndex(0, 1)).R;

        @tensor Q2[le, lc, rue, ruc] := C2[α, β] * T3[β, le, γ] * T2[rue, α, δ] * R[ruc, δ, γ, lc];

        Q2 = reshape(Q2, (size(Q2, 1) * size(Q2, 2), :));
        Q3 = reshape(Q3, (size(Q3, 1) * size(Q3, 2), :));
        HD = Q3 * Q2;

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


function calculate_error_ctm_v2(uc::UnitCell)
    unique_tensors = unique(uc.pattern);
    coord_unique = [findfirst(t -> t == type_tensor, uc.pattern) for type_tensor ∈ unique_tensors];

    Χ = uc.E[1, 1].Χ;
    ϵ = 0.0;
    ϵs = zeros(4, length(coord_unique));
    i = 0;
    for coord ∈ coord_unique
        i += 1;
        for n ∈ 1:4
            # Calculate error
            S_ref = uc.E[coord].spectra[n];
            _, S_new, _ = svd(uc.E[coord].C[n]);
            ϵ += sum(abs.(S_new[1:length(S_ref)]/maximum(abs.(S_new)) - S_ref/maximum(abs.(S_ref))))

            # Update spectra
            uc.E[coord].spectra[n] = copy(S_new);

            #! Diag
            ϵs[n, i] = sum(abs.(S_new[1:length(S_ref)] - S_ref))
        end
    end

    return ϵ/(4*length(coord_unique) * Χ), ϵs
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



##################
# Former methods #
##################


function do_ctmrg_iteration_v2!(
    unitcell::UnitCell,
    projectors::Projectors{EachMove}, Χ::Int64)

    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];


    #= Left move for every unit-cell tensor. Sweeps from left to right, column by column =#
    for j ∈ 1:Nj
        for n ∈ 0:Nj-1

            # 1) Calculate all projectors along the column, i.e. all P_(i, j+n) for fixed j+n
            for i ∈ 1:Ni
                S = calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), n, LEFT, Χ); # P_(i,j)
                @info (i,j,n)
                @info S[1:5]
                unitcell.E[i, j].spectra[1] = S;
            end

            # 2) Absorb column j+n tensors in environment tensors of all tensors (i,j) with fixed j and renormalize
            for i ∈ 1:Ni
                do_ctm_move!(unitcell, projectors, LEFT, CartesianIndex(i,j), n);
            end
        end
    end


    #= Right move for every unit-cell tensor. Sweeps from left to right, column by column =#
    for j ∈ 1:Nj
        for n ∈ 0:Nj-1

            # 1) Calculate all projectors for the column, i.e. all P_(i, j+n) for fixed j+n
            for i ∈ 1:Ni
                S = calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), n, RIGHT, Χ); # P_(i,j)
                unitcell.E[i, j].spectra[2] = S;
            end

            # 2) Absorb column j-n tensors in environment tensors of all tensors (i,j) with fixed j and renormalize
            for i ∈ 1:Ni
                do_ctm_move!(unitcell, projectors, RIGHT, CartesianIndex(i,j), n);
            end
        end
    end

    #= Up move for every unit-cell tensor. Sweeps from top to bottom, row by row =#
    for i ∈ 1:Ni
        for n ∈ 0:Ni-1

            # 1) Calculate all projectors for the row, i.e. all P_(i+n, j) for fixed i+n
            for j ∈ 1:Nj
                S = calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), n, UP, Χ); # P_(i,j)
                unitcell.E[i, j].spectra[3] = S;

            end

            # 2) Absorb row i+n tensors in environment tensors of all tensors (i,j) with fixed i and renormalize
            for j ∈ 1:Nj
                do_ctm_move!(unitcell, projectors, UP, CartesianIndex(i,j), n);
            end
        end
    end


    #= Down move for every unit-cell tensor. Sweeps from top to bottom, row by row =#
    for i ∈ 1:Ni
        for n ∈ 0:Ni-1

            # 1) Calculate all projectors for the row, i.e. all P_(i+n, j) for fixed i+n
            for j ∈ 1:Nj
                S = calc_projectors_ctmrg!(unitcell, projectors, CartesianIndex(i,j), n, DOWN, Χ); # P_(i,j)
                unitcell.E[i, j].spectra[4] = S;

            end

            # 2) Absorb row i-n tensors in environment tensors of all tensors (i,j) with fixed i and renormalize
            for j ∈ 1:Nj
                do_ctm_move!(unitcell, projectors, DOWN, CartesianIndex(i,j), n);
            end
        end
    end


end

"""
    function do_ctm_iteration!(
    unitcell::UnitCell,
    environment::Environment,
    projectors::Projectors{T}) where {T<:Renormalization}

    Performs a single CTM iteration, i.e. up, down, left and right moves, for arbitrary size rectangular unit-cells

### Arguments
### Returns
"""

function do_ctm_iteration!(
    unitcell::UnitCell,
    projectors::Projectors{EachMove})

    # u,d,l,r: direction of bond. e,c: environment or cell bond. E.g: bond re corresponds to a right-environment bond
    # greek letters for contracted symbols

    Χ = unitcell.E[1,1].Χ;
    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];

    for i ∈ 1:Ni, j ∈ 1:Nj

        #= Left move. Absorbs unit-cell tensors and environment tensors of the whole column, row by row.=#
        for n ∈ 0:Nj - 1
            pos_left = [i, mod(j + n - 1, Nj) + 1];

            # Renormalize
            calc_projectors!(unitcell, projectors, [i, j], pos_left, LEFT, Χ);
            dctm_move!(unitcell, projectors, LEFT, [i, j], pos_left)
        end

        #= Right move. Absorbs unit-cell tensors and environment tensors of the whole column, row by row =#
        for n ∈ 0:Nj - 1
            pos_right = [i, mod(j - n - 1, Nj) + 1];

            # Renormalize
            calc_projectors!(unitcell, projectors, [i, j], pos_right, RIGHT, Χ)
            dctm_move!(unitcell, projectors, RIGHT, [i, j], pos_right)
        end

        #= Up move. Absorbs unit-cell tensors and environment tensors of the whole row, column by column =#
        for n ∈ 0:Ni - 1
            #@info "Up move, step $(n+1)"
            pos_up = [mod(i + n - 1, Ni) + 1, j];

            # Renormalize
            calc_projectors!(unitcell, projectors, [i, j], pos_up, UP, Χ)
            dctm_move!(unitcell, projectors, UP, [i, j], pos_up)
        end

        #= Down move. Absorbs unit-cell tensors and environment tensors of the whole row, column by column =#
        for n ∈ 0:Ni - 1
            #@info "Down move, step $(n+1)"
            pos_down = [mod(i - n - 1, Ni) + 1, j];

            # Renormalize
            calc_projectors!(unitcell, projectors, [i, j], pos_down, DOWN, Χ)
            dctm_move!(unitcell, projectors, DOWN, [i, j], pos_down)
        end
    end
end


function do_ctm_iteration!(
    unitcell::UnitCell,
    projectors::Projectors{Start})

    # u,d,l,r: direction of bond. e,c: environment or cell bond. E.g: bond re corresponds to a right-environment bond
    # greek letters for contracted symbols

    Χ = unitcell.E[1,1].Χ;
    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];

    for i ∈ 1:Ni, j ∈ 1:Nj

        #Nlong = max(Ni, Nj);

        for n ∈ 0:max(Ni, Nj) - 1
            # Position in unit-cell to be absorbed
            pos_left = [i, mod(j + n - 1, Nj) + 1];
            pos_right = [i, mod(j - n - 1, Nj) + 1];
            pos_up = [mod(i + n - 1, Ni) + 1, j];
            pos_down = [mod(i - n - 1, Ni) + 1, j];


            n < Nj && calc_projectors!(unitcell, projectors, [i, j], pos_left, LEFT, Χ);
            n < Nj && calc_projectors!(unitcell, projectors, [i, j], pos_right, RIGHT, Χ)
            n < Ni && calc_projectors!(unitcell, projectors, [i, j], pos_up, UP, Χ)
            n < Ni && calc_projectors!(unitcell, projectors, [i, j], pos_down, DOWN, Χ)

            n < Nj && dctm_move!(unitcell, projectors, LEFT, [i, j], pos_left)
            n < Nj && dctm_move!(unitcell, projectors, RIGHT, [i, j], pos_right)
            n < Ni && dctm_move!(unitcell, projectors, UP, [i, j], pos_up)
            n < Ni && dctm_move!(unitcell, projectors, DOWN, [i, j], pos_down)

        end

    end
end

function do_ctm_iteration!(
    unitcell::UnitCell,
    projectors::Projectors{EachMoveCirc})

    # u,d,l,r: direction of bond. e,c: environment or cell bond. E.g: bond re corresponds to a right-environment bond
    # greek letters for contracted symbols

    Χ = unitcell.E[1,1].Χ;
    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];

    for i ∈ 1:Ni, j ∈ 1:Nj

        #Nlong = max(Ni, Nj);

        for n ∈ 0:max(Ni, Nj) - 1
            # Position in unit-cell to be absorbed
            pos_left = [i, mod(j + n - 1, Nj) + 1];
            pos_right = [i, mod(j - n - 1, Nj) + 1];
            pos_up = [mod(i + n - 1, Ni) + 1, j];
            pos_down = [mod(i - n - 1, Ni) + 1, j];

            if n < Nj
                calc_projectors!(unitcell, projectors, [i, j], pos_left, LEFT, Χ);
                dctm_move!(unitcell, projectors, LEFT, [i, j], pos_left);
                calc_projectors!(unitcell, projectors, [i, j], pos_right, RIGHT, Χ);
                dctm_move!(unitcell, projectors, RIGHT, [i, j], pos_right);
            end

            if n < Ni
                calc_projectors!(unitcell, projectors, [i, j], pos_up, UP, Χ)
                dctm_move!(unitcell, projectors, UP, [i, j], pos_up)
                calc_projectors!(unitcell, projectors, [i, j], pos_down, DOWN, Χ)
                dctm_move!(unitcell, projectors, DOWN, [i, j], pos_down)
            end

        end

    end
end

"""
    function calc_projectors!(
    uc::UnitCell{T},
    projectors::Projectors{HalfSystem}, loc::Tuple{Int64, Int64}, direction::String, step::Int64,
    Χ::Int64;
    kwargs...)

    Calculates environment projectors using two corner tensors without the intermediate QR factorization,
    following 10.1103/PhysRevB.80.094403.

### Arguments
### Returns
- P and P̃
### Notes
- the second leg of the projector corresponds to the auxiliary bond of the unit-cell
"""
function calc_projectors!(
    uc::UnitCell,
    projectors::Projectors,
    loc::Vector{Int64},
    loc_add::Vector{Int64},
    direction::Direction,
    Χ::Int64)


    if direction == LEFT

        """
        C4(i,j)  --   T1(i,j+s) --
           |             |
           |             |
        T4(i,j)  --   R(i,j+s)  --
           |             |


           |             |
        T4(i,j)  --   R(i,j+s)  --
           |             |
           |             |
        C3(i,j)  --   T3(i,j+s) --

        """


        E_loc = uc.E[loc...];
        E_add = uc.E[loc_add...];
        R_add = uc.R[loc_add...];

        @tensor C4T4T1A[ure, urc, de, dc] := E_loc.C[4][α, δ] * E_loc.T[4][δ, de, γ] * E_add.T[1][ure, α, β] * R_add.R[β, urc, dc, γ];
        C4T4T1A = reshape(C4T4T1A, (size(C4T4T1A, 1) * size(C4T4T1A, 2), :));

        #if uc.symmetry == XY || uc.symmetry == C4
        #    HL = C4T4T1A;
            #C3T4T3A = C4T4T1A';
        #else
            @tensor C3T4T3A[ue, uc, dre, drc] := E_loc.C[3][α, δ] * E_loc.T[4][ue, α, β] * E_add.T[3][dre, δ, γ] * R_add.R[uc, drc, γ, β];
            C3T4T3A = reshape(C3T4T3A, (size(C3T4T3A, 1) * size(C3T4T3A, 2), :));
            HL = C4T4T1A * C3T4T3A;
        #end

        U, Sinvsqrt, V = factorize_rho(HL, Χ)
        P̃ = C3T4T3A * V * Sinvsqrt;
        P = Sinvsqrt * U' * C4T4T1A;

        projectors.Pl = [P̃, P];

        @debug "Left renormalization" norm(C4T4T1A * C3T4T3A - C4T4T1A * P̃ * P * C3T4T3A);

    elseif direction == RIGHT


        """
        -- T1(i,j-s) --   C1(i,j)
               |             |
               |             |
        -- R(i,j-s)  --   T2(i,j)
               |             |


              |             |
        -- R(i,j-s)  --   T2(i,j)
              |             |
              |             |
        -- T3(i,j-s) --   C2(i,j)

        """

        E_loc = uc.E[loc...];
        E_add = uc.E[loc_add...];
        R_add = uc.R[loc_add...];

        @tensor C1T2T1A[ule, ulc, de, dc] := E_loc.C[1][α, δ] * E_loc.T[2][α, de, β] * E_add.T[1][δ, ule, γ] * R_add.R[γ, β, dc, ulc];
        C1T2T1A = reshape(C1T2T1A, (size(C1T2T1A, 1) * size(C1T2T1A, 2), :));

        #if uc.symmetry == XY || uc.symmetry == C4
            #HR = C1T2T1A;
            #C2T3T2A = C1T2T1A';
        #else
            @tensor C2T3T2A[ue, uc, dle, dlc] := E_loc.C[2][α, β] * E_add.T[3][β, dle, γ] * E_loc.T[2][ue, α, δ] * R_add.R[uc, δ, γ, dlc];
            C2T3T2A = reshape(C2T3T2A, (size(C2T3T2A, 1) * size(C2T3T2A, 2), :));
            HR = C1T2T1A * C2T3T2A;
        #end

        U, Sinvsqrt, V = factorize_rho(HR, Χ)
        P̃ = C2T3T2A * V * Sinvsqrt;
        P = Sinvsqrt * U' * C1T2T1A;

        projectors.Pr = [P̃, P];

        @debug "Right renormalization" norm(C1T2T1A * C2T3T2A - C1T2T1A * P̃ * P * C2T3T2A);
    elseif direction == UP

        """
        C4(i,j)   --   T1(i,j)  --
           |             |
           |             |
        T4(i+s,j) --   R(i+s,j) --
           |             |


        -- T1(i,j)   --   C1(i,j)
              |             |
              |             |
        -- R(i+s,j)  --   T2(i+s,j)
              |             |

        """

        E_loc = uc.E[loc...];
        E_add = uc.E[loc_add...];
        R_add = uc.R[loc_add...];

        @tensor C4T1T4A[lde, ldc, re, rc] := E_loc.C[4][α, δ] * E_loc.T[1][re, α, β] * E_add.T[4][δ, lde, γ] * R_add.R[β, rc, ldc, γ];
        C4T1T4A = reshape(C4T1T4A, (size(C4T1T4A, 1) * size(C4T1T4A, 2), :));


        #if uc.symmetry == XY || uc.symmetry == C4
            #HU = C4T1T4A;
            #C1T1T2A = C4T1T4A';
        #else
            @tensor C1T1T2A[le, lc, rde, rdc] := E_loc.C[1][α, δ] * E_add.T[2][α, rde, β] * E_loc.T[1][δ, le, γ] * R_add.R[γ, β, rdc, lc]
            C1T1T2A = reshape(C1T1T2A, (size(C1T1T2A, 1) * size(C1T1T2A, 2), :));
            HU = C4T1T4A * C1T1T2A;
        #end

        U, Sinvsqrt, V = factorize_rho(HU, Χ)
        P̃ = C1T1T2A * V * Sinvsqrt;
        P = Sinvsqrt * U' * C4T1T4A;

        projectors.Pu = [P̃, P];

        @debug "Up renormalization" norm(C4T1T4A * C1T1T2A - C4T1T4A * P̃ * P * C1T1T2A);

    elseif direction == DOWN

        """
           |             |
        T4(i-s,j) --   R(i-s,j) --
           |             |
           |             |
        C3(i,j)   --   T3(i,j)  --


              |             |
        -- R(i-s,j)  --   T2(i-s,j)
              |             |
              |             |
        -- T3(i,j)   --   C2(i,j)

        """

        E_loc = uc.E[loc...];
        E_add = uc.E[loc_add...];
        R_add = uc.R[loc_add...];

        @tensor C3T4T3A[lue, luc, re, rc] := E_loc.C[3][α, δ] * E_add.T[4][lue, α, β] * E_loc.T[3][re, δ, γ] * R_add.R[luc, rc, γ, β]
        C3T4T3A = reshape(C3T4T3A, (size(C3T4T3A, 1) * size(C3T4T3A, 2), :));

        #if uc.symmetry == XY || uc.symmetry == C4
            #HD = C3T4T3A;
            #C2T3T2A = C3T4T3A';
        #else
            @tensor C2T3T2A[le, lc, rue, ruc] := E_loc.C[2][α, β] * E_loc.T[3][β, le, γ] * E_add.T[2][rue, α, δ] * R_add.R[ruc, δ, γ, lc];
            C2T3T2A = reshape(C2T3T2A, (size(C2T3T2A, 1) * size(C2T3T2A, 2), :));
            HD = C3T4T3A * C2T3T2A;
        #end

        U, Sinvsqrt, V = factorize_rho(HD, Χ)
        P̃ = C2T3T2A * V * Sinvsqrt;
        P = Sinvsqrt * U' * C3T4T3A;

        projectors.Pd = [P̃, P];

        @debug "Down renormalization" norm(C3T4T3A * C2T3T2A - C3T4T3A * P̃ * P * C2T3T2A);

    end
end

#################################
# Old/obsolete/untested methods #
#################################


function calc_projectors_ctmrg_v1!(
    uc::UnitCell,
    projectors::Projectors,
    loc::CartesianIndex,
    step::Int64,
    direction::Direction,
    Χ::Int64)

    @info "Step" step


    if direction == LEFT

        """
        C4(i-1,j-1) --   T1(i-1,j+s) --
            |             |
            |             |
        T4(i,j-1)   --    R(i,j+s)   --
            |             |


            |             |
        T4(i+1,j-1)  --   R(i+1,j+s) --
            |             |
            |             |
        C3(i+2,j-1)  --   T3(i+2,j+s)--

        """

        # Build expanded corners
        C4 = uc(Environment, loc + CartesianIndex(-1, -1)).C[4];
        T4 = uc(Environment, loc + CartesianIndex(0, -1)).T[4];
        T1 = uc(Environment, loc + CartesianIndex(-1, step)).T[1];
        R = uc(ReducedTensor, loc + CartesianIndex(0, step)).R;

        @tensor Q4[ure, urc, de, dc] := C4[α, δ] * T4[δ, de, γ] * T1[ure, α, β] * R[β, urc, dc, γ];

        C3 = uc(Environment, loc + CartesianIndex(2, -1)).C[3];
        T4 = uc(Environment, loc + CartesianIndex(1, -1)).T[4];
        T3 = uc(Environment, loc + CartesianIndex(2, step)).T[3];
        R = uc(ReducedTensor, loc + CartesianIndex(1, step)).R;

        @tensor Q3[ue, uc, dre, drc] := C3[α, δ] * T4[ue, α, β] * T3[dre, δ, γ] * R[uc, drc, γ, β];

        # Calculate half-system density matrix
        Q4 = reshape(Q4, (size(Q4, 1) * size(Q4, 2), :));
        Q3 = reshape(Q3, (size(Q3, 1) * size(Q3, 2), :));
        HL = Q4 * Q3;

        # Calculate projectors
        U, Sinvsqrt, V = factorize_rho(HL, Χ)
        P̃ = Q3 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q4;

        projectors.Pl[loc] = [P̃, P];

    elseif direction == RIGHT

        """
        -- T1(i-1,j-s) --  C1(i-1,j+1)
            |                 |
            |                 |
        -- R(i,j-s)    --   T2(i,j+1)
            |                 |


            |                 |
        -- R(i+1,j-s)  --  T2(i+1,j+1)
            |                 |
            |                 |
        -- T3(i+2,j-s) --  C2(i+2,j+1)

        """

        C1 = uc(Environment, loc + CartesianIndex(-1, 1)).C[1];
        T2 = uc(Environment, loc + CartesianIndex(0, 1)).T[2];
        T1 = uc(Environment, loc + CartesianIndex(-1, -step)).T[1];
        R = uc(ReducedTensor, loc + CartesianIndex(0, -step)).R;

        @tensor Q1[ule, ulc, de, dc] := C1[α, δ] * T2[α, de, β] * T1[δ, ule, γ] * R[γ, β, dc, ulc];

        C2 = uc(Environment, loc + CartesianIndex(2, 1)).C[2];
        T2 = uc(Environment, loc + CartesianIndex(1, 1)).T[2];
        T3 = uc(Environment, loc + CartesianIndex(2, -step)).T[3];
        R = uc(ReducedTensor, loc + CartesianIndex(1, -step)).R;

        @tensor Q2[ue, uc, dle, dlc] := C2[α, β] * T3[β, dle, γ] * T2[ue, α, δ] * R[uc, δ, γ, dlc];

        Q1 = reshape(Q1, (size(Q1, 1) * size(Q1, 2), :));
        Q2 = reshape(Q2, (size(Q2, 1) * size(Q2, 2), :));
        HR = Q1 * Q2;

        U, Sinvsqrt, V = factorize_rho(HR, Χ)
        P̃ = Q2 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q1;

        projectors.Pr[loc] = [P̃, P];

    elseif direction == UP

        """
        C4(i-1,j-1) --  T1(i-1,j) --
            |             |
            |             |
        T4(i+s,j-1) --  R(i+s,j)  --
            |             |


        -- T1(i-1,j+1) --   C1(i-1,j+2)
                |             |
                |             |
        -- R(i+s,j+1)  --   T2(i+s,j+2)
                |             |

        """

        C4 = uc(Environment, loc + CartesianIndex(-1, -1)).C[4];
        T1 = uc(Environment, loc + CartesianIndex(-1, 0)).T[1];
        T4 = uc(Environment, loc + CartesianIndex(step, -1)).T[4];
        R = uc(ReducedTensor, loc + CartesianIndex(step, 0)).R;

        @tensor Q4[lde, ldc, re, rc] := C4[α, δ] * T1[re, α, β] * T4[δ, lde, γ] * R[β, rc, ldc, γ];

        C1 = uc(Environment, loc + CartesianIndex(-1, 2)).C[1];
        T1 = uc(Environment, loc + CartesianIndex(-1, 1)).T[1];
        T2 = uc(Environment, loc + CartesianIndex(step, 2)).T[2];
        R = uc(ReducedTensor, loc + CartesianIndex(step, 1)).R;

        @tensor Q1[le, lc, rde, rdc] := C1[α, δ] * T2[α, rde, β] * T1[δ, le, γ] * R[γ, β, rdc, lc];

        Q4 = reshape(Q4, (size(Q4, 1) * size(Q4, 2), :));
        Q1 = reshape(Q1, (size(Q1, 1) * size(Q1, 2), :));
        HU = Q4 * Q1;

        U, Sinvsqrt, V = factorize_rho(HU, Χ)
        P̃ = Q1 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q4;

        projectors.Pu[loc] = [P̃, P];


    elseif direction == DOWN

        """
            |             |
        T4(i-s,j-1) --  R(i-s,j)  --
            |             |
            |             |
        C3(i+1,j-1) --  T3(i+1,j) --


                |             |
        -- R(i-s,j+1)  --   T2(i-s,j+2)
                |             |
                |             |
        -- T3(i+1,j+1) --   C2(i+1,j+2)

        """

        C3 = uc(Environment, loc + CartesianIndex(1, -1)).C[3];
        T3 = uc(Environment, loc + CartesianIndex(1, 0)).T[3];
        T4 = uc(Environment, loc + CartesianIndex(-step, -1)).T[4];
        R = uc(ReducedTensor, loc + CartesianIndex(-step, 0)).R;

        @tensor Q3[lue, luc, re, rc] := C3[α, δ] * T4[lue, α, β] * T3[re, δ, γ] * R[luc, rc, γ, β]

        C2 = uc(Environment, loc + CartesianIndex(1, 2)).C[2];
        T3 = uc(Environment, loc + CartesianIndex(1, 1)).T[3];
        T2 = uc(Environment, loc + CartesianIndex(-step, 2)).T[2];
        R = uc(ReducedTensor, loc + CartesianIndex(-step, 1)).R;

        @tensor Q2[le, lc, rue, ruc] := C2[α, β] * T3[β, le, γ] * T2[rue, α, δ] * R[ruc, δ, γ, lc];

        Q2 = reshape(Q2, (size(Q2, 1) * size(Q2, 2), :));
        Q3 = reshape(Q3, (size(Q3, 1) * size(Q3, 2), :));
        HD = Q3 * Q2;

        U, Sinvsqrt, V = factorize_rho(HD, Χ)
        P̃ = Q2 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q3;

        projectors.Pd[loc] = [P̃, P];

    end


end

function calc_projectors_ctmrg_v2!(
    uc::UnitCell,
    projectors::Projectors,
    loc::CartesianIndex,
    step::Int64,
    direction::Direction,
    Χ::Int64)


    if direction == LEFT

        """
        C4(i-1,j) --   T1(i-1,j+s+1) --
            |             |
            |             |
        T4(i,j)   --    R(i,j+s+1)   --
            |             |


            |             |
        T4(i+1,j)  --   R(i+1,j+s+1) --
            |             |
            |             |
        C3(i+2,j)  --   T3(i+2,j+s+1)--

        """

        # Build expanded corners

        C4 = uc(Environment, loc + CartesianIndex(-1, 0)).C[4];
        T4 = uc(Environment, loc + CartesianIndex(0, 0)).T[4];
        T1 = uc(Environment, loc + CartesianIndex(-1, 1 + step)).T[1];
        R = uc(ReducedTensor, loc + CartesianIndex(0, 1 + step)).R;

        @tensor Q4[ure, urc, de, dc] := C4[α, δ] * T4[δ, de, γ] * T1[ure, α, β] * R[β, urc, dc, γ];

        C3 = uc(Environment, loc + CartesianIndex(2, 0)).C[3];
        T4 = uc(Environment, loc + CartesianIndex(1, 0)).T[4];
        T3 = uc(Environment, loc + CartesianIndex(2, 1 + step)).T[3];
        R = uc(ReducedTensor, loc + CartesianIndex(1, 1 + step)).R;

        @tensor Q3[ue, uc, dre, drc] := C3[α, δ] * T4[ue, α, β] * T3[dre, δ, γ] * R[uc, drc, γ, β];

        # Calculate half-system density matrix
        Q4 = reshape(Q4, (size(Q4, 1) * size(Q4, 2), :));
        Q3 = reshape(Q3, (size(Q3, 1) * size(Q3, 2), :));
        HL = Q4 * Q3;

        # Calculate projectors
        #U, Sinvsqrt, V = factorize_rho(HL, Χ)
        U, Sinvsqrt, V, S = factorize_rho(HL, Χ)
        P̃ = Q3 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q4;

        projectors.Pl[loc] = [P̃, P];

        @debug "Left renormalization" norm(Q4 * Q3 - Q4 * P̃ * P * Q3);

    elseif direction == RIGHT

        """
        -- T1(i-1,j-s-1) --  C1(i-1,j)
              |                 |
              |                 |
        -- R(i,j-s-1)    --   T2(i,j)
              |                 |


              |                 |
        -- R(i+1,j-s-1)  --  T2(i+1,j)
              |                 |
              |                 |
        -- T3(i+2,j-s-1) --  C2(i+2,j)

        """

        C1 = uc(Environment, loc + CartesianIndex(-1, 0)).C[1];
        T2 = uc(Environment, loc + CartesianIndex(0, 0)).T[2];
        T1 = uc(Environment, loc + CartesianIndex(-1, -1 - step)).T[1];
        R = uc(ReducedTensor, loc + CartesianIndex(0, -1 - step)).R;

        @tensor Q1[ule, ulc, de, dc] := C1[α, δ] * T2[α, de, β] * T1[δ, ule, γ] * R[γ, β, dc, ulc];

        C2 = uc(Environment, loc + CartesianIndex(2, 0)).C[2];
        T2 = uc(Environment, loc + CartesianIndex(1, 0)).T[2];
        T3 = uc(Environment, loc + CartesianIndex(2, -1 - step)).T[3];
        R = uc(ReducedTensor, loc + CartesianIndex(1, -1 - step)).R;

        @tensor Q2[ue, uc, dle, dlc] := C2[α, β] * T3[β, dle, γ] * T2[ue, α, δ] * R[uc, δ, γ, dlc];

        Q1 = reshape(Q1, (size(Q1, 1) * size(Q1, 2), :));
        Q2 = reshape(Q2, (size(Q2, 1) * size(Q2, 2), :));
        HR = Q1 * Q2;

        U, Sinvsqrt, V, S = factorize_rho(HR, Χ)
        P̃ = Q2 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q1;

        projectors.Pr[loc] = [P̃, P];

    elseif direction == UP

        """
         C4(i,j-1)   --   T1(i,j)   --
             |               |
             |               |
        T4(i+1+s,j-1) -- R(i+1+s,j) --
             |               |


        -- T1(i,j+1)   --   C1(i,j+2)
                |              |
                |              |
        -- R(i+s+1,j+1) --  T2(i+s+1,j+2)
                |              |

        """

        C4 = uc(Environment, loc + CartesianIndex(0, -1)).C[4]; #!
        T1 = uc(Environment, loc + CartesianIndex(0, 0)).T[1];
        T4 = uc(Environment, loc + CartesianIndex(1 + step, -1)).T[4];
        R = uc(ReducedTensor, loc + CartesianIndex(1 + step, 0)).R;


        @tensor Q4[lde, ldc, re, rc] := C4[α, δ] * T1[re, α, β] * T4[δ, lde, γ] * R[β, rc, ldc, γ];

        C1 = uc(Environment, loc + CartesianIndex(0, 2)).C[1];
        T1 = uc(Environment, loc + CartesianIndex(0, 1)).T[1];
        T2 = uc(Environment, loc + CartesianIndex(1 + step, 2)).T[2];
        R = uc(ReducedTensor, loc + CartesianIndex(1 + step, 1)).R;

        @tensor Q1[le, lc, rde, rdc] := C1[α, δ] * T2[α, rde, β] * T1[δ, le, γ] * R[γ, β, rdc, lc];

        Q4 = reshape(Q4, (size(Q4, 1) * size(Q4, 2), :));
        Q1 = reshape(Q1, (size(Q1, 1) * size(Q1, 2), :));
        HU = Q4 * Q1;

        U, Sinvsqrt, V, S = factorize_rho(HU, Χ)
        P̃ = Q1 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q4;

        projectors.Pu[loc] = [P̃, P];

    elseif direction == DOWN

        """
              |              |
        T4(i-s-1,j-1) -- R(i-s-1,j) --
              |              |
              |              |
         C3(i,j-1)  --   T3(i,j)   --


                 |             |
        -- R(i-s-1,j+1) -- T2(i-s-1,j+2)
                 |             |
                 |             |
        --  T3(i,j+1) --   C2(i,j+2)

        """

        C3 = uc(Environment, loc + CartesianIndex(0, -1)).C[3];
        T3 = uc(Environment, loc + CartesianIndex(0, 0)).T[3];
        T4 = uc(Environment, loc + CartesianIndex(-1 - step, -1)).T[4];
        R = uc(ReducedTensor, loc + CartesianIndex(-1 - step, 0)).R;

        @tensor Q3[lue, luc, re, rc] := C3[α, δ] * T4[lue, α, β] * T3[re, δ, γ] * R[luc, rc, γ, β]

        C2 = uc(Environment, loc + CartesianIndex(0, 2)).C[2];
        T3 = uc(Environment, loc + CartesianIndex(0, 1)).T[3];
        T2 = uc(Environment, loc + CartesianIndex(-1 - step, 2)).T[2];
        R = uc(ReducedTensor, loc + CartesianIndex(-1 - step, 1)).R;

        @tensor Q2[le, lc, rue, ruc] := C2[α, β] * T3[β, le, γ] * T2[rue, α, δ] * R[ruc, δ, γ, lc];

        Q2 = reshape(Q2, (size(Q2, 1) * size(Q2, 2), :));
        Q3 = reshape(Q3, (size(Q3, 1) * size(Q3, 2), :));
        HD = Q3 * Q2;

        U, Sinvsqrt, V, S = factorize_rho(HD, Χ)
        P̃ = Q2 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q3;

        projectors.Pd[loc] = [P̃, P];

    end


end


function do_ctmrg_iteration_old!(
    unitcell::UnitCell,
    projectors::Projectors{EachMove})

    # u,d,l,r: direction of bond. e,c: environment or cell bond. E.g: bond re corresponds to a right-environment bond
    # greek letters for contracted symbols

    Χ = unitcell.E[1,1].Χ;
    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];

    for ij ∈ CartesianIndices((1:Ni, 1:Nj))

        #= Left move. Absorbs unit-cell tensors and environment tensors of the whole column, row by row.=#
        for n ∈ 0:Nj - 1
            # Renormalize
            ij[1] == 1 ? i_1j = CartesianIndex(Ni, ij[2]) : i_1j = CartesianIndex(ij[1] - 1, ij[2]);

            calc_projectors_ctmrg!(unitcell, projectors, ij, n, LEFT, Χ); # P_(i,j)
            calc_projectors_ctmrg!(unitcell, projectors, i_1j, n, LEFT, Χ); # P_(i-1,j)

            do_ctm_move!(unitcell, projectors, LEFT, ij, n);
        end

        #= Right move. Absorbs unit-cell tensors and environment tensors of the whole column, row by row =#
        for n ∈ 0:Nj - 1

            # Renormalize
            ij[1] == 1 ? i_1j = CartesianIndex(Ni, ij[2]) : i_1j = CartesianIndex(ij[1] - 1, ij[2]);

            calc_projectors_ctmrg!(unitcell, projectors, ij, n, RIGHT, Χ);
            calc_projectors_ctmrg!(unitcell, projectors, i_1j, n, RIGHT, Χ); # P_(i-1,j)

            do_ctm_move!(unitcell, projectors, RIGHT, ij, n);
        end

        #= Up move. Absorbs unit-cell tensors and environment tensors of the whole row, column by column =#
        for n ∈ 0:Ni - 1
            #@info "Up move, step $(n+1)"

            # Renormalize
            ij[2] == 1 ? ij_1 = CartesianIndex(ij[1], Nj) : ij_1 = CartesianIndex(ij[1], ij[2]-1);

            calc_projectors_ctmrg!(unitcell, projectors, ij, n, UP, Χ);
            calc_projectors_ctmrg!(unitcell, projectors, ij_1, n, UP, Χ); # P_(i,j-1)

            do_ctm_move!(unitcell, projectors, UP, ij, n);
        end

        #= Down move. Absorbs unit-cell tensors and environment tensors of the whole row, column by column =#
        for n ∈ 0:Ni - 1
            #@info "Down move, step $(n+1)"

            # Renormalize
            ij[2] == 1 ? ij_1 = CartesianIndex(ij[1], Nj) : ij_1 = CartesianIndex(ij[1], ij[2]-1);

            calc_projectors_ctmrg!(unitcell, projectors, ij, n, DOWN, Χ);
            calc_projectors_ctmrg!(unitcell, projectors, ij_1, n, DOWN, Χ);
            do_ctm_move!(unitcell, projectors, DOWN, ij, n);
        end
    end
end

function do_ctm_iteration_old!(
    ::Type{S},
    unitcell::UnitCell,
    projectors::Projectors) where {S<:Union{Type{R4}, Type{XY}}}

    # u,d,l,r: direction of bond. e,c: environment or cell bond. E.g: bond re corresponds to a right-environment bond
    # greek letters for contracted symbols

    @assert "Method is outdated. calc_projectors! call has wrong syntaxis. See other do_ctm_interation methods"

    Χ = unitcell.E[1,1].Χ;
    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];

    for i ∈ 1:Ni, j ∈ 1:Nj

        E_loc = uc.E[i, j];

        #= Left/Right moves. Absorbs unit-cell tensors and environment tensors of the whole column, row by row.=#
        for n ∈ 0:Nj - 1
            j_n = mod(j + n - 1, Nj) + 1; # jₙ = 1 + mod(j - 1, nⱼ)

            E_add = uc.E[i, j_n];
            R_add = uc.R[i, j_n];

            # Grow environment
            @tensor C4T1[re, de, dc] := E_loc.C[4][α, de] * E_add.T[1][re, α, dc]
            @tensor T4A[ue, uc, de, dc, rc] := E_loc.T[4][ue, de, α] * R_add.R[uc, rc, dc, α]
            @tensor C3T3[ue, uc, re] := E_loc.C[3][ue, α] * E_add.T[3][re, α, uc]

            # Renormalize
            calc_projectors!(unitcell, projectors, [i,j], LEFT, n, Χ);

            T4A = reshape(T4A, (prod(size(T4A)[1:2]), prod(size(T4A)[3:4]), size(T4A, 5)));

            C̃4 = reshape(C4T1, (size(C4T1, 1), :)) * projectors.Pl[1]; #(r, d)
            C̃3 = projectors.Pl[2] * reshape(C3T3, (:, size(C3T3, 3))); #(u, r)
            @tensor T̃4[ue, de, rc] := projectors.Pl[2][ue, α] * T4A[α, β, rc] * projectors.Pl[1][β, de];

            # Update tensors environment
            update_tensors!(unitcell, [C̃3, T̃4, C̃4], LEFT, [i,j]);

            C̃1 = adjoint(C̃4);            T̃2 = T̃4;            C̃2 = C̃3;
            update_tensors!(unitcell, [C̃1, T̃2, C̃2], RIGHT, [i,j]);

        end


        #= Up move. Absorbs unit-cell tensors and environment tensors of the whole row, column by column =#
        for n ∈ 0:Ni - 1
            #@info "Up move, step $(n+1)"
            i_n = mod(i + n - 1, Ni) + 1;

            E_add = uc.E[i_n, j];
            R_add = uc.R[i_n, j];

            # Grow environment
            @tensor C4T4[re, rc, de] := E_loc.C[4][re, α] * E_add.T[4][α, de, rc];
            @tensor T1A[le, lc, re, rc, dc] := E_loc.T[1][re, le, α] * R_add.R[α, rc, dc, lc];
            @tensor C1T2[de, le, lc] := E_loc.C[1][α, le] * E_add.T[2][α, de, lc]; #! indices permuted

            # Renormalize
            calc_projectors!(unitcell, projectors, [i,j], UP, n, Χ)

            C4T4 = transpose(reshape(C4T4, (:, size(C4T4, 3))));
            C1T2 = transpose(reshape(C1T2, (size(C1T2, 1), :)));
            T1A = reshape(T1A, (prod(size(T1A)[1:2]), prod(size(T1A)[3:4]), size(T1A, 5)));

            C̃4 = transpose(C4T4 * projectors.Pu[1]); #(d,r) -> (r,d)
            C̃1 = transpose(projectors.Pu[2] * C1T2); #(l,d) -> (d,l)
            @tensor T̃1[le, re, dc] := projectors.Pu[2][le, α] * T1A[α, β, dc] * projectors.Pu[1][β, re];

            # Update tensors environment
            update_tensors!(unitcell, [C̃4, T̃1, C̃1], UP, [i,j]);

            C̃3 = adjoint(C̃4);            T̃3 = T̃1;            C̃2 = C̃1;
            update_tensors!(unitcell, [C̃2, T̃3, C̃3], DOWN, [i,j]);


        end

    end
end

"""
    function calc_projectors!(
    uc::UnitCell{T},
    projectors::Projectors{HalfSystem}, loc::Tuple{Int64, Int64}, direction::String, step::Int64,
    Χ::Int64;
    kwargs...)

    Calculates environment projectors using two corner tensors, following 10.1103/PhysRevB.80.094403.

### Arguments
### Returns
- P and P̃
### Notes
- the second leg of the projector corresponds to the auxiliary bond of the unit-cell
"""

function calc_projectors_untested!(
    uc::UnitCell{T},
    projectors::Projectors{TwoCorners},
    loc::Tuple{Int64, Int64},
    direction::String,
    step::Int64,
    Χ::Int64;
    kwargs...) where {T}

    @warn "Method not tested yet"

    if direction == "left"

        ## Upper half density matrix. #! Updated, however still missing a good definition of location_step

        loc_step = loc
        loc_step[2] = mod(loc_step[2] + step - 1, uc.Nj) + 1;

        @tensor C4T1T4A[re, rc, lde, ldc] := uc.E[loc...].C[4][α, δ] * uc.E[loc...].T[1][re, α, β] * uc.E[loc_step...].T[4][δ, lde, γ] * uc.R[loc_step...].R[β, rc, ldc, γ]
        @tensor C1T1T2A[rde, rdc, le, lc] := uc.E[loc...].C[1][α, δ] * uc.E[loc_step...].T[2][α, rde, β] * uc.E[loc...].T[1][δ, le, γ] * uc.R[loc_step...].R[γ, β, rdc, lc]

        @tensor HU[lde, ldc, rde, rdc] := C4T1T4A[α, β, lde, ldc] * C2T1T2A[rde, rdc, α, β]

        HU = reshape(HU, size(HU, 1) * size(HU, 2), :); #! Merge into one line

        ## Lower half density matrix
        @tensor C3T4T3A[lue, luc, re, rc] := uc[loc...].E.C[3][α, δ] * uc[loc_step...].E.T[4][lue, α, β] * uc[loc...].E.T[3][re, δ, γ] * uc.R[loc_step...].R[luc, rc, γ, β]
        @tensor C2T3T2A[rue, ruc, le, lc] := uc[loc...].E.C[2][α, β] * uc[loc...].E.T[3][β, le, γ] * uc[loc_step...].E.T[2][rue, α, δ] * uc.R[loc_step...].R[ruc, δ, γ, lc]
        @tensor HD[lue, luc, rue, ruc] := C3T4T3A[lue, luc, α, β] * C2T3T2A[rue, ruc, α, β]

        HD = reshape(HD, size(HD, 1) * size(HD, 2), :);

        ### Left/Right move projectors
        projectors.Pl = projectors_from_identity(transpose(HU), transpose(HD), Χ; kwargs...);

    elseif direction == "right"

        loc = loc_step;
        loc_step[2] = mod(loc_step[2] - step - 1, uc.dims[2]) + 1;

        @tensor C4T1T4A[re, rc, lde, ldc] := uc.E[loc...].C[4][α, δ] * uc.E[loc...].T[1][re, α, β] * uc.E[loc_step...].T[4][δ, lde, γ] * uc.R[loc_step...].R[β, rc, ldc, γ]
        @tensor C1T1T2A[rde, rdc, le, lc] := uc.E[loc...].C[1][α, δ] * uc.E[loc_step...].T[2][α, rde, β] * uc.E[loc...].T[1][δ, le, γ] * uc.R[loc_step...].R[γ, β, rdc, lc]
        @tensor HU[lde, ldc, rde, rdc] := C4T1T4A[α, β, lde, ldc] * C2T1T2A[rde, rdc, α, β]

        HU = reshape(HU, size(HU, 1) * size(HU, 2), :); #! Merge into one line

        ## Lower half density matrix
        @tensor C3T4T3A[lue, luc, re, rc] := uc[loc...].E.C[3][α, δ] * uc[loc_step...].E.T[4][lue, α, β] * uc[loc...].E.T[3][re, δ, γ] * uc.R[loc_step...].R[luc, rc, γ, β]
        @tensor C2T3T2A[rue, ruc, le, lc] := uc[loc...].E.C[2][α, β] * uc[loc...].E.T[3][β, le, γ] * uc[loc_step...].E.T[2][rue, α, δ] * uc.R[loc_step...].R[ruc, δ, γ, lc]
        @tensor HD[lue, luc, rue, ruc] := C3T4T3A[lue, luc, α, β] * C2T3T2A[rue, ruc, α, β]

        HD = reshape(HD, size(HD, 1) * size(HD, 2), :);
        projectors.Pr = projectors_from_identity(HU, HD, Χ; kwargs...);


    elseif direction == "up"


        loc_step = loc
        loc_step[1] = mod(loc_step[1] + step - 1, uc.dims[1]) + 1;

        ## Left half density matrix
        @tensor C4T4T1A[ure, urc, de, dc] := uc.E[loc...].C[4][α, δ] * uc.E[loc...].T[4][δ, de, γ] * uc.E[loc_step...].T[1][ure, α, β] * uc.R[loc_step...].R[β, urc, dc, γ]
        @tensor C3T4T3A[ue, uc, dre, drc] := uc.E[loc...].C[3][α, δ] * uc.E[loc...].T[4][ue, α, β] * uc.E[loc_step...].T[3][dre, δ, γ] * uc.R[loc_step...].R[uc, drc, γ, β]
        @tensor HL[ure, urc, dre, drc] := C4E4E1A[ure, urc, α, β] * C3E4E3A[α, β, dre, drc]

        HL = reshape(HL, size(HL, 1) * size(HL, 2), :); #! Merge into one line

        ## Right half density matrix
        @tensor C1T2T1A[de, dc, ule, ulc] := uc.E[loc...].C[1][α, δ] * uc.E[loc...].T[2][α, de, β] * uc.E[loc_step...].T[1][δ, ule, γ] * uc.R[loc_step...].R[γ, β, dc, ulc]
        @tensor C2T3T2A[ue, uc, dle, dlc] := uc.E[loc...].C[2][α, β] * uc.E[loc_step...].T[3][β, dle, γ] * uc.E[loc...].T[2][ue, α, δ] * uc.R[loc_step...].R[uc, δ, γ, dlc]
        @tensor HR[ule, ulc, dle, dlc] := C1E2E1A[α, β, ule, ulc] * C2E3E2A[α, β, dle, dlc]

        HR = reshape(HR, size(HR, 1) * size(HR, 2), :);

        ### Up/down move projectors
        projectors.Pu = projectors_from_identity(transpose(HL), transpose(HR), Χ; kwargs...);

    elseif direction == "down"

        loc_step = loc
        loc_step[1] = mod(loc_step[1] - step - 1, uc.dims[1]) + 1;

        ## Left half density matrix
        @tensor C4T4T1A[ure, urc, de, dc] := uc.E[loc...].C[4][α, δ] * uc.E[loc...].T[4][δ, de, γ] * uc.E[loc_step...].T[1][ure, α, β] * uc.R[loc_step...].R[β, urc, dc, γ]
        @tensor C3T4T3A[ue, uc, dre, drc] := uc.E[loc...].C[3][α, δ] * uc.E[loc...].T[4][ue, α, β] * uc.E[loc_step...].T[3][dre, δ, γ] * uc.R[loc_step...].R[uc, drc, γ, β]
        @tensor HL[ure, urc, dre, drc] := C4E4E1A[ure, urc, α, β] * C3E4E3A[α, β, dre, drc]

        HL = reshape(HL, size(HL, 1) * size(HL, 2), :); #! Merge into one line

        ## Right half density matrix
        @tensor C1T2T1A[de, dc, ule, ulc] := uc.E[loc...].C[1][α, δ] * uc.E[loc...].T[2][α, de, β] * uc.E[loc_step...].T[1][δ, ule, γ] * uc.R[loc_step...].R[γ, β, dc, ulc]
        @tensor C2T3T2A[ue, uc, dle, dlc] := uc.E[loc...].C[2][α, β] * uc.E[loc_step...].T[3][β, dle, γ] * uc.E[loc...].T[2][ue, α, δ] * uc.R[loc_step...].R[uc, δ, γ, dlc]
        @tensor HR[ule, ulc, dle, dlc] := C1E2E1A[α, β, ule, ulc] * C2E3E2A[α, β, dle, dlc]


        HR = reshape(HR, size(HR, 1) * size(HR, 2), :);

        ### Up/down move projectors
        projectors.Pd = projectors_from_identity(HL, HR, Χ; kwargs...);


    end
end

function do_ctm_move_v3!(unitcell::UnitCell, projectors::Projectors, direction::Direction, ij::CartesianIndex, step::Int64)
    E_loc = unitcell(Environment, ij);
    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];

    if direction == LEFT

        """
        C4(i,j)  --  T1(i,j+1+s) --
            |             |


            |             |
        T4(i,j)  --   R(i,j+1+s) --
            |             |


            |             |
        C3(i,j)  --  T3(i,j+1+s) --
        """

        E_add = unitcell(Environment, ij + CartesianIndex(0, 1 + step));
        R_add = unitcell(ReducedTensor, ij + CartesianIndex(0, 1 + step));

        # Grow environment
        @tensor C4T1[re, de, dc] := E_loc.C[4][α, de] * E_add.T[1][re, α, dc]
        @tensor T4A[ue, uc, de, dc, rc] := E_loc.T[4][ue, de, α] * R_add.R[uc, rc, dc, α]
        @tensor C3T3[ue, uc, re] := E_loc.C[3][ue, α] * E_add.T[3][re, α, uc]

        T4A = reshape(T4A, (prod(size(T4A)[1:2]), prod(size(T4A)[3:4]), size(T4A, 5)));
        C4T1 = reshape(C4T1, (size(C4T1, 1), :));
        C3T3 = reshape(C3T3, (:, size(C3T3, 3)));

        # Renormalize
        ij[1] == 1 ? i_1j = CartesianIndex(Ni, ij[2]) : i_1j = CartesianIndex(ij[1] - 1, ij[2]);

        C̃4 = C4T1 * projectors.Pl[ij][1]; #(r, d)
        C̃3 = projectors.Pl[i_1j][2] * C3T3; #(u, r)
        @tensor T̃4[ue, de, rc] := projectors.Pl[i_1j][2][ue, α] * T4A[α, β, rc] * projectors.Pl[ij][1][β, de];

        # Update tensors environment
        update_tensors!(unitcell, [C̃3, T̃4, C̃4], LEFT, ij);

    elseif direction == RIGHT

        """
        -- T1(i,j-1-s)  --    C1(i,j)
               |                 |

               |                 |
        -- R(i,j-1-s)   --    T2(i,j)
               |                 |

               |                 |
        -- T3(i,j-1-s)  --   C2(i,j)

        """

        E_add = unitcell(Environment, ij + CartesianIndex(0, -1 - step));
        R_add = unitcell(ReducedTensor, ij + CartesianIndex(0, -1 - step));

        # Grow environment
        @tensor C1T1[de, dc, le] := E_loc.C[1][de, α] * E_add.T[1][α, le, dc];
        @tensor T2A[ue, uc, de, dc, lc] := E_loc.T[2][ue, de, α] * R_add.R[uc, α, dc, lc];
        @tensor C2T3[ue, uc, le] := E_loc.C[2][ue, α] * E_add.T[3][α, le, uc]; #! indices permuted

        C1T1 = transpose(reshape(C1T1, (:, size(C1T1, 3))));
        C2T3 = reshape(C2T3, (:, size(C2T3, 3)));
        T2A = reshape(T2A, (prod(size(T2A)[1:2]), prod(size(T2A)[3:4]), size(T2A, 5)));

        # Renormalize
        ij[1] == 1 ? i_1j = CartesianIndex(Ni, ij[2]) : i_1j = CartesianIndex(ij[1] - 1, ij[2]);

        C̃1 = transpose(C1T1 * projectors.Pr[ij][1]); # (l, d) -> (d, l)
        C̃2 = projectors.Pr[i_1j][2] * C2T3; # (u, l)
        @tensor T̃2[ue, de, lc] := projectors.Pr[i_1j][2][ue, α] * T2A[α, β, lc] * projectors.Pr[ij][1][β, de];

        # Update tensors environment
        update_tensors!(unitcell, [C̃1, T̃2, C̃2], RIGHT, ij);

    elseif direction == UP

        """
         C4(i,j)   --      --T1(i,j)--       --C1(i,j)
            |                   |                 |
            |                   |                 |
        T4(i+1+s,j)--    --R(i+1+s,j)--    --T2(i+1+s,j)
            |                   |                 |

        """

        E_add = unitcell(Environment, ij + CartesianIndex(1 + step, 0));
        R_add = unitcell(ReducedTensor, ij + CartesianIndex(1 + step, 0));

        # Grow environment
        @tensor C4T4[re, rc, de] := E_loc.C[4][re, α] * E_add.T[4][α, de, rc];
        @tensor T1A[le, lc, re, rc, dc] := E_loc.T[1][re, le, α] * R_add.R[α, rc, dc, lc];
        @tensor C1T2[de, le, lc] := E_loc.C[1][α, le] * E_add.T[2][α, de, lc]; #! indices permuted

        C4T4 = transpose(reshape(C4T4, (:, size(C4T4, 3))));
        C1T2 = transpose(reshape(C1T2, (size(C1T2, 1), :)));
        T1A = reshape(T1A, (prod(size(T1A)[1:2]), prod(size(T1A)[3:4]), size(T1A, 5)));

        # Renormalize
        ij[2] == 1 ? ij_1 = CartesianIndex(ij[1], Nj) : ij_1 = CartesianIndex(ij[1], ij[2]-1);

        C̃4 = transpose(C4T4 * projectors.Pu[ij][1]); #(d,r) -> (r,d)
        C̃1 = transpose(projectors.Pu[ij_1][2] * C1T2); #(l,d) -> (d,l)
        @tensor T̃1[le, re, dc] := projectors.Pu[ij_1][2][le, α] * T1A[α, β, dc] * projectors.Pu[ij][1][β, re];

        # Update tensors environment
        update_tensors!(unitcell, [C̃4, T̃1, C̃1], UP, ij);

    elseif direction == DOWN

        """
             |                 |                  |
        T4(i-1-s,j)--    --R(i-1-s,j)--    --T2(i-1-s,j)
             |                 |                  |
             |                 |                  |
          C3(i,j) --      --T3(i,j)--        --C2(i,j)

        """

        E_add = unitcell(Environment, ij + CartesianIndex(-1 - step, 0));
        R_add = unitcell(ReducedTensor, ij + CartesianIndex(-1 - step, 0));

        # Grow environment
        @tensor C3T4[ue, re, rc] := E_loc.C[3][α, re] * E_add.T[4][ue, α, rc]; #! indices permuted
        @tensor T3A[le, lc, re, rc, uc] := E_loc.T[3][re, le, α] * R_add.R[uc, rc, α, lc];
        @tensor C2T2[ue, le, lc] := E_loc.C[2][α, le] * E_add.T[2][ue, α, lc];

        C3T4 = reshape(C3T4, (size(C3T4, 1), :));
        C2T2 = transpose(reshape(C2T2, (size(C2T2, 1), :)));
        T3A = reshape(T3A, (prod(size(T3A)[1:2]), prod(size(T3A)[3:4]), size(T3A, 5)));

        # Renormalize
        ij[2] == 1 ? ij_1 = CartesianIndex(ij[1], Nj) : ij_1 = CartesianIndex(ij[1], ij[2]-1);

        C̃3 = C3T4 * projectors.Pd[ij][1]; #(u,r)
        C̃2 = transpose(projectors.Pd[ij_1][2] * C2T2); #(l,u) -> (u,l)
        @tensor T̃3[le, re, uc] := projectors.Pd[ij_1][2][le, α] * T3A[α, β, uc] * projectors.Pd[ij][1][β, re];

        # Update tensors environment
        update_tensors!(unitcell, [C̃2, T̃3, C̃3], DOWN, ij);
    end
end


function do_ctm_move_v1!(unitcell::UnitCell, projectors::Projectors, direction::Direction, ij::CartesianIndex, step::Int64)

    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];

    if direction == LEFT

        """
        C4(i-1,j-1) -- T1(i-1,j+s) --
            |             |


            |             |
        T4(i,j-1) --  R(i,j+s) --
            |             |


            |             |
        C3(i+1,j-1) -- T3(i+1,j+s) --

        """

        C4 = unitcell(Environment, ij + CartesianIndex(-1, -1)).C[4];
        T1 = unitcell(Environment, ij + CartesianIndex(-1, step)).T[1];

        T4 = unitcell(Environment, ij + CartesianIndex(0, -1)).T[4];
        R = unitcell(ReducedTensor, ij + CartesianIndex(0, step)).R;

        C3 = unitcell(Environment, ij + CartesianIndex(1, -1)).C[4];
        T3 = unitcell(Environment, ij + CartesianIndex(1, step)).T[3];

        @info "Size C4" size(C4)
        @info "Size C3" size(C3)
        @info "Size T3" size(T3)
        @info "Size T1" size(T1)
        @info "Size T4" size(T4)

        # Grow environment
        @tensor C4T1[re, de, dc] := C4[α, de] * T1[re, α, dc]
        @tensor T4A[ue, uc, de, dc, rc] := T4[ue, de, α] * R[uc, rc, dc, α]
        @tensor C3T3[ue, uc, re] := C3[ue, α] * T3[re, α, uc]

        T4A = reshape(T4A, (prod(size(T4A)[1:2]), prod(size(T4A)[3:4]), size(T4A, 5)));
        C4T1 = reshape(C4T1, (size(C4T1, 1), :));
        C3T3 = reshape(C3T3, (:, size(C3T3, 3)));

        # Renormalize
        ij[1] == 1 ? i_1j = CartesianIndex(Ni, ij[2]) : i_1j = CartesianIndex(ij[1] - 1, ij[2]);

        C̃4 = C4T1 * projectors.Pl[ij][1]; #(r, d)
        C̃3 = projectors.Pl[i_1j][2] * C3T3; #(u, r)
        @tensor T̃4[ue, de, rc] := projectors.Pl[i_1j][2][ue, α] * T4A[α, β, rc] * projectors.Pl[ij][1][β, de];

        # Update tensors environment
        update_tensors!(unitcell, [C̃3, T̃4, C̃4], LEFT, ij);

    elseif direction == RIGHT

        """
        -- T1(i-1,j-s) --  C1(i-1,j+1)
              |                 |

              |                 |
        -- R(i,j-s)    --   T2(i,j+1)
              |                 |

              |                 |
        -- T3(i+1,j-s) --  C2(i+1,j+1)

        """

        C1 = unitcell(Environment, ij + CartesianIndex(-1, 1)).C[1];
        T1 = unitcell(Environment, ij + CartesianIndex(-1, -step)).T[1];

        T2 = unitcell(Environment, ij + CartesianIndex(0, 1)).T[2];
        R = unitcell(ReducedTensor, ij + CartesianIndex(0, -step)).R;

        C2 = unitcell(Environment, ij + CartesianIndex(1, 1)).C[2];
        T3 = unitcell(Environment, ij + CartesianIndex(1, -step)).T[3];


        # Grow environment
        @tensor C1T1[de, dc, le] := C1[de, α] * T1[α, le, dc];
        @tensor T2A[ue, uc, de, dc, lc] := T2[ue, de, α] * R[uc, α, dc, lc];
        @tensor C2T3[ue, uc, le] := C2[ue, α] * T3[α, le, uc]; #! indices permuted

        C1T1 = transpose(reshape(C1T1, (:, size(C1T1, 3))));
        C2T3 = reshape(C2T3, (:, size(C2T3, 3)));
        T2A = reshape(T2A, (prod(size(T2A)[1:2]), prod(size(T2A)[3:4]), size(T2A, 5)));

        # Renormalize
        ij[1] == 1 ? i_1j = CartesianIndex(Ni, ij[2]) : i_1j = CartesianIndex(ij[1] - 1, ij[2]);

        C̃1 = transpose(C1T1 * projectors.Pr[ij][1]); # (l, d) -> (d, l)
        C̃2 = projectors.Pr[i_1j][2] * C2T3; # (u, l)
        @tensor T̃2[ue, de, lc] := projectors.Pr[i_1j][2][ue, α] * T2A[α, β, lc] * projectors.Pr[ij][1][β, de];

        # Update tensors environment
        update_tensors!(unitcell, [C̃1, T̃2, C̃2], RIGHT, ij);

    elseif direction == UP

        """
        C4(i-1,j-1)--   --T1(i-1,j)--   --C1(i-1,j+1)
            |                 |                |
            |                 |                |
        T4(i+s,j-1)--   --R(i+s,j) --   --T2(i+s,j+1)
            |                 |                |

        """

        C4 = unitcell(Environment, ij + CartesianIndex(-1, -1)).C[4];
        T4 = unitcell(Environment, ij + CartesianIndex(step, -1)).T[4];

        T1 = unitcell(Environment, ij + CartesianIndex(-1, 0)).T[1];
        R = unitcell(ReducedTensor, ij + CartesianIndex(step, 0)).R;

        C1 = unitcell(Environment, ij + CartesianIndex(-1, 1)).C[1];
        T2 = unitcell(Environment, ij + CartesianIndex(step, 1)).T[2];

        # Grow environment
        @tensor C4T4[re, rc, de] := C4[re, α] * T4[α, de, rc];
        @tensor T1A[le, lc, re, rc, dc] := T1[re, le, α] * R[α, rc, dc, lc];
        @tensor C1T2[de, le, lc] := C1[α, le] * T2[α, de, lc]; #! indices permuted

        C4T4 = transpose(reshape(C4T4, (:, size(C4T4, 3))));
        C1T2 = transpose(reshape(C1T2, (size(C1T2, 1), :)));
        T1A = reshape(T1A, (prod(size(T1A)[1:2]), prod(size(T1A)[3:4]), size(T1A, 5)));

        # Renormalize
        ij[2] == 1 ? ij_1 = CartesianIndex(ij[1], Nj) : ij_1 = CartesianIndex(ij[1], ij[2]-1);

        C̃4 = transpose(C4T4 * projectors.Pu[ij][1]); #(d,r) -> (r,d)
        C̃1 = transpose(projectors.Pu[ij_1][2] * C1T2); #(l,d) -> (d,l)
        @tensor T̃1[le, re, dc] := projectors.Pu[ij_1][2][le, α] * T1A[α, β, dc] * projectors.Pu[ij][1][β, re];

        # Update tensors environment
        update_tensors!(unitcell, [C̃4, T̃1, C̃1], UP, ij);

    elseif direction == DOWN

        """
            |                 |                |
        T4(i-s,j-1)--   --R(i-s,j)--    --T2(i-s,j+1)
            |                 |                |
            |                 |                |
        C3(i+1,j-1)--   --T3(i+1,j)--   --C2(i+1,j+1)

        """

        C3 = unitcell(Environment, ij + CartesianIndex(1, -1)).C[3];
        T4 = unitcell(Environment, ij + CartesianIndex(-step, -1)).T[4];

        T3 = unitcell(Environment, ij + CartesianIndex(1, 0)).T[3];
        R = unitcell(ReducedTensor, ij + CartesianIndex(-step, 0)).R;

        C2 = unitcell(Environment, ij + CartesianIndex(1, 1)).C[2];
        T2 = unitcell(Environment, ij + CartesianIndex(-step, 1)).T[2];

        # Grow environment
        @tensor C3T4[ue, re, rc] := C3[α, re] * T4[ue, α, rc]; #! indices permuted
        @tensor T3A[le, lc, re, rc, uc] := T3[re, le, α] * R[uc, rc, α, lc];
        @tensor C2T2[ue, le, lc] := C2[α, le] * T2[ue, α, lc];

        C3T4 = reshape(C3T4, (size(C3T4, 1), :));
        C2T2 = transpose(reshape(C2T2, (size(C2T2, 1), :)));
        T3A = reshape(T3A, (prod(size(T3A)[1:2]), prod(size(T3A)[3:4]), size(T3A, 5)));

        # Renormalize
        ij[2] == 1 ? ij_1 = CartesianIndex(ij[1], Nj) : ij_1 = CartesianIndex(ij[1], ij[2]-1);

        C̃3 = C3T4 * projectors.Pd[ij][1]; #(u,r)
        C̃2 = transpose(projectors.Pd[ij_1][2] * C2T2); #(l,u) -> (u,l)
        @tensor T̃3[le, re, uc] := projectors.Pd[ij_1][2][le, α] * T3A[α, β, uc] * projectors.Pd[ij][1][β, re];

        # Update tensors environment
        update_tensors!(unitcell, [C̃2, T̃3, C̃3], DOWN, ij);
    end
end


function projectors_from_identity(
    densitymatrix_U_or_L::Array{T,2},
    densitymatrix_D_or_R::Array{T,2},
    Χmax::Int64;
    ϵmax::Float64 = 0.0) where {T}

    _, R = qr(densitymatrix_U_or_L);
    _, R̃ = qr(densitymatrix_D_or_R);

    U, S, V = svd(R * R̃);

    Χcut  = cutoff(S, Χmax, ϵmax);
    Sinvsqrt = (S[1:Χcut ]).^(-1/2);

    P̃ = R̃ * V[:, 1:Χcut ] * diagm(Sinvsqrt);
    P = diagm(Sinvsqrt) * U'[1:Χcut , :] * R;

    return [P, P̃]
end
