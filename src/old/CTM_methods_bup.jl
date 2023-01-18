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

    do_ctmrg_iteration!(unitcell, projectors);
    while ϵ > simulation.tol_ctm
        i += 1;

        Eref = deepcopy(unitcell.E)
        do_ctmrg_iteration!(unitcell, projectors);
        ϵ = calculate_error_ctm(Eref, unitcell.E);

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
    unitcell::UnitCell{T},
    projectors::Projectors{EachMove}) where {T<:Union{Float64, ComplexF64}}

    Nx = unitcell.dims[1];
    Ny = unitcell.dims[2];

    # Makes a copy of the initial full cell environment to be used when updating each tensor environment
    initial_environment = deepcopy(unitcell.E);
    new_environment = Array{Environment{T}, 2}(undef, unitcell.dims);

    ##### Left move #####
    for xy ∈ CartesianIndices(unitcell.dims)
        unitcell.E = deepcopy(initial_environment); # Reset environment to initial one
        @debug "Updating environment tensors at $xy"

        #= Left move for every unit-cell tensor. Sweeps from left to right, column by column =#
        for j ∈ 1:Ny
            @debug "Left move, adding column $i"
            # 1) Calculate all projectors along the column, i.e. all P_(i, j+n) for fixed j+n
            for i ∈ 0:Nx-1
                loc = coord(xy + (i, j), unitcell.dims);
                calculate_projectors_ctmrg!(unitcell, projectors, loc, LEFT); # P_(i,j)
            end

            # 2) Absorb column j+n tensors in environment tensors of all tensors (i,j) with fixed j and renormalize
            for i ∈ 0:Nx-1
                loc = coord(xy + (i, j), unitcell.dims);
                do_ctm_move!(unitcell, projectors, LEFT, loc);
            end
        end

        # Updates new environment
        new_environment[xy] = deepcopy(unitcell.E[xy]);
    end

    initial_environment = deepcopy(new_environment);

    ##### Right move #####
    for xy ∈ CartesianIndices(unitcell.dims)
        unitcell.E = deepcopy(initial_environment); # Reset environment to initial one
        #= Right move for every unit-cell tensor. Sweeps from right to left, column by column =#
        for j ∈ 1:Ny
            @debug "Right move, adding column $i"
            # 1) Calculate all projectors for the column, i.e. all P_(i, j+n) for fixed j+n
            for i ∈ 0:Nx-1
                loc = coord(xy + (i, -j), unitcell.dims);
                calculate_projectors_ctmrg!(unitcell, projectors, loc, RIGHT); # P_(i,j)
            end

            # 2) Absorb column j-n tensors in environment tensors of all tensors (i,j) with fixed j and renormalize
            for i ∈ 0:Nx-1
                loc = coord(xy + (i, -j), unitcell.dims);
                do_ctm_move!(unitcell, projectors, RIGHT, loc);
            end
        end

        # Updates new environment
        new_environment[xy] = deepcopy(unitcell.E[xy]);
    end

    initial_environment = deepcopy(new_environment);

    ##### Up move #####
    for xy ∈ CartesianIndices(unitcell.dims)
        unitcell.E = deepcopy(initial_environment); # Reset environment to initial one
        #= Up move for every unit-cell tensor. Sweeps from top to bottom, row by row =#
        for i ∈ 1:Nx
            @debug "Up move, adding row $j"
            # 1) Calculate all projectors for the row, i.e. all P_(i+n, j) for fixed i+n
            for j ∈ 0:Ny-1
                loc = coord(xy + (i, j), unitcell.dims);
                calculate_projectors_ctmrg!(unitcell, projectors, loc, UP); # P_(i,j)
            end

            # 2) Absorb row i+n tensors in environment tensors of all tensors (i,j) with fixed i and renormalize
            for j ∈ 0:Ny-1
                loc = coord(xy + (i, j), unitcell.dims);
                do_ctm_move!(unitcell, projectors, UP, loc);
            end
        end

        # Updates new environment
        new_environment[xy] = deepcopy(unitcell.E[xy]);
    end

    initial_environment = deepcopy(new_environment);

    ##### Down move #####
    for xy ∈ CartesianIndices(unitcell.dims)
        unitcell.E = deepcopy(initial_environment); # Reset environment to initial one

        #= Down move for every unit-cell tensor. Sweeps from bottom to top, row by row =#
        for i ∈ 1:Nx
            @debug "Down move, adding row $j"
            # 1) Calculate all projectors for the row, i.e. all P_(i+n, j) for fixed i+n
            for j ∈ 0:Ny-1
                loc = coord(xy + (-i, j), unitcell.dims);
                calculate_projectors_ctmrg!(unitcell, projectors, loc, DOWN); # P_(i,j)
            end

            # 2) Absorb row i-n tensors in environment tensors of all tensors (i,j) with fixed i and renormalize
            for j ∈ 0:Ny-1
                loc = coord(xy + (-i, j), unitcell.dims);
                do_ctm_move!(unitcell, projectors, DOWN, loc);
            end
        end

        # Updates new environment
        new_environment[xy] = deepcopy(unitcell.E[xy]);
    end

    unitcell.E = deepcopy(new_environment);
end


function grow_T_tensor(T::Array{X,4}, Aket::Array{X,5}, Abra::Array{X,5}, dir::Direction) where {X<:Union{Float64, ComplexF64}}
    if dir == UP
        @tensor Tket[re, rk, le, lk, dk, db, p] := T[re, le, α, db] * Aket[α, rk, dk, lk, p];
        @tensor Tbraket[re, rk, rb, le, lk, lb, dk, db] := Tket[re, rk, le, lk, dk, α, β] * conj(Abra)[α, rb, db, lb, β];

    elseif dir == RIGHT
        @tensor Tket[ue, uk, de, dk, lk, lb, p] := T[ue, de, α, lb] * Aket[uk, α, dk, lk, p];
        @tensor Tbraket[ue, uk, ub, de, dk, db, lk, lb] := Tket[ue, uk, de, dk, lk, α, β] * conj(Abra)[ub, α, db, lb, β];

    elseif dir == DOWN
        @tensor Tket[re, rk, le, lk, uk, ub, p] := T[re, le, α, ub] * Aket[uk, rk, α, lk, p];
        @tensor Tbraket[re, rk, rb, le, lk, lb, uk, ub] := Tket[re, rk, le, lk, uk, α, β] * conj(Abra)[ub, rb, α, lb, β];

    elseif dir == LEFT
        @info "Size T and A", size(T), size(Aket)
        @tensor Tket[ue, uk, de, dk, rk, rb, p] := T[ue, de, α, rb] * Aket[uk, rk, dk, α, p];
        @tensor Tbraket[ue, uk, ub, de, dk, db, rk, rb] := Tket[ue, uk, de, dk, rk, α, β] * conj(Abra)[ub, rb, db, α, β];

    end

    return Tbraket
end

function do_ctm_move!(unitcell::UnitCell, projectors::Projectors, direction::Direction, loc::CartesianIndex)
    E_loc = unitcell(Environment, loc);
    A = unitcell(Tensor, loc).A;
    isdefined(unitcell, :B) == true ? (B = unitcell(BTensor, loc).A) : (B = conj(A));

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

        E_add = unitcell(Environment, loc + (0, -1));

        # Grow environment
        @tensor C4T1[re, de, dk, db] := E_add.C[4][α, de] * E_loc.T[1][re, α, dk, db]
        @tensor C3T3[ue, uk, ub, re] := E_add.C[3][ue, α] * E_loc.T[3][re, α, uk, ub]
        T4AB = grow_T_tensor(E_add.T[4], A, B, LEFT);

        C4T1 = reshape(C4T1, (size(C4T1, 1), :));
        C3T3 = reshape(C3T3, (:, size(C3T3, 4)));
        T4AB = reshape(T4AB, (prod(size(T4AB)[1:3]), prod(size(T4AB)[4:6]), size(T4AB, 7), size(T4AB, 8)));


        # Renormalize
        P̃ = projectors(LEFT, loc)[1];
        P = projectors(LEFT, loc + (-1, 0))[2];

        C̃4 = C4T1 * P̃; #(r, d)
        C̃3 = P * C3T3; #(u, r)
        @tensor T̃4[ue, de, rk, rb] := P[ue, α] * T4AB[α, β, rk, rb] * P̃[β, de];

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

        E_add = unitcell(Environment, loc + (0, 1));

        # Grow environment
        @tensor C1T1[de, dk, db, le] := E_add.C[1][de, α] * E_loc.T[1][α, le, dk, db];
        @tensor C2T3[ue, uk, ub, le] := E_add.C[2][ue, α] * E_loc.T[3][α, le, uk, ub]; #! indices permuted
        T2AB = grow_T_tensor(E_add.T[2], A, B, RIGHT);

        C1T1 = transpose(reshape(C1T1, (:, size(C1T1, 4))));
        C2T3 = reshape(C2T3, (:, size(C2T3, 4)));
        T2AB = reshape(T2AB, (prod(size(T2AB)[1:3]), prod(size(T2AB)[4:6]), size(T2AB, 7), size(T2AB, 8)));


        # Renormalize
        P̃ = projectors(RIGHT, loc)[1];
        P = projectors(RIGHT, loc + (-1, 0))[2];

        C̃1 = transpose(C1T1 * P̃); # (l, d) -> (d, l)
        C̃2 = P * C2T3; # (u, l)
        @tensor T̃2[ue, de, lk, lb] := P[ue, α] * T2AB[α, β, lk, lb] * P̃[β, de];

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

        E_add = unitcell(Environment, loc + (-1, 0));

        # Grow environment
        @tensor C4T4[re, rk, rb, de] := E_add.C[4][re, α] * E_loc.T[4][α, de, rk, rb];
        @tensor C1T2[de, le, lk, lb] := E_add.C[1][α, le] * E_loc.T[2][α, de, lk, lb]; #! indices permuted
        T1AB = grow_T_tensor(E_add.T[1], A, B, UP);


        C4T4 = transpose(reshape(C4T4, (:, size(C4T4, 4))));
        C1T2 = transpose(reshape(C1T2, (size(C1T2, 1), :)));
        T1AB = reshape(T1AB, (prod(size(T1AB)[1:3]), prod(size(T1AB)[4:6]), size(T1AB, 7), size(T1AB, 8)));

        # Renormalize
        P̃ = projectors(UP, loc)[1];
        P = projectors(UP, loc + (0, -1))[2];

        C̃4 = transpose(C4T4 * P̃); #(d,r) -> (r,d)
        C̃1 = transpose(P * C1T2); #(l,d) -> (d,l)
        @tensor T̃1[re, le, dk, db] := P[le, α] * T1AB[α, β, dk, db] * P̃[β, re];

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

        E_add = unitcell(Environment, loc + (1, 0));

        # Grow environment
        @tensor C3T4[ue, re, rk, rb] := E_add.C[3][α, re] * E_loc.T[4][ue, α, rk, rb]; #! indices permuted
        @tensor C2T2[ue, le, lk, lb] := E_add.C[2][α, le] * E_loc.T[2][ue, α, lk, lb];
        T3AB = grow_T_tensor(E_add.T[3], A, B, DOWN);

        C3T4 = reshape(C3T4, (size(C3T4, 1), :));
        C2T2 = transpose(reshape(C2T2, (size(C2T2, 1), :)));
        T3AB = reshape(T3AB, (prod(size(T3AB)[1:3]), prod(size(T3AB)[4:6]), size(T3AB, 7), size(T3AB, 8)));

        # Renormalize
        P̃ = projectors(DOWN, loc)[1];
        P = projectors(DOWN, loc + (0, -1))[2];

        C̃3 = C3T4 * P̃; #(u,r)
        C̃2 = transpose(P * C2T2); #(l,u) -> (u,l)
        @tensor T̃3[re, le, uk, ub] := P[le, α] * T3AB[α, β, uk, ub] * P̃[β, re];

        # Update tensors environment
        update_tensors!(unitcell, [C̃2, T̃3, C̃3], DOWN, loc);
    end
end


function calculate_projectors_ctmrg!(
    uc::UnitCell,
    projectors::Projectors,
    loc::CartesianIndex,
    direction::Direction;
    Χ::Int64=0)

    if direction == UP

        # Calculate half-system density matrix
        Q4, Q1 = calculate_enlarged_corners(uc, loc, UP)
        HU = Q4 * Q1;

        # Calculate projectors
        if Χ == 0
            Χ = size(uc(Environment, loc).T[1], 1);
        end

        U, Sinvsqrt, V, S = factorize_rho(HU, Χ);
        P̃ = Q1 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q4;

        projectors.Pu[loc] = [P̃, P];

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S;
        uc.E[loc].spectra[1] = S_f;


    elseif direction == RIGHT

        # Calculate half-system density matrix
        Q1, Q2 = calculate_enlarged_corners(uc, loc, RIGHT)
        HR = Q1 * Q2;

        # Calculate projectors
        if Χ == 0
            Χ = size(uc(Environment, loc).T[2], 2);
        end

        U, Sinvsqrt, V, S = factorize_rho(HR, Χ)
        P̃ = Q2 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q1;

        projectors.Pr[loc] = [P̃, P];

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S;
        uc.E[loc].spectra[2] = S_f;

    elseif direction == DOWN

        # Calculate half-system density matrix
        Q3, Q2 = calculate_enlarged_corners(uc, loc, DOWN)
        HD = Q3 * Q2;

        if Χ == 0
            Χ = size(uc(Environment, loc).T[3], 1);
        end

        U, Sinvsqrt, V, S = factorize_rho(HD, Χ)
        P̃ = Q2 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q3;

        projectors.Pd[loc] = [P̃, P];

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S;
        uc.E[loc].spectra[3] = S_f;

    elseif direction == LEFT

        # Calculate half-system density matrix
        Q4, Q3 = calculate_enlarged_corners(uc, loc, LEFT)
        HL = Q4 * Q3;

        # Calculate projectors
        if Χ == 0
            Χ = size(uc(Environment, loc).T[4], 2);
        end

        U, Sinvsqrt, V, S = factorize_rho(HL, Χ)
        P̃ = Q3 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q4;
        projectors.Pl[loc] = [P̃, P];

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S;
        uc.E[loc].spectra[4] = S_f;

    end


end

function calculate_enlarged_corners(
    uc::UnitCell,
    loc::CartesianIndex,
    direction::Direction)

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

        ##### Q4 #####
        # Load tensors
        @info "Loc error", loc
        C4 = uc(Environment, loc + (-1, -1)).C[4];
        T1 = uc(Environment, loc  + (-1, 0)).T[1];
        T4 = uc(Environment, loc + (0, -1)).T[4];
        A = uc(Tensor, loc).A;
        isdefined(uc, :B) == true ? (B = uc(BTensor, loc).A) : (B = conj(A));

        # Build enlarged corner
        ABT4 = grow_T_tensor(T4, A, B, LEFT);
        @tensor ABT4C4[re, uk, ub, de, dk, db, rk, rb] := C4[re, α] * ABT4[α, uk, ub, de, dk, db, rk, rb];
        @tensor Q4[re, rk, rb, de, dk, db] := ABT4C4[α, β, γ, de, dk, db, rk, rb] * T1[re, α, β, γ];
        Q4 = reshape(Q4, (prod(size(Q4)[1:3]), :));


        ##### Q3 #####
        # Load tensors
        C3 = uc(Environment, loc + (2, -1)).C[3];
        T3 = uc(Environment, loc + (2, 0)).T[3];
        T4 = uc(Environment, loc + (1, -1)).T[4];
        A = uc(Tensor, loc + (1, 0)).A;
        isdefined(uc, :B) == true ? (B = uc(BTensor, loc + (1, 0)).A) : (B = conj(A));


        # Build enlarged corner
        ABT4 = grow_T_tensor(T4, A, B, LEFT);
        @tensor ABT4C3[re, ue, uk, ub, dk, db, rk, rb] := C3[α, re] * ABT4[ue, uk, ub, α, dk, db, rk, rb];
        @tensor Q3[ue, uk, ub, re, rk, rb] := T3[re, α, β, γ] * ABT4C3[α, ue, uk, ub, β, γ, rk, rb];
        Q3 = reshape(Q3, (prod(size(Q3)[1:3]), :));

        return Q4, Q3

    elseif direction == RIGHT

        """
        -- T1(x,y-1) --  C1(x+1,y-1)
              |             |
              |             |
        -- R(x,y)    -- T2(x+1,y)
              |             |


              |             |
        -- R(x,y+1)  --  T2(x+1,y+1)
              |             |
              |             |
        -- T3(x,y+2) --  C2(x+1,y+2)

        """

        ##### Q1 #####
        # Load tensors
        C1 = uc(Environment, loc + (-1, 1)).C[1];
        T1 = uc(Environment, loc + (-1, 0)).T[1];
        T2 = uc(Environment, loc + (0, 1)).T[2];
        A = uc(Tensor, loc).A;
        isdefined(uc, :B) == true ? (B = uc(BTensor, loc).A) : (B = conj(A));

        # Build enlarged corner
        ABT2 = grow_T_tensor(T2, A, B, RIGHT);
        @tensor ABT2C1[le, uk, ub, de, dk, db, lk, lb] := C1[α, le] * ABT2[α, uk, ub, de, dk, db, lk, lb];
        @tensor Q1[le, lk, lb, de, dk, db] := T1[α, le, β, γ] * ABT2C1[α, β, γ, de, dk, db, lk, lb];
        Q1 = reshape(Q1, (prod(size(Q1)[1:3]), :));

        ##### Q2 #####
        # Load tensors
        C2 = uc(Environment, loc + (2, 1)).C[2];
        T3 = uc(Environment, loc + (2, 0)).T[3];
        T2 = uc(Environment, loc + (1, 1)).T[2];
        A = uc(Tensor, loc + (1, 0)).A;
        isdefined(uc, :B) == true ? (B = uc(BTensor, loc + (1, 0)).A) : (B = conj(A));

        # Build enlarged corner
        ABT2 = grow_T_tensor(T2, A, B, RIGHT);
        @tensor ABT2C2[le, ue, uk, ub, dk, db, lk, lb] := C2[α, le] * ABT2[ue, uk, ub, α, dk, db, lk, lb];
        @tensor Q2[ue, uk, ub, le, lk, lb] := T3[α, le, β, γ] * ABT2C2[α, ue, uk, ub, β, γ, lk, lb];
        Q2 = reshape(Q2, (prod(size(Q2)[1:3]), :));

        return Q1, Q2

    elseif direction == UP

        """
         C4(x-1,y-1) --  T1(x,y-1)--     -- T1(x+1,y-1) --  C1(x+2,y-1)
            |                |                 |                |
            |                |                 |                |
         T4(x-1,y)   --    R(x,y) --     --  R(x+1,y)   --   T2(x+2,y)
             |               |                 |                |

        """

        ##### Q4 #####
        # Load tensors
        C4 = uc(Environment, loc + (-1, -1)).C[4]; #!
        T1 = uc(Environment, loc + (-1, 0)).T[1];
        T4 = uc(Environment, loc + (0, -1)).T[4];
        A = uc(Tensor, loc).A;
        isdefined(uc, :B) == true ? (B = uc(BTensor, loc).A) : (B = conj(A));


        # Build enlarged corner #! copied from left move
        ABT4 = grow_T_tensor(T4, A, B, LEFT);
        @tensor ABT4C4[re, uk, ub, de, dk, db, rk, rb] := C4[re, α] * ABT4[α, uk, ub, de, dk, db, rk, rb];
        @tensor Q4[de, dk, db, re, rk, rb] := ABT4C4[α, β, γ, de, dk, db, rk, rb] * T1[re, α, β, γ];
        Q4 = reshape(Q4, (prod(size(Q4)[1:3]), :));


        ##### Q1 #####
        # Load tensors
        C1 = uc(Environment, loc + (-1, 2)).C[1];
        T1 = uc(Environment, loc + (-1, 1)).T[1];
        T2 = uc(Environment, loc + (0, 2)).T[2];
        A = uc(Tensor, loc + (0, 1)).A;
        isdefined(uc, :B) == true ? (B = uc(BTensor, loc + (0, 1)).A) : (B = conj(A));

        # Build enlarged corner #! copied from right move
        ABT2 = grow_T_tensor(T2, A, B, RIGHT);
        @tensor ABT2C1[le, uk, ub, de, dk, db, lk, lb] := C1[α, le] * ABT2[α, uk, ub, de, dk, db, lk, lb];
        @tensor Q1[le, lk, lb, de, dk, db] := T1[α, le, β, γ] * ABT2C1[α, β, γ, de, dk, db, lk, lb];
        Q1 = reshape(Q1, (prod(size(Q1)[1:3]), :));

        return Q4, Q1

    elseif direction == DOWN

        """
            |              |                    |               |
        T4(x-1,y)   --  R(x,y)   --     --  R(x+1,y)   --    T2(x+2,y)
            |              |                    |               |
            |              |                    |               |
        C3(x-1,y+1) -- T3(x,y+1) --     -- T3(x+1,y+1) --  C2(x+2,y+1)

        """

        ##### Q3 #####
        # Load tensors
        C3 = uc(Environment, loc + (1, -1)).C[3];
        T3 = uc(Environment, loc + (1, 0)).T[3];
        T4 = uc(Environment, loc + (0, -1)).T[4];
        A = uc(Tensor, loc).A;
        isdefined(uc, :B) == true ? (B = uc(BTensor, loc).A) : (B = conj(A));

        # Build enlarged corner #! Copied from left move
        ABT4 = grow_T_tensor(T4, A, B, LEFT);
        @tensor ABT4C3[re, ue, uk, ub, dk, db, rk, rb] := C3[α, re] * ABT4[ue, uk, ub, α, dk, db, rk, rb];
        @tensor Q3[ue, uk, ub, re, rk, rb] := T3[re, α, β, γ] * ABT4C3[α, ue, uk, ub, β, γ, rk, rb];
        Q3 = reshape(Q3, (prod(size(Q3)[1:3]), :));

        ##### Q2 #####
        # Load tensors
        C2 = uc(Environment, loc + (1, 2)).C[2];
        T3 = uc(Environment, loc + (1, 1)).T[3];
        T2 = uc(Environment, loc + (0, 2)).T[2];
        A = uc(Tensor, loc + (0, 1)).A;
        isdefined(uc, :B) == true ? (B = uc(BTensor, loc + (0, 1)).A) : (B = conj(A));

        # Build enlarged corner #! Copied from right move
        ABT2 = grow_T_tensor(T2, A, B, RIGHT);
        @tensor ABT2C2[le, ue, uk, ub, dk, db, lk, lb] := C2[α, le] * ABT2[ue, uk, ub, α, dk, db, lk, lb];
        @tensor Q2[le, lk, lb, ue, uk, ub] := T3[α, le, β, γ] * ABT2C2[α, ue, uk, ub, β, γ, lk, lb];
        Q2 = reshape(Q2, (prod(size(Q2)[1:3]), :));

        return Q3, Q2

    end

end


function update_tensors!(uc::UnitCell, tensors::Vector{T}, direction::Direction, loc::CartesianIndex; renormalize::Bool = true) where {T<:AbstractArray}


    if direction == UP
        pos = [4, 1, 1];
    elseif direction == RIGHT
        pos = [1, 2, 2];
    elseif direction == DOWN
        pos = [2, 3, 3];
    elseif direction == LEFT
        pos = [3, 4, 4];
    end

    if renormalize == true
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

function factorize_rho(rho::Array{T,2}, Χ::Int64) where {T<:Union{Float64, ComplexF64}}

    U, S, V = svd(rho);
    Χ > length(S) && (Χ = length(S);)
    Winvsqrt = diagm(S[1:Χ].^(-1/2));

    return U[:, 1:Χ], Winvsqrt, V[:, 1:Χ], S[1:Χ]/maximum(S[1:Χ])
    #return U[:, 1:Χ], Winvsqrt, V[:, 1:Χ]
end


function calculate_error_ctm(Eref::Array{Environment{T},2}, Eupd::Array{Environment{T},2}) where {T}

    ϵ = 0.0;
    for xy ∈ CartesianIndices(size(Eref))
        ϵ += sum([sum(abs.(Eupd[xy].spectra[n] - Eref[xy].spectra[n])) for n ∈ 1:4])
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

#= function calculate_projectors_ctmrg_old!(
    uc::UnitCell,
    projectors::Projectors,
    loc::CartesianIndex,
    direction::Direction;
    Χ::Int64=0)

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
        C4 = uc(Environment, loc + (-1, -1)).C[4];
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
        if Χ == 0
            Χ = size(uc(Environment, loc).T[4], 2);
        end

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
        if Χ == 0
            Χ = size(uc(Environment, loc).T[2], 2);
        end

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
        if Χ == 0
            Χ = size(uc(Environment, loc).T[1], 1);
        end

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

        if Χ == 0
            Χ = size(uc(Environment, loc).T[3], 1);
        end

        U, Sinvsqrt, V, S = factorize_rho(HD, Χ)
        P̃ = Q2 * V * Sinvsqrt;
        P = Sinvsqrt * U' * Q3;

        projectors.Pd[loc] = [P̃, P];

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S;
        uc.E[loc].spectra[3] = S_f;

    end


end =#

#= function do_ctm_move_old!(unitcell::UnitCell, projectors::Projectors, direction::Direction, loc::CartesianIndex)
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
        @debug "Left move at $loc adding $(coord(loc + CartesianIndex(-1, 0), unitcell.dims))"

        # Grow environment
        @tensor C4T1[re, de, dc] := E_add.C[4][α, de] * E_loc.T[1][re, α, dc]
        @tensor T4A[ue, uc, de, dc, rc] := E_add.T[4][ue, de, α] * R_loc.R[uc, rc, dc, α]
        @tensor C3T3[ue, uc, re] := E_add.C[3][ue, α] * E_loc.T[3][re, α, uc]

        T4A = reshape(T4A, (prod(size(T4A)[1:2]), prod(size(T4A)[3:4]), size(T4A, 5)));
        C4T1 = reshape(C4T1, (size(C4T1, 1), :));
        C3T3 = reshape(C3T3, (:, size(C3T3, 3)));

        #! debug
        E_loc.TS[4] = E_add.TS[4] * "_l" * string(Tuple(loc));

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
        @debug "Right move at $loc adding $(coord(loc + CartesianIndex(1, 0), unitcell.dims))"

        # Grow environment
        @tensor C1T1[de, dc, le] := E_add.C[1][de, α] * E_loc.T[1][α, le, dc];
        @tensor T2A[ue, uc, de, dc, lc] := E_add.T[2][ue, de, α] * R_loc.R[uc, α, dc, lc];
        @tensor C2T3[ue, uc, le] := E_add.C[2][ue, α] * E_loc.T[3][α, le, uc]; #! indices permuted

        C1T1 = transpose(reshape(C1T1, (:, size(C1T1, 3))));
        C2T3 = reshape(C2T3, (:, size(C2T3, 3)));
        T2A = reshape(T2A, (prod(size(T2A)[1:2]), prod(size(T2A)[3:4]), size(T2A, 5)));

        #! debug
        E_loc.TS[2] = E_add.TS[2] * "_r" * string(Tuple(loc));
        #@info "T2 env", E_loc.TS[2]

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

        #! debug
        E_loc.TS[1] = E_add.TS[1] * "_u" * string(Tuple(loc));

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

        #! debug
        E_loc.TS[3] = E_add.TS[3] * "_d" * string(Tuple(loc));

        # Renormalize

        P̃ = projectors(DOWN, loc)[1];
        P = projectors(DOWN, loc + CartesianIndex(-1, 0))[2];

        C̃3 = C3T4 * P̃; #(u,r)
        C̃2 = transpose(P * C2T2); #(l,u) -> (u,l)
        @tensor T̃3[re, le, uc] := P[le, α] * T3A[α, β, uc] * P̃[β, re];

        # Update tensors environment
        update_tensors!(unitcell, [C̃2, T̃3, C̃3], DOWN, loc);
    end
end =#
