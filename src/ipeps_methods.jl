
"""
    initialize_environment!(uc::UnitCell{X}) where {X}

Initializes environment reduced tensors by contracting legs pointing outwards at each lattice site. See PRB. 84, 041108(R) 2011

"""
function initialize_environment!(uc::UnitCell{X}) where {X<:Union{Float64, ComplexF64}}

    cell_environment = Array{Environment{X}, 2}(undef, uc.dims);

    unique_tensors = unique(uc.pattern);

    # for unit-cell with repeating tensors
    if length(unique_tensors) != length(uc.pattern)
        for tensor_type ∈ unique_tensors
            coords = findall(t -> t == tensor_type, uc.pattern);
            Cs, Ts = generate_environment_tensors(uc, first(coords));
            for coord in coords
                cell_environment[coord] = Environment(Cs, Ts, coord);
            end
        end

    # if unit-cell consists of unique tensors
    else
        for ij ∈ CartesianIndices(uc.dims)
            Cs, Ts = generate_environment_tensors(uc, ij);
            cell_environment[ij] = Environment(Cs, Ts, ij);
        end
    end

    # Initializes field
    uc.E = cell_environment;
end


function braket_unitcell_old(bra::UnitCell{X}, ket::UnitCell{X}) where {X}

    #* Create new state
    braket = UnitCell{X}();
    braket.D = ket.D;
    braket.dims = ket.dims;
    braket.pattern = ket.pattern;
    braket.A = deepcopy(ket.A);
    braket.B = deepcopy(bra.A);

    #* Reconstruct environment
    if ket.D != bra.D
        E = reinitialize_environment(ket, bra);
        braket.E = deepcopy(E);
    else
        initialize_environment!(braket);
    end

    #* Recreate reduced tensors
    R_cell = Array{ReducedTensor{X}, 2}(undef,  ket.dims);
    for ij ∈ CartesianIndices(ket.dims)
        A = ket(Tensor, ij).A;
        B = bra(Tensor, ij).A;

        @tensor R[uk, ub, rk, rb, dk, db, lk, lb] := A[uk, rk, dk, lk, α] * conj(B)[ub, rb, db, lb, α]
        R_cell[ij] = ReducedTensor(reshape(R, (size(A) .* size(B))[1:4]), ket.symmetry);
    end
    braket.R = deepcopy(R_cell);

    return braket

end


function braket_unitcell(bra::UnitCell{X}, ket::UnitCell{X}) where {X}

    #* Recreate reduced tensors
    R_cell = Array{ReducedTensor{X}, 2}(undef,  ket.dims);
    for ij ∈ CartesianIndices(ket.dims)
        A = ket(Tensor, ij).A;
        B = bra(Tensor, ij).A;

        @tensor R[uk, ub, rk, rb, dk, db, lk, lb] := A[uk, rk, dk, lk, α] * conj(B)[ub, rb, db, lb, α]
        R_cell[ij] = ReducedTensor(reshape(R, (size(A) .* size(B))[1:4]), ket.symmetry);
    end
    braket = UnitCell(R_cell);
    braket.A = deepcopy(ket.A);
    braket.B = deepcopy(bra.A);
    braket.D = ket.D;

    #* Reconstruct environment
    if ket.D != bra.D
        reinitialize_environment!(braket);
    else
        initialize_environment!(braket);
    end

    return braket

end

"""
    reinitialize_environment(ket::UnitCell{X}, bra::UnitCell{X}) where {X}

Creates a new environment from two different states as required when calculating overlaps.
"""
function reinitialize_environment!(braket::UnitCell{X}) where {X}

    cell_environment = Array{Environment{X}, 2}(undef, braket.dims);

    #* Create environment tensors
    for ij ∈ CartesianIndices(braket.dims)
        Cs, Ts = generate_environment_tensors(braket, ij);
        cell_environment[ij] = Environment(Cs, Ts, ij);
    end

    braket.E = cell_environment;
end


function generate_isometries_overlap!(isos::Projectors{BraKetOverlap}, state::UnitCell)

    for ij ∈ CartesianIndices(state.dims)
        shifts = [(-1, 0), (0, 1), (1, 0), (0, -1)]; # up, right, down, left

        for dir ∈ [UP, RIGHT, DOWN, LEFT]
            nij = coord(ij + shifts[Int(dir)], state.dims);
            nR = state(ReducedTensor, nij).R;

            # Factorize
            if dir == UP
                nR = permutedims(nR, (3, 1, 2, 4));
                U, S, _ = tensor_svd(nR,[[1], [2, 3, 4]], full_svd = true);
            elseif dir == RIGHT
                nR = permutedims(nR, (4, 1, 2, 3));
                U, S, _ = tensor_svd(nR,[[1], [2, 3, 4]], full_svd = true);
            elseif dir == DOWN
                U, S, _ = tensor_svd(nR,[[1], [2, 3, 4]], full_svd = true);
            elseif dir == LEFT
                nR = permutedims(nR, (2, 1, 3, 4));
                U, S, _ = tensor_svd(nR,[[1], [2, 3, 4]], full_svd = true);
            end

            sqrtS = sqrt(S[1]);
            IA = U[:, 1] * sqrtS; #legs: 1:in, 2:out
            IA = reshape(IA, (state.D, :)); #legs: (1)ket, (2)bra

            Id = diagm(ones(size(IA, 1)));

            # Save isometries
            if dir == UP
                isos.Pu[ij] = [IA, Id];
            elseif dir == RIGHT
                isos.Pr[ij] = [IA, Id];
            elseif dir == DOWN
                isos.Pd[ij] = [IA, Id];
            elseif dir == LEFT
                isos.Pl[ij] = [IA, Id];
            end

        end
    end

end


"""
    generate_environment_tensors_proj(braket::UnitCell, isos::Projectors, loc::CartesianIndex)

Creates the environments tensors at a unit-cell site for the case where the ket and the bra states are different
"""
function generate_environment_tensors_proj(braket::UnitCell, isos::Projectors, loc::CartesianIndex)
    Aij = braket(Tensor, loc).A;
    isdefined(braket, :B) == true ? (Bij = braket(BraTensor, loc).A) : (Bij = Aij);

    @tensor Rij[uk, ub, rk, rb, dk, db, lk, lb] := Aij[uk, rk, dk, lk, α] * conj(Bij)[ub, rb, db, lb, α]; # Contract physical index

    IA_u = isos(UP, loc)[1];    IA_r = isos(RIGHT, loc)[1];    IA_d = isos(DOWN, loc)[1];    IA_l = isos(LEFT, loc)[1];

    @tensor C1[dk, db, lk, lb] := Rij[α, β, γ, δ, dk, db, lk, lb] * IA_u[α, β] * IA_r[γ, δ];
    @tensor C2[uk, ub, lk, lb] := Rij[uk, ub, α, β, γ, δ, lk, lb] * IA_r[α, β] * IA_d[γ, δ];
    @tensor C3[uk, ub, rk, rb] := Rij[uk, ub, rk, rb, α, β, γ, δ] * IA_d[α, β] * IA_l[γ, δ];
    @tensor C4[rk, rb, dk, db] := Rij[γ, δ, rk, rb, dk, db, α, β] * IA_l[α, β] * IA_u[γ, δ];

    @tensor T1[rk, rb, lk, lb, dk, db] := Rij[α, β, rk, rb, dk, db, lk, lb] * IA_u[α, β];
    @tensor T2[uk, ub, dk, db, lk, lb] := Rij[uk, ub, α, β, dk, db, lk, lb] * IA_r[α, β];
    @tensor T3[rk, rb, lk, lb, uk, ub] := Rij[uk, ub, rk, rb, α, β, lk, lb] * IA_d[α, β];
    @tensor T4[uk, ub, dk, db, rk, rb] := Rij[uk, ub, rk, rb, dk, db, α, β] * IA_l[α, β];

    C1 = reshape(C1, (prod(size(C1)[1:2]), :));
    C2 = reshape(C2, (prod(size(C2)[1:2]), :));
    C3 = reshape(C3, (prod(size(C3)[1:2]), :));
    C4 = reshape(C4, (prod(size(C4)[1:2]), :));

    T1 = reshape(T1, (prod(size(T1)[1:2]), prod(size(T1)[3:4]), size(T1, 5), size(T1, 6)));
    T2 = reshape(T2, (prod(size(T2)[1:2]), prod(size(T2)[3:4]), size(T2, 5), size(T2, 6)));
    T3 = reshape(T3, (prod(size(T3)[1:2]), prod(size(T3)[3:4]), size(T3, 5), size(T3, 6)));
    T4 = reshape(T4, (prod(size(T4)[1:2]), prod(size(T4)[3:4]), size(T4, 5), size(T4, 6)));


    Cs = [C1, C2, C3, C4];
    Ts = [T1, T2, T3, T4];

    return Cs, Ts

end

function generate_environment_tensors(unitcell::UnitCell, coord::CartesianIndex)
    Aij = unitcell(Tensor, coord).A;

    if isdefined(unitcell, :B) == true
        Bij = unitcell(BraTensor, coord).A

        @tensor Rij[uk, ub, rk, rb, dk, db, lk, lb] := Aij[uk, rk, dk, lk, α] * conj(Bij)[ub, rb, db, lb, α]; # Contract physical index

        C1 = Rij[1, 1, 1, 1, :, :, :, :];
        C2 = Rij[:, :, 1, 1, 1, 1, :, :];
        C3 = Rij[:, :, :, :, 1, 1, 1, 1];
        C4 = Rij[1, 1, :, :, :, :, 1, 1];

        T1 = Rij[1, 1, :, :, :, :, :, :];
        T2 = Rij[:, :, 1, 1, :, :, :, :];
        T3 = Rij[:, :, :, :, 1, 1, :, :];
        T4 = Rij[:, :, :, :, :, :, 1, 1];
    else
        @tensor Rij[uk, ub, rk, rb, dk, db, lk, lb] := Aij[uk, rk, dk, lk, α] * conj(Aij)[ub, rb, db, lb, α]; # Contract physical index

        @tensor C1[dk, db, lk, lb] := Rij[α, α, β, β, dk, db, lk, lb];
        @tensor C2[uk, ub, lk, lb] := Rij[uk, ub, α, α, β, β, lk, lb];
        @tensor C3[uk, ub, rk, rb] := Rij[uk, ub, rk, rb, α, α, β, β];
        @tensor C4[rk, rb, dk, db] := Rij[α, α, rk, rb, dk, db, β, β];

        @tensor T1[rk, rb, lk, lb, dk, db] := Rij[α, α, rk, rb, dk, db, lk, lb];
        @tensor T2[uk, ub, dk, db, lk, lb] := Rij[uk, ub, α, α, dk, db, lk, lb];
        @tensor T3[rk, rb, lk, lb, uk, ub] := Rij[uk, ub, rk, rb, α, α, lk, lb];
        @tensor T4[uk, ub, dk, db, rk, rb] := Rij[uk, ub, rk, rb, dk, db, α, α];
    end


    C1 = reshape(C1, (prod(size(C1)[1:2]), :));
    C2 = reshape(C2, (prod(size(C2)[1:2]), :));
    C3 = reshape(C3, (prod(size(C3)[1:2]), :));
    C4 = reshape(C4, (prod(size(C4)[1:2]), :));

    T1 = reshape(T1, (prod(size(T1)[1:2]), prod(size(T1)[3:4]), size(T1, 5), size(T1, 6)));
    T2 = reshape(T2, (prod(size(T2)[1:2]), prod(size(T2)[3:4]), size(T2, 5), size(T2, 6)));
    T3 = reshape(T3, (prod(size(T3)[1:2]), prod(size(T3)[3:4]), size(T3, 5), size(T3, 6)));
    T4 = reshape(T4, (prod(size(T4)[1:2]), prod(size(T4)[3:4]), size(T4, 5), size(T4, 6)));


    Cs = [C1, C2, C3, C4];
    Ts = [T1, T2, T3, T4];

    return Cs, Ts

end



"""
    apply_operator(state::UnitCell, op::Operator)

Applies an operator (only single-site at the moment) to an state and updates the simple-update and reduced tensors of the location.

"""
function apply_operator(state::UnitCell, op::Operator)
    op_state = deepcopy(state);
    if op.nsites == 1
        A = op_state(Tensor, op.loc[1]);

        # Apply operator and update S tensor
        @tensor Op_A_M[u, r, d, l, p] := A.A[u, r, d, l, α] * op.O[α, p];
        Op_A = Tensor(Op_A_M);
        op_state.A[op.loc[1]] = Op_A;

        # Update R tensor
        @tensor R[uk, ub, rk, rb, dk, db, lk, lb] := A.A[uk, rk, dk, lk, α] * conj(Op_A.A)[ub, rb, db, lb, α];
        R = reshape(R, (state.D * state.D, state.D * state.D, state.D * state.D, :));
        op_state.R[op.loc[1]] = deepcopy(ReducedTensor(R, UNDEF));

    elseif op.nsite != 1
        @info "Not implemented yet"
    end
    return op_state
end

"""
    calculate_rdm(unitcell::UnitCell, loc::CartesianIndex)


    Contracts:

    C4(x-1, y-1) -- T1(x, y-1) -- C1(x+1, y-1)
        |               |             |
        |               |             |
    T4(x-1, y)   --   R(x,y)   -- T2(x+1, y)
        |               |             |
        |               |             |
    C3(x-1, y+1) -- T3(x, y+1) -- C2(x+1, y+1)

"""
function calculate_rdm(unitcell::UnitCell, loc::CartesianIndex)
    C1 = unitcell(Environment, loc + CartesianIndex(-1, 1)).C[1];
    C2 = unitcell(Environment, loc + CartesianIndex(1, 1)).C[2];
    C3 = unitcell(Environment, loc + CartesianIndex(1, -1)).C[3];
    C4 = unitcell(Environment, loc + CartesianIndex(-1, -1)).C[4];

    T1 = unitcell(Environment, loc  + CartesianIndex(-1, 0)).T[1];
    T2 = unitcell(Environment, loc + CartesianIndex(0, 1)).T[2];
    T3 = unitcell(Environment, loc + CartesianIndex(1, 0)).T[3];
    T4 = unitcell(Environment, loc + CartesianIndex(0, -1)).T[4];

    A = unitcell(Tensor, loc).A;
    if isdefined(unitcell, :B) == true
        B = unitcell(BraTensor, loc).A;
    else
        B = conj(A);
    end


    # cw: clockwise, ccw: counterclockwise, u(k,b): unitcell leg
    @tensor T1C1[ccw, cw, uk, ub] := T1[α, ccw, uk, ub] * C1[cw, α]
    @tensor C4T1C1[ccw, cw, uk, ub] := C4[α, ccw] * T1C1[α, cw, uk, ub]
    @tensor T4C4T1C1[ccw, cw, uk, ub, lk, lb] := T4[α, ccw, lk, lb] * C4T1C1[α, cw, uk, ub]
    @tensor C3T4C4T1C1[ccw, cw, uk, ub, lk, lb] := C3[α, ccw] * T4C4T1C1[α, cw, uk, ub, lk, lb]
    @tensor T3C3T4C4T1C1[ccw, cw, uk, ub, lk, lb, dk, db] := T3[ccw, α, dk, db] * C3T4C4T1C1[α, cw, uk, ub, lk, lb]
    @tensor C2T3C3T4C4T1C1[ccw, cw, uk, ub, lk, lb, dk, db] := C2[ccw, α] * T3C3T4C4T1C1[α, cw, uk, ub, lk, lb, dk, db]
    @tensor E[uk, ub, lk, lb, dk, db, rk, rb] := T2[β, α, rk, rb] * C2T3C3T4C4T1C1[α, β, uk, ub, lk, lb, dk, db]
    E = permutedims(E, (1, 2, 7, 8, 5, 6, 3, 4)); #u,r,d,l

    # Contract with bra and ket tensors
    @tensor rho[pk, pb] := E[αk, αb, βk, βb, γk, γb, δk, δb] * A[αk, βk, γk, δk, pk] * B[αb, βb, γb, δb, pb];

    return rho

end



#= function overlap(ket::UnitCell{T}, bra::UnitCell{T}, ctm_parms::Simulation; init_env::Bool = true, loc::CartesianIndex=CartesianIndex(1,1)) where {T}
    if ket.D != bra.D
        # Reinitialize environments with fused bra-ket legs
        E = reinitialize_environment(ket, bra);

        # Calculate reduced tensors of unit-cell
        R_cell = Array{ReducedTensor{T}, 2}(undef,  ket.dims);

        for xy ∈ CartesianIndices(ket.dims)
            A = ket(Tensor, xy);
            B = bra(Tensor, xy);
            @tensor R[uk, ub, rk, rb, dk, db, lk, lb] := A.A[uk, rk, dk, lk, α] * B.A[ub, rb, db, lb, α];
            R = reshape(R, (ket.D * bra.D, ket.D * bra.D, ket.D * bra.D, :));
            R_cell[xy] = ReducedTensor(R, UNDEF);
        end

        braket = UnitCell(R_cell);
        braket.E = deepcopy(E);

    else ket.D == ket.D

        # Calculate reduced tensors of unit-cell
        R_cell = Array{ReducedTensor{T}, 2}(undef,  ket.dims);

        for xy ∈ CartesianIndices(ket.dims)
            A = ket(Tensor, xy);
            B = bra(Tensor, xy);
            @tensor R[uk, ub, rk, rb, dk, db, lk, lb] := A.A[uk, rk, dk, lk, α] * B.A[ub, rb, db, lb, α];
            R = reshape(R, (ket.D^2, ket.D^2, ket.D^2, :));
            R_cell[xy] = ReducedTensor(R, UNDEF);
        end

        braket = UnitCell(R_cell);

        # Recreate environment tensors
        if init_env == true
            E = Array{Environment{T}, 2}(undef, braket.dims);

            for xy ∈ CartesianIndices(braket.dims)
                E[xy] = generate_environment_tensors(braket, xy);
            end

            braket.E = deepcopy(E);
        else
            braket.E = deepcopy(ket.E)
        end


    end

    # Converge environment
    projectors = Projectors{EachMove}(braket);
    error_CTM = update_environment!(braket, projectors, ctm_parms);

    # Contract
    λ = do_full_contraction(braket, loc)

    return λ
end =#

#=
"""
    do_full_contraction(unitcell::UnitCell, loc::CartesianIndex)

    Contracts:

                C4(x-1, y-1) -- T1(x, y-1) -- C1(x+1, y-1)
                    |               |             |
                    |               |             |
                T4(x-1, y)   --   R(x,y)   -- T2(x+1, y)
                    |               |             |
                    |               |             |
                C3(x-1, y+1) -- T3(x, y+1) -- C2(x+1, y+1)

"""
function do_full_contraction(unitcell::UnitCell, loc::CartesianIndex)
    C1 = unitcell(Environment, loc + CartesianIndex(1, -1)).C[1];
    C2 = unitcell(Environment, loc + CartesianIndex(1, 1)).C[2];
    C3 = unitcell(Environment, loc + CartesianIndex(-1, 1)).C[3];
    C4 = unitcell(Environment, loc + CartesianIndex(-1, -1)).C[4];

    T1 = unitcell(Environment, loc  + CartesianIndex(0, -1)).T[1];
    T2 = unitcell(Environment, loc + CartesianIndex(1, 0)).T[2];
    T3 = unitcell(Environment, loc + CartesianIndex(0, 1)).T[3];
    T4 = unitcell(Environment, loc + CartesianIndex(-1, 0)).T[4];


    R = unitcell(ReducedTensor, loc).R;

    # cw: clockwise, ccw: counterclockwise, uc: unitcell leg
    @tensor T1C1[ccw, cw, u] := T1[α, ccw, u] * C1[cw, α]
    @tensor C4T1C1[ccw, cw, u] := C4[α, ccw] * T1C1[α, cw, u]
    @tensor T4C4T1C1[ccw, cw, u, l] := T4[α, ccw, l] * C4T1C1[α, cw, u]
    @tensor C3T4C4T1C1[ccw, cw, u, l] := C3[α, ccw] * T4C4T1C1[α, cw, u, l]
    @tensor T3C3T4C4T1C1[ccw, cw, u, l, d] := T3[ccw, α, d] * C3T4C4T1C1[α, cw, u, l]
    @tensor C2T3C3T4C4T1C1[ccw, cw, u, l, d] := C2[ccw, α] * T3C3T4C4T1C1[α, cw, u, l, d]
    @tensor E[u, l, d, r] := T2[β, α, r] * C2T3C3T4C4T1C1[α, β, u, l, d]
    E = permutedims(E, (1, 4, 3, 2)); #u,r,d,l


    # Contract with reduced tensor
    λ = 0.0;
    @tensor λ = E[u, r, d, l] * R[u, r, d, l]

    return λ

end =#

#=
"""
    do_full_contraction(unitcell::UnitCell, loc::CartesianIndex)

    Contracts:

                C4(x-1, y-1) -- T1(x, y-1) -- C1(x+1, y-1)
                    |               |             |
                    |               |             |
                T4(x-1, y)   --   R(x,y)   -- T2(x+1, y)
                    |               |             |
                    |               |             |
                C3(x-1, y+1) -- T3(x, y+1) -- C2(x+1, y+1)

"""
function do_full_contraction_bigmem(unitcell::UnitCell, loc::CartesianIndex)
    C1 = unitcell(Environment, loc + CartesianIndex(1, -1)).C[1];
    C2 = unitcell(Environment, loc + CartesianIndex(1, 1)).C[2];
    C3 = unitcell(Environment, loc + CartesianIndex(-1, 1)).C[3];
    C4 = unitcell(Environment, loc + CartesianIndex(-1, -1)).C[4];

    T1 = unitcell(Environment, loc  + CartesianIndex(0, -1)).T[1];
    T2 = unitcell(Environment, loc + CartesianIndex(1, 0)).T[2];
    T3 = unitcell(Environment, loc + CartesianIndex(0, 1)).T[3];
    T4 = unitcell(Environment, loc + CartesianIndex(-1, 0)).T[4];


    R = unitcell(ReducedTensor, loc).R;

    @tensor C4T4C3[ur, cr, dr] := C4[ur, α] * T4[α, β, cr] * C3[β, dr];
    @tensor T1RT3[ul, cl, dl, ur, cr, dr] := T1[ur, ul, α] * R[α, cr, β, cl] * T3[dr, dl, β];
    @tensor C1T2C2[ul, cl, dl] := C1[α, ul] * T2[α, β, cl] * C2[β, dl];

    λ = 0;
    @tensor λ = C4T4C3[α, β, γ] * T1RT3[α, β, γ, δ, ϵ, ζ] * C1T2C2[δ, ϵ, ζ];

    return λ
end =#


###############
# Old methods #
###############

#= """
    implode(R::Array{T, 4}, E::Environment) where {T}


    Contracts:

                C4 -- T1 -- C1
                |     |      |
                T4 -- R1 -- T2
                |     |      |
                C3 -- T3 -- C2
"""
function implode(R::Array{T, 4}, E::Environment) where {T}
    # cw: clockwise, ccw: counterclockwise, uc: unitcell leg
    @tensor TCu[ccw, cw, u] := E.T[1][α, ccw, u] * E.C[1][cw, α]
    @tensor Eu[ccw, cw, u] := E.C[4][α, ccw] * TCu[α, cw, u]

    @tensor TCd[ccw, cw, d] := E.T[3][α, ccw, d] * E.C[2][ccw, α]
    @tensor Ed[ccw, cw, d] := E.C[3][cw, α] * TCd[α, ccw, d]

    # Contracts environment cyclicically??
    @tensor Eur[ccw, cw, u, r] := Eu[ccw, α, u] * E.T[2][α, cw, r]
    @tensor Eurd[ccw, cw, u, r, d] := Eur[ccw, α, u, r] * Ed[α, cw]
    @tensor Eurdl[u, r, d, l] := Eurd[α, β, u, r, d] * E.T[4][α, β, l]

    # Contract with reduced tensor
    RE = 0.0;
    @tensor RE = Eurdl[u, r, d, l] * R[u, r, d, l]

    return RE

end


"""
    implode(R::Vector{Array{T, 4}}, E::Environment)

    Contracts:

                C4 -- T1 -- C1
                |     |      |
                T6 -- R1 -- T2
                |     |      |
                T5 -- R2 -- T3
                |     |      |
                C3 -- T4 -- C2

    by contracting first the upper and lower halfs independently.

## Notes
    For two tensors aligned horizontally, the environment has to be rotated before contraction

"""
function implode(R::Vector{Array{T, 4}}, E::Environment) where {T}

    # Half-cell upper environment
    @tensor TCu[ccw, cw, u] :=  E.T[1][α, ccw, u] * E.C[1][cw, α] #T1C1
    @tensor Eu[ccw, cw, u] :=  E.C[4][α, ccw] * TCu[α, cw, u] #C4T1C1
    @tensor Eur[ccw, cw, u, r] :=  Eu[α, ccw, u] * E.T[2][α, cw, r] #C4T1C1T2
    @tensor Eurl[ccw, cw, u, r, l] := E.T[6][α, ccw, l] * Eur[α, cw, u, r]  #T6C4T1C1T2

    # Half-cell lower environment
    @tensor TCd[ccw, cw, d] :=  E.T[4][α, ccw, d] * E.C[2][ccw, α] #T4C2
    @tensor Ed[ccw, cw, d] :=  E.C[3][cw, α] * TCd[α, ccw, d] #C3T4C2
    @tensor Edr[ccw, cw, d, r] := Ed[α, cw, d] * E.T[3][ccw, α, r]; #C3T4C2T3
    @tensor Edrl[ccw, cw, d, r, l] := E.T[5][ccw, α, l] * Edr[α, cw, d, r]

    # Contract with reduced tensors
    @tensor EuR1[ccw, cw, d] := Eurl[ccw, cw, α, β, γ] * R[1][α, β, γ, d]
    @tensor EdR2[ccw, cw, u] := Edrl[ccw, cw, β, α, γ] * R[2][u, α, β, γ]

    RE = 0.0;
    @tensor RE = EuR1[α, β, γ] * EdR2[β, α, γ];

    return RE

end =#


#= function calculate_exp_val(unitcell::UnitCell, operator::Operator{X}) where {X}
    # Extract environment
    # Apply operator to ket layer
    # Contract

    if operator.nsites == 1
        T = cast_tensor(Tensor, unitcell.S[operator.loc[1]]);
        @tensor TO[u, r, d, l, p] := T[u, r, d, l, α] * operator.O[α, p];
        @tensor RO[u, r, d, l] := TO[uk, r, d, ];

        @tensor TOT[uk, ub, rk, rb, dk, db, lk, lb] := TO[uk, rk, dk, lk, α] * conj(T)[ub, rb, db, lb, α]
        TOT = reshape(TOT, (T.D^2, T.D^2, T.D^2, T.D^2));

        exp_val = implode(TOT, unitcell.E[operator.loc[1]]);

    elseif operator.nsites == 2 # only two-site NN case
        @info "Only for NN operators"
        T1 = cast_tensor(Tensor, unitcell.S[operator.loc[1]]);
        T2 = cast_tensor(Tensor, unitcell.S[operator.loc[2]]);
        E1 = unitcell.E[operator.loc[1]];
        E2 = unitcell.E[operator.loc[2]];

        if operator.loc[1][1] == operator.loc[2][1] # operator along horizontal bonds
            @tensor T1T2O[u1, u2, r2, d1, d2, l1, p1, p2] := T1[u1, α, d1, l1, β] *
            T2[u2, r2, d2, α, γ] * operator.O[p1, p2, β, γ];

            E1E2 = Environment(
            [E1.C[4] E2.C[1] E2.C[2] E1.C[3]],
            [E1.T[4]; E1.T[1]; E2.T[1:3]; E1.T[3]], operator.loc[1]); # environment is rotated CW π/2

            exp_val = implode(T1T2O, E1E2);

        elseif operator.loc[1][2] == operator.loc[2][2] # operator along vertical bonds
            @tensor T1T2O[u1, r1, r2, d2, l1, l2, p1, p2] := T1[u1, r1, α, l1, β] *
            T2[α, r2, d2, l2, γ] * operator.O[p1, p2, β, γ];

            E1E2 = Environment(
            [E1.C[1] E2.C[2] E2.C[3] E1.C[4]],
            [E1.T[1:2]; E2.T[2:4]; E1.T[4]], operator.loc[1]);

            exp_val = implode(T1T2O, E1E2);

        end

    end

    return exp_val
end =#
