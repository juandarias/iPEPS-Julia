
"""
function (uc::UnitCell)(i, j; reduced::Bool = false)

Functor to return tensors of unit cell. Probably will become obsolete soon.

### Arguments
### Returns
"""
function (uc::UnitCell)(i, j; reduced::Bool = false)

    if size(uc.pattern) != uc.dims
        Ni = size(uc.pattern, 1);
        Nj = size(uc.pattern, 2);
        reduced == true && return uc.R[mod(i -1 , Ni) + 1, mod(j - 1, Nj) + 1]
        reduced == false && return uc.S[mod(i -1 , Ni) + 1, mod(j - 1, Nj) + 1]
    else
        reduced == true && return uc.R[i,j];
        reduced == false && return uc.S[i,j];
    end

end


"""
    initialize_environment!(uc::UnitCell{X}) where {X}

Initializes environment reduced tensors by contracting legs pointing outwards at each lattice site. See PRB. 84, 041108(R) 2011

"""
function initialize_environment!(uc::UnitCell{X}) where {X}

    cell_environment = Array{Environment{X}, 2}(undef, uc.dims);

    unique_tensors = unique(uc.pattern);

    # for unit-cell with repeating tensors
    if length(unique_tensors) != length(uc.pattern)
        for tensor_type ∈ unique_tensors
            coords = findall(t -> t == tensor_type, uc.pattern);
            Cs, Ts = generate_environment_tensors(uc, first(coords));
            for coord in coords
                cell_environment[coord] = Environment(Cs, Ts, Tuple(coord));
            end
        end

    # if unit-cell consists of unique tensors
    else
        for i ∈ 1:uc.dims[1], j ∈ 1:uc.dims[2]
            Cs, Ts = generate_environment_tensors(uc, CartesianIndex(i, j));
            cell_environment[i, j] = Environment(Cs, Ts, (i, j));
        end
    end

    # Initializes field
    uc.E = cell_environment;
end


function generate_environment_tensors(unitcell::UnitCell{U}, coord::CartesianIndex; fuse_bra_ket::Bool = true) where {U}

    D = unitcell.S[coord].D;
    Aij = cast_tensor(Tensor, unitcell.S[coord].S);
    @tensor Rij[uk, ub, rk, rb, dk, db, lk, lb] := Aij[uk, rk, dk, lk, α] * conj(Aij)[ub, rb, db, lb, α]; # Contract physical index

    @tensor C1[dk, db, lk, lb] := C4[α, α, β, β, dk, db, lk, lb];
    @tensor C2[uk, ub, lk, lb] := C4[uk, ub, α, α, β, β, lk, lb];
    @tensor C3[uk, ub, rk, rb] := C4[uk, ub, rk, rb, α, α, β, β];
    @tensor C4[rk, rb, dk, db] := C4[α, α, rk, rb, dk, db, β, β];

    @tensor T1[rk, rb, lk, lb, dk, db] := C4[α, α, rk, rb, dk, db, lk, lb];
    @tensor T2[uk, ub, dk, db, lk, lb] := C4[uk, ub, α, α, dk, db, lk, lb];
    @tensor T3[rk, rb, lk, lb, uk, ub] := C4[uk, ub, rk, rb, α, α, lk, lb];
    @tensor T4[uk, ub, dk, db, rk, rb] := C4[uk, ub, rk, rb, dk, db, α, α];


    if fuse_bra_ket == true
        C1 = reshape(C1, (D^2, D^2));    C2 = reshape(C2, (D^2, D^2));    C3 = reshape(C3, (D^2, D^2));    C4 = reshape(C4, (D^2, D^2));

        T1 = reshape(T1, (D^2, D^2, D^2));    T2 = reshape(T2, (D^2, D^2, D^2));    T3 = reshape(T3, (D^2, D^2, D^2));    T4 = reshape(T4, (D^2, D^2, D^2));
    end

    Cs = [C1, C2, C3, C4];
    Ts = [T1, T2, T3, T4];

    return Cs, Ts
end

function prepare_su_tensor(S::SimpleUpdateTensor, gate_direction::Direction)

    gate_direction == UP ? (n = 3) : (gate_direction == RIGHT ?
    (n = 2) : (gate_direction == DOWN ? (n = 1) : (n = 0)));
    order = [circshift([1, 2, 3, 4], n)..., 5];

    S.S = permutedims(S.S, order); # permute legs such that aux. leg towards gate is before last one
    weights = diagm.(S.weights[order[1:3]]);

    # Absorb weights
    @tensor M[1, 2, 3, g, p] := S.S[α, β, γ, g, p] * weights[1][α, 1] *
        weights[2][β, 2] * weights[3][γ, 3];


    # Factorize
    M = reshape(M, (S.D^3, S.D * S.d));
    Q, R = qr(M);
    R = reshape(Matrix(R), (:, S.D, S.d));
    Q = reshape(Matrix(Q), (S.D, S.D, S.D, :));

    return Q, R
end

function restore_su_tensor(Q::Array{X, 4}, R::Array{T,3}, weights::Vector{Vector{Float64}}, gate_direction::Direction) where {T,X}

    gate_direction == UP ? (n = 3) : (gate_direction == RIGHT ?
    (n = 2) : (gate_direction == DOWN ? (n = 1) : (n = 0)));
    order = circshift([1, 2, 3, 4], n);

    #x⁻¹(x) = x.^(-1)
    inv_weights = [diagm(weights[order[n]].^-1) for n ∈ 1:3]

    @tensor S[1, 2, 3, 4, p] := Q[α, β, γ, δ] * inv_weights[1][α, 1] * inv_weights[2][β, 2] *
    inv_weights[3][γ, 3] * R[δ, p, 4];


    # Restore order of legs
    order = [circshift([1, 2, 3, 4], -n)..., 5];
    S = permutedims(S, order);
    return S
end

function apply_gate(G::Array{S, 4}, RA::Array{T, 3}, RB::Array{T, 3}, W::Array{Float64,2}, Dmax::Int64) where {T,S}
    @tensor RRGw[l, u1, u2, r] := RA[l, γ, α] * G[u1, u2, α, β] * W[γ, δ] * RB[r, δ, β];

    # Split and truncate
    RRGw = reshape(RRGw, (prod(size(RRGw)[1:2]), :));
    RAg, Wg, RBg = svd(RRGw);
    R̃A = reshape(RAg[:, 1:Dmax], (size(RA, 1), 2, Dmax)); #! (l,d,Dmax)
    R̃B = reshape(RBg[:, 1:Dmax], (2, size(RB, 1), Dmax)); #! (d,r,Dmax)
    W̃ = Wg[1:Dmax];

    return R̃A, R̃B, W̃
end


function update_cell!(unitcell::UnitCell, S::Array{T, 5}, weights::Vector{Float64}, label::Char, direction::Direction) where {T}
    coords = findall(t -> t == label, unitcell.pattern);
    for coord in coords
        unitcell.S[coord].S = S/norm(S); #! normalize
        unitcell.S[coord].weights[Int(direction)] = weights/sqrt(sum(weights.^2));
    end

end

function calculate_exp_val(unitcell::UnitCell, operator::Operator{X}) where {X}
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
end


"""
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
    @tensor TCu[ccw, cw, u] :=  E.T[1][α, ccw, u] * E.C[1][cw, α]
    @tensor Eu[ccw, cw, u] :=  E.C[4][α, ccw] * TCu[α, cw, u]

    @tensor TCd[ccw, cw, d] :=  E.T[3][α, ccw, d] * E.C[2][ccw, α]
    @tensor Ed[ccw, cw, d] :=  E.C[3][cw, α] * TCd[α, ccw, d]

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

end


##############################
# Old and to be discontinued #
##############################

function initialize_random_environment!(uc::UnitCell{X}, Χ::Int64; unique_environments::Bool = false) where {X}

    cell_environment = Array{Environment{X}, 2}(undef, uc.dims);

    if unique_environments == false #* uses the same environment for all tensors. Helps with convergence

        Cs, Ts = generate_environment_tensors(uc, Χ);
        for i ∈ 1:uc.dims[1], j ∈ 1:uc.dims[2]
            cell_environment[i, j] = Environment(Cs, Ts, (i, j));
        end

    else #* for each unique tensor initializes an environment
        unique_tensors = unique(uc.pattern);

        # for unit-cell with repeating tensors
        if length(unique_tensors) != length(uc.pattern)
            for tensor_type ∈ unique_tensors
                Cs, Ts = generate_environment_tensors(uc, Χ);
                coords = findall(t -> t == tensor_type, uc.pattern);
                for coord in coords
                    cell_environment[coord] = Environment(Cs, Ts, Tuple(coord));
                end
            end

        # if unit-cell consists of unique tensors
        else
            for i ∈ 1:uc.dims[1], j ∈ 1:uc.dims[2]
                Cs, Ts = generate_environment_tensors(uc, Χ);
                cell_environment[i, j] = Environment(Cs, Ts, (i, j));
            end
        end
    end

    # Initializes field
    uc.E = cell_environment;
end

function generate_random_environment_reduced(uc::UnitCell{U}, Χ::Int64) where {U}
    D = uc.D;
    Χsqrt = Int64(sqrt(Χ));

    uc.symmetry == C4 && (C_max = 1; T_max = 1;)
    uc.symmetry == XY && (C_max = 1; T_max = 2;)
    uc.symmetry == UNDEF && (C_max = 4; T_max = 4;)

    Cs = Array{U, 2}[];
    for n ∈ 1:C_max
        Ci = rand(U, Χsqrt, Χsqrt, 2);

        # Symmetrize and normalize
        uc.symmetry == C4 && (Ci = Ci + permutedims(Ci, (2, 1, 3)));
        Ci = Ci + conj(Ci);
        normalize!(Ci);

        # Create reduced environment tensors
        @tensor C[k1, b1, k2, b2] := Ci[k1, k2, α] * conj(Ci)[b1, b2, α];
        push!(Cs, reshape(C, Χ, Χ));
    end

    Ts = Array{U, 3}[];
    for m ∈ 1:T_max
        Ti = rand(U, Χsqrt, Χsqrt, D, 2);

        # Symmetrize and normalize
        uc.symmetry != UNDEF && (Ti = Ti + permutedims(Ti, (2, 1, 3, 4)));
        Ti = Ti + conj(Ti);
        normalize!(Ti);

        # Create reduced environment tensors
        @tensor T[k1, b1, k2, b2, kc, bc] := Ti[k1, k2, kc, α] * conj(Ti)[b1, b2, bc, α];

        push!(Ts, reshape(T, Χ, Χ, D^2));
    end

    if uc.symmetry == C4
        Cs = [Cs[1], Cs[1], Cs[1], collect(transpose(Cs[1]))];
        Ts = fill(Ts[1], 4);
    elseif uc.symmetry == XY
        Cs = [Cs[1], Cs[1], Cs[1], collect(transpose(Cs[1]))];
        Ts = [Ts; Ts];
    end

    return Cs,Ts
end

function generate_random_environment(uc::UnitCell{U}, Χ::Int64) where {U}
    D = uc.D;
    #Χsqrt = Int64(sqrt(Χ));

    uc.symmetry == C4 && (C_max = 1; T_max = 1;)
    uc.symmetry == XY && (C_max = 1; T_max = 2;)
    uc.symmetry == UNDEF && (C_max = 4; T_max = 4;)

    Cs = Array{U, 2}[];
    for n ∈ 1:C_max
        Ci = rand(U, Χ, Χ);

        # Symmetrize and normalize
        uc.symmetry == C4 && (Ci = Ci + transpose(Ci));
        Ci = Ci + conj(Ci);
        Ci = Ci/opnorm(Ci);
        push!(Cs, Ci)
    end

    Ts = Array{U, 3}[];
    for m ∈ 1:T_max
        Ti = rand(U, Χ, Χ, D^2);

        # Symmetrize and normalize
        uc.symmetry != UNDEF && (Ti = Ti + permutedims(Ti, (2, 1, 3)));
        Ti = Ti + conj(Ti);
        normalize!(Ti);
        push!(Ts, Ti);
    end

    if uc.symmetry == C4
        Cs = [Cs[1], Cs[1], Cs[1], collect(transpose(Cs[1]))];
        Ts = fill(Ts[1], 4);
    elseif uc.symmetry == XY
        Cs = [Cs[1], Cs[1], Cs[1], collect(transpose(Cs[1]))];
        Ts = [Ts; Ts];
    end

    return Cs,Ts
end

#=
function initialize_environment!(uc::UnitCell{T}, Χ::Int64) where {T}
    cell_environment = Array{Environment{T}, 2}(undef, uc.dims);
    unique_tensors = unique(uc.pattern);
    D = uc.D;

    if length(unique_tensors) != length(uc.pattern) # for unit-cell with repeating tensors
        for n ∈ eachindex(unique_tensors)
            coords = findall(t -> t == unique_tensors[n], uc.pattern);
            En = Environment{T}((nothing, nothing), Χ, D);
            for coord in coords
                En.loc = Tuple(coord);
                cell_environment[coord] = En;
            end
        end

    else # if unit-cell consists of no repeating tensors
        for i ∈ 1:uc.dims[1], j ∈ 1:uc.dims[2]
            cell_environment[i, j] = Environment{T}((i,j), Χ, D);
        end
    end

    # Initializes field
    uc.E = cell_environment;
end
 =#
