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
    while ϵ > simulation.tol_ctm
        i += 1;
        old_cell = deepcopy(unitcell);
        do_ctm_iteration!(Type{unitcell.symmetry}, unitcell, projectors);

        ϵ = calculate_error_ctm(old_cell, unitcell, simulation);

        @info "CTM iteration $i using $(typeof(projectors)).
        Convergence error = $(round(abs(ϵ), sigdigits = 4))";

        if i == simulation.max_ctm_steps
            @warn "!!! Maximum number of iterations reached !!!"
            simulation.conv_ctm_steps = i;
            simulation.ctm_error = ϵ;
            #return ϵ
        end
    end

    simulation.conv_ctm_steps = i;
    simulation.ctm_error = ϵ;
    #return ϵ
end


function calculate_error_ctm(old::UnitCell, new::UnitCell, simulation::Simulation)
    unique_tensors = unique(old.pattern);
    coord_unique = [findfirst(t -> t == type_tensor, old.pattern) for type_tensor ∈ unique_tensors];

    ϵ = 0.0;
    if simulation.ctm_convergence == OnlyCorners
        for coord ∈ coord_unique
            if new.S[coord].symmetry == C4
                ϵ += sum([abs(tr(new.E[coord].C[n] - old.E[coord].C[n])) for n ∈ 1:4]);
            else new.S[coord].symmetry == XY
                ϵ += sum([opnorm(new.E[coord].C[n] - old.E[coord].C[n]) for n ∈ 1:4]);
            end
        end

    elseif simulation.ctm_convergence == Full
        for coord ∈ coord_unique
            ϵ += sum([abs(tr(new.E[coord].C[n] - old.E[coord].C[n])) for n ∈ 1:4]);
            ϵ += sum([abs(tr(new.E[coord].T[n][:, :, i] - old.E[coord].T[n][:, :, i])) for i ∈ 1:2, n ∈ 1:4]);
        end
    end

    return ϵ/length(coord_unique)
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
    ::Type{S},
    unitcell::UnitCell,
    projectors::Projectors) where {S<:Type{UNDEF}}

    # u,d,l,r: direction of bond. e,c: environment or cell bond. E.g: bond re corresponds to a right-environment bond
    # greek letters for contracted symbols

    Χ = unitcell.E[1,1].Χ;
    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];

    for i ∈ 1:Ni, j ∈ 1:Nj

        E_loc = uc.E[i, j];

        #= Left move. Absorbs unit-cell tensors and environment tensors of the whole column, row by row.=#
        for n ∈ 0:Nj - 1
            #@info "Left move, step $(n+1)"
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

        end

        #= Right move. Absorbs unit-cell tensors and environment tensors of the whole column, row by row =#
        for n ∈ Nj - 1:-1:0
            #@info "Right move, step $(n+1)"

                j_n = mod(j + n + Nj - 1, Nj) + 1; # jₙ = 1 + mod(j + nⱼ - 1, nⱼ)

                E_add = uc.E[i, j_n];
                R_add = uc.R[i, j_n];

                # Grow environment
                @tensor C1T1[de, dc, le] := E_loc.C[1][de, α] * E_add.T[1][α, le, dc];
                @tensor T2A[ue, uc, de, dc, lc] := E_loc.T[2][ue, de, α] * R_add.R[uc, α, dc, lc];
                @tensor C2T3[ue, uc, le] := E_loc.C[2][ue, α] * E_add.T[3][α, le, uc]; #! indices permuted

                # Renormalize
                calc_projectors!(unitcell, projectors, [i,j], RIGHT, n, Χ)

                C1T1 = transpose(reshape(C1T1, (:, size(C1T1, 3))));
                T2A = reshape(T2A, (prod(size(T2A)[1:2]), prod(size(T2A)[3:4]), size(T2A, 5)));

                C̃1 = transpose(C1T1 * projectors.Pr[1]); # (l, d) -> (d, l)
                C̃2 = projectors.Pr[2] * reshape(C2T3, (:, size(C2T3, 3))); # (u, l)
                @tensor T̃2[ue, de, lc] := projectors.Pr[2][ue, α] * T2A[α, β, lc] * projectors.Pr[1][β, de];

            # Update tensors environment
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

        end

        #= Down move. Absorbs unit-cell tensors and environment tensors of the whole row, column by column =#
        for n ∈ Ni - 1:-1:0
            #@info "Down move, step $(n+1)"
            i_n = mod(i + n + Ni - 1, Ni) + 1;

            E_add = uc.E[i_n, j];
            R_add = uc.R[i_n, j];

            # Grow environment
            @tensor C3T4[ue, re, rc] := E_loc.C[3][α, re] * E_add.T[4][ue, α, rc]; #! indices permuted
            @tensor T3A[le, lc, re, rc, uc] := E_loc.T[3][re, le, α] * R_add.R[uc, rc, α, lc];
            @tensor C2T2[ue, le, lc] := E_loc.C[2][α, le] * E_add.T[2][ue, α, lc];


            # Renormalize
            calc_projectors!(unitcell, projectors, [i,j], DOWN, n, Χ)

            C2T2 = transpose(transpose(reshape(C2T2, (size(C2T2, 1), :))));
            T3A = reshape(T3A, (prod(size(T3A)[1:2]), prod(size(T3A)[3:4]), size(T3A, 5)));


            C̃3 = reshape(C3T4, (size(C3T4, 1), :)) * projectors.Pd[1]; #(u,r)
            C̃2 = transpose(projectors.Pd[2] * C2T2); #(l,u) -> (u,l)
            @tensor T̃3[le, re, uc] := projectors.Pd[2][le, α] * T3A[α, β, uc] * projectors.Pd[1][β, re];

            # Update tensors environment
            update_tensors!(unitcell, [C̃2, T̃3, C̃3], DOWN, [i,j]);

        end

    end
end


function do_ctm_iteration!(
    ::Type{S},
    unitcell::UnitCell,
    projectors::Projectors) where {S<:Union{Type{C4}, Type{XY}}}

    # u,d,l,r: direction of bond. e,c: environment or cell bond. E.g: bond re corresponds to a right-environment bond
    # greek letters for contracted symbols

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


function update_tensors!(uc::UnitCell, tensors::Vector{T}, direction::Direction, loc::Vector{Int64}; normalize::Bool = true) where {T<:AbstractArray}

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
        #TC(A) = A/opnorm(A);
        #TT(A) = A/norm(A);
        uc.E[loc...].C[pos[1]] = tensors[1]/opnorm(tensors[1]);
        uc.E[loc...].T[pos[2]] = tensors[2]/norm(tensors[2]);
        uc.E[loc...].C[pos[3]] = tensors[3]/opnorm(tensors[3]);
    else
        uc.E[loc...].C[pos[1]] = collect(tensors[1]);
        uc.E[loc...].T[pos[2]] = collect(tensors[2]);
        uc.E[loc...].C[pos[3]] = collect(tensors[3]);
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

function calc_projectors!(
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
    projectors::Projectors{TwoCornersSimple},
    loc::Vector{Int64},
    direction::Direction,
    step::Int64,
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

        loc_step = loc
        loc_step[2] = mod(loc_step[2] + step - 1, uc.dims[2]) + 1;

        E_loc = uc.E[loc...];
        E_add = uc.E[loc_step...];
        R_add = uc.R[loc_step...];

        @tensor C4T4T1A[ure, urc, de, dc] := E_loc.C[4][α, δ] * E_loc.T[4][δ, de, γ] * E_add.T[1][ure, α, β] * R_add.R[β, urc, dc, γ];
        C4T4T1A = reshape(C4T4T1A, (size(C4T4T1A, 1) * size(C4T4T1A, 2), :));

        if uc.symmetry == XY || uc.symmetry == C4
            C3T4T3A = C4T4T1A'
        else
            @tensor C3T4T3A[ue, uc, dre, drc] := E_loc.C[3][α, δ] * E_loc.T[4][ue, α, β] * E_add.T[3][dre, δ, γ] * R_add.R[uc, drc, γ, β];
            C3T4T3A = reshape(C3T4T3A, (size(C3T4T3A, 1) * size(C3T4T3A, 2), :));
        end

        HL = C4T4T1A * C3T4T3A
        U, Sinvsqrt, V = factorize_rho(HL, Χ, uc.symmetry)
        P̃ = C3T4T3A * V * Sinvsqrt;
        P = Sinvsqrt * U' * C4T4T1A;

        projectors.Pl = [P̃, P];

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

        loc_step = loc;
        loc_step[2] = mod(loc_step[2] + step + uc.dims[2] - 1, uc.dims[2]) + 1;

        E_loc = uc.E[loc...];
        E_add = uc.E[loc_step...];
        R_add = uc.R[loc_step...];

        @tensor C1T2T1A[ule, ulc, de, dc] := E_loc.C[1][α, δ] * E_loc.T[2][α, de, β] * E_add.T[1][δ, ule, γ] * R_add.R[γ, β, dc, ulc];
        C1T2T1A = reshape(C1T2T1A, (size(C1T2T1A, 1) * size(C1T2T1A, 2), :));

        if uc.symmetry == XY || uc.symmetry == C4
            C2T3T2A = C1T2T1A';
        else
            @tensor C2T3T2A[ue, uc, dle, dlc] := E_loc.C[2][α, β] * E_add.T[3][β, dle, γ] * E_loc.T[2][ue, α, δ] * R_add.R[uc, δ, γ, dlc];
            C2T3T2A = reshape(C2T3T2A, (size(C2T3T2A, 1) * size(C2T3T2A, 2), :));
        end

        HR = C1T2T1A * C2T3T2A;
        U, Sinvsqrt, V = factorize_rho(HR, Χ, uc.symmetry)
        P̃ = C2T3T2A * V * Sinvsqrt;
        P = Sinvsqrt * U' * C1T2T1A;

        projectors.Pr = [P̃, P];

    elseif direction == UP

        loc_step = loc
        loc_step[1] = mod(loc_step[1] + step - 1, uc.dims[1]) + 1;

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
        E_add = uc.E[loc_step...];
        R_add = uc.R[loc_step...];

        @tensor C4T1T4A[lde, ldc, re, rc] := E_loc.C[4][α, δ] * E_loc.T[1][re, α, β] * E_add.T[4][δ, lde, γ] * R_add.R[β, rc, ldc, γ];
        C4T1T4A = reshape(C4T1T4A, (size(C4T1T4A, 1) * size(C4T1T4A, 2), :));


        if uc.symmetry == XY || uc.symmetry == C4
            C1T1T2A = C4T1T4A';
        else
            @tensor C1T1T2A[le, lc, rde, rdc] := E_loc.C[1][α, δ] * E_add.T[2][α, rde, β] * E_loc.T[1][δ, le, γ] * R_add.R[γ, β, rdc, lc]
            C1T1T2A = reshape(C1T1T2A, (size(C1T1T2A, 1) * size(C1T1T2A, 2), :));
        end

        HU = C4T1T4A * C1T1T2A;
        U, Sinvsqrt, V = factorize_rho(HU, Χ, uc.symmetry)
        P̃ = C1T1T2A * V * Sinvsqrt;
        P = Sinvsqrt * U' * C4T1T4A;

        projectors.Pu = [P̃, P];

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

        loc_step = loc;
        loc_step[1] = mod(loc_step[1] + step + uc.dims[1] - 1, uc.dims[1]) + 1;

        E_loc = uc.E[loc...];
        E_add = uc.E[loc_step...];
        R_add = uc.R[loc_step...];

        @tensor C3T4T3A[lue, luc, re, rc] := E_loc.C[3][α, δ] * E_add.T[4][lue, α, β] * E_loc.T[3][re, δ, γ] * R_add.R[luc, rc, γ, β]
        C3T4T3A = reshape(C3T4T3A, (size(C3T4T3A, 1) * size(C3T4T3A, 2), :));

        if uc.symmetry == XY || uc.symmetry == C4
            C2T3T2A = C3T4T3A';
        else
            @tensor C2T3T2A[le, lc, rue, ruc] := E_loc.C[2][α, β] * E_loc.T[3][β, le, γ] * E_add.T[2][rue, α, δ] * R_add.R[ruc, δ, γ, lc];
            C2T3T2A = reshape(C2T3T2A, (size(C2T3T2A, 1) * size(C2T3T2A, 2), :));
        end

        HD = C3T4T3A * C2T3T2A;
        U, Sinvsqrt, V = factorize_rho(HD, Χ, uc.symmetry)
        P̃ = C2T3T2A * V * Sinvsqrt;
        P = Sinvsqrt * U' * C3T4T3A;

        projectors.Pd = [P̃, P];

    end
end


####################
# Helper functions #
####################

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

function factorize_rho(rho::Array{T,2}, Χ::Int, symmetry::LatticeSymmetry) where {T}
    if symmetry == XY || symmetry == C4
        #@info "Is hermitian $(rho ≈ rho')"
        Λ, U = eigen(Hermitian(rho), sortby= x -> -1 * real(x))
        Winvsqrt = diagm(Λ[1:Χ].^(-1/2));
        return U[:, 1:Χ], Winvsqrt, U[:, 1:Χ]
    else
        U, S, V = svd(rho);
        Winvsqrt = diagm(S[1:Χ].^(-1/2));
        return U[:, 1:Χ], Winvsqrt, V[:, 1:Χ]
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
