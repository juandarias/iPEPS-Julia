################################################
# Directional CTM (10.1103/PhysRevB.80.094403) #
################################################

"""
function update_environment!(unitcell::UnitCell, projectors::Projectors, simulation::Simulation)

    Performs CTM iterations till convergence of the environment. Convergence is verified by
    comparing eigenvalues of all environment tensors (`Full`) or just the corner tensors (`OnlyCorners`).

"""
function update_environment!(unitcell::UnitCell, projectors::Projectors, simulation::Simulation)
    #criteria::ConvergenceCriteria,
    #tol::Float64;
    #max_iters::Int64 = 50)

    #* Calculates trace of all environment tensors, needed to calculate convergence
    unique_tensors = unique(unitcell.pattern);
    C_tr = zeros(4, length(unique_tensors)); T_tr = zeros(4, length(unique_tensors));

    for (t, type_tensor) ∈ enumerate(unique_tensors)
        coord = findfirst(t -> t == type_tensor, unitcell.pattern);
        C_tr[:, t] = [tr(unitcell.E[coord].C[n]) for n ∈ 1:4];
        T_tr[:, t] = [tr(unitcell.E[coord].T[n]) for n ∈ 1:4];
    end

    ϵ = 0.0;
    i = 0
    while ϵ > simulation.tol_ctm
        i += 1;
        do_ctm_iteration!(unitcell, projectors);

        if simulation.ctm_convergence == OnlyCorners
            #ϵ =  sum([C_tr[n] - tr(environment.C[n]) for n ∈ 1:4]);
            for (t, type_tensor) ∈ enumerate(unique_tensors)
                coord = findfirst(t -> t == type_tensor, unitcell.pattern);
                ϵ += sum([C_tr[n, t] - tr(unitcell.E[coord].C[n]) for n ∈ 1:4]);
            end

        elseif simulation.ctm_convergence == Full
            #ϵ = sum([C_tr[n] - tr(environment.C[n])  + T_tr[n] -
            #tr(environment.T[n]) for n ∈ 1:4]);

            for (t, type_tensor) ∈ enumerate(unique_tensors)
                coord = findfirst(t -> t == type_tensor, unitcell.pattern);
                ϵ += sum([C_tr[n, t] - tr(unitcell.E[coord].C[n]) for n ∈ 1:4]);
                ϵ += sum([T_tr[n, t] - tr(unitcell.E[coord].T[n]) for n ∈ 1:4]);
            end

        end

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
    projectors::Projectors{T}) where {T<:Renormalization}

    # u,d,l,r: direction of bond. e,c: environment or cell bond. E.g: bond re corresponds to a right-environment bond
    # greek letters for contracted symbols

    Χ = unitcell.E[1,1].Χ;
    Ni = unitcell.dims[1];
    Nj = unitcell.dims[2];

    for i ∈ 1:Ni, j ∈ 1:Nj

        #= Left move. Absorbs unit-cell tensors and environment tensors of the whole column, row by row.=#
        for n ∈ 0:Nj - 1
            # Grow environment
            @tensor C4T1[re, de, dc] := unitcell.E[i, j].C[4][α, de] * unitcell.E[i, j + n].T[1][re, α, dc]
            @tensor T4A[ue, uc, de, dc, rc] := unitcell.E[i, j].T[4][ue, de, α] * unitcell.R[i, j + n][uc, rc, dc, α]
            @tensor C3T3[ue, uc, re] := unitcell.E[i, j].C[3][ue, α] * unitcell.E[i, j + n].T[3][α, re, uc]

            # Renormalize
            #! Calculate projectors here
            calc_projectors!(unitcell, projectors, [i,j], "left", n, Χ)

            C̃4 = reshape(C4T1, (size(C4T1, 1), :)) * projectors.Pl[2];
            C̃3 = projectors.Pl[1] * reshape(C3T3, (:, size(C3T3, 3)));
            T4A = reshape(T4A, (prod(size(T4A)[1:2]), prod(size(T4A)[3:4], :)));
            @tensor T̃4[ue, de, rc] := projectors.Pl[1][ue, α] * T4A[α, β, rc] * projectors.Pl[2][β, de];

            # Update environment
            environment.C[4] = symmetrize(C̃4);
            environment.C[3] = symmetrize(C̃3);
            environment.T[4] = symmetrize(T̃4);
        end

        #= Right move. Absorbs unit-cell tensors and environment tensors of the whole column, row by row =#
        for n ∈ Nj - 1:-1:0
            # Grow environment
            @tensor C1T1[le, de, dc] := unitcell.E[i, j].C[1][de, α] * unitcell.E[i, j + n].T[1][α, dc, le];
            @tensor T2A[ue, uc, de, dc, lc] := unitcell.E[i, j].T[2][ue, de, α] * unitcell.R[i,j + n][uc, α, dc, lc];
            @tensor C2T3[ue, uc, le] := unitcell.E[i,j].C[2][ue, α] * unitcell.E[i, j + n].T[3][uc, α, le];

            # Renormalize
            calc_projectors!(unitcell, projectors, [i,j], "right", n, Χ)

            C̃1 = reshape(C1T1, (size(C1T1, 1), :)) * projectors.Pr[2];
            C̃2 = projectors.Pr[1] * reshape(C2T3, (:, size(C2T3, 3)));
            T2A = reshape(T2A, (prod(size(T2A)[1:2]), prod(size(T2A)[3:4]), :));
            @tensor T̃2[ue, de, lc] := projectors.Pr[1][ue, α] * T2A[α, β, lc] * projectors.Pr[2][β, de];

            # Update environment
            environment.C[1] = symmetrize(C̃1);
            environment.C[2] = symmetrize(C̃2);
            environment.T[2] = symmetrize(T̃2);

        end


        #= Up move. Absorbs unit-cell tensors and environment tensors of the whole row, column by column =#
        for n ∈ 0:Ni - 1
            # Grow environment
            @tensor C4T4[de, re, rc] := unitcell.E[i, j].C[4][re, α] * unitcell.E[i + n, j].T[4][α, rc, de];
            @tensor T1A[le, lc, re, rc, dc] := unitcell.E[i, j].T[1][le, re, α] * unitcell.R[i + n, j][α, rc, dc, lc];
            @tensor C1T2[le, lc, de] := unitcell.E[i,j].C[1][le, α] * unitcell.E[i + n, j].T[2][α, de, lc];

            # Renormalize
            calc_projectors!(unitcell, projectors, [i,j], "top", n, Χ)

            C̃4 = reshape(C4T4, (size(C4T4, 1), :)) * projectors.Pt[2];
            C̃1 = projectors.Pu[1] * reshape(C1T2, (:, size(C1T2, 3)));
            T2A = reshape(T2A, (prod(size(T2A)[1:2]), prod(size(T2A)[3:4]), :));
            @tensor T̃2[le, re, dc] := projectors.Pu[1][le, α] * T2A[α, β, dc] * projectors.Pt[2][β, re];

            # Update environment
            environment.C[4] = symmetrize(C̃4);
            environment.C[1] = symmetrize(C̃1);
            environment.T[2] = symmetrize(T̃2);

        end


        #= Down move. Absorbs unit-cell tensors and environment tensors of the whole row, column by column =#
        for n ∈ Ni - 1:-1:0
            # Grow environment
            @tensor C2T2[ue, le, lc] := unitcell.E[i, j].C[2][le, α] * unitcell.E[i + n, j].T[2][ue, lc, α];
            @tensor T3A[le, lc, re, rc, uc] := unitcell.E[i, j].T[3][α, re, le] * unitcell.R[i + n, j][uc, rc, α, lc];
            @tensor C3T4[re, rc, ue] := unitcell.E[i,j].C[3][α, re] * unitcell.E[i + n, j].T[4][ue, rc, α];

            # Renormalize
            calc_projectors!(unitcell, projectors, [i,j], "bottom", n, Χ)

            C̃2 = reshape(C2T2, (size(C2T2, 1), :)) * projectors.Pt[2];
            C̃3 = projectors.Pd[1] * reshape(C3T4, (:, size(C3T4, 3)));
            T3A = reshape(T3A, (prod(size(T3A)[1:2]), prod(size(T3A)[3:4]), :));
            @tensor T̃3[le, re, uc] := projectors.Pd[1][le, α] * T3A[α, β, uc] * projectors.Pt[2][β, re];

            # Update environment
            environment.C[2] = symmetrize(C̃2);
            environment.C[3] = symmetrize(C̃3);
            environment.T[3] = symmetrize(T̃3);

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

function calc_projectors!(
    uc::UnitCell{T},
    projectors::Projectors{TwoCorners},
    loc::Tuple{Int64, Int64},
    direction::String,
    step::Int64,
    Χ::Int64;
    kwargs...)


    if direction == "left"

        ## Upper half density matrix. #! Updated, however still missing a good definition of location_step

        loc_step = loc
        loc_step[2] = mod(loc_step[2] + step - 1, uc.Nj) + 1;

        @tensor C4T1T4A[re, rc, lde, ldc] := uc.E[loc...].C[4][α, δ] * uc.E[loc_step...].T[1][re, α, β] * uc.E[loc_step...].T[4][δ, lde, γ] * uc.R[loc_step...][β, rc, ldc, γ]

        @tensor C1T1T2A[rde, rdc, le, lc] := uc.E[loc...].C[1][α, δ] * uc.E[loc_step...].T[2][α, rde, β] * uc.E[loc_step...].T[1][δ, le, γ] * uc.R[loc_step...][γ, β, rdc, lc]

        @tensor HU[lde, ldc, rde, rdc] := C4T1T4A[α, β, lde, ldc] * C2T1T2A[rde, rdc, α, β]

        HU = reshape(HU, size(HU, 1) * size(HU, 2), :); #! Merge into one line

        ## Lower half density matrix
        @tensor C3T4T3A[lue, luc, re, rc] := uc[loc...].E.C[3][α, δ] * uc[loc_step...].E.T[4][lue, α, β] * uc[loc_step...].E.T[3][re, δ, γ] * uc.R[loc_step][luc, rc, γ, β]
        @tensor C2T3T2A[rue, ruc, le, lc] := uc[loc...].E.C[2][α, β] * uc[loc_step...].E.T[3][β, le, γ] * uc[loc_step...].E.T[2][rue, α, δ] * uc.R[loc_step][ruc, δ, γ, lc]
        @tensor HD[lue, luc, rue, ruc] := C3T4T3A[lue, luc, α, β] * C2T3T2A[rue, ruc, α, β]

        HD = reshape(HD, size(HD, 1) * size(HD, 2), :);

        ### Left/Right move projectors
        projectors.Pl = projectors_from_identity(transpose(HU), transpose(HD), Χ; kwargs...);

    elseif direction == "right"

        loc = loc_step;
        loc_step[2] = mod(loc_step[2] - step - 1, uc.dims[2]) + 1;

        @tensor C4T1T4A[re, rc, lde, ldc] := uc.E[loc...].C[4][α, δ] * uc.E[loc_step...].T[1][re, α, β] * uc.E[loc_step...].T[4][δ, lde, γ] * uc.R[loc_step...][β, rc, ldc, γ]

        @tensor C1T1T2A[rde, rdc, le, lc] := uc.E[loc...].C[1][α, δ] * uc.E[loc_step...].T[2][α, rde, β] * uc.E[loc_step...].T[1][δ, le, γ] * uc.R[loc_step...][γ, β, rdc, lc]

        @tensor HU[lde, ldc, rde, rdc] := C4T1T4A[α, β, lde, ldc] * C2T1T2A[rde, rdc, α, β]

        HU = reshape(HU, size(HU, 1) * size(HU, 2), :); #! Merge into one line

        ## Lower half density matrix
        @tensor C3T4T3A[lue, luc, re, rc] := uc[loc...].E.C[3][α, δ] * uc[loc_step...].E.T[4][lue, α, β] * uc[loc_step...].E.T[3][re, δ, γ] * uc.R[loc_step][luc, rc, γ, β]
        @tensor C2T3T2A[rue, ruc, le, lc] := uc[loc...].E.C[2][α, β] * uc[loc_step...].E.T[3][β, le, γ] * uc[loc_step...].E.T[2][rue, α, δ] * uc.R[loc_step][ruc, δ, γ, lc]
        @tensor HD[lue, luc, rue, ruc] := C3T4T3A[lue, luc, α, β] * C2T3T2A[rue, ruc, α, β]

        HD = reshape(HD, size(HD, 1) * size(HD, 2), :);
        projectors.Pr = projectors_from_identity(HU, HD, Χ; kwargs...);


    elseif direction == "up"


        loc_step = loc
        loc_step[1] = mod(loc_step[1] + step - 1, uc.dims[1]) + 1;

        ## Left half density matrix
        @tensor C4T4T1A[ure, urc, de, dc] := uc.E[loc...].C[4][α, δ] * uc.E[loc_step...].T[4][δ, de, γ] * uc.E[loc_step...].T[1][ure, α, β] * uc.R[loc_step...][β, urc, dc, γ]
        @tensor C3T4T3A[ue, uc, dre, drc] := uc.E[loc...].C[3][α, δ] * uc.E[loc_step...].T[4][ue, α, β] * uc.E[loc_step...].T[3][dre, δ, γ] * uc.R[loc_step][uc, drc, γ, β]
        @tensor HL[ure, urc, dre, drc] := C4E4E1A[ure, urc, α, β] * C3E4E3A[α, β, dre, drc]

        HL = reshape(HL, size(HL, 1) * size(HL, 2), :); #! Merge into one line

        ## Right half density matrix
        @tensor C1T2T1A[de, dc, ule, ulc] := uc.E[loc...].C[1][α, δ] * uc.E[loc_step...].T[2][α, de, β] * uc.E[loc_step...].T[1][δ, ule, γ] * uc.R[loc_step][γ, β, dc, ulc]
        @tensor C2T3T2A[ue, uc, dle, dlc] := uc.E[loc...].C[2][α, β] * uc.E[loc_step...].T[3][β, dle, γ] * uc.E[loc_step...].T[2][ue, α, δ] * uc.R[loc_step][uc, δ, γ, dlc]
        @tensor HR[ule, ulc, dle, dlc] := C1E2E1A[α, β, ule, ulc] * C2E3E2A[α, β, dle, dlc]

        HR = reshape(HR, size(HR, 1) * size(HR, 2), :);

        ### Up/down move projectors
        projectors.Pu = projectors_from_identity(transpose(HL), transpose(HR), Χ; kwargs...);

    elseif direction == "down"

        loc_step = loc
        loc_step[1] = mod(loc_step[1] - step - 1, uc.dims[1]) + 1;

        ## Left half density matrix
        @tensor C4T4T1A[ure, urc, de, dc] := uc.E[loc...].C[4][α, δ] * uc.E[loc_step...].T[4][δ, de, γ] * uc.E[loc_step...].T[1][ure, α, β] * uc.R[loc_step...][β, urc, dc, γ]
        @tensor C3T4T3A[ue, uc, dre, drc] := uc.E[loc...].C[3][α, δ] * uc.E[loc_step...].T[4][ue, α, β] * uc.E[loc_step...].T[3][dre, δ, γ] * uc.R[loc_step][uc, drc, γ, β]
        @tensor HL[ure, urc, dre, drc] := C4E4E1A[ure, urc, α, β] * C3E4E3A[α, β, dre, drc]

        HL = reshape(HL, size(HL, 1) * size(HL, 2), :); #! Merge into one line

        ## Right half density matrix
        @tensor C1T2T1A[de, dc, ule, ulc] := uc.E[loc...].C[1][α, δ] * uc.E[loc_step...].T[2][α, de, β] * uc.E[loc_step...].T[1][δ, ule, γ] * uc.R[loc_step][γ, β, dc, ulc]
        @tensor C2T3T2A[ue, uc, dle, dlc] := uc.E[loc...].C[2][α, β] * uc.E[loc_step...].T[3][β, dle, γ] * uc.E[loc_step...].T[2][ue, α, δ] * uc.R[loc_step][uc, δ, γ, dlc]
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
    uc::UnitCell{T},
    projectors::Projectors{TwoCornersSimple},
    loc::Tuple{Int64, Int64},
    direction::String,
    step::Int64,
    Χ::Int64;
    ϵmax::Float64=0.0)


    if direction == "up"

        ## Two-upper corners

        loc_step = loc
        loc_step[2] = mod(loc_step[2] + step - 1, uc.dims[2]) + 1;

        @tensor C4T1T4A[re, rc, lde, ldc] := uc.E[loc...].C[4][α, δ] * uc.E[loc_step...].T[1][re, α, β] * uc.E[loc_step...].T[4][δ, lde, γ] * uc.R[loc_step...][β, rc, ldc, γ]

        @tensor C1T1T2A[rde, rdc, le, lc] := uc.E[loc...].C[1][α, δ] * uc.E[loc_step...].T[2][α, rde, β] * uc.E[loc_step...].T[1][δ, le, γ] * uc.R[loc_step...][γ, β, rdc, lc]

        @tensor HU[lde, ldc, rde, rdc] := C4T1T4A[α, β, lde, ldc] * C2T1T2A[rde, rdc, α, β]

        HU = reshape(HU, size(HU, 1) * size(HU, 2), :); #! Merge into one line
        U, S, V = svd(HU);

        Χmax = cutoff(S, Χ, ϵmax);
        Sinvsqrt = (S[1:Χmax ]).^(-1/2);

        C4T1T4A = transpose(reshape(C4T1T4A, (prod(size(C4T1T4A)[1:2]), :)));
        C1T1T2A = transpose(reshape(C1T1T2A, (prod(size(C1T1T2A)[1:2]), :)));
        P̃ = C1T1T2A * V[:, 1:Χcut] * diagm(Sinvsqrt);
        P = diagm(Sinvsqrt) * U'[1:Χcut , :] * C4T1T4A;

        projectors.Pu = [P̃, P];

    elseif direction == "down"


        ## Lower half density matrix
        @tensor C3T4T3A[lue, luc, re, rc] := uc[loc...].E.C[3][α, δ] * uc[loc_step...].E.T[4][lue, α, β] * uc[loc_step...].E.T[3][re, δ, γ] * uc.R[loc_step][luc, rc, γ, β]
        @tensor C2T3T2A[rue, ruc, le, lc] := uc[loc...].E.C[2][α, β] * uc[loc_step...].E.T[3][β, le, γ] * uc[loc_step...].E.T[2][rue, α, δ] * uc.R[loc_step][ruc, δ, γ, lc]
        @tensor HD[lue, luc, rue, ruc] := C3T4T3A[lue, luc, α, β] * C2T3T2A[rue, ruc, α, β]

        HD = reshape(HD, size(HD, 1) * size(HD, 2), :);
        U, S, V = svd(HD);

        Χmax = cutoff(S, Χ, ϵmax);
        Sinvsqrt = (S[1:Χmax ]).^(-1/2);

        C3T4T3A = transpose(reshape(C3T4T3A, (prod(size(C3T4T3A)[1:2]), :)));
        C2T3T2A = reshape(C2T3T2A, (prod(size(C2T3T2A)[1:2]), :));
        P̃ = C2T3T2A * V[:, 1:Χcut] * diagm(Sinvsqrt);
        P = diagm(Sinvsqrt) * U'[1:Χcut , :] * C3T4T3A;

        projectors.Pd = [P̃, P];


    elseif direction == "left"


        loc_step = loc
        loc_step[1] = mod(loc_step[1] + step - 1, uc.dims[1]) + 1;

        ## Left half density matrix
        @tensor C4T4T1A[ure, urc, de, dc] := uc.E[loc...].C[4][α, δ] * uc.E[loc_step...].T[4][δ, de, γ] * uc.E[loc_step...].T[1][ure, α, β] * uc.R[loc_step...][β, urc, dc, γ]
        @tensor C3T4T3A[ue, uc, dre, drc] := uc.E[loc...].C[3][α, δ] * uc.E[loc_step...].T[4][ue, α, β] * uc.E[loc_step...].T[3][dre, δ, γ] * uc.R[loc_step][uc, drc, γ, β]
        @tensor HL[ure, urc, dre, drc] := C4E4E1A[ure, urc, α, β] * C3E4E3A[α, β, dre, drc]

        HL = reshape(HL, size(HL, 1) * size(HL, 2), :); #! Merge into one line
        U, S, V = svd(HL);

        Χmax = cutoff(S, Χ, ϵmax);
        Sinvsqrt = (S[1:Χmax ]).^(-1/2);

        C4T4T1A = reshape(C4T4T1A, (prod(size(C4T4T1A)[1:2]), :));
        C3T4T3A = reshape(C3T4T3A, (prod(size(C3T4T3A)[1:2]), :));
        P̃ = C3T4T3A * V[:, 1:Χcut] * diagm(Sinvsqrt);
        P = diagm(Sinvsqrt) * U'[1:Χcut , :] * C4T4T1A;

        projectors.Pl = [P̃, P];

    elseif == "right"

        ## Right half density matrix
        @tensor C1T2T1A[de, dc, ule, ulc] := uc.E[loc...].C[1][α, δ] * uc.E[loc_step...].T[2][α, de, β] * uc.E[loc_step...].T[1][δ, ule, γ] * uc.R[loc_step][γ, β, dc, ulc]
        @tensor C2T3T2A[ue, uc, dle, dlc] := uc.E[loc...].C[2][α, β] * uc.E[loc_step...].T[3][β, dle, γ] * uc.E[loc_step...].T[2][ue, α, δ] * uc.R[loc_step][uc, δ, γ, dlc]
        @tensor HR[ule, ulc, dle, dlc] := C1E2E1A[α, β, ule, ulc] * C2E3E2A[α, β, dle, dlc]

        HR = reshape(HR, size(HR, 1) * size(HR, 2), :);
        U, S, V = svd(HR);

        Χmax = cutoff(S, Χ, ϵmax);
        Sinvsqrt = (S[1:Χmax ]).^(-1/2);

        C1T2T1A = transpose(reshape(C1T2T1A, (prod(size(C1T2T1A)[1:2]), :)));
        C2T3T2A = reshape(C2T3T2A, (prod(size(C2T3T2A)[1:2]), :));
        P̃ = C2T3T2A * V[:, 1:Χcut] * diagm(Sinvsqrt);
        P = diagm(Sinvsqrt) * U'[1:Χcut , :] * C1T2T1A;

        projectors.Pr = [P̃, P];

        loc_step = loc
        loc_step[1] = mod(loc_step[1] - step - 1, uc.dims[1]) + 1;

        ## Left half density matrix
        @tensor C4T4T1A[ure, urc, de, dc] := uc.E[loc...].C[4][α, δ] * uc.E[loc_step...].T[4][δ, de, γ] * uc.E[loc_step...].T[1][ure, α, β] * uc.R[loc_step...][β, urc, dc, γ]
        @tensor C3T4T3A[ue, uc, dre, drc] := uc.E[loc...].C[3][α, δ] * uc.E[loc_step...].T[4][ue, α, β] * uc.E[loc_step...].T[3][dre, δ, γ] * uc.R[loc_step][uc, drc, γ, β]
        @tensor HL[ure, urc, dre, drc] := C4E4E1A[ure, urc, α, β] * C3E4E3A[α, β, dre, drc]

        HL = reshape(HL, size(HL, 1) * size(HL, 2), :); #! Merge into one line

        ## Right half density matrix
        @tensor C1T2T1A[de, dc, ule, ulc] := uc.E[loc...].C[1][α, δ] * uc.E[loc_step...].T[2][α, de, β] * uc.E[loc_step...].T[1][δ, ule, γ] * uc.R[loc_step][γ, β, dc, ulc]
        @tensor C2T3T2A[ue, uc, dle, dlc] := uc.E[loc...].C[2][α, β] * uc.E[loc_step...].T[3][β, dle, γ] * uc.E[loc_step...].T[2][ue, α, δ] * uc.R[loc_step][uc, δ, γ, dlc]
        @tensor HR[ule, ulc, dle, dlc] := C1E2E1A[α, β, ule, ulc] * C2E3E2A[α, β, dle, dlc]

        HR = reshape(HR, size(HR, 1) * size(HR, 2), :);

        ### Up/down move projectors
        projectors.Pd = projectors_from_identity(HL, HR, Χ; kwargs...);


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

function projectors_from_identity(
    densitymatrix_U_or_L::Array{T,2},
    densitymatrix_D_or_R::Array{T,2},
    Χmax::Int64;
    ϵmax::Float64 = 0.0)

    _, R = qr(densitymatrix_U_or_L);
    _, R̃ = qr(densitymatrix_D_or_R);

    U, S, V = svd(R * R̃);

    Χcut  = cutoff(S, Χmax, ϵmax);
    Sinvsqrt = (S[1:Χcut ]).^(-1/2);

    P̃ = R̃ * V[:, 1:Χcut ] * diagm(Sinvsqrt);
    P = diagm(Sinvsqrt) * U'[1:Χcut , :] * R;

    return [P, P̃]
end
