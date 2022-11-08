
################################################
# Directional CTM (10.1103/PhysRevB.80.094403) #
################################################

"""
    do_move!(
    unitcell::UnitCell{T},
    environment::Environment{T},
    projectors::Projectors{X}) where {X<:Renormalization}

TBW
"""
function do_ctm_iteration!(
    unitcell::UnitCell,
    environment::Environment,
    projectors::Projectors{X}) where {X<:Renormalization}

    # u,d,l,r: direction of bond. e,c: environment or cell bond. E.g: bond re corresponds to a right-environment bond
    # greek letters for contracted symbols

    Χ = environment.Χ;

    #= Projectors for up/down moves =#
    calc_projectors!(unitcell, environment, projectors, "up_down", Χ);

    #= Up move =#
    begin
        # Grow environment
        @tensor C4E4[re, rc, de] := environment.C[4][re, α] * environment.E[4][α, de, rc]
        @tensor E1A[re, rc, le, lc, dc] := environment.E[1][re, le, α] * unitcell.R[1][α, rc, dc, lc]
        @tensor C1E2[de, le, lc] := environment.C[1][α, le] * environment.E[2][α, de, lc]

        # Renormalize
        C̃4 = transpose(projectors.Pu[2]) * reshape(C4E4, (:, size(C4E4, 3)));
        C̃1 = reshape(C1E2, (size(C1E2, 1), :)) * transpose(projectors.Pu[1]);
        E1A = reshape(E1A, (prod(size(E1A)[1:2]), prod(size(E1A)[3:4]), :));
        @tensor Ẽ1[re, le, dc] := projectors.Pu[1][le, α] * E1A[β, α, dc] * projectors.Pu[2][β, re];

        # Update environment
        environment.C[4] = symmetrize(C̃4);
        environment.C[1] = symmetrize(C̃1);
        environment.E[1] = symmetrize(Ẽ1);
    end

    #= Down move =#
    begin
        # Grow environment
        @tensor C3E4[ue, re, rc] := environment.C[3][α, re] * environment.E[4][ue, α, rc]
        @tensor E3A[re, rc, le, lc, uc] := environment.E[3][re, le, α] * unitcell.R[1][uc, rc, α, lc]
        @tensor C2E2[ue, le, lc] := environment.C[2][α, le] * environment.E[2][ue, α, lc]

        # Renormalize
        C̃3 = reshape(C3E4, (size(C3E4, 1), :)) * projectors.Pd[2];
        C̃2 = reshape(C2E2, (size(C2E2, 1), :)) * transpose(projectors.Pd[1])
        E3A = reshape(E3A, (prod(size(E3A)[1:2]), prod(size(E3A)[3:4]), :));
        @tensor Ẽ3[re, le, uc] := projectors.Pd[1][le, α] * E3A[β, α, uc] * projectors.Pd[2][β, re];

        # Update environment
        environment.C[3] = symmetrize(C̃3);
        environment.C[2] = symmetrize(C̃2);
        environment.E[3] = symmetrize(Ẽ3);
    end

    #= Projectors for left/right moves =#
    calc_projectors!(unitcell, environment, projectors, "left_right", Χ);

    #= Left move =#
    begin
        # Grow environment
        @tensor C4E1[re, de, dc] := environment.C[4][α, de] * environment.E[1][re, α, dc]
        @tensor E4A[ue, uc, de, dc, rc] := environment.E[4][ue, de, α] * unitcell.R[1][uc, rc, dc, α]
        @tensor C3E3[ue, uc, re] := environment.C[3][ue, α] * environment.E[3][α, re, uc]

        # Renormalize
        C̃4 = reshape(C4E1, (size(C4E1, 1), :)) * projectors.Pl[2];
        C̃3 = projectors.Pl[1] * reshape(C3E4, (:, size(C3E3, 3)));
        E4A = reshape(E4A, (prod(size(E4A)[1:2]), prod(size(E4A)[3:4], :)));
        @tensor Ẽ4[ue, de, rc] := projectors.Pl[1][ue, α] * E4A[α, β, rc] * projectors.Pl[2][β, de];

        # Update environment
        environment.C[4] = symmetrize(C̃4);
        environment.C[3] = symmetrize(C̃3);
        environment.E[4] = symmetrize(Ẽ4);
    end

    ## Right move
    begin
        # Grow environment
        @tensor C1E1[de, dc, le] := environment.C[1][de, α] * environment.E[1][α, le, dc]
        @tensor E2A[ue, uc, de, dc, lc] := environment.E[2][ue, de, α] * unitcell.R[1][uc, α, dc, lc]
        @tensor C2E3[ue, uc, le] := environment.C[2][ue, α] * environment.E[3][α, le, uc]

        # Renormalize
        C̃1 = transpose(projectors.Pr[2]) * reshape(C1E1, (:, size(C1E1, 3)));
        C̃2 = projectors.Pr[1] * reshape(C2E3, (:, size(C2E3, 3)));
        E2A = reshape(E2A, (prod(size(E2A)[1:2]), prod(size(E2A)[3:4], :)));
        @tensor Ẽ2[ue, de, lc] := projectors.Pr[1][ue, α] * E2A[α, β, lc] * projectors.Pr[2][β, de];

        # Update environment
        environment.C[1] = symmetrize(C̃1);
        environment.C[2] = symmetrize(C̃2);
        environment.E[2] = symmetrize(Ẽ2);
    end


    #return ?
end


function calc_projectors!(
    unitcell::UnitCell{T},
    environment::Environment{T},
    projectors::Projectors{HalfSystem}, direction::String,
    Χ::Int64;
    kwargs...)

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

    if direction == "left_right"

        ## Upper half density matrix
        @tensor C4E1E4A[re, rc, lde, ldc] := environment.C[4][α, δ] * environment.E[1][re, α, β] * environment.E[4][δ, lde, γ] * unitcell.R[1][β, rc, ldc, γ]
        @tensor C1E1E2A[rde, rdc, le, lc] := environment.C[1][α, δ] * environment.E[2][α, rde, β] * environment.E[1][δ, le, γ] * unitcell.R[1][γ, β, rdc, lc]
        @tensor HU[lde, ldc, rde, rdc] := C4E1E4A[α, β, lde, ldc] * C2E1E2A[rde, rdc, α, β]

        HU = reshape(HU, size(HU, 1) * size(HU, 2), :); #! Merge into one line

        ## Lower half density matrix
        @tensor C3E4E3A[lue, luc, re, rc] := environment.C[3][α, δ] * environment.E[4][lue, α, β] * environment.E[3][re, δ, γ] * unitcell.R[1][luc, rc, γ, β]
        @tensor C2E3E2A[rue, ruc, le, lc] := environment.C[2][α, β] * environment.E[3][β, le, γ] * environment.E[2][rue, α, δ] * unitcell.R[1][ruc, δ, γ, lc]
        @tensor HD[lue, luc, rue, ruc] := C3E4E3A[lue, luc, α, β] * C2E3E2A[rue, ruc, α, β]

        HD = reshape(HD, size(HD, 1) * size(HD, 2), :);

        ### Left/Right move projectors
        projectors.Pl = projectors_from_identity(transpose(HU), transpose(HD), Χ; kwargs...);
        projectors.Pr = projectors_from_identity(HU, HD, Χ; kwargs...);

    elseif direction == "up_down"

        ## Left half density matrix
        @tensor C4E4E1A[ure, urc, de, dc] := environment.C[4][α, δ] * environment.E[4][δ, de, γ] * environment.E[1][ure, α, β] * unitcell.R[1][β, urc, dc, γ]
        @tensor C3E4E3A[ue, uc, dre, drc] := environment.C[3][α, δ] * environment.E[4][ue, α, β] * environment.E[3][dre, δ, γ] * unitcell.R[1][uc, drc, γ, β]
        @tensor HL[ure, urc, dre, drc] := C4E4E1A[ure, urc, α, β] * C3E4E3A[α, β, dre, drc]

        HL = reshape(HL, size(HL, 1) * size(HL, 2), :); #! Merge into one line

        ## Right half density matrix
        @tensor C1E2E1A[de, dc, ule, ulc] := environment.C[1][α, δ] * environment.E[2][α, de, β] * environment.E[1][δ, ule, γ] * unitcell.R[1][γ, β, dc, ulc]
        @tensor C2E3E2A[ue, uc, dle, dlc] := environment.C[2][α, β] * environment.E[3][β, dle, γ] * environment.E[2][ue, α, δ] * unitcell.R[1][uc, δ, γ, dlc]
        @tensor HR[ule, ulc, dle, dlc] := C1E2E1A[α, β, ule, ulc] * C2E3E2A[α, β, dle, dlc]

        HR = reshape(HR, size(HR, 1) * size(HR, 2), :);

        ### Up/down move projectors
        projectors.Pu = projectors_from_identity(transpose(HL), transpose(HR), Χ; kwargs...);
        projectors.Pd = projectors_from_identity(HL, HR, Χ; kwargs...);

    end
end

function update_environment!(
    environment::Environment,
    unitcell::UnitCell,
    projectors::Projectors,
    criteria::ConvergenceCriteria,
    tol::Float64;
    max_iters::Int64 = 50)

    C_tr = [tr(environment.C[n]) for n ∈ 1:4];
    T_tr = [tr(environment.T[n]) for n ∈ 1:4];

    ϵ = 0.0;
    i = 0
    while ϵ > tol
        i += 1;
        do_ctm_iteration!(unitcell, environment, projectors);

        # Checks difference in eigenvalues of environment tensors
        if criteria == OnlyCorners
            ϵ =  sum([C_tr[n] - tr(environment.C[n]) for n ∈ 1:4]);
        elseif criteria == Full
            ϵ = sum([C_tr[n] - tr(environment.C[n])  + T_tr[n] -
            tr(environment.T[n]) for n ∈ 1:4]);
        end

        log_message("\n-> CTM iteration $i using $(typeof(projectors)).
        Convergence error = $(round(abs(ϵ), sigdigits = 4))\n", color = :blue);

        if i == max_iters
            log_message("\n!!! Maximum number of iterations reached !!!", color = :yellow);
            return ϵ
        end
    end

    return ϵ
end
