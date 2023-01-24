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
    i = 0;
    expval = 1.0;

    do_ctmrg_iteration!(unitcell, projectors, Χ = simulation.Χ);
    while ϵ > simulation.tol_ctm
        i += 1;

        Eref = deepcopy(unitcell.E)
        do_ctmrg_iteration!(unitcell, projectors, Χ = simulation.Χ);
        ϵ = calculate_error_ctm(Eref, unitcell.E);

        log_message("\nCTM iteration $i, convergence error = $(round(abs(ϵ), sigdigits = 4))\n", color = :blue);

        if simulation.ctm_convergence == Observable
            ref_expval = copy(expval);
            expval, n = calculate_error_ctm(unitcell, simulation)

            if abs((ref_expval - expval)/expval) < simulation.tol_expval
                log_message("\n!!! CTM converged !!!\n", color = :green)
                simulation.conv_ctm_steps = i;
                simulation.ctm_error = ϵ;
                return true
            end
        end

        if i > simulation.max_ctm_steps
            log_message("\n!!! Maximum number of iterations reached !!!\n", color = :red)
            simulation.conv_ctm_steps = i;
            simulation.ctm_error = ϵ;
            return false
        end

    end

    simulation.conv_ctm_steps = i;
    simulation.ctm_error = ϵ;
    return true
end


"""
    do_ctmrg_iteration!(
    unitcell::UnitCell{T},
    projectors::Projectors{EachMove}; Χ::Int64=0) where {T<:Union{Float64, ComplexF64}}


    Convention axis and indices in unit-cell

    --→ x(j)
    |
    ↓
    y(i)

    Convention indices in arrays: (i,j) i.e (y,x)

"""
function do_ctmrg_iteration!(
    unitcell::UnitCell{T},
    projectors::Projectors{EachMove}; Χ::Int64=0) where {T<:Union{Float64, ComplexF64}}

    Ni = unitcell.dims[1]; # y-axis
    Nj = unitcell.dims[2]; # x-axis

    # Makes a copy of the initial full cell environment to be used when updating each tensor environment
    initial_environment = deepcopy(unitcell.E);
    new_environment = Array{Environment{T}, 2}(undef, unitcell.dims);

    ##### Left move #####
    for ij ∈ CartesianIndices(unitcell.dims)
        unitcell.E = deepcopy(initial_environment); # Reset environment to initial one

        #= Left move for every unit-cell tensor. Sweeps from left to right (x-axis), column by column =#
        for j ∈ 0:Nj-1

            # 1) Calculate all projectors along the column, i.e. all P_(i, j+n) for fixed j+n
            for i ∈ 0:Ni-1
                loc = coord(ij + (i, j), unitcell.dims);
                calculate_projectors_ctmrg!(unitcell, projectors, loc, LEFT, Χ=Χ); # P_(i,j)
            end

            # 2) Absorb column j+n tensors in environment tensors of all tensors (i,j) with fixed j and renormalize
            for i ∈ 0:Ni-1
                loc = coord(ij + (i, j), unitcell.dims);
                do_ctm_move!(unitcell, projectors, LEFT, loc);
            end
        end

        # Updates new environment
        new_environment[ij] = deepcopy(unitcell.E[ij]);
    end

    initial_environment = deepcopy(new_environment);

    ##### Right move #####
    for ij ∈ CartesianIndices(unitcell.dims)
        unitcell.E = deepcopy(initial_environment); # Reset environment to initial one

        #= Right move for every unit-cell tensor. Sweeps from right to left, column by column =#
        for j ∈ 0:Nj-1

            # 1) Calculate all projectors for the column, i.e. all P_(i, j+n) for fixed j+n
            for i ∈ 0:Ni-1
                loc = coord(ij + (i, -j), unitcell.dims);
                calculate_projectors_ctmrg!(unitcell, projectors, loc, RIGHT, Χ=Χ); # P_(i,j)
            end

            # 2) Absorb column j-n tensors in environment tensors of all tensors (i,j) with fixed j and renormalize
            for i ∈ 0:Ni-1
                loc = coord(ij + (i, -j), unitcell.dims);
                do_ctm_move!(unitcell, projectors, RIGHT, loc);
            end
        end

        # Updates new environment
        new_environment[ij] = deepcopy(unitcell.E[ij]);
    end

    initial_environment = deepcopy(new_environment);

    ##### Up move #####
    for ij ∈ CartesianIndices(unitcell.dims)
        unitcell.E = deepcopy(initial_environment); # Reset environment to initial one

        #= Up move for every unit-cell tensor. Sweeps from top to bottom (y-axis), row by row =#
        for i ∈ 0:Ni-1

            # 1) Calculate all projectors for the row, i.e. all P_(i+n, j) for fixed i+n
            for j ∈ 0:Nj-1
                loc = coord(ij + (i, j), unitcell.dims);
                calculate_projectors_ctmrg!(unitcell, projectors, loc, UP, Χ=Χ); # P_(i,j)
            end

            # 2) Absorb row i+n tensors in environment tensors of all tensors (i,j) with fixed i and renormalize
            for j ∈ 0:Nj-1
                loc = coord(ij + (i, j), unitcell.dims);
                do_ctm_move!(unitcell, projectors, UP, loc);
            end
        end

        # Updates new environment
        new_environment[ij] = deepcopy(unitcell.E[ij]);
    end

    initial_environment = deepcopy(new_environment);

    ##### Down move #####
    for ij ∈ CartesianIndices(unitcell.dims)
        unitcell.E = deepcopy(initial_environment); # Reset environment to initial one

        #= Down move for every unit-cell tensor. Sweeps from bottom to top, row by row =#
        for i ∈ 0:Ni-1

            # 1) Calculate all projectors for the row, i.e. all P_(i+n, j) for fixed i+n
            for j ∈ 0:Nj-1
                loc = coord(ij + (-i, j), unitcell.dims);
                calculate_projectors_ctmrg!(unitcell, projectors, loc, DOWN, Χ=Χ); # P_(i,j)
            end

            # 2) Absorb row i-n tensors in environment tensors of all tensors (i,j) with fixed i and renormalize
            for j ∈ 0:Nj-1
                loc = coord(ij + (-i, j), unitcell.dims);
                do_ctm_move!(unitcell, projectors, DOWN, loc);
            end
        end

        # Updates new environment
        new_environment[ij] = deepcopy(unitcell.E[ij]);
    end

    unitcell.E = deepcopy(new_environment);
end


function renormalize_tensor(T::Array{X, 4}, A::Array{X, 5}, B::Array{X,5}, P::Vector{Array{ComplexF64, 4}}, dir::Direction) where {X<:Union{ComplexF64, Float64}}#P[1] -> P̃, P[2] -> P
    if dir == UP
        @tensoropt T̃[re, le, dk, db] := T[α, β, γ, δ] * A[γ, ξ, dk, θ, p] * conj(B)[δ, κ, db, λ, p] * P[1][α, ξ, κ, re] * P[2][β, θ, λ, le];
    elseif dir == RIGHT
        @tensoropt T̃[ue, de, lk, lb] := T[α, β, γ, δ] * A[θ, γ, ξ, lk, p] * conj(B)[λ, δ, κ, lb, p] * P[1][β, ξ, κ, de] * P[2][α, θ, λ, ue];
    elseif dir == DOWN
        @tensoropt T̃[re, le, uk, ub] := T[α, β, γ, δ] * A[uk, ξ, γ, θ, p] * conj(B)[ub, κ, δ, λ, p] * P[1][α, ξ, κ, re] * P[2][β, θ, λ, le];
    elseif dir == LEFT
        @tensoropt T̃[ue, de, rk, rb] := T[α, β, γ, δ] * A[θ, rk, ξ, γ, p] * conj(B)[λ, rb, κ, δ, p] * P[1][β, ξ, κ, de] * P[2][α, θ, λ, ue];
    end

    return T̃
end

"""
    renormalize_tensor(C::Array{ComplexF64, 2}, T::Array{ComplexF64, 4}, P::Array{ComplexF64, 4}, dir::Direction)


    UP:                         RIGHT:
        ___ β    β ___           T₁ ____   α___
    C₄ᵀ |            | C₁            ||       |  C₁ᵀ
        α            α                        β

        |__        __|
    T₄  |__        __| T₂                     β
        |            |           T₃ _||_  α___|  C₂ᵀ


    DOWN:                       LEFT:
        |__        __|              ___ α  ____ T₁
    T₄  |__        __| T₂       C₄  |       ||
        |            |              β

        α            α              β
    C₃  |__β      β__| C₂       C₃ᵀ |__ α  _||_ T₃


"""
function renormalize_tensor(C::AbstractArray{X, 2}, T::Array{X, 4}, P::Array{ComplexF64, 4}, dir::Direction) where {X<:Union{ComplexF64, Float64}}
    if dir == UP
        @tensoropt (α=>χ, β=>χ, de=>χ, T1=>χ) C̃[de, T1] := C[α, β] * T[α, de, γ, δ] * P[β, γ, δ, T1]
    elseif dir == RIGHT
        @tensoropt (α=>χ, β=>χ, le=>χ, T2=>χ) C̃[le, T2] := C[α, β] * T[α, le, γ, δ] * P[β, γ, δ, T2]
    elseif dir == DOWN
        @tensoropt (α=>χ, β=>χ, ue=>χ, T3=>χ) C̃[ue, T3] := C[α, β] * T[ue, α, γ, δ] * P[β, γ, δ, T3]
    elseif dir == LEFT
        @tensoropt (α=>χ, β=>χ, re=>χ, T4=>χ) C̃[re, T4] := C[α, β] * T[re, α, γ, δ] * P[β, γ, δ, T4]
    end
    return C̃
end

function do_ctm_move!(unitcell::UnitCell, projectors::Projectors, direction::Direction, loc::CartesianIndex)
    E_loc = unitcell(Environment, loc);
    A = unitcell(Tensor, loc).A;
    isdefined(unitcell, :B) == true ? (B = unitcell(BraTensor, loc).A) : (B = A);

    if direction == UP

        """
         C4(x,y-1)--      --T1(x,y-1)--     --C1(x,y-1)
            |                   |                 |
            |                   |                 |
        T4(x,y)--          --R(x,y)--        --T2(x,y)
            |                   |                 |

        """

        E_add = unitcell(Environment, loc + (-1, 0));

        # Renormalize
        P̃ = projectors(UP, loc)[1];
        P = projectors(UP, loc + (0, -1))[2];

        C̃4 = transpose(renormalize_tensor(transpose(E_add.C[4]), E_loc.T[4], P̃, UP));
        C̃1 = renormalize_tensor(E_add.C[1], E_loc.T[2], P, UP);
        T̃1 = renormalize_tensor(E_add.T[1], A, B, [P̃, P], UP);


        # Update tensors environment
        update_tensors!(unitcell, [C̃4, T̃1, C̃1], UP, loc);


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

        # Renormalize
        P̃ = projectors(RIGHT, loc)[1];
        P = projectors(RIGHT, loc + (-1, 0))[2];

        C̃1 = transpose(renormalize_tensor(transpose(E_add.C[1]), E_loc.T[1], P̃, RIGHT));
        C̃2 = transpose(renormalize_tensor(transpose(E_add.C[2]), E_loc.T[3], P, RIGHT));
        T̃2 = renormalize_tensor(E_add.T[2], A, B, [P̃, P], RIGHT);

        # Update tensors environment
        update_tensors!(unitcell, [C̃1, T̃2, C̃2], RIGHT, loc);


    elseif direction == DOWN

        """
             |                 |                  |
          T4(x,y)--       --R(x,y)--        --T2(x,y)
             |                 |                  |
             |                 |                  |
          C3(x,y+1)--    --T3(x,y+1)--      --C2(x,y+1)

        """

        E_add = unitcell(Environment, loc + (1, 0));

        # Renormalize
        P̃ = projectors(DOWN, loc)[1];
        P = projectors(DOWN, loc + (0, -1))[2];

        C̃3 = renormalize_tensor(E_add.C[3], E_loc.T[4], P̃, DOWN);
        C̃2 = renormalize_tensor(E_add.C[2], E_loc.T[2], P, DOWN);
        T̃3 = renormalize_tensor(E_add.T[3], A, B, [P̃, P], DOWN);

        # Update tensors environment
        update_tensors!(unitcell, [C̃2, T̃3, C̃3], DOWN, loc);

    elseif direction == LEFT

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

        # Renormalize
        P̃ = projectors(LEFT, loc)[1];
        P = projectors(LEFT, loc + (-1, 0))[2];

        C̃4 = renormalize_tensor(E_add.C[4], E_loc.T[1], P̃, LEFT);
        C̃3 = transpose(renormalize_tensor(transpose(E_add.C[3]), E_loc.T[3], P, LEFT));
        T̃4 = renormalize_tensor(E_add.T[4], A, B, [P̃, P], LEFT);

        # Update tensors environment
        update_tensors!(unitcell, [C̃3, T̃4, C̃4], LEFT, loc);
    end
end


#= """
    projectors_half_system(Qa::Array{T, 6}, Qb::Array{T, 6}, Χ::Int64)

    Index convention projectors
                        /|-- environment (1)
                     __/ |
      environment(4)   \ |-- ket (2)
                        \|-- bra (3)

""" =#
function projectors_half_system(Qa::Array{T, 6}, Qb::Array{T, 6}, Χ::Int64) where {T<:Union{Float64, ComplexF64}}
    @tensor rho[lde, ldk, ldb, rde, rdk, rdb] := Qa[lde, ldk, ldb, α, β, γ] * Qb[α, β, γ, rde, rdk, rdb];

    # Calculate projectors
    U, S, Vt = tensor_svd(rho, [[1,2,3], [4,5,6]], Χ = Χ, full_svd = false);
    Sinvsqrt = diagm(S.^(-1/2));

    @tensor P̃[le, lk, lb, re] := Qb[le, lk, lb, α, β, γ] * Vt[δ, α, β, γ] * Sinvsqrt[δ, re];
    @tensor P[re, rk, rb, le] := Sinvsqrt[le, δ] * conj(U)[α, β, γ, δ] * Qa[α, β, γ, re, rk, rb];

    # Spectra of half-system
    S_f = zeros(Χ);
    S_f[1:length(S)] = S/maximum(S);

    return P̃, P, S_f
end



function calculate_projectors_ctmrg!(
    uc::UnitCell,
    projectors::Projectors,
    loc::CartesianIndex,
    direction::Direction;
    Χ::Int64=0)

    if direction == UP

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
        isdefined(uc, :B) == true ? (B = uc(BraTensor, loc).A) : (B = A);


        # Build enlarged corner
        @tensor Q4[de, dk, db, re, rk, rb] := C4[α, β] * T1[re, α, γ, ξ] * T4[β, de, δ, θ] * A[γ, rk, dk, δ, p] * conj(B)[ξ, rb, db, θ, p];


        ##### Q1 #####
        # Load tensors
        C1 = uc(Environment, loc + (-1, 2)).C[1];
        T1 = uc(Environment, loc + (-1, 1)).T[1];
        T2 = uc(Environment, loc + (0, 2)).T[2];
        A = uc(Tensor, loc + (0, 1)).A;
        isdefined(uc, :B) == true ? (B = uc(BraTensor, loc + (0, 1)).A) : (B = A);

        # Build enlarged corner
        @tensor Q1[le, lk, lb, de, dk, db] := C1[α, β] * T1[β, le, δ, θ] * T2[α, de, γ, ξ]  * A[δ, γ, dk, lk, p] * conj(B)[θ, ξ, db, lb, p];


        #return Q4, Q1

        # Calculate projectors
        Χ == 0 && (Χ = size(uc(Environment, loc).T[1], 1);)
        P̃, P, S_f = projectors_half_system(Q4, Q1, Χ);
        projectors.Pu[loc] = [P̃, P];

        # Save spectra
        uc.E[loc].spectra[1] = S_f;


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
        isdefined(uc, :B) == true ? (B = uc(BraTensor, loc).A) : (B = A);

        # Build enlarged corner
        @tensor Q1[le, lk, lb, de, dk, db] := C1[α, β] * T1[β, le, δ, θ] * T2[α, de, γ, ξ]  * A[δ, γ, dk, lk, p] * conj(B)[θ, ξ, db, lb, p];


        ##### Q2 #####
        # Load tensors
        C2 = uc(Environment, loc + (2, 1)).C[2];
        T3 = uc(Environment, loc + (2, 0)).T[3];
        T2 = uc(Environment, loc + (1, 1)).T[2];
        A = uc(Tensor, loc + (1, 0)).A;
        isdefined(uc, :B) == true ? (B = uc(BraTensor, loc + (1, 0)).A) : (B = A);

        # Build enlarged corner
        @tensor Q2[ue, uk, ub, le, lk, lb] := C2[α, β] * T2[ue, α, γ, ξ] * T3[β, le, δ, θ] * A[uk, γ, δ, lk, p] * conj(B)[ub, ξ, θ, lb, p];


        #return Q1, Q2

        ##### Calculate projectors #####
        Χ == 0 && (Χ = size(uc(Environment, loc).T[2], 2);)
        P̃, P, S_f = projectors_half_system(Q1, Q2, Χ);
        projectors.Pr[loc] = [P̃, P];

        # Save spectra
        uc.E[loc].spectra[2] = S_f;

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
        isdefined(uc, :B) == true ? (B = uc(BraTensor, loc).A) : (B = A);

        # Build enlarged corner
        @tensor Q3[ue, uk, ub, re, rk, rb] := C3[α, β] * T4[ue, α, γ, ξ] * T3[re, β, δ, θ] * A[uk, rk, δ, γ, p] * conj(B)[ub, rb, θ, ξ, p];


        ##### Q2 #####
        # Load tensors
        C2 = uc(Environment, loc + (1, 2)).C[2];
        T3 = uc(Environment, loc + (1, 1)).T[3];
        T2 = uc(Environment, loc + (0, 2)).T[2];
        A = uc(Tensor, loc + (0, 1)).A;
        isdefined(uc, :B) == true ? (B = uc(BraTensor, loc + (0, 1)).A) : (B = A);

        # Build enlarged corner
        @tensor Q2[le, lk, lb, ue, uk, ub] := C2[α, β] * T2[ue, α, γ, ξ] * T3[β, le, δ, θ] * A[uk, γ, δ, lk, p] * conj(B)[ub, ξ, θ, lb, p];

        ##### Calculate projectors #####
        Χ == 0 && (Χ = size(uc(Environment, loc).T[3], 1);)
        P̃, P, S_f = projectors_half_system(Q3, Q2, Χ);
        projectors.Pd[loc] = [P̃, P];

        # Save spectra
        uc.E[loc].spectra[3] = S_f;

    elseif direction == LEFT

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
        C4 = uc(Environment, loc + (-1, -1)).C[4];
        T1 = uc(Environment, loc  + (-1, 0)).T[1];
        T4 = uc(Environment, loc + (0, -1)).T[4];
        A = uc(Tensor, loc).A;
        isdefined(uc, :B) == true ? (B = uc(BraTensor, loc).A) : (B = A);

        # Build enlarged corner
        @tensor Q4[re, rk, rb, de, dk, db] := C4[α, β] * T1[re, α, γ, ξ] * T4[β, de, δ, θ] * A[γ, rk, dk, δ, p] * conj(B)[ξ, rb, db, θ, p];


        ##### Q3 #####
        # Load tensors
        C3 = uc(Environment, loc + (2, -1)).C[3];
        T3 = uc(Environment, loc + (2, 0)).T[3];
        T4 = uc(Environment, loc + (1, -1)).T[4];
        A = uc(Tensor, loc + (1, 0)).A;
        isdefined(uc, :B) == true ? (B = uc(BraTensor, loc + (1, 0)).A) : (B = A);

        # Build enlarged corner
        @tensor Q3[ue, uk, ub, re, rk, rb] := C3[α, β] * T4[ue, α, γ, ξ] * T3[re, β, δ, θ] * A[uk, rk, δ, γ, p] * conj(B)[ub, rb, θ, ξ, p];


        ##### Calculate projectors #####
        Χ == 0 && (Χ = size(uc(Environment, loc).T[4], 2);)
        P̃, P, S_f = projectors_half_system(Q4, Q3, Χ);
        projectors.Pl[loc] = [P̃, P];

        # Save spectra
        uc.E[loc].spectra[4] = S_f;

        return Q4, Q3
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
        ϵ += sum([sum(abs.(Eupd[xy].spectra[n] - Eref[xy].spectra[n])) for n ∈ 1:4])/4
    end

    return ϵ/prod(size(Eref))
end

function calculate_error_ctm(unitcell::UnitCell, simulation::Simulation)

    O = simulation.observables[1].O;
    loc = simulation.observables[1].loc[1];
    name_O = simulation.observables[1].name;

    rho = calculate_rdm(unitcell, loc);
    n = tr(rho);
    @tensor expval = (1/n) * rho[α, β] * O[α, β]

    log_message("$name_O at $(Tuple(loc)) = $(round(expval, sigdigits=4)), norm = $(round(n, sigdigits = 4)) \n", color = :blue);

    if length(simulation.observables) > 1
        @assert "Not implemented yet"
    end

    return expval, n
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



#=
function do_ctm_move_old!(unitcell::UnitCell, projectors::Projectors, direction::Direction, loc::CartesianIndex)
    E_loc = unitcell(Environment, loc);
    A = unitcell(Tensor, loc).A;
    isdefined(unitcell, :B) == true ? (B = unitcell(BraTensor, loc).A) : (B = conj(A));


    """
    Convention axis and indices in unit-cell
    ____ x(j)
    |
    |
    y(i)

    Convention indices in arrays: (i,j) i.e (y,x)

    """

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
        @tensor T̃1[re, le, dk, db] := P[le, β] * T1AB[α, β, dk, db] * P̃[α, re];

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
        @tensor T̃3[re, le, uk, ub] := P[le, β] * T3AB[α, β, uk, ub] * P̃[α, re];

        # Update tensors environment
        update_tensors!(unitcell, [C̃2, T̃3, C̃3], DOWN, loc);
    end
end =#


#= function calculate_projectors_ctmrg_old!(
    uc::UnitCell,
    projectors::Projectors,
    loc::CartesianIndex,
    direction::Direction;
    Χ::Int64=0)

    if direction == UP

        # Calculate half-system density matrix
        Q4, Q1 = calculate_enlarged_corners(uc, loc, UP)
        @tensor HU[lde, ldk, ldb, rde, rdk, rdb] := Q4[α, β, γ, lde, ldk, ldb] * Q1[α, β, γ, rde, rdk, rdb];

        # Calculate projectors
        if Χ == 0
            Χ = size(uc(Environment, loc).T[1], 1);
        end
        U, S, Vt = tensor_svd(HU, [[1,2,3], [4,5,6]], Χ = Χ);
        Sinvsqrt = diagm(S.^(-1/2));

        @tensor P̃[re, le, lk, lb] := Q1[le, lk, lb, α, β, γ] * Vt[δ, α, β, γ] * Sinvsqrt[δ, re];
        @tensor P[re, rk, rb, le] := Sinvsqrt[le, δ] * conj(U)[α, β, γ, δ] * Q4[α, β, γ, re, rk, rb];

        projectors.Pu[loc] = [P̃, P];

        # Save spectra of half-system
        S_f = zeros(Χ);
        S_f[1:length(S)] = S/maximum(S);
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


end =#
