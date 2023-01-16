##############################################################################
# Simple update (10.1103/PhysRevLett.101.090603; 10.1103/PhysRevB.86.195137) #
##############################################################################
calculate_gate(hamiltonian::Hamiltonian, β::Float64) = reshape(exp(-β * hamiltonian.hij), (2, 2, 2, 2));


function do_su_iteration!(unitcell::UnitCell, hamiltonian::Hamiltonian, τ::Float64)


    G = calculate_gate(hamiltonian, τ);
    Dmax = unitcell.S[1].D[1];

    if unitcell.dims == (1,1) #! Update this case using large unit cell methods

        #? If system if rot. invariant, the direction is irrelevant.
        #? I could as well apply four times the gate to the same bond

        for gate_direction ∈ [UP, RIGHT, DOWN, LEFT]

            @assert false "Outdated" #!
            # Absorb weights and reshape tensor
            QA, RA = prepare_su_tensor(unitcell.S[i,j], gate_direction);


            # Apply gate
            weights = diagm(unitcell.S[1].weights[Int(gate_direction)]);
            R̃A, _, new_weights = apply_gate(G, RA, RB, weights, Dmax);

            # New SU tensors
            S̃A = restore_su_tensor(QA, R̃A, gate_direction);

            # Update cell
            unitcell.S[1].S = S̃A;
            unitcell.S[1].weights = new_weights;
        end

    else
        for gate_direction ∈ [DOWN, RIGHT]

            gate_direction == DOWN && (gate_direction_neighbor = UP);
            gate_direction == RIGHT && (gate_direction_neighbor = LEFT);

            unique_tensors = unique(unitcell.pattern);

            for type_tensor in unique_tensors #* Loops through unique tensors in unit-cell
                coord = findfirst(t -> t == type_tensor, unitcell.pattern);

                #= Old =#
                #for i ∈ 1:unitcell.dims[1], j ∈ 1:unitcell.dims[2]
                #gate_direction == DOWN && (neighbor_coord = [mod(i, unitcell.dims[1]) + 1, j];)
                #gate_direction == RIGHT && (neighbor_coord = [i, mod(j, unitcell.dims[2]) + 1];)
                #= Old =#

                gate_direction == DOWN && (neighbor_coord = CartesianIndex(mod(coord[1], unitcell.dims[1]) + 1, coord[2]);)
                gate_direction == RIGHT && (neighbor_coord = CartesianIndex(coord[1], mod(coord[2], unitcell.dims[2]) + 1);)

                # Absorb weights and reshape tensor
                QA, RA = prepare_su_tensor(unitcell.S[coord], gate_direction);
                QB, RB = prepare_su_tensor(unitcell.S[neighbor_coord], gate_direction_neighbor)

                # Apply gate
                weights = diagm(unitcell.S[coord].weights[Int(gate_direction)]);
                R̃A, R̃B, new_weights = apply_gate(G, RA, RB, weights, Dmax);

                # Contract new SU tensors
                S̃A = restore_su_tensor(QA, R̃A, unitcell.S[coord].weights, gate_direction);
                S̃B = restore_su_tensor(QB, permutedims(R̃B, (2,1,3)), unitcell.S[coord].weights, gate_direction_neighbor);

                #!
                #println(type_tensor)
                #show(sum(unitcell.S[coord].weights[Int(gate_direction)].^2))

                # Update unit-cell
                update_cell!(unitcell, S̃A, new_weights, unitcell.pattern[coord], gate_direction);
                update_cell!(unitcell, S̃B, new_weights, unitcell.pattern[neighbor_coord], gate_direction_neighbor);

            end

        end

    end
end

function time_evolution_su(unitcell::UnitCell, hamiltonian::Hamiltonian, su_sweep::Simulation)

    initialize_environment!(unitcell, su_sweep.Χ);
    projectors = Projectors{su_sweep.ctm_type}();
    energy_ver = [];
    energy_hor = [];
    ϵ_energy = 0.0;

    for s ∈ 1:su_sweep.max_su_steps
        #! do_su_iteration!(unitcell, hamiltonian, su_sweep.τ);
        if mod(s, su_sweep.eval_freq) == 0 # Calculate energies and expectation values
            update_environment!(unitcell, projectors, su_sweep);
            hij_hor = Operator(reshape(hamiltonian.hij, (2, 2, 2, 2)), [(1, 1), (1, 2)]);
            hij_ver = Operator(reshape(hamiltonian.hij, (2, 2, 2, 2)), [(1, 1), (2, 1)]);
            push!(energy_hor, calculate_exp_val(unitcell, hij_hor)); # Calculate energy along a single horizontal bond
            push!(energy_ver, calculate_exp_val(unitcell, hij_ver)); # Calculate energy along a single vertical bond
        end

        # Calculates error energies
        conv_order = su_sweep.conv_order;
        ΔE_ver = sum([abs(energy_ver[end] - energy_ver[end-n]) for n ∈ 1:conv_order])/conv_order;
        ΔE_hor = sum([abs(energy_hor[end] - energy_hor[end-n]) for n ∈ 1:conv_order])/conv_order;
        ϵ_energy = 0.5 * (ΔE_hor + ΔE_ver);

        if ϵ_energy < su_sweep.tol_energy
            @info "Converged after $s steps with τ=$τ to a bond energy = $(energy_hor[s])"

            su_sweep.β = s * su_sweep.τ;
            su_sweep.energies_hor = energy_hor;
            su_sweep.energies_ver = energy_ver;

            return 0
        end
    end

    @warn "Reached maximum number of SU steps without convergence. Final error = $(0.5 * (ΔE_hor + ΔE_ver))"

    return ϵ_energy
end
