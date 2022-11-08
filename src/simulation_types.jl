
#abstract type Simulation end #! already defined in ipeps_ctm module

#= Structure for a ground state simple-update optimization =#
mutable struct GROUNDSTATE_SU2 <: Simulation

    #= Input parameters =#
    hamiltonian::Hamiltonian

    ctm_type # Type of renormalization for CTM
    ctm_Χ::Int64
    ctm_convergence # Convergence criteria for CTM
    max_ctm_steps::Int64
    tol_ctm::Float64

    τ::Float64
    max_su_steps::Int64 # Max number of SU steps
    conv_order::Int64 # Number of energy calculations for convergence check
    eval_freq::Int64 # Frequency of calculation of energies/observables
    tol_energy::Float64

    #= Output =#
    conv_ctm_steps::Int64 # Number of CTM steps used for convergence
    ctm_error::Float64

    β::Float64 # Final β of SU


    energies_hor::Vector{Vector{Float64}}
    energies_ver::Vector{Vector{Float64}}

    observables::Vector{Operator}
    expectation_values::Vector{Vector{Float64}}

    GROUNDSTATE_SU2() = new();
end
