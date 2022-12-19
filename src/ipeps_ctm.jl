module ipeps_ctm

    @info "Code is not type-stable, leading to long compilation times. See https://www.juliabloggers.com/writing-type-stable-julia-code/ for tips on how to fix this!"
    ###########
    # Modules #
    ###########

    import LinearAlgebra: svd, qr, norm, opnorm, tr, diagm, normalize, Hermitian, eigen, normalize!, ishermitian
    import Combinatorics: permutations
    using TensorOperations
    using PrecompileSignatures: @precompile_signatures #* Speeds up first call of methods


    ###############
    # Definitions #
    ###############

    abstract type Simulation end
    abstract type Hamiltonian end


    ###########
    # Exports #
    ###########

    #= Other types =#
    export Simulation
    export Hamiltonian

    #= iPEPS types =#
    export Tensor, SimpleUpdateTensor, ReducedTensor
    export UnitCell, Environment
    export Operator
    export LatticeSymmetry, R4, XY, UNDEF

    #= iPEPS methods =#
    export initialize_environment!, generate_environment_tensors
    export update_cell!
    export prepare_su_tensor
    export restore_su_tensor
    export apply_gate
    export calculate_exp_val
    export implode #? Is there a better name?

    #= CTM types =#
    export Direction, UP, RIGHT, DOWN, LEFT, VERTICAL, HORIZONTAL # CTM moves and legs direction
    export Renormalization, Projectors, Start, EachMove

    # Projectors

    #= CTM methods =#
    export update_environment!
    export do_ctmrg_iteration!
    export calc_projectors_ctmrg!

    #= General methods =#
    export cast_tensor, cast_tensor!
    export symmetrize

    #= Others =#
    #export GROUNDSTATE_SU2

    ###########
    # Imports #
    ###########

    #= Types =#
    include("./ipeps_types.jl")
    include("./CTM_types.jl")
    #include("./simulation_types.jl")

    #= Methods =#
    include("./tensor_methods.jl")
    include("./ipeps_methods.jl")
    include("./CTM_methods.jl")

    #= Others =#
    #include("./simulation_types.jl")

    ##########
    # Others #
    ##########
    @precompile_signatures(ipeps_ctm)

    include("precompiles.jl")
    _precompile_();

end
