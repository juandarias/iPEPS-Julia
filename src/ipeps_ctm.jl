module ipeps_ctm

    ###########
    # Modules #
    ###########

    import LinearAlgebra: svd, qr, norm, opnorm, tr, diagm, normalize, Hermitian, eigen, normalize!, ishermitian
    import Combinatorics: permutations
    using TensorOperations

    #! Debug
    import UnicodePlots: heatmap, scatterplot


    ###############
    # Definitions #
    ###############

    abstract type Simulation end
    abstract type Hamiltonian end
    SvC1 = [];

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
    export LatticeSymmetry, C4, XY, UNDEF

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
    export Renormalization, Projectors, TwoCorners, TwoCornersSimple, HalfSystem, Start, EachMove, EachMoveCirc # Projectors
    export ConvergenceCriteria, OnlyCorners, Full

    #= CTM methods =#
    export update_environment!
    export do_ctm_iteration!
    export calc_projectors!

    #! Debug
    export factorize_rho
    export dctm_move!
    export update_tensors!
    #! Debug



    #= General methods =#
    export cast_tensor, cast_tensor!
    export symmetrize

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

end
