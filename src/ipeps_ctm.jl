module ipeps_ctm

    #@info "Code is not type-stable, leading to long compilation times. See https://www.juliabloggers.com/writing-type-stable-julia-code/ and https://blog.sintef.com/industry-en/writing-type-stable-julia-code/ for tips on how to fix this! and https://docs.julialang.org/en/v1/manual/performance-tips/#man-code-warntype"
    ###########
    # Modules #
    ###########

    import LinearAlgebra: svd, qr, norm, opnorm, tr, diagm, normalize, Hermitian, eigen, normalize!, SVD
    import Base: +
    #import IterativeSolvers: svdl
    #import KrylovKit: svdsolve
    import PROPACK: tsvd as ptsvd #! only one working
    #import Combinatorics: permutations
    using TensorOperations
    #using PrecompileSignatures: @precompile_signatures #* Speeds up first call of methods


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
    export Tensor, SimpleUpdateTensor, ReducedTensor, BraTensor
    export UnitCell, Environment
    export Operator
    export LatticeSymmetry, R4, XY, UNDEF

    #= iPEPS methods =#
    export initialize_environment!, reinitialize_environment!
    export braket_unitcell
    export apply_operator
    export overlap
    export do_full_contraction
    export calculate_rdm


    #= SU methods =#
    #export update_cell!
    #export prepare_su_tensor
    #export restore_su_tensor
    #export apply_gate

    #= CTM types =#
    export Direction, UP, RIGHT, DOWN, LEFT, VERTICAL, HORIZONTAL # CTM moves and legs direction
    export Renormalization, Projectors, Start, EachMove
    export Convergence, Observable, SingularValues

    # Projectors

    #= CTM methods =#
    export update_environment!
    export do_ctmrg_iteration!
    #export calculate_projectors_ctmrg!
    #export do_ctm_move!

    #= General methods =#
    export cast_tensor, cast_tensor!
    export symmetrize
    export tensor_svd

    #= Aux/helper methods =#
    export coord

    #= Others =#
    #export GROUNDSTATE_SU2
    export log_message


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
    include("./aux_methods.jl")
    include("./logger.jl")

    ##########
    # Others #
    ##########
    #@precompile_signatures(ipeps_ctm)

    include("precompiles.jl")
    _precompile_();

end
