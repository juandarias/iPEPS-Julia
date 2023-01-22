using Revise
using MKL
using DrWatson
@quickactivate

push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5
using LinearAlgebra
using TensorOperations

using ipeps_ctm
using argparser
include(srcdir("simulation_types.jl"))
include(srcdir("load_methods.jl"))


##############
# Parameters #
##############

args_dict = collect_args(ARGS)

const infolder = get_param!(args_dict, "infolder", "None")
const outfolder = get_param!(args_dict, "outfolder", "None")
const dt = get_param!(args_dict, "dt", 0.01);
const steps = get_param!(args_dict, "steps", 10);
const step_start = get_param!(args_dict, "step_start", 10);
const step_end = get_param!(args_dict, "step_end", 15);
const D0 = get_param!(args_dict, "D0", 3);
const Dt = get_param!(args_dict, "Dt", 4);
const Chi0 = get_param!(args_dict, "Chi0", 50);
const Chi = get_param!(args_dict, "Chi", 50);
const SC = get_param!(args_dict, "SC", 15);
const h = get_param!(args_dict, "h", 2.5);


const tol_ctm = get_param!(args_dict, "tol_ctm", 1e-8);
const ctm_steps = get_param!(args_dict, "ctm_steps", 20);


##################
# CTM parameters #
##################
dims = (SC, SC);

ctm = GENERIC_CTM()
ctm.Χ = Chi;
ctm.max_ctm_steps = ctm_steps;
ctm.tol_ctm = tol_ctm;


##############################
# Calculation of observables #
##############################

##### Load ground state #####
gs_file = projectdir("input/Ising/$(SC)x$(SC)/Psi0_VU_15x15_B2.5_D3_X40.h5")
Psi0_As, Psi0_Rs = load_ctm_matlab(gs_file, dims; load_environment = false);
Ψ0 = UnitCell(deepcopy(Psi0_Rs), deepcopy(Psi0_As));

psit_file_root = "$(SC)x$(SC)_B$(h)_D0$(D0)_Dt$(Dt)_X$(Chi)_t";
rho_xy = Array{Array{ComplexF64,2},2}(undef, dims);

#for s ∈ step_start:step_end
s =1
    t = s * dt;

    # Location of results
    f = h5open(datadir("Ising/$(SC)x$(SC)/$(psit_file_root)$(t).h5"), "w");
    create_group(f, "Re_rho");
    create_group(f, "Im_rho");
    create_group(f, "S+S+");
    create_group(f, "S+S-");
    create_group(f, "N");


    # Load time-evolved state
    filepath = projectdir("input/Ising/$(SC)x$(SC)/$(psit_file_root)$(t).h5")
    As, Rs = load_ctm_matlab(filepath, dims; load_environment = false);
    Ψt = UnitCell(deepcopy(Rs), deepcopy(As));


    # Creates new unitcell with new environment and two layers of tensors
    ΨΦ = braket_unitcell(Ψt, Ψ0);

    # Reconverge environment
    @info "Reconverging environment for time $t"

    projectors = Projectors{EachMove}(ΨΦ);
    error_CTM = update_environment!(ΨΦ, projectors, ctm)


    get_param!(args_dict, "ctm_error", error_CTM);


    # Calculate reduced density matrices
    @info "Calculating reduced density matrices and observables time $t"
    for xy ∈ CartesianIndices(dims)
        rho = calculate_rdm(ΨΦ, xy);
        rho_xy[xy] = rho;
        f["Re_rho/x$(xy[1])_y$(xy[2])"] = real(rho);
        f["Im_rho/x$(xy[1])_y$(xy[2])"] = imag(rho);
    end

    # Calculate observables
    Sp = [0 0.5; 0 0];
    Sm = [0 0; 0.5 0];
    for xy ∈ CartesianIndices(dims)
        rho = rho_xy[xy];
        @tensor SpSp = rho[α, β] * Sp[α, β];
        @tensor SpSm = rho[α, β] * Sm[α, β];
        n = tr(rho);
        f["S+S+/x$(xy[1])_y$(xy[2])"] = real(SpSp);
        f["S-S+/x$(xy[1])_y$(xy[2])"] = real(SpSm);
        f["N/x$(xy[1])_y$(xy[2])"] = real(n);
    end

    bson(datadir("Ising/$(SC)x$(SC)/$(psit_file_root)$t.bson"), args_dict);
    close(f);
#end
