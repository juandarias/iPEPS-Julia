using MKL
using DrWatson
@quickactivate

push!(LOAD_PATH, srcdir());
push!(LOAD_PATH, scriptsdir());

using HDF5
using BSON
using LinearAlgebra
using TensorOperations

using argparser
args_dict = collect_args(ARGS)
global JOB_ID = get_param!(args_dict, "ID", "r_id$(string(rand(1:100)))");

using ipeps_ctm

#include(srcdir("simulation_types.jl"))
include(srcdir("load_methods.jl"))

@info "Modules loaded!!!"

##############
# Parameters #
##############

const datastore = get_param!(args_dict, "datastore", projectdir(""))
const infolder = get_param!(args_dict, "infolder", "$(datastore)input/Ising/")
const outfolder = get_param!(args_dict, "outfolder", "$(datastore)data/Ising/")
const dt = get_param!(args_dict, "dt", 0.01);
const steps = get_param!(args_dict, "steps", 10);
const step_start = get_param!(args_dict, "step_start", 1);
const step_end = get_param!(args_dict, "step_end", 5);
const D0 = get_param!(args_dict, "D0", 3);
const Dt = get_param!(args_dict, "Dt", 4);
const Chi = get_param!(args_dict, "Chi", 50);
const Chi0 = get_param!(args_dict, "Chi0", 50);
const SC = get_param!(args_dict, "SC", 5);
const h = get_param!(args_dict, "h", 2.5);


const tol_ctm = get_param!(args_dict, "tol_ctm", 1e-12);
const tol_expval = get_param!(args_dict, "tol_expval", 1e-3);
const ctm_steps = get_param!(args_dict, "ctm_steps", 2);
const op_loc = get_param!(args_dict, "op_loc", (2,2));
const conv_obs = get_param!(args_dict, "conv_obs", true);

log_message("\n ##### Parameters #####", color = :blue)
log_message("\n$(string(args_dict)[16:end-1])\n", color = :blue)


##################
# CTM parameters #
##################
dims = (SC, SC);

ctm = GENERIC_CTM()
ctm.Χ = Chi;
ctm.max_ctm_steps = ctm_steps;
ctm.tol_ctm = tol_ctm;
ctm.full_svd = false

if conv_obs == true
    ctm.tol_expval = tol_expval;
    ctm.ctm_convergence = Observable
    Sz_op = Operator(0.5*[1.0 0.0 ; 0.0 -1.0], [op_loc]);
    Sz_op.name = "⟨Z⟩";
    ctm.observables = [Sz_op];
end


##############################
# Calculation of observables #
##############################


##### Load ground state #####
gs_file = "$(infolder)$(SC)x$(SC)/Psi0_VU_1x1_B2.5_D3_X40.h5"
#gs_file = "$(infolder)$(SC)x$(SC)/Psi0_VU_$(SC)x$(SC)_B2.5_D3_X40.h5"
log_message("\nInitial state: $gs_file ")
Psi0_As, Psi0_Rs = load_ctm_matlab(gs_file, dims; load_environment = false);
Ψ0 = UnitCell(deepcopy(Psi0_Rs), deepcopy(Psi0_As));
log_message(" -> Loaded \n")

s = 1;
t = Float64(s * dt);
file_name = "$(SC)x$(SC)_B$(h)_D0$(D0)_Dt$(Dt)_X$(Chi)_t$(t)";

# RDM
filepath = "$(infolder)$(SC)x$(SC)/$(file_name).h5"
As, Rs = load_ctm_matlab(filepath, dims; load_environment = false);
log_message("\nLoaded state: $file_name \n");

Ψt = UnitCell(deepcopy(Rs), deepcopy(As));

# Creates new unitcell with new environment and two layers of tensors
ΨΦ = braket_unitcell(Ψt, Ψ0);

# Reconverge environment
log_message("\nReconverging environment\n")

projectors = Projectors{EachMove}(ΨΦ);
converged = update_environment!(ΨΦ, projectors, ctm)
