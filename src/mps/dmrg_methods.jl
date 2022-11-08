module dmrg_methods

using MKL
import Base: conj, copy, prod
import LinearAlgebra: normalize!, svd, diagm, norm, tr, qr

using Dates: now, Time, DateTime #! for logger
using LinearMaps 
#using IterativeSolvers: cg!, gmres!, bicgstabl! #! for mps_compressor
#using KrylovKit: linsolve #! for mps_compressor
#using Optim #! for mps_compressor
#using OptimBase #! for mps_compressor
#using LineSearches: HagerZhang #! for mps_compressor
using TensorOperations
using UnicodePlots: scatterplot #! for mps_compressor


##### Exports #####
###################

## Methods
export copy
export overlap
export prod, prod!
export norm

export cast_mps
export cast_mpo
#export cast_uMPS #! incomplete
export calc_env
export calc_expval
export calc_Ut
export calc_Wt

#export conj
export sweep_qr!, sweep_svd!
export grow_mps_tensors!
export normalize!

export vector_to_mps
export operator_to_mpo

## Compressor
export compress_tensor #, compress_tensor_cg, compress_tensor_lbfgs
export mps_compress_var, mps_compress_svd! #, mps_compress_cg, mps_compress_lbfgs
export mpo_compress

## Tools
export log_message

##### Imports #####
###################

## Types
include("./dmrg_types.jl")
include("./compressor_types.jl")

## General
include("./dmrg_timedep.jl")
include("./mpo_methods.jl")
include("./mps_methods.jl")

## Compressor
include("./mps_compressor.jl")

## Tools
#import ....JOB_ID
include("./logger.jl")
#using logger


##### Methods #####
###################

"""
    function prod(mps::MPS, mpo::WII)

Calculates the product between a finite, generic MPS and the ``W_II`` representation of time-evolution propagator

"""
function prod(mps::MPS, mpo::WII)
    L = mps.L;
    new_mps = Vector{Array{ComplexF64,3}}();
    
    @tensor Ai_new[lmps, lmpo, phys, rmps, rmpo] := mps.Ai[1][lmps, x, rmps]*mpo.W1[x, lmpo, phys, rmpo];
    push!(new_mps, reshape(Ai_new, 1, 2, mps.D[1]*(mpo.Nj+1))); #! add to WII type dimensions
    
    
    for i ∈ 2:L-1 #! can be parallelized!
        @tensor Ai_new[lmps, lmpo, phys, rmps, rmpo] := mps.Ai[i][lmps, x, rmps]*mpo.Wi[x, lmpo, phys, rmpo];
        push!(new_mps, reshape(Ai_new, mps.D[i-1]*(mpo.Ni+1), 2, mps.D[i]*(mpo.Nj+1)))
    end

    @tensor Ai_new[lmps, lmpo, phys, rmps, rmpo] := mps.Ai[L][lmps, x, rmps]*mpo.WN[x, lmpo, phys, rmpo];
    push!(new_mps, reshape(Ai_new, mps.D[L-1]*(mpo.Ni+1), 2, 1)); #! add to WII type dimensions
    
    return MPS(new_mps)
end

"""
    function prod(mps::MPS, mpo::MPO)

Calculates the product between a finite, generic MPS and a generic MPO

"""
function prod(mps::MPS, mpo::MPO)
    L = mps.L;
    new_mps = Vector{Array{ComplexF64,3}}();
    
    for i ∈ 1:L #! can be parallelized!
        @tensor Ai_new[lmps, lmpo, phys, rmps, rmpo] := mps.Ai[i][lmps, x, rmps]*mpo.Wi[i][x, lmpo, phys, rmpo];
        i==1 && push!(new_mps, reshape(Ai_new, 1, 2, mps.D[i]*mpo.D[i]));
        i==L && push!(new_mps, reshape(Ai_new, mps.D[i-1]*mpo.D[i-1], 2, 1));
        i!=1 && i!=L && push!(new_mps, reshape(Ai_new, mps.D[i-1]*mpo.D[i-1], 2, mps.D[i]*mpo.D[i]));
    end

    return MPS(new_mps)
end


function calc_expval(mps::MPS, op::MPO, loc::Int; mixed_canonical::Bool=false)
    @assert false "Method requires defining contraction A*O*A† for ALL sites"
    L = mps.L;
    ci_ket = mps.Ai[1];
    𝟙 = [1 0; 0 1];
    if mixed_canonical == false
        for i ∈ 1:L-1 # contracts left to right
            if i == loc
                @tensor EiL[a,b] := ci_ket[x,u,a]*op[u,d]*conj(mps.Ai[i])[x,d,b]# A_i*O*A†_i
            else
                @tensor EiL[a,b] := ci_ket[x,u,a]*𝟙[u,d]*conj(mps.Ai[i])[x,d,b] # A_i*A†_i
            end
            @tensor ci_ket[a,b,c] := EiL[x,a]*mps.Ai[i+1][x,b,c] 
        end
        loc == L ? (@tensor expval = ci_ket[x,u,z]*op[u,d]*conj(mps.Ai[L])[x,d,z]) : (@tensor expval = ci_ket[x,u,z]*𝟙[u,d]*conj(mps.Ai[L])[x,d,z])
    else
        @assert loc == mps.oc "Ortohonality center and location of operator are different"
        @tensor expval = mps.Ai[loc][x, u, y]*op[u ,d]*conj(mps.Ai[loc])[x, d, y]
    end

    return expval
end


function calc_expval(mps::MPS, op::Array{T, 2}, loc::Int; mixed_canonical::Bool=false) where {T}
    L = mps.L;
    ci_ket = mps.Ai[1];
    𝟙 = [1 0; 0 1];
    if mixed_canonical == false
        for i ∈ 1:L-1 # contracts left to right
            if i == loc
                @tensor EiL[a,b] := ci_ket[x,u,a]*op[u,d]*conj(mps.Ai[i])[x,d,b]# A_i*O*A†_i
            else
                @tensor EiL[a,b] := ci_ket[x,u,a]*𝟙[u,d]*conj(mps.Ai[i])[x,d,b] # A_i*A†_i
            end
            @tensor ci_ket[a,b,c] := EiL[x,a]*mps.Ai[i+1][x,b,c] 
        end
        loc == L ? (@tensor expval = ci_ket[x,u,z]*op[u,d]*conj(mps.Ai[L])[x,d,z]) : (@tensor expval = ci_ket[x,u,z]*𝟙[u,d]*conj(mps.Ai[L])[x,d,z])
    else
        @assert loc == mps.oc "Ortohonality center and location of operator are different"
        @tensor expval = mps.Ai[loc][x, u, y]*op[u ,d]*conj(mps.Ai[loc])[x, d, y]
    end

    return expval
end


end