
conj(mps::MPO) = MPO(Base.conj(mps.Wi))

function prod!(c::Number, mpo::MPO)
    mpo.Wi = mpo.Wi * c;
    return nothing
end

"""
    function norm(Op::MPO)

## Description
Returns the Frobenius norm of the operator
"""
function norm(Op::MPO)
    L = Op.L;
    Li = ones(1,1); # dummy identity matrix
    for i ∈ 1:L
        @tensor Li_U[u, r2, d, r1] := Li[x, r2]*Op.Wi[i][u, x, d, r1] 
        @tensor Li[r1, r2] := Li_U[x, y, z, r1]*conj(Op.Wi[i])[x, y, z, r2] 
    end
    return sqrt(abs(Li[1,1]))
end

function normalize!(Op::MPO)
    L = Op.L;
    n_op = norm(Op);
    prod!(n_op^(-1/L), Op)
    #for n ∈ 1:L
    #    Op.Wi[n] = Op.Wi[n]/(n_op^(1/L));
    #end
    return nothing
end


function update_tensor!(mpo::MPO, tensor::Array{T, 4}, loc::Int) where {T}
    mpo.Wi[loc] = tensor;
    loc == mpo.L && (mpo.D[loc-1] = size(tensor)[1]);
    loc != mpo.L && (mpo.D[loc] = size(tensor)[end]);
    return nothing    
end

"""
    function cast_mps(mpo::MPO; L::Int=mpo.L, normalize=false)

Casts a finite, generic `MPO` into `MPS` form. If the length `L` is not provided, the new `MPS` will have the same length as the original `MPO`. 

### Notes 
- To obtain back an `MPO` use the ``cast_mpo`` method.
- If `normalize = true`, normalizes the MPS by ``\\sqrt \\langle O, O \\rangle`` which is equivalent to the Frobenius norm ``\\sqrt \\tr O^\\dagger O``. 
- Method has been tested by rebuilding MPO from MPS after casting
- TODO: Implement uMPS and uMPO types to avoid copying tensors
- TODO: In case of translation symmetric MPO, containing only the boundary tensors and one bulk tensor, the bulk tensor shall be copied to give the desired length `MPS`. 
"""
function cast_mps(mpo::MPO{T}; L::Int = mpo.L, normalize = false, Dmax::Int = 100) where {T}
    
    mps = MPS([zeros(T, 0,0,0) for i ∈ 1:L]); # initializes MPS of appropiate type
    
    update_tensor!(mps, reshape(permutedims(mpo.Wi[1], (2,1,3,4)), 1, 4, mpo.D[1]), 1);
    for i ∈ 2:L-1
        update_tensor!(mps, reshape(permutedims(mpo.Wi[i], (2,1,3,4)), mpo.D[i-1], 4, mpo.D[i]), i);
    end
    update_tensor!(mps, reshape(permutedims(mpo.Wi[end], (2,1,3,4)), mpo.D[end], 4, 1), L);

    mps.physical_space = BraKet();
    mps.d = mps.d^2;
    
    if normalize == true
        if maximum(mps.D) > Dmax # For large tensors, reduces the memory cost of calculating the norm at the price of doing a QR sweep of the MPS
            sweep_qr!(mps);
            println("Doing a sweep")
        end
        n = norm(mps);
        nxs = n^(-1/L);
        prod!(nxs, mps);
        #for n ∈ 1:L
        #    mps.Ai[n] = mps.Ai[n]/nxs;
        #end
    end

    return mps
end


"""
    function cast_mps(mpo::MPO; normalize=false)

Cast a a ``W_II`` mpo into MPS form. The size of the MPS is obtained form the field `L` of the `mpo`.
"""
function cast_mps(mpo::WII; normalize=false)
    L = mpo.L;   
    Ai = Vector{Array{ComplexF64,3}}();
    
    A1 = reshape(permutedims(mpo.W1, (2,1,3,4)), 1, 4, mpo.Ni+1);
    Ai = reshape(permutedims(mpo.Wi, (2,1,3,4)), mpo.Ni+1, 4, mpo.Nj+1);
    AN = reshape(permutedims(mpo.WN, (2,1,3,4)), mpo.Nj+1, 4, 1);

    mps = MPS([A1, fill(Ai, L)..., AN]); # Creates MPS
    mps.physical_space = BraKet();

    if normalize == true
        n = sqrt(abs(overlap(mps,mps)));
        nxs = n^(-1/L);
        prod!(nxs, mps);
        #for n ∈ 1:L
        #    mps.Ai[n] = mps.Ai[n]/nxs;
        #end
    end

    return mps
end

"""
    function cast_uMPS(umpo::MPO, L=Int; normalize=false)

Casts an uniform open-boundary MPS from an uniform open-boundary MPO, where all the central tensors ``W_i`` are equal. #TODO: might require defining an uMPO and uMPS types.
"""
function cast_uMPS(umpo::MPO, L=Int; normalize=false)
    @assert false "this function requires a uMPS type which is not ready"
    A1 = reshape(umpo.Wi[1], 1, 4, :);
    Ai = reshape(permutedims(umpo.Wi[2], (2,1,3,4)), umpo.D[1], 4, umpo.D[2]);
    AN = reshape(permutedims(umpo.Wi[3], (2,1,3)), :, 4, 1); # aux left, phys down      
    mps = MPS([A1,Ai,AN]);
    
    if normalize == true
        n = sqrt(abs(overlap(mps,mps)));
        nxs = n^(1/L);
        for n ∈ 1:3
            mps.Ai[n] = mps.Ai[n]/nxs;
        end
    end

    return mps
end


"""
    function cast_mpo(mps::MPS{T}) where {T}

Cast a finite MPS with doubled physical Hilbert space into a MPO

"""
function cast_mpo(mps::MPS{T}) where {T}
    L = mps.L;
    Wi = Vector{Array{T,4}}();
    for i ∈ 1:L
        push!(Wi, permutedims(reshape(mps.Ai[i], (:, 2, 2, size(mps.Ai[i])[end])), (2, 1, 3, 4)));
    end
    return MPO(Wi)
end

function operator_to_mpo(operator::Matrix{T}; Dmax::Int=2^50) where {T<:Number}
    L = Int(log2(size(operator)[end]));
    tensors = Vector{Array{T,4}}();

    ck = permutedims(reshape(operator, fill(2, 2*L)...), vcat([[i; L+i] for i ∈ 1:L]...));
    Dleft = 1;
    for k ∈ 1:L-1
        Ψk = reshape(ck, (4*Dleft, 4^(L-k))); #reshape vector to matrix
        Uk, Sk, Vk = svd(Ψk); 
        Dright = min(length(Sk), Dmax);
        push!(tensors, permutedims(reshape(view(Uk, :, 1:Dright), (Dleft, 2, 2, Dright)), (2, 1, 3, 4))); #add site tensors to vector
        ck = diagm(Sk[1:Dright])*Vk'[1:Dright,:]; #remaining state to be factorized
        Dleft = Dright;
    end
    push!(tensors, permutedims(reshape(ck, (:, 2, 2, 1)), (2, 1, 3, 4))); 
    return MPO(tensors)
end


"""
    function prod(mpo_top::MPO{T}, mpo_bottom::MPO{T}; compress::Bool=false, kwargs...) where {T}

Calculates the product of two `MPO`s and compresses the result if required. 

"""
function prod(
    mpo_top::MPO{T}, 
    mpo_bottom::MPO{T}; 
    compress::Bool=false,
    kwargs...
    ) where {T}

    Wi_prod = Vector{Array{T,4}}();
    L = mpo_top.L;
    
    for i ∈ 1:L
        @tensor Wi[u, l1, l2, d, r1, r2] := mpo_top.Wi[i][u, l1, x, r1]*mpo_bottom.Wi[i][x, l2, d, r2];
        i == 1  && push!(Wi_prod, reshape(Wi, (2, 1, 2, mpo_top.D[i]*mpo_bottom.D[i])));
        i != 1 && i != L && push!(Wi_prod, reshape(Wi, (2, mpo_top.D[i-1]*mpo_bottom.D[i-1], 2, mpo_top.D[i]*mpo_bottom.D[i])));
        i == L && push!(Wi_prod, reshape(Wi, (2, mpo_top.D[i-1]*mpo_bottom.D[i-1], 2, 1)));
    end
    
    if compress == false
        return MPO(Wi_prod);
    elseif compress == true
        println("returning compressed MPO")
        mpo, ϵ_c = mpo_compress(MPO(Wi_prod); kwargs...);
        return mpo, ϵ_c
    end
end

function mpo_compress(mpo::MPO; METHOD::COMPRESSOR = SVD, seed::MPS = MPS([zeros(2,2,2)]), normalize::Bool = false, kwargs...)
   
    if METHOD == SIMPLE #* Uses canonical forms of MPS to do a single-site optimization of the MPS. The operator is normalized by default. Requires a normalized seed.
        @assert seed.L == mpo.L "Length of seed MPO and input MPO are not equal, provide a proper seed"
        mps_mpo = cast_mps(mpo; normalize = true);
        mps_mpo, ϵ_c = mps_compress_var(mps_mpo, seed; kwargs...); # Seed must be normalized
        mpo_comp = cast_mpo(mps_mpo);
    elseif METHOD == VAR_CG
        @assert seed.L == mpo.L "Length of seed MPO and input MPO are not equal, provide a proper seed"
        mps_mpo = cast_mps(mpo; normalize = false);
        #mps_mpo = mps_compress_cg(mps_mpo, seed);
        normalize == true && normalize!(mps_mpo);
        mpo_comp = cast_mpo(mps_mpo; kwargs...);
        ϵ_c = 10; #! TODO: not implemented. mps_compress_cg has to return the compression error
    elseif METHOD == VAR_OPTIM
        @assert seed.L == mpo.L "Length of seed MPO and input MPO are not equal, provide a proper seed"
        mps_mpo = cast_mps(mpo; normalize = false);
        #mps_mpo = mps_compress_lbfgs(mps_mpo, seed);
        normalize == true && normalize!(mps_mpo);
        mpo_comp = cast_mpo(mps_mpo; kwargs...);
        ϵ_c = 10; #! TODO: not implemented. mps_compress_lbfgs has to return the compression error
    elseif METHOD == SVD
        mps_mpo = cast_mps(mpo; normalize = false);
        ϵ_c = mps_compress_svd!(mps_mpo; kwargs...);
        mpo_comp = cast_mpo(mps_mpo);
    end
    return mpo_comp, ϵ_c
end

#### Old methods ####
#####################

#= 
function cast_mps_old(mpo::MPO; L::Int=mpo.L, normalize=false)
    Ai = Vector{Array{ComplexF64,3}}();
    
    push!(Ai,reshape(permutedims(mpo.Wi[1], (2,1,3,4)), 1, 4, mpo.D[1]))
    for i ∈ 2:L-1
        # creates a mps of lenght L from a mpo of length L
        L == mpo.L && (push!(Ai,reshape(permutedims(mpo.Wi[i], (2,1,3,4)), mpo.D[i-1], 4, mpo.D[i])));
        # creates a mps of lenght L from a mpo of length 3, e.g. if the mpo is translational 
        # invariant within the boudaries
        L > mpo.L && (push!(Ai,reshape(permutedims(mpo.Wi[2], (2,1,3,4)), mpo.D[1], 4, mpo.D[2])));
    end
    push!(Ai,reshape(permutedims(mpo.Wi[end], (2,1,3,4)), mpo.D[end], 4, 1))

    mps = MPS(Ai); # Creates MPS
    mps.physical_space = BraKet();
    mps.d = mps.d^2;
    
    if normalize == true
        if maximum(mps.D) > 100 # For large tensors, reduces the memory cost of calculating the norm at the price of canonizing the MPS
            canonize!(mps);
        end
        n = norm(mps);
        nxs = n^(1/L);
        for n ∈ 1:L
            mps.Ai[n] = mps.Ai[n]/nxs;
        end
    end

    return mps
end =#