############################## Methods ##############################

conj(mps::MPS) = MPS(Base.conj(mps.Ai));

function prod!(c::Number, mps::MPS)
    mps.Ai = mps.Ai * c;
    return nothing
end

function norm(mps::MPS)
    if mps.oc != -1 && mps.canonical != None();
        @tensor norm_sq = mps.Ai[mps.oc][α, β, γ] * conj(mps.Ai[mps.oc])[α, β, γ];
    else
        norm_sq = overlap(mps, mps);
    end

    return sqrt(abs(norm_sq))
end

function copy(mps::MPS)
    new_mps = MPS(copy(mps.Ai)); # copies tensors
    for field in propertynames(mps)
        setfield!(new_mps, field, getfield(mps, field));
    end
    return new_mps
end

function update_tensor!(mps::MPS, tensor::Array{T, 3}, loc::Int) where {T}
    mps.Ai[loc] = tensor;
    loc == mps.L && (mps.D[loc-1] = size(tensor)[1]);
    loc == 1 && (mps.D[1] = size(tensor)[3]);
    loc != mps.L && (mps.D[loc] = size(tensor)[3]);
    loc != 1 && (mps.D[loc-1] = size(tensor)[1]);
    return nothing    
end

function normalize!(mps::MPS)
    L = mps.L;
    if mps.canonical == None() && maximum(mps.D) > 100 # For large tensors, reduces the memory cost of calculating the norm at the price of canonizing the MPS
        sweep_qr!(mps);
    end
    n = norm(mps);
    nxs = n^(-1/L);
    prod!(nxs, mps);
    #for n ∈ 1:L
    #    mps.Ai[n] = mps.Ai[n]/nxs;
    #end
    return nothing
end


"""
    function overlap(ket::MPS, bra::MPS)

Generic method to calculate the overlap between two MPS's

"""
function overlap(ket::MPS, bra::MPS)
    L = ket.L;
    bra = conj(bra); # transposition is taken care in label of legs
    overlap = 0.0;
    
    ci_ket = ket.Ai[1];
    for i ∈ 1:L-1
        @tensor EiL[r1, r2] := ci_ket[α, β, r1] * bra.Ai[i][α, β, r2]; # Calculates transfer matrix E_i
        @tensor ci_ket[l, u, r] := EiL[α, l] * ket.Ai[i+1][α, u, r]; # Contracts E_i with ket A_i+1
        # @tensor EiL[a,b] := ci_ket[x,y,a]*bra.Ai[i][x,y,b]
    end
    @tensor overlap = ci_ket[x,y,z] * bra.Ai[L][x,y,z]

    return overlap
    
end



"""
    function overlap(ket::MPS, bra::MPS, L::Int)

Method to calculate the overlap between two translation invariant MPS's. Calculates the power transfer matrix of the translation invariant region

"""
function overlap(ket::MPS, bra::MPS, L::Int)
    bra = conj(bra); # transposition is taken care in label of legs
    overlap = 0.0;
    
    @tensor Ei[a,b,c,d] := ket.Ai[2][a,x,c]*bra.Ai[2][b,x,d]; # Calculates transfer matrix E_i
    EN = Ei^L;
    ### TODO
    @assert false "Function is not complete";
    # contract left and right tensors A1, AN
    return overlap
    
end


"""
    function sweep_svd!(mps::MPS; final_site::Int = mps.L, direction::String = "left", Dmax::Int = 2^50, ϵmax::Float64 = 0.0, spectrum::Bool = false)

Calculates the canonical form up to site `final_site` starting from the leftmost (first) or rightmost (last) tensor using a SVD at each site. If the input `MPS` is already in left(right)-canonical form, brings it into mixed canonical form by choosing the sweep `direction` to be `right`(`left`). The maximum number of kept values can be controlled by ``D_\\text{max}`` or such that the sum of the square discarded values is below a cut-off ``\\epsilon_\\text{max}``. If the MPS bond dimension is truncated, the preserved singular values are rescaled to return a norm one state. The singular values are returned 

## Arguments
- mps : MPS to be compressed
- Dmax : Maximum number of singular values to be kept
- ϵmax : Cut-off for the sum of the square discarded values
- spectrum : Returns the spectrum of singular values at each bond

"""
function sweep_svd!(mps::MPS; final_site::Int = mps.L, direction::String = "left", Dmax::Int = 2^50, ϵmax::Float64 = 0.0, spectrum::Bool = false, normalize::Bool = true)
    #Ai_new = Vector{Array{ComplexF64,3}}();
    L = mps.L;
    d = mps.d;

    local Daux;
    local ϵcomp = 0.0;
    local singular_values = Vector{Vector{Float64}}();

    function Dcutoff(s::Vector{Float64}, Dmax::Int, ϵ::Float64)
        D = length(s);
        sum_disc = 0.0;
        n = 0;
        if ϵ != 0.0
            while sum_disc < ϵ && n < D
                sum_disc += s[end - n]^2
                n += 1;
            end
            n += -1; # to cancel the last step
        end
        Dkeep = D - n;
        return min(Dkeep, Dmax)
    end
    
    if direction == "left" && mps.oc != L
        mps.canonical == None() && (mps.oc = 1;) 
        @assert mps.oc < final_site "New orthogonality center must be right from current one"
        
        Atilde = reshape(mps.Ai[mps.oc], (:, mps.D[mps.oc]));
        
        for i ∈ mps.oc:final_site-1
            Ui, Si, Vi = svd(Atilde);
            push!(singular_values, Si/sum(Si.^2));
            Daux = Dcutoff(Si, Dmax, ϵmax);
            if Daux < length(Si)
                ϵcomp += sum(Si[Daux+1:end].^2); # sum of discarded values
            end
            
            normalize == true && (Si = Si/sqrt(sum(Si[1:Daux].^2));) # Normalization
            Atilde = diagm(Si[1:Daux]) * Vi[:,1:Daux]' * reshape(mps.Ai[i+1], (mps.D[i], :)); #! Sᵢ*Vᵢ†*Aᵢ₊₁
            Atilde = reshape(Atilde, (d * Daux, :));
            update_tensor!(mps, reshape(Ui[:, 1:Daux], :, d, Daux), i);
        end
        update_tensor!(mps, reshape(Atilde, (Daux, d, :)), final_site);

        mps.canonical == None() && direction == "left" && (mps.canonical = Left();)
        mps.canonical == Right() && direction == "left" && (mps.canonical = Mixed();)
        mps.canonical == Right() && direction == "left" && final_site == L && (mps.canonical = Left();)
        mps.oc = final_site;

    elseif direction == "right" && mps.oc != 1
        mps.canonical == None() && (mps.oc = L;) 
        @assert mps.oc > final_site "New orthogonality center must be left from current one"
        
        Atilde = reshape(mps.Ai[mps.oc], (mps.D[mps.oc-1], :));
    
        for i ∈ mps.oc:-1:final_site+1
            Ui, Si, Vi = svd(Atilde);
            push!(singular_values, Si/sum(Si.^2));
            Daux = Dcutoff(Si, Dmax, ϵmax); 
            if Daux < length(Si)
                ϵcomp += sum(Si[Daux+1:end].^2); 
            end
            normalize == true && (Si = Si/sqrt(sum(Si[1:Daux].^2));) # Normalization
            Atilde = reshape(mps.Ai[i-1], :, mps.D[i-1]) * Ui[:, 1:Daux] * diagm(Si[1:Daux]);
            #Atilde = diagm(Si)*Vi'*reshape(mps.Ai[i-1], (mps.D[i-1], :)); #! Sᵢ*Vᵢ†*Aᵢ₊₁
            Atilde = reshape(Atilde, (:, d * Daux));
            update_tensor!(mps, reshape(collect(Vi[:,1:Daux]'), (Daux, d, :)), i);
        end
        reverse!(singular_values);
        update_tensor!(mps, reshape(Atilde, (:, d, Daux)), final_site);
        
        mps.canonical == None() && direction == "right" && (mps.canonical = Right();)
        mps.canonical == Left() && direction == "right" && (mps.canonical = Mixed();)
        mps.canonical == Left() && direction == "right" && final_site == 1 && (mps.canonical = Right();)
        mps.oc = final_site;
    end 
    
    if spectrum == true
        return ϵcomp, singular_values
    else
        return ϵcomp
    end
end


function sweep_qr!(mps::MPS; final_site::Int = mps.L, direction::String = "left")
    #Ai_new = Vector{Array{ComplexF64,3}}();
    L = mps.L;
    d = mps.d;
    
    if direction == "left" && mps.oc != L
        mps.canonical == None() && (mps.oc = 1;) 
        @assert mps.oc < final_site "New orthogonality center must be right from current one"
        
        Atilde = reshape(mps.Ai[mps.oc], (:, mps.D[mps.oc]));
        
        for i ∈ mps.oc:final_site-1
            Qi, Ri = qr(Atilde);
            Qi = Matrix(Qi);
            update_tensor!(mps, reshape(Qi, Int(size(Qi,1)/d), d, :), i);

            Atilde = Ri * reshape(mps.Ai[i+1], (size(Ri,2), :)); 
            Atilde = reshape(Atilde, (d * mps.D[i], :));
        end
        update_tensor!(mps, reshape(Atilde, (:, d, size(Atilde, 2))), final_site);

        mps.canonical == None() && direction == "left" && (mps.canonical = Left();)
        mps.canonical == Right() && direction == "left" && (mps.canonical = Mixed();)
        mps.oc = final_site;

    elseif direction == "right" && mps.oc != 1
        mps.canonical == None() && (mps.oc = L;) 
        @assert mps.oc > final_site "New orthogonality center must be left from current one"
        
        Atilde = reshape(mps.Ai[mps.oc], (mps.D[mps.oc-1], :));
    
        for i ∈ mps.oc:-1:final_site+1
            Qi, Ri = qr(adjoint(Atilde)); # Ai = R†Q†
            Qi = Matrix(Qi);
            update_tensor!(mps, reshape(collect(Qi'), (:, d, Int(size(Qi, 1)/d))), i);
            Atilde = reshape(mps.Ai[i-1], :, size(Ri, 2)) * adjoint(Ri); #Ai-1*Ri†
            Atilde = reshape(Atilde, (Int(size(Atilde, 1)/d), :));
        end
        update_tensor!(mps, reshape(Atilde, (size(Atilde, 1), d, :)), final_site);
        
        mps.canonical == None() && direction == "right" && (mps.canonical = Right();)
        mps.canonical == Left() && direction == "right" && (mps.canonical = Mixed();)
        mps.oc = final_site;
    end 
    
    return nothing
end


function vector_to_mps(state::Vector{T}; Dmax::Int=Int(sqrt(length(state)) ÷ 1 )) where {T<:Number}
    L = Int(log2(length(state)));
    tensors = Vector{Array{T,3}}();
    ck = state;
    Dleft = 1;
    for k ∈ 1:L-1
        Ψk = reshape(ck, (2*Dleft, 2^(L-k))); #reshape vector to matrix
        Uk, Sk, Vk = svd(Ψk); 
        Dright = min(length(Sk), Dmax);
        push!(tensors, reshape(view(Uk, :, 1:Dright), (Dleft, 2, Dright))); #add site tensors to vector
        ck = diagm(Sk[1:Dright])*Vk'[1:Dright,:]; #remaining state to be factorized
        Dleft = Dright;
    end
    push!(tensors, reshape(ck, (:, 2, 1))); #add site tensors to 
    return MPS(tensors)
end


function grow_mps_tensors!(mps::MPS, Dmax::Int64)
    L = mps.L;
    d = mps.d;
    D = copy(mps.D);
    mps.oc = -1;
    mps.canonical = None();

    for i ∈ 1:L
        Dl = Dr = Dmax;
        i == 1 && (Dl=1);
        i == L && (Dr=1);
        new_ten = zeros(ComplexF64, Dl, d, Dr);
        for di ∈ 1:d
            (i != 1 && i != L) && (new_ten[1:D[i-1], di, 1:D[i]] = mps.Ai[i][:, di, :]);
            i == 1  && (new_ten[1, di, 1:D[i]] = mps.Ai[i][:, di, :]);
            i == L && (new_ten[1:D[i-1], di, 1] = mps.Ai[i][:, di, :]);
        end
        update_tensor!(mps, new_ten, i);
    end
    
    return nothing
end


