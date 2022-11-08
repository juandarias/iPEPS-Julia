function reconstructcomplexState(mps) #* if the MPS have been transposed during saving to HDF5 correct by setting tranpose = true


    #* Sort A matrices by lattice site. Sites are labelled from right to left in exported states
    sites = length(mps) ÷ 4;
    keys_mps = collect(keys(mps));
    index_key(A_key) = findfirst(x-> x==A_key, keys_mps)

    Atensors =Any[];
    for site ∈ sites-1:-1:0
        Akey_1 = string(site)*"_1_((),())"
        Akey_0 = string(site)*"_0_((),())"
        AiRe_0 = index_key(Akey_0*"Re");
        AiIm_0 = index_key(Akey_0*"Im");
        AiRe_1 = index_key(Akey_1*"Re");
        AiIm_1 = index_key(Akey_1*"Im");
        Acomplex_0 = im*mps[keys_mps[AiIm_0]] + mps[keys_mps[AiRe_0]]; #Re[A] + i Im[A]
        Acomplex_1 = im*mps[keys_mps[AiIm_1]] + mps[keys_mps[AiRe_1]]; #Re[A] + i Im[A]
        
        if site == sites-1
            push!(Atensors, vcat(Acomplex_0,Acomplex_1)); #first site
        elseif site == 0
            push!(Atensors, hcat(Acomplex_0,Acomplex_1)); #last site
        else
            Ai = zeros(ComplexF64, 2,size(Acomplex_0)...)
            Ai[1,:,:] = Acomplex_0;
            Ai[2,:,:] = Acomplex_1;
            push!(Atensors, Ai)
        end
    end
    
    #* Contract tensors
    IndexArray = [[-n,n-1,n] for n=2:sites-1]
    prepend!(IndexArray, [[-1,1]]); #first site 
    append!(IndexArray, [[sites-1,-sites]]); #last site
    res = ncon(Atensors, IndexArray);
    state = reshape(res, 2^sites);
    Atensors = nothing;
    return state
end;



function MPS_tensors(mps)
    
    #* Sort A matrices by lattice site. Sites are labelled from right to left in exported states
    sites = length(mps) ÷ 4;
    keys_mps = collect(keys(mps));
    index_key_A(A_key) = findfirst(x-> x==A_key, keys_mps)

    Atensors = Array{ComplexF64}[];
    for site ∈ sites-1:-1:0
        key_1 = string(site)*"_1_((),())"
        key_0 = string(site)*"_0_((),())"
        AiRe_0_A = index_key_A(key_0*"Re");
        AiIm_0_A = index_key_A(key_0*"Im");
        AiRe_1_A = index_key_A(key_1*"Re");
        AiIm_1_A = index_key_A(key_1*"Im");
        Acomplex_0 = im*mps[keys_mps[AiIm_0_A]] + mps[keys_mps[AiRe_0_A]]; #Re[A] + i Im[A]
        Acomplex_1 = im*mps[keys_mps[AiIm_1_A]] + mps[keys_mps[AiRe_1_A]]; #Re[A] + i Im[A]
        
        
        if site == sites-1
            push!(Atensors, vcat(Acomplex_0,Acomplex_1)); #first site
        elseif site == 0
            push!(Atensors, hcat(Acomplex_0,Acomplex_1)); #last site
        else
            Ai = zeros(ComplexF64, 2,size(Acomplex_0)...)
            Ai[1,:,:] = Acomplex_0;
            Ai[2,:,:] = Acomplex_1;
            push!(Atensors, Ai)
        end
    end
    
    return Atensors
end


"""
# Will calculate the conjugate of B before calculating he overlap
"""
function MPS_fidelity(tensors_A, tensors_B) #* if the MPS have been transposed during saving to HDF5 correct by setting tranpose = true
    
    #* Contract tensors. Indices starting with a 9 label physical indices, with a 7 those of the ket and with an 8 those of the bra

    #     71  72
    #   A---A---B---
    # 91| 92|   |
    #   B---A---A---
    #     81  82

    sites = length(tensors_A);
    tensors = [tensors_A..., conj.(tensors_B)...]
    pInt(AS) = parse.(Int, AS)
    IndexArray_Ket = [pInt(["9$n", "7$(n-1)", "7$n"]) for n ∈ 2:sites-1]
    prepend!(IndexArray_Ket, [[91,71]]); #first site 
    append!(IndexArray_Ket, [pInt(["7$(sites-1)", "9$sites"])]); #first site 

    IndexArray_Bra = [pInt(["9$n", "8$(n-1)", "8$n"]) for n ∈ 2:sites-1]
    prepend!(IndexArray_Bra, [[91,81]]); #first site 
    append!(IndexArray_Bra, [pInt(["8$(sites-1)", "9$sites"])]); #first site 
    
    IndexArray = [IndexArray_Ket..., IndexArray_Bra...]
    order_indices = vcat([pInt(["9$n", "7$n", "8$n"]) for n ∈ 1:sites-1]...)
    push!(order_indices, parse(Int,"9$sites"))
    

    fid = ncon(tensors, IndexArray,con_order=order_indices);
    return abs(fid)
end;


function generate_indices(L, sites_out)
    pInt(AS) = parse.(Int, AS)
    indices_ket = [];
    indices_bra = [];
    1 ∈ sites_out ? (push!(indices_ket, [91,71]);) : (push!(indices_ket, [-1,71]);)
    1 ∈ sites_out ? (push!(indices_bra, [91,81]);) : (push!(indices_bra, [-2,81]);)
    1 ∈ sites_out ? (nc = 0;) : (nc = -2;)
    for n in 2:L-1
        if n ∈ sites_out
            push!(indices_ket, pInt(["9$n", "7$(n-1)", "7$n"]))
            push!(indices_bra, pInt(["9$n", "8$(n-1)", "8$n"]))
        else
            push!(indices_ket, pInt(["$(nc-1)", "7$(n-1)", "7$n"]))
            push!(indices_bra, pInt(["$(nc-2)", "8$(n-1)", "8$n"]))
            nc-=2;
        end
    end
    L ∈ sites_out ? (push!(indices_ket,pInt(["7$(L-1)", "9$L"]));) : (push!(indices_ket, pInt(["7$(L-1)", "$(nc-1)"]));)
    L ∈ sites_out ? (push!(indices_bra,pInt(["8$(L-1)", "9$L"]));) : (push!(indices_bra, pInt(["8$(L-1)", "$(nc-2)"]));)

    return vcat(indices_bra, indices_ket)
end

function generate_indices(L; edge="none")
    pInt(AS) = parse.(Int, AS)
    indices_ket = [];
    indices_bra = [];
    for n in 1:L
        push!(indices_ket, pInt(["$(-n)", "7$(n-1)", "7$n"]))
        push!(indices_bra, pInt(["$(-n-L)", "8$(n-1)", "8$n"]))
    end
    if edge=="left"
        indices_ket[1] = pInt(["$(-1)", "71"]) 
        indices_bra[1] = pInt(["$(-1-L)", "81"]) 
    elseif edge=="right"
        indices_ket[L] = pInt(["$(-L)", "7$(L-1)"]) 
        indices_bra[L] = pInt(["$(-2L)", "8$(L-1)"])
    end
    return vcat(indices_bra, indices_ket)
end

function density_matrix(tensors_A, sites_out) #* if the MPS have been transposed during saving to HDF5 correct by setting tranpose = true
    
    #* Contract tensors. Indices starting with a 9 label physical indices, with a 7 those of the ket and with an 8 those of the bra

    #     71  72
    #   A---A---B---
    # 91| 92|   |
    #   B---A---A---
    #     81  82

    L = length(tensors_A);
    Lin = L - length(sites_out);
    tensors = [tensors_A..., conj.(tensors_A)...]
    pInt(AS) = parse.(Int, AS)
    index_con =  generate_indices(L, sites_out);
    
    order_indices=[];
    for n in 1:L-1
        if n ∈ sites_out
            push!(order_indices, pInt(["9$n","7$n", "8$n"])...)
        else
            push!(order_indices, pInt(["7$n", "8$n"])...)
        end
    end
    L ∈ sites_out && push!(order_indices, parse(Int,"9$L"))

    ρ = TensorOperations.ncon(tensors, index_con, order=reverse(order_indices));
    #ρ = ncon(tensors, index_con);
    #return index_con, order_indices
    return reshape(ρ,(2^Lin,2^Lin))
end;


function reconstructcomplexStatev1(mps; transpose=false) #* if the MPS have been transposed during saving to HDF5 correct by setting tranpose = true

    #* basis of Hilbert space
    basis(n)= Vector{Bool}(digits(n, base=2, pad=N))
    basis(n,D)= Vector{Bool}(digits(n, base=2, pad=D))

    #* Sort A matrices by lattice site. Apparently the sites are labelled from right to left
    sites = length(mps) ÷ 4;
    keys_mps = collect(keys(mps));
    index_key(A_key) = findfirst(x-> x==A_key, keys_mps)

    Azero = [];     
    for site ∈ 0:sites-1
        Akey = string(site)*"_0_((),())"
        AiRe = index_key(Akey*"Re");
        AiIm = index_key(Akey*"Im");
        Acomplex = im*mps[keys_mps[AiIm]] + mps[keys_mps[AiRe]]; #Re[A] + i Im[A]
        transpose == false ? push!(Azero, Acomplex) : push!(Azero, Acomplex');
    end

    Aone = [];     
    for site ∈ 0:sites-1
        Akey = string(site)*"_1_((),())"
        AiRe = index_key(Akey*"Re");
        AiIm = index_key(Akey*"Im");
        Acomplex = im*mps[keys_mps[AiIm]] + mps[keys_mps[AiRe]]; #Re[A] + i Im[A]
        transpose == false ? push!(Aone, Acomplex) : push!(Aone, Acomplex');
    end

    if transpose == false
        Aone = reverse(Aone)
        Azero = reverse(Azero)
    end
    
    #* Rebuild state
    state = zeros(ComplexF64, 2^sites);
    Threads.@threads for n ∈ 0:2^sites-1    
        tensors = Matrix{ComplexF64}[];
        basis_n=basis(n, sites);
        for σ ∈ 1:sites
            basis_n[σ] == 0 ? push!(tensors, Azero[σ]) : push!(tensors, Aone[σ]);
        end
        state[n+1] = prod(tensors)[1] #Calculates the coefficient of the basis state n
    end 
    return state
end;


function reconstructcomplexStatev2(mps) #* if the MPS have been transposed during saving to HDF5 correct by setting tranpose = true

    #* basis of Hilbert space
    basis(n)= Vector{Bool}(digits(n, base=2, pad=N))
    basis(n,D)= Vector{Bool}(digits(n, base=2, pad=D))

    #* Sort A matrices by lattice site. Sites are labelled from right to left in exported states
    sites = length(mps) ÷ 4;
    keys_mps = collect(keys(mps));
    index_key(A_key) = findfirst(x-> x==A_key, keys_mps)

    Aone = []; Azero = [];
    for site ∈ 0:sites-1
        Akey_1 = string(site)*"_1_((),())"
        Akey_0 = string(site)*"_0_((),())"
        AiRe_0 = index_key(Akey_0*"Re");
        AiIm_0 = index_key(Akey_0*"Im");
        AiRe_1 = index_key(Akey_1*"Re");
        AiIm_1 = index_key(Akey_1*"Im");
        Acomplex_0 = im*mps[keys_mps[AiIm_0]] + mps[keys_mps[AiRe_0]]; #Re[A] + i Im[A]
        Acomplex_1 = im*mps[keys_mps[AiIm_1]] + mps[keys_mps[AiRe_1]]; #Re[A] + i Im[A]
        push!(Aone, Acomplex_1);
        push!(Azero, Acomplex_0);
    end

    #* Reverses order of tensors to match left to right site labelling
    Aone = reverse(Aone)
    Azero = reverse(Azero)
    
    #* Rebuild state
    state = zeros(ComplexF64, 2^sites);
    Threads.@threads for n ∈ 0:2^sites-1    
        mps_contr = 1;
        basis_n = basis(n, sites);
        for σ ∈ 1:sites
            basis_n[σ] == 0 ?  (mps_contr *=  Azero[σ]) :  (mps_contr *=  Aone[σ]);
        end
        state[n+1] = mps_contr[1] #Calculates the coefficient of the basis state n
    end
    return state
end;

