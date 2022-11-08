module operators_basis

    using LinearAlgebra, SparseArrays

    export σˣᵢσˣⱼ, σʸᵢσʸⱼ, σᶻᵢσᶻⱼ, σᶻᵢ, σˣᵢ, σʸᵢ, SˣᵢSˣⱼ, SʸᵢSʸⱼ, SᶻᵢSᶻⱼ, Sᶻᵢ, Sˣᵢ, Sʸᵢ
    export generate_basis, Z₂basis, generate_bit_basis

    #* Operators. Convention bigO = Oₙ ⊗ ... O₂ ⊗ O₁

    function σˣᵢσˣⱼ(i,j,N)
        σˣ = sparse([0 1; 1 0]);
        II = sparse([1 0; 0 1]);
        i==1 || j==1 ? (op = σˣ) : (op = II)
        for n=2:N
            n==i || n==j ? (op = kron(σˣ,op)) : (op = kron(II, op))
        end
        return op
    end
    
    function σʸᵢσʸⱼ(i,j,N)
        σʸ = sparse([0 -im; im 0]);
        II = sparse([1 0; 0 1]);
        i==1 || j==1 ? (op = σʸ) : (op = II)
        for n=2:N
            n==i || n==j ? (op = kron(σʸ,op)) : (op = kron(II, op))
        end
        return op
    end
    
    function σᶻᵢσᶻⱼ(i,j,N)
        σᶻ = sparse([1 0; 0 -1]);
        II = sparse([1 0; 0 1]);
        i==1 || j==1 ? (op = σᶻ) : (op = II)
        for n=2:N
            n==i || n==j ? (op = kron(σᶻ,op)) : (op = kron(II, op))
        end
        return op
    end

    function σᶻᵢ(i,N)
        σᶻ = sparse([1 0; 0 -1]);
        II = sparse([1 0; 0 1]);
        return kron([n==i ? σᶻ : II for n in N:-1:1]...)
    end

    function σˣᵢ(i,N)
        σˣ = sparse([0 1; 1 0]);
        II = sparse([1 0; 0 1]);
        return kron([n==i ? σˣ : II for n in N:-1:1]...)
    end

    function σʸᵢ(i,N)
        σʸ = sparse([0 -im; im 0]);
        II = sparse([1 0; 0 1]);
        return kron([n==i ? σʸ : II for n in N:-1:1]...)
    end

    Sˣᵢ(i,N) = 0.5*σˣᵢ(i,N);
    Sʸᵢ(i,N) = 0.5*σʸᵢ(i,N);
    Sᶻᵢ(i,N) = 0.5*σᶻᵢ(i,N);

    SᶻᵢSᶻⱼ(i,j,N) = (1/4)*σᶻᵢσᶻⱼ(i,j,N);
    SʸᵢSʸⱼ(i,j,N) = (1/4)*σʸᵢσʸⱼ(i,j,N);
    SˣᵢSˣⱼ(i,j,N) = (1/4)*σˣᵢσˣⱼ(i,j,N);


    #* Basis
    """
    Binary `BitArray` representation of the given integer `num`, padded to length `N`.
    """
    bit_rep(num::Integer, N::Integer) = Vector{Bool}(digits(num, base=2, pad=N))
    
    state_number(state::BitArray) = parse(Int,join(Int64.(state)),base=2)

    """
        generate_bit_basis(N::Integer) -> basis

    Generates a basis (`Vector{BitArray}`) spanning the Hilbert space of `N` spins.
    """
    function generate_bit_basis(N::Integer)
        nstates = 2^N
        basis = Vector{BitArray{1}}(undef, nstates)
        for i in 0:nstates-1
            basis[i+1] = bit_rep(i, N)
        end
        return basis
    end

    function generate_basis(N::Integer)
        ↑ = [1, 0];
        ↓ = [0, 1];
        basis = [↑, ↓];
        for i in 2:N
            basis_N = Vector{Vector{Int8}}();
            for n ∈ 1:length(basis)
                push!(basis_N, kron(↑, basis[n]));
                push!(basis_N, kron(↓, basis[n]));
            end
            basis = basis_N
        end
        return basis
    end


    """
    Gives two basis of the Hilbert space for the two symmetry sectors of a Ζ₂ symmetric Hamiltonian with z as quantization axis
    """
    function Z₂basis(full_basis)
        lowsector_basis = Int64[];
        highsector_basis = Int64[];
        size_basis = length(full_basis);
        num_sites = Int(log2(size_basis));
        for n in 1:size_basis÷2
            state = full_basis[n]
            state_hw = sum(state) #hadamard weight of state
            neg_state_number = state_number(reverse(.!state))+1
            neg_state_hw = sum(bit_rep(state_number(reverse(.!full_basis[n])),num_sites)) #hadamard weight of state
            state_hw < neg_state_hw ? (push!(lowsector_basis, n); push!(highsector_basis, neg_state_number)) : (push!(lowsector_basis, neg_state_number); push!(highsector_basis, n))
        end
        return lowsector_basis, highsector_basis
    end
end


"==="
# Old
"==="


function σᶻᵢσᶻⱼold(i,j,N)
    σᶻ = sparse([1 0; 0 -1]);
    II = sparse([1 0; 0 1]);
    return kron([n==i || n==j ? σᶻ : II for n in 1:N]...)
end