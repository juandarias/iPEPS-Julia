##########################################################
# General methods applicable to tensors and its networks #
##########################################################

function cast_tensor!(R::ReducedTensor, S::SimpleUpdateTensor; renormalize::Bool = false)

    sqrtW =  [sqrt.(S.weights[n]) for n ∈ eachindex(S.weights)];

    @tensor L[u, r, d, l, p] := diagm(sqrtW[1])[u, α] * S.S[α, r, d, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[2])[r, α] * L[u, α, d, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[3])[d, α] * L[u, r, α, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[4])[l, α] * L[u, r, d, α, p]

    @tensor LLdag[uk, ub, rk, rb, dk, db, lk, lb] := L[uk, rk, dk, lk, α] * conj(L)[ub, rb, db, lb, α]

    renormalize == false && (R.R = reshape(LLdag, (size(S.S)[1:4]).^2));
    renormalize == true && (R.R = normalize(reshape(LLdag, (size(S.S)[1:4]).^2)));
    R.D = (S.D).^2;
end


function cast_tensor(::Type{ReducedTensor}, S::SimpleUpdateTensor; renormalize::Bool = false)


    sqrtW =  [sqrt.(S.weights[n]) for n ∈ eachindex(S.weights)];

    @tensor L[u, r, d, l, p] := diagm(sqrtW[1])[u, α] * S.S[α, r, d, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[2])[r, α] * L[u, α, d, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[3])[d, α] * L[u, r, α, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[4])[l, α] * L[u, r, d, α, p]

    @tensor LLdag[uk, ub, rk, rb, dk, db, lk, lb] := L[uk, rk, dk, lk, α] * conj(L)[ub, rb, db, lb, α]

    R = reshape(LLdag, (size(S.S)[1:4]).^2);
    renormalize == true && (R = normalize(R);)

    return ReducedTensor(R, S.symmetry)
end

function cast_tensor(::Type{ReducedTensor}, A::Tensor; renormalize::Bool = false)


    @tensor LLdag[uk, ub, rk, rb, dk, db, lk, lb] := A.A[uk, rk, dk, lk, α] * conj(A.A)[ub, rb, db, lb, α]

    R = reshape(LLdag, (size(A.A)[1:4]).^2);
    renormalize == true && (R = normalize(R);)

    return ReducedTensor(R, A.symmetry)
end


function cast_tensor!(A::Tensor, S::SimpleUpdateTensor)

    sqrtW =  [sqrt.(S.weights[n]) for n ∈ eachindex(S.weights)];

    @tensor A.A[u, r, d, l, p] = diagm(sqrtW[1])[u, α] * S.S[α, r, d, l, p]
    @tensor A.A[u, r, d, l, p] = diagm(sqrtW[2])[r, α] * A.A[u, α, d, l, p]
    @tensor A.A[u, r, d, l, p] = diagm(sqrtW[3])[d, α] * A.A[u, r, α, l, p]
    @tensor A.A[u, r, d, l, p] = diagm(sqrtW[4])[l, α] * A.A[u, r, d, α, p]

    A.D = collect(size(S.S)[1:4]);
    A.d = S.d;
end


function cast_tensor(::Type{Tensor}, S::SimpleUpdateTensor)

    sqrtW =  [sqrt.(S.weights[n]) for n ∈ eachindex(S.weights)];

    @tensor X[u, r, d, l, p] := diagm(sqrtW[1])[u, α] * S.S[α, r, d, l, p]
    @tensor X[u, r, d, l, p] := diagm(sqrtW[2])[r, α] * X[u, α, d, l, p]
    @tensor X[u, r, d, l, p] := diagm(sqrtW[3])[d, α] * X[u, r, α, l, p]
    @tensor X[u, r, d, l, p] := diagm(sqrtW[4])[l, α] * X[u, r, d, α, p]

    return Tensor(X, S.symmetry)
end


function symmetrize(C::Array{T,2}; renormalize::Bool = true) where {T}
    C = C + adjoint(C);
    renormalize == true && return C/norm(C)
    renormalize == false && return C
end

function symmetrize(T::Array{X,3}; renormalize::Bool = true) where {X}
    T = T + permutedims(T, (2, 1, 3));
    #T = T + conj(permutedims(T, (2, 1, 3)));
    renormalize == true && return T/norm(T)
    renormalize == false && return T
end

function symmetrize(S::Array{T,5}; renormalize::Bool = true, hermitian::Bool = true, symmetry::LatticeSymmetry = XY) where {T}

    if symmetry == XY
        S = S + permutedims(S, (3, 2, 1, 4, 5)) + permutedims(S, (1, 4, 3, 2, 5)) + permutedims(S, (3, 4, 1, 2, 5));
    elseif symmetry == R4
        Ssym = zero(S);
        for p ∈ permutations([1,2,3,4])
            Ssym += permutedims(S, [p; 5])
        end
        S = Ssym;
    end
    hermitian == true && (S = S + conj(S));
    renormalize == true && return S/norm(S)
    renormalize == false && return S
end

function tensor_svd(A::Array{T}, indices_partition::Vector{Vector{Int64}}; Χ::Int64=0) where {T<:Union{ComplexF64, Float64}}
    rA = reshape(A, (prod(size(A)[indices_partition[1]]), prod(size(A)[indices_partition[2]])));

    fA = svd(rA);
    (Χ > length(fA.S) || Χ == 0) && (Χ = length(fA.S);)

    U = fA.U[:, 1:Χ];
    S = fA.S[1:Χ];
    Vt = fA.Vt[1:Χ, :];

    U = reshape(U, (size(A)[indices_partition[1]]..., size(U, 2)));
    Vt = reshape(Vt, (size(Vt, 1), size(A)[indices_partition[2]]...));

    return U, S, Vt
end
