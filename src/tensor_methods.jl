##########################################################
# General methods applicable to tensors and its networks #
##########################################################

function cast_tensor!(R::ReducedTensor, S::SimpleUpdateTensor)

    sqrtW =  [sqrt.(S.weights[n]) for n ∈ eachindex(S.weights)];

    @tensor L[u, r, d, l, p] := diagm(sqrtW[1])[u, α] * S.S[α, r, d, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[2])[r, α] * L[u, α, d, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[3])[d, α] * L[u, r, α, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[4])[l, α] * L[u, r, d, α, p]

    @tensor LLdag[uk, ub, rk, rb, dk, db, lk, lb] := L[uk, rk, dk, lk, α] * conj(L)[ub, rb, db, lb, α]

    R.R = reshape(LLdag, (S.D^2, S.D^2, S.D^2, S.D^2));
    R.D = S.D^2;
end


function cast_tensor(::Type{ReducedTensor}, S::SimpleUpdateTensor{T}) where {T}

    R = ReducedTensor{T}();

    sqrtW =  [sqrt.(S.weights[n]) for n ∈ eachindex(S.weights)];

    @tensor L[u, r, d, l, p] := diagm(sqrtW[1])[u, α] * S.S[α, r, d, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[2])[r, α] * L[u, α, d, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[3])[d, α] * L[u, r, α, l, p]
    @tensor L[u, r, d, l, p] := diagm(sqrtW[4])[l, α] * L[u, r, d, α, p]

    @tensor LLdag[uk, ub, rk, rb, dk, db, lk, lb] := L[uk, rk, dk, lk, α] * conj(L)[ub, rb, db, lb, α]

    R.R = reshape(LLdag, (size(S.S, 1)^2, size(S.S, 2)^2, size(S.S, 3)^2, size(S.S, 4)^2));
    R.D = S.D^2;
    R.symmetry = S.symmetry;
    return R
end

function cast_tensor!(A::Tensor, S::SimpleUpdateTensor)

    sqrtW =  [sqrt.(S.weights[n]) for n ∈ eachindex(S.weights)];

    @tensor A.A[u, r, d, l, p] = diagm(sqrtW[1])[u, α] * S.S[α, r, d, l, p]
    @tensor A.A[u, r, d, l, p] = diagm(sqrtW[2])[r, α] * A.A[u, α, d, l, p]
    @tensor A.A[u, r, d, l, p] = diagm(sqrtW[3])[d, α] * A.A[u, r, α, l, p]
    @tensor A.A[u, r, d, l, p] = diagm(sqrtW[4])[l, α] * A.A[u, r, d, α, p]

    A.D = [S.D[1], S.D[2], S.D[3], S.D[4]];
    A.d = S.d;
end


function cast_tensor(::Type{Tensor}, S::SimpleUpdateTensor{T}) where {T}

    A = Tensor{T}();
    sqrtW =  [sqrt.(S.weights[n]) for n ∈ eachindex(S.weights)];

    @tensor X[u, r, d, l, p] := diagm(sqrtW[1])[u, α] * S.S[α, r, d, l, p]
    @tensor X[u, r, d, l, p] := diagm(sqrtW[2])[r, α] * X[u, α, d, l, p]
    @tensor X[u, r, d, l, p] := diagm(sqrtW[3])[d, α] * X[u, r, α, l, p]
    @tensor X[u, r, d, l, p] := diagm(sqrtW[4])[l, α] * X[u, r, d, α, p]

    A.A = X;
    A.D = [S.D, S.D, S.D, S.D];
    A.d = S.d;
    A.symmetry = S.symmetry;

    return A
end


function symmetrize(C::Array{T,2}; normalize::Bool = true) where {T}
    C = C + adjoint(C);
    normalize == true && return C/norm(C)
    normalize == false && return C
end

function symmetrize(T::Array{X,3}; normalize::Bool = true) where {X}
    T = T + permutedims(T, (2, 1, 3));
    #T = T + conj(permutedims(T, (2, 1, 3)));
    normalize == true && return T/norm(T)
    normalize == false && return T
end

function symmetrize(S::Array{T,5}; normalize::Bool = true, hermitian::Bool = true, symmetry::LatticeSymmetry = XY) where {T}

    if symmetry == XY
        S = S + permutedims(S, (3, 2, 1, 4, 5)) + permutedims(S, (1, 4, 3, 2, 5)) + permutedims(S, (3, 4, 1, 2, 5));
    elseif symmetry == C4
        Ssym = zero(S);
        for p ∈ permutations([1,2,3,4])
            Ssym += permutedims(S, [p; 5])
        end
        S = Ssym;
    end
    hermitian == true && (S = S + conj(S));
    normalize == true && return S/norm(S)
    normalize == false && return S
end
