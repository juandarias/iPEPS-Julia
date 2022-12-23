function prepare_su_tensor(S::SimpleUpdateTensor, gate_direction::Direction)

    gate_direction == UP ? (n = 3) : (gate_direction == RIGHT ?
    (n = 2) : (gate_direction == DOWN ? (n = 1) : (n = 0)));
    order = [circshift([1, 2, 3, 4], n)..., 5];

    S.S = permutedims(S.S, order); # permute legs such that aux. leg towards gate is before last one
    weights = diagm.(S.weights[order[1:3]]);

    # Absorb weights
    @tensor M[1, 2, 3, g, p] := S.S[α, β, γ, g, p] * weights[1][α, 1] *
        weights[2][β, 2] * weights[3][γ, 3];


    # Factorize
    M = reshape(M, (S.D^3, S.D * S.d));
    Q, R = qr(M);
    R = reshape(Matrix(R), (:, S.D, S.d));
    Q = reshape(Matrix(Q), (S.D, S.D, S.D, :));

    return Q, R
end

function restore_su_tensor(Q::Array{X, 4}, R::Array{T,3}, weights::Vector{Vector{Float64}}, gate_direction::Direction) where {T,X}

    gate_direction == UP ? (n = 3) : (gate_direction == RIGHT ?
    (n = 2) : (gate_direction == DOWN ? (n = 1) : (n = 0)));
    order = circshift([1, 2, 3, 4], n);

    #x⁻¹(x) = x.^(-1)
    inv_weights = [diagm(weights[order[n]].^-1) for n ∈ 1:3]

    @tensor S[1, 2, 3, 4, p] := Q[α, β, γ, δ] * inv_weights[1][α, 1] * inv_weights[2][β, 2] *
    inv_weights[3][γ, 3] * R[δ, p, 4];


    # Restore order of legs
    order = [circshift([1, 2, 3, 4], -n)..., 5];
    S = permutedims(S, order);
    return S
end

function apply_gate(G::Array{S, 4}, RA::Array{T, 3}, RB::Array{T, 3}, W::Array{Float64,2}, Dmax::Int64) where {T,S}
    @tensor RRGw[l, u1, u2, r] := RA[l, γ, α] * G[u1, u2, α, β] * W[γ, δ] * RB[r, δ, β];

    # Split and truncate
    RRGw = reshape(RRGw, (prod(size(RRGw)[1:2]), :));
    RAg, Wg, RBg = svd(RRGw);
    R̃A = reshape(RAg[:, 1:Dmax], (size(RA, 1), 2, Dmax)); #! (l,d,Dmax)
    R̃B = reshape(RBg[:, 1:Dmax], (2, size(RB, 1), Dmax)); #! (d,r,Dmax)
    W̃ = Wg[1:Dmax];

    return R̃A, R̃B, W̃
end

function update_cell!(unitcell::UnitCell, S::Array{T, 5}, weights::Vector{Float64}, label::Char, direction::Direction) where {T}
    coords = findall(t -> t == label, unitcell.pattern);
    for coord in coords
        unitcell.S[coord].S = S/norm(S); #! normalize
        unitcell.S[coord].weights[Int(direction)] = weights/sqrt(sum(weights.^2));
    end

end
