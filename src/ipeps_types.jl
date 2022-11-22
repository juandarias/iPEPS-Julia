#########
# Types #
#########

#abstract type LatticeSymmetry end
#struct C4 <: LatticeSymmetry end
#struct XY <: LatticeSymmetry end

@enum LatticeSymmetry UNDEF C4=1 XY

@warn "LatticeSymmetry type not properly implemented in methods"

"""
    mutable struct Environment{T}

## Arguments

## Notes
    The indices of the environment bonds are labelled in ascending order clockwise, with the first index at 12 o'clock.
    The index corresponding to auxiliary bonds are always the last ones.
"""
struct Environment{U}
    loc::Tuple #? Shall it accept multiple locations, e.g. for environment of multiple sites
    "Corner tensors"
    C::Vector{Array{U,2}}
    "Column/row tensors. The last index correspond to the system auxiliary space"
    T::Vector{Array{U,3}}
    Χ::Int64
    "Corner matrices spectrum"
    spectra::Vector{Vector{Float64}}

    function Environment(C::Vector{Array{U,2}}, T::Vector{Array{U,3}}, loc::Tuple) where {U}
        Χ = size(T[1], 1);
        spectra = fill(zeros(Χ), 4);
        new{U}(loc, copy(C), copy(T), Χ, spectra)
    end
end

function Environment{U}(Χ::Int64, D::Int64, loc::Tuple) where {U}
    Ci = rand(U, Χ, Χ);
    Ci = normalize(Ci); #! Note Ci is not symmetrical
    Ti = rand(U, Χ, Χ, D^2);
    Ti = normalize(Ti);
    Environment([Ci, Ci, Ci, collect(transpose(Ci))], fill(Ti, 4), loc)
end


#= Implemented for the simplest case i.e. a translational and rotational invariant state
    with a 1x1 unitcell =#

mutable struct Tensor{T}
    "The last dimension corresponds to the physical space. Auxiliary indices
    are labelled clockwise"
    A::Array{T,5}
    D::Vector{Int64}
    d::Int64
    symmetry::LatticeSymmetry

    Tensor{T}() where {T} = new{T}();

    function Tensor(Ai::Array{T,5}, symmetry::X = UNDEF) where {T, X<:LatticeSymmetry}
        D = collect(size(Ai)[1:4]);
        d = size(Ai, 4);
        new{T}(Ai, D, d, symmetry);
    end

end

function Tensor{T}(D::Int64, symmetry::X = UNDEF) where {T, X<:LatticeSymmetry}
    d = 2;
    A = rand(T, [D, D, D, D, d]...);
    A = symmetrize(A; symmetry = symmetry);
    Tensor(A, symmetry);
end

mutable struct SimpleUpdateTensor{T}
    "The last dimension corresponds to the physical space"
    S::Array{T,5}
    D::Int64
    d::Int64
    "Weights (λᵢ) on auxiliary bonds, e.g. a simple update environment, such that ∑λᵢ = 1"
    #weights::Vector{Float64}
    weights::Vector{Vector{Float64}}
    symmetry::LatticeSymmetry

    SimpleUpdateTensor{T}() where {T} = new{T}();

    function SimpleUpdateTensor(
        Si::Array{T,5},
        weights::Vector{Vector{Float64}},
        symmetry::LatticeSymmetry = UNDEF) where {T}
        D = size(Si, 1);
        d = size(Si, 5);
        new{T}(Si, D, d, weights, symmetry);
    end
end

function SimpleUpdateTensor{T}(D::Int64, symmetry::X = UNDEF) where {T, X<:LatticeSymmetry}
    d = 2;

    #= Generate symmetric tensor =#
    A = rand(T, [D, D, D, D, d]...);
    A = symmetrize(A; symmetry = symmetry);

    #= Generate weights =#
    if symmetry == UNDEF
        λi = [rand(Float64, D) for _ ∈ 1:4];
        λs = [λi[n]/sum(λi[n]) for n ∈ 1:4];
    elseif symmetry == XY
        λi = [rand(Float64, D) for _ ∈ 1:2];
        λi = [λi[n]/sum(λi[n]) for n ∈ 1:2];
        λs = [λi; λi]
    elseif symmetry == C4
        λi = rand(Float64, D);
        λi = λi/sum(λi);
        λs = fill(λi, 4);
    end

    invsqrtλ =  [diagm(λs[n].^(-1/2)) for n ∈ 1:4];

    @tensor Si[u, r, d, l, p] := A[α, β, γ, δ, p] * invsqrtλ[1][α, u] * invsqrtλ[2][β, r] *
    invsqrtλ[3][γ, d] *  invsqrtλ[4][δ, l];

    SimpleUpdateTensor(Si, λs, symmetry);
end

#function SimpleUpdateTensor(Si::Array{T,5}, symmetry::X = UNDEF) where {T, X<:LatticeSymmetry}
#    λi = rand(Float64, size(Si, 1));
#    λi = λi/sum(λi);
#    SimpleUpdateTensor(Si, fill(λi, 4), symmetry);
#end

mutable struct ReducedTensor{T}
    R::Array{T,4}
    D::Int64
    E::Environment{T}
    symmetry::LatticeSymmetry

    ReducedTensor{T}() where {T} = new{T}();

    function ReducedTensor(R::Array{T,4}) where {T}
        D = size(R, 1);
        new{T}(R, D);
    end
end


#= Considering only single and two-site ops =#
mutable struct Operator{T}
    O::Array{T};
    loc::Vector{CartesianIndex{2}};
    nsites::Int64;
    name::String

    Operator(O::Array{T}, loc::Vector{CartesianIndex{2}}) where {T} = new{T}(O, loc, length(loc))

    Operator(O::Array{T}, loc::Vector{NTuple{2, Int64}}) where {T} = new{T}(O, CartesianIndex.(loc), length(loc))

end

mutable struct UnitCell{T}
    D::Int64
    dims::Tuple
    pattern::Array{Char}
    symmetry::LatticeSymmetry
    R::Array{ReducedTensor{T}}
    S::Array{SimpleUpdateTensor{T}}
    E::Array{Environment{T}}

    UnitCell{T}() where {T} = new{T}();


    function UnitCell(R::ReducedTensor{T}) where {T}
        D = sqrt(R.D);
        pattern = ['a'];
        new{T}(D, (1, 1), pattern, R.symmetry, [R])
    end

    function UnitCell(S::SimpleUpdateTensor{T}) where {T}
        D = S.D;
        pattern = ['a'];
        R = cast_tensor(ReducedTensor, S);
        new{T}(D, (1, 1), pattern, S.symmetry, [R], [S])
    end

    function UnitCell(
        D::Int64,
        dims::Tuple,
        pattern::Array{Char},
        symmetry::LatticeSymmetry,
        R_cell::Array{ReducedTensor{T}},
        S_cell::Array{SimpleUpdateTensor{T}}) where {T}

        new{T}(D, dims, pattern, symmetry, copy(R_cell), copy(S_cell))

    end

end

function UnitCell{T}(D::Int64, dims::Tuple, pattern::Array{Char, 2}, symmetry::LatticeSymmetry = XY) where {T}

    S_cell = Array{SimpleUpdateTensor{T}, 2}(undef,  size(pattern));
    R_cell = Array{ReducedTensor{T}, 2}(undef,  size(pattern));
    Ni = size(pattern, 1);
    Nj = size(pattern, 2);

    #= Creates minimal unit cell =#
    unique_tensors = unique(pattern);

    if length(unique_tensors) != length(pattern) # for unit-cell with repeating tensors
        for type_tensor ∈ unique_tensors
            Si = SimpleUpdateTensor{T}(D, symmetry);
            coords = findall(t -> t == type_tensor, pattern);
            for coord in coords
                S_cell[coord] = Si;
                R_cell[coord] = cast_tensor(ReducedTensor, Si);
            end
        end
    else # if minimal unit-cell consists of no repeating tensors
        for i ∈ 1:Ni, j ∈ 1:Nj
            S_cell[i,j] = SimpleUpdateTensor{T}(D, symmetry);
            R_cell[i,j] = cast_tensor(ReducedTensor, S_cell[i,j]);
        end
    end

    #= Expands minimal unit cell to final unit cell if needed =#
    if Ni == dims[1] && Nj == dims[2]
        UnitCell(D, dims, pattern, symmetry, R_cell, S_cell)
    elseif Ni < dims[1] || Nj < dims[2]
        pattern_large = Array{Char, 2}(undef, dims);
        R_cell_large = Array{ReducedTensor{T}, 2}(undef, dims);
        S_cell_large = Array{SimpleUpdateTensor{T}, 2}(undef, dims);
        for i ∈ 1:dims[1], j ∈ 1:dims[2]
            pattern_large[i, j] = pattern[mod(i - 1, Ni) + 1, mod(j - 1, Nj) + 1]
            S_cell_large[i, j] = S_cell[mod(i - 1, Ni) + 1, mod(j - 1, Nj) + 1];
            R_cell_large[i, j] = R_cell[mod(i - 1, Ni) + 1, mod(j - 1, Nj) + 1];
        end

        UnitCell(D, dims, pattern_large, symmetry, R_cell_large, S_cell_large)
    end
end

function UnitCell{T}(D::Int64, dims::Tuple, pattern::Array{Char, 1}, symmetry::LatticeSymmetry = XY) where {T}


    Si = SimpleUpdateTensor{T}(D, symmetry);
    Ri = cast_tensor(ReducedTensor, Si);

    pattern_large = Array{Char, 2}(undef, dims);
    R_cell_large = Array{ReducedTensor{T}, 2}(undef, dims);
    S_cell_large = Array{SimpleUpdateTensor{T}, 2}(undef, dims);

    for i ∈ 1:dims[1], j ∈ 1:dims[2]
        pattern_large[i, j] = pattern[1]
        S_cell_large[i, j] = Si;
        R_cell_large[i, j] = Ri;
    end

    UnitCell(D, dims, pattern_large, symmetry, R_cell_large, S_cell_large)

end

function UnitCell(pattern::Array{Char},
    symmetry::LatticeSymmetry,
    R_cell::Array{ReducedTensor{T}},
    S_cell::Array{SimpleUpdateTensor{T}}) where {T}

    dims = size(S_cell);
    D = size(S_cell[1,1].S, 1);

    UnitCell(D, dims, pattern, symmetry, R_cell, S_cell);
end

#=
#= Builds an unitcell of dimension `dims` from a smaller pattern of SU tensors =#
function UnitCell(S::Array{SimpleUpdateTensor{T}}, dims::Tuple, pattern::Array{Char, 2}) where {T}

    D = S[1].D;
    full_pattern = Array{Char, 2}(undef, dims);
    R_cell_large = Array{ReducedTensor{T}, 2}(undef, dims);
    full_cell_simple = Array{SimpleUpdateTensor{T}, 2}(undef, dims);
    Ni = size(pattern, 1);
    Nj = size(pattern, 2);

    if Ni < dims[1] || Nj < dims[2]
        for i ∈ 1:dims[1], j ∈ 1:dims[2]
            full_pattern[i, j] = pattern[mod(i - 1, Ni) + 1, mod(j - 1, Nj) + 1]
            full_cell_reduced[i, j] = cast_tensor(ReducedTensor, S[mod(i - 1, Ni) + 1, mod(j - 1, Nj) + 1]);
            full_cell_simple[i, j] = S[mod(i - 1, Ni) + 1, mod(j - 1, Nj) + 1]/norm(full_cell_reduced[i, j].R);
        end
    else
        for i ∈ 1:Ni, j ∈ 1:Nj
            full_cell_reduced[i, j] = cast_tensor(ReducedTensor, S[i, j]);
        end
        full_cell_simple = S;
        full_pattern = pattern;
    end

    new{T}(D, dims, full_pattern, full_cell_reduced, full_cell_simple)
end

#= Builds an unitcell of dimension `dims` from a smaller pattern of reduced tensors =#
function UnitCell(R::Array{ReducedTensor{T}}, dims::Tuple, pattern::Array{Char, 2}) where {T}

    D = R[1].D;
    full_pattern = Array{Char, 2}(undef, dims)
    full_cell = Array{ReducedTensor{T}, 2}(undef, dims)
    Ni = size(pattern, 1);
    Nj = size(pattern, 2);

    if Ni < dims[1] || Nj < dims[2]
        for i ∈ 1:dims[1], j ∈ 1:dims[2]
            full_pattern[i, j] = pattern[mod(i - 1, Ni) + 1, mod(j - 1, Nj) + 1]
            full_cell[i, j] = R[mod(i - 1, Ni) + 1, mod(j - 1, Nj) + 1]
        end
    else
        full_cell = R;
        full_pattern = pattern;
    end

    new{T}(D, dims, full_pattern, full_cell)
end =#

#= Builds an unitcell of dimension `dims` by specifying a pattern and a bond dimension.
Tensors will be initialized randomly. =#
#=
function UnitCell{T}(D::Int64, dims::Tuple, pattern::Array{Char,2}, symmetry::LatticeSymmetry = XY) where {T}

    # Initialize tensors for pattern
    Ni = size(pattern, 1);
    Nj = size(pattern, 2);
    minimal_cell = Array{SimpleUpdateTensor{T}, 2}(undef,  size(pattern));
    unique_tensors = unique(pattern);

    if length(unique_tensors) != length(pattern) # for unit-cell with repeating tensors
        for n ∈ eachindex(unique_tensors)
            Si = SimpleUpdateTensor{T}(D, symmetry);
            coords = findall(t -> t == unique_tensors[n], pattern);
            for coord in coords
                minimal_cell[coord] = Si;
            end
        end

    else # if minimal unit-cell consists of no repeating tensors
        for i ∈ 1:Ni, j ∈ 1:Nj
            minimal_cell[i,j] = SimpleUpdateTensor{T}(D, symmetry);
        end
    end

    # Build unit cell from pattern
    UnitCell(minimal_cell, dims, pattern);
end
=#
