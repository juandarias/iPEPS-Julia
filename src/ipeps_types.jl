#########
# Types #
#########

#abstract type LatticeSymmetry end
#struct C4 <: LatticeSymmetry end
#struct XY <: LatticeSymmetry end

@enum LatticeSymmetry UNDEF R4=1 XY


"""
    mutable struct Environment{T}

## Arguments

## Notes

    Labels of environment tensors:

    C4-  -T1-  -C1
     |    |     |
    T4-        -T2
     |    |     |
    C3-  -T3-  -C2

    The indices of the environment bonds are labelled in ascending order clockwise, with the first index at 12 o'clock, e.g. the corner C1 elements are labelled C1(down, left).
    The index corresponding to auxiliary bonds are always the last ones.
"""
struct Environment{U<:Union{Float64, ComplexF64}}
    loc::CartesianIndex #? Shall it accept multiple locations, e.g. for environment of multiple sites
    "Corner tensors"
    C::Vector{Array{U,2}}
    "Column/row tensors. The last index correspond to the system auxiliary space"
    T::Vector{Array{U,4}}
    Χ::Int64
    "Corner matrices spectrum"
    spectra::Vector{Vector{Float64}}
    TS::Vector{String} #! debug

    function Environment(C::Vector{Array{U,2}}, T::Vector{Array{U,4}}, loc::CartesianIndex) where {U<:Union{Float64, ComplexF64}}
        Χ = size(T[1], 1);
        spectra = fill(ones(Χ), 4);
        TS = fill("s_", 4); #! debug
        new{U}(loc, deepcopy(C), deepcopy(T), Χ, spectra, TS)
    end
end

#= function Environment{U}(Χ::Int64, D::Int64, loc::Tuple) where {U}
    Ci = rand(U, Χ, Χ);
    Ci = normalize(Ci); #! Note Ci is not symmetrical
    Ti = rand(U, Χ, Χ, D^2);
    Ti = normalize(Ti);
    Environment([Ci, Ci, Ci, collect(transpose(Ci))], fill(Ti, 4), loc)
end =#


#= Implemented for the simplest case i.e. a translational and rotational invariant state
    with a 1x1 unitcell =#

mutable struct Tensor{T<:Union{Float64, ComplexF64}}
    "The last dimension corresponds to the physical space. Auxiliary indices
    are labelled clockwise"
    A::Array{T,5}
    D::Vector{Int64}
    d::Int64
    symmetry::LatticeSymmetry

    Tensor{T}() where {T<:Union{Float64, ComplexF64}} = new{T}();

    function Tensor(Ai::Array{T,5}, symmetry::X = UNDEF) where {T<:Union{Float64, ComplexF64}, X<:LatticeSymmetry}
        D = collect(size(Ai)[1:4]);
        d = size(Ai, 5);
        new{T}(Ai, D, d, symmetry);
    end

end

function Tensor{T}(D::Int64, symmetry::X = UNDEF) where {T<:Union{Float64, ComplexF64}, X<:LatticeSymmetry}
    d = 2;
    A = rand(T, [D, D, D, D, d]...);
    A = symmetrize(A; symmetry = symmetry);
    Tensor(A, symmetry);
end

function Tensor{T}(D::Vector{Int64}, symmetry::X = UNDEF) where {T<:Union{Float64, ComplexF64}, X<:LatticeSymmetry}
    d = 2;
    A = rand(T, [D..., d]...);
    A = symmetrize(A; symmetry = symmetry);
    Tensor(A, symmetry);
end

#Dummy struct for bra tensors
mutable struct BraTensor end

mutable struct SimpleUpdateTensor{T<:Union{Float64, ComplexF64}}
    "The last dimension corresponds to the physical space"
    S::Array{T,5}
    D::Vector{Int64}
    d::Int64
    "Weights (λᵢ) on auxiliary bonds, e.g. a simple update environment, such that ∑λᵢ = 1"
    #weights::Vector{Float64}
    weights::Vector{Vector{Float64}}
    symmetry::LatticeSymmetry

    SimpleUpdateTensor{T}() where {T<:Union{Float64, ComplexF64}} = new{T}();

    function SimpleUpdateTensor(
        Si::Array{T,5},
        weights::Vector{Vector{T}},
        symmetry::LatticeSymmetry = UNDEF) where {T<:Union{Float64, ComplexF64}}
        #D = size(Si, 1);
        D = collect(size(Si)[1:4]);
        d = size(Si, 5);
        new{T}(Si, D, d, weights, symmetry);
    end
end

function SimpleUpdateTensor{T}(D::Vector{Int64}, symmetry::X = UNDEF) where {T<:Union{Float64, ComplexF64}, X<:LatticeSymmetry}
    d = 2;

    #= Generate symmetric tensor =#
    A = rand(T, vcat(D, d)...);
    A = symmetrize(A; symmetry = symmetry);

    #= Generate weights =#
    if symmetry == UNDEF
        λi = [rand(Float64, D[n]) for n ∈ 1:4];
        λs = [λi[n]/sum(λi[n]) for n ∈ 1:4];
    elseif symmetry == XY
        λi = [rand(Float64, D[n]) for n ∈ 1:2];
        λi = [λi[n]/sum(λi[n]) for n ∈ 1:2];
        λs = [λi; λi]
    elseif symmetry == R4
        λi = rand(Float64, D[1]);
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

mutable struct ReducedTensor{T<:Union{Float64, ComplexF64}}
    R::Array{T,4}
    D::Vector{Int64} #! here D is the product of ket and bra D's
    symmetry::LatticeSymmetry
    E::Environment{T} #! this field is unused

    ReducedTensor{T}() where {T<:Union{Float64, ComplexF64}} = new{T}();

    function ReducedTensor(R::Array{T,4}, symmetry) where {T<:Union{Float64, ComplexF64}}
        D = collect(size(R));
        new{T}(R, D, symmetry);
    end
end


#= Considering only single and two-site ops =#
mutable struct Operator{T<:Union{Float64, ComplexF64}}
    O::Array{T};
    loc::Vector{CartesianIndex{2}};
    nsites::Int64;
    name::String

    Operator(O::Array{T}, loc::Vector{CartesianIndex{2}}) where {T} = new{T}(O, loc, length(loc))

    Operator(O::Array{T}, loc::Vector{NTuple{2, Int64}}) where {T} = new{T}(O, CartesianIndex.(loc), length(loc))

end

mutable struct UnitCell{T<:Union{Float64, ComplexF64}}
    D::Int64
    dims::Tuple
    pattern::Array{Char}
    symmetry::LatticeSymmetry
    R::Array{ReducedTensor{T}}
    A::Array{Tensor{T}}
    E::Array{Environment{T}}
    B::Array{Tensor{T}} #* to store tensors of bra layer required for calculations such as ⟨Ψ|O|Φ⟩

    UnitCell{T}() where {T} = new{T}();


    function UnitCell(R::ReducedTensor{T}) where {T<:Union{Float64, ComplexF64}}
        D = sqrt(R.D);
        pattern = ['a'];
        new{T}(D, (1, 1), pattern, R.symmetry, [R])
    end

    #= function UnitCell(S::SimpleUpdateTensor{T}) where {T<:Union{Float64, ComplexF64}}
        D = S.D;
        pattern = ['a'];
        R = cast_tensor(ReducedTensor, S);
        new{T}(D, (1, 1), pattern, S.symmetry, [R], [S])
    end =#

    function UnitCell(
        D::Int64,
        dims::Tuple,
        pattern::Array{Char},
        symmetry::LatticeSymmetry,
        R_cell::Array{ReducedTensor{T}}) where {T<:Union{Float64, ComplexF64}}

        new{T}(D, dims, pattern, symmetry, deepcopy(R_cell))

    end

    function UnitCell(
        D::Int64,
        dims::Tuple,
        pattern::Array{Char},
        symmetry::LatticeSymmetry,
        R_cell::Array{ReducedTensor{T}},
        A_cell::Array{Tensor{T}}) where {T<:Union{Float64, ComplexF64}}

        new{T}(D, dims, pattern, symmetry, deepcopy(R_cell), deepcopy(A_cell))

    end

end


function UnitCell{T}(D::Int64, dims::Tuple, pattern::Array{Char, 2}, symmetry::LatticeSymmetry = XY) where {T<:Union{Float64, ComplexF64}}

    A_cell = Array{Tensor{T}, 2}(undef,  size(pattern));
    R_cell = Array{ReducedTensor{T}, 2}(undef,  size(pattern));
    Ni = size(pattern, 1);
    Nj = size(pattern, 2);

    #= Creates minimal unit cell =#
    unique_tensors = unique(pattern);

    if length(unique_tensors) != length(pattern) # for unit-cell with repeating tensors
        for type_tensor ∈ unique_tensors
            Ai = Tensor{T}(D, symmetry);
            coords = findall(t -> t == type_tensor, pattern);
            for coord in coords
                A_cell[coord] = Ai;
                R_cell[coord] = cast_tensor(ReducedTensor, Ai);
            end
        end
    else # if minimal unit-cell consists of no repeating tensors
        for ij ∈ CartesianIndices(size(pattern))
            A_cell[ij] = Tensor{T}(D, symmetry);
            R_cell[ij] = cast_tensor(ReducedTensor, A_cell[ij]);
        end
    end

    #= Expands minimal unit cell to final unit cell if needed =#
    if Ni == dims[1] && Nj == dims[2]
        UnitCell(D, dims, pattern, symmetry, R_cell, A_cell)
    elseif Ni < dims[1] || Nj < dims[2]
        pattern_large = Array{Char, 2}(undef, dims);
        R_cell_large = Array{ReducedTensor{T}, 2}(undef, dims);
        A_cell_large = Array{Tensor{T}, 2}(undef, dims);
        for ij ∈ CartesianIndices(dims)
            loc = coord(ij, (Ni, Nj));
            pattern_large[ij] = pattern[loc]
            A_cell_large[ij] = A_cell[loc];
            R_cell_large[ij] = R_cell[loc];
        end

        UnitCell(D, dims, pattern_large, symmetry, R_cell_large, A_cell_large)
    end
end

function UnitCell{T}(D::Int64, dims::Tuple, pattern::Array{Char, 1}, symmetry::LatticeSymmetry = XY) where {T<:Union{Float64, ComplexF64}}


    Ai = Tensor{T}(D, symmetry);
    Ri = cast_tensor(ReducedTensor, Ai);

    pattern_large = Array{Char, 2}(undef, dims);
    R_cell_large = Array{ReducedTensor{T}, 2}(undef, dims);
    A_cell_large = Array{Tensor{T}, 2}(undef, dims);

    for xy ∈ CartesianIndices(dims)
        pattern_large[xy] = pattern[1]
        A_cell_large[xy] = Ai;
        R_cell_large[xy] = Ri;
    end

    UnitCell(D, dims, pattern_large, symmetry, R_cell_large, A_cell_large)

end

function UnitCell(
    pattern::Array{Char},
    symmetry::LatticeSymmetry,
    R_cell::Array{ReducedTensor{T}},
    A_cell::Array{Tensor{T}}) where {T<:Union{Float64, ComplexF64}}

    dims = size(A_cell);
    D = size(A_cell[1,1].A, 1);

    UnitCell(D, dims, pattern, symmetry, R_cell, A_cell);
end

function UnitCell(R_cell::Array{ReducedTensor{T}}) where {T<:Union{Float64, ComplexF64}}
    dims = size(R_cell);
    D = 0;

    if length(dims) == 1
        pattern = ['a']
    else
        pattern = [Char(i * dims[1] + j) for i ∈ 0:dims[1]-1, j ∈ 1:dims[2]];
    end

    symmetry = R_cell[1,1].symmetry;
    UnitCell(D, dims, pattern, symmetry, R_cell)
end

function UnitCell(R_cell::Array{ReducedTensor{T}},
    A_cell::Array{Tensor{T}}) where {T<:Union{Float64, ComplexF64}}
    dims = size(R_cell);
    D = size(A_cell[1,1].A, 1);

    if length(dims) == 1
        pattern = ['a']
    else
        pattern = [Char(i * dims[1] + j) for i ∈ 0:dims[1]-1, j ∈ 1:dims[2]];
    end

    symmetry = R_cell[1,1].symmetry;
    UnitCell(D, dims, pattern, symmetry, R_cell, A_cell)
end

function (uc::UnitCell)(::Type{T}, loc::CartesianIndex) where {T<:Union{Tensor, BraTensor, ReducedTensor, Environment}}

    loc = coord(loc, uc.dims);

    if T == Tensor
        return deepcopy(uc.A[loc])
    elseif T == BraTensor
        return deepcopy(uc.B[loc])
    elseif T == ReducedTensor
        return deepcopy(uc.R[loc])
    elseif T == Environment
        return deepcopy(uc.E[loc])
    end

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
