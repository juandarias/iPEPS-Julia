############################## Types ##############################
export MPS
export MPO
export WII
export HilbertSpace
export Bra, Ket, BraKet
export CanonicalForm
export None, Left, Right, Mixed


abstract type HilbertSpace end
struct Ket <: HilbertSpace end
struct Bra <: HilbertSpace end
struct BraKet <: HilbertSpace end

abstract type CanonicalForm end
struct None <: CanonicalForm end
struct Left <: CanonicalForm end
struct Right <: CanonicalForm end
struct Mixed <: CanonicalForm end


"""
Structure for a finite `MPS`

## Fields
- Ai: Vector with rank-3 tensors, where the indices labels are: left auxiliary (first), physical (second) and right auxiliary (third) spaces
- L: lenght of `MPS`
- D: Vector bond dimensions between tensor ``i`` and ``i+1``
- canonical: canonical form of tensor, `None` for general tensors
- oc: orthogonality center for MPS in a canonical form. `-1` for `MPS` not in canonical form
- physical_space: Hilbert Space encoded by physical leg, e.g. `Ket`, `Bra` or `BraKet`, the last one used for vectorized forms of operators or density matrices
- d: Physical bond dimension

"""
mutable struct MPS{T}
    Ai::Vector{Array{T,3}} # legs labels is aux left, phys, aux right
    L::Int64
    D::Array{Int64,1}
    "Canonical form of tensor, `None` for general tensors"
    canonical::CanonicalForm
    "Orthogonality center for MPS in a canonical form. `-1` for `MPS` not in canonical form"
    oc::Int64 # orthogonality center
    "Hilbert Space encoded by physical leg, e.g. `Ket`, `Bra` or `BraKet`"
    physical_space::HilbertSpace
    d::Int
    
    function MPS(Ai::Vector{Array{T,3}}) where {T}
        L = length(Ai);
        D = []
        for i in 1:L-1
            append!(D, size(Ai[i])[end])
        end
        new{T}(Ai, L, D, None(), -1, Ket(), 2)
    end
end

mutable struct MPO{T}
    Wi::Vector{Array{T,4}} # leg labels; phys in, aux left, phys out, aux right
    L::Int64
    D::Array{Int64,1}
    
    function MPO(Wi::Vector{Array{T,4}}) where {T}
        L = length(Wi);
        D = []
        for i in 1:L-1
            append!(D, size(Wi[i])[end])
        end
        new{T}(Wi, L, D)
    end
end