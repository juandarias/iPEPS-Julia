
@enum Direction UP=1 RIGHT DOWN LEFT VERTICAL HORIZONTAL

abstract type Renormalization end
abstract type FullSystem <: Renormalization end # DOI: 10.1143/JPSJ.65.891
abstract type HalfSystem <: Renormalization end # DOI: 10.1103/PhysRevB.84.041108
abstract type TwoCornersSym <: Renormalization end # DOI: 10.1103/PhysRevB.80.094403
abstract type TwoCorners <: Renormalization end # DOI: 10.1103/PhysRevLett.113.046402
abstract type TwoCornersSimple <: Renormalization end # DOI: 10.1103/PhysRevLett.113.046402

# When are the projectors calculated
abstract type Start <: Renormalization end
abstract type EachMove <: Renormalization end
abstract type BraKetOverlap <: Renormalization end

"""
    mutable struct Projectors{T<:Renormalization}

Parametric type for renormalization projectors.

### Fields
- `Pl` : Vector with left-move projectors into upper and lower subspace
- `Pr` : Vector with right-move projectors into upper and lower subspace
- `Pu` : Vector with up-move projectors into left and rigth subspace
- `Pd` : Vector with down-move projectors into left and right subspace

### Notes

 |                    |
 ▽ : Lower subspace;  △ : Upper subspace; --◁-- : Left subspace; --▷-- : Right subspace
 |                    |

 """

mutable struct Projectors{T<:Renormalization}

# |                    |
# ▽ : Lower subspace;  △ : Upper subspace; --◁-- : Left subspace; --▷-- : Right subspace
# |                    |

    dims::Tuple
    Pu::Array{Vector{Array{ComplexF64}}} # [Left subspace, Right subspace]
    Pr::Array{Vector{Array{ComplexF64}}} # [Upper subspace, Lower subspace]
    Pd::Array{Vector{Array{ComplexF64}}} # [Left subspace, Right subspace]
    Pl::Array{Vector{Array{ComplexF64}}} # [Upper subspace, Lower subspace]

    Projectors{T}() where {T<:Renormalization} = new{T}()

    function Projectors{T}(unitcell::UnitCell) where {T<:Renormalization}

        dims = unitcell.dims;
        Pu = Array{Vector{Array{ComplexF64}}}(undef, dims);
        Pr = Array{Vector{Array{ComplexF64}}}(undef, dims);
        Pd = Array{Vector{Array{ComplexF64}}}(undef, dims);
        Pl = Array{Vector{Array{ComplexF64}}}(undef, dims);

        new{T}(dims, Pu, Pr, Pd, Pl)
    end
end

function (proj::Projectors)(direction::Direction, loc::CartesianIndex)

    loc = coord(loc, proj.dims);

    if direction == UP
        return proj.Pu[loc]
    elseif direction == RIGHT
        return proj.Pr[loc]
    elseif direction == DOWN
        return proj.Pd[loc]
    elseif direction == LEFT
        return proj.Pl[loc]
    end

end
