@warn "Debugging types present in projectors types"

@enum Direction UP=1 RIGHT DOWN LEFT VERTICAL HORIZONTAL

abstract type Renormalization end
abstract type FullSystem <: Renormalization end # DOI: 10.1143/JPSJ.65.891
abstract type HalfSystem <: Renormalization end # DOI: 10.1103/PhysRevB.84.041108
abstract type TwoCornersSym <: Renormalization end # DOI: 10.1103/PhysRevB.80.094403
abstract type TwoCorners <: Renormalization end # DOI: 10.1103/PhysRevLett.113.046402
abstract type TwoCornersSimple <: Renormalization end # DOI: 10.1103/PhysRevLett.113.046402

#! debug
abstract type Start <: Renormalization end
abstract type EachMove <: Renormalization end
abstract type EachMoveCirc <: Renormalization end

abstract type ConvergenceCriteria end
struct OnlyCorners <: ConvergenceCriteria end
struct Full <: ConvergenceCriteria end

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

    Pl::Vector{Array{ComplexF64, 2}} # [Upper subspace, Lower subspace]
    Pr::Vector{Array{ComplexF64, 2}} # [Upper subspace, Lower subspace]
    Pu::Vector{Array{ComplexF64, 2}} # [Left subspace, Right subspace]
    Pd::Vector{Array{ComplexF64, 2}} # [Left subspace, Right subspace]
    Projectors{T}() where {T<:Renormalization} = new{T}()
end
