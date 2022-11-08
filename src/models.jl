#abstract type Hamiltonian end #! already defined in ipeps_ctm module

mutable struct HeisenbergXYZ <: Hamiltonian
    Jx::Float64
    Jy::Float64
    Jz::Float64

    hx::Float64
    hy::Float64
    hz::Float64

    hij::Array{ComplexF64,2}

    function HeisenbergXYZ(J::Vector{Float64}, h::Vector{Float64})
        ⊗ = kron;
        𝟙 = [1 0; 0 1]; X = [0 1; 1 0]; Z = [1 0; 0 -1]; Y = [0 -im; im 0];

        hNN = J[1] * X ⊗ X + J[2] * Y ⊗ Y + J[3] * Z ⊗ Z + h[1] * (𝟙 ⊗ X + X ⊗ 𝟙) +
        h[2] * (𝟙 ⊗ Y + Y ⊗ 𝟙) + h[3] * (𝟙 ⊗ Z + Z ⊗ 𝟙);

        new(J..., h..., hNN);
    end

end
