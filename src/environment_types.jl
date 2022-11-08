#########
# Types #
#########


mutable struct CTM{T}
    unitcell::UnitCell{T}
    cell_environment::Array{Environment{T}, 2}
    energy_site::Array{Float64,2}
    energy_hbond::Array{Float64,2}
    energy_vbond::Array{Float64,2}

    function CTM(
        unitcell::UnitCell{T},
        cell_environment::Array{Environment{T}, 2}) where {T}

        energy_site = zeros(Float64, unitcell.dims);
        energy_hbond = zeros(Float64, unitcell.dims .- 1);
        energy_vbond = zeros(Float64, unitcell.dims .- 1);

        new{T}(unitcell, cell_environment, energy_site, energy_hbond, energy_vbond)
    end

    function CTM(unitcell::UnitCell{T}) where {T}
        dims = unitcell.dims;
        C = fill(zeros(T, (1,1)), 4);
        E = fill(zeros(T, (1,1,1)), 4);
        empty_environment = [Environment(C, E, (i, j)) for i ∈ 1:dims[1], j ∈ 1:dims[2]];

        CTM(unitcell, empty_environment);

    end
end

mutable struct bMPS{T}
end
