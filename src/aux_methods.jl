+(coord::CartesianIndex, ij::Tuple{Int64, Int64}) = coord + CartesianIndex(ij)

function coord(ij::CartesianIndex, dims::Tuple{Int64, Int64})
    Ni = dims[1];
    Nj = dims[2];

    Ni != 1 && (ij[1] > Ni || ij[1] < 0) && (ij = CartesianIndex(mod(ij[1], Ni), ij[2]);)
    Ni == 1 && (ij[1] > Ni || ij[1] < 0) && (ij = CartesianIndex(1, ij[2]);)

    Nj != 1 && (ij[2] > Nj || ij[2] < 0) && (ij = CartesianIndex(ij[1], mod(ij[2], Nj));)
    Nj == 1 && (ij[2] > Nj || ij[2] < 0) && (ij = CartesianIndex(ij[1], 1);)

    ij[1] == 0 && (ij = CartesianIndex(Ni, ij[2]);)
    ij[2] == 0 && (ij = CartesianIndex(ij[1], Nj);)

    return ij;
end
