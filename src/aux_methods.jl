+(xy::CartesianIndex, ij::Tuple{Int64, Int64}) = xy + CartesianIndex(ij)

function coord(xy::CartesianIndex, dims::Tuple{Int64, Int64})
    Nx = dims[1];
    Ny = dims[2];

    Nx != 1 && (xy[1] > Nx || xy[1] < 0) && (xy = CartesianIndex(mod(xy[1], Nx), xy[2]);)
    Nx == 1 && (xy[1] > Nx || xy[1] < 0) && (xy = CartesianIndex(1, xy[2]);)

    Ny != 1 && (xy[2] > Ny || xy[2] < 0) && (xy = CartesianIndex(xy[1], mod(xy[2], Ny));)
    Ny == 1 && (xy[2] > Ny || xy[2] < 0) && (xy = CartesianIndex(xy[1], 1);)

    xy[1] == 0 && (xy = CartesianIndex(Nx, xy[2]);)
    xy[2] == 0 && (xy = CartesianIndex(xy[1], Ny);)

    return xy;
end
