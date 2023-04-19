function load_sutensor_matlab(folder, filename)
    data = h5open(folder*filename, "r");
    S = read(data["S"]);
    S = permutedims(S, (3, 2, 5, 4, 1));
    wh = diag(read(A_M["w_h"]));
    wv = diag(read(A_M["w_v"]));

    return SimpleUpdateTensor(S, [wv, wh, wv, wh])
end

function load_ctm_matlab(filepath, target_cell_size; load_environment::Bool = true)

    """
    Convention axis and indices in unit-cell
    ____ x(j)
    |
    |
    y(i)

    Convention indices in arrays: (i,j) i.e (y,x)

    """
    psi_c = h5open(filepath);
    original_cell_size = read(psi_c["cell_size"]);

    As = Array{Tensor{ComplexF64}}(undef, target_cell_size);
    Rs = Array{ReducedTensor{ComplexF64}}(undef, target_cell_size);
    Es = Array{Environment{ComplexF64}}(undef, target_cell_size);
    #whs = []; wvs = [];

    if original_cell_size == [1 1]
        # Read cell tensors
        Re_A = permutedims(read(psi_c["Re_x1y1/A"]), (3, 2, 5, 4, 1));
        Im_A = permutedims(read(psi_c["Im_x1y1/A"]), (3, 2, 5, 4, 1));
        A = Tensor(Re_A + im * Im_A)
        rA = cast_tensor(ReducedTensor, A;  renormalize = false);

        # Read environment
        if load_environment == true
            Cs = Matrix{ComplexF64}[];
            Ts = Array{ComplexF64, 4}[];
            for m ∈ 1:4
                Re_Cm = read(psi_c["Re_x1y1/C$m"])
                Im_Cm = read(psi_c["Im_x1y1/C$m"])
                push!(Cs, Re_Cm + im * Im_Cm);
                Re_Tibk = read(psi_c["Re_x1y1/T$m"]);
                Im_Tibk = read(psi_c["Im_x1y1/T$m"]);
                Ti = Re_Tibk + im * Im_Tibk;
                push!(Ts, Ti);
            end
        end

        for x ∈ 1:target_cell_size[1] , y ∈ 1:target_cell_size[2]
            As[y, x] = A;
            Rs[y, x] = rA;
            if load_environment == true
                E = Environment(Cs, Ts, CartesianIndex(y, x));
                Es[y, x] = E
            end
        end
    else
        for x ∈ 1:target_cell_size[1] , y ∈ 1:target_cell_size[2]

            # Read cell tensors
            Re_A = permutedims(read(psi_c["Re_x$(x)y$(y)/A"]), (3, 2, 5, 4, 1));
            Im_A = permutedims(read(psi_c["Im_x$(x)y$(y)/A"]), (3, 2, 5, 4, 1));
            A = Tensor(Re_A + im * Im_A)
            rA = cast_tensor(ReducedTensor, A;  renormalize = false);

            #= # SU tensors: first remove weights from tensor
            A0W = SimpleUpdateTensor(A.A, [wv_inv, wh_inv, wv_inv, wh_inv]);
            A0 = cast_tensor(Tensor, A0W);
            S = SimpleUpdateTensor(normalize(A0.A), [wv, wh, wv, wh]); =#

            #= # Read SU weights
            wh = ComplexF64.(diag(read(psi_c["x$(i)y$(j)/wh"])));
            wh_inv = 1 ./wh;

            push!(whs, wh);
            wv = ComplexF64.(diag(read(psi_c["x$(i)y$(j)/wv"])));
            wv_inv = 1 ./wv;

            push!(wvs, wv);
    =#

            As[y, x] = A;
            #Ss[i, j] = S;
            Rs[y, x] = rA;

            # Read environment
            if load_environment == true
                Cs = Matrix{ComplexF64}[];
                Ts = Array{ComplexF64, 4}[];
                for m ∈ 1:4
                    Re_Cm = read(psi_c["Re_x$(x)y$(y)/C$m"])
                    Im_Cm = read(psi_c["Im_x$(x)y$(y)/C$m"])
                    push!(Cs, Re_Cm + im * Im_Cm);
                    Re_Tibk = read(psi_c["Re_x$(x)y$(y)/T$m"]);
                    Im_Tibk = read(psi_c["Im_x$(x)y$(y)/T$m"]);
                    Ti = Re_Tibk + im * Im_Tibk;
                    push!(Ts, Ti);
                end
                E = Environment(Cs, Ts, CartesianIndex(y, x));
                Es[y, x] = E
            end
        end
    end

    load_environment == true && (return As, Rs, Es)
    return As, Rs
end




#= MATLAB method to export a state in a format compatible with load_ctm_matlab


function export_hdf5(data, name)
    filename = strcat('/mnt/c/Users/Juan/surfdrive/QuantumSimulationPhD/Code/iPEPS-Julia/input/', name, '.h5')
    sc = data.allsimpars{1}.unitcell;
    for sx = 1:sc(1)
        for sy = 1:sc(2)
            loc = strcat("/x", string(sx), "y", string(sy), "/");

            h5create(filename, strcat(loc, "A"), size(data.myc.xA{sx,sy}));
            h5create(filename, strcat(loc, "wh"), size(data.myc.xah{sx,sy}));
            h5create(filename, strcat(loc, "wv"), size(data.myc.xav{sx,sy}));

            h5create(filename, strcat(loc, "C4"), size(data.myc.xC1{sx,sy}));
            h5create(filename, strcat(loc, "C1"), size(transpose(data.myc.xC2{sx,sy})));
            h5create(filename, strcat(loc, "C2"), size(data.myc.xC3{sx,sy}));
            h5create(filename, strcat(loc, "C3"), size(transpose(data.myc.xC4{sx,sy})));


            h5create(filename, strcat(loc, "T1"), size(data.myc.xT1{sx,sy}));
            h5create(filename, strcat(loc, "T2"), size(data.myc.xT2{sx,sy}));
            h5create(filename, strcat(loc, "T3"), size(data.myc.xT3{sx,sy}));
            h5create(filename, strcat(loc, "T4"), size(data.myc.xT4{sx,sy}));

            h5write(filename, strcat(loc, "A"), data.myc.xA{sx,sy});
            h5write(filename, strcat(loc, "wh"), data.myc.xah{sx,sy});
            h5write(filename, strcat(loc, "wv"), data.myc.xav{sx,sy});

            h5write(filename, strcat(loc, "C4"), data.myc.xC1{sx,sy});
            h5write(filename, strcat(loc, "C1"), transpose(data.myc.xC2{sx,sy}));
            h5write(filename, strcat(loc, "C2"), data.myc.xC3{sx,sy});
            h5write(filename, strcat(loc, "C3"), transpose(data.myc.xC4{sx,sy}));

            h5write(filename, strcat(loc, "T1"), data.myc.xT1{sx,sy});
            h5write(filename, strcat(loc, "T2"), data.myc.xT2{sx,sy});
            h5write(filename, strcat(loc, "T3"), data.myc.xT3{sx,sy});
            h5write(filename, strcat(loc, "T4"), data.myc.xT4{sx,sy});

        end
    end

end =#
