function loadstate(folder, file_name, step)
    data = h5open(folder*file_name*"_state_step=$(step).h5", "r");
    state = reconstructcomplexState(read(data["mps"]));
    return state
end

function load_sutensor_matlab(folder, filename)
    data = h5open(folder*filename, "r");
    S = read(data["S"]);
    S = permutedims(S, (3, 2, 5, 4, 1));
    wh = diag(read(A_M["w_h"]));
    wv = diag(read(A_M["w_v"]));

    return SimpleUpdateTensor(S, [wv, wh, wv, wh])
end

function load_ctm_matlab(filepath, dims)
    psi_c = h5open(filepath);
    A0s = Array{SimpleUpdateTensor{Float64}}(undef, 2,2);
    As = Array{Tensor{Float64}}(undef, 2,2);
    Rs = Array{ReducedTensor{Float64}}(undef, 2,2);
    Es = Array{Environment}(undef, 2,2);
    whs = []; wvs = [];

    for i ∈ 1:dims[1] , j ∈ 1:dims[2]
        # Read SU weights
        wh = diag(read(psi_c["x$(i)y$(j)/wh"]));
        push!(whs, wh);
        wv = diag(read(psi_c["x$(i)y$(j)/wv"]));
        push!(wvs, wv);

        # Read tensors
        A = permutedims(read(psi_c["x$(i)y$(j)/A"]), (3, 2, 5, 4, 1));
        A0 = SimpleUpdateTensor(normalize!(A), [wv, wh, wv, wh])
        A = Tensor(normalize!(A))
        rA = cast_tensor(ReducedTensor, A);

        As[i, j] = A;
        A0s[i, j] = A0;
        Rs[i, j] = rA;

        # Read environment
        Cs = Matrix{Float64}[];
        Ts = Array{Float64, 3}[];
        for m ∈ 1:4
            Cm = read(psi_c["x$(i)y$(j)/C$m"])
            push!(Cs, Cm);
            Tibk = read(psi_c["x$(i)y$(j)/T$m"]);
            Ti = reshape(Tibk, (size(Tibk, 1), size(Tibk, 2), :));
            push!(Ts, Ti);
        end
        E = Environment(Cs, Ts, (i, j));
        Es[i, j] = E
    end
end
