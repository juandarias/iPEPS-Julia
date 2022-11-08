############################## Methods ##############################

function calc_env(ket::MPS, bra::MPS, loc::Int64)
    L = ket.L;
    bra = conj(bra); # transposition is taken care in label of legs
    
    # Left env
    EiL = 1;
    if loc != 1
        @tensor EiL[a,b] := ket.Ai[1][dummy,y,a]*bra.Ai[1][dummy,y,b];
        for i ∈ 2:loc-1 # Contract ket and then bra to tranfer matrix E_i-1 and generate E_i
            @tensor ci_ket[a,b,c] := EiL[x,a]*ket.Ai[i][x,b,c]; # aux right bra, phys down ket, aux right ket
            @tensor EiL[a,b] := ci_ket[x,y,a]*bra.Ai[i][x,y,b];
        end
    end
    
    # right env    
    EiR = 1;
    if loc != L
        @tensor EL[a,b] := ket.Ai[L][a,x,dummy]*bra.Ai[L][b,x,dummy]; # aux left ket, aux left bra
        EiR = EL; # 1st right transfer matrix
        for i ∈ L-1:-1:loc+1
            @tensor ci_ket[a,b,c] := EiR[x,c]*ket.Ai[i][a,b,x]; # aux left ket, phys down ket, aux left bra
            @tensor EiR[a,b] := ci_ket[a,x,y]*bra.Ai[i][b,x,y];
        end
    end

    return EiL, EiR
end


"""
    function mps_compress_var(bigket::MPS, guessket::MPS; tol_compr::Float64, Dmax::Int=200, rate=Float64=1.1)

## Description
Approximates a target `MPS` |Ψ⟩ by a `MPS` |Φ⟩ of smaller bond dimension by iteratively updating the site tensors |Φ⟩. |Φ⟩ is brought into mixed canonical form such that the orthogonality center corresponds to the tensor to update at each step. Both input `MPS` have to be normalized and |Φ⟩ has to passed in left or right canonical form. The bond dimension of |Φ⟩ is increased if ‖|Ψ⟩ - |Φ⟩‖² > ϵ, where ϵ is the compression tolerance

## Arguments
- bigket: target (normalized) `MPS` |Ψ⟩ to be approximated. 
- guessket: a guess, initial (normalized) `MPS` |Φ⟩ of the desired bond dimension. Can be obtained by SVD truncation
- tol_compr: maximum accepted error of ‖|Ψ⟩ - |Φ⟩‖²
- Dmax: maximum bond dimension allowed during compression
- rate: rate at which the bond dimension of |Φ⟩ can be increased

"""
function mps_compress_var(bigket::MPS, guessket::MPS; tol_compr::Float64 = 1e-6, Dmax::Int = 200, rate=Float64 = 1.1, normalize::Bool = true)
    L = bigket.L;
    ϵ = 2;
    s = 0;

    comprket = deepcopy(guessket);
    
    # Do one round sweep. 
    if comprket.oc == L
        sweep_qr!(comprket, direction = "right", final_site = 1)
    end
    @assert comprket.oc == 1 "Orthogonality center of seed should be at first site to continue" 
    
    
    while abs(ϵ) > tol_compr
        s += 1;
        for n in 1:L-1 # Left-right sweep
            log_message("Left sweep at site $(n), ", color = :blue);# 
            
            #* Calculate x = b for each Aσᵢ
            Lenv, Renv = calc_env(bigket, comprket, n);
            Mnew = zeros(ComplexF64, size(comprket.Ai[n])); # Allocate array for updated tensor
            for d ∈ 1:4
                M_d = bigket.Ai[n][:,d,:];
                Mnew[:,d,:] = transpose(Lenv)*M_d*Renv;
            end
            
            comprket.Ai[n] = Mnew; # update orthogonality center tensor in MPS
            sweep_qr!(comprket; final_site = n + 1); # move oc center left for next step

            ϵ = 1 - sum(tr(Mnew[:,d,:]'*Mnew[:,d,:]) for d ∈ 1:4);
            log_message("with error $(round(abs(ϵ), sigdigits = 4))\n", color = :blue);# 
        end

        for n in L:-1:1 # right-left sweep
            log_message("Right sweep at site $(n), ", color = :blue);# 
            
            #* Calculate x = b for each Aσᵢ
            Lenv, Renv = calc_env(bigket, comprket, n);
            Mnew = zeros(ComplexF64, size(comprket.Ai[n])); # Allocate array for updated tensor
            for d ∈ 1:4
                M_d = bigket.Ai[n][:,d,:];
                Mnew[:,d,:] = transpose(Lenv)*M_d*Renv;
            end
            
            comprket.Ai[n] = Mnew; # update tensor in MPS
            sweep_qr!(comprket; final_site = n - 1, direction = "right"); # move oc center right for next step

            ϵ = 1 - sum(tr(Mnew[:,d,:]'*Mnew[:,d,:]) for d ∈ 1:4);
            log_message("with error $(round(abs(ϵ), sigdigits = 4))\n", color = :blue);# 
        end

        println("\n Sweep : $(s). Error: $(round(abs(ϵ), sigdigits = 4)) \n")

        if abs(ϵ) > tol_compr
            if maximum(comprket.D) >= Dmax
                log_message("\n##### Maximum bond dimension reached before convergence. Returning not converged state with error = $(round(abs(ϵ), sigdigits = 4)) ##### \n "; color = :yellow)
                return comprket, abs(ϵ)
            end

            Dincr = max(floor(Int, maximum(comprket.D) * rate), maximum(comprket.D) + 5);
            newD = min(Dmax, Dincr); 
            log_message("\nError is above tolerance, increasing bond dimension to $(newD) \n\n", color = :yellow)
            grow_mps_tensors!(comprket, newD);
            
            # Do one round sweep
            #sweep_qr!!(comprket, direction = "left");
            sweep_qr!(comprket; direction = "right", final_site = 1) 
        end
    end
    log_message("\n##### CONVERGED!! #####\n", color = :green);# 
    
    if normalize == true
        nxs = (sqrt(1 - ϵ))^(-1/L)
        prod!(nxs, comprket);
    end
    return comprket, abs(ϵ)
end

"""
    function mps_compress_svd!(ket::MPS; kwargs...)

## Description
Inplace method to compress a MPS by truncating the number of singular values at each bond to maximum number of kept values ``D_\\text{max}`` or such that the sum of the square discarded values is below a cut-off ``\\epsilon_\\text{max}``. 

## Arguments
- ket : MPS to be compressed
- Dmax : Maximum number of singular values to be kept
- ϵmax : Cut-off for the sum of the square discarded values

## Notes
- `Dmax` and/or `ϵmax` have to be provided to compress the MPS
- The method does not work for mixed canonical MPS.
"""
function mps_compress_svd!(ket::MPS; kwargs...)
    L = ket.L;
    local ϵ_c = 0.0;
    if ket.canonical == None()
        sweep_qr!(ket); # prepares left-canonical form
        ϵ_c = sweep_svd!(ket, final_site = 1, direction = "right"; kwargs...); # generates mixed canonical form and compress
    elseif ket.canonical == Left() && ket.oc == L
        ϵ_c = sweep_svd!(ket, final_site = 1, direction = "right"; kwargs...);
    elseif ket.canonical == Right() && ket.oc == 1
        ϵ_c = sweep_svd!(ket; kwargs...);
    else
        @assert false "The method doesn't accept mixed or ``incomplete`` canonical forms. The orthogonality center should be at the edges of the chain"
    end
    return ϵ_c
end

############################## Methods for generical MPS ##############################
# The methods below were written for MPS that for one or another reason cannot be
# brought into canonical form


function LxMxR(x, Ltilde, Rtilde, dims) # Linear map for matrix-less evaluation of Ax
    x = reshape(x, dims);
    #@tensor Ox[a,b] := Ltilde[x,a]*x[x,y]*Rtilde[y,b] #! this and next line give *almost* the same result
    Ox = transpose(Ltilde)*(x*Rtilde); # works even when one of the environments is = 1
    Ox = reshape(Ox, :, 1);
    return Ox
end

#= 
function mps_compress_cg(bigket::MPS, guessket::MPS; tol_compr::Float64, max_sweeps::Int=10, Dmax::Int=200, kwargs...)
    L = bigket.L;
    ϵ = 2;
    s = 0;
    comprket = copy(guessket); #! this is needed as otherwise guessket is modified, which can occur inadvertently
    o_bigket = overlap(bigket, bigket);
    while abs(ϵ) > tol_compr
        s += 1;
        for n in 1:L # Left-right sweep
            log_message("\n\nLeft sweep at site $(n)\n\n", color = :blue);# 
            Mnew = compress_tensor_cg(bigket, comprket, n; kwargs...);
            comprket.Ai[n] = Mnew; # update tensor in MPS
        end

        for n in L-1:-1:2 # right-left sweep
            log_message("\n\nRight sweep at site $(n)\n\n", color = :blue);# 
            Mnew = compress_tensor_cg(bigket, comprket, n; kwargs...);
            comprket.Ai[n] = Mnew; # update tensor in MPS
            # 
        end

        o_comprket = overlap(comprket, comprket);
        o_bigcompr = overlap(bigket, comprket);
        ϵ = (o_bigket + o_comprket - o_bigcompr - conj(o_bigcompr))/o_bigket;
        println("\n Sweep : $(s). Error: $(ϵ) \n")

        if s == max_sweeps
            newD = min(Dmax, Int(floor(comprket.D[1]*1.1))); 
            log_message("\n Reached max number of sweeps, increasing bond dimension to $(newD) \n", color = :yellow)
            comprket = grow_mps_tensors!(comprket, newD);
            s = 0;
        end
    end
    log_message("\n##### CONVERGED!! #####\n", color = :green);# 
    return comprket
end


function mps_compress_lbfgs(bigket::MPS, guessket::MPS; tol_compr::Float64, max_sweeps::Int=10, Dmax::Int=200, kwargs...)
    L = bigket.L;
    ϵ = 2;
    s = 0;
    comprket = MPS(copy(guessket.Ai)); #! this is needed as otherwise guessket is modified, which can pass inadvertently
    o_bigket = overlap(bigket, bigket);
    while abs(ϵ) > tol_compr
        s += 1;
        for n in 1:L # Left-right sweep
            log_message("\n\nLeft sweep at site $(n)\n\n", color = :blue);# 
            Mnew = compress_tensor_lbfgs(bigket, comprket, n; kwargs...);
            comprket.Ai[n] = Mnew; # update tensor in MPS
        end

        for n in L-1:-1:2 # right-left sweep
            log_message("\n\nRight sweep at site $(n)\n\n", color = :blue);# 
            Mnew = compress_tensor_lbfgs(bigket, comprket, n; kwargs...);
            comprket.Ai[n] = Mnew; # update tensor in MPS
            # 
        end

        o_comprket = overlap(comprket, comprket);
        o_bigcompr = overlap(bigket, comprket);
        ϵ = (o_bigket + o_comprket - o_bigcompr - conj(o_bigcompr))/o_bigket;
        
        println("\n Sweep : $(s). Error: $(ϵ) \n")
        if s == max_sweeps
            newD = min(Dmax, Int(floor(comprket.D[1]*1.1))); 
            log_message("\n Reached max number of sweeps, increasing bond dimension to $(newD) \n", color = :yellow)
            comprket = grow_mps_tensors!(comprket, newD); #? what is happenning to the canonical form if I pad the tensors with zeros?
            s = 0;
        end
    end
    log_message("\n##### CONVERGED!! #####\n", color = :green);# 
    return comprket
end

function compress_tensor(bigket::MPS, guessket::MPS, loc::Int, tol_compr::Float64)
    # Calc environments
    Ltilde, Rtilde = calc_env(guessket, guessket, loc);
    L, R = calc_env(bigket, guessket, loc);
    
    Mnew = zeros(size(guessket.Ai[loc])); # Allocate array for updated tensor
    for d = 1:4 # Sweep physical dimension
        M_d = bigket.Ai[loc][:,d,:];
        Mseed_d = guessket.Ai[loc][:,d,:];
                
        # Define Ax=b
        A = x -> LxMxR(x, Ltilde, Rtilde, size(Mseed_d));
        b = reshape(transpose(L)*M_d*R, :, 1);
        xseed = reshape(Mseed_d, : ,1);
        
        converged = 0;
        t = 0;
        while converged != 1 # Solve
            t += 1;
            Mnew_d, info = linsolve(A, b, xseed; tol=tol_compr);
            converged = info.converged;
            converged == 0 && (println("Not converged for site $(loc) and d_i=$(d), trying again"))
            xseed = Mnew_d;
            @assert t == 10 "Reached maximum number of attemps"
        end
        Mnew[:,d,:] = reshape(Mnew_d, size(Mseed_d)...); # build updated site-tensor
    end
    return Mnew; # update tensor in MPS
    #
end


function compress_tensor_cg(
    bigket::MPS, 
    guessket::MPS, 
    loc::Int; 
    fav_solver=gmres!,
    alt_solver=cg!,
    verb_compressor=0, 
    max_solver_runs=10, 
    kwargs...
    )
    
    # Calc environments
    Ltilde, Rtilde = calc_env(guessket, guessket, loc);
    L, R = calc_env(bigket, guessket, loc);
    
    Mnew = zeros(ComplexF64, size(guessket.Ai[loc])); # Allocate array for updated tensor
    for d = 1:4 # Sweep physical dimension
        M_d = bigket.Ai[loc][:,d,:];
        Mseed_d = guessket.Ai[loc][:,d,:];
        b_dim = prod(size(Mseed_d));
                
        # Define Ax=b. 
        #! A is not completely hermitian or positive 
        #! @tensor introduces some numerical error which leads to some eigenvalues ~= 0 but < 0
        A = (L, R, dims) -> LinearMap(b_dim, ishermitian=true, isposdef=true) do Ox, x 
            x = reshape(x, dims);
            Ox[:] = transpose(L)*(x*R); # works even when one of the environments is = 1
            return Ox
        end
        b = reshape(transpose(L)*M_d*R, :, 1);
        xseed = reshape(Mseed_d, : ,1);
        #xseed =  zeros(ComplexF64, prod(size(Mseed_d)));
        
        
        converged = 0;
        t = 0; # counter for each solver
        attempts = [0]; # total number of attemps
        attempts_no_iter = 0;
        solver = fav_solver; # sets the solver to conjugate gradient. Works only for positive-definite symmetric matrices. 
        log_message("Using $(fav_solver) solver\n"; color = :blue)

        local Mnew_d; # makes Mnew_d accesible to while loop
        local hist, residual;
        while converged != true # Solve
            t += 1;

            # Solve
            if t == max_solver_runs # tries a new seed for the last run of each solver
                log_message("\nTrying optimization with a new seed\n"; color = :blue)
                xseed = rand(ComplexF64, size(xseed)...);
                Mnew_d_new, hist_new = solver(xseed, A(Ltilde, Rtilde, size(Mseed_d)), b; log=true, verbose=false, kwargs...)
                if hist_new[:resnorm][end] < residual # updates tensor and convergence info with new calculation
                    Mnew_d = Mnew_d_new;
                    hist = hist_new;
                end
            else
                Mnew_d, hist = solver(xseed, A(Ltilde, Rtilde, size(Mseed_d)), b; log=true, verbose=false, kwargs...)
            end
            
            # If no iterations were runned, try again with different solvers, seeds or tolerances till the tensor is optimized
            if hist.iters == 0 && attempts_no_iter == 0
                log_message("\nNo solver iterations, trying new solver, tolerances and seed\n"; color = :yellow);
                log_message("Trying with alternative solver $(alt_solver)\n"; color = :yellow)
                Mnew_d, hist = alt_solver(xseed, A(Ltilde, Rtilde, size(Mseed_d)), b; log=true, verbose=false, kwargs...)
                if hist.iters == 0
                    log_message("Trying with a new seed\n"; color = :yellow)
                    rseed = zeros(ComplexF64, size(xseed)...);
                    Mnew_d, hist = solver(rseed, A(Ltilde, Rtilde, size(Mseed_d)), b; log=true, verbose=false, kwargs...)
                    if hist.iters == 0
                        log_message("Trying with default tolerances\n"; color = :yellow)
                        Mnew_d, hist = solver(xseed, A(Ltilde, Rtilde, size(Mseed_d)), b; log=true, verbose=false)
                        if hist.iters == 0
                            log_message("\nFailed to optimize tensor\n!!!"; color = :red);
                        end
                    end 
                end
                attempts_no_iter = 1;
            end

            # Convergence info
            #res = 0;
            if hist.iters != 0 
                #res = hist[:resnorm][end];
                if verb_compressor == 2
                    plthist = scatterplot(hist[:resnorm], yscale=:log10)
                    display(plthist)
                end
            end
                                   
            converged, residual = assert_convergence!(attempts, hist, loc, d, max_solver_runs; kwargs...);
            
            # Change solver if max attemps are reached
            if t == max_solver_runs && converged == false
                solver = alt_solver;
                xseed = zeros(ComplexF64, size(xseed)...);
                log_message("Changing to alternative solver : $(alt_solver)\n"; color = :blue)
                t = 0; # resetting counts
            end
            
            #xseed = Mnew_d;
        end
        Mnew[:,d,:] = reshape(Mnew_d, size(Mseed_d)...); # build updated site-tensor
    end
    return Mnew; # update tensor in MPS
    #
end

function compress_tensor_lbfgs(bigket::MPS, guessket::MPS, loc::Int; verb_compressor::Int = 0, options_solver::Optim.Options{Float64, Nothing} = Optim.Options())
    # Converts a vector of dim 2L^2 to a complex array of dim LxL
    dims = size(guessket.Ai[loc][:,1,:]);
    vc(v) = reshape(v[1:Int(end/2)] + im*v[Int(end/2)+1:end], dims); 
    #? or
    #= @assert false "Has to be tested"
    if loc != 1 && loc != bigket.L
        vc(v) = reshape(v[1:Int(end/2)] + im*v[Int(end/2)+1:end], (Int(sqrt(length(v)/2)), Int(sqrt(length(v)/2)))); 
    elseif loc == 1
        vc(v) = reshape(v[1:Int(end/2)] + im*v[Int(end/2)+1:end], (1, Int(length(v)/2))); 
    elseif loc == bigket.L
        vc(v) = reshape(v[1:Int(end/2)] + im*v[Int(end/2)+1:end], (1, Int(length(v)/2))); 
    end
     =#

    # Converts a complex vector of dim L to a real vector of dim 2L
    cv(v) = vcat(real.(v), imag.(v));

    
    # Calc environments
    Ltilde, Rtilde = calc_env(guessket, guessket, loc);
    L, R = calc_env(bigket, guessket, loc);
    
    Mnew = zeros(ComplexF64, size(guessket.Ai[loc])); # Allocate array for updated tensor
        
    for d ∈ 1:4 # Sweep physical dims
        M_d = bigket.Ai[loc][:,d,:];
        Mseed_d = guessket.Ai[loc][:,d,:];
        xseed = cv(reshape(Mseed_d, : ,1));
    
        ϵ(Mtilde) = norm(transpose(Ltilde)*(vc(Mtilde)*Rtilde) - transpose(L)*M_d*R); 

        # Solver options
        algorithm = LBFGS(linesearch=HagerZhang());
        # options_solver = Optim.Options(store_trace = true, f_tol = f_tol);

        local Mnew_d;
        converged = false;
        while converged == false
            res = Optim.optimize(ϵ, xseed, algorithm, options_solver, autodiff = :forward);
            
            Mnew_d = res.minimizer;
            xseed = Mnew_d;
            
            converged = Optim.converged(res);
            converged == true && log_message("\nConverged with $(Optim.iterations(res)) iterations and final error $(minimum(res))\n"; color = :green)
            converged == false && log_message("\nNot converged after $(Optim.iterations(res)) iterations and final error $(minimum(res)). Trying again\n"; color = :yellow)
            
            if verb_compressor == 2
                plthist = scatterplot(f_trace(res), yscale=:log10)
                display(plthist)
            end
        end
        #Mnew[:,d,:] = reshape(Mnew_d, size(Mseed_d)...); # build updated site-tensor
        Mnew[:,d,:] = vc(Mnew_d); # build updated site-tensor
    end
    return Mnew
end

function assert_convergence!(
    attempts::Vector{Int64},
    hist, loc::Int,
    d::Int,
    max_solver_runs::Int;
    abstol::Float64=1e-8,
    kwargs...
    )
    #hist.iters != 0 ? (residual = hist[:resnorm][end]) : (residual = 0);
    attempts[1] += 1;

    residual = 1;
    try
        residual = hist[:resnorm][end];
    catch
        log_message("No iterations in last run\n"; color = :red)
    end
    
    hist.isconverged == false && (log_message("\nNot converged for site $(loc) and d_i=$(d), residual ‖Ax-b‖ = $(residual). Trying again\n"; color = :yellow))
    
    hist.isconverged == true && (log_message("\nConverged for site $(loc) and d_i=$(d), residual ‖Ax-b‖ = $(residual).\n"; color = :green))

    if attempts[1] == 2*max_solver_runs && residual > abstol  #! Stops optimization if residual is still too large
        log_message("\n Reached maximum number of attemps, residual are too high!!! Will continue with next tensor anyways \n"; color = :red)
        #! Increase D_max
        return true, residual; 
    elseif attempts[1] == 2*max_solver_runs && residual < abstol
        return true, residual; 
    else
        return hist.isconverged, residual;
    end
end

 =#
############################## OLD ##############################
#= 
function mps_compress(bigket::MPS, guessket::MPS, tol_compr::Float64)
    L = bigket.L;
    ϵ = 2;
    s = 0;
    comprMPS = MPS(copy(guessket.Ai)); #! this is needed as otherwise guessket is modified, which can pass inadvertently
    while abs(ϵ) > tol_compr
        s += 1;
        for n in 1:L # Left-right sweep
            println("Left sweep at site $(n)");# 
            Mnew = compress_tensor(bigket, comprMPS, n, tol_compr/L);
            comprMPS.Ai[n] = Mnew; # update tensor in MPS
        end

        for n in L-1:-1:1 # right-left sweep
            println("Right sweep at site $(n)");# 
            Mnew = compress_tensor(bigket, comprMPS, n,tol_compr/L);
            comprMPS.Ai[n] = Mnew; # update tensor in MPS
            # 
        end

        ϵ = 1 - abs(overlap(bigket, comprMPS));
        println("Sweep : $(s). Error: $(ϵ)")
    end
end =#