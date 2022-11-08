module fidelities

"=================================="
#### TODO
# ---------------------------------
#
#### Notes
# -unitary operator basis is obtained from generalized pauli matrices (see wiki reference), in particular shift and clock matrices as defined by silvester.
# -verified calculations of Process Fidelity (13/03/2020)
# -ProcessFidelity and Silvester basis method could be parallelized
# -check if GC and resetting unitaries to 0.0 is effective
#### Bugs
# -FIXED: lower Process Fidelities obtained if dimFS = 8*k or 8*k + 1
#### References
# Zhu, PRL 97(050505), 2006; Kim, PRL 103(120502), 2009; Sorensen, PRA 62(022311), 2000; Nielsen, PLA 303(249), 2002
# https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices
#----------------------------------
"=================================="



"================"
# Packages and modules
"================"

    using LinearAlgebra, SparseArrays, QuantumInformation, Distributed #Arpack


"================"
# Export
"================"

    export SylvesterBasis, ProcessFidelity, ProcessFidelityBasis, HaarThermalFidelity, QuantumChannel, StateFidelity, FidelitySector, OperatorDensity;

"================"
# Definitions and functions
"================"

    ⊗ = kron
    generalX(dims) = spdiagm(-1=>ones(dims-1),dims-1=>[1+0.0*im]); #shift matrix
    generalZ(dims) = spdiagm(0=>[round(exp(im*2*pi*(j/dims)),digits=10) for j in 0:dims-1]); #clock matrix


"================"
# Methods
"================"

    function SylvesterBasis(dimensions::Int64) #Normalized basis
        XX = generalX(dimensions);
        ZZ = generalZ(dimensions);
        return [(1/sqrt(dimensions))*(XX^m*ZZ^n) for n in 0:dimensions-1, m in 0:dimensions-1]
    end

    function SylvesterBasis(dimensions::Int64, index::Tuple{Int64,Int64}) #Normalized basis
        XX = generalX(dimensions);
        ZZ = generalZ(dimensions);
        BE = (1/sqrt(dimensions))*(XX^index[1]*ZZ^index[2])
        return BE
    end


    function OperatorDensity(operator::T, sites_out::Array{Int64}) where T<:SparseMatrixCSC{Complex{Float64},Int64}
        dims_tot = size(gate)[1];
        dims_out = length(sites_out);
        dims_red = dims_tot - dims_out;
        W_red = ptrace(operator, [dims_red, dims_out], [2]);
        
        ρ_b = 0.0
        mn_pairs = collect(Iterators.product(0:dims_red-1,0:dims_red-1));
        if parallel == false
            for (j,mn) in enumerate(mn_pairs)
                Uj = SylvesterBasis(dims, mn);
                ρ_b += abs(tr(W_red*Uj));
            end
        elseif parallel == true
            ρ_b = @distributed (+) for mn in mn_pairs
                Uj = SylvesterBasis(dims, mn);
                abs(tr(W_red*Uj));
            end
        end    #GC.gc()
        return ρ_b
    end

    function QuantumChannel(operator_basis::T, unitary::T; n_quanta::Int=0) where T<:SparseMatrixCSC{Complex{Float64},Int64};
        dimSS = Int64(size(operator_basis)[1])
        dimFS = Int64(size(unitary)[1]/dimSS);
        #n_quanta =0; #assuming ground state. It might be more precise to use a coherent state or thermal state.
        ρ_n = spdiagm(0=>insert!(zeros(dimFS-1),n_quanta+1,1));
        ρ_i = ρ_n ⊗ operator_basis
        #ρ_i = kron(operator_basis, ρ_n)
        #ρ_f = unitary*ρ_i*unitary'
        return sparse(ptrace(Array(unitary*ρ_i*unitary'), [dimFS, dimSS], [1]));#tensor contractions ptrace definition. Last index is index to trace out
    end


    function ProcessFidelity(gate::T, unitary::T; parallel=false) where T<:SparseMatrixCSC{Complex{Float64},Int64}
        dims = size(gate)[1]; Fidelity = 0.0 + 0.0*im;
        mn_pairs = collect(Iterators.product(0:dims-1,0:dims-1));
        if parallel == false
            SumTr = 0.0 +0.0*im;
            for mn in mn_pairs
                Uj = SylvesterBasis(dims, mn);
                Ub = gate*Uj'*gate'*QuantumChannel(Uj, unitary);
                SumTr += tr(Ub);
                Uj=0.0;Ub=0.0;
            end
            Fidelity = (1/((dims+1)*dims^2))*(dims*SumTr + dims^2)
            #GC.gc()
            return Fidelity
        end
        if parallel == true
            SumTr = @distributed (+) for mn in mn_pairs
                Uj = SylvesterBasis(dims, mn);
                Ub=gate*Uj'*gate'*QuantumChannel(Uj, unitary);
                trUb=tr(Ub)
                Uj=0.0;Ub=0.0;
                trUb
            end
            Fidelity = (1/((dims+1)*dims^2))*(dims*SumTr + dims^2)
            GC.gc()
            return Fidelity
        end
    end


    function ProcessFidelityBasis(gate::T, unitary::T; basis::B, parallel=false) where T<:SparseMatrixCSC{Complex{Float64},Int64} where B<:Array{SparseMatrixCSC{Complex{Float64},Int64},2}
        dims = size(gate)[1]; Fidelity = 0.0 + 0.0*im;
        ONB = basis;
        parallel == false && (Fidelity = (1/((dims+1)*dims^2))*(dims*sum([tr(gate*Uj'*gate'*QuantumChannel(Uj, unitary)) for Uj in ONB])+dims^2));
        if parallel == true
            SumTr = @distributed (+) for Uj in ONB
                tr(gate*Uj'*gate'*QuantumChannel(Uj, unitary))
            end
            Fidelity = (1/((dims+1)*dims^2))*(dims*SumTr + dims^2)
            GC.gc()
        end
        return Fidelity
    end



    function FidelitySector(Uideal::Array, Uexp::Array, mode::Int, Nions::Int)
        sectorU(U,mode)=U[1+(mode-1)*2^Nions:mode*2^Nions,1+(mode-1)*2^Nions:mode*2^Nions]
        Uideal_sector = sectorU(Uideal,mode)
        Uexp_sector = sectorU(Uexp,mode)
        return 1/(2^Nions)*abs(tr(Uideal_sector*adjoint(Uexp_sector)))
    end


    function StateFidelity(channel::SparseMatrixCSC{Complex{Float64},Int64}
, gate::SparseMatrixCSC{Complex{Float64},Int64}
, n_qbits::Int, dim_fock::Int, haar_states::Array{Array{Complex{Float64},1},1}
, n_quanta::Int)
        global Fidelity = []
        for ket_haar in haar_states
            #ket_haar = rand(haar);
            ket_n = insert!(zeros(dim_fock-1),n_quanta+1,1);
            full_rho = kron(ket_n,ket_haar)*(kron(ket_n,ket_haar)')
            Fidelity += 1/(2^n_qbits)*abs(tr((gate*full_rho)*adjoint(channel*full_rho)))
        end
        return Fidelity/length(haar_states)
    end

    function HaarFidelity(channel::SparseMatrixCSC{Complex{Float64},Int64}
, gate::SparseMatrixCSC{Complex{Float64},Int64}
, n_qbits::Int, dim_fock::Int, haar_states::Array{Array{Complex{Float64},1},1}
, n_quanta::Int)
        global Fidelity = 0.0
        for ket_haar in haar_states
            #ket_haar = rand(haar);
            ket_n = insert!(zeros(dim_fock-1),n_quanta+1,1);
            Ψ = kron(ket_n,ket_haar)
            Fidelity += ((gate*Ψ)')*(channel*Ψ)*((channel*Ψ)')*(gate*Ψ)
        end
        return Fidelity/length(haar_states)
    end


    function HaarThermalFidelity(channel, gate, n_qbits, dim_fock, haar_states, nbar)
        global ThermalFidelity = 0.0
        for n in 0:dim_fock-1
            ThermalFidelity += nbar^n/((1+nbar)^(n+1))*HaarFidelity(channel, gate, n_qbits, dim_fock, haar_states, n)
        end
        return ThermalFidelity
    end

    function ThermalFidelityError(range_nbar, range_g, fidelity_arrays)
        TF = zeros(length(range_nbar),length(range_g))
        fidelity_arrays = real.(fidelity_arrays)
        for i in 1:length(range_nbar)
            for j in 1:length(range_g)
                range_nbar[i] == 0 && (TF[i,j] = fidelity_arrays[range_g[j],1])
                range_nbar[i] != 0 && (TF[i,j] = sum([(range_nbar[i]^n/((1+range_nbar[i])^(n+1)))*fidelity_arrays[range_g[j],n+1] for n in 0:5]))
            end
        end
        return abs(-TF.+1)
    end




end  # module fidelities


"======="
# Old
"======="
    #
    #
    # function ProcessFidelityOld(gate::T, unitary::T; parallel=false) where T<:SparseMatrixCSC{Complex{Float64},Int64}
    #     dims = size(gate)[1]; Fidelity = 0.0 + 0.0*im;
    #     ONB = SylvesterBasis(dims);
    #     parallel == false && (Fidelity = (1/((dims+1)*dims^2))*(dims*sum([tr(gate*Uj'*gate'*QuantumChannel(Uj, unitary)) for Uj in ONB])+dims^2));
    #     if parallel == true
    #         SumTr = @distributed (+) for Uj in ONB #memory "leak"
    #             tr(gate*Uj'*gate'*QuantumChannel(Uj, unitary))
    #         end
    #         Fidelity = (1/((dims+1)*dims^2))*(dims*SumTr + dims^2)
    #         GC.gc()
    #     end
    #     GC.gc()
    #     return Fidelity
    # end
