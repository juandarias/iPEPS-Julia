############################## Types ##############################

export COMPRESSOR
export SIMPLE, SVD, VAR_CG, VAR_OPTIM
export ALGORITHM
export CG, GMRES, LBFGS


@enum COMPRESSOR begin
    SVD = 1 #* Uses canonical forms of MPS to do a single-site optimization/compression of the MPS.
    SIMPLE = 2
    VAR_CG = 3
    VAR_OPTIM = 4
end

@enum ALGORITHM begin
    CG = 1
    GMRES = 2
    L_BFGS = 3
end


