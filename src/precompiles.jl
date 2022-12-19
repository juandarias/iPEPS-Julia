function _precompile_()

    pattern = ['a' 'b'; 'c' 'd'];
    uc = UnitCell{Float64}(2, (2, 2), pattern, UNDEF);
    initialize_environment!(uc);
    projectors = Projectors{EachMove}(uc);
    do_ctmrg_iteration!(uc, projectors);
end
