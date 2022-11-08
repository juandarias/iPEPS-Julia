module argparser

    export collect_args, get_param!

    function collect_args(arguments)
        args_dict = Dict();
        for i=1:length(arguments)
            indexequal = findfirst(isequal('='), arguments[i])
            if indexequal != nothing
                label = arguments[i][2:indexequal-1]
                value = arguments[i][indexequal+1:end]
                args_dict[label] = value;
            end
        end
        return args_dict
    end

    function get_param!(dict::Dict, label::String, def_value::T) where {T}
        value = get!(dict,label,def_value)
        if typeof(value) == String && T != String
            return parse(T, value);
        else
            return value;
        end
    end
    
end