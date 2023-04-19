using Crayons
using Dates: now, Time, DateTime
using DrWatson


function log_message(message::String; color::Symbol=:blue, time::Bool=true)
    script_name = PROGRAM_FILE;
    if findlast("/", script_name) !== nothing
        script_name = script_name[findlast("/", script_name)[1]+1:end-3];

        log_name = "logs/$(script_name)_$(Main.JOB_ID).txt";
        log = open(projectdir(log_name), "a");
        time == true && write(log, "\nTime: "*string(Time(now()))[1:8]*"->")
        write(log, message)
        close(log)
        print(Crayon(foreground = color, bold = true), message);
        flush(stdout);
    end
end
