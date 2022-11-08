using Crayons
using Dates: now, Time, DateTime


function log_message(message::String; color::Symbol=:green, time::Bool=true)
    script_name = PROGRAM_FILE;
    script_name = script_name[findlast("/", script_name)[1]+1:end-3];

    log_name = "logs/$(script_name)_$(Main.JOB_ID).txt";
    log = open(datadir(log_name), "a");
    time == true && write(log, "\nTime: "*string(Time(now()))[1:8]*"->")
    write(log, message)
    close(log)
    print(Crayon(foreground = color, bold = true), message);
    flush(stdout);
end