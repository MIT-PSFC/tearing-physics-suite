# Persistent jGPEC server for TPS: reads run directories (each with a gpec.toml) from stdin, one per line,
# runs GeneralizedPerturbedEquilibrium.main([dir]) with output in dir/jgpec_terminal_output.txt, and replies
# "TPS_DONE ok" or "TPS_DONE err <message>". "TPS_QUIT" stops it. JIT happens once, on the first run.
using Logging
using GeneralizedPerturbedEquilibrium

println("TPS_READY")
flush(stdout)
for line in eachline(stdin)
    dir = strip(line)
    isempty(dir) && continue
    dir == "TPS_QUIT" && break
    status, msg = "ok", ""
    try
        open(joinpath(dir, "jgpec_terminal_output.txt"), "w") do io
            with_logger(ConsoleLogger(io, Logging.Info)) do
                redirect_stdout(io) do
                    redirect_stderr(io) do
                        GeneralizedPerturbedEquilibrium.main([String(dir)])
                    end
                end
            end
        end
    catch e
        status, msg = "err", replace(sprint(showerror, e), r"\s+" => " ")
    end
    println("TPS_DONE $status $msg")
    flush(stdout)
end
