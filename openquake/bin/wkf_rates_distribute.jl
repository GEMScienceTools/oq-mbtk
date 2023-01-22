#!/usr/bin/env julia

using Glob
using ArgParse
using Printf
using PSHAModelBuilder
using TOML


"""
    distribute_rates(folder_smooth, fname_config, folder_out, [eps_b, eps_rate])

Distributes the rates using the output of a seismicity smoothing function. 
The `folder_smooth` contains the output of a smoothing algorithm i.e. .csv 
files with three columns consisting of longitude, latitude and 

# Examples
```julia-repl
julia> distribute_rates('./smooth', './conf.toml', './out')
1
```
"""
function distribute_rates(folder_smooth::String, fname_config::String, folder_out::String, eps_b::Float64=0.0, eps_rate::Float64=0.0)

    # Parse the configuration file
    config = TOML.parsefile(fname_config)
    
    # Loop over the zones
    pattern = joinpath(folder_smooth, "*.csv")
    for tmps in glob(pattern)

        fname = splitdir(tmps)[2]
        source_id = split(fname, '.')[1]
        
        # bgr
        σb = 0.0
        if haskey(config["sources"][source_id], "bgr_sig")
            σb = config["sources"][source_id]["bgr_sig"]
        end
        bgr = config["sources"][source_id]["bgr"] + σb * eps_b

        # If for the current source the a value of the GR exists in the 
        # configuration file and the epsilon is larger than 0
        if haskey(config["sources"][source_id], "rmag")

            # This adds support for the uncertainty on the rate
            rmag = config["sources"][source_id]["rmag"]
            λ_rmag = config["sources"][source_id]["rmag_rate"]
            λ_rmag_sig = config["sources"][source_id]["rmag_rate_sig"]

            # This is the total agr
            agr = log10((λ_rmag + eps_rate * λ_rmag_sig) / (10^(-bgr*rmag)))

        else

            # In this case we do not support uncertainty
            if eps_rate != 0.0
                error("eps_rate must be equal to 0 since rmag is not defined")
            end

            # This is the original behaviour
            agr = config["sources"][source_id]["agr"]

        end

        fname_out = joinpath(folder_out, @sprintf("%s.csv", source_id))
        PSHAModelBuilder.distribute_total_rates(agr, bgr, tmps, fname_out)
    end
        
end

"""
function distribute_rates(folder_smooth::String, fname_config::String, folder_out::String, eps_a::Float64=0.0, eps_b::Float64=0.0)

    # Parse the configuration file
    config = TOML.parsefile(fname_config)
    
    # Loop over the zones
    pattern = joinpath(folder_smooth, "*.csv")
    for tmps in glob(pattern)
        fname = splitdir(tmps)[2]
        source_id = split(fname, '.')[1]
        σa = config["sources"][source_id]["agr_sig"]
        σb = config["sources"][source_id]["bgr_sig"]
        agr = config["sources"][source_id]["agr"] + σa * eps_a
        bgr = config["sources"][source_id]["bgr"] + σb * eps_b
        fname_out = joinpath(folder_out, @sprintf("%s.csv", source_id))
        PSHAModelBuilder.distribute_total_rates(agr, bgr, tmps, fname_out)
    end
        
end
"""

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "smooth_folder"
            help = "name of the .csv file produced by box counting"
            arg_type = String
            required = true
        "config"
            help = "name of the .toml configuration file"
            arg_type = String
            required = true
        "folder_out"
            help = "folder where to save output files"
            arg_type = String
            required = true
        "--eps_rate", "-r"
            help = "epsilon for the rate above m_ref"
            arg_type = Float64
            default = 0.0
        "--eps_b", "-b"
            help = "epsilon for bgr"
            arg_type = Float64
            default = 0.0
    end

    return parse_args(s)
end


function main()
    
    # Parsing command line args
    args = parse_commandline()

    # Creating output folder
    if !isdir(args["folder_out"])
        mkpath(args["folder_out"])
    end

    # Running smoothing
    distribute_rates(args["smooth_folder"], args["config"], args["folder_out"], args["eps_b"], args["eps_rate"])

end

main()
