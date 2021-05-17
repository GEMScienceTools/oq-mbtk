#!/usr/bin/env julia

using ArgParse
using PSHAModelBuilder


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "count"
            help = "name of the .csv file produced by box counting"
			arg_type = String
			required = true
        "config"
            help = "name of the .toml configuration file"
			arg_type = String
            required = true
		"fname_out"
		    help = "Folder where to save output files"
			arg_type = String
		    required = true
    end

    return parse_args(s)
end


function main()
	
	# Parsing command line args
    args = parse_commandline()

	# Creating output folder
	
	tmp = splitdir(args["fname_out"])
	if !isdir(tmp[1])
		mkpath(tmp[1])
	end

	# Running smoothing
	PSHAModelBuilder.smoothing(args["count"], args["config"], args["fname_out"])

end

main()