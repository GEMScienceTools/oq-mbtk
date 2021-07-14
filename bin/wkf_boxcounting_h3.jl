#!/usr/bin/env julia

using Dates
using ArgParse
using PSHAModelBuilder

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "cat"
            help = "name of the .csv file with the catalogue"
			arg_type = String
			required = true
        "mapping"
            help = "name of the .csv file with the mapping i.e. link between h3 point and zone"
			arg_type = String
			required = true
        "config"
            help = "name of the .toml configuration file"
			arg_type = String
            required = true
	    "h3_level"
	        help = "h3 level i.e. the grid resolution"
			arg_type = Int64
	        required = true
		"folder_out"
		    help = "Folder where to save output files"
			arg_type = String
		    required = true
		"--year_end", "-y"
		    help = "latest year in the catalogue"
			arg_type = Int
			default = Dates.year(today())
		"--weighting", "-w"
            help = "weighting option for events [one, mfd, completeness]"
			arg_type = String
			default = "one"
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

	# Running boxcounting
	PSHAModelBuilder.boxcounting(args["cat"], args["h3_level"], 
                args["mapping"], args["config"], 
	            args["folder_out"], args["year_end"], args["weighting"]);

end

main()
