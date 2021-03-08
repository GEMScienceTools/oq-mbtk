#!/usr/bin/env julia

using DataFrames
using StatsBase
using CSV
using Dates
using HDF5
using Printf

using ArgParse
using PSHAModelBuilder


function extract(fname_in, fname_out, depth_min, depth_max, mag_min, mag_max)

    # Read input file - The catalogue should follow the oq-hmtk standard
    data = DataFrame(CSV.File(fname_in));

    # Filter
    filter!(row -> row.depth <= depth_max, data);
    filter!(row -> row.depth > depth_min, data);
    filter!(row -> row.magnitude <= mag_max, data);
    filter!(row -> row.magnitude >= mag_min, data);
    
    data.month[data.month .> 12] .= 12 
    data.month[data.month .< 1] .= 1;
    data.day[data.day .< 1] .= 1;
    data.day[data.day .> 12] .= 12;

    # Save file
    println("Saving ", fname_out);
    CSV.write(fname_out, data);

end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "fname_in"
            help = "name of the .csv file containing the original catalogue"
			arg_type = String
			required = true
        "fname_out"
            help = "name of output .csv file"
			arg_type = String
            required = true
		"--depth_min", "-n"
		    help = "Minimum hypocentral depth"
			arg_type = Float64
            default = 0.0
		"--depth_max", "-x"
		    help = "Maximum hypocentral depth"
			arg_type = Float64
            default = 600.0
		"--mag_min", "-l"
		    help = "Minimum magnitude"
			arg_type = Float64
            default = -1.0
		"--mag_max", "-u"
		    help = "Maximum magnitude"
			arg_type = Float64
            default = 11.0
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
	extract(args["fname_in"], args["fname_out"], args["depth_min"], 
            args["depth_max"], args["mag_min"], args["mag_max"])

end

main()
