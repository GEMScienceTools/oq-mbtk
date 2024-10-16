#!/usr/bin/env julia

using DataFrames
using StatsBase
using CSV
using Dates
using HDF5
using Printf
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "fname"
            help = "name of the .csv file with the input catalogue"
			arg_type = String
			required = true
        "fname_out"
            help = "name of the .csv file with the output catalogue"
			arg_type = String
			required = true
		"--year_start"
		    help = "first year in the catalogue"
			arg_type = Int
			default = -1000
		"--year_end"
		    help = "latest year in the catalogue"
			arg_type = Int
			default = Dates.year(today())
		"--min_mag"
		    help = "minimum magnitude to keep in the catalogue"
			arg_type = Float64
			default = -5.0
		"--max_mag"
		    help = "maximum magnitude to keep in the catalogue"
			arg_type = Float64
			default = 20.0
    end

    return parse_args(s)
end


function prepare(fname, fname_out, minyear, maxyear, minmag, maxmag)

    # Keep events within a geographic area and above a magnitude threshold
    minlo = -360.0
    minla = -91.0
    maxlo = 360.0
    maxla = 91.0

    mindate = Date(minyear, 1, 1)

    # Read data 
    column_types = Dict(:fdepth=>Float32)
    data = DataFrame(CSV.File(fname, types=column_types));

    #show(data_isc_ext, allcols=true, allrows=false)
    # Cleaning eqks without depth
    dropmissing!(data, :depth)

    # Filtering
    filter!(row -> row.latitude >= minla, data);
    filter!(row -> row.latitude <= maxla, data);
    filter!(row -> row.longitude >= minlo, data);
    filter!(row -> row.longitude <= maxlo, data);
    filter!(row -> row.magnitude >= minmag, data);
    filter!(row -> row.magnitude <= maxmag, data);
    filter!(row -> row.year >= minyear, data);
    filter!(row -> row.year <= maxyear, data);

    select!(data, :eventID, :year, :month, :day, :magnitude, :longitude, :latitude, :depth);

    # Save file
    dname = dirname(fname_out)
    if !isdir(dname)
        mkpath(dname)
    end
    CSV.write(fname_out, data);
end


function main()
	
	# Parsing command line args
    args = parse_commandline()

	# Running 
    prepare(args["fname"], args["fname_out"], args["year_start"], 
            args["year_end"], args["min_mag"], args["max_mag"]); 

end


main()
