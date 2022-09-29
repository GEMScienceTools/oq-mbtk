#!/usr/bin/env julia

# This is a command-line version of the accompanying jupyter notebook.

using ArgParse
using NetCDF
using Printf
using TOML


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "config"
            help = "name of the .toml configuration file"
			arg_type = String
            required = true
        "fname_sites"
            help = "name of the .csv file with the coordinates of the sites"
			arg_type = String
            required = true
        "fname_out"
            help = "name of the output .csv file with the site model"
			arg_type = String
            required = true
    end

    return parse_args(s)
end

function get_index(vec, value)
    dff = vec .- value
    return findmin(abs.(dff))
end

function get_sites(ncfile)
    lons = Float64[]
    lats = Float64[]
    file=open(ncfile, "r")
    c = 0
    while(!eof(file))
        line = readline(file)
        svec = split(line, ",")
        if !all(isletter, svec[1])
            push!(lons, parse(Float64, svec[1]))
            push!(lats, parse(Float64, svec[2])) 
            c += 1
        end
    end
    return lons, lats
end

function get_vs30(slons, slats, lons, lats, vs30)
    vals = Float64[]
    for (lo, la) in zip(slons, slats)
        idxx = get_index(lons, lo)
        idxy = get_index(lats, la)
        push!(vals, vs30[idxx[2], idxy[2]])
    end
    return vals
end

function get_z1pt0(vs30)
    c1 = 571^4.
    c2 = 1360.0^4.
    out = exp.((-7.15 / 4.0) * log.((vs30.^4. .+ c1) ./ (c2 + c1)))
    return out
end

function get_z2pt5(vs30)
    c1 = 7.089
    c2 = -1.144
    out = exp.(c1 .+ log.(vs30) .* c2)
    return out
end

      
function write_site_file(ncfile, lons, lats, vs30)
    z1pt0 = get_z1pt0(vs30)
    z2pt5 = get_z2pt5(vs30)
    f = open(ncfile, "w")
    write(f, "lon,lat,vs30,z1pt0,z2pt5,vs30measured\n")
    c = 0
    for (lo, la, vs, z1, z2) in zip(lons, lats, vs30, z1pt0, z2pt5)
        tmp = @sprintf("%.6f,%.6f,%.2f,%.2f,%.2f,%d\n", lo, la, vs, z1, z2, 0)
        write(f, tmp)
        c += 1
    end
    close(f)
end


function main()

    args = parse_commandline()

    conf = TOML.parsefile(args["config"])
    ncfile = conf["site_model"]["ncfile"]

    # info = ncinfo(ncfile);
    # println(info)

    xs = ncread(ncfile, "x");
    ys = ncread(ncfile, "y");
    zs = ncread(ncfile, "z"); 

    fname_in = args["fname_sites"]
    lons, lats = get_sites(fname_in)
    vs30 = get_vs30(lons, lats, xs, ys, zs);

    if size(vs30) != size(lons)
        error("Size of Vs30 different from lon, lat size")
    end

    fname_out = args["fname_out"]
    write_site_file(fname_out, lons, lats, vs30)

end

main()
