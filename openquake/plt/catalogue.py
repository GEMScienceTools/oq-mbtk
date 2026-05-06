# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2026 GEM Foundation and Électricité de France
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This script is produced within the scope of Work Package 5, named Simulation 
# platform, under SIGMA3 project. For more detailed information about 
# the project, please visit to <https://sigma-programs.com/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

"""
module :mod:`openquake.plt.catalogue` provides functions for a seismicity map using 
Generic Mapping Tools (GMT).
"""

import os
import subprocess
import pandas as pd
import geopandas as gpd

def prepare_geometry(geojson_path: str, output_xy: str = "polygon.xy"):
    """
    Converts a GeoJSON boundary file to a GMT-compatible XY format.
    """
    gdf = gpd.read_file(geojson_path)
    with open(output_xy, "w") as f:
        for geom in gdf.geometry:
            if geom.geom_type == "Polygon":
                coords = [geom.exterior.coords]
            elif geom.geom_type == "MultiPolygon":
                coords = [p.exterior.coords for p in geom.geoms]
            
            for c in coords:
                for lon, lat in c:
                    f.write(f"{lon} {lat}\n")
                f.write(">\n")
    return output_xy

def split_catalogue_dynamic(df: pd.DataFrame, bins: list):
    """
    Splits the catalogue into dynamic magnitude bins provided by the user.
    """
    temp_files = []
    for i, bin_info in enumerate(bins):
        low = bin_info["min_mag"]
        high = bin_info.get("max_mag", None)
        fname = f"m{i+1}.xyz"
        
        if high is not None:
            subset = df[(df["magnitude"] >= low) & (df["magnitude"] < high)]
        else:
            subset = df[df["magnitude"] >= low]
        
        subset[['longitude', 'latitude']].to_csv(fname, sep=" ", header=False, index=False)
        temp_files.append(fname)
    return temp_files

def visualize_catalogue(catalogue: str, 
                        polygon: str, 
                        region: list, 
                        output_png: str, 
                        bins: list = None):
    """
    Generates a seismicity map using Generic Mapping Tools (GMT).
    
    Args:
        catalogue (str): Path to the HMTK-formatted CSV catalogue.
        polygon (str): Path to the GeoJSON file for the study area boundary.
        region (list): Map extent [West, East, South, North].
        output_png (str): Name of the output image file.
        bins (list): List of dictionaries containing magnitude bin parameters.
    """
    # Default magnitude bins and formats
    if bins is None:
        bins = [
            {"min_mag": 3.5, "max_mag": 4.5, "color": "blue", "size": "0.08c", "label": "3.5-4.5"},
            {"min_mag": 4.5, "max_mag": 5.5, "color": "green", "size": "0.12c", "label": "4.5-5.5"},
            {"min_mag": 5.5, "max_mag": 6.5, "color": "orange", "size": "0.18c", "label": "5.5-6.5"},
            {"min_mag": 6.5, "max_mag": None, "color": "red", "size": "0.25c", "label": "> 6.5"}
        ]

    df = pd.read_csv(catalogue)
    df.columns = [c.lower() for c in df.columns]
    
    xy_boundary = prepare_geometry(polygon)
    temp_xyz_files = split_catalogue_dynamic(df, bins)
    
    west, east, south, north = region
    map_name = output_png.replace('.png', '')

    # Legends
    legend_lines = ""
    for i, b in enumerate(bins):
        fname = f"m{i+1}.xyz"
        color_val = f"{b['color']}@40"
        
        if b['max_mag'] is not None:
            label_text = f"{b['min_mag']} <= M < {b['max_mag']}"
        else:
            label_text = f"M >= {b['min_mag']}"
            
        legend_lines += f"S 0.2c c {b['size']} {color_val} 0.1p,{b['color']} 0.5c {label_text}\n"

    # GMT SCRIPT
    gmt_script = f"""#!/usr/bin/env bash
rm -rf ~/.gmt/sessions/*

gmt begin {map_name} png
    gmt set FONT_ANNOT_PRIMARY 8p,Helvetica
    gmt basemap -R{west}/{east}/{south}/{north} -JQ6i -Bxa4f2 -Bya4f2 -BWSen
    
    # Boundary fill
    gmt plot {xy_boundary} -W0.5p,gray40 -Ggray80@60
    
"""
    
    for i, b in enumerate(bins):
        fname = f"m{i+1}.xyz"
        gmt_script += f"    [ -s {fname} ] && gmt plot {fname} -Sc{b['size']} -G{b['color']}@40 -W0.1p,{b['color']}\n"

    gmt_script += f"""
    gmt coast -W0.2p,black -N1/0.5p,black,dash -Df                  # Geographic features
    gmt legend -DjBL+o0.3c/0.3c+w3.0c -F+p0.5p+gwhite@30 << EOF     # Legend
H 10p,Helvetica-Bold Magnitude (M)
{legend_lines}EOF
gmt end
"""
    
    with open("temp_plot.sh", "w") as f:
        f.write(gmt_script)
    
    try:
        subprocess.run(["bash", "temp_plot.sh"], check=True, env=os.environ)
        print(f"Done! {output_png} generated.")
    except subprocess.CalledProcessError as e:
        print(f"GMT Error: The script failed. Check if GMT is installed correctly.")
        raise e
    finally:
        for f in temp_xyz_files + [xy_boundary, "temp_plot.sh"]:
            if os.path.exists(f):
                os.remove(f)

