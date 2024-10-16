import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from openquake.baselib import sap


def main(fippath):#, eventid):
    """
    Converts an event from the sourcemod database of the format
    .fsp to a geojson file

    Note: first download events from http://equake-rc.info/srcmod/ 

    Example:

        oqm sub srcmod_to_json srcmod_events/s2013SCOTIA01HAYE.fsp

    """
    fin = f'{fippath}'
    root = fin.split('/')[-1].replace('.fsp','')
    outfi = f'{root}.geojson'

    file1 = open(fin, 'r')
    Lines = file1.readlines()

    lons, lats, depths, slips = [],[],[],[]
    for line in Lines:
        if (line[0] != '%') & (line != ' \n'):
            parts = line.strip().split()
            lons.append(parts[1]); lats.append(parts[0])
            depths.append(float(parts[4])); slips.append(float(parts[5]))
    df = pd.DataFrame({'lon': lons, 'lat': lats,
                       'depth': depths, 'slip': slips})

    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in \
            zip(df['lon'], df['lat'])], crs="epsg:4326")
    gdf.to_file(outfi, driver="GeoJSON")
    print(f'Written to {outfi}')

main.fippath = 'path to .fsp file'

if __name__ == "__main__":
    sap.run(main)
