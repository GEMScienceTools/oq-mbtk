import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from openquake.baselib import sap


def main(datapath, eventid):
    fin = f'{datapath}/seismicity/srcmod/{eventid}.fsp'
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
    df = pd.DataFrame({'lon': lons, 'lat': lats, 'depth': depths, 'slip': slips})

    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df['lon'], df['lat'])], crs="epsg:4326")
    gdf.to_file(outfi, driver="GeoJSON")

main.datapath = 'path to gem_hazard_data'
main.eventid = 'name of event to convert'

if __name__ == "__main__":
    sap.run(main)
