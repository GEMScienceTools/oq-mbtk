#!/usr/bin/env python
# coding: utf-8

import os
import h3
import pandas as pd
from openquake.wkf.utils import create_folder
from openquake.baselib import sap


def main(h3_mapping: str, h3_level: int, folder_out: str):

    # Reading the input
    df = pd.read_csv(h3_mapping, names=['key', 'sid'])
    df.head()

    # Create the output folder - If needed
    create_folder(folder_out)

    # Preparing the dataframe
    lons = []
    lats = []
    for i, row in df.iterrows():
        la, lo = h3.cell_to_latlng(row.key)
        lons.append(lo)
        lats.append(la)
    df['lon'] = lons
    df['lat'] = lats
    df['nocc'] = 1.

    # Writing output
    for sid in df.sid.unique():
        if isinstance(sid, str):
            tmps = '{:s}.csv'.format(sid[0:3])
        else:
            tmps = '{:02d}.csv'.format(sid)
        fname_out = os.path.join(folder_out, tmps)
        print(fname_out)
        tdf = df.loc[df.sid == sid]
        tdf.to_csv(fname_out, columns=['lon', 'lat', 'nocc'], index=False)


descr = 'The .csv file containing the mapping between h3 cells and zones'
main.h3_mapping = descr
descr = 'The h3 level used to discretize the zones'
main.h3_level = descr
descr = 'The name of the folder where to save output'
main.folder_out = descr

if __name__ == '__main__':
    sap.run(main)
