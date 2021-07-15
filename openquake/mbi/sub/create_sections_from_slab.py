#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from openquake.baselib import sap
from openquake.sub.cross_sections import CrossSection, Slab2pt0


def main(fname_slab: str, pname_out: str, fname_cross_sections: str, *,
         bffer: float = 10):

    # Read the traces of the cross-sections
    names = ['olo', 'ola', 'sde', 'sle', 'azi', 'num', 'ini']
    df = pd.read_csv(fname_cross_sections, names=names, sep='\\s+')
    css = []
    for i, l in df.iterrows():
        num = '{:03d}'.format(int(l.num))
        css.append(CrossSection(l.olo, l.ola, l.sle, l.azi, num))

    # Create and save profiles
    slb = Slab2pt0.from_file(fname_slab, css)
    slb.compute_profiles(bffer)
    slb.write_profiles(pname_out)


descr = 'Name of the Slab2.0 file with the geometry of the top of the slab'
main.fname_slab = descr
main.pname_out = 'Name of the folder where to store the profiles'
descr = 'Name of the file with the traces of the cross-sections'
main.fname_cross_sections = descr
main.bffer = 'Distance [km] to select the slab points close to a cross-section'

if __name__ == '__main__':
    sap.run(main)
