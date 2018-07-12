#!/usr/bin/env python

import os
import sys
import numpy
import pickle

from openquake.hmtk.seismicity.catalogue import Catalogue
from openquake.hmtk.parsers.catalogue.csv_catalogue_parser import CsvCatalogueParser
from openquake.hmtk.seismicity.selector import CatalogueSelector

def main(argv):

    filename = argv[0]
    parser = CsvCatalogueParser(filename)
    cat = parser.read_file()

    output_path = './catalogue_ori.p'
    fou = open(output_path,'wb')
    pickle.dump(cat, fou)
    fou.close()

    lomin = -180
    lomax = +180
    lamin = -90
    lamax = +90

    if len(argv) > 1:
        lomin = float(argv[1])
    if len(argv) > 2:
        lomax = float(argv[2])
    if len(argv) > 3:
        lamin = float(argv[3])
    if len(argv) > 4:
        lamax = float(argv[4])

    idxo = numpy.nonzero((cat.data['longitude'] >= lomin) &
                         (cat.data['longitude'] <= lomax) &
                         (cat.data['latitude'] >= lamin) &
                         (cat.data['latitude'] <= lamax))
    idxs = idxo[0].astype(int)

    boo = numpy.zeros_like(cat.data['magnitude'], dtype=int)
    boo[idxs] = 1

    selector = CatalogueSelector(cat, create_copy=True)
    newcat = selector.select_catalogue(boo)

    output_path = './catalogue_ext.p'
    fou = open(output_path,'wb')
    pickle.dump(newcat, fou)
    fou.close()

if __name__ == "__main__":
    main(sys.argv[1:])
