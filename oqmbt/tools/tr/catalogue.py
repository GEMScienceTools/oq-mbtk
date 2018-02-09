from datetime import datetime
import os
import pickle

from pathlib import Path
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser
from openquake.hmtk.seismicity.selector import CatalogueSelector


def get_catalogue(catalogue_filename):
    """
    """
    ext = Path(catalogue_filename).suffix
    path, name = os.path.split(catalogue_filename)
    cat_pickle_filename = os.path.join(path, Path(name).stem+'.pkl')

    if ext == '.csv' or ext == '.hmtk':
        parser = CsvCatalogueParser(catalogue_filename)
        catalogue = parser.read_file()
        catalogue.sort_catalogue_chronologically()
        sel = CatalogueSelector(catalogue, create_copy=True)
        catalogue = sel.within_time_period(start_time=datetime(1960, 1, 1, 0, 0, 0),end_time=None)
        pickle.dump(catalogue, open(cat_pickle_filename, 'wb'))
    elif ext == '.pkl' or ext == '.p':
        catalogue = pickle.load(open(catalogue_filename, 'rb'))
        catalogue.sort_catalogue_chronologically()
        sel = CatalogueSelector(catalogue, create_copy=True)
        catalogue = sel.within_time_period(start_time=datetime(1960, 1, 1, 0, 0, 0),end_time=None)
    else:
        raise ValueError('File with an unkown extension')
    return catalogue
