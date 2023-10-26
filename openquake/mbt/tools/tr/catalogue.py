import os
import pickle

from pathlib import Path
from openquake.hmtk.parsers.catalogue import CsvCatalogueParser
# MN: 'CatalogueSelector' imported but not used
from openquake.hmtk.seismicity.selector import CatalogueSelector

from openquake.hmtk.parsers.catalogue.gcmt_ndk_parser import ParseNDKtoGCMT


def get_catalogue(catalogue_filename, force_csv=False):
    """
    """

    ext = Path(catalogue_filename).suffix
    path, name = os.path.split(catalogue_filename)
    cat_pickle_filename = os.path.join(path, Path(name).stem+'.pkl')

    if (ext == '.csv' or ext == '.hmtk') or (force_csv):
        parser = CsvCatalogueParser(catalogue_filename)
        catalogue = parser.read_file()
        pickle.dump(catalogue, open(cat_pickle_filename, 'wb'))
    elif ext in ['.pkl', '.p', '.pickle']:
        catalogue = pickle.load(open(catalogue_filename, 'rb'))
    elif ext == '.ndk':
        parser = ParseNDKtoGCMT(catalogue_filename)
        catalogue = parser.read_file()
        pickle.dump(catalogue, open(cat_pickle_filename, 'wb'))
    else:
        raise ValueError('File with an unkown extension')
    return catalogue
