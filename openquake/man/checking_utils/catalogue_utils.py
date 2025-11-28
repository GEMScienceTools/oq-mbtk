import pickle
import pathlib

from openquake.hmtk.parsers.catalogue import CsvCatalogueParser


def load_catalogue(catalogue_fname):
    ext = pathlib.Path(catalogue_fname).suffix
    if ext == '.pkl' or ext == '.p':
        
        # Load pickle file
        cat = pickle.load(open(catalogue_fname, 'rb'))
    elif ext == '.csv' or ext == '.hmtk':
        
        # Load hmtk file
        parser = CsvCatalogueParser(catalogue_fname)
        cat = parser.read_file()
        cat.sort_catalogue_chronologically()
        
    return cat
