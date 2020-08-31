import os
import copy
import unittest
import nbformat

from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
from nbconvert.exporters import HTMLExporter, export


def run(notebook_filename, inps, reports_folder=None, key=None):
    """
    :parameter notebook_filename:

    :parameter inps:
        A string with parameters replacing the content of the first cell
        of the notebook
    :parameter reports_folder:
        The name of the folder where the report will be created
    """
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    #
    # replacing the content of the first cell
    if len(inps):
        nb['cells'][0]['source'] = inps
    #
    # prepare execution
    ep = ExecutePreprocessor(timeout=100000, kernel_name='python')
    ok = False
    try:
        #
        # returns a 'nb node' and 'resources'
        out = ep.preprocess(nb, {'metadata': {'path': './'}})
        ok = True
    except CellExecutionError as cee:
        msg = 'Error executing the notebook "%s".\n\n' % notebook_filename
        msg += 'See notebook for the traceback.'
        if 'mpl_toolkits.basemap' in cee.traceback:
            raise unittest.SkipTest('Missing basemap')
        else:
            raise
    finally:
        #
        # creating report
        if reports_folder is not None and key is not None:
            #
            # filtering cells
            ocells = []
            for cell in out[0]['cells']:
                if cell['cell_type'] == 'code':
                    cell['source'] = ''
                ocells.append(cell)
            node = copy.deepcopy(out[0])
            node['cells'] = ocells
            #
            # creating the exporter
            # html_exporter = HTMLExporter()
            html_exporter = HTMLExporter(html_exporter='nbextensions.tpl')
            shtml = export(html_exporter, node)
            #
            #
            filename = os.path.join(reports_folder, '%s.html' % key)
            with open(filename, 'w') as f:
                f.write(shtml[0])
            print('Report in {:s}'.format(filename))
            ok = True
    return ok
