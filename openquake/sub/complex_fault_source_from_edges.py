#!/usr/bin/env python

import os
import sys


from openquake.baselib import sap
from openquake.sub.edges_set import EdgesSet

from openquake.hazardlib.sourcewriter import write_source_model


def complex_fault_src_from_edges(edges_folder, out_nrml='source.xml'):
    """
    :param edges_folder:
    :param out_nrml:
    """
    #
    # check edges folder
    assert os.path.exists(edges_folder)
    #
    #
    es = EdgesSet.from_files(edges_folder)
    src = es.get_complex_fault()
    print(out_nrml)
    write_source_model(out_nrml, [src], 'Name')


def runner(argv):

    p = sap.Script(complex_fault_src_from_edges)
    p.arg(name='edges_folder', help='Name of the folder containing the edges')
    p.arg(name='out_nrml', help='Name of the output file')

    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()


if __name__ == "__main__":
    runner(sys.argv[1:])
