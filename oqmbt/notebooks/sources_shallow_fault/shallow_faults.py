# coding: utf-8

import os
import sys
import re

from openquake.man.model import read
from openquake.hazardlib.source import SimpleFaultSource
from oqmbt.oqt_project import OQtProject, OQtSource
from oqmbt.mfd_tools import get_moment_from_mfd


def read_faults(faults_xml_filename=None):
    """
    Reads the information on faults included in an .xml file

    :parameter faults_xml_filename:
        The name of the .xml file with the faults
    """
    #
    # loading project
    project_pickle_filename = os.environ.get('OQMBT_PROJECT')
    oqtkp = OQtProject.load_from_file(project_pickle_filename)
    model_id = oqtkp.active_model_id
    model = oqtkp.models[model_id]
    if faults_xml_filename is None:
        faults_xml_filename = os.path.join(oqtkp.directory,
                                           getattr(model,'faults_xml_filename'))
    #
    # read .xml file content
    sources, _ = read(faults_xml_filename)
    #
    # save the information
    for f in sources:
        #
        # fixing the id of the fault source
        sid = str(f.source_id)
        if not re.search('^fs_', sid):
            sid = 'fs_{:s}'.format(sid)
        if isinstance(f, SimpleFaultSource):
            src = OQtSource(sid, 'SimpleFaultSource')
            src.trace = f.fault_trace
            src.msr = f.magnitude_scaling_relationship
            src.mfd = f.mfd
            src.rupture_aspect_ratio = f.rupture_aspect_ratio
            src.trt = f.tectonic_region_type
            src.dip = f.dip
            src.upper_seismogenic_depth = f.upper_seismogenic_depth
            src.lower_seismogenic_depth = f.lower_seismogenic_depth
            src.name = f.name
            src.rake = f.rake
            model.sources[sid] = src
        else:
            raise ValueError('Unsupported fault type')
    #
    # save the project
    oqtkp.models[model_id] = model
    oqtkp.save()

if __name__ == "__main__":
    main()
