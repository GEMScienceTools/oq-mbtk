import os
import re
# MN: 'sys' imported but not used
import sys
import glob

# MN: 'OQtModel' imported but not used
from openquake.mbt.oqt_project import OQtProject, OQtModel


def clean_project_component(model_id, source_type):
    """
    :param model_id:
    :param source_type:
    """
    #
    # load project
    project_pickle_filename = os.environ.get('OQMBT_PROJECT')
    oqtkp = OQtProject.load_from_file(project_pickle_filename)
    print('Project: {:s}'.format(oqtkp.name))
    #
    # set model
    model = oqtkp.models[model_id]
    #
    # delete filename
    if hasattr(model, 'faults_shp_filename'):
        del model.faults_shp_filename
    # directories:
    # - reports
    # - focal_mech
    # - hypo_depths
    # - nrml
    folder_reports = os.path.join(oqtkp.directory, 'reports')
    #
    # hdf5 files:
    # - completeness.hdf5
    # - eqk_rates.hdf5
    # - focal_mechanisms.hdf5
    # - hypo_close_to_faults.hdf5 - NOT USED
    # - hypo_depths.hdf5
    # - <model>_hypo_dist.hdf5
    # - <model>_nodal_plane_dist.hdf5

    #
    # deleting sources
    keys = list(model.sources.keys())
    for key in keys:
        stype = model.sources[key].source_type
        if stype == source_type:
            del model.sources[key]
            #
            # delete reports
            _delete_reports(folder_reports, key)
            #
            # remove information about faults
            if source_type == 'SimpleFaultSource':
                for src_key in model.sources:
                    if (model.sources[src_key].source_type == 'AreaSource' and
                            hasattr(model.sources[src_key],
                                    'ids_faults_inside')):
                        del model.sources[src_key].ids_faults_inside
    #
    # saving model and project
    oqtkp.models[model_id] = model
    oqtkp.save()


def _delete_reports(foldername, src_id):
    src_id = re.sub('[a-z]', '', src_id)
    foldername = os.path.join(foldername, '*{:s}*'.format(src_id))
    for filename in glob.glob(foldername):
        tmp = re.split('-', re.split('\\.', os.path.basename(filename))[0])
        print(tmp[-1])
        if tmp[-1] == src_id:
            print('----', filename)
