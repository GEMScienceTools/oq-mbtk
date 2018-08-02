import os
import h5py

from openquake.mbt.oqt_project import OQtProject


def set_completeness_for_sources(completeness_table, dataset_list):
    """
    :parameter completeness_table:
    :parameter dataset_list:
    """
    #
    # load the project
    project_pickle_filename = os.environ.get('OQMBT_PROJECT')
    oqtkp = OQtProject.load_from_file(project_pickle_filename)
    oqtkp.directory = os.path.dirname(project_pickle_filename)
    model_id = oqtkp.active_model_id
    # MN: 'model' assigned but never used
    model = oqtkp.models[model_id]
    #
    # closing file
    filename = os.path.join(oqtkp.directory, oqtkp.compl_hdf5_filename)
    fhdf5 = h5py.File(filename, 'a')
    print('Updating {:s}'.format(filename))
    #
    # update/create group (i.e. model) containing the completeness table
    if model_id in fhdf5.keys():
        print('    Group {:s} exists'.format(model_id))
        grp = fhdf5[model_id]
    else:
        print('    Creating group: %s' % (model_id))
        grp = fhdf5.create_group(model_id)
    #
    # update/create the dataset containing the completeness table
    for dataset_name in dataset_list:
        if dataset_name in grp:
            del fhdf5[model_id][dataset_name]
            print('    Updating dataset: %s' % (dataset_name))
            # MN: 'dataset' assigned but never used
            dataset = grp.create_dataset(dataset_name, data=completeness_table)
        else:
            print('    Creating dataset: %s' % (dataset_name))
            # MN: 'dataset' assigned but never used
            dataset = grp.create_dataset(dataset_name, data=completeness_table)
    #
    # closing the .hdf5 completeness file
    fhdf5.close()
