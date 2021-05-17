import os
import re
import sys
import h5py
# MN: 'shutil' imported but not used
import shutil

from openquake.mbt.tests.tools.tools import delete_and_create_project_dir
from openquake.mbt.oqt_project import OQtProject, OQtModel
from configparser import ConfigParser

HDF5_FILES = ['completeness.hdf5', 'eqk_rates.hdf5',
              'hypo_close_to_faults.hdf5', 'hypo_depths.hdf5']


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def create_default_files(prj, project_dir):
    """
    Creates the basic set of hdf5 files used to store information within a
    project

    :parameter prj:
        An instance of :class:`OQtProject`
    :parameter project_dir:
        The path to the folder where the project is created
    """
    #
    # create the completeness .hdf5 file
    prj.compl_hdf5_filename = 'completeness.hdf5'
    compl_hdf5_filename = os.path.join(project_dir, prj.compl_hdf5_filename)
    fhdf5 = h5py.File(compl_hdf5_filename, 'a')
    fhdf5.close()
    #
    # create the .hdf5 file containing information on eqk rates for the
    # various sources
    prj.eqk_rates_hdf5_filename = 'eqk_rates.hdf5'
    eqk_rates_hdf5_filename = os.path.join(project_dir,
                                           prj.eqk_rates_hdf5_filename)
    fhdf5 = h5py.File(eqk_rates_hdf5_filename, 'a')
    fhdf5.close()
    #
    # create the .hdf5 file with info on eqks close to faults
    prj.hypo_close_to_flts_hdf5_filename = 'hypo_close_to_faults.hdf5'
    hypo_close_to_flts_hdf5_filename = os.path.join(
        project_dir, prj.hypo_close_to_flts_hdf5_filename)
    fhdf5 = h5py.File(hypo_close_to_flts_hdf5_filename, 'a')
    fhdf5.close()
    #
    # create the .hdf5 with hypocentral depth information
    prj.hypo_depths_hdf5_filename = 'hypo_depths.hdf5'
    hypod_hdf5_filename = os.path.join(project_dir,
                                       prj.hypo_depths_hdf5_filename)
    fhdf5 = h5py.File(hypod_hdf5_filename, 'a')
    fhdf5.close()
    # create the folder where to store hypo depth data
    hypo_folder = os.path.join(project_dir, 'hypo_depths')
    if not os.path.exists(hypo_folder):
        os.mkdir(hypo_folder)
    #
    # create the .hdf5 with focal mechanism information
    prj.focal_mech_hdf5_filename = 'focal_mechs.hdf5'
    focm_hdf5_filename = os.path.join(project_dir,
                                      prj.hypo_depths_hdf5_filename)
    fhdf5 = h5py.File(focm_hdf5_filename, 'a')
    fhdf5.close()
    # create the folder where to store focal mechanism data
    focm_folder = os.path.join(project_dir, 'focal_mechs')
    if not os.path.exists(focm_folder):
        os.mkdir(focm_folder)


def load_model_info(config, key):
    """
    Loads into a model the information included in a .ini file.

    :parameter config:
        An instance of :class:`ConfigParser`
    :parameter key:
        The key identifying the section with the information to be loaded
    """
    inimodel = getattr(config._sections, key)
    model = OQtModel(model_id=inimodel.id, name=inimodel.name)
    project_dir = os.path.abspath(config._sections.project.directory)
    #
    # loop over the variables assigned to a model
    for mkey in inimodel.keys():
        if not re.search('^__', mkey):
            if re.search('filename$', mkey):
                #
                # if the key is for a filename we add the path
                tmp = os.path.abspath(getattr(inimodel, mkey))
                path = os.path.relpath(tmp)
                setattr(model, mkey, path)
            else:
                setattr(model, mkey, getattr(inimodel, mkey))
    #
    # updating hdf5 files
    for fname in HDF5_FILES:
        hdf5_filename = os.path.join(project_dir, fname)
        fhdf5 = h5py.File(hdf5_filename, 'a')
        if model.model_id not in fhdf5:
            # MN: 'grp' assigned but never used
            grp = fhdf5.create_group(model.model_id)
        fhdf5.close()
    #
    # add the model to the project
    return model


def add_subfolders(project_dir):
    """
    This creates the subfolders within a project

    :parameter project_dir:
        The path to the directory where folders will be created
    """
    subfolders = ['reports']
    for subfolder in subfolders:
        path = os.path.join(project_dir, subfolder)
        if not os.path.exists(path):
            os.makedirs(path)


def project_create(argv):
    """
    This creates a new `oqmbt` project

    :parameter argv:
        A list. The first argument contains the path to the folder where the
        project will be created. The second parameter is the folder (this
        overrides the `directory` parameter in the `project` section of the
        .ini file
    """
    ini_filename = argv[0]
    print('Reading project information from: \n{:s}'.format(ini_filename))
    assert os.path.exists(ini_filename)
    #
    # reading the .ini file
    config = ConfigParser(dict_type=AttrDict)
    config.read(ini_filename)
    #
    # set project dir and name
    if len(argv) > 1:
        project_dir = argv[1]
        config._sections.project.directory = project_dir
    else:
        project_dir = os.path.abspath(config._sections.project.directory)
    project_name = config._sections.project.name
    #
    # info
    print('Project directory : {:s}'.format((project_dir)))
    print('Project name      : {:s}'.format((project_name)))
    #
    # create a clean project directory
    delete_and_create_project_dir(project_dir)
    #
    # create the project
    prj = OQtProject(project_name, project_dir)
    # MN: 'project_filename' assigned but never used
    project_filename = os.path.join(project_dir, prj._get_filename())
    #
    # create default files
    create_default_files(prj, project_dir)
    #
    # add standard subfolders
    add_subfolders(project_dir)
    #
    # load information for the various models
    for key in config._sections.keys():
        #
        # search for sections containing model information
        if re.search('^model', key):
            model = load_model_info(config, key)
            prj.add_model(model)
    #
    # save the project
    prj.save()


if __name__ == "__main__":
    project_create(sys.argv[1:])
