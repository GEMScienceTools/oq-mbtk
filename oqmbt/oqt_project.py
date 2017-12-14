"""
Module :mod:`oqmbt.oqt_project` defines :class:`OQtProject`, :class:`OQtModel`,
and :class:`OQtSource`. A :class:`OQtProject` instance is a container for
the inofrmation needed to create a PSHA source model.
"""

import re
import os
import warnings
import inspect
try:
    import cPickle as pickle
except:
    import pickle
import openquake.hazardlib.source as oqsrc

from copy import deepcopy


# Default paramters for a model
DEFAULTS_PRJ = {'name': 'Unknown project',
                'csv_catalogues': {},
                'directory': '',
                'models': {},
                'active_model_id': None
                }

# Default paramters for a model
DEFAULTS_MOD = {'name': 'Unknown model',
                'model_id': '',
                'sources': {}
                }

# List of valid attributes for an area source
AREAS_ATTRIBUTES = set(['source_id', 'name', 'tectonic_region_type', 'mfd',
                        'rupture_mesh_spacing',
                        'magnitude_scaling_relationship',
                        'rupture_aspect_ratio', 'temporal_occurrence_model',
                        'upper_seismogenic_depth', 'lower_seismogenic_depth',
                        'nodal_plane_distribution', 'hypocenter_distribution',
                        'polygon', 'area_discretization'])
AREAS_ATTRIBUTES |= set(['gr_aval', 'gr_bval', 'source_type'])

# List of valid attributes for a simple source
SIMPLE_FAULT_ATTRIBUTES = set(['source_id', 'name', 'tectonic_region_type',
                               'mfd', 'rupture_mesh_spacing',
                               'magnitude_scaling_relationship',
                               'rupture_aspect_ratio',
                               'temporal_occurrence_model',
                               'upper_seismogenic_depth',
                               'lower_seismogenic_depth', 'fault_trace',
                               'dip', 'rake', 'hypo_list', 'slip_list'])
SIMPLE_FAULT_ATTRIBUTES |= set(['gr_aval', 'gr_bval', 'source_type'])

# Create the set of valid source types
SOURCE_TYPES = set()
for name, obj in inspect.getmembers(oqsrc):
    if inspect.isclass(obj):
        if not re.search('Rupture', name):
            SOURCE_TYPES.add(name)


class OQtSource(object):
    """
    A container for information necessary to build and/or characterise an
    earthquake source

    :parameter str source_id:
        The ID of the source
    :parameter str source_type:
        Source type i.e. Object name amongst the ones admitted in the
        OpenQuake Hazardlib.

    """
    def __init__(self, *args, **kwargs):
        # Checks
        if len(args):
            self.source_id = args[0]
            if len(args) > 1:
                self.source_type = args[1]
        if len(kwargs):
            self.__dict__.update(kwargs)
        # Check mandatory attributes
        if 'source_id' not in self.__dict__:
            raise ValueError('Source must have an ID')
        elif not isinstance(self.source_id, str):
            raise ValueError('ID must be a string')
        #
        if 'source_type' not in self.__dict__:
            raise ValueError('Source must have a type')
        if self.source_type not in SOURCE_TYPES:
            raise ValueError('Unrecognized source type: %s' % self.source_type)
        # Find the set
        if 'source_type' in self.__dict__:
            attribute_set = AREAS_ATTRIBUTES
        elif 'source_type' in self.__dict__:
            attribute_set = SIMPLE_FAULT_ATTRIBUTES
        else:
            raise ValueError('Unsupported source type')
        # Check attributes
        for key in self.__dict__:
            if key not in attribute_set:
                print ('Attribute set {:s}'.format(attribute_set))
                msg = 'Parameter %s not compatible with this source' % (key)
                raise ValueError(msg)

    def get_info(self):
        for key in self.__dict__:
            print ('%30s:' % (key), getattr(self, key))


class OQtProject(object):
    """
    A container for all the information needed and produced to create a
    source model.

    :parameter str name:
        Name of the projects
    :parameter str project_directory:
        Name of the directory that will contain the project
    :parameter dict source_models:
        A dictionary of source models. Keys of the dictionary are the
        models IDs.
    """
    def __init__(self, *args, **kwargs):
        # Set defaults
        self.__dict__.update(DEFAULTS_PRJ)
        # Set args
        if len(args):
            self.name = args[0]
            self.directory = args[1]
        # Set kwargs
        if len(kwargs):
            self.__dict__.update(kwargs)
        # Check if model exists
        filename = self._get_filename()
        output_path = os.path.join(self.directory, filename)
        if os.path.exists(output_path):
            msg = 'File exists! Change project name or delete project folder'
            raise ValueError(msg)

    def add_model(self, model):
        """
        :param model:
            An instance of `class`:oqt_project.OQtModel
        """
        assert isinstance(model, OQtModel)
        self.models[model.model_id] = (model)
        print ('Model \'%s\' added to project' % (model.model_id))

    def _get_filename(self):
        filename = re.sub('\s', '_', self.name).lower()
        filename += '.oqmbtp'
        return filename

    def save(self, log=False):
        """
        :parameter str output_dir:
            Output directory
        """
        #
        # set filename
        filename = self._get_filename()
        #
        # save pickle file
        output_path = os.path.join(self.directory, filename)
        #
        # delete old file
        if os.path.exists(output_path):
            os.remove(output_path)
        #
        # write new project file
        fou = open(output_path, 'wb')
        pickle.dump(self, fou)
        fou.close()
        if log:
            print ('Project saved into file %s' % (output_path))

    @classmethod
    def load_from_file(cls, filename):
        """
        :parameter str filename:
            The name of the file containing the toolkit project
        :returns:
            An instance of :py:class:`~oqmbt.oqt_project.OQtProject`
        """
        if not os.path.exists(filename):
            print (filename)
            warnings.warn("File does not exist", UserWarning)
            return None
        else:
            fin = open(filename, 'rb')
            prj = pickle.load(fin)
            fin.close()
            return prj

    def activate_model(self, model_id, **kwargs):
        """
        parameter str model_id:
            The id of a source model
        """
        if model_id in self.models:
            self.active_model_id = model_id
            print ('\nActivating model already available in the project')
        else:
            self.add_model(OQtModel(model_id=model_id))
            print ('   Initialising model not available in the project')
            if len(kwargs):
                self.__dict__.update(kwargs)

    def get_info(self):
        print ('\nInformation for project: %s' % (self.name))
        print ('\n  -- parameters:')
        for key in self.__dict__:
            print ('   %-40s:' % (key), getattr(self, key))
        print ('\n  -- models:')
        if len(self.models):
            for key in self.models:
                print ('   %-40s: %2d sources' % (key,
                                                 len(self.models[key].sources)))
        else:
            print ('   %-30s' % ('None'))


class OQtModel(object):
    """
    :parameter str model_id:
        Model ID
    :parameter str name:
        Model name
    :parameter dict sources:
        A dictionary where key is source_id and value is an instance of the
        :py:class:`~oqt_project.OQtSource`.
    :parameter str catalogue_csv_filename:
        Relative pats to the catalogue used for the analysis.
    :parameter str area_shapefile_filename:
        Relative paths to the shapefile with geometry of the area sources
    """
    def __init__(self, *args, **kwargs):
        # Set defaults
        self.__dict__.update(DEFAULTS_MOD)
        # Set parameters
        if len(args):
            self.model_id = args[0]
            self.name = args[1]
        if len(args) > 2:
            assert isinstance(args[2], dict)
            self.sources = args[2]
        # Set kwargs
        if len(kwargs):
            self.__dict__.update(kwargs)
        # Check that IDs are unique
        # if not len(set(self.sources.keys)) == len(self.sources.keys):
        #    raise ValueError('The list of sources contains non-unique IDs')

    def clean_sources(self, source_type=None):
        """
        :parameter source_type:
            A string
        """
        tdic = deepcopy(self.sources)
        if source_type is 'All':
            self.sources = {}
        elif source_type in set(['AreaSource',
                                 'SimpleFaultSource',
                                 'ComplexFaultSource']):
            for key in self.sources:
                src = self.sources[key]
                if src.source_type == source_type:
                    del tdic[key]
        else:
            raise ValueError('Unrecognized source type')
        self.sources = tdic

    def add_source(self, source, skip=False):
        """
        Add a source to the model

        :parameter source:
            An instance
        :parameter skip:
            Boolean, when true makes the process more flexible i.e. if we
            attenpt to add a source with the same ID of the source we're
            about we skip instead of raising an error
        """
        assert isinstance(source, OQtSource)
        sid = source.source_id
        if sid in self.sources and not skip:
            msg = 'This model already contains a source with ID %s' % (sid)
            raise ValueError(msg)
        else:
            if sid in self.sources:
                msg = 'Skipping source with ID %s ' % (sid)
                msg += 'since already included'
                print (msg)
            else:
                self.sources[sid] = source
                print ('Adding source: %s' % (sid))

    def update_source(self, source):
        """
        Updates a source already included in the model

        :parameter source:
            An instance of :py:class:`~oqmbt.oqt_project.OQtSource`.
        """
        assert isinstance(source, OQtSource)
        if source.source_id is not self.sources:
            sid = source.source_id
            msg = 'This model does not contain a source with ID %s' % (sid)
            raise ValueError(msg)
        else:
            self.sources[source.source_id] = source
            print ('Updating source: %s' % (source.source_id))

    def get_info(self):
        """
        Prints info on the project content
        """
        print ('\nInformation for model: %s' % (self.name))
        for key in sorted(self.__dict__):
            if key in ['sources', 'nrml_sources']:
                pass
            else:
                print ('   %-40s:' % (key), getattr(self, key))

    def get_source(self, source_id):
        """
        """
        if source_id in self.sources:
            return self.sources[source_id]
        else:
            return None

    def get_area_source_ids(self):
        ids = []
        for key in self.sources:
            src = self.sources[key]
            if src.source_type == 'AreaSource':
                ids.append(key)
        return sorted(ids)
