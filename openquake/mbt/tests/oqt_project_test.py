import unittest

from openquake.mbt.oqt_project import OQtSource, OQtProject, OQtModel


class TestOQtSource(unittest.TestCase):

    def test_wrong_id(self):
        """
        Check that an error is raised when a wrong source ID is set
        """
        sid = 1
        self.assertRaises(ValueError, OQtSource, sid, 'AreaSource')

    def test_missing_source_type(self):
        """
        Check that an error is raised when a source type is missing
        """
        sid = '1'
        self.assertRaises(ValueError, OQtSource, sid, )

    def test_create_area_source01(self):
        """
        Check the instantiation of an area source (test 1)
        """
        sid = '1'
        stype = 'AreaSource'
        src = OQtSource(sid, stype)
        self.assertEqual(sid, src.source_id)
        self.assertEqual(stype, src.source_type)

    def test_create_area_source02(self):
        """
        Check the instantiation of an area source (test 2)
        """
        sid = '1'
        stype = 'AreaSource'
        src = OQtSource(source_id=sid, source_type=stype)
        self.assertEqual(sid, src.source_id)
        self.assertEqual(stype, src.source_type)


class TestOQtModel(unittest.TestCase):

    def setUp(self):
        project_name = 'Test project'
        project_dir = './'
        prj = OQtProject(project_name, project_dir)
        model_id = 'model01'
        model = OQtModel(model_id=model_id, name='Test model')
        prj.add_model(model)
