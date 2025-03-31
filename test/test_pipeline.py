import unittest
import os
import tempfile
from morphomics.pipeline import Pipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.test_data_location = "test_data"
        self.test_extension = "corrected.swc"
        self.test_conditions = ['Region', 'Model', 'Sex', 'Animal']
        
        # Create basic test parameters
        self.basic_params = {
            'Input': {
                'data_location_filepath': self.test_data_location,
                'extension': self.test_extension,
                'conditions': self.test_conditions,
                'separated_by': None,
                'morphoframe_name': "input",
                'save_data': False,
                'save_folderpath': None,
                'save_filename': None
            }
        }

    def test_pipeline_initialization(self):
        """Test basic pipeline initialization"""
        pipeline = Pipeline(parameters=self.basic_params, Parameters_ID='test')
        self.assertEqual(pipeline.Parameters_ID, 'test')
        self.assertEqual(pipeline.parameters, self.basic_params)

    def test_pipeline_with_invalid_params(self):
        """Test pipeline initialization with invalid parameters"""
        invalid_params = {
            'Input': {
                'wrong_parameter': 'value'
            }
        }
        with self.assertRaises(ValueError):
            Pipeline(parameters=invalid_params, Parameters_ID='test')

    def test_pipeline_with_save_data(self):
        """Test pipeline with save_data enabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            params = self.basic_params.copy()
            params['Input']['save_data'] = True
            params['Input']['save_folderpath'] = tmpdir
            params['Input']['save_filename'] = 'test_save'
            
            pipeline = Pipeline(parameters=params, Parameters_ID='test')
            self.assertTrue(pipeline.parameters['Input']['save_data'])
            self.assertEqual(pipeline.parameters['Input']['save_folderpath'], tmpdir)

    def test_separated_by_parameter(self):
        """Test pipeline with separated_by parameter"""
        params = self.basic_params.copy()
        params['Input']['separated_by'] = 'Animal'
        
        pipeline = Pipeline(parameters=params, Parameters_ID='test')
        self.assertEqual(pipeline.parameters['Input']['separated_by'], 'Animal')

if __name__ == '__main__':
    unittest.main() 