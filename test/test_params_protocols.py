import sys
import os
import shutil
import unittest
from morphomics import pipeline
from morphomics.io import toml
from morphomics import utils
import tempfile

class TestToml(unittest.TestCase):
    def setUp(self):
        self.test_params = {
            'Input': {
                'data_location_filepath': 'test_data',
                'extension': 'corrected.swc',
                'conditions': ['Region', 'Model', 'Sex', 'Animal'],
                'separated_by': None,
                'morphoframe_name': "input",
                'save_data': False,
                'save_folderpath': None,
                'save_filename': None
            },
            'Protocols': ['Input', 'Process', 'Analyze']
        }

    def test_toml_load_and_run(self):
        """Test loading and running TOML configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(self.test_params, f)
            f.flush()
            
            loaded_params = toml.load_toml(f.name)
            self.assertEqual(loaded_params['Input'], self.test_params['Input'])
            self.assertEqual(loaded_params['Protocols'], self.test_params['Protocols'])
            
            # Test running the pipeline with the loaded parameters
            protocol = pipeline.Pipeline(parameters=loaded_params, Parameters_ID='test')
            self.assertEqual(protocol.Parameters_ID, 'test')
            
        os.unlink(f.name)

class TestProtocols(unittest.TestCase):
    def setUp(self):
        self.test_params = {
            'Input': {
                'data_location_filepath': 'test_data',
                'extension': 'corrected.swc',
                'conditions': ['Region', 'Model', 'Sex', 'Animal'],
                'separated_by': None,
                'morphoframe_name': "input",
                'save_data': False,
                'save_folderpath': None,
                'save_filename': None
            },
            'Protocols': ['Input']
        }
        
        # Create test data directory and files
        os.makedirs('test_data', exist_ok=True)
        self.test_files = [
            'Region1_Model1_Sex1_Animal1.corrected.swc',
            'Region1_Model1_Sex1_Animal2.corrected.swc'
        ]
        for file in self.test_files:
            with open(os.path.join('test_data', file), 'w') as f:
                f.write("# Test SWC file\n")

    def tearDown(self):
        # Clean up test data
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')

    def test_protocol_execution(self):
        """Test protocol execution sequence"""
        protocol = pipeline.Pipeline(parameters=self.test_params, Parameters_ID='test')
        
        # Test Input protocol
        protocol.Input()
        self.assertTrue(hasattr(protocol, 'morphoframe'))
        self.assertEqual(protocol.morphoframe.name, "input")

    def test_save_last_instance(self):
        """Test saving last instance functionality"""
        save_path = 'test_save'
        os.makedirs(save_path, exist_ok=True)
        
        params = self.test_params.copy()
        params['save_last_instance'] = True
        params['path_to_last_instance'] = save_path
        
        protocol = pipeline.Pipeline(parameters=params, Parameters_ID='test')
        protocol.Input()
        
        save_file = os.path.join(save_path, f'last_instance_{params["Parameters_ID"]}')
        self.assertTrue(os.path.exists(save_file))
        
        # Clean up
        shutil.rmtree(save_path)

if __name__ == '__main__':
    unittest.main()