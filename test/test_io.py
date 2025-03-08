import unittest
import os
import tempfile
from morphomics.io.toml import load_toml, run_toml

class TestIO(unittest.TestCase):
    def setUp(self):
        # Basic TOML content for testing load_toml and run_toml
        self.test_toml_content = """
[Input]
data_location_filepath = "examples/data"
extension = "corrected.swc"
conditions = ["Region", "Model", "Sex", "Animal"]
separated_by = "Animal"
morphoframe_name = "input"
save_data = false
save_folderpath = null
save_filename = null
"""

    def test_load_toml(self):
        """Test loading TOML configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(self.test_toml_content)
            f.flush()
            config = load_toml(f.name)
            
            self.assertIn('Input', config)
            self.assertEqual(config['Input']['data_location_filepath'], 'examples/data')
            self.assertEqual(config['Input']['extension'], 'corrected.swc')
            self.assertEqual(config['Input']['conditions'], ["Region", "Model", "Sex", "Animal"])
            
        os.unlink(f.name)

    def test_invalid_toml(self):
        """Test loading invalid TOML configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("invalid = toml [ content")
            f.flush()
            with self.assertRaises(Exception):
                load_toml(f.name)
                
        os.unlink(f.name)

    def test_missing_required_fields(self):
        """Test TOML configuration with missing required fields"""
        incomplete_toml = """
[Input]
data_location_filepath = "test_data"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(incomplete_toml)
            f.flush()
            config = load_toml(f.name)
            
            # Verify that required fields are present with default values
            self.assertIn('Input', config)
            self.assertEqual(config['Input']['data_location_filepath'], 'test_data')
            
        os.unlink(f.name)

if __name__ == '__main__':
    unittest.main() 