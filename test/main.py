import sys
import os
import shutil
import unittest

class TestCases_0(unittest.TestCase):
    def test_0(self):
        os.system('pip install -U morphomics')

        from morphomics import protocols, utils
        import tomli

        parameters_filepath = "./test/Morphomics.Parameters.toml"

        with open(parameters_filepath, mode="rb") as _parameter_file:
            parameters = tomli.load(_parameter_file)

        if parameters["load_previous_instance"]:
            protocol = utils.load_obj("%s/last_instance_%d"%(parameters["path_to_last_instance"], parameters["Parameters_ID"]))
        else:
            protocol = protocols.Protocols(parameters, parameters["Parameters_ID"])

        script_sequence = parameters["Protocols"]
        for sequence in script_sequence:
            print("Doing %s..."%sequence)
            perform_this = getattr(protocol, sequence)
            perform_this()
            if parameters["save_last_instance"]:
                save_path = os.path.join(parameters["path_to_last_instance"], f'last_instance_{parameters["Parameters_ID"]}')
                utils.save_obj(protocol, save_path)

if __name__ == '__main__':
    unittest.main()