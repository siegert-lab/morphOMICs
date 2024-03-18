from morphomics import protocols, utils
import tomli
import sys

parameters_filepath = sys.argv[1]
Parameters_ID = int(parameters_filepath.split('/')[-1].split('.')[-2])

with open(parameters_filepath, mode="rb") as _parameter_file:
    parameters = tomli.load(_parameter_file)
parameters["Parameters_ID"] = Parameters_ID

if parameters["load_previous_instance"]:
    protocol = utils.load_obj("%s/last_instance_%d"%(parameters["path_to_last_instance"], parameters["Parameters_ID"]))
else:
    protocol = protocols.Protocols(parameters, Parameters_ID)

script_sequence = parameters["Protocols"]
for sequence in script_sequence:
    print("Doing %s..."%sequence)
    perform_this = getattr(protocol, sequence)
    perform_this()
    if parameters["save_last_instance"]:
        utils.save_obj(protocol, "%s/last_instance_%d"%(parameters["path_to_last_instance"], parameters["Parameters_ID"]))