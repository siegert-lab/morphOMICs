from morphomics import pipeline
import tomli

def load_toml(parameters_filepath):
    # read the toml file
    with open(parameters_filepath, mode="rb") as _parameter_file:
        parameters = tomli.load(_parameter_file)
    return parameters

def run_toml(parameters, morphoframe = {}):
    my_pipeline = pipeline.Pipeline(parameters, parameters["Parameters_ID"], morphoframe)

    # run the protocols in row
    protocol_list = parameters["Protocols"]
    for protocol in protocol_list:
        print("Doing %s..."%protocol)
        perform_this = getattr(my_pipeline, protocol)
        perform_this()
    
    return my_pipeline