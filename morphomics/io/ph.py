##
# Part to save PHs in .txt files.
##

# def write_ph(ph, output_file="test.txt"):
#     """Writes a persistence diagram in an output file."""
#     with open(output_file, "w", encoding="utf-8") as wfile:
#         for p in ph:
#             wfile.write(str(p[0]) + " " + str(p[1]) + "\n")


# def extract_ph(tree, feature="radial_distance", output_file="test.txt", sort=False, **kwargs):
#     """Extracts persistent homology from tree."""
#     ph = get_persistence_diagram(tree, feature=feature, **kwargs)

#     if sort:
#         p = sort_ph(ph)
#     else:
#         p = ph

#     write_ph(p, output_file)


# def extract_ph_neuron(
#     neuron, feature="radial_distance", output_file=None, neurite_type="all", sort=False, **kwargs
# ):
#     """Extracts persistent homology from tree."""
#     ph = get_ph_neuron(neuron, feature=feature, neurite_type="all", **kwargs)

#     if sort:
#         sort_ph(ph)

#     if output_file is None:
#         output_file = "PH_" + neuron.name + "_" + neurite_type + ".txt"

#     write_ph(ph, output_file)