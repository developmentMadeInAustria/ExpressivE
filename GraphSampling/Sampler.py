import sys
import getopt

import pandas as pd
import networkx as nx

from ForestFire import ForestFire

if __name__ == '__main__':

    input_file = ""
    output_file = ""
    size = None
    p_forward = 0.7
    p_backward = 0.2

    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:s:", ["input_file=", "output_file=", "size=", "p_forward=", "p_backward="])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input_file"):
            input_file = arg
        if opt in ("-o", "--output_file"):
            output_file = arg
        elif opt in ("-s", "--size"):
            size = int(arg)
        elif opt == "--p_forward":
            p_forward = float(arg)
        elif opt == "--p_backward":
            p_backward = float(arg)

    if input_file == "":
        raise ValueError("Did not specify input file!")

    if output_file == "":
        raise ValueError("Did not specify output file!")

    if size is None or size <= 0:
        raise ValueError("Size must be specified and greater than 0!")

    edges = pd.read_csv(input_file, names=["source", "type", "target"], sep="\t")

    full_graph = nx.MultiDiGraph()
    edges.apply(lambda edge: full_graph.add_edge(edge["source"], edge["target"], type=edge["type"]), axis=1)

    forest_fire = ForestFire()
    sampled_graph = forest_fire.sample(full_graph, size, p_forward, p_backward)

    sampled_edges = nx.to_pandas_edgelist(sampled_graph)
    sampled_edges.to_csv(output_file, sep='\t', columns=["source", "type", "target"], header=False, index=False)
