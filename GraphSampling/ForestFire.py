# Based on: https://github.com/Ashish7129/Graph_Sampling/tree/master
# License: https://github.com/Ashish7129/Graph_Sampling/blob/master/LICENSE

import random

import networkx as nx


# New sampling is based on old graph -> start with small samples and increase size stepwise
class ForestFire:
    def __init__(self):
        self.G1 = nx.Graph()

    # graph: Original Graph
    # size: size of the sampled graph
    # p_forward: probability for forward sampling
    # r_backward: ratio
    def sample(self, graph: nx.MultiDiGraph, size: int, p_forward: 0.7, p_backward: 0.2):
        list_nodes = set(graph.nodes())
        burnt_nodes = set()
        random_node = random.sample(list(list_nodes), 1)[0]

        burning_nodes = set()
        burning_nodes.add(random_node)

        while len(self.G1.nodes()) < size:
            if len(burning_nodes) > 0:
                initial_node = burning_nodes.pop()

                if initial_node not in burnt_nodes:
                    burnt_nodes.add(initial_node)

                    out_edges = list(graph.out_edges(initial_node))
                    in_edges = list(graph.in_edges(initial_node))

                    for out_edge in out_edges:
                        if len(self.G1.nodes()) < size:
                            num_attributes = len(graph.adj[out_edge[0]][out_edge[1]])

                            for attribute_idx in range(0, num_attributes):
                                randomness = random.random()
                                if randomness > p_forward:
                                    continue

                                attribute = graph.adj[out_edge[0]][out_edge[1]][attribute_idx]
                                self.G1.add_edge(out_edge[0], out_edge[1], type=attribute['type'])

                                burning_nodes.add(out_edge[1])

                        else:
                            break

                    for in_edge in in_edges:
                        if len(self.G1.nodes()) < size:
                            num_attributes = len(graph.adj[in_edge[0]][in_edge[1]])

                            for attribute_idx in range(0, num_attributes):
                                randomness = random.random()
                                if randomness > p_backward:
                                    continue

                                attribute = graph.adj[in_edge[0]][in_edge[1]][attribute_idx]
                                self.G1.add_edge(in_edge[0], in_edge[1], type=attribute['type'])

                                burning_nodes.add(in_edge[0])

                        else:
                            break

                else:
                    continue

            else:
                random_node = random.sample(list(list_nodes.difference(burnt_nodes)), 1)[0]
                burning_nodes.add(random_node)

        burning_nodes.clear()
        return self.G1
