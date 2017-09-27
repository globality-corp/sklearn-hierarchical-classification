"""
Graph processing helpers.

"""
from collections import defaultdict

from networkx import all_simple_paths


def make_flat_hierarchy(targets, root):
    """
    Create a trivial (flat) hiearchy, linking all given targets to given root node.

    """
    adjacency = defaultdict(list)
    for target in targets:
        adjacency[root].append(target)
    return adjacency


def rollup_nodes(graph, source, targets):
    """
    Perform a "roll-up" of given target nodes up to the nodes immediately below
    given source node in given graph.

    """
    resultset = []
    for node_id in targets:
        all_paths = all_simple_paths(G=graph, source=source, target=node_id)
        resultset.append([
            path[1]
            for path in all_paths
        ])

    assert len(resultset) == len(targets)

    return resultset


def root_nodes(graph):
    return (
        node
        for node, in_degree in graph.in_degree()
        if in_degree == 0
    )


def terminal_nodes(graph):
    return (
        node
        for node, out_degree in graph.out_degree()
        if out_degree == 0
    )
