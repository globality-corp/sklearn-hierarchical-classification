"""
Graph processing helpers.

"""
from collections import defaultdict

from networkx import all_simple_paths
from numpy import ndarray


def make_flat_hierarchy(targets, root):
    """Create a trivial (flat) hiearchy, linking all given targets to given root node."""
    adjacency = defaultdict(list)
    for target in targets:
        adjacency[root].append(target)
    return adjacency


def rollup_nodes(graph, source, targets, mlb=None):
    """Perform a "roll-up" of given target nodes up to the nodes immediately below
    given source node in given graph.

    """
    result_cache = {}
    resultset = []
    for node_id in targets:
        if mlb and type(node_id) == ndarray:
            # multi-label binarizer was passed and node_id is an array, perform a roll-up
            # for multi-label targets.
            result_row = []
            for label in node_id.nonzero()[0]:
                if label not in result_cache:
                    result_cache[label] = list(all_simple_paths(G=graph, source=source, target=mlb.classes_[label]))
                all_paths = result_cache[label]
                result_row.extend([
                    path[1]
                    for path in all_paths
                ])
            resultset.append(result_row)
        else:
            if node_id not in result_cache:
                result_cache[node_id] = list(all_simple_paths(G=graph, source=source, target=node_id))
            all_paths = result_cache[node_id]
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
