"""
Graph processing helpers.

"""
import logging
from collections import defaultdict

from networkx import all_simple_paths


def make_flat_hierarchy(targets, root):
    """
    Create a trivial "flat" hiearchy, linking all given targets to an artificial ROOT node.

    """
    adjacency = defaultdict(list)
    for target in targets:
        adjacency[root].append(target)
    return adjacency


def rollup_nodes(graph, root, targets):
    """
    Perform a "roll-up" of given target nodes up to the nodes immediately below
    given root node in given graph.

    """
    resultset = []
    for node_id in targets:
        all_paths = all_simple_paths(G=graph, source=root, target=node_id)
        # XXX need to generalize training logic to support DAG - rolling up
        # a rolled up node may result in multiple nodes, e.g rolling up D to A:
        #
        #        B
        #      /   \
        #    A - C - D
        #
        # for path in all_paths:
        #    resultset.append(path[1])
        try:
            resultset.append(next(all_paths)[1])
        except StopIteration:
            logging.error(
                "Could not find path from root='{}' to node_id='{}', targets: {}".format(
                    root,
                    node_id,
                    targets,
                ),
            )
            raise

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
