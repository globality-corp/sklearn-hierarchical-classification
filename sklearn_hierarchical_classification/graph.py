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


def rollup_nodes_1d(graph, source, targets):
    r"""Perform a "roll-up" of given targets (nodes) up to the nodes sitting immediately below
    the given source node in the given (directed) graph.

    For each target, we find all (simple) paths between it and the source node, and then
    accumulate all the immediate child nodes of the source node corresponding to those paths.

    Easily understood by way of example, assuming the following graph:

                        SOURCE
                        /    \
                       A      B
                     /  \  /  |
                    C    D    E

    >>> rollup_nodes(graph, source="SOURCE", targets=["C, "D", "E"])
    [["A"], ["A", "B"], ["B"]]

    Parameters
    ----------
    graph : `networkx.DiGraph` instance
        The graph instance we're performing roll up over

    source : hashable type
        The identifier of the source node in the graph which are performing roll up relative to.

    targets : list
        The list of target labels (y).
        These targets should correspond to node ids in the given graph since we will be using them
        to perform path search on the graph.

    Returns
    -------
    resultset: list-of-lists
        A list-of-lists, with each nested list corresponding to an item in the input `targets` parameter,
        and containing the node ids of the immediate children of the source node which were reached
        as part of enumerating all simple paths from the target to the source node.

    """
    cache = {}
    resultset = []
    for node_id in targets:
        if node_id not in cache:
            cache[node_id] = list(all_simple_paths(G=graph, source=source, target=node_id))
        all_paths = cache[node_id]

        # Note we unique-ify this by creating as a set and then casting to a list.
        resultset.append(list({
            path[1]
            for path in all_paths
        }))

    assert len(resultset) == len(targets)

    return resultset


def rollup_nodes_2d(graph, source, targets):
    r"""Perform a "roll-up" of given targets (nodes) up to the nodes sitting immediately below
    the given source node in the given (directed) graph.

    This function is used for the case where targets are a 2D 0-1 array (matrix), corresponding
    to the format used for multi-label classification, e.g. by sklearn.MultiLabelBinarizer.

    Easily understood by way of example, assuming the following graph:

                        SOURCE
                        /    \
                       A      B
                     /  \  /  |
                    C    D    E

    >>> rollup_nodes(graph, source="SOURCE", targets=["C, "D", "E"])
    [["A"], ["A", "B"], ["B"]]

    Parameters
    ----------
    graph : `networkx.DiGraph` instance
        The graph instance we're performing roll up over

    source : hashable type
        The identifier of the source node in the graph which are performing roll up relative to.

    targets : (sparse) array-like, shape = [n_samples, n_classes]
        These targets should correspond to node ids in the given graph since we will be using them
        to perform path search on the graph.

    Returns
    -------
    resultset: list-of-lists
        A list-of-lists, with each nested list corresponding to an item in the input `targets` parameter,
        and containing the node ids of the immediate children of the source node which were reached
        as part of enumerating all simple paths from the target to the source node.

    """
    cache = {}
    resultset = []
    for row in targets:
        # multi-label binarizer was passed and node_id is an array, perform a roll-up
        # for multi-label targets.
        result_row = []
        for idx in row.nonzero()[0]:
            if idx not in cache:
                cache[idx] = list(all_simple_paths(G=graph, source=source, target=idx))
            all_paths = cache[idx]
            result_row.extend(list({
                path[1]
                for path in all_paths
            }))
        resultset.append(result_row)

    assert len(resultset) == len(targets)

    return resultset


def root_nodes(graph):
    """Return all nodes in graph which are considered as root nodes (having in-degree of zero).

    Parameters
    ----------
    graph : `networkx.DiGraph` instance
        The graph instance we're searching over

    Returns
    -------
    nodes: generator
        Generator expression over graph node objects corresponding to the found root nodes if any

    """
    return (
        node
        for node, in_degree in graph.in_degree()
        if in_degree == 0
    )


def terminal_nodes(graph):
    """Return all nodes in graph which are considered as terminal nodes (having out-degree of zero).

    Parameters
    ----------
    graph : `networkx.DiGraph` instance
        The graph instance we're searching over

    Returns
    -------
    nodes: generator
        Generator expression over graph node objects corresponding to the found terminal nodes if any

    """
    return (
        node
        for node, out_degree in graph.out_degree()
        if out_degree == 0
    )
