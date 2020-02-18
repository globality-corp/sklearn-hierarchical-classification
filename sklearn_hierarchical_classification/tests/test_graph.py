"""
Unit-tests for the graph module.

"""
from hamcrest import assert_that, equal_to, is_
from networkx import DiGraph
from parameterized import parameterized

from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.graph import rollup_nodes_1d, rollup_nodes_2d


def make_simple_dag():
    r"""Create a  DAG graph for testing purposes.
    Graph would look like this:

                 ROOT
                /    \
               A      B
             /  \  /  |  \
            C    D    E   |
          / |    |    | \ |
         F  G    H    I   J

    """
    graph = DiGraph()
    graph.add_edge(ROOT, "A")
    graph.add_edge("A", "C")
    graph.add_edge("A", "D")
    graph.add_edge(ROOT, "B")
    graph.add_edge("B", "D")
    graph.add_edge("B", "E")
    graph.add_edge("B", "J")
    graph.add_edge("C", "F")
    graph.add_edge("C", "G")
    graph.add_edge("D", "H")
    graph.add_edge("E", "I")
    graph.add_edge("E", "J")

    return graph


@parameterized([
    # targets, expected
    (["C", "D", "E"], [["A"], ["A", "B"], ["B"]]),
    (["F", "H", "I"], [["A"], ["A", "B"], ["B"]]),
    (["F", "H", "I", "J"], [["A"], ["A", "B"], ["B"], ["B"]]),
])
def test_rollup_nodes_1d(targets, expected):
    graph = make_simple_dag()

    result = rollup_nodes_1d(graph=graph, source=ROOT, targets=targets)

    # For comparison purposes below we ensure each item in result is sorted.
    result = [sorted(lst) for lst in result]

    assert_that(result, is_(equal_to(expected)))


@parameterized([
    # targets, expected
    # TODO
])
def test_rollup_nodes_2d(targets, expected):
    graph = make_simple_dag()

    result = rollup_nodes_2d(graph=graph, source=ROOT, targets=targets)

    assert_that(result, is_(equal_to(expected)))
