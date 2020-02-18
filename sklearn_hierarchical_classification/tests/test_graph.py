"""
Unit-tests for the graph module.

"""
from hamcrest import assert_that, equal_to, is_
from networkx import DiGraph
from parameterized import parameterized

from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.graph import rollup_nodes


def make_simple_dag():
    """Create a simple DAG graph for testing purposes.
    Graph would look like this:

                 ROOT
                /    \
               A      B
             /  \  /  |
            C    D    E

    """
    graph = DiGraph()
    graph.add_edge(ROOT, "A")
    graph.add_edge("A", "C")
    graph.add_edge("A", "D")
    graph.add_edge(ROOT, "B")
    graph.add_edge("B", "D")
    graph.add_edge("B", "E")

    return graph


@parameterized([
    (["C", "D", "E"], [["A"], ["A", "B"], ["B"]]),
])
def test_rollup_nodes(targets, expected):
    graph = make_simple_dag()

    result = rollup_nodes(graph=graph, source=ROOT, targets=targets)

    assert_that(result, is_(equal_to(expected)))
