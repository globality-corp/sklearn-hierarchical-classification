"""
Hamcrest matchers for unit-tests.

"""
from hamcrest.core.base_matcher import BaseMatcher
from networkx import info, is_isomorphic


class GraphMatcher(BaseMatcher):

    def __init__(self, graph):
        self.graph = graph

    def _matches(self, graph):
        return is_isomorphic(graph, self.graph)

    def describe_to(self, description):
        description.append_text("graph ").append_text(info(self.graph))


def matches_graph(graph):
    return GraphMatcher(graph)
