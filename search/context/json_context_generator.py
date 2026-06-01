from networkx import Graph

from search.context.context_generator import ContextGenerator
from networkx.readwrite import json_graph

class JSONContextGenerator(ContextGenerator):


    def __init__(self):
        super().__init__()

    def generate_context(self, graph_list:list[Graph], node_properties:set, edge_properties:set):
        context_list = []
        for g in graph_list:
            copied_graph = g.copy()
            for n, d in copied_graph.nodes(data=True):
                props = set(d.keys())
                props.difference_update(node_properties)
                for p  in props:
                    d.pop(p, None)  # Remove unwanted attribute
            for u, v, d in copied_graph.edges(data=True):
                props = set(d.keys())
                props.difference_update(edge_properties)
                for p in props:
                    d.pop(p, None)  # Remove unwanted attribute
            json_data = json_graph.node_link_data(copied_graph)
            context_list.append(json_data)
        return context_list