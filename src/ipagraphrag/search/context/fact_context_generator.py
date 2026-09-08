import networkx as nx
from networkx import Graph

from ipagraphrag.search.context.context_generator import ContextGenerator


class FactContextGenerator(ContextGenerator):

    def __init__(self):
        super().__init__()

    def generate_context(self, graph_list:list[Graph], node_properties:set, edge_properties:set):
        context_list = []
        for g in graph_list:
            facts = ""
            for u, v, d in g.edges(data=True):
                u_node_data = g.nodes[u]
                v_node_data = g.nodes[v]
                u_properties = ', '.join([f"{p}: '{u_node_data[p]}'" for p in node_properties if p in u_node_data])
                v_properties = ', '.join([f"{p}: '{v_node_data[p]}'" for p in node_properties if p in v_node_data])
                u_node = f"({u}:{", ".join(u_node_data["key"])} {{{u_properties}}})"
                v_node = f"({v}:{", ".join(v_node_data["key"])} {{{v_properties}}})"
                facts += u_node + f"-[:{d["key"]}]-"+v_node +","
            context_list.append(facts)
        return context_list
