from ipagraphrag.kg_construction.data.chunk import Chunk


def create_chunk_node_queries(c: Chunk) -> list[str]:
    queries = []
    label = "chunk"
    properties = "text: '{}'".format(c.text)
    chunk_query = f"MERGE (n:{label} {{chunk_pos: {c.pos}, source: '{c.data_source}', {properties}}})"
    queries.append(chunk_query)
    for m in c.mentions:
        label = "mention"
        properties = "text: '{}', type: '{}'".format(m.term, m.type)
        mention_query = f"MERGE (n:{label} {{id: '{m.data_source+'-'+str(m.pos)}', source: '{c.data_source}', {properties}}})"
        queries.append(mention_query)
        label = "has_mention"
        edge_query = (f"MATCH (a {{chunk_pos: {c.pos}, source:'{c.data_source}'}}), "
                      f"(b {{id: '{m.data_source+'-'+str(m.pos)}'}})"
                      f"MERGE (a)-[r:{label}]->(b)")
        queries.append(edge_query)
    return queries


def create_chunk_edge_queries(c1: Chunk, c2: Chunk) -> list[str]:
    queries = []
    label = "next_chunk"
    edge_query = (f"MATCH (a {{chunk_pos: {c1.pos}, source:'{c1.data_source}'}}), "
                  f"(b {{chunk_pos: {c2.pos}, source:'{c2.data_source}'}}) "
                  f"MERGE (a)-[r:{label}]->(b)")
    queries.append(edge_query)
    print(edge_query)
    return queries
