"""Relation schema used by extraction and graph-build. Single source of truth."""

# Extraction-time allowed relation types (neo4j-graphrag schema + prompt).
EXTRACTION_RELATIONS = [
    "TREATS", "PREVENTS", "CAUSES", "PREDISPOSES",
    "INCREASES", "DECREASES",
    "COEXISTS_WITH", "ASSOCIATED_WITH", "NOT_ASSOCIATED_WITH",
    "REGULATES", "PART_OF",
    "COMPARED_WITH", "SUPERIOR_TO", "EQUIVALENT_TO",
]

# Symmetric relations: A-B and B-A are the same fact. build_graph.py sorts
# endpoints canonically so both directions merge into one weighted edge.
SYMMETRIC = {
    "ASSOCIATED_WITH", "COEXISTS_WITH",
    "NOT_ASSOCIATED_WITH", "EQUIVALENT_TO", "COMPARED_WITH",
}