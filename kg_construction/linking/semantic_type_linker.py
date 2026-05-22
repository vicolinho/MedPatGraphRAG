import argparse
import json
import os

from tqdm import tqdm

from llm import llm_agent
from kg_construction.linking.prompt import LINK_WITH_CONTEXT
import spacy

from kg_construction.io.neo4j.importer import Neo4jImport

# Optional: Neo4j enrichment
try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


def load_scispacy_model():
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    return nlp


def find_unlinked_mentions(driver):
    query = """MATCH (m:mention)-[]-(c:chunk)
            WHERE NOT (m)-[]-(:SemanticType)
            RETURN m.mention_id as mention_id, m.text AS term, m.type AS mention_type, c.text AS chunk"""
    mention_dict = {}
    records, summary, keys = driver.execute_query(query)
    for record in records:
        mention_dict[record['mention_id']] = {"text": record['term'], "mention_type": record['mention_type'],
                                              "chunk": record['chunk']}
    return mention_dict

def get_semantic_type(driver):
    query = """MATCH(sty: SemanticType) 
    RETURN sty.name AS name, sty.definition AS definition"""
    semantic_type_list = []
    records, summary, keys = driver.execute_query(query)
    for record in records:
        semantic_type_list.append({'name':record['name'], 'definition': record['definition']})
    return semantic_type_list

def link_mentions(mention_dict, semantic_type_dict)->list[tuple]:
    links = []
    for mention_id, mention in tqdm(mention_dict.items()):
        link_prompt = LINK_WITH_CONTEXT + ": mention: {} chunk: {} semantic_types: {}".format(mention['text'], mention['chunk'],
                                                                                              json.dumps(semantic_type_dict))

        result = llm_agent.llm_query(os.getenv('LLM_STUB_URL'), os.getenv('API_KEY'), link_prompt, os.getenv("LLM_MODEL"))
        data = json.loads(result.text)
        semantic_type = json.loads(data['choices'][0]['message']['content'])
        links.append((mention_id,semantic_type['name']))
        #print("mention {} semantic_type: {}".format(mention, semantic_type))
    return links

def save_links(driver, links):
    neo4j_importer = Neo4jImport(driver)
    edge_queries = []
    for mention_id, sem_type in links:
        edge_query = (f"MATCH (a {{mention_id: '{mention_id}'}}), "
                      f"(b:SemanticType {{name: '{sem_type}'}}) "
                      f"MERGE (a)-[r:annotated_with]->(b)")
        edge_queries.append(edge_query)
    neo4j_importer.execute_queries(driver, edge_queries)



def find_mention_entity(doc, mention):
    # Try to find an entity in doc that matches the mention (case-insensitive)
    for ent in doc.ents:
        if ent.text.lower() == mention.lower():
            return ent
    # If not found, try substring match
    for ent in doc.ents:
        if mention.lower() in ent.text.lower():
            return ent
    return None

def query_neo4j_semtype(tui, driver):
    query = """
    MATCH (sty:SemanticType {ui: $tui})
    RETURN sty.name AS name, sty.definition AS definition, sty.tree_number AS tree_number
    """
    with driver.session() as session:
        result = session.run(query, tui=tui)
        record = result.single()
        if record:
            return dict(record)
    return None


