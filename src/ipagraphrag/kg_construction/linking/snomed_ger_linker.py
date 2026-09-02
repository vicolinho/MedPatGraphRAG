import json
import os

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import XLMRobertaTokenizer

from ipagraphrag.kg_construction.io.neo4j.importer import Neo4jImport
from ipagraphrag.kg_construction.linking.prompt import LINK_WITH_CONTEXT, LINK_WITH_CONTEXT_SNOMED
from ipagraphrag.llm import llm_agent


def get_concept_candidates(session, mention_id):
    print(mention_id)
    cypher = """
    MATCH (p:mention)
    WHERE p.mention_id = $mention_id
    WITH p.embedding AS queryEmbedding
    CALL
    db.index.vector.queryNodes($index, $k, queryEmbedding)
    YIELD
    node, score
    RETURN
    node, elementId(node) as id, score
    ORDER
    BY
    score
    DESC
    """
    params = {"index": "concept_description_vector", "k": 20, "mention_id": mention_id}
    results = []
    for record in session.run(cypher, **params):
        concept_query = """
        MATCH (c:ObjectConcept)--(cd:ConceptDescription)
        WHERE elementId(cd)=$description_id
        RETURN c as node, elementId(c) as id
        """
        props = dict(record["node"])
        params = {"description_id":record['id']}
        #print(params)
        for record_2 in session.run(concept_query, **params):
            props_2 = dict(record_2["node"])
            #print(props_2)
            results.append({'id': record_2['id'], 'FSN': props_2['FSN'], 'description': props['term']})
    return results


def link_mentions(session, mention_dict)->list[tuple]:
    links=[]
    for mention_id, mention in tqdm(mention_dict.items()):
        semantic_type_dict = get_concept_candidates(session, mention_id)
        link_prompt = LINK_WITH_CONTEXT_SNOMED + ": mention: {} type:{} chunk: {} snomed_concepts: {}".format(mention['text'],
                                                                                            mention['type'],
                                                                                           mention['chunk'],
                                                                                           json.dumps(
                                                                                               semantic_type_dict))

        result = llm_agent.llm_query(os.getenv('BASE_URL'), os.getenv('API_KEY'), link_prompt,
                                  os.getenv("LLM_MODEL"))
        data = json.loads(result.text)
        semantic_type = json.loads(data['choices'][0]['message']['content'])
        links.append((mention_id, semantic_type['id']))
    return links
        # print("mention {} semantic_type: {}".format(mention, semantic_type))

def save_links(driver, links):
    neo4j_importer = Neo4jImport(driver)
    edge_queries = []
    for mention_id, sem_type in links:
        edge_query = (f"MATCH (a {{mention_id: '{mention_id}'}}), "
                      f"(b:ObjectConcept) "
                      f"WHERE elementId(b)='{sem_type}'"
                      f"MERGE (a)-[r:annotated_with]->(b)")
        edge_queries.append(edge_query)
    neo4j_importer.execute_queries(driver, edge_queries)


def cos_sim(a, b):
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

# cosine similarity of first entity with all the entities

