import asyncio
import json
import os
import pickle
import time

import torch

from ipagraphrag.kg_construction.io.neo4j.importer import Neo4jImport
from ipagraphrag.kg_construction.linking.prompt import LINK_WITH_CONTEXT_SNOMED
from ipagraphrag.llm import llm_agent


def get_concept_candidates(session, id):
    cypher = """
    MATCH (p:mention)
    WHERE p.id = $id
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
    params = {"index": "concept_description_vector", "k": 20, "id": id}
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
    async def run_all():
        todo = [(m_id, mention) for m_id, mention in mention_dict.items()]
        total = len(todo)
        sem = asyncio.Semaphore(8)
        lock = asyncio.Lock()
        counter = {"n": 0}

        async def worker(m_id, mention):
            async with sem:
                t0 = time.time()
                try:
                    semantic_type_dict = get_concept_candidates(session, m_id)
                    link_prompt = LINK_WITH_CONTEXT_SNOMED + ": mention: {} type:{} chunk: {} snomed_concepts: {}".format(
                        mention['text'],
                        mention['type'],
                        mention['chunk'],
                        json.dumps(
                            semantic_type_dict))
                    result = llm_agent.llm_query(os.getenv('BASE_URL'), os.getenv('API_KEY'), link_prompt,
                                                                                os.getenv("LLM_MODEL"))
                    data = json.loads(result.text)
                    msg = f"{len(semantic_type_dict)} candidates, found"
                    semantic_type = json.loads(data['choices'][0]['message']['content'])
                except Exception as e:
                    msg = f"ERROR {len(semantic_type_dict)} candidates"
            async with lock:
                links.append((m_id, semantic_type['id']))
                with open("links.tmp", "wb") as f:
                    pickle.dump(links, f)
                counter["n"] += 1
                print(f"  [{counter['n']}/{total}] {m_id} - {msg}", flush=True)
        await asyncio.gather(*(worker(m_id, mention) for m_id, mention in todo))
    asyncio.run(run_all())
    # for mention_id, mention in tqdm(mention_dict.items()):
    #     semantic_type_dict = get_concept_candidates(session, mention_id)
    #     link_prompt = LINK_WITH_CONTEXT_SNOMED + ": mention: {} type:{} chunk: {} snomed_concepts: {}".format(mention['text'],
    #                                                                                         mention['type'],
    #                                                                                        mention['chunk'],
    #                                                                                        json.dumps(
    #                                                                                            semantic_type_dict))
    #
    #     result = llm_agent.llm_query(os.getenv('BASE_URL'), os.getenv('API_KEY'), link_prompt,
    #                               os.getenv("LLM_MODEL"))
    #     data = json.loads(result.text)
    #     semantic_type = json.loads(data['choices'][0]['message']['content'])
    #     links.append((mention_id, semantic_type['id']))
    return links
        # print("mention {} semantic_type: {}".format(mention, semantic_type))

def _save_links_tx(tx, batch):
    edge_query = """UNWIND $batch as row
                      MATCH (a {id: row.mention_id}), 
                      (b:ObjectConcept) "
                      WHERE elementId(b)=row.sem_type
                      MERGE (a)-[r:annotated_with]->(b))"""
    result = tx.run(edge_query, batch=batch)


def save_links(driver, links, batch_size=500):
    neo4j_importer = Neo4jImport(driver)
    edge_queries = []
    with driver.session() as session:
        for i in range(0,len(links), batch_size):
            batch = links[i, i+batch_size]
            session.execute_write(_save_links_tx, batch)
        # for mention_id, sem_type in links:
        #     edge_query = (f"MATCH (a {{id: '{mention_id}'}}), "
        #                   f"(b:ObjectConcept) "
        #                   f"WHERE elementId(b)='{sem_type}'"
        #                   f"MERGE (a)-[r:annotated_with]->(b)")
        #     edge_queries.append(edge_query)
        # neo4j_importer.execute_queries(session, edge_queries)


def cos_sim(a, b):
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

# cosine similarity of first entity with all the entities

