

def find_unlinked_mentions(driver, ontology_label, chunk_labels):
    query = f"""MATCH (m:mention)-[]-(c:{":".join(chunk_labels)})
            WHERE NOT (m)-[]-({ontology_label})
            RETURN m.id as mention_id, m.text AS term, m.type AS type, c.text AS chunk"""
    mention_dict = {}
    records, summary, keys = driver.execute_query(query)
    for record in records:
        mention_dict[record['mention_id']] = {"text": record['term'], "type": record['type'],
                                              "chunk": record['chunk']}
    return mention_dict


