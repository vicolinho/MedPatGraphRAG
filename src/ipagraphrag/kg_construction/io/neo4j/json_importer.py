import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from huggingface_hub.utils import tqdm
from neo4j import GraphDatabase
from torchgen.api.lazy import process_ir_type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def load_json(file_path: Path) -> dict:
    """Load and return the JSON data from *file_path*."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded JSON from %s", file_path)
    return data


def extract_nodes_and_triples(data: dict) -> tuple[list[dict], list[tuple]]:
    """
    Walk the nested JSON structure and collect:
      - nodes: list of dicts ready for Neo4j import
      - triples: list of (src_id, rel_type, tgt_id) tuples

    The `type` field is always normalised to a list of strings so that
    downstream Cypher can apply multiple labels consistently.
    """
    all_nodes: list[dict] = []
    all_triples: list[tuple] = []

    for model_key, documents in data.items():
        logger.info("Processing model key: %s", model_key)
        for doc_id, doc_data in documents.items():
            # ---- Nodes ----
            for node in doc_data.get("nodes", []):
                # Normalise `type` to a list (it may arrive as a bare string)
                raw_type = node.get("type", [])
                labels: list[str] = (
                    raw_type if isinstance(raw_type, list) else [raw_type]
                )
                all_nodes.append(
                    {
                        "id": node["id"],
                        "text": node.get("text", ""),
                        "data_source": node.get("data_source", doc_id),
                        "labels": labels,
                    }
                )

            # ---- Triples ----
            for triple in doc_data.get("triples", []):
                if len(triple) >=3 :
                    all_triples.append((triple["source"], triple["type"], triple["target"]))
                else:
                    logger.warning(
                    "Skipping malformed triple in doc %s: %s", doc_id, triple)

    logger.info(
        "Extracted %d nodes and %d triples total.",
        len(all_nodes),
        len(all_triples),
    )
    return all_nodes, all_triples


# ---------------------------------------------------------------------------
# Neo4j write transactions
# ---------------------------------------------------------------------------
BATCH_SIZE = 500  # Number of nodes/relationships per Cypher UNWIND batch


def _create_nodes_tx(tx, batch: list[dict]) -> int:
    """
    Transaction function: MERGE nodes in *batch* and apply dynamic labels
    via APOC.  Returns the number of nodes created in this batch.

    Cypher logic:
      1. UNWIND the batch as individual node parameter maps.
      2. MERGE on the `id` property (prevents duplicates on re-runs).
      3. SET text and data_source properties.
      4. Call apoc.create.addLabels to attach all labels from the `labels` list.
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (n {id: row.id})
    SET   n.text        = row.text,
          n.data_source = row.data_source
    WITH  n, row.labels AS lbls
    CALL  apoc.create.addLabels(n, lbls) YIELD node
    RETURN count(node) AS created
    """
    result = tx.run(cypher, batch=batch)
    summary = result.consume()
    return summary.counters.nodes_created


def create_nodes(session, nodes: list[dict]) -> int:
    """Batch-insert all nodes, BATCH_SIZE at a time."""
    total_created = 0
    for start in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[start : start + BATCH_SIZE]
        created = session.execute_write(_create_nodes_tx, batch)
        total_created += created
        logger.debug("Node batch %d–%d: %d created.", start, start + len(batch), created)
    logger.info("Nodes created: %d", total_created)
    return total_created

def _create_realtionships_tx(tx, batch, rel_type):
    query = f"""
        UNWIND $batch AS row
        MATCH (src:mention {{id: row.src_id}})
        MATCH (tgt:mention {{id: row.target_id}})
        MERGE (src)-[:{rel_type}]->(tgt)
        """
    result = tx.run(query, batch=batch)
    summary = result.consume()
    return summary.counters.relationships_created

def _create_relationships_abstract_tx(tx, batch, rel_type):
    query = f"""
        UNWIND $batch AS row
        MATCH (src:abstract {{id: row.src_id}})
        MATCH (tgt:mention {{id: row.target_id}})
        MERGE (src)-[:{rel_type}]->(tgt)
        """
    result = tx.run(query, batch=batch)
    summary = result.consume()
    return summary.counters.relationships_created

def create_relationships(session, triples, batch_size=1000):
    """
    Imports triples (src_id, target_id, rel_type) into a Neo4j database in transactions of 1000 rows.

    Args:
        session: Neo4j session object for database connection.
        triples: A set of triples (src_id, target_id, rel_type).
        batch_size: Number of triples to process in a single transaction (default: 1000).
    """
    # Convert the set of triples to a list for processing
    rel_type_groups = {}
    for src_id, rel_type,  target_id in triples:
        if rel_type not in rel_type_groups:
            rel_type_groups[rel_type] = []
        rel_type_groups[rel_type].append({"src_id": src_id, "target_id": target_id})
    print("grouping finished")
    print("import {} edges ".format(len(triples)))
    # Process the triples in batches
    for rel_type, triples_for_type in rel_type_groups.items():
        if rel_type == 'has_mention':
            for i in tqdm(range(0, len(triples_for_type), batch_size)):
                # Extract the current batch
                batch = triples_for_type[i:i + batch_size]
                print("Process batch {} with {} edges".format(i, len(batch)))
                # Prepare the batch data for the query
                batch_data = batch
                # Execute the query
                if rel_type != 'has_mention':
                    result = session.execute_write(_create_realtionships_tx, batch=batch_data, rel_type=rel_type)
                else:
                    result = session.execute_write(_create_relationships_abstract_tx, batch=batch_data, rel_type=rel_type)
                print(result)

# ---------------------------------------------------------------------------
# Constraint / index setup (optional but recommended for performance)
# ---------------------------------------------------------------------------
def ensure_constraints(session) -> None:
    """
    Create a uniqueness constraint on the `id` property for all nodes
    labelled `Entity` (a generic anchor label).  This accelerates MERGE
    look-ups during import.  Safe to call even if the constraint exists.
    """
    try:
        session.run(
            "CREATE CONSTRAINT unique_node_id IF NOT EXISTS "
            "FOR (n:Entity) REQUIRE n.id IS UNIQUE"
        )
        logger.info("Uniqueness constraint on :Entity(id) ensured.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not create constraint (may already exist): %s", exc)
