"""
umls_to_neo4j.py
================
Imports the UMLS Semantic Network into Neo4j from two pipe-delimited source files:

  • SRDEF   – defines Semantic Types (STY rows) and Relations (RL rows)
  • SRSTRE1 – lists directed edges between Semantic Types

Usage
-----
1. Install the Neo4j Python driver:
       pip install neo4j

2. Edit the CONFIG block below (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
   SRDEF_PATH, SRSTRE1_PATH).

3. Run:
       python umls_to_neo4j.py

The script uses MERGE statements and batched writes (BATCH_SIZE records per
committed transaction) so it is safe to re-run without creating duplicates.

SRDEF column layout (0-indexed after splitting on '|'):
  0  RT    – record type: STY | RL
  1  UI    – unique identifier (T001, T202, …)
  2  Name  – semantic type name or relation name
  3  STN/RTN – tree number
  4  DEF   – definition
  5  EX    – examples (STY only)
  6  UN    – usage note (STY only)
  7  NH    – non-human flag (STY only)
  8  ABR   – abbreviation
  9  RIN   – inverse relation name (RL only)

SRSTRE1 column layout (0-indexed):
  0  subject_ui    – UI of the source SemanticType
  1  relation_name – name of the relation (matches name field in SRDEF RL rows)
  2  object_ui     – UI of the target SemanticType
"""

import csv
import os
from pathlib import Path

from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# CONFIG – edit these values before running
# ---------------------------------------------------------------------------

SRDEF_PATH   = "SRDEF"     # path to pipe-delimited UMLS SRDEF file
SRSTRE1_PATH = "SRSTRE1"   # path to pipe-delimited UMLS SRSTRE1 file

BATCH_SIZE = 500            # records per committed transaction batch
# ---------------------------------------------------------------------------


# ── helpers ─────────────────────────────────────────────────────────────────

def _clean(value: str):
    """Return stripped string or None for empty / NULL placeholder values."""
    v = value.strip()
    return None if v in ("", "NULL") else v


def batch(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ── parsers ──────────────────────────────────────────────────────────────────

def parse_srdef(filepath: str):
    """
    Parse a pipe-delimited SRDEF file.

    Returns
    -------
    sty_nodes  : list[dict]  – property dicts for SemanticType nodes
    rl_nodes   : list[dict]  – property dicts for Relation nodes
    rl_by_name : dict        – relation name -> rl property dict (for edge enrichment)
    """
    sty_nodes  = []
    rl_nodes   = []
    rl_by_name = {}

    with open(filepath, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="|")
        for lineno, row in enumerate(reader, start=1):
            if not row:
                continue
            # Pad row to at least 10 fields so index access is always safe
            row = row + [""] * max(0, 10 - len(row))
            rt = row[0].strip()

            if rt == "STY":
                node = {
                    "ui":          _clean(row[1]),
                    "name":        _clean(row[2]),
                    "tree_number": _clean(row[3]),
                    "definition":  _clean(row[4]),
                    "examples":    _clean(row[5]),
                    "usage_note":  _clean(row[6]),
                    "non_human":   _clean(row[7]),
                    "abbreviation": _clean(row[8]),
                }
                sty_nodes.append(node)

            elif rt == "RL":
                node = {
                    "ui":           _clean(row[1]),
                    "name":         _clean(row[2]),
                    "tree_number":  _clean(row[3]),
                    "definition":   _clean(row[4]),
                    "abbreviation": _clean(row[8]),
                    "inverse":      _clean(row[9]),
                }
                rl_nodes.append(node)
                # Index by relation name for fast edge enrichment lookups
                if node["ui"]:
                    rl_by_name[node["ui"]] = node

    return sty_nodes, rl_nodes, rl_by_name


def parse_srstre1(filepath: str):
    """
    Parse a pipe-delimited SRSTRE1 file.

    Returns
    -------
    list[dict] – each dict has keys: subject_ui, relation_name, object_ui
    """
    edges = []
    with open(filepath, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="|")
        for row in reader:
            if not row or len(row) < 3:
                continue
            subject_ui    = _clean(row[0])
            relation_name = _clean(row[1])
            object_ui     = _clean(row[2])
            if subject_ui and relation_name and object_ui:
                edges.append({
                    "subject_ui":    subject_ui,
                    "ui": relation_name,
                    "object_ui":     object_ui,
                })
    return edges


# ── Neo4j helpers ────────────────────────────────────────────────────────────

def create_indexes(driver) -> None:
    """Create indexes on SemanticType(ui) and Relation(ui) for fast MERGE."""
    with driver.session() as session:
        session.run(
            "CREATE INDEX IF NOT EXISTS FOR (n:SemanticType) ON (n.ui)"
        )
        session.run(
            "CREATE INDEX IF NOT EXISTS FOR (r:Relation) ON (r.ui)"
        )
    print("  Indexes ensured.")


def import_sty_nodes(driver, sty_nodes: list) -> None:
    """
    MERGE SemanticType nodes in batches.

    Uses SET n += row so existing properties are updated on re-runs.
    None-valued keys are omitted per-row before sending to avoid overwriting
    existing data with null.
    """
    cypher = """
    UNWIND $batch AS row
    MERGE (n:SemanticType:Concept {ui: row.ui, name: row.name, 
    definition: row.definition, abbreviation:row.abbreviation})
    """
    total = 0
    for chunk in batch(sty_nodes, BATCH_SIZE):
        # Strip None values so SET n += row does not store null properties
        clean_chunk = [{k: v for k, v in row.items() if v is not None}
                       for row in chunk]
        with driver.session() as session:
            session.run(cypher, batch=clean_chunk)
        total += len(chunk)
        print(f"  SemanticType nodes merged: {total} / {len(sty_nodes)}")


def read_rl_nodes(rl_nodes: list) -> None:
    rl_by_name = {}
    for rl_node in rl_nodes:
        relation = {k: v for k, v in rl_node.items() if v is not None}
        rl_by_name[relation["ui"]] = relation
    return rl_by_name


def import_relationships(driver, relationships: list, rl_by_name: dict) -> None:
    """
    MERGE directed RELATED_TO edges between SemanticType nodes.

    Each edge's Cypher properties dict (row.props) includes:
      - relation_name
      - rel_ui, rel_tree_number, rel_definition, rel_abbreviation, rel_inverse
        (enriched from rl_by_name when a matching RL entry exists)
    """
    # Pre-enrich each relationship with RL metadata
    enriched = []
    print(rl_by_name)
    for e in relationships:
        print(e)
        rl = rl_by_name.get(e["ui"], {})
        props = {
            "rel_definition":  rl.get("definition"),
            "rel_abbreviation": rl.get("abbreviation"),
            "rel_inverse":     rl.get("inverse"),
        }
        # Remove None values so they are not written as null properties
        props = {k: v for k, v in props.items() if v is not None}
        enriched.append({
            "subject_ui":    e["subject_ui"],
            "object_ui":     e["object_ui"],
            "relation_name": rl_by_name[e['ui']]['name'],
            "props":         props,
        })

    cypher = """
    UNWIND $batch AS row
    MATCH (a:SemanticType {ui: row.subject_ui})
    MATCH (b:SemanticType {ui: row.object_ui})
    CALL apoc.create.relationship(a, row.relation_name, row.props, b) YIELD rel
    RETURN rel
    """
    total = 0
    for chunk in batch(enriched, BATCH_SIZE):
        with driver.session() as session:
            session.run(cypher, batch=chunk)
        total += len(chunk)
        print(f"  Relationships merged: {total} / {len(enriched)}")


#