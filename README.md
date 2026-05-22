
# iPA GraphRAG

## Motivation

This project provides a pipeline to build a GraphRAG system for electronic health records
and to search via the realized GraphRAG system.

## Approach

### GraphRAG Building

The pipeline of building the GraphRAG process consists of the following tasks:

1. **Backend** ```kg_construction/io```: As the backend, we utilize a Neo4j graph database to store the knowledge graph.
As a backend graph, we import several ontologies, along with their concepts and relations. The intention
is to connect the different mentions from the medical documents via the common ontology graph.
Currently, we support the import of the Semantic Network from UMLS to roughly categorize the medical
mentions.
2. **Extraction** ```kg_construction/extraction```: We extract from the documents medical mentions such as medication, diagnoses,
and symptoms. Therefore, we split the document into chunks and use an LLM to extract the mentions
for each chunk. Each chunk is connected to the identified mentions and to the next chunk.
3. **Entity Linking** ```kg_construction/linking```: The goal of the entity linking process is to annotate each mention with a concept from
the backbone knowledge graph. The input of the entity linking process is the unlinked mentions and the
concepts from an ontology. The process stores the determined annotations as edges in the underlying KG.
Currently, we utilize an LLM to determine the annotations for each mention and the corresponding semantic type.
In the future, we aim to support larger ontologies. Therefore, we realize a 
blocking step to reduce the set of concept candidates.

### GraphRAG Search

We use the GraphRAG system to enable an LLM-based querying with patient-related context information. The main steps of 
using context information with LLMs are the following:

1. **Retrieval** The cornerstone of a GraphRAG system is the identification of relevant context information from the 
knowledge graph. The retrieval methods are located at ```search.retrieval``` package. Currently, we have implemented
a simple node retrieval method identifying the Top5 nodes for a query based on the precalculated text-embeddings.
We further plan the following retrieval methods:
   - 
