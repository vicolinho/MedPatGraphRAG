
EXTRACT_MENTIONS = """Developer: You are an intelligent assistant designed to extract medical mentions from a text. 
Analyze the input contents and extract mentions related to medical terms. Focus on diagnoses, symptoms, and 
medication. 
The output should be valid JSON and formatted as follows:
{
  "medical_mentions": [
    {"term": "XX", "type": "diagnosis|symptom|medication"},
    ...
  ]
}
Ensure that every object in the "medical_mentions" array includes both the "term" and "type" keys."""


EXTRACT_MENTIONS_CONCEPT = """Developer: You are an intelligent assistant designed to extract medical mentions from a text. 
Analyze the input contents and extract mentions related to medical terms and concepts. Focus on diagnoses, symptoms, and 
medication. 
The output should be valid JSON and formatted as follows:
{
  "medical_mentions": [
    {"term": "XX", "type": "diagnosis|symptom|medication"},
    ...
  ]
}
Ensure that every object in the "medical_mentions" array includes both the "term" and "type" keys."""



"""Every piece of prompt text phase 1 sends to the model, in one place.

Text only -- no schema structure, no API settings. extract_neo4j.py wraps
these strings into the neo4j-graphrag schema dicts (the two *_DESCRIPTION
constants become property descriptions, EXTRA_INSTRUCTIONS is passed as the
extractor's `examples` argument, which the ERExtractionTemplate appends to
its own instructions verbatim).

Editing this file changes what the model is asked to do, and therefore what
lands in the graph -- a change here invalidates an existing extraction the
same way a model change would.

COUPLING: the type definitions in EXTRA_INSTRUCTIONS must stay in sync with
ALLOWED_NODES (extract_neo4j.py) and EXTRACTION_RELATIONS (common/relations.py)
-- a type in the schema but missing here is allowed through with no definition
to guide the model, one defined here but not in the schema is extracted and
then silently dropped by STRICT enforcement. extract_neo4j.py checks this at
import; the definitions are the "- <TYPE>: ..." lines it looks for.
"""

# Appended to neo4j-graphrag's own extraction prompt (the `examples` argument).
EXTRA_INSTRUCTIONS = """
ABBREVIATIONS: if the text defines an abbreviation (e.g. "myocardial infarction
(MI)" or "MI (myocardial infarction)"), use the FULL EXPANDED FORM as the node
name every time that entity is mentioned in this text, even where only the
abbreviation appears afterward. If an abbreviation is used WITHOUT ever being
defined in this text, keep it as-is (do not guess an expansion).

Use the following definitions to assign the correct node type:

- Disease: pathological conditions, disorders, syndromes (e.g. Hypertension, Asthma, Myocardial Infarction).
- Drug: pharmaceutical agents and chemical compounds used therapeutically (e.g. Aspirin, Dexamethasone, Metformin).
- Procedure: medical, surgical, or diagnostic procedures performed on a patient (e.g. Colonoscopy, Laparotomy, MRI).
- Intervention: non-pharmacological, non-surgical treatments such as behavioural, educational, or organisational measures (e.g. Telephone Counselling, Screening Programme).
- AnatomicalStructure: organs, tissues, and body parts (e.g. Liver, Left Ventricle, Talus).
- Symptom: clinical signs and symptoms experienced by a patient (e.g. Vomiting, Dyspnoea, Pain).
- ClinicalOutcome: measurable results of treatment or disease progression, including rates and endpoints (e.g. Mortality, Graft Survival, Physician Revisitation Rate, Symptomatic Return To Baseline).
- BiologicalProcess: cellular or molecular processes (e.g. Apoptosis, Inflammation, Programmed Cell Death).
- CellularComponent: subcellular structures and organelles (e.g. Mitochondria, Chloroplast).
- MedicalSpecialty: clinical or surgical specialties only (e.g. General Surgery, Emergency Medicine, Cardiology).
- GeneticVariant: genes, mutations, polymorphisms, and genetic predispositions (e.g. BRCA1, HDL Mutation, SNP).
- Pathogen: infectious agents such as bacteria, viruses, fungi, and parasites (e.g. Staphylococcus aureus, HIV, Influenza virus).

Use the following definitions to assign the correct relationship type:

- TREATS: A Drug, Procedure, or Intervention reduces or eliminates a Disease or Symptom (therapeutic intent). The target must be a Disease or Symptom, never a ClinicalOutcome.
- PREVENTS: An entity prevents the occurrence of a Disease or ClinicalOutcome (prophylactic intent).
- CAUSES: An entity directly produces or leads to a Disease or Symptom (e.g. adverse effects, direct causation). Target must be a Disease or Symptom.
- PREDISPOSES: An entity increases susceptibility to a Disease without directly causing it (risk factor relationships).
- INCREASES: An entity raises the value, frequency, rate, or magnitude of a ClinicalOutcome or Symptom. Capture ONLY the literal direction stated in the text; do NOT judge whether it is beneficial or harmful.
- DECREASES: An entity lowers the value, frequency, rate, or magnitude of a ClinicalOutcome or Symptom (literal direction only, no benefit/harm judgment).
- COEXISTS_WITH: Two conditions frequently co-occur in the same patient population (comorbidity).
- ASSOCIATED_WITH: A general statistical or clinical association between two entities where NONE of the more specific relationships above (TREATS, CAUSES, PREDISPOSES, INCREASES, DECREASES, REGULATES, etc.) applies. Weak and non-causal - use it only as a fallback, never instead of a more specific relation the text actually supports.
- NOT_ASSOCIATED_WITH: The study explicitly reports NO significant association or NO effect between two entities (a null/negative finding). Symmetric.
- REGULATES: A BiologicalProcess or molecule controls another process or molecular entity.
- PART_OF: An AnatomicalStructure is a component of a larger anatomical structure.
- COMPARED_WITH: Two Drugs/Procedures/Interventions are evaluated against each other for the same outcome, but the text states NO clear result. If a result IS stated, use SUPERIOR_TO or EQUIVALENT_TO instead.
- SUPERIOR_TO: The source entity is reported as MORE effective (or safer) than the target for the same indication/outcome. Directional: source = the better entity.
- EQUIVALENT_TO: Two entities are reported as equally effective, or showing NO significant difference, for the same outcome. Symmetric.

CRITICAL -- CAPTURE NULL / NEGATIVE FINDINGS. When the abstract reports that something was tested and NO significant effect, association, or difference was found, you MUST extract it as a relationship - it is a key clinical finding, not a method:
- "X had no significant effect on Y" / "X was not associated with Y"  ->  X NOT_ASSOCIATED_WITH Y
- "A and B showed no significant difference" / "A was comparable to B"  ->  A EQUIVALENT_TO B
Do NOT silently drop null results, and do NOT turn a null result into INCREASES, DECREASES, or ASSOCIATED_WITH. A null finding is the answer to many questions.

CRITICAL EXCLUSION RULE -- the following describe HOW a study was conducted, not its medical content. They are NEVER nodes and must be completely ignored:
- Any statistical, mathematical, or computational method, model, metric, or test used to analyse data.
- Any study design, data source, or data-collection procedure (e.g. trial types, surveys, database searches, sampling).
- Any research infrastructure, registry, or care setting used to run the study.
Note the distinction: a statistical TEST (e.g. "chi-square test") is a method and is ignored, but the RESULT it reports ("no significant difference") IS medical content - capture that result as NOT_ASSOCIATED_WITH / EQUIVALENT_TO between the real medical entities.

Precedence: when a comparative claim has a stated result, emit SUPERIOR_TO or EQUIVALENT_TO and do NOT also emit COMPARED_WITH for that pair. Extract only the underlying medical entities (diseases, drugs, procedures, symptoms, outcomes, genes, pathogens) and the clinical relations between them.

For every relationship, also fill in its "quote" property: the exact sentence or shortest verbatim fragment from the input text that states it. Copy the original wording as closely as possible - do not paraphrase, summarize, or invent one.
"""

# Description of the "name" property every entity type carries. STRICT mode
# drops any node left with no allowed properties, so the schema MUST declare
# "name" (which the extraction template already has the model emit) - otherwise
# every node is stripped to {} and discarded, taking all edges with it.
NAME_PROPERTY_DESCRIPTION = (
    "The entity's name or label as it appears in the text."
)

# Description of the "quote" property every relation type carries. Captures the
# extractor's own verbatim supporting sentence via neo4j-graphrag's schema
# "properties" mechanism (ERExtractionTemplate's own prompt example already asks
# the model for a properties dict per edge).
QUOTE_PROPERTY_DESCRIPTION = (
    "The exact sentence or shortest verbatim fragment from the input text that "
    "states this relationship. Copy the original wording as closely as possible "
    "- do NOT paraphrase or summarize. One sentence (or short clause) is "
    "enough; do not include unrelated surrounding text."
)