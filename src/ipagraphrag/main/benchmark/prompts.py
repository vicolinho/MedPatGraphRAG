"""The system prompt eval_qa.py sends, per evaluation mode.

Kept apart from eval_qa.py for the same reason as phase 1's prompts.py: prompt
wording is the experiment's independent variable, and it should be editable
without reading around the retrieval and scoring code. The user message is
assembled in eval_qa.build_prompt -- it is structure (question + context
block), not wording, so it stays there.

The three modes differ ONLY in where the answer is supposed to come from;
everything else is shared, so a wording change applies to all of them and the
comparison between modes stays fair.
"""

# Answer format. Parsed by eval_qa.parse(), which takes the first yes/no/maybe
# token of the first line -- hence "EXACTLY one word on the first line".
SYSTEM_PREFIX = (
    "You are a biomedical QA assistant. Answer with EXACTLY one word on the "
    "first line: yes, no, or maybe. Then one sentence of justification. "
)

# PubMedQA's "maybe" is a real class (~10% of PQA-L), but an unprompted model
# retreats to it whenever evidence is thin, which inflates it at the cost of
# yes/no recall. Identical in all three modes so it never explains a difference
# between them.
_MAYBE_RULE = (
    "Don't default to 'maybe', but use it when there is equally "
    "convincing evidence for 'no' and 'yes', or a general lack of "
    "evidence for either."
)

# All three grant the fallback to the model's own knowledge: without it the
# graph modes are penalised for the retrieval misses rather than measured on
# the facts they do supply.
MODE_INSTRUCTIONS = {
    "graph": ("Base your answer on the provided knowledge-graph facts. If the "
              "provided facts are not enough to answer the question, use your "
              "biomedical knowledge. Facts with a source sentence reflect the "
              "study's own stated result - treat them as reliable evidence. "
              + _MAYBE_RULE),
    "textrag": ("Base your answer on the retrieved abstracts below. If the "
                "retrieved abstracts are not enough to answer the question, use your "
                "biomedical knowledge. "
                + _MAYBE_RULE),
    "baseline": ("Base your answer on your biomedical knowledge. "
                 + _MAYBE_RULE),
}


def system_prompt(mode):
    return SYSTEM_PREFIX + MODE_INSTRUCTIONS[mode]