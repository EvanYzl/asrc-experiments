CONSTRAINT_REASON_PROMPT = """Please determine whether the input triple satisfies the following three constraints based on the provided knowledge graph evidence.

structure Constraint:
{path_Connectivity}

type Constraint:
{fewshot_triples}

background Constraint:
{background_evidence}

The triple to be evaluated:
{test_triple}

Please return 'Y' if the input triple satisfies all three constraints and is plausible; otherwise, return 'N'. Do not say anything else except your determination.
"""


SUBGRAPH_REASON_PROMPT = """Please determine whether the relation in the input can be reliably inferred between the head and tail entities, based on a set of neighbor triples and reasoning paths from the knowledge graph.
A set of neighbor triples from the knowledge graph are:
{neighbor_triples}
A set of reasoning paths from the knowledge graph are:
{reasoning_paths}
The relation to be inferred is:
{test_triple}
Please return 'Y' if there is sufficient evidence from the knowledge graph to infer the relation, otherwise return 'N'. Do not say anything else except your determination.
"""

