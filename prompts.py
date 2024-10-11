from langchain.prompts import PromptTemplate


INIT_PROMPT = """Given a question that requires multi-step retrieval to collect necessary knowledge triples and offer the final answer, you are an advanced knowledge reasoner and retrieval facilitator. In this step, you should first extract the initial topic entities for retrieving relevant knowledge, and then propose a detailed and concrete retrieval guidance for each topic entity to reflect which aspect of knowledge related to this entity you want to retrieve. Be sure to only provide the retrieval guidance for necessary knowledge. Don't overthink the entities that you don't yet know their complete names, but instead explore knowledge starting from concrete entities in the given question.

Please strictly use the following template and do not provide any unnecessary explanations:
- Entity name 1: propose a detailed and concrete retrieval guidance for this entity
- Entity name 2: propose a detailed and concrete retrieval guidance for this entity
- ...

{examples}

Question: {question}
Retrieval Guidance:
"""

DIRECT_REASON_PROMPT = """You are given a question and some associated knowledge triples. Each knowledge triple takes the form of (subject entity, relation, object entity). You are asked to reason out the final answer based on the provided knowledge and your own knowledge.

Please strictly use the following inference template without anything else:
Thought: think step by step to reason out the final answer
Answer: the final answer

{examples}

Question: {question}
Knowledge triples collected in previous steps: {paths}
"""

REASON_PROMPT = """Given a question that requires multi-step retrieval to collect necessary knowledge triples and offer the final answer, you are an advanced knowledge reasoner and retrieval facilitator. The knowledge triples that you collected in previous steps take the form of (subject entity; relation; object entity).

In this step, you should first carefully determine whether the collected knowledge triples are sufficient for you to offer the final answer. Don't answer with uncertainty. Please strictly use the following judgment template:
Whether the given knowledge triples are sufficient for answering: Yes or No

If Yes, then think and offer the final answer based on the collected knowledge triples. Please strictly use the following inference template:
Thought: think step by step to reason out the final answer
Answer: the final answer

If No, then provide a high-quality concrete guidance for the retrieval step to collect more necessary knowledge triples. You should first provide a set of entities that need futher retrieval in the retrieval step, and then propose a detailed and concrete retrieval guidance for each entity to reflect which aspect of knowledge related to this entity you want to retrieve. Be sure to only provide the retrieval guidance for necessary knowledge. Don't overthink the entities that you don't yet know their complete names. Please strictly use the following template and do not provide any unnecessary explanations.
Retrieval Guidance:
- Entity name 1: propose a detailed and concrete retrieval guidance for this entity
- Entity name 2: propose a detailed and concrete retrieval guidance for this entity
- ...

{examples}

Question: {question}
Knowledge triples collected in previous steps: {paths}
"""


REFINE_PROMPT = """Given a set of documents and a specified entity with a guidance on the entity-related knowledge, you are an advanced relevant knowledge extractor. According to the provided knowledge guidance, you should extract sufficient information from the documents to construct the structured knowledge triples that are related to the input entity. The constructed knowledge triples must take the complete form of (subject entity; relation; object entity), in which the relation must be detailed and concrete. You must provide the knowledge triples without any vague expressions such as "not found" or "N/A". Use newline characters as separators between multiple knowledge triples. Feel free to ignore irrelevant knowledge in the documents.

The input entity with knowledge guidance are organized as follows:
- Input entity name: a knowledge guidance for this entity

Please strictly use the following triple template, and do not provide any unnecessary explanations or notes.
(subject entity; relation; object entity)\n(subject entity; relation; object entity)\n...

{examples}

Documents: 
{docs}
Input Entity with Knowledge Guidance:
{entities}
Structured Knowledge Triple(s): """

EXTRACT_PROMPT = """Give a list of knowledge triples, each of which takes the form of (subject entity, relation, object entity), you are asked to extract all the entities in these triples. Use semicolons as separators between multiple entities.

Here are some examples:
{examples}

Triples: {triples}
Entities: """


init_prompt_template = PromptTemplate(
                        input_variables=["examples", "question"],
                        template = INIT_PROMPT,
                        )

direct_reason_prompt_template = PromptTemplate(
                        input_variables=["examples", "question", "paths"],
                        template = DIRECT_REASON_PROMPT,
                        )

reason_prompt_template = PromptTemplate(
                        input_variables=["examples", "question", "paths"],
                        template = REASON_PROMPT,
                        )

refine_prompt_template = PromptTemplate(
                        input_variables=["examples", "docs", "entities"],
                        template = REFINE_PROMPT,
                        )

extract_prompt_template = PromptTemplate(
                        input_variables=["examples", "triples"],
                        template = EXTRACT_PROMPT,
                        )
