
import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"


SYSTEM_PROMPT = """
You are an evidence-grounded aircraft maintenance research assistant.

Your task is to answer questions using only the supplied maintenance
records.

STRICT RULES:
1. Use only facts supported by the supplied records.
2. Cite supporting records using their source IDs, for example:
   [SDR_2019_00000087].
3. Distinguish documented facts from possible interpretations.
4. Do not invent root causes, inspection steps, repair procedures,
   part numbers, limits, or maintenance instructions.
5. If the evidence does not establish the cause, explicitly say so.
6. If records conflict or are ambiguous, explain the uncertainty.
7. Do not treat a repair reported in one incident as a universal fix.
8. Do not claim that a record proves a component caused a failure
   unless the record explicitly supports that conclusion.
9. For aircraft-specific maintenance actions, direct the reader to
   the applicable approved maintenance documentation and authorized
   maintenance personnel.
10. Keep the answer concise and technically clear.

Use these sections when appropriate:
- Summary
- Evidence
- Limitations

If the retrieved records do not adequately answer the question,
state that the available evidence is insufficient.
"""


def build_context(results):
    """
    Convert retrieved Qdrant results into evidence for the LLM.
    """
    context_parts = []

    for index, result in enumerate(results, start=1):
        source_id = result.get("source_record_id", "UNKNOWN")
        score = result.get("score", 0.0)
        text = result.get("text", "")
        metadata = result.get("metadata", {})

        context_parts.append(
            f"""
[EVIDENCE {index}]
Source ID: {source_id}
Retrieval score: {score:.4f}
Aircraft: {metadata.get('aircraft_make', '')}
Model: {metadata.get('aircraft_model', '')}
Component: {metadata.get('component_part_name', '')}
Condition: {metadata.get('component_part_condition', '')}

Record text:
{text}
[/EVIDENCE {index}]
"""
        )

    return "\n".join(context_parts)


class AnswerGenerator:
    def __init__(
        self,
        model,
        ollama_url=DEFAULT_OLLAMA_URL,
        timeout=120,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout

    def generate(self, question, results):
        """
        Generate an answer grounded in retrieved records.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not results:
            return (
                "The available indexed records did not provide evidence "
                "to answer this question. Consult the applicable approved "
                "aircraft maintenance documentation."
            )

        context = build_context(results)

        user_prompt = f"""
QUESTION:
{question.strip()}

EVIDENCE RECORDS:
{context}

INSTRUCTIONS:
- Read every evidence record before answering.
- The records above are the authoritative evidence for this task.
- Record IDs identify individual reports.
- Do not claim that a type of record is absent if a supplied
  record explicitly mentions it.
- Distinguish between what a report states and what can be
  inferred from it.
- Do not invent causes, procedures, or technical conclusions.
- Cite claims using the exact source ID, such as
  [SDR_2019_00000087].
- If the records contain relevant information, summarize it.
- If they do not establish the underlying cause, explicitly
  state that the cause is not established by these records.

Answer the following question using the supplied evidence:
{question.strip()}
"""


        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "options": {
                    "temperature": 0,
                },
            },
            timeout=self.timeout,
        )

        response.raise_for_status()
        data = response.json()

        answer = data.get("message", {}).get("content", "").strip()

        if not answer:
            raise RuntimeError(
                "The language model returned an empty answer."
            )

        return answer
