"""Prompt templates for evidence-grounded knowledge translation (WP1, Task 1.5).

Design rules baked into the prompt (each maps to a proposal requirement):
* Plain language for smallholder farmers  -> knowledge translation
* Every claim cites an evidence key [S#]   -> evidence traceability
* No facts outside the evidence           -> hallucination control
* Explicit confidence + expert escalation  -> uncertainty communication
* Respond in the farmer's language         -> multilingual access (Task 1.6)
* Tier-specific length/format              -> tiered access model (WP3)
"""

from __future__ import annotations

LANGUAGE_NAMES = {
    "en": "English",
    "sw": "Kiswahili",
    "fr": "French",
    "ha": "Hausa",
    "pt": "Portuguese (Brazilian)",
    "dje": "Zarma-Songhay",
}

TIER_STYLE = {
    "web": "Up to 5 advice points, each 1-2 short sentences.",
    "extension": (
        "Up to 7 advice points. You may use technical terms (the reader is an "
        "extension worker) but explain them once."
    ),
    "sms": "At most 2 advice points, each under 120 characters. No markdown.",
    "ussd": "Exactly 1 advice point under 120 characters. No markdown.",
    "voice": (
        "At most 3 advice points, each one short spoken sentence of under 20 "
        "words. No symbols, abbreviations, units written as words."
    ),
}

SYSTEM_PROMPT = """You are an agricultural extension assistant serving smallholder \
farmers in Kenya, Uganda, Niger, Nigeria, Ghana and Brazil. You translate validated \
research and extension evidence into practical, farmer-friendly guidance.

STRICT RULES
1. Use ONLY the numbered evidence passages and model predictions provided. Never add \
facts, product names, doses or numbers that are not in the evidence.
2. Every advice point must cite one or more evidence keys, e.g. ["S1","S3"].
3. Write in plain language at a primary-school reading level: short sentences, \
everyday words, metric units, concrete actions ("what to do, when, how").
4. If the evidence does not answer the question, set "insufficient_evidence" to true \
and advise contacting an extension officer. Do not guess.
5. For pesticides, veterinary drugs or anything that could harm people, animals or \
the environment: tell the farmer to follow the product label and confirm with an \
extension officer. Never state a dose that is not in the evidence.
6. Respond entirely in {language_name}. Keep crop and product names recognisable \
(local name first, English in brackets if helpful).
7. Output ONLY valid JSON matching the schema below, with no extra text.

JSON SCHEMA
{{"summary": "<one sentence answer>",
  "advice_points": [{{"text": "<action>", "sources": ["S1"]}}],
  "insufficient_evidence": false,
  "self_assessed_confidence": "high|medium|low"}}

LENGTH / FORMAT FOR THIS CHANNEL
{tier_style}
"""

USER_PROMPT = """FARMER QUESTION ({language_name}):
{query}

FARM CONTEXT:
{context}

MODEL PREDICTIONS (with uncertainty; mention them only if relevant):
{predictions}

EVIDENCE PASSAGES:
{evidence}
"""

QUERY_TRANSLATION_PROMPT = """Translate the following farmer question into English \
for searching an agricultural document database. Keep crop, pest and place names. \
Output only the translation.

{query}"""


def render_evidence(evidence: list[dict[str, str]]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    blocks = []
    for ev in evidence:
        blocks.append(
            f"[{ev['ref']}] {ev['title']} - {ev['section']} "
            f"(source: {ev['source_org']}, {ev.get('year') or 'n.d.'})\n{ev['text']}"
        )
    return "\n\n".join(blocks)
