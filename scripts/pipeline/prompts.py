"""Prompt templates for the Stage 2 pipeline.

All prompts are plain strings with Python ``str.format()`` placeholders.
No Jinja2 here — Jinja2 lives only inside the Data Designer recipe YAML.
"""

# ── Stage 1A: Logical Entailment extraction ──────────────────────────────────

LE_SYSTEM = (
    "You are a technical documentation analyst. Extract logical structure "
    "from text as JSON. Return ONLY valid JSON — no explanation, no markdown fences."
)

LE_USER = """\
Extract ALL distinct logical entailments from this documentation passage.
A passage may cover multiple topics — each warrants its own entailment.

For each entailment:
- conclusion: the central claim or main topic of that thread (one sentence)
- premises: one or more supporting facts that logically justify this conclusion;
  derive from the text if not explicit, and include all facts needed for support
- context: ancillary technical details specific to this entailment (1-2 sentences)
- entities: comma-separated list of notable entities for this entailment

Rules:
- All content must come ONLY from the passage — no outside knowledge
- Ensure every truth assignment satisfying the premises also satisfies the
  corresponding conclusion
- Return as many entailments as needed to cover the distinct topics the
  passage contains

Note: 'recommendations' field exists in LogEntailment with default="" for
downstream compatibility; not requested here to keep output tight.

Return JSON:
{{
  "entailments": [
    {{
      "conclusion": "...",
      "premises": ["...", "..."],
      "context": "...",
      "entities": "..."
    }}
  ]
}}

Passage:
{text}"""


# ── Stage 1A: KVP generation (premise→Q, conclusion→A) ───────────────────────

KVP_SYSTEM = (
    "You are a technical documentation analyst. Generate a QA pair as JSON. "
    "Return ONLY valid JSON — no explanation, no markdown fences."
)

KVP_USER = """\
Given a premise and conclusion from a documentation passage, generate a substantive
question-answer pair grounded in the source text.

Rules for the QUESTION (one clear sentence, derived from the PREMISE):
- Must require specific, verifiable knowledge — an env var name, command,
  parameter, version, configuration value, or product-specific identifier.
- Avoid yes/no questions and avoid generic "what is X" questions when X is a
  well-known term; phrase the question so the answer must cite a detail from
  the source.

Rules for the ANSWER (2-4 sentences, 50-120 tokens):
- State the core claim that the CONCLUSION conveys.
- Cite the specific identifier(s), value(s), command(s), or version-specific
  details from the source text that justify the claim.
- May add one sentence of immediately-actionable context (when to use, what
  it interacts with) — but only using facts present in the source.
- Use complete declarative sentences. Do not begin with "Yes" or "No".

Both Q and A must be derivable ONLY from the source text — no outside knowledge.
Use the PREMISE and CONCLUSION as seeds; you may rewrite them for grammar and
specificity, but do not introduce facts not present in the source.

Return JSON:
{{
  "question": "...",
  "answer": "..."
}}

Premise:    {premise}
Conclusion: {conclusion}
Source text:
{text}"""


# ── Stage 1B: kNN neighborhood synthesis ─────────────────────────────────────

SYNTHESIS_SYSTEM = (
    "You are a technical documentation analyst. Generate QA pairs as JSON. "
    "Return ONLY valid JSON — no explanation, no markdown fences."
)

SYNTHESIS_USER = """\
Given the following related passages from {domain} documentation,
generate two high-value questions that require synthesizing information across passages.

- BRIDGING: a question whose complete answer requires combining facts from two or more
  of the provided passages (e.g. "Given that X requires Y, and Y depends on Z,
  what must be true when configuring...?")
- CONTRASTIVE: a question that asks how two related concepts, models, or configurations
  differ (e.g. "How does deploying NIM on Kubernetes differ from bare-metal deployment?")

Rules:
- Questions must be answerable ONLY from the provided passages — no outside knowledge
- Answers must be complete sentences, 1-3 sentences max
- Do not fabricate facts not present in the passages

Passages:
{neighborhood_context}

Output JSON:
{{
  "pairs": [
    {{"type": "bridging",    "question": "...", "answer": "..."}},
    {{"type": "contrastive", "question": "...", "answer": "..."}}
  ]
}}"""


# ── Stage 1C: Instruction diversity ──────────────────────────────────────────

INSTRUCTION_SYSTEM = (
    "You are a technical writer. Generate instruction-following training examples "
    "as JSON. Return ONLY valid JSON."
)

INSTRUCTION_USER = """\
Given the following {domain} documentation passage, generate instruction-following
training examples in different formats:

1. SUMMARY:    "Summarize the key points of [topic] from the following..."
2. LISTICLE:   "List all [configuration options / prerequisites / steps] for..."
3. PROCEDURAL: "Provide step-by-step instructions for..."
   (Omit type 3 if the passage is not procedural — no numbered steps, no command sequence.)

Rules:
- Answer ONLY from the provided text — no outside knowledge.
- Each answer 50-200 tokens.

Passage:
{passage}

Output JSON (2 or 3 pairs — omit procedural if not applicable):
{{
  "pairs": [
    {{"type": "summary",  "question": "...", "answer": "..."}},
    {{"type": "listicle", "question": "...", "answer": "..."}}
  ]
}}"""


# ── Stage 2: QA admission/refinement ─────────────────────────────────────────

QA_EVAL_SYSTEM = (
    "You are a frontier-grade technical documentation QA gate. Verify and refine "
    "question-answer pairs against their source text. Return ONLY valid JSON."
)

QA_EVAL_USER = """\
Evaluate the question and answer below against the source text.

Rules:
1. Use ONLY the source text as factual support.
2. If the QA is grounded but phrasing can be improved, rewrite it.
3. If the QA is not logically associated with the source, rewrite it only when a
   corrected grounded pair can be derived from the source text.
4. If the QA is ungrounded and cannot be repaired from the source text, set
   admit=false and explain why.
5. If the QA is grounded and logically consistent, return it unchanged with
   admit=true.

Question: {question}
Answer:   {answer}
Source text:
{context}

Output JSON:
{{
  "admit": true,
  "prompt": "...",
  "completion": "...",
  "grounded": true,
  "answer_fidelity": true,
  "no_hallucination": true,
  "repairable": true,
  "reason": "one concise audit note"
}}"""


# ── Stage 4: External judge ─────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an impartial grader of question-answer pairs against their source text. "
    "Return ONLY valid JSON with fields: grounded, answer_fidelity, no_hallucination "
    "(booleans), and reason (string)."
)

JUDGE_USER = """\
Grade this Q+A pair against the source text on three criteria.

Question: {question}
Answer:   {answer}
Source text:
{context}

Criteria (each a boolean):
- grounded:        every factual claim in the answer is supported by the source text
- answer_fidelity: the answer directly addresses the question
- no_hallucination: the answer contains NO entities, version numbers, or commands
                    that are not present in the source text

Output JSON:
{{
  "grounded":         true,
  "answer_fidelity":  true,
  "no_hallucination": true,
  "reason":           "<one-sentence justification>"
}}"""


# ── Stage 1.5: Data Designer recipe user template (also embedded in YAML) ────

GAPFILL_SYSTEM = (
    "You are a precise NVIDIA technical assistant. Generate Q+A pairs grounded ONLY "
    "in the provided documentation chunks. Return ONLY valid JSON — no explanation, "
    "no markdown fences."
)

GAPFILL_RECIPE_USER = """\
Generate {pairs_count} question-answer pairs about {product_family} that are answerable
ONLY from the following retrieved documentation chunks. Vary the question styles
using these examples as reference:
{seed_styles}

Documentation chunks:
{retrieved_chunks}

Output JSON:
{{
  "pairs": [
    {{"question": "...", "answer": "..."}}
  ]
}}"""
