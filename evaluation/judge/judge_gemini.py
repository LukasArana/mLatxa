import argparse
import json
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

JUDGE_PROMPT_TEMPLATE = """
# System Instructions: Basque Text Quality Evaluation

You are an expert annotator fluent in Basque (*Euskara*). Your task is to evaluate the linguistic quality of the provided Basque text based on two distinct criteria: **Grammar, Orthography, and Punctuation** and **Style and Naturalness**. 

---

#Task Description:
A model-generated answer in Basque to evaluate and two score rubrics are given.
1. For each rubric, write a brief feedback assessing the answer strictly based on the criteria.
2. After writing the feedback for each rubric, write the corresponding score.
3. The output format should look as follows:
{{
  "linguistic_correctness_feedback": (write a feedback for grammar, orthography, and punctuation),
  "linguistic_correctness": (0, 1, 2, 3, 4 or 5),
  "style_naturalness_feedback": (write a feedback for style and naturalness),
  "style_naturalness": (0, 1, 2, 3, 4 or 5)
}}
4. Please do not generate any other opening, closing, or explanations. Respond only with a JSON object containing the feedback and scores as specified above.

###Answer to evaluate:
{answer}

---

## Criterion 1: Grammar, Orthography, and Punctuation (1-5)
Evaluate the linguistic correctness of the response regarding grammar, spelling, typos, and punctuation. Limit the evaluation strictly to **formal correctness**, ignoring aspects like syntactic variety or lexical richness.

* **Score 1 (Inintelligible):** The text contains extremely severe structural, grammatical, and discursive errors that destroy global coherence. There is no structure allowing for a coherent message, making the text completely unintelligible.
* **Score 2 (Severe Errors):** The text contains numerous errors in individual words or basic verbal paradigms (e.g., severe auxiliary verb/nor-nork confusion) that severely hinder comprehension. While isolated words or fragments can be recognized, the accumulation of errors makes it difficult to reconstruct the global message.
* **Score 3 (Noticeable Deviations):** The text contains several errors (e.g., missed ergative *-ek* cases, incorrect declensions, or clear typos). They do not prevent comprehension, but they represent clear deviations from *Euskaltzaindia* (Basque Language Academy) norms that an adult native speaker would not make.
* **Score 4 (Minor Slips):** The text is correct as a whole, though it contains a few minor errors or typos. These do not affect comprehension and represent slight, occasional deviations from the norm that an adult native speaker might accidentally make in casual writing.
* **Score 5 (Flawless):** The text is completely correct regarding grammar, orthography, and punctuation. There is no substantial room for improvement in correctness.

---

## Criterion 2: Style and Naturalness (1-5)
Evaluate the stylistic adequacy, fluency, and naturalness of the response. Does the response sound like authentic Basque, or does it feel like a mechanical translation?

* **Score 1 (Completely Antinatural):** The text is completely artificial. It includes discursive structures that no native speaker would ever use, or introduces invented words, non-existent loanwords, or structurally incoherent expressions.
* **Score 2 (Awkward & Heavily Calqued):** The text feels highly awkward and unnatural. It relies heavily on forced collocations, word-for-word dictionary translations, or structural calques from Spanish/French.
* **Score 3 (Acceptable but Rigid):** The text is acceptable and understandable, but it feels rigid or robotic. It suffers from repetitive vocabulary, overuse of nominalizations (e.g., excessive use of *-tzea* or *-tzerakoan* where relative clauses are preferred), or minor semantic calques. It lacks natural linguistic flow.
* **Score 4 (Good Quality):** The text is of good quality and sounds natural. It might feature slight lexical repetition or minor lack of structural variety, but it flows well. Any minor awkwardness does not affect global coherence or the authentic flavor of the language.
* **Score 5 (Native/Idiomatic Flow):** The text flows with total naturalness and elegance. It actively employs genuine Basque expressions (*esamoldeak*), rich vocabulary, proper word order, and appropriate connectors (*bada*, *izan ere*, etc.). The register adapts perfectly to the communicative context.

---

## Specific Guidelines for Basque Evaluation

When scoring, keep an eye out for these common pitfalls specific to Basque text generation:

1.  **The Ergative Case (*-ek*):** Check if the subject of transitive verbs correctly carries the ergative case. Dropping it or misapplying it to intransitive verbs is a severe grammatical error (lowering Criterion 1).
2.  **The *Galdegaia* (Focus Rule):** Basque is highly flexible but relies on placing the most important information right before the main verb. If a text has perfect grammar but copies Western SVO (Subject-Verb-Object) word order blindly, it sounds unnatural (lowering Criterion 2).
3.  **Romance Calques:** Look out for literal translations from Spanish or French. For example, using *suportatu* (to endure) when *jasan* is appropriate, or overusing passive/nominalized structures instead of active verbal relative clauses (*-nean*, *-lako*, etc.).

"""


JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "linguistic_correctness_feedback": {"type": "STRING"},
        "linguistic_correctness": {
            "type": "INTEGER",
            "enum": ["0", "1", "2", "3", "4", "5"],
        },
        "style_naturalness_feedback": {"type": "STRING"},
        "style_naturalness": {
            "type": "INTEGER",
            "enum": ["0", "1", "2", "3", "4", "5"],
        },
    },
    "required": [
        "linguistic_correctness_feedback",
        "linguistic_correctness",
        "style_naturalness_feedback",
        "style_naturalness",
    ],
}


def call_judge(client: genai.Client, model: str, prompt: str, max_retries: int = 3) -> dict:
    """
    Sends a single judging request to Gemini and returns the parsed JSON result.
    Retries on transient errors with simple linear backoff.
    """
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JUDGE_SCHEMA,
        max_output_tokens=1500,
    )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return json.loads(response.text)
        except Exception as e:  # noqa: BLE001 - want to retry on any transient failure
            last_err = e
            wait = 2 * attempt
            print(f"    WARNING: request failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"All {max_retries} attempts failed: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", required=True, help="Path to the predictions JSONL file.")
    parser.add_argument("--output_file",       required=True, help="Output path for the judged results JSONL.")
    parser.add_argument("--model_name",        default="gemini-2.5-flash", help="Gemini model to use for judging.")
    parser.add_argument("--sleep",             type=float, default=30.0,
                         help="Optional seconds to sleep between requests (helps avoid rate limits).")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_file)
    if not predictions_path.exists():
        sys.exit(f"ERROR: predictions file not found: {predictions_path}")

    # Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment by default.
    client = genai.Client()

    # Read predictions
    predictions = []
    with open(predictions_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping malformed line {lineno}: {e}")

    print(f"Read {len(predictions)} predictions from {predictions_path}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0
    failed = 0

    with open(output_path, "w") as out:
        for i, entry in enumerate(predictions, 1):
            idx         = entry.get("idx")
            question_id = entry.get("question_id")
            prediction  = entry.get("prediction", "").strip()

            if question_id is None:
                print(f"  WARNING: missing question_id at idx={idx}, skipping.")
                skipped += 1
                continue

            if not prediction:
                print(f"  WARNING: empty prediction for question_id={question_id} (idx={idx}), skipping.")
                skipped += 1
                continue

            prompt = JUDGE_PROMPT_TEMPLATE.format(answer=prediction)

            print(f"[{i}/{len(predictions)}] Judging idx={idx} question_id={question_id}...")
            try:
                result = call_judge(client, args.model_name, prompt)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR: giving up on idx={idx} question_id={question_id}: {e}")
                failed += 1
                continue

            record = {
                "idx": idx,
                "question_id": question_id,
                "prediction": prediction,
                **result,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            written += 1

            if args.sleep:
                time.sleep(args.sleep)

    print(f"\nResults written to {output_path}")
    print(f"  Judged  : {written}")
    print(f"  Skipped : {skipped}")
    print(f"  Failed  : {failed}")


if __name__ == "__main__":
    main()