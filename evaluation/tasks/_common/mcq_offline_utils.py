"""Shared helpers for offline vLLM multiple-choice benchmark scripts.

Factored out of mmstar_offline.py so mmlu_pro/mmmu_pro/mmmu_val (and any
future MC benchmark) don't each reimplement the vLLM batching loop, the
logprob-based option scoring, and letter extraction from scratch.

Deliberately self-contained (no import of the `lmms_eval` package) so these
offline scripts keep working without the harness installed.
"""
import ast
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from tqdm import tqdm

ALL_CHOICE_LETTERS = [chr(ord("A") + i) for i in range(16)]  # up to a 16-way MC, plenty of headroom


# ---------------------------------------------------------------------------
# Robust letter extraction (vendored from
# lmms_eval.tasks._task_utils.mcq_extract.extract_mcq_answer)
# ---------------------------------------------------------------------------

_ANSWER_PHRASES = [
    "the answer is",
    "answer is",
    "the correct answer is",
    "correct answer is",
    "the best answer is",
    "best answer is",
    "the correct option is",
    "correct option is",
    "the best option is",
    "best option is",
    "the choice is",
    "choice is",
    "the correct choice is",
    "correct choice is",
    "i choose",
    "i select",
    "i pick",
    "my answer is",
    "my choice is",
]

_FORMAT_PRIORITY = {
    "start": 10,
    "end": 9,
    "phrase": 7,
    "parentheses": 6,
    "period": 5,
    "colon": 4,
    "right_paren": 3,
    "space": 2,
    "fallback": 0,
}


def extract_mcq_answer(response: str, choices: Optional[Sequence[str]] = None) -> str:
    """Extract a multiple-choice answer letter from free-form model output.

    Returns the uppercase choice letter, or "" if none found. See
    lmms_eval.tasks._task_utils.mcq_extract for the original.
    """
    if not response or not response.strip():
        return ""

    all_choices = list(choices) if choices else ALL_CHOICE_LETTERS[:8]

    text = response.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        text = text.strip(char)
    text = " " + text + " "

    candidates: list = []

    for ch in all_choices:
        if f"({ch})" in text:
            candidates.append((ch, text.rfind(f"({ch})"), "parentheses"))
    for ch in all_choices:
        if f"{ch}." in text:
            candidates.append((ch, text.rfind(f"{ch}."), "period"))
    for ch in all_choices:
        if f"{ch}:" in text:
            candidates.append((ch, text.rfind(f"{ch}:"), "colon"))
    for ch in all_choices:
        if f"{ch})" in text:
            candidates.append((ch, text.rfind(f"{ch})"), "right_paren"))
    for ch in all_choices:
        if f"{ch} " in text:
            candidates.append((ch, text.rfind(f"{ch} "), "space"))

    text_lower = text.lower()
    for phrase in _ANSWER_PHRASES:
        idx = text_lower.find(phrase)
        if idx != -1:
            after = idx + len(phrase)
            for ch in all_choices:
                ch_pos = text.find(ch, after)
                if ch_pos != -1:
                    candidates.append((ch, ch_pos, "phrase"))

    stripped = text.strip()
    for ch in all_choices:
        if stripped.startswith(ch) and (len(stripped) == 1 or not stripped[1].isalpha()):
            candidates.append((ch, 0, "start"))
    for ch in all_choices:
        if stripped.endswith(ch) and (len(stripped) == 1 or not stripped[-2].isalpha()):
            candidates.append((ch, len(text) - 1, "end"))

    if not candidates:
        for ch in all_choices:
            if ch in text:
                candidates.append((ch, text.rfind(ch), "fallback"))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: (_FORMAT_PRIORITY.get(x[2], 0), x[1]), reverse=True)
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Option formatting shared by mmlu_pro / mmmu_pro / mmmu (parse_options
# convention: "A. <option>\nB. <option>\n...")
# ---------------------------------------------------------------------------

def format_options_block(options: Sequence[str]) -> (str, List[str]):
    """Return (formatted options block, list of valid letters) for a doc's options."""
    valid_letters = ALL_CHOICE_LETTERS[: len(options)]
    block = "\n".join(f"{letter}. {opt}" for letter, opt in zip(valid_letters, options))
    return block, valid_letters


def parse_options_field(options_field) -> List[str]:
    """MMMU/MMMU-Pro store `options` as a stringified Python list."""
    if isinstance(options_field, str):
        return ast.literal_eval(options_field)
    return list(options_field)


# ---------------------------------------------------------------------------
# Multi-image interleaving for MMMU-style "<image N>" tokens
# ---------------------------------------------------------------------------

_IMAGE_TOKEN_RE = re.compile(r"<image (\d+)>")


def build_interleaved_content(text: str, doc: dict) -> List[dict]:
    """Split `text` on "<image N>" tokens and interleave with the
    corresponding PIL images from doc["image_N"], preserving order.

    Mirrors MMMU/MMMU-Pro's own doc_to_visual convention (image_1..image_7
    fields referenced by <image N> tokens inside the question text).
    """
    content: List[dict] = []
    last_end = 0
    for m in _IMAGE_TOKEN_RE.finditer(text):
        pre = text[last_end : m.start()].strip()
        if pre:
            content.append({"type": "text", "text": pre})
        field = f"image_{m.group(1)}"
        image = doc[field]
        content.append({"type": "image_pil", "image_pil": image.convert("RGB")})
        last_end = m.end()
    tail = text[last_end:].strip()
    if tail:
        content.append({"type": "text", "text": tail})
    if not content:
        content.append({"type": "text", "text": text})
    return content


# ---------------------------------------------------------------------------
# CLI args shared across all offline MC scripts
# ---------------------------------------------------------------------------

def add_common_vllm_args(parser, default_max_new_tokens: int = 10):
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_new_tokens", type=int, default=default_max_new_tokens, help="Maximum number of new tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k for sampling")
    parser.add_argument("--presence_penalty", type=float, default=2.0, help="Presence penalty")
    parser.add_argument("--repetition_penalty", type=float, default=2.0, help="Repetition penalty")
    parser.add_argument("--output_file", type=str, default="results_offline.jsonl", help="Output file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to evaluate (for testing)")
    return parser


def make_sampling_params(args, stop: Optional[List[str]] = None, logprobs: int = 20):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        stop=stop or ["\n"],
        logprobs=logprobs,
    )


# ---------------------------------------------------------------------------
# Logprob-based MC scoring
# ---------------------------------------------------------------------------

def score_option_logprobs(output, valid_letters: Sequence[str]) -> Dict[str, float]:
    """Max logprob assigned to each valid letter as the *first* generated token."""
    option_log_probs = {letter: float("-inf") for letter in valid_letters}
    log_probs_list = output.outputs[0].logprobs
    if log_probs_list and log_probs_list[0] is not None:
        first_token_logprobs = log_probs_list[0]
        for _token_id, logprob_obj in first_token_logprobs.items():
            token_str = logprob_obj.decoded_token.strip().upper()
            if token_str in option_log_probs:
                option_log_probs[token_str] = max(option_log_probs[token_str], logprob_obj.logprob)
    return option_log_probs


def best_option(option_log_probs: Dict[str, float]) -> Optional[str]:
    best = max(option_log_probs, key=option_log_probs.get)
    if option_log_probs[best] == float("-inf"):
        return None
    return best


# ---------------------------------------------------------------------------
# Generic batched vLLM chat loop with jsonl output
# ---------------------------------------------------------------------------

def run_batched_chat(
    llm,
    sampling_params,
    dataset,
    batch_size: int,
    prepare_inputs_fn: Callable[[object, List[int]], List[list]],
    process_output_fn: Callable[[int, dict, object], dict],
    output_path,
    chat_template_kwargs: Optional[dict] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    """Run llm.chat in batches, score each output via process_output_fn, and
    stream results to a jsonl file. Returns the full list of result dicts.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(dataset) if limit is None else min(limit, len(dataset))
    results: List[dict] = []

    with open(output_path, "w") as f:
        for start_idx in tqdm(range(0, n, batch_size)):
            end_idx = min(start_idx + batch_size, n)
            batch_indices = list(range(start_idx, end_idx))
            batch_done = 0
            try:
                messages_batch = prepare_inputs_fn(dataset, batch_indices)
                outputs = llm.chat(
                    messages=messages_batch,
                    sampling_params=sampling_params,
                    chat_template_kwargs=chat_template_kwargs or {"enable_thinking": False},
                )
                for idx, output in zip(batch_indices, outputs):
                    result = process_output_fn(idx, dataset[idx], output)
                    results.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    batch_done += 1
                f.flush()
            except Exception as e:
                print(f"\nError processing batch {start_idx}-{end_idx}: {e}")
                for idx in batch_indices[batch_done:]:
                    # keep failed examples in results so accuracy counts them as wrong
                    error_result = {"idx": idx, "error": str(e)}
                    results.append(error_result)
                    f.write(json.dumps(error_result) + "\n")
                f.flush()

    return results


def strip_think_tags(text: str) -> str:
    """Mirror --reasoning-parser qwen3: strip any <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def summarize_mc_results(results: List[dict], output_path, keys=("correct_exact_match", "correct_logprobs")):
    total = len(results)
    n_errors = sum(1 for r in results if "error" in r)
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"Total: {total}")
    if n_errors:
        print(f"WARNING: {n_errors} examples failed (batch errors) and are counted as wrong")
    for key in keys:
        n_correct = sum(1 for r in results if r.get(key))
        acc = (n_correct / total * 100) if total else 0.0
        print(f"Correct ({key}): {n_correct} ({acc:.2f}%)")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
