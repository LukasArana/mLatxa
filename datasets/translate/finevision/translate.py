import argparse
import json
import os
import re
import sys
import time
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Default model / vLLM configuration (used as argparse defaults below)
# -----------------------------------------------------------------------------
MODEL = "HiTZ/Latxa-Llama-3.1-70B-Instruct"
TENSOR_PARALLEL_SIZE = 4          # Leonardo Booster node = 4 GPUs
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 256
GPU_MEMORY_UTILIZATION = 0.90
DTYPE = "bfloat16"
SEED = 0
IMAGE_TOKEN = "<image>"

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text from English "
    "into Basque. Output ONLY the translation, with no explanations, notes, "
    "labels or surrounding quotation marks. Preserve numbers, symbols, line "
    "breaks, punctuation, LaTeX and any code exactly as they appear."
)

# Two few-shot examples (FineVision-style VQA) so the model emits ONLY the
# translation and nothing else.
FEWSHOT = [
    ("Subtract 0 red cylinders. How many objects are left?",
     "Kendu 0 zilindro gorri. Zenbat objektu geratzen dira?"),
    ("What is the color of the large metallic sphere on the right?",
     "Zer kolore du eskuineko esfera metaliko handiak?"),
]

_NUMERIC_ONLY = re.compile(r"^[\s\d.,;:%+\-*/()=\[\]<>]*$")

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shard-translate chat JSONL EN->Basque with Latxa + vLLM.")
    p.add_argument("--input", required=True,
                   help="Path to the input data. Either a JSON array (one big list) "
                        "or JSONL (one JSON object per line).")
    p.add_argument("--output-dir", required=True, help="Directory for per-job output files.")
    p.add_argument("--samples-per-job", type=int, default=20000,
                   help="Number of input lines each job translates.")
    p.add_argument("--job-index", type=int, default=None,
                   help="0-based job index. Defaults to $SLURM_ARRAY_TASK_ID.")
    p.add_argument("--resume", action="store_true",
                   help="Skip the job if its output file is already complete.")

    # vLLM / model config (defaults come from the constants above).
    p.add_argument("--model", default=MODEL,
                   help="HF model id to load.")
    p.add_argument("--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE,
                   help="Number of GPUs to shard the model across.")
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN,
                   help="Maximum model context length.")
    p.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS,
                   help="Maximum number of sequences batched per step.")
    p.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION,
                   help="Fraction of GPU memory vLLM may use (0-1).")
    p.add_argument("--dtype", default=DTYPE,
                   help="Model dtype (e.g. bfloat16, float16, auto).")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Random seed for sampling.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    return p.parse_args()

def is_trivial(text: str) -> bool:
    """True if the string carries no translatable letters."""
    return text.strip() == "" or _NUMERIC_ONLY.match(text) is not None

def split_on_token(content: str) -> List[str]:
    """
    Split content on the image token, keeping the token as its own segment.
    "<image>How many?" -> ["", "<image>", "How many?"]
    Empty strings are preserved so we can reconstruct exactly.
    """
    parts = content.split(IMAGE_TOKEN)
    out: List[str] = []
    for i, part in enumerate(parts):
        out.append(part)
        if i < len(parts) - 1:
            out.append(IMAGE_TOKEN)
    return out

def clean_translation(raw: str) -> str:
    """Trim chatty prefixes / wrapping quotes the model may add."""
    t = raw.strip()
    t = re.sub(r"^(itzulpena|translation|euskaraz)\s*[:\-]\s*", "", t, flags=re.IGNORECASE)
    quote_pairs = {'"': '"', "'": "'", "«": "»", "“": "”", "‘": "’"}
    if len(t) >= 2 and t[0] in quote_pairs and t[-1] == quote_pairs[t[0]]:
        t = t[1:-1].strip()
    return t

def build_messages(text: str) -> List[dict]:
    """Translation prompt with two few-shot turns (Llama-3 chat format)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for en, eu in FEWSHOT:
        msgs.append({"role": "user", "content": en})
        msgs.append({"role": "assistant", "content": eu})
    msgs.append({"role": "user", "content": text})
    return msgs

def read_slice(path: str, start: int, end: int) -> List[dict]:
    """
    Return only the samples whose global index lies in [start, end).

    Accepts either a JSON array (the whole file is one big list) or JSONL
    (one JSON object per line). JSONL is streamed, so each job parses only
    the lines it actually needs instead of the whole file.
    """
    with open(path, "r", encoding="utf-8") as f:
        # Peek at the first non-whitespace character to detect the format.
        first = ""
        while True:
            ch = f.read(1)
            if ch == "":
                return []
            if not ch.isspace():
                first = ch
                break
        f.seek(0)

        if first == "[":
            data = json.load(f)
            return data[start:end]

        out: List[dict] = []
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if idx >= end:
                break
            if idx >= start:
                out.append(json.loads(line))
            idx += 1
        return out

def main() -> None:
    args = parse_args()

    job_index = args.job_index
    if job_index is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            sys.exit("ERROR: provide --job-index or run under a SLURM array ($SLURM_ARRAY_TASK_ID).")
        job_index = int(env)

    start = job_index * args.samples_per_job
    end = start + args.samples_per_job

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"translated_{job_index:06d}.jsonl")
    tmp_path = out_path + ".tmp"

    samples = read_slice(args.input, start, end)

    if not samples:
        print(f"[job {job_index}] no samples in range [{start}, {end}); nothing to do.", flush=True)
        open(out_path, "a", encoding="utf-8").close()  # touch for consistent merges/--resume
        return

    if args.resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            done = sum(1 for _ in f)
        if done == len(samples):
            print(f"[job {job_index}] already complete ({done} lines); skipping.", flush=True)
            return
        print(f"[job {job_index}] incomplete output ({done}/{len(samples)}); recomputing.", flush=True)

    print(f"[job {job_index}] lines [{start}, {start + len(samples)}) -> {out_path}", flush=True)

    seg_store: List[List[List[str]]] = []
    todo_refs: List[Tuple[int, int, int]] = []
    todo_texts: List[str] = []

    for si, sample in enumerate(samples):
        per_sample: List[List[str]] = []
        for mi, msg in enumerate(sample.get("messages", [])):
            segs = split_on_token(msg.get("content", ""))
            per_sample.append(segs)
            for gi, seg in enumerate(segs):
                if seg == IMAGE_TOKEN or is_trivial(seg):
                    continue
                todo_refs.append((si, mi, gi))
                todo_texts.append(seg)
        seg_store.append(per_sample)

    print(f"[job {job_index}] {len(todo_texts)} segments across {len(samples)} samples.", flush=True)

    # --- deduplicate -----------------------------------------------------------
    text_to_uid: dict = {}
    unique_texts: List[str] = []
    ref_uid: List[int] = []
    for t in todo_texts:
        uid = text_to_uid.get(t)
        if uid is None:
            uid = len(unique_texts)
            text_to_uid[t] = uid
            unique_texts.append(t)
        ref_uid.append(uid)
    if todo_texts:
        print(f"[job {job_index}] dedup: {len(unique_texts)} unique / {len(todo_texts)} total "
              f"({100 * (1 - len(unique_texts) / len(todo_texts)):.1f}% saved).", flush=True)

    # --- load model & translate ------------------------------------------------
    translations_by_uid: List[str] = unique_texts[:]  # fallback = source text
    if unique_texts:
        from vllm import LLM, SamplingParams  # imported here so --help stays fast

        t0 = time.time()

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            seed=args.seed,
            trust_remote_code=True,
        )

        print(f"[job {job_index}] model loaded in {time.time() - t0:.0f}s.", flush=True)

        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        conversations = [build_messages(t) for t in unique_texts]

        t0 = time.time()
        breakpoint()
        outputs = llm.chat(conversations, sampling, use_tqdm=True)
        dt = time.time() - t0
        print(f"[job {job_index}] translated {len(unique_texts)} strings in {dt:.0f}s "
              f"({len(unique_texts) / max(dt, 1e-9):.1f} str/s).", flush=True)

        for uid, out in enumerate(outputs):
            translations_by_uid[uid] = clean_translation(out.outputs[0].text)

    # --- write the translated segments back ------------------------------------
    for ref_i, (si, mi, gi) in enumerate(todo_refs):
        seg_store[si][mi][gi] = translations_by_uid[ref_uid[ref_i]]

    n_written = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for si, sample in enumerate(samples):
            for mi, msg in enumerate(sample.get("messages", [])):
                msg["content"] = "".join(seg_store[si][mi])
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_written += 1
    os.replace(tmp_path, out_path)  # atomic: never leave a half-written file

    print(f"[job {job_index}] DONE: wrote {n_written} samples -> {out_path}", flush=True)


if __name__ == "__main__":
    main()