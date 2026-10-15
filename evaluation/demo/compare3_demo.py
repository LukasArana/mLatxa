# Configure Gradio to use a user-writable temp directory
import os

GRADIO_TMP_DIR = os.environ.get("GRADIO_TMP_DIR", os.path.expanduser("~/.gradio_tmp"))
os.makedirs(GRADIO_TMP_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TMP_DIR

import argparse
import threading

import gradio as gr
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer


# =====================================================================
# Command Line Arguments
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Side-by-side comparison of 4 multimodal Latxa/Qwen models")
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs=4,
        default=[
            "/proiektuak/ilenia/models-instruct/Latxa-Qwen3.5-9B-v1_multimodal_4",
            "/proiektuak/ilenia/models-instruct/Latxa-Qwen3.5-9B-v1-multimodal",
            "/proiektuak/ilenia/models-instruct/Latxa-Qwen3.5-9B-v2-multimodal_1024",
            "Qwen/Qwen3.5-9B",
        ],
        help="The 4 model paths/HF ids to compare",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs=4,
        default=[
            "Latxa-Qwen3.5-9B-v1_multimodal_4",
            "Latxa-Qwen3.5-9B-v1-multimodal",
            "Latxa-Qwen3.5-9B-v2-multimodal_1024",
            "Qwen3.5-9B",
        ],
        help="Display names for the 4 models",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs=4,
        default=["cuda:0", "cuda:0", "cuda:1", "cuda:1"],
        help=(
            "CUDA device for each of the 4 models, e.g. 'cuda:0 cuda:0 cuda:1 cuda:1' (2 GPUs, "
            "2 models packed on each) or 'cuda:0 cuda:1 cuda:2 cuda:3' (1 GPU per model). "
            "Each model is pinned entirely to its device (no CPU offload) so all four "
            "stay resident in VRAM simultaneously."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens per answer")
    parser.add_argument("--gradio-port", type=int, default=7863, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    return parser.parse_args()


args = parse_args()

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful Artificial Intelligence assistant called Latxa, "
    "created and developed by HiTZ, the Basque Center for Language Technology research center. "
    "Answer the user's questions about the provided image, or chat normally if no image is given."
)

# Serializes GPU generation so concurrent Gradio users don't OOM the card
GEN_LOCK = threading.Lock()


# =====================================================================
# Model Loading
# =====================================================================

def load_model(path, device):
    print(f"Loading model: {path} -> {device}", flush=True)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype="auto",
        # Pin the whole model to a single device instead of "auto": this fails loudly
        # with CUDA OOM if it doesn't fit, rather than silently spilling to CPU RAM,
        # which would defeat the goal of keeping all 4 models resident in VRAM at once.
        device_map={"": device},
    )
    model.eval()
    print(f"Loaded: {path}", flush=True)
    return processor, model


MODELS = []  # list of {"name": str, "path": str, "device": str, "processor": ..., "model": ..., "error": str | None}

for name, path, device in zip(args.model_names, args.model_paths, args.devices):
    entry = {"name": name, "path": path, "device": device, "processor": None, "model": None, "error": None}
    try:
        entry["processor"], entry["model"] = load_model(path, device)
    except Exception as e:
        entry["error"] = str(e)
        print(f"Warning: could not load {name} ({path}) on {device}: {e}", flush=True)
    MODELS.append(entry)


# =====================================================================
# Generation
# =====================================================================

def stream_answer(entry, system_prompt, message, image, enable_thinking, max_new_tokens, temperature, top_p, top_k):
    """Stream an answer from a single model entry. Yields growing text."""
    if entry["model"] is None:
        yield f"⚠️ Model failed to load: {entry['error']}"
        return

    processor, model = entry["processor"], entry["model"]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": message})
    messages.append({"role": "user", "content": content})

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    ).to(model.device)

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(max_new_tokens),
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        top_k=int(top_k),
    )

    with GEN_LOCK:
        thread = threading.Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        partial = ""
        # Stream raw text as-is: <think>...</think> blocks are never stripped.
        for chunk in streamer:
            partial += chunk
            yield partial
        thread.join()


def compare(system_prompt, message, image, enable_thinking, max_new_tokens, temperature, top_p, top_k):
    message = (message or "").strip()
    if not message and image is None:
        gr.Warning("Write a message and/or upload an image first.")
        yield "", "", ""
        return

    outputs = ["⏳ ..." for _ in MODELS]
    yield tuple(outputs)

    for i, entry in enumerate(MODELS):
        for outputs[i] in stream_answer(
            entry, system_prompt, message, image, enable_thinking, max_new_tokens, temperature, top_p, top_k
        ):
            yield tuple(outputs)


# =====================================================================
# Gradio Interface
# =====================================================================

with gr.Blocks(title="Multimodal 4-Way Compare") as demo:
    model_list_md = "\n".join(f"        {i}. `{name}`" for i, name in enumerate(args.model_names, start=1))
    gr.Markdown(
        f"""
        # 🖼️ Multimodal 4-Way Compare
        Upload an image (optional), write a prompt, and compare the responses of:
{model_list_md}
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Image (optional)")
        with gr.Column(scale=2):
            message_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe this image / Ask a question about it...",
                lines=4,
            )
            with gr.Accordion("Options", open=False):
                system_prompt = gr.Textbox(
                    label="System prompt",
                    value=DEFAULT_SYSTEM_PROMPT,
                    lines=3,
                )
                enable_thinking = gr.Checkbox(
                    value=False,
                    label="Enable thinking",
                    info="Lets the models reason inside <think>...</think> before answering. The raw <think> tags are always shown, never stripped.",
                )
                with gr.Row():
                    max_new_tokens = gr.Slider(16, 2048, value=args.max_new_tokens, step=16, label="Max new tokens")
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Top-P")
                    top_k = gr.Slider(1, 100, value=20, step=1, label="Top-K")
            submit_btn = gr.Button("🚀 Compare", variant="primary")

    with gr.Row():
        outputs = [
            gr.Textbox(label=name, lines=16, interactive=False)
            for name in args.model_names
        ]

    inputs = [system_prompt, message_input, image_input, enable_thinking, max_new_tokens, temperature, top_p, top_k]

    submit_btn.click(compare, inputs=inputs, outputs=outputs)
    message_input.submit(compare, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    print("\n" + "=" * 80, flush=True)
    print("Starting Gradio interface...", flush=True)
    for entry in MODELS:
        status = "✓ loaded" if entry["model"] is not None else f"✗ FAILED ({entry['error']})"
        print(f"  [{entry['device']}] {entry['name']} ({entry['path']}): {status}", flush=True)
    print("=" * 80 + "\n", flush=True)

    demo.launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.gradio_port,
        show_error=True,
    )
