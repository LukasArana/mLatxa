"""
Prediction Comparison Viewer
=============================
Streamlit app that:
  1. Reads one or more prediction files (JSON array or JSONL), each record like:
     {"idx": 0, "question_id": "test_01535", "question": "...", "prediction": "..."}
  2. Reads a HuggingFace dataset that has an id column (e.g. "example_id") and
     an "image" column.
  3. Shows a table with columns: question_id | image | <prediction file 1> | ...

Run with the sidebar UI (upload files by hand, type in the dataset):
    streamlit run app.py

Or pass everything on the command line instead:
    streamlit run app.py -- \\
        --files preds_model_a.jsonl preds_model_b.jsonl \\
        --dataset your-org/your-dataset \\
        --id-column example_id

Everything after "--" is forwarded to this script as sys.argv, which is why
the CLI flags come after that separator. All flags are optional — anything
not passed on the command line can still be filled in / overridden in the
sidebar.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys

import pandas as pd
import streamlit as st
from datasets import load_dataset
from PIL import Image

st.set_page_config(page_title="Prediction Comparison Viewer", layout="wide")
st.title("Prediction Comparison Viewer")
st.caption(
    "Compare predictions from several files side by side, with the source image "
    "pulled in from a HuggingFace dataset."
)


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_cli_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--files", nargs="+", default=None, help="Paths to local prediction files")
    parser.add_argument("--dataset", default=None, help="HF dataset repo id")
    parser.add_argument("--config", default=None, help="HF dataset config name")
    parser.add_argument("--split", default="test", help="HF dataset split")
    parser.add_argument("--token", default=None, help="HF token (for private/gated datasets)")
    parser.add_argument("--id-column", dest="id_column", default="example_id", help="Dataset id column")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only compare the first N question_ids (default: all)"
    )
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    return args


cli_args = parse_cli_args()

# ---------------------------------------------------------------------------
# Sidebar inputs — pre-filled from CLI args when given, editable either way
# ---------------------------------------------------------------------------
st.sidebar.header("1. Prediction files")
if cli_args.files:
    st.sidebar.success("Using files from the command line:\n" + "\n".join(f"- {f}" for f in cli_args.files))
    uploaded_files = None
else:
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more prediction files (.jsonl or .json)",
        type=["jsonl", "json", "txt"],
        accept_multiple_files=True,
    )

st.sidebar.header("2. HuggingFace dataset")
hf_dataset_name = st.sidebar.text_input("Dataset repo id (e.g. 'user/dataset')", value=cli_args.dataset or "")
hf_config = st.sidebar.text_input("Config name (optional)", value=cli_args.config or "")
hf_split = st.sidebar.text_input("Split", value=cli_args.split or "test")
hf_token = st.sidebar.text_input(
    "HF token (optional, for private/gated datasets)", value=cli_args.token or "", type="password"
)
hf_id_column = st.sidebar.text_input(
    "Dataset column that matches question_id",
    value=cli_args.id_column,
    help=(
        "The column in the HF dataset that corresponds to the prediction files' "
        "'question_id' field. Change this if your dataset uses a different name."
    ),
)

st.sidebar.header("3. How many to compare")
instance_limit = st.sidebar.number_input(
    "Number of instances to compare (0 = all)",
    min_value=0,
    value=cli_args.limit if cli_args.limit is not None else 0,
    step=1,
    help="Caps the table to the first N question_ids (in the order they appear in your files). Set to 0 for all.",
)

load_button = st.sidebar.button("Load / Refresh", type="primary")

# If enough was passed on the CLI, auto-run once on startup so you don't have
# to click the button by hand.
auto_load = bool(cli_args.files and cli_args.dataset and "auto_loaded" not in st.session_state)
if auto_load:
    st.session_state["auto_loaded"] = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_predictions_text(text: str) -> dict:
    """Parse prediction file contents (JSON array or JSONL) into {question_id: prediction}."""
    text = text.strip()
    records = {}
    if not text:
        return records

    # Try as a single JSON array first.
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for row in data:
                qid = row.get("question_id")
                if qid is not None:
                    records[qid] = row.get("prediction", "")
            return records
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL: one JSON object per line.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        qid = row.get("question_id")
        if qid is not None:
            records[qid] = row.get("prediction", "")
    return records


def load_prediction_sources(uploaded_files, cli_files):
    """Returns {display_name: {qid: prediction}} from either uploads or local file paths."""
    file_records = {}
    if cli_files:
        for path in cli_files:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            file_records[path] = parse_predictions_text(text)
    elif uploaded_files:
        for f in uploaded_files:
            file_records[f.name] = parse_predictions_text(f.getvalue().decode("utf-8"))
    return file_records


def image_to_data_uri(img) -> str:
    """Convert a PIL image (as stored in a HF dataset) to a base64 data URI."""
    if img is None or not isinstance(img, Image.Image):
        return ""
    buf = io.BytesIO()
    fmt = (img.format or "PNG").upper()
    try:
        img.save(buf, format=fmt)
    except Exception:
        fmt = "PNG"
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


@st.cache_data(show_spinner=False)
def load_hf_dataset(name: str, config: str | None, split: str, token: str | None):
    kwargs = {}
    if config:
        kwargs["name"] = config
    if token:
        kwargs["token"] = token
    return load_dataset(name, split=split, **kwargs)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
if load_button or auto_load:
    if not cli_args.files and not uploaded_files:
        st.warning("Please upload at least one prediction file (or pass --files on the command line).")
        st.stop()
    if not hf_dataset_name:
        st.warning("Please provide a HuggingFace dataset repo id (or pass --dataset on the command line).")
        st.stop()

    with st.spinner("Parsing prediction files..."):
        file_records = load_prediction_sources(uploaded_files, cli_args.files)
        all_qids = []
        seen = set()
        for recs in file_records.values():
            for qid in recs:
                if qid not in seen:
                    seen.add(qid)
                    all_qids.append(qid)

    if instance_limit and instance_limit > 0:
        all_qids = all_qids[: int(instance_limit)]

    with st.spinner(f"Loading dataset '{hf_dataset_name}'..."):
        try:
            ds = load_hf_dataset(hf_dataset_name, hf_config or None, hf_split, hf_token or None)
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    if hf_id_column not in ds.column_names:
        st.error(f"The dataset does not have a '{hf_id_column}' column. Available columns: {ds.column_names}")
        st.stop()
    if "image" not in ds.column_names:
        st.error("The dataset does not have an 'image' column.")
        st.stop()

    with st.spinner("Finding matching rows without decoding all images..."):
        # Only look at the id column first -- this avoids triggering image
        # decoding for the whole dataset, which is the expensive part.
        id_only = ds.select_columns([hf_id_column])
        wanted = set(all_qids)
        matching_indices = [i for i, row in enumerate(id_only) if row[hf_id_column] in wanted]

    with st.spinner(f"Decoding images for {len(matching_indices)} matching row(s)..."):
        selected = ds.select(matching_indices)
        qid_to_image = {row[hf_id_column]: row["image"] for row in selected}

    with st.spinner("Building comparison table..."):
        rows = []
        missing = []
        for qid in all_qids:
            row_data = {"question_id": qid}
            img = qid_to_image.get(qid)
            if img is None:
                missing.append(qid)
            row_data["image"] = image_to_data_uri(img)
            for fname, recs in file_records.items():
                row_data[fname] = recs.get(qid, "")
            rows.append(row_data)

        df = pd.DataFrame(rows)

    st.session_state["df"] = df
    st.session_state["file_names"] = list(file_records.keys())
    st.session_state["missing"] = missing

if "df" in st.session_state:
    df = st.session_state["df"]
    file_names = st.session_state["file_names"]
    missing = st.session_state.get("missing", [])
    n = len(df)

    if missing:
        preview = ", ".join(str(m) for m in missing[:10])
        more = "..." if len(missing) > 10 else ""
        st.warning(f"{len(missing)} question_id(s) had no matching image in the dataset: {preview}{more}")

    if n == 0:
        st.info("No matching question_ids to show.")
        st.stop()

    if "page_idx" not in st.session_state:
        st.session_state["page_idx"] = 0
    st.session_state["page_idx"] = max(0, min(st.session_state["page_idx"], n - 1))

    st.subheader(f"Comparing {n} example(s) across {len(file_names)} file(s)")

    # --- Gallery navigation -------------------------------------------------
    nav_prev, nav_jump, nav_next = st.columns([1, 2, 1])
    with nav_prev:
        if st.button("⬅ Previous", use_container_width=True, disabled=st.session_state["page_idx"] <= 0):
            st.session_state["page_idx"] -= 1
            st.rerun()
    with nav_next:
        if st.button("Next ➡", use_container_width=True, disabled=st.session_state["page_idx"] >= n - 1):
            st.session_state["page_idx"] += 1
            st.rerun()
    with nav_jump:
        jump_to = st.number_input(
            f"Example (1–{n})",
            min_value=1,
            max_value=n,
            value=st.session_state["page_idx"] + 1,
            step=1,
        )
        if jump_to - 1 != st.session_state["page_idx"]:
            st.session_state["page_idx"] = jump_to - 1
            st.rerun()

    row = df.iloc[st.session_state["page_idx"]]

    # --- Image, centered -----------------------------------------------------
    st.markdown(
        f"<h4 style='text-align:center;'>question_id: {html.escape(str(row['question_id']))}"
        f" &nbsp;·&nbsp; {st.session_state['page_idx'] + 1} / {n}</h4>",
        unsafe_allow_html=True,
    )

    if row["image"]:
        image_html = (
            '<div style="display:flex; justify-content:center; margin-bottom:1.5rem;">'
            f'<img src="{row["image"]}" style="max-width:520px; max-height:520px; '
            'border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,0.18);" />'
            "</div>"
        )
        st.markdown(image_html, unsafe_allow_html=True)
    else:
        st.info("No image found in the dataset for this question_id.")

    # --- Predictions, one card per file, wrapping onto new lines as needed ---
    cards = ['<div style="display:flex; flex-wrap:wrap; gap:1rem; justify-content:center;">']
    for fname in file_names:
        fname_name = fname.split("results/")[-1]
        pred_text = html.escape(str(row.get(fname, "") or ""))
        card_html = (
            '<div style="flex:0 0 320px; width:320px; border:1px solid rgba(128,128,128,0.35); '
            'border-radius:10px; padding:0.85rem 1.1rem; background:rgba(128,128,128,0.06);">'
            f'<div style="font-weight:600; margin-bottom:0.45rem; word-break:break-all;">{html.escape(fname_name)}</div>'
            f'<div style="white-space:pre-wrap; line-height:1.4;">{pred_text}</div>'
            "</div>"
        )
        cards.append(card_html)
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)

    st.divider()
    csv_bytes = df.drop(columns=["image"]).to_csv(index=False).encode("utf-8")
    st.download_button("Download full table as CSV (without images)", csv_bytes, "comparison.csv", "text/csv")
else:
    st.info(
        "Upload prediction files and set the dataset info in the sidebar (or pass them as "
        "command-line arguments), then click 'Load / Refresh'."
    )