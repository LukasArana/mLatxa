"""Rewrite a cached (`swift export`) dataset so `messages[]` matches the text dataset schema.

The cached text datasets store messages as  {role, content, loss: float64}
while the multimodal ones store             {role, content, loss: bool, loss_scale: float64}.
`concatenate_datasets` refuses to align the two. Both `loss` and `loss_scale` are always
None, so we drop `loss_scale` and recast `loss` to float64 (values stay None).

Only the `messages` column is touched; images, lengths, etc. are copied as-is.
Shards are processed one record batch at a time, so memory stays low.

Usage:
    python normalize_messages_schema.py SRC_DIR DST_DIR
    e.g. python normalize_messages_schema.py \
        ~/scratch/data/multimodal_eu_124/train ~/scratch/data/multimodal_eu_124_fixed/train
"""
import argparse
import json
import os
import shutil

import pyarrow as pa
import pyarrow.compute as pc

TARGET_ITEM = pa.struct([("role", pa.string()), ("content", pa.string()), ("loss", pa.float64())])
TARGET_FEATURE = [{
    "role": {"dtype": "string", "_type": "Value"},
    "content": {"dtype": "string", "_type": "Value"},
    "loss": {"dtype": "float64", "_type": "Value"},
}]


def fix_messages(col: pa.Array) -> pa.Array:
    items = col.values  # flattened struct array (keeps list offsets intact below)
    for name in ("loss", "loss_scale"):
        if name in [f.name for f in items.type]:
            vals = pc.struct_field(items, name)
            if vals.null_count != len(vals):
                raise ValueError(f"messages[].{name} has non-null values; refusing to drop/recast")
    new_items = pa.StructArray.from_arrays(
        [
            pc.struct_field(items, "role"),
            pc.struct_field(items, "content"),
            pa.nulls(len(items), pa.float64()),
        ],
        fields=list(TARGET_ITEM),
        mask=items.is_null() if items.null_count else None,
    )
    return pa.ListArray.from_arrays(col.offsets, new_items, mask=col.is_null() if col.null_count else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        raise SystemExit(f"{args.dst} already exists; refusing to overwrite")
    os.makedirs(args.dst)

    with open(os.path.join(args.src, "state.json")) as f:
        state = json.load(f)
    with open(os.path.join(args.src, "dataset_info.json")) as f:
        info = json.load(f)
    info["features"]["messages"] = TARGET_FEATURE
    hf_meta = json.dumps({"info": {"features": info["features"]}}).encode()

    total = 0
    # only the shards listed in state.json; ignore leftover cache-*.arrow files
    for entry in state["_data_files"]:
        name = entry["filename"]
        reader = pa.ipc.open_stream(os.path.join(args.src, name))
        schema = reader.schema
        idx = schema.get_field_index("messages")
        new_schema = schema.set(idx, pa.field("messages", pa.list_(TARGET_ITEM))).with_metadata({b"huggingface": hf_meta})
        n = 0
        with pa.OSFile(os.path.join(args.dst, name), "wb") as sink, pa.ipc.new_stream(sink, new_schema) as writer:
            for batch in reader:
                cols = list(batch.columns)
                cols[idx] = fix_messages(batch.column(idx))
                writer.write_batch(pa.RecordBatch.from_arrays(cols, schema=new_schema))
                n += batch.num_rows
        total += n
        print(f"{name}: {n} rows")

    state["_fingerprint"] = state["_fingerprint"] + "_lossf64"
    with open(os.path.join(args.dst, "state.json"), "w") as f:
        json.dump(state, f, indent=2)
    with open(os.path.join(args.dst, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"done: {total} rows -> {args.dst}")

    # sanity check: loads with HF datasets and has the expected messages feature
    from datasets import load_from_disk
    ds = load_from_disk(args.dst)
    assert len(ds) == total, (len(ds), total)
    print(ds.features["messages"])


if __name__ == "__main__":
    main()
