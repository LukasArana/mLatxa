# MMStar Evaluation - Comparison of Approaches

## Critical Bug Fixed
**The original script (`mmstar_qwen3_5.py`) never sent images to the model!** It extracted `img = i["image"]` but never included it in the payload. This made the evaluation meaningless for a vision-language model.

## Three Approaches

### 1. ❌ Original (Broken)
**File:** `mmstar_qwen3_5.py` + `mmstar_qwen3.5.sh`

**Issues:**
- 🚨 **Images never sent to model**
- Uses wrong endpoint (`/v1/completions` instead of `/v1/chat/completions`)
- Inefficient: recreates few-shot examples on every iteration
- Raw HTTP requests instead of OpenAI client
- No metrics calculation or result saving
- No error handling

---

### 2. ✅ Server + OpenAI Client (Recommended for flexibility)
**Files:** `mmstar_openai_client.py` + `mmstar_openai_client.sh`

**Advantages:**
- ✅ **Actually sends images properly** (base64-encoded data URIs)
- Uses OpenAI client library (cleaner code)
- Uses `/v1/chat/completions` endpoint (multimodal support)
- Few-shot examples created once
- Saves results incrementally to JSONL
- Calculates accuracy
- Better error handling
- Can connect to existing/shared vLLM server

**Use when:**
- You want to share a server across multiple evaluations
- Running multiple concurrent evaluations
- Need flexibility to switch models without reloading

**Performance:**
- Memory: ~Model size (loaded in server)
- Speed: Network overhead per request (minimal if localhost)
- Startup: Wait for server initialization

---

### 3. 🚀 Offline vLLM (Most Efficient)
**Files:** `mmstar_offline.py` + `mmstar_offline.sh`

**Advantages:**
- ✅ **Most efficient** - no server needed
- ✅ **Batched inference** (8 examples at once by default)
- ✅ Loads model once, processes all examples
- No network overhead
- Better GPU utilization through batching
- Saves results incrementally

**Use when:**
- Running single evaluation task
- Want maximum throughput
- Have dedicated GPU resources
- Don't need concurrent access

**Performance:**
- Memory: ~Model size (loaded directly)
- Speed: **Fastest** due to batching
- Startup: One-time model load then continuous inference

---

## Performance Comparison

For 1500 examples on a single GPU:

| Approach | Images Sent? | Time Estimate | GPU Efficiency | Code Complexity |
|----------|--------------|---------------|----------------|-----------------|
| Original | ❌ No        | N/A (broken)  | 0%             | Simple          |
| Server   | ✅ Yes       | ~15-20 min    | Medium         | Simple          |
| Offline  | ✅ Yes       | ~8-12 min     | **High**       | Simple          |

---

## Usage Examples

### Server Approach
```bash
cd /sorgin1/users/larana/LatxaTxat/evaluation/scripts
./mmstar_openai_client.sh
```

### Offline Approach (Recommended)
```bash
cd /sorgin1/users/larana/LatxaTxat/evaluation/scripts
./mmstar_offline.sh
```

---

## Output Format

Both improved scripts save results to JSONL with this format:

```json
{
  "idx": 0,
  "question": "What color is the sky?",
  "ground_truth": "blue",
  "prediction": "blue",
  "correct": true
}
```

Final accuracy is printed and saved to the results file.

---

## Recommendation

**Use the offline approach (`mmstar_offline.sh`)** for:
- Single evaluation runs
- Maximum efficiency
- Best GPU utilization

**Use the server approach (`mmstar_openai_client.sh`)** for:
- Connecting to existing servers
- Running multiple evaluations concurrently
- Flexibility to change models frequently
