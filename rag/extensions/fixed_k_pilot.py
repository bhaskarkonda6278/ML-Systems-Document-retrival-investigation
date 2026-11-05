import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever
from src.context_builder import SimpleContextBuilder
from src.generator import ELMGenerator

load_dotenv(ROOT / ".env")

output_dir = ROOT / "results/fixed_k_pilot"
questions = [
    json.loads(line)
    for line in (output_dir / "inputs.jsonl").read_text().splitlines()
    if line.strip()
]

# Preserve any previous run rather than overwrite its answers.
ks = [1, 3, 5, 10, 20]
for k in ks:
    if (output_dir / f"answers_k{k}.jsonl").exists():
        raise SystemExit(f"Results for k={k} already exist. Stop and check them.")

data_dir = ROOT / "data/rag_corpus"
data = CourseData(data_dir)
encoder = LocalEncoder(ROOT / "models/enc_a", device="cpu")
retriever = ExactDenseRetriever(
    data_dir / "embeddings/corpus_enc_a.npy",
    data_dir / "embeddings/corpus_enc_a_ids.json",
)
builder = SimpleContextBuilder()
generator = ELMGenerator()

# Encode each question once; every k uses exactly the same vector.
vectors = {
    row["query_id"]: encoder.encode(row["transcript"])
    for row in questions
}

# Warm up retrieval without making an ELM request.
retriever.retrieve(vectors[questions[0]["query_id"]], k=20)

# Interleave settings to reduce dependence on when each setting runs.
schedule = [(row, k) for row in questions for k in ks]
random.Random(42).shuffle(schedule)

for row, k in schedule:
    start = time.perf_counter()
    retrieved = retriever.retrieve(vectors[row["query_id"]], k=k)
    built = builder.build(row["transcript"], retrieved["record_ids"], data)

    generation_start = time.perf_counter()
    response = generator.client.chat.completions.create(
        model=generator.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use the supplied evidence to answer the user's question. "
                    "Do not invent facts that are unsupported by the evidence."
                ),
            },
            {"role": "user", "content": built["prompt"]},
        ],
        temperature=0.1,
    )
    generation_ms = (time.perf_counter() - generation_start) * 1000
    downstream_ms = (time.perf_counter() - start) * 1000
    usage = response.usage

    result = {
        **row,
        "k": k,
        "model": generator.model,
        "answer": response.choices[0].message.content or "",
        "retrieved_record_ids": retrieved["record_ids"],
        "retrieval_scores": retrieved["scores"],
        "prompt": built["prompt"],
        "context_characters": sum(len(text) for text in built["contexts"]),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "timing": {
            "retrieval_ms": retrieved["latency_ms"],
            "generation_ms": generation_ms,
            "downstream_ms": downstream_ms,
        },
    }

    with (output_dir / f"answers_k{k}.jsonl").open("a") as output:
        output.write(json.dumps(result) + "\n")

    print(
        f'{row["query_id"]} | k={k:2} | '
        f'tokens={result["prompt_tokens"]} | '
        f'downstream={downstream_ms:.0f} ms',
        flush=True,
    )

print("Finished: 25 answers saved.")
