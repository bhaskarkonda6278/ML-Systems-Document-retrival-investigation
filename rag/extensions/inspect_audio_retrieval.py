"""Inspect ASR, retrieval and evidence without fabricating an ELM answer."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.asr_adapter import ASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", choices=["none", "all"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = CourseData(ROOT / "data/rag_corpus")
    item = data.audio_items("practice")[0]
    with ASRAdapter(cuda_groups=args.cuda) as asr:
        asr.prepare()
        start = time.perf_counter()
        transcript = asr.transcribe(ROOT / "data/rag_corpus" / item["audio_file"])
        asr_ms = (time.perf_counter() - start) * 1000
    encoder = LocalEncoder(ROOT / "models/enc_a", device="cpu")
    vector = encoder.encode(transcript)
    directory = ROOT / "data/rag_corpus/embeddings"
    retrieved = ExactDenseRetriever(directory / "corpus_enc_a.npy", directory / "corpus_enc_a_ids.json").retrieve(vector, k=5)
    evidence = SimpleContextBuilder().build(transcript, retrieved["record_ids"], data)
    row = {"query_id": item["query_id"], "cuda_groups": args.cuda,
           "transcript": transcript, "asr_ms_excluding_model_load": asr_ms,
           "retrieval": retrieved, "contexts": evidence["contexts"],
           "prompt_characters": len(evidence["prompt"]),
           "generation_status": "Not run: awaiting the user's ELM API key"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
