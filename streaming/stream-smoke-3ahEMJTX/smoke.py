import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

from generator import ELMGenerator

out = Path(__file__).with_name("smoke.json")
records = {}
stage = "initialise"
clock = time.perf_counter
try:
    generator = ELMGenerator()
    api = generator.client.chat.completions
    stage = "baseline"
    with patch.object(api, "create", wraps=api.create) as observed:
        baseline = generator.generate("Reply with the single word ready.")
    records["baseline"] = baseline
    request = dict(observed.call_args.kwargs)
    records["request"] = {
        k: request[k] for k in ("model", "messages", "temperature",
                                "max_tokens", "max_completion_tokens") if k in request
    }
    stage = "stream"
    stream = records["stream"] = {"answer": "", "first_text_ms": None,
                                  "completion_ms": None, "finish_reason": None}
    request["stream"] = True
    start = clock()
    with api.create(**request) as chunks:
        for chunk in chunks:
            arrived = clock()
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            piece = choice.delta.content or ""
            if piece:
                if stream["first_text_ms"] is None:
                    stream["first_text_ms"] = round((arrived - start) * 1000, 1)
                stream["answer"] += piece
            if choice.finish_reason:
                stream["finish_reason"] = choice.finish_reason
        elapsed = (clock() - start) * 1000
    if not stream["answer"].strip() or not stream["finish_reason"]:
        raise ValueError("Incomplete stream")
    stream["completion_ms"] = round(elapsed, 1)
    print("Baseline completion (ms):", round(baseline["latency_ms"], 1))
    print("Stream first non-empty text (ms):", stream["first_text_ms"])
    print("Stream completion (ms):", stream["completion_ms"])
except Exception as exc:
    records["failure"] = {"stage": stage, "type": type(exc).__name__,
                          "http_status": getattr(exc, "status_code", None)}
    print("Smoke test failed:", records["failure"])
finally:
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print("Saved:", out)
sys.exit(int("failure" in records))
