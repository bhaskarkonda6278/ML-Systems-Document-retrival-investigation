#!/usr/bin/env python3
"""A5: encode only the real audio instead of the full 30 s window.

The processor pads every clip to 30 s (3000 mel frames), so the audio encoder does
the same work for a 2 s question as for a 30 s one (A4: encoder time is flat). This
script cuts the mel features to the real audio length (plus an optional margin or a
fixed bucket) before the encoder, then decodes with the KV cache.

The trim changes what the encoder sees: its attention is not masked, so the padded
frames normally take part in it. So every window is checked for quality:

  part 1, timing:  a few clips, every window x kernel set, warm-ups + timed runs
  part 2, quality: every clip of a split (or --limit of them), one run per window;
                   tokens identical to the full window? word error rate (WER) against
                   the full-window transcript; one transcript cache file per window,
                   which scripts/evaluate_rag.py --transcript-cache turns into
                   Recall@k / MRR / NDCG (does retrieval still find the same documents?)

Windows (mel frames, 100 per second, each rounded up to a multiple of 8 so the
stride-2 conv and the 4-frame pooling stay aligned):
  full      the processor's 3000 frames (the baseline, as in model.generate)
  trim      the real audio only
  trim+1s   the real audio + 100 frames
  bucket5   rounded up to the next multiple of 500 frames (5 s, 10 s, ...): only 6
            possible shapes, which is friendlier to batching and to CUDA Graphs

Run from hw1-asr (next to kv_cache.py), on the H200 3g node; see a5_trim.sbatch.
"""

import argparse
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FULL = 3000
WINDOWS = ["full", "trim", "trim+1s", "bucket5"]


# ---------------- helpers (no GPU needed) ----------------

def window_frames(name, real, full=FULL):
    """Number of mel frames to keep for window `name`, given `real` frames of audio."""
    def up8(n):
        return -(-n // 8) * 8
    if name == "full":
        return full
    if name == "trim":
        n = up8(real)
    elif name == "trim+1s":
        n = up8(real + 100)
    elif name == "bucket5":
        n = -(-real // 500) * 500
    else:
        raise ValueError(f"unknown window {name!r}")
    return int(min(full, max(n, 8)))


def norm_words(s):
    return re.sub(r"[^\w\s']", " ", s.lower()).split()


def wer(hyp, ref):
    """Word error rate of hyp against ref (Levenshtein over words)."""
    h, r = norm_words(hyp), norm_words(ref)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


def wav_seconds(path):
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def pick_by_duration(items, n):
    """n items at evenly spaced duration quantiles (same idea as the A4 clip choice)."""
    items = sorted(items, key=lambda it: it["seconds"])
    if n >= len(items):
        return items
    idx = np.round(np.linspace(0, len(items) - 1, n)).astype(int)
    return [items[i] for i in idx]


def cut(feats, mask, frames):
    """First `frames` mel frames of the features and the mask (padding is at the end)."""
    return feats[..., :frames].contiguous(), mask[..., :frames].contiguous()


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--rag-data", type=Path, default=HERE.parent / "rag" / "data" / "rag_corpus",
                    help="folder with student_audio_manifest.jsonl and audio/")
    ap.add_argument("--split", default="practice", choices=["practice", "dev"])
    ap.add_argument("--windows", default=",".join(WINDOWS))
    ap.add_argument("--timing-clips", type=int, default=10,
                    help="clips for part 1, at evenly spaced durations (0 = skip part 1)")
    ap.add_argument("--timing-kernels", default="none;all", help="kernel sets for part 1, ';'-separated")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="part 2: first N clips of the split (0 = all)")
    ap.add_argument("--quality-kernels", default="none",
                    help="kernel set for part 2 (default none: fastest; A4 showed identical "
                         "transcripts for none and all)")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("a5_out"))
    args = ap.parse_args()
    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    if windows[0] != "full":
        windows = ["full"] + [w for w in windows if w != "full"]
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from benchmark import use_cache_if_present
    from glm_asr_cuda import ops
    from glm_asr_cuda.model import MODEL_NAME, load_model
    from kv_cache import build_prompt_embeds, generate_kv

    if not torch.cuda.is_available():
        sys.exit("No GPU visible. Run this on a GPU node.")
    gpu = torch.cuda.get_device_name()
    print(f"GPU: {gpu}")
    use_cache_if_present(MODEL_NAME)
    (model, processor), load_ms = ops.wall_clock_ms(load_model)
    print(f"Model loaded in {load_ms / 1000:.1f} s (not timed)")

    manifest = args.rag_data / "student_audio_manifest.jsonl"
    items = [json.loads(l) for l in manifest.open(encoding="utf-8") if l.strip()]
    items = [it for it in items if it["split"] == args.split]
    for it in items:
        it["path"] = args.rag_data / it["audio_file"]
        it["seconds"] = wav_seconds(it["path"])
    print(f"{len(items)} {args.split} clips, {min(i['seconds'] for i in items):.2f} to "
          f"{max(i['seconds'] for i in items):.2f} s")

    from benchmark import read_wav

    def prepare(it):
        audio = read_wav(it["path"])
        inputs = processor.apply_transcription_request(audio)
        feats = inputs["input_features"].to("cuda", torch.float32)
        ids = inputs["input_ids"].to("cuda")
        mask = inputs["input_features_mask"].to("cuda")
        real = int(mask[0].sum())
        if feats.shape[-1] != FULL or not bool(mask[0, :real].all()):
            raise RuntimeError(f"{it['query_id']}: expected {FULL} frames with the real audio first")
        return feats, ids, mask, real

    def decode(tokens):
        return processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    # ---------- part 1: timing ----------
    timing_rows = []
    if args.timing_clips > 0:
        clips = pick_by_duration(items, args.timing_clips)
        print("\n=== Part 1: timing on " + ", ".join(f"{c['query_id']} ({c['seconds']:.2f} s)" for c in clips))
        for it in clips:
            feats, ids, mask, real = prepare(it)
            for ks in [k.strip() for k in args.timing_kernels.split(";") if k.strip()]:
                ops.set_cuda_groups(ks)
                ref_tokens = None
                for w in windows:
                    n = window_frames(w, real)
                    f, m = cut(feats, mask, n)
                    for _ in range(args.warmup):
                        generate_kv(model, f, ids, m, args.max_new_tokens)
                    enc = [ops.wall_clock_ms(build_prompt_embeds, model, f, ids, m)[1]
                           for _ in range(args.runs)]
                    torch.cuda.reset_peak_memory_stats()
                    tot, tokens = [], None
                    for _ in range(args.runs):
                        (tokens, _), ms = ops.wall_clock_ms(generate_kv, model, f, ids, m,
                                                            args.max_new_tokens)
                        tot.append(ms)
                    if ref_tokens is None:
                        ref_tokens = tokens
                    row = {"query_id": it["query_id"], "clip_s": round(it["seconds"], 2),
                           "kernels": ks, "window": w, "frames": n, "real_frames": real,
                           "encoder_ms": round(float(np.median(enc)), 1),
                           "encoder_iqr": round(float(np.subtract(*np.percentile(enc, [75, 25]))), 1),
                           "total_ms": round(float(np.median(tot)), 1),
                           "total_iqr": round(float(np.subtract(*np.percentile(tot, [75, 25]))), 1),
                           "tokens": len(tokens), "same_tokens_as_full": tokens == ref_tokens,
                           "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           "transcript": decode(tokens)}
                    timing_rows.append(row)
                    print("RESULT_JSON " + json.dumps(row, ensure_ascii=False), flush=True)

        print(f"\n{'clip':>10} {'s':>6} {'kernels':>7} {'window':>8} {'frames':>6} {'enc ms':>8} "
              f"{'total ms':>9} {'enc x':>6} {'tot x':>6} same")
        base = {(r["query_id"], r["kernels"]): r for r in timing_rows if r["window"] == "full"}
        for r in timing_rows:
            b = base[(r["query_id"], r["kernels"])]
            print(f"{r['query_id']:>10} {r['clip_s']:6.2f} {r['kernels']:>7} {r['window']:>8} "
                  f"{r['frames']:6d} {r['encoder_ms']:8.1f} {r['total_ms']:9.1f} "
                  f"{b['encoder_ms'] / r['encoder_ms']:6.2f} {b['total_ms'] / r['total_ms']:6.2f} "
                  f"{'YES' if r['same_tokens_as_full'] else 'NO'}")
        (args.out / "a5_timing.json").write_text(json.dumps(timing_rows, indent=1, ensure_ascii=False))

    # ---------- part 2: quality over the split ----------
    qset = items[: args.limit] if args.limit else items
    ops.set_cuda_groups(args.quality_kernels)
    print(f"\n=== Part 2: quality on {len(qset)} {args.split} clips, kernels {ops.describe()}")
    caches = {w: (args.out / f"a5_transcripts_{args.split}_{w.replace('+', 'p')}.jsonl").open(
        "w", encoding="utf-8") for w in windows}
    qrows, t0 = [], time.time()
    for i, it in enumerate(qset, 1):
        feats, ids, mask, real = prepare(it)
        ref_tokens = ref_text = None
        for w in windows:
            n = window_frames(w, real)
            f, m = cut(feats, mask, n)
            torch.cuda.synchronize()
            s = time.perf_counter()
            tokens, _ = generate_kv(model, f, ids, m, args.max_new_tokens)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - s) * 1000
            text = decode(tokens)
            if ref_tokens is None:
                ref_tokens, ref_text = tokens, text
            row = {"query_id": it["query_id"], "clip_s": round(it["seconds"], 2), "window": w,
                   "frames": n, "ms": round(ms, 1), "tokens": len(tokens),
                   "same_tokens_as_full": tokens == ref_tokens,
                   "wer_vs_full": round(wer(text, ref_text), 4),
                   "hit_max_tokens": len(tokens) >= args.max_new_tokens, "transcript": text}
            qrows.append(row)
            caches[w].write(json.dumps({"query_id": it["query_id"], "transcript": text},
                                       ensure_ascii=False) + "\n")
        if i % 25 == 0 or i == len(qset):
            print(f"  {i}/{len(qset)} clips, {time.time() - t0:.0f} s", flush=True)
    for fh in caches.values():
        fh.close()
    (args.out / f"a5_quality_{args.split}.json").write_text(json.dumps(qrows, indent=1, ensure_ascii=False))

    print(f"\n{'window':>8} {'clips':>6} {'identical':>10} {'mean WER':>9} {'max WER':>8} "
          f"{'hit max':>8} {'mean ms':>8} {'frames':>7}")
    for w in windows:
        rs = [r for r in qrows if r["window"] == w]
        print(f"{w:>8} {len(rs):6d} {sum(r['same_tokens_as_full'] for r in rs) / len(rs):9.1%} "
              f"{np.mean([r['wer_vs_full'] for r in rs]):9.4f} {max(r['wer_vs_full'] for r in rs):8.3f} "
              f"{sum(r['hit_max_tokens'] for r in rs):8d} {np.mean([r['ms'] for r in rs]):8.1f} "
              f"{np.mean([r['frames'] for r in rs]):7.0f}")
    worst = sorted((r for r in qrows if r["window"] != "full"), key=lambda r: -r["wer_vs_full"])[:5]
    if worst and worst[0]["wer_vs_full"] > 0:
        print("\nLargest differences from the full window:")
        full_text = {r["query_id"]: r["transcript"] for r in qrows if r["window"] == "full"}
        for r in worst:
            if r["wer_vs_full"] == 0:
                break
            print(f"  {r['query_id']} {r['window']} WER {r['wer_vs_full']:.2f}\n"
                  f"    full: {full_text[r['query_id']]}\n    {r['window']:>4}: {r['transcript']}")
    print(f"\nTranscript caches for evaluate_rag.py are in {args.out}/")


if __name__ == "__main__":
    main()
