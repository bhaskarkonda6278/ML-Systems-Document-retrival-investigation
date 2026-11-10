"""Persistent ASR worker: isolate Py3.11 ASR from the Py3.12 RAG environment."""
import faulthandler
faulthandler.enable()
import signal
faulthandler.register(signal.SIGUSR1, all_threads=True)
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asr-root', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.asr_root.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'partner_asr'))
    with redirect_stdout(sys.stderr):
        print("START: importing torch", file=sys.stderr, flush=True)
        import torch
        print("DONE: importing torch", file=sys.stderr, flush=True)
        from benchmark import read_wav, use_cache_if_present
        from glm_asr_cuda import ops
        from glm_asr_cuda.model import load_model, MODEL_NAME
        from kv_cache import generate_kv
        from encoder_trim import cut, window_frames
        print("START: checking CUDA", file=sys.stderr, flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU allocation required')
        print("DONE: checking CUDA", file=sys.stderr, flush=True)
        ops.set_cuda_groups('none')
        print("START: checking model cache", file=sys.stderr, flush=True)
        use_cache_if_present(MODEL_NAME)
        print("DONE: checking model cache", file=sys.stderr, flush=True)
        print("START: loading model", file=sys.stderr, flush=True)
        model, processor = load_model()
        print("DONE: loading model", file=sys.stderr, flush=True)
        faulthandler.cancel_dump_traceback_later()
    print(json.dumps({'ready': True, 'gpu': torch.cuda.get_device_name(),
                      'torch': torch.__version__, 'cuda': torch.version.cuda}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('action') == 'close':
                break
            with redirect_stdout(sys.stderr), torch.no_grad():
                start = time.perf_counter()
                audio = read_wav(Path(request['audio_path']))
                inputs = processor.apply_transcription_request(audio)
                features = inputs['input_features'].to('cuda', torch.float32)
                ids = inputs['input_ids'].to('cuda')
                mask = inputs['input_features_mask'].to('cuda')
                real = int(mask[0].sum())
                frames = features.shape[-1]
                mode = request['mode']
                if mode == 'kv_trim':
                    if frames != 3000 or not bool(mask[0, :real].all()):
                        raise RuntimeError('Unexpected mel-feature padding layout')
                    frames = window_frames('trim', real)
                    features, mask = cut(features, mask, frames)
                elif mode != 'baseline':
                    raise ValueError('Unknown ASR mode')
                torch.cuda.synchronize()
                prepared = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                function = model.generate if mode == 'baseline' else lambda *a, **kw: generate_kv(model, *a, **kw)
                if mode == 'baseline':
                    tokens, _ = function(features, ids, mask, max_new_tokens=100)
                else:
                    tokens, _ = function(features, ids, mask, max_new_tokens=100)
                torch.cuda.synchronize()
                inferred = time.perf_counter()
                text = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                result = {'transcript': text, 'tokens': tokens, 'mode': mode,
                    'real_frames': real, 'frames': frames, 'hit_max_tokens': len(tokens) >= 100,
                    'preprocessing_ms': (prepared-start)*1000,
                    'inference_ms': (inferred-prepared)*1000,
                    'asr_ms': (time.perf_counter()-start)*1000,
                    'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30}
        except Exception as exc:
            result = {'error': type(exc).__name__ + ': ' + str(exc)}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
