"""Investigation 2 (A2): greedy decoding with a KV cache, on the same kernels.

The hand-in model (glm_asr_cuda/model.py) recomputes the whole sequence for every new
token. Here the prompt is run once (prefill) and every layer keeps the keys and values it
computed; each later step runs only the new token and attends to the cache. Every
operation still goes through glm_asr_cuda.ops, so ASR_CUDA / ops.set_cuda_groups decides
whether our kernels or PyTorch run it. Nothing in glm_asr_cuda/ or csrc/ is changed.

Two details:
  * RoPE. ops.rope_ rotates positions 0..S-1 only. For a step at position P the new q
    and k are placed in row P of a zero buffer of length P + 1 and rotated there, so the
    same RoPE code (Task 1 kernel or reference) applies the angle for position P.
  * Attention. ops.attention(q [B,1,H,D], k/v cache [B,P+1,Hkv,D], causal=True) already
    handles seq_q != seq_k: the query sits at position seq_k - 1 and sees every key.
"""

import time

import torch

from glm_asr_cuda import ops


def _sync_ms(t0):
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def prompt_embeds(model, features, input_ids, features_mask):
    """Encoder + projector, and the prompt with audio embeddings swapped in (as model.generate)."""
    audio = model.encode_audio(features, features_mask)
    embeds = ops.embedding(input_ids, model.w["language_model.model.embed_tokens.weight"])
    is_audio = (input_ids[0] == model.c.audio_token_id).nonzero().flatten()
    first, last = int(is_audio[0]), int(is_audio[-1])
    return torch.cat([embeds[:, :first], audio[None], embeds[:, last + 1:]], dim=1)


def _rope_at(model, q, k, pos):
    """Rotate q [B,S,Hq,D], k [B,S,Hk,D] as positions pos .. pos+S-1."""
    if pos == 0:
        ops.rope_(q, k, model.text_inv_freq, model.text_rotary_dim)
        return q, k
    B, S = q.shape[:2]
    qb = q.new_zeros(B, pos + S, *q.shape[2:])
    kb = k.new_zeros(B, pos + S, *k.shape[2:])
    qb[:, pos:] = q
    kb[:, pos:] = k
    ops.rope_(qb, kb, model.text_inv_freq, model.text_rotary_dim)
    return qb[:, pos:].contiguous(), kb[:, pos:].contiguous()


class KVCache:
    def __init__(self, model, batch, max_len, device):
        c = model.c
        shape = (c.text_layers, batch, max_len, c.text_kv_heads, model.text_head_dim)
        self.k = torch.empty(shape, device=device, dtype=torch.float32)
        self.v = torch.empty(shape, device=device, dtype=torch.float32)
        self.len = 0

    def nbytes(self):
        return self.k.numel() * 4 * 2


def decoder_layer_kv(model, h, i, cache, pos):
    """model.decoder_layer, except K/V of earlier positions come from the cache."""
    c, w, p = model.c, model.w, f"language_model.model.layers.{i}."
    B, S, _ = h.shape
    x = ops.rmsnorm(h, w[p + "input_layernorm.weight"], c.rms_eps)
    q = ops.linear(x, w[p + "self_attn.q_proj.weight"]).view(B, S, c.text_heads, model.text_head_dim)
    k = ops.linear(x, w[p + "self_attn.k_proj.weight"]).view(B, S, c.text_kv_heads, model.text_head_dim)
    v = ops.linear(x, w[p + "self_attn.v_proj.weight"]).view(B, S, c.text_kv_heads, model.text_head_dim)
    q, k = _rope_at(model, q, k, pos)
    cache.k[i, :, pos:pos + S] = k
    cache.v[i, :, pos:pos + S] = v
    keys = cache.k[i, :, :pos + S]                   # contiguous: B = 1 and the slice is a prefix
    vals = cache.v[i, :, :pos + S]
    a = ops.attention(q, keys, vals, causal=True).view(B, S, c.text_hidden)
    h = h + ops.linear(a, w[p + "self_attn.o_proj.weight"])

    x = ops.rmsnorm(h, w[p + "post_attention_layernorm.weight"], c.rms_eps)
    gate = ops.linear(x, w[p + "mlp.gate_proj.weight"])
    up = ops.linear(x, w[p + "mlp.up_proj.weight"])
    return h + ops.linear(ops.silu_mul(gate, up), w[p + "mlp.down_proj.weight"])


def _head(model, h):
    last = ops.rmsnorm(h[:, -1], model.w["language_model.model.norm.weight"], model.c.rms_eps)
    logits = ops.linear(last, model.w["language_model.lm_head.weight"])
    return ops.softmax(logits)


@torch.no_grad()
def generate_timed(model, features, input_ids, features_mask, use_cache, max_new_tokens=100):
    """Greedy decoding, with or without the KV cache.

    Returns (tokens, probs, timing) where timing has encoder_ms, prefill_ms (first token),
    step_ms (list, one per later token), decoder_ms and total_ms. Without the cache the
    first step plays the role of prefill so the two modes report the same fields.
    """
    emb_table = model.w["language_model.model.embed_tokens.weight"]
    torch.cuda.synchronize()
    t_all = time.perf_counter()
    embeds = prompt_embeds(model, features, input_ids, features_mask)
    encoder_ms = _sync_ms(t_all)

    tokens, probs, step_ms = [], [], []
    prompt_len = embeds.shape[1]
    cache = KVCache(model, embeds.shape[0], prompt_len + max_new_tokens, embeds.device) if use_cache else None
    x, pos = embeds, 0
    prefill_ms = None
    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        if use_cache:
            h = x
            for i in range(model.c.text_layers):
                h = decoder_layer_kv(model, h, i, cache, pos)
            pos += x.shape[1]
            p = _head(model, h)
        else:
            p = model.next_token_probs(x)              # the hand-in path, unchanged
        best = torch.argmax(p, dim=-1)
        tokens.append(int(best[0]))                    # int() synchronises, like model.generate
        probs.append(float(p[0, best[0]]))
        ms = _sync_ms(t0)
        if prefill_ms is None:
            prefill_ms = ms
        else:
            step_ms.append(ms)
        if tokens[-1] in model.c.eos_token_ids:
            break
        new = ops.embedding(best[:, None], emb_table)
        x = new if use_cache else torch.cat([x, new], dim=1)
    total_ms = _sync_ms(t_all)
    timing = {"encoder_ms": encoder_ms, "prefill_ms": prefill_ms, "step_ms": step_ms,
              "decoder_ms": total_ms - encoder_ms, "total_ms": total_ms,
              "prompt_len": prompt_len,
              "cache_mib": cache.nbytes() / 2**20 if cache else 0.0}
    return tokens, probs, timing


# ---- names used by encoder_trim.py (A5) ----

@torch.no_grad()
def build_prompt_embeds(model, features, input_ids, features_mask):
    """Encoder + projector + prompt embeddings (same as prompt_embeds)."""
    return prompt_embeds(model, features, input_ids, features_mask)


def generate_kv(model, features, input_ids, features_mask, max_new_tokens=100):
    """Greedy decoding with the KV cache. Returns (tokens, probs)."""
    tokens, probs, _ = generate_timed(model, features, input_ids, features_mask, True, max_new_tokens)
    return tokens, probs
