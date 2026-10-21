#!/usr/bin/env python3
"""transformer_xray: what one model's transformer costs you, read from its config.json.

    python3 transformer_xray.py <config.json | model folder | hub id> [--prompt FILE | --tokens N]
                                [--answer N] [--gpu-gb G] [--conversations C]

It reads the configuration file a model ships with and prints:
  1. the blocks: how many, how wide, and how attention shares its keys and values
  2. where the weights sit (dense models): feed-forward, attention, token tables
  3. the key-value cache per token, counted block by block, so shared key-value heads,
     compressed latents, sliding-window blocks and Mamba blocks are each counted as they are
  4. your prompt: its tokens, its cache, and whether prompt plus answer fits the window
  5. with --gpu-gb: how many conversations of your length fit beside the weights

Standard library only. The token count is exact when the `tokenizers` package is installed
and a tokenizer.json sits beside the config; otherwise it says it is an estimate.
Exit 0 when the prompt fits, 3 when prompt plus answer is over the window, 1 on an input
it cannot read.
"""
import argparse
import json
import math
import os
import sys
import urllib.request

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float8_e4m3fn": 1}


# ------------------------------------------------------------------ reading the config
def load(src):
    """(config, folder or None, label). src is a config.json, a folder holding one, or a hub id."""
    if os.path.isdir(src):
        path = os.path.join(src, "config.json")
        return json.load(open(path, encoding="utf-8")), src, os.path.basename(os.path.abspath(src))
    if os.path.isfile(src):
        return json.load(open(src, encoding="utf-8")), os.path.dirname(os.path.abspath(src)), os.path.basename(src)
    if src.count("/") == 1 and not src.startswith((".", "/")):
        url = f"https://huggingface.co/{src}/resolve/main/config.json"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8")), None, src
        except Exception as e:
            raise ValueError(f"could not fetch {url}: {e}. A gated model needs its config.json "
                             f"downloaded first; pass the file instead.")
    raise ValueError(f"{src}: not a config.json, a folder holding one, or a hub id like owner/name")


def g(c, *keys, default=None):
    for k in keys:
        if c.get(k) is not None:
            return c[k]
    return default


class Model:
    def __init__(self, c):
        self.c = c
        self.kind = c.get("model_type", "unknown")
        self.layers = g(c, "num_hidden_layers", "n_layer")
        self.width = g(c, "hidden_size", "n_embd")
        self.heads = g(c, "num_attention_heads", "n_head")
        if not (self.layers and self.width and self.heads):
            raise ValueError("the config names no layer count, width or head count, so this is not a "
                             "transformer config this tool can read")
        self.kv_heads = g(c, "num_key_value_heads", default=self.heads)
        self.head_dim = g(c, "head_dim", default=self.width // self.heads)
        self.window = g(c, "max_position_embeddings", "n_positions", "max_sequence_length")
        self.vocab = g(c, "vocab_size")
        dtype = g(c, "torch_dtype", "dtype", default="float32" if self.kind == "gpt2" else "bfloat16")
        self.dtype = str(dtype).replace("torch.", "")
        self.bytes = DTYPE_BYTES.get(self.dtype, 2)
        self.types = self.layer_types()

    def layer_types(self):
        """One entry per block: full, sliding, latent (compressed keys and values) or mamba."""
        c = self.c
        lt = c.get("layer_types") or c.get("layers_block_type")
        if c.get("kv_lora_rank"):
            return ["latent"] * self.layers
        if lt:
            m = {"full_attention": "full", "attention": "full", "sliding_attention": "sliding",
                 "mamba": "mamba", "linear_attention": "mamba"}
            return [m.get(x, "full") for x in lt][:self.layers]
        if c.get("sliding_window") and c.get("use_sliding_window", True) and self.kind in ("mistral",):
            return ["sliding"] * self.layers
        return ["full"] * self.layers

    # -------------------------------------------------------------- the cache
    def numbers_per_token(self, t):
        """Numbers one block stores per token in the key-value cache."""
        if t == "mamba":
            return 0
        if t == "latent":
            return self.c["kv_lora_rank"] + self.c.get("qk_rope_head_dim", 0)
        return 2 * self.kv_heads * self.head_dim

    def cache_bytes(self, tokens):
        """Cache bytes for one conversation of `tokens` tokens, block by block."""
        win = self.c.get("sliding_window") or 0
        total = 0
        for t in self.types:
            held = min(tokens, win) if (t == "sliding" and win) else tokens
            total += self.numbers_per_token(t) * held
        return total * self.bytes

    def naive_cache_bytes(self, tokens):
        """The textbook formula: every block keeps full keys and values for every head."""
        kdim = self.head_dim
        if self.c.get("kv_lora_rank"):
            kdim = self.c.get("qk_nope_head_dim", 0) + self.c.get("qk_rope_head_dim", 0)
        vdim = self.c.get("v_head_dim", self.head_dim)
        return self.layers * self.heads * (kdim + vdim) * tokens * self.bytes

    # -------------------------------------------------------------- the weights
    def weights(self):
        """{part: count} for dense layouts this tool can count exactly, else None."""
        c, H, L = self.c, self.width, self.layers
        q, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        if self.kind == "gpt2":
            ff = g(c, "n_inner", default=4 * H)
            att = (H * 3 * H + 3 * H) + (H * H + H)
            mlp = (H * ff + ff) + (ff * H + H)
            norms = L * 4 * H + 2 * H
            tables = self.vocab * H + self.window * H
            return {"feed-forward": L * mlp, "attention": L * att, "token tables": tables, "norms": norms}
        if self.kind in ("llama", "mistral", "qwen2", "qwen3", "gemma2", "gemma"):
            if any(g(c, k) for k in ("num_local_experts", "num_experts", "n_routed_experts")):
                return None
            ff = c["intermediate_size"]
            bias = (q + 2 * kv) if (c.get("attention_bias") or self.kind == "qwen2") else 0
            att = H * q + 2 * H * kv + q * H + bias
            if self.kind == "qwen3":
                att += 2 * self.head_dim            # q_norm and k_norm
            mlp = 3 * H * ff
            per_norms = 4 if self.kind == "gemma2" else 2
            norms = L * per_norms * H + H
            tied = c.get("tie_word_embeddings", self.kind.startswith("gemma"))
            tables = self.vocab * H * (1 if tied else 2)
            return {"feed-forward": L * mlp, "attention": L * att, "token tables": tables, "norms": norms}
        return None


# ------------------------------------------------------------------ the prompt
def count_tokens(text, folder):
    """(count, how). Exact with `tokenizers` and a tokenizer.json, else an estimate."""
    tj = os.path.join(folder, "tokenizer.json") if folder else None
    if tj and os.path.exists(tj):
        try:
            from tokenizers import Tokenizer
            return len(Tokenizer.from_file(tj).encode(text, add_special_tokens=False).ids), "tokenizer.json"
        except ImportError:
            pass
    return math.ceil(len(text) / 4), "estimate at 4 characters per token: pip install tokenizers and put tokenizer.json beside the config for the exact count"


# ------------------------------------------------------------------ output
def gb(b):
    return f"{b / 1e9:,.2f} GB"


def report(m, label, prompt_tokens=None, how=None, answer=0, gpu_gb=None, conversations=1, out=print):
    out(f"transformer_xray · {label} ({m.kind})")
    n = len(m.types)
    kinds = {t: m.types.count(t) for t in ("full", "sliding", "latent", "mamba") if t in m.types}
    shape = f"{m.layers} blocks, {m.width:,} wide, {m.heads} query heads"
    if "latent" in kinds:
        shape += f", keys and values compressed to {m.numbers_per_token('latent')} numbers per token"
    elif m.kv_heads != m.heads:
        shape += f" sharing {m.kv_heads} key-value heads"
    out(f"blocks        {shape}")
    if len(kinds) > 1:
        parts = []
        if "full" in kinds:
            parts.append(f"{kinds['full']} read the whole context")
        if "sliding" in kinds:
            parts.append(f"{kinds['sliding']} read the last {m.c.get('sliding_window'):,} tokens")
        if "latent" in kinds:
            parts.append(f"{kinds['latent']} store a compressed latent")
        if "mamba" in kinds:
            parts.append(f"{kinds['mamba']} are Mamba, with a fixed-size state and no per-token cache")
        out(f"              of {n}: " + ", ".join(parts))
    out(f"window        {m.window:,} tokens" if m.window else "window        not stated in the config")

    w = m.weights()
    if w:
        total = sum(w.values())
        split = ", ".join(f"{k} {100 * v / total:.1f}%" for k, v in w.items() if k != "norms")
        out(f"weights       {total:,} ({split}); {gb(total * m.bytes)} in {m.dtype}")
    else:
        out("weights       not counted for this layout (mixture of experts or hybrid); the cache below still is")

    per = m.cache_bytes(1)
    naive = m.naive_cache_bytes(1)
    line = f"cache         {per:,} bytes per token ({m.dtype})"
    if naive > per * 1.05:
        line += f"; the full-heads formula says {naive:,}, {naive / per:.1f} times too much"
    out(line)
    if m.window:
        out(f"              {gb(m.cache_bytes(m.window))} for one conversation at the full window")

    code = 0
    if prompt_tokens is not None:
        out(f"prompt        {prompt_tokens:,} tokens ({how})")
        out(f"              {gb(m.cache_bytes(prompt_tokens))} of cache, "
            f"{prompt_tokens * (prompt_tokens + 1) // 2:,} attention scores per head per full block")
        need = prompt_tokens + answer
        if m.window and need > m.window:
            out(f"fits          NO: {prompt_tokens:,} prompt + {answer:,} answer = {need:,}, "
                f"over the {m.window:,}-token window by {need - m.window:,}")
            code = 3
        elif m.window:
            out(f"fits          yes: {need:,} of {m.window:,} tokens, {m.window - need:,} to spare")
    if gpu_gb is not None:
        length = (prompt_tokens or 0) + answer if prompt_tokens is not None else (m.window or 0)
        wbytes = sum(w.values()) * m.bytes if w else None
        if code == 3:
            out("gpu           not counted: no conversation of this length runs, since it is over the window")
        elif wbytes is None:
            out("gpu           weights not counted, so the space they leave is unknown")
        else:
            left = gpu_gb * 1e9 - wbytes
            each = m.cache_bytes(length)
            if left <= 0:
                out(f"gpu           the weights alone, {gb(wbytes)}, do not fit in {gpu_gb:g} GB")
            else:
                fit = int(left // each) if each else 0
                out(f"gpu           {gpu_gb:g} GB minus {gb(wbytes)} of weights leaves {gb(left)}: "
                    f"{fit:,} conversations of {length:,} tokens fit")
                if conversations > fit:
                    out(f"              you asked for {conversations:,}: short by {conversations - fit:,}")
    return code


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", help="config.json, a folder holding one, or a hub id like openai-community/gpt2")
    p.add_argument("--prompt", help="a text file holding the prompt you send")
    p.add_argument("--tokens", type=int, help="prompt length in tokens, when you know it")
    p.add_argument("--answer", type=int, default=0, help="tokens you reserve for the answer (max_tokens)")
    p.add_argument("--gpu-gb", type=float, help="GPU memory in GB, to count conversations that fit")
    p.add_argument("--conversations", type=int, default=1, help="conversations you want to serve at once")
    a = p.parse_args(argv)
    try:
        c, folder, label = load(a.model)
        m = Model(c)
    except (OSError, ValueError, KeyError) as e:
        print(f"transformer_xray: cannot read {a.model}: {e}", file=sys.stderr)
        return 1
    tokens, how = a.tokens, "given"
    if a.prompt:
        try:
            text = open(a.prompt, encoding="utf-8").read()
        except OSError as e:
            print(f"transformer_xray: cannot read {a.prompt}: {e}", file=sys.stderr)
            return 1
        tokens, how = count_tokens(text, folder)
        how = f"{os.path.basename(a.prompt)}, {how}"
    return report(m, label, tokens, how, a.answer, a.gpu_gb, a.conversations)


if __name__ == "__main__":
    sys.exit(main())
