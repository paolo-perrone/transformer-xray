#!/usr/bin/env python3
"""transformer_xray: watch a transformer write the next word of YOUR prompt, block by block.

    python3 transformer_xray.py "The capital of Poland is"
    python3 transformer_xray.py "Your own prompt" --model Qwen/Qwen2.5-0.5B

For one forward pass on your prompt it prints:
  1. the model's best guess for the next token after every block (read through the final layer)
  2. where the final answer comes from: its probability after each block's attention and after
     each block's feed-forward network, and the step where it rises most
  3. for each block, the attention head that pulls the most into the last token, and from where
  4. when the biggest rise is a feed-forward network, it switches that network off and runs again

Runs on a laptop CPU. Needs torch and transformers (pip install torch transformers). Works with
GPT-2-style and Llama-style models from the Hugging Face hub (GPT-2, Llama, Qwen, Mistral, SmolLM).
Exit 0 on success, 1 on a model or prompt it cannot run.
"""
import argparse
import sys


def load(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager",
                                                 dtype=torch.float32)
    model.eval()
    return tok, model


def parts(model):
    """(blocks, final_norm, output_layer) for the two layouts most open models use."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):        # GPT-2 style
        return model.transformer.h, model.transformer.ln_f, model.lm_head
    if hasattr(model, "model") and hasattr(model.model, "layers"):               # Llama style
        return model.model.layers, model.model.norm, model.lm_head
    raise ValueError("unknown layout: this tool reads GPT-2-style and Llama-style models")


def piece(tok, i):
    return repr(tok.decode([int(i)]))


def guess(model, norm, head, h):
    """Probabilities for the next token from one hidden row, read through the final layer."""
    return head(norm(h)).softmax(-1)


def attn_module(blk):
    return blk.attn if hasattr(blk, "attn") else blk.self_attn


def inspect(tok, model, prompt, top=2, ablate=True, out=print):
    import torch
    blocks, norm, head = parts(model)
    ids = tok(prompt, return_tensors="pt").input_ids
    toks = [tok.decode([i]) for i in ids[0]]
    attn_out = {}
    hooks = [attn_module(b).register_forward_hook(
        lambda m, i, o, k=k: attn_out.__setitem__(k, o[0] if isinstance(o, tuple) else o))
        for k, b in enumerate(blocks)]
    try:
        with torch.no_grad():
            res = model(ids, output_hidden_states=True, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()
    probs = res.logits[0, -1].softmax(-1)
    best = int(probs.argmax())
    n = len(blocks)
    name = getattr(model.config, "_name_or_path", "model")
    out(f"transformer_xray · {name} · {n} blocks · {len(toks)} tokens")
    out("tokens        " + " | ".join(f"{t!r} {int(i)}" for t, i in zip(toks, ids[0])))

    def lens(h, last=False):
        # The final hidden state already has the final norm applied (GPT-2 and Llama layouts).
        return (head(h) if last else head(norm(h))).softmax(-1)

    out("")
    out("1. the next-token guess after each block")
    with torch.no_grad():
        for b in range(n):
            p = lens(res.hidden_states[b + 1][0, -1], last=b == n - 1)
            vals, idx = p.topk(top)
            row = "  ".join(f"{piece(tok, i):>12} {100 * v:5.1f}%" for v, i in zip(vals, idx))
            out(f"  block {b + 1:>2}   {row}")

    out("")
    out(f"2. where {piece(tok, best)} comes from: its probability after each step")
    out("  block   after attention   after feed-forward")
    rises = []
    with torch.no_grad():
        for b in range(n):
            pre = float(lens(res.hidden_states[b][0, -1])[best]) if b else 0.0
            mid_h = res.hidden_states[b][0, -1] + attn_out[b][0, -1]
            mid = float(lens(mid_h)[best])
            post = float(lens(res.hidden_states[b + 1][0, -1], last=b == n - 1)[best])
            rises.append((mid - pre, "attention", b))
            rises.append((post - mid, "feed-forward network", b))
            out(f"  {b + 1:>5}   {100 * mid:13.1f}%   {100 * post:16.1f}%")
    up, kind, b = max(rises)
    out(f"  biggest rise: block {b + 1}'s {kind}, +{100 * up:.1f} points")

    out("")
    out("3. attention into the last token, first token left out (it soaks up spare attention)")
    for k, att in enumerate(res.attentions):
        row = att[0, :, -1, 1:] if len(toks) > 1 else att[0, :, -1, :]
        share, tokidx = row.max(-1)
        h = int(share.argmax())
        t = int(tokidx[h]) + (1 if len(toks) > 1 else 0)
        out(f"  block {k + 1:>2}   head {h + 1:>2} puts {float(share[h]):.3f} on {toks[t]!r}")

    result = {"tokens": toks, "best": tok.decode([best]), "p_best": float(probs[best]),
              "rise": (kind, b + 1, up)}
    if ablate and kind == "feed-forward network":
        hook = blocks[b].mlp.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
        try:
            with torch.no_grad():
                q = model(ids).logits[0, -1].softmax(-1)
        finally:
            hook.remove()
        nb = int(q.argmax())
        out("")
        out(f"4. check: switch off block {b + 1}'s feed-forward network and run again")
        out(f"  {piece(tok, best)} {100 * float(probs[best]):.1f}% -> {100 * float(q[best]):.1f}%, "
            f"top guess becomes {piece(tok, nb)} at {100 * float(q[nb]):.1f}%")
        result["ablated"] = (float(q[best]), tok.decode([nb]), float(q[nb]))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prompt", help="the text the model continues")
    ap.add_argument("--model", default="openai-community/gpt2",
                    help="a hub id or a local folder (default: openai-community/gpt2)")
    ap.add_argument("--top", type=int, default=2, help="guesses shown per block (default 2)")
    ap.add_argument("--no-ablate", action="store_true", help="skip the feed-forward check")
    a = ap.parse_args(argv)
    try:
        tok, model = load(a.model)
        inspect(tok, model, a.prompt, a.top, not a.no_ablate)
    except (OSError, ValueError) as e:
        print(f"transformer_xray: cannot run {a.model}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
