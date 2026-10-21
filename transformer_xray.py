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
  5. with --context FILE: whether the answer comes from that context or from training, by running
     the prompt with and without it, and which blocks read the context most

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


def context_check(tok, model, context, prompt, out=print):
    """Did the answer come from the context or from training? Runs the prompt with and without it."""
    import torch
    blocks, _norm, _head = parts(model)
    c_ids = tok(context.rstrip() + "\n", return_tensors="pt").input_ids[0]
    p_ids = tok(prompt, return_tensors="pt").input_ids[0]
    both = torch.cat([c_ids, p_ids]).unsqueeze(0)
    with torch.no_grad():
        w = model(both, output_attentions=True)
        wo = model(p_ids.unsqueeze(0))
    pw, po = w.logits[0, -1].softmax(-1), wo.logits[0, -1].softmax(-1)
    a_w, a_o = int(pw.argmax()), int(po.argmax())
    out("")
    out(f"5. context or training? ({len(c_ids)} context tokens)")
    out(f"  with the context:     {piece(tok, a_w)} at {100 * float(pw[a_w]):.1f}%")
    extra = "" if a_o == a_w else f", and {piece(tok, a_w)} at {100 * float(po[a_w]):.1f}%"
    out(f"  without the context:  {piece(tok, a_o)} at {100 * float(po[a_o]):.1f}%{extra}")
    gain = float(pw[a_w]) - float(po[a_w])
    if a_w != a_o:
        verdict = "from the context: without it, the model writes a different answer"
    elif gain >= 0.10:
        verdict = (f"from training, and the context backs it up: +{100 * gain:.1f} points "
                   "for the same answer")
    else:
        verdict = "from training: the model writes the same answer without the context"
    out(f"  verdict: the answer comes {verdict}")
    n_ctx = len(c_ids)
    share = []
    for b, att in enumerate(w.attentions):
        row = att[0, :, -1, :]                        # heads x tokens, for the last token
        share.append((float(row[:, 1:n_ctx].sum(-1).mean()), b))   # first token left out
    share.sort(reverse=True)
    out("  blocks whose attention reads the context most (average share over heads): " +
        ", ".join(f"block {b + 1} {100 * s:.0f}%" for s, b in share[:3]))
    return {"with": (tok.decode([a_w]), float(pw[a_w])), "without": (tok.decode([a_o]), float(po[a_o])),
            "verdict": verdict, "top_block": share[0][1] + 1}


# ------------------------------------------------------------------ API mode
# Closed models expose no blocks, so this mode answers only the question that matters in production:
# with your context, does the model you ship give a different answer than without it?
# Standard library only. Keys come from OPENAI_API_KEY or ANTHROPIC_API_KEY.

ASK = "Answer in as few words as possible."


def _post(url, headers, body):
    import json
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def ask_api(provider, model, question, context=None, post=_post):
    """(answer text, {token: probability} for the first answer token, or None) from one API call."""
    import math
    import os
    text = (f"Context:\n{context.strip()}\n\nQuestion: {question}\n{ASK}" if context
            else f"Question: {question}\n{ASK}")
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("set OPENAI_API_KEY")
        r = post("https://api.openai.com/v1/chat/completions",
                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                 {"model": model, "messages": [{"role": "user", "content": text}], "temperature": 0,
                  "max_tokens": 30, "logprobs": True, "top_logprobs": 5})
        ch = r["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        first = {c["token"]: math.exp(c["logprob"]) for c in lp[0]["top_logprobs"]} if lp else None
        return ch["message"]["content"].strip(), first
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("set ANTHROPIC_API_KEY")
        r = post("https://api.anthropic.com/v1/messages",
                 {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                 {"model": model, "max_tokens": 30, "temperature": 0,
                  "messages": [{"role": "user", "content": text}]})
        return "".join(b.get("text", "") for b in r["content"]).strip(), None
    raise ValueError(f"unknown provider {provider!r}: use openai or anthropic")


def _norm(s):
    return " ".join(s.lower().strip(" .!\n\"'").split())


def api_check(provider, model, question, context, runs=3, out=print, post=_post):
    """Majority answer with and without the context, over `runs` calls each, and a verdict."""
    from collections import Counter
    res = {}
    for label, ctx in (("with", context), ("without", None)):
        answers, firsts = [], []
        for _ in range(runs):
            a, f = ask_api(provider, model, question, ctx, post)
            answers.append(a)
            firsts.append(f)
        top, n = Counter(_norm(a) for a in answers).most_common(1)[0]
        shown = next(a for a in answers if _norm(a) == top)
        res[label] = (shown, n, firsts[0])
    (aw, nw, fw), (ao, no, fo) = res["with"], res["without"]
    out(f"transformer_xray · {provider} · {model} · {runs} calls each, temperature 0")
    out(f"  with the context:     {aw!r}  ({nw} of {runs} calls)")
    out(f"  without the context:  {ao!r}  ({no} of {runs} calls)")
    if fw and fo:
        tok = max(fw, key=fw.get)
        out(f"  first answer token {tok!r}: {100 * fw[tok]:.1f}% with the context, "
            f"{100 * fo.get(tok, 0):.1f}% without")
    if _norm(aw) != _norm(ao):
        verdict = "from the context: without it, the model gives a different answer"
    else:
        verdict = "from training: the model gives the same answer without the context"
    out(f"  verdict: the answer comes {verdict}")
    return {"with": aw, "without": ao, "verdict": verdict}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prompt", help="the text the model continues")
    ap.add_argument("--model", default="openai-community/gpt2",
                    help="a hub id or a local folder (default: openai-community/gpt2)")
    ap.add_argument("--top", type=int, default=2, help="guesses shown per block (default 2)")
    ap.add_argument("--no-ablate", action="store_true", help="skip the feed-forward check")
    ap.add_argument("--api", choices=["openai", "anthropic"],
                    help="check a closed model through its API (needs --context and --model)")
    ap.add_argument("--runs", type=int, default=3, help="API calls per side in --api mode (default 3)")
    ap.add_argument("--context", help="a text file with the context your app pastes before the prompt "
                    "(retrieved passages, a document); compares the answer with and without it")
    a = ap.parse_args(argv)
    try:
        if a.api:
            if not a.context or a.model == "openai-community/gpt2":
                raise ValueError("--api needs --context FILE and --model, e.g. --model gpt-4o-mini")
            api_check(a.api, a.model, a.prompt, open(a.context, encoding="utf-8").read(), a.runs)
            return 0
        tok, model = load(a.model)
        if a.context:
            context = open(a.context, encoding="utf-8").read()
            full = context.rstrip() + "\n" + a.prompt
            inspect(tok, model, full, a.top, not a.no_ablate)
            context_check(tok, model, context, a.prompt)
        else:
            inspect(tok, model, a.prompt, a.top, not a.no_ablate)
    except (OSError, ValueError, KeyError) as e:
        print(f"transformer_xray: cannot run {a.model}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
