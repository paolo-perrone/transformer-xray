# transformer-xray

> Tested 2026-10-21 on GPT-2 small (openai-community/gpt2) on an Apple M3 Pro, torch 2.14.1, transformers 5.20.2, and on tiny random GPT-2 and Llama test models. Not yet run on a full-size Llama-style checkpoint. The --api mode is tested on recorded OpenAI and Anthropic responses, not yet on a live call.

Companion to [What is a Transformer?](https://theaiengineer.substack.com/p/what-is-a-transformer), The AI Engineer.

For engineers who build on language models and need to know where an answer came from: the context your app pasted in, or what the model learned in training. It also shows which word the model is guessing after each block, and which part of the model the answer depends on.

It runs one forward pass of an open model on your prompt and prints the next-token guess after every block, the answer's probability after each attention step and each feed-forward step, the token each block's strongest attention head reads, and, when the answer comes from a feed-forward network, what happens when you switch that network off.

    pip install torch transformers
    python3 transformer_xray.py "Your question" --context retrieved.txt

Runs on a laptop CPU, no GPU and no API key. GPT-2 small by default; pass --model for any GPT-2-style or Llama-style model on the Hugging Face hub or on disk.

## What it prints, on the issue's example

```
python3 transformer_xray.py "The capital of Poland is"
```

```
transformer_xray · openai-community/gpt2 · 12 blocks · 5 tokens
tokens        'The' 464 | ' capital' 3139 | ' of' 286 | ' Poland' 12873 | ' is' 318

1. the next-token guess after each block
  block  1         ' not'  34.5%        ' now'  10.6%
  block  2         ' not'  27.2%        ' now'  22.4%
  block  3         ' now'  35.2%        ' not'  16.6%
  block  4         ' now'  51.2%       ' also'  13.5%
  block  5         ' now'  57.7%        ' not'   7.8%
  block  6         ' now'  52.7%      ' still'   8.4%
  block  7         ' now'  50.4%        ' not'  10.3%
  block  8         ' now'  62.5%       ' also'   7.9%
  block  9     ' located'  40.1%        ' now'  34.0%
  block 10      ' Poland'  87.2%     ' Warsaw'   5.6%
  block 11      ' Warsaw'  43.1%     ' Poland'  39.5%
  block 12      ' Warsaw'  10.3%        ' the'   9.1%

2. where ' Warsaw' comes from: its probability after each step
  block   after attention   after feed-forward
      1             0.0%                0.0%
      2             0.0%                0.0%
      3             0.0%                0.0%
      4             0.0%                0.0%
      5             0.0%                0.0%
      6             0.0%                0.0%
      7             0.0%                0.0%
      8             0.0%                0.0%
      9             0.0%                0.0%
     10             3.6%                5.6%
     11            18.1%               43.1%
     12            12.5%               10.3%
  biggest rise: block 11's feed-forward network, +25.0 points

3. attention into the last token, first token left out (it soaks up spare attention)
  block  1   head  2 puts 0.977 on ' is'
  block  2   head 12 puts 0.855 on ' is'
  block  3   head 10 puts 0.477 on ' Poland'
  block  4   head 12 puts 0.852 on ' capital'
  block  5   head 12 puts 1.000 on ' Poland'
  block  6   head  8 puts 0.455 on ' capital'
  block  7   head  2 puts 0.348 on ' of'
  block  8   head  9 puts 0.294 on ' capital'
  block  9   head  8 puts 0.294 on ' is'
  block 10   head  9 puts 0.837 on ' Poland'
  block 11   head  1 puts 0.561 on ' Poland'
  block 12   head 11 puts 0.505 on ' Poland'

4. check: switch off block 11's feed-forward network and run again
  ' Warsaw' 10.3% -> 2.6%, top guess becomes ' the' at 9.9%
```

" Warsaw" barely registers until block 10, rises most in block 11's feed-forward network, and falls to 2.6 percent when that network is switched off. The prompt never says Warsaw, so the capital comes from what GPT-2 learned in training.

## Did the answer come from your context or from training?

If your app pastes retrieved passages or a document before the question, pass them with --context. It runs the prompt with and without them and tells you which one the answer came from.

    python3 transformer_xray.py "The capital of Poland is" --context examples/wrong_fact.txt

A context that says the capital is Kraków changes the answer:

```
5. context or training? (11 context tokens)
  with the context:     ' K' at 67.2%
  without the context:  ' Warsaw' at 10.3%, and ' K' at 0.3%
  verdict: the answer comes from the context: without it, the model writes a different answer
  blocks whose attention reads the context most (average share over heads): block 1 40%, block 2 38%, block 12 33%
```

A context that is about Poland but never names the capital leaves the trained answer in place, and only makes the model surer of it:

```
5. context or training? (17 context tokens)
  with the context:     ' Warsaw' at 27.0%
  without the context:  ' Warsaw' at 10.3%
  verdict: the answer comes from training, and the context backs it up: +16.7 points for the same answer
  blocks whose attention reads the context most (average share over heads): block 1 47%, block 2 45%, block 3 22%
```

Run it on a question your RAG pipeline answers wrong. If the verdict says the answer came from training, the retrieved passage never reached the answer, so look at what you retrieved and where you put it in the prompt.

## The model you ship: --api

Closed models like GPT and Claude don't expose their blocks, so on them the tool answers the production question only: with your context, does the model give a different answer than without it? It calls the model 3 times on each side at temperature 0 and compares the majority answers. On OpenAI models it also prints the first answer token's probability with and without the context.

    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY
    python3 transformer_xray.py "What is the capital of Poland?" --context retrieved.txt --api openai --model gpt-4o-mini

The verdict reads "from the context" when the answers differ and "from training" when they match. Standard library only in this mode; torch is not needed.

## On your own prompt and model

    python3 transformer_xray.py "Your prompt" --model Qwen/Qwen2.5-0.5B
    python3 transformer_xray.py "Your prompt" --top 3 --no-ablate

The guess after each block is read by sending that block's output through the model's final layer. Attention lists leave out the first token, which soaks up attention a head has nowhere else to put.

## Also in the repo: kv_cache.py

How much GPU memory each token of a conversation takes, read from a model's config.json, counted block by block (shared key-value heads, compressed latents, sliding-window blocks and Mamba blocks each as they are):

```
python3 kv_cache.py fixtures/deepseek-v3.json
```

```
kv_cache · deepseek-v3.json (deepseek_v3)
blocks        61 blocks, 7,168 wide, 128 query heads, keys and values compressed to 576 numbers per token
window        163,840 tokens
weights       not counted for this layout (mixture of experts or hybrid); the cache below still is
cache         70,272 bytes per token (bfloat16); the full-heads formula says 4,997,120, 71.1 times too much
              11.51 GB for one conversation at the full window
```

Standard-library Python. See `python3 kv_cache.py --help`.

## Tests

    python3 -m unittest test_transformer_xray test_kv_cache -v

The GPT-2 tests check the numbers above, the Kraków context changing the answer and the neutral one leaving it: " Warsaw" at 10.3 percent, the biggest rise in block 11's feed-forward network, 2.6 percent with it switched off, and block 10's head 9 putting 0.837 on " Poland".

## Support

Tested 2026-10-21. Supported through 2027-10. Issues and pull requests welcome. MIT License.
