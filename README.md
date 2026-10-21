# transformer-xray

> Tested on 5 shipped config.json files (GPT-2, Llama 3 70B, Gemma 2 9B, DeepSeek-V3, Granite 4.0-H Small) on 2026-10-21 with Python 3.14; CI runs 3.9, 3.12 and 3.13. Weight totals match the published counts for the 3 dense models. Not yet run on a mixture-of-experts weight count, which it declines to guess.

Companion to [What is a Transformer?](https://theaiengineer.substack.com/p/what-is-a-transformer), The AI Engineer.

For engineers who serve or call a language model and need to know, before the bill or the out-of-memory error, how much GPU memory each conversation takes and whether a prompt fits.

It reads the config.json your model already ships with and counts the key-value cache block by block: shared key-value heads, compressed latents, sliding-window blocks and Mamba blocks each counted as they are. Then it checks your prompt against the window, and with --gpu-gb, counts how many conversations of your length fit beside the weights.

    python3 transformer_xray.py path/to/config.json --prompt my_prompt.txt --answer 1024 --gpu-gb 80

One file, standard-library Python, no GPU and no API key. Exit 0 when the prompt fits, 3 when prompt plus answer is over the window, 1 on an input it cannot read.

## What it prints, on cases with a known answer

DeepSeek-V3 stores a compressed latent where the textbook formula assumes full keys and values for all 128 heads:

```
python3 transformer_xray.py fixtures/deepseek-v3.json
```

```
transformer_xray · deepseek-v3.json (deepseek_v3)
blocks        61 blocks, 7,168 wide, 128 query heads, keys and values compressed to 576 numbers per token
window        163,840 tokens
weights       not counted for this layout (mixture of experts or hybrid); the cache below still is
cache         70,272 bytes per token (bfloat16); the full-heads formula says 4,997,120, 71.1 times too much
              11.51 GB for one conversation at the full window
```

Check it by hand: (512 + 64) numbers x 61 blocks x 2 bytes = 70,272 bytes per token.

The request from an OpenAI forum report, 1,750 tokens of messages with 8,100 reserved for the answer, on Llama 3 70B's 8,192-token window:

```
python3 transformer_xray.py fixtures/llama-3-70b.json --tokens 1750 --answer 8100
```

```
transformer_xray · llama-3-70b.json (llama)
blocks        80 blocks, 8,192 wide, 64 query heads sharing 8 key-value heads
window        8,192 tokens
weights       70,553,706,496 (feed-forward 79.9%, attention 17.1%, token tables 3.0%); 141.11 GB in bfloat16
cache         327,680 bytes per token (bfloat16); the full-heads formula says 2,621,440, 8.0 times too much
              2.68 GB for one conversation at the full window
prompt        1,750 tokens (given)
              0.57 GB of cache, 1,532,125 attention scores per head per full block
fits          NO: 1,750 prompt + 8,100 answer = 9,850, over the 8,192-token window by 1,658
```

Gemma 2 9B on a 24 GB card, at its full window:

```
python3 transformer_xray.py fixtures/gemma-2-9b.json --tokens 8192 --gpu-gb 24
```

```
transformer_xray · gemma-2-9b.json (gemma2)
blocks        42 blocks, 3,584 wide, 16 query heads sharing 8 key-value heads
              of 42: 21 read the whole context, 21 read the last 4,096 tokens
window        8,192 tokens
weights       9,241,705,984 (feed-forward 70.1%, attention 20.0%, token tables 9.9%); 18.48 GB in bfloat16
cache         344,064 bytes per token (bfloat16); the full-heads formula says 688,128, 2.0 times too much
              2.11 GB for one conversation at the full window
prompt        8,192 tokens (given)
              2.11 GB of cache, 33,558,528 attention scores per head per full block
fits          yes: 8,192 of 8,192 tokens, 0 to spare
gpu           24 GB minus 18.48 GB of weights leaves 5.52 GB: 2 conversations of 8,192 tokens fit
```

## On your own model

Pass the config.json, the folder that holds it, or a public hub id (`openai-community/gpt2`). A gated model's config has to be downloaded first. For an exact token count, `pip install tokenizers` and put the model's tokenizer.json beside the config; without it the prompt count is an estimate at 4 characters per token, and the output says so.

## What it counts, and what it does not

- **Cache:** per block, 2 x key-value heads x head size for full and sliding blocks (sliding blocks stop at their window), the latent size for compressed-latent blocks, nothing for Mamba blocks, whose state stays a fixed size.
- **Weights:** exact for GPT-2 and for dense Llama, Mistral, Qwen 2 and 3, and Gemma layouts. Mixture-of-experts and hybrid layouts are not counted, and the output says so instead of guessing.
- **Not counted:** activations, the serving engine's own overhead, and the cache block rounding engines like vLLM add. Treat the conversation count as a ceiling.

## Tests

    python3 -m unittest test_transformer_xray -v

Each test is a number you can rebuild by hand from the config: GPT-2's 124,439,808 weights match the tensor count of its weights file, Llama 3 70B's cache is 2 x 80 x 8 x 128 x 2 bytes per token.

## Support

Tested 2026-10-21. Supported through 2027-10. Issues and pull requests welcome.

MIT License.
