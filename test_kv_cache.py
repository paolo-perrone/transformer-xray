"""Known answers for kv_cache, each one reconstructible by hand from the config."""
import io
import os
import unittest
from contextlib import redirect_stdout

import kv_cache as tx

HERE = os.path.dirname(os.path.abspath(__file__))


def model(name):
    c, _folder, _label = tx.load(os.path.join(HERE, "fixtures", name + ".json"))
    return tx.Model(c)


def run(*argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = tx.main(list(argv))
    return code, buf.getvalue()


class Weights(unittest.TestCase):
    def test_gpt2_matches_its_weights_file(self):
        # 124,439,808 is the tensor count of openai-community/gpt2's model.safetensors,
        # causal-mask buffers left out.
        self.assertEqual(sum(model("gpt2").weights().values()), 124_439_808)

    def test_llama_3_70b(self):
        self.assertEqual(sum(model("llama-3-70b").weights().values()), 70_553_706_496)

    def test_gemma_2_9b(self):
        self.assertEqual(sum(model("gemma-2-9b").weights().values()), 9_241_705_984)

    def test_mixture_of_experts_is_not_guessed(self):
        self.assertIsNone(model("deepseek-v3").weights())


class Cache(unittest.TestCase):
    def test_llama_shares_8_key_value_heads(self):
        # 2 (keys, values) x 80 blocks x 8 heads x 128 numbers x 2 bytes
        self.assertEqual(model("llama-3-70b").cache_bytes(1), 327_680)

    def test_deepseek_stores_a_576_number_latent(self):
        # (512 + 64) numbers x 61 blocks x 2 bytes
        self.assertEqual(model("deepseek-v3").cache_bytes(1), 70_272)

    def test_gemma_sliding_blocks_stop_at_4096(self):
        m = model("gemma-2-9b")
        per_block = 2 * 8 * 256 * 2
        self.assertEqual(m.cache_bytes(8192), 21 * per_block * 8192 + 21 * per_block * 4096)

    def test_granite_caches_only_its_4_attention_blocks(self):
        self.assertEqual(model("granite-4.0-h-small").cache_bytes(1), 4 * 2 * 8 * 128 * 2)

    def test_gpt2_is_float32(self):
        self.assertEqual(model("gpt2").cache_bytes(1), 2 * 12 * 768 * 4)


class Output(unittest.TestCase):
    def test_the_forum_request_is_over_the_window(self):
        code, out = run(os.path.join(HERE, "fixtures", "llama-3-70b.json"), "--tokens", "1750", "--answer", "8100")
        self.assertEqual(code, 3)
        self.assertIn("1,750 prompt + 8,100 answer = 9,850, over the 8,192-token window by 1,658", out)

    def test_gemma_on_a_24_gb_card(self):
        code, out = run(os.path.join(HERE, "fixtures", "gemma-2-9b.json"), "--tokens", "8192", "--gpu-gb", "24")
        self.assertEqual(code, 0)
        self.assertIn("2 conversations of 8,192 tokens fit", out)

    def test_deepseek_names_the_overcount(self):
        _code, out = run(os.path.join(HERE, "fixtures", "deepseek-v3.json"))
        self.assertIn("71.1 times too much", out)

    def test_estimate_is_labeled(self):
        p = os.path.join(HERE, "fixtures", "prompt.txt")
        _code, out = run(os.path.join(HERE, "fixtures", "gpt2.json"), "--prompt", p)
        self.assertIn("estimate at 4 characters per token", out)

    def test_unreadable_input_exits_1(self):
        code, _out = run(os.path.join(HERE, "fixtures", "missing.json"))
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
