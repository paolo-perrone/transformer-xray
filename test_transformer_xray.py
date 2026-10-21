"""Known answers for transformer_xray.

The layout tests run on tiny random models (seconds, no real weights). The GPT-2 test runs the
real openai-community/gpt2 on "The capital of Poland is" and checks the numbers the issue prints.
"""
import os
import unittest

import transformer_xray as tx

GPT2 = os.environ.get("GPT2", "openai-community/gpt2")


def quiet(_):
    pass


class Layouts(unittest.TestCase):
    def test_gpt2_layout_runs(self):
        tok, m = tx.load("hf-internal-testing/tiny-random-gpt2")
        r = tx.inspect(tok, m, "The capital of Poland is", out=quiet)
        self.assertTrue(r["tokens"])

    def test_llama_layout_runs(self):
        tok, m = tx.load("hf-internal-testing/tiny-random-LlamaForCausalLM")
        r = tx.inspect(tok, m, "The capital of Poland is", out=quiet)
        self.assertTrue(r["tokens"])

    def test_unknown_layout_is_refused(self):
        class Odd:
            pass
        with self.assertRaises(ValueError):
            tx.parts(Odd())


class GPT2KnownAnswer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok, m = tx.load(GPT2)
        lines = []
        cls.r = tx.inspect(tok, m, "The capital of Poland is", out=lines.append)
        cls.text = "\n".join(lines)

    def test_answer_and_probability(self):
        self.assertEqual(self.r["best"], " Warsaw")
        self.assertAlmostEqual(self.r["p_best"], 0.103, places=3)

    def test_biggest_rise_is_block_11_feed_forward(self):
        kind, block, up = self.r["rise"]
        self.assertEqual((kind, block), ("feed-forward network", 11))
        self.assertAlmostEqual(up, 0.250, places=2)

    def test_switching_it_off_drops_warsaw(self):
        p, top, ptop = self.r["ablated"]
        self.assertAlmostEqual(p, 0.026, places=3)
        self.assertEqual(top, " the")

    def test_wrong_fact_in_context_wins(self):
        tok, m = tx.load(GPT2)
        r = tx.context_check(tok, m, "The capital of Poland is Kraków.", "The capital of Poland is", out=quiet)
        self.assertEqual(r["with"][0], " K")
        self.assertEqual(r["without"][0], " Warsaw")
        self.assertTrue(r["verdict"].startswith("from the context"))

    def test_neutral_context_leaves_training_answer(self):
        tok, m = tx.load(GPT2)
        r = tx.context_check(tok, m, "Poland is a country in Central Europe with a population of about 37 million.",
                             "The capital of Poland is", out=quiet)
        self.assertEqual(r["with"][0], " Warsaw")
        self.assertTrue(r["verdict"].startswith("from training"))

    def test_block_10_head_9_reads_poland(self):
        self.assertIn("block 10   head  9 puts 0.837 on ' Poland'", self.text)


if __name__ == "__main__":
    unittest.main()


class ApiMode(unittest.TestCase):
    """Recorded response shapes from the OpenAI and Anthropic APIs, no network."""

    def fake(self, answer_with, answer_without):
        def post(url, headers, body):
            text = body["messages"][0]["content"]
            ans = answer_with if text.startswith("Context:") else answer_without
            if "openai" in url:
                return {"choices": [{"message": {"content": ans},
                                     "logprobs": {"content": [{"top_logprobs": [
                                         {"token": ans.split()[0], "logprob": -0.1}]}]}}]}
            return {"content": [{"type": "text", "text": ans}]}
        return post

    def setUp(self):
        os.environ.setdefault("OPENAI_API_KEY", "test")
        os.environ.setdefault("ANTHROPIC_API_KEY", "test")

    def test_context_changes_the_answer(self):
        r = tx.api_check("openai", "m", "What is the capital of Poland?", "The capital is Kraków.",
                         runs=2, out=quiet, post=self.fake("Kraków", "Warsaw"))
        self.assertTrue(r["verdict"].startswith("from the context"))

    def test_same_answer_means_training(self):
        r = tx.api_check("anthropic", "m", "What is the capital of Poland?", "Poland is in Europe.",
                         runs=2, out=quiet, post=self.fake("Warsaw.", "warsaw"))
        self.assertTrue(r["verdict"].startswith("from training"))
