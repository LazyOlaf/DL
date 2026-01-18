import unittest
from dlchat.llm import build_prompt, build_tinyllama_prompt


class PromptTests(unittest.TestCase):
    def test_mistral_includes_system(self):
        prompt = build_prompt(system_prompt="SYS", history=[], user_message="USER")
        self.assertIn("SYS", prompt)
        self.assertIn("USER", prompt)
        self.assertTrue(prompt.startswith("<s>[INST]"))

    def test_mistral_system_once(self):
        prompt = build_prompt(system_prompt="SYS", history=[("u1", "a1"), ("u2", "a2")], user_message="u3")
        self.assertEqual(prompt.count("SYS"), 1)

    def test_tinyllama_format(self):
        prompt = build_tinyllama_prompt(system_prompt="SYS", history=[], user_message="USER")
        self.assertIn("<|system|>", prompt)
        self.assertIn("<|user|>", prompt)
        self.assertIn("<|assistant|>", prompt)


if __name__ == "__main__":
    unittest.main()
