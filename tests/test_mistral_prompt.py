import unittest

from dlchat.llm.mistral_instruct import build_mistral_instruct_prompt


class MistralPromptTests(unittest.TestCase):
    def test_includes_system_on_first_turn(self) -> None:
        prompt = build_mistral_instruct_prompt(
            system_prompt="SYS",
            history=[],
            user_message="USER",
        )
        self.assertIn("SYS", prompt)
        self.assertIn("USER", prompt)
        self.assertTrue(prompt.startswith("<s>[INST]"))

    def test_system_only_once(self) -> None:
        prompt = build_mistral_instruct_prompt(
            system_prompt="SYS",
            history=[("u1", "a1"), ("u2", "a2")],
            user_message="u3",
        )
        self.assertEqual(prompt.count("SYS"), 1)


if __name__ == "__main__":
    unittest.main()
