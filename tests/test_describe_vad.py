import unittest

from dlchat.affect.describe_vad import describe_vad
from dlchat.logging.schema import AffectVAD


class DescribeVADTests(unittest.TestCase):
    def test_neutral_axis(self) -> None:
        vad = AffectVAD(valence=0.50, arousal=0.50, dominance=0.50)
        desc = describe_vad(vad)
        self.assertIn("valence: neutral", desc[0])
        self.assertIn("arousal: neutral", desc[1])
        self.assertIn("dominance: neutral", desc[2])

    def test_stress_composite(self) -> None:
        vad = AffectVAD(valence=0.20, arousal=0.85, dominance=0.40)
        desc = describe_vad(vad)
        self.assertTrue(any("stress-leaning" in d for d in desc))


if __name__ == "__main__":
    unittest.main()
