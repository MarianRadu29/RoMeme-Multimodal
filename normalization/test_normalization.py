import unittest
from normalization import TextNormalizer


class TextNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = TextNormalizer(spell_check=False)

    def test_keeps_short_meme_lines(self):
        text = "NU\nTE\nSUPARA"
        self.assertEqual(self.normalizer.normalize(text), "nu\nte\nsupara")

    def test_keeps_short_phrase_on_one_line(self):
        self.assertEqual(self.normalizer.normalize("nu te"), "nu te")

    def test_repairs_mixed_alnum_word(self):
        self.assertEqual(self.normalizer.normalize("R0man1a"), "romania")

    def test_preserves_plain_numbers(self):
        self.assertEqual(self.normalizer.normalize("anul 2024"), "anul 2024")

    def test_does_not_corrupt_numeric_suffixes(self):
        self.assertEqual(self.normalizer.normalize("clasa a 12-a"), "clasa a 12-a")

    def test_repairs_glued_and_short_ocr_words_without_symspell(self):
        text = "CuAGHEASMA SUU PRESIUNE\nSA FII MAFIO"
        self.assertEqual(
            self.normalizer.normalize(text),
            "cu agheasma sub presiune\nsa fii mafiot",
        )

    def test_drops_probable_artifact_lines_from_tesseract_noise(self):
        text = 'o ASTAINSEAMNA 4,\n\nA it\n\nț!\n\nA\n\nSA FII MAFIO\n\na'
        self.assertEqual(
            self.normalizer.normalize(text),
            "o asta inseamna 4,\nsa fii mafiot",
        )

    def test_repairs_known_spalatorie_phrase(self):
        text = '7oRIE DE PACATE SPA\n\n| —\n\nPoe\n\n"CU AGHEASMA\n\nSUB PRESIUNE'
        self.assertEqual(
            self.normalizer.normalize(text),
            'spalatorie de pacate spa\ncu agheasma\nsub presiune',
        )


if __name__ == "__main__":
    unittest.main()
