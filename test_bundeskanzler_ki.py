import unittest
from bundeskanzler_ki import preprocess, detect_lang, corpus_original

class TestBundeskanzlerKI(unittest.TestCase):
    def test_preprocess_de(self):
        text = "Die Regierung plant Investitionen in Bildung und Infrastruktur."
        result = preprocess(text, lang='de')
        self.assertIsInstance(result, str)
        self.assertNotIn("die", result)
        self.assertNotIn("und", result)

    def test_preprocess_en(self):
        text = "The chancellor said he will increase taxes for the rich."
        result = preprocess(text, lang='en')
        self.assertIsInstance(result, str)
        self.assertNotIn("the", result)
        self.assertNotIn("for", result)

    def test_detect_lang_de(self):
        text = "Die Bundesregierung diskutiert über neue Klimaschutzmaßnahmen."
        lang = detect_lang(text)
        self.assertEqual(lang, 'de')

    def test_detect_lang_en(self):
        text = "Angela Merkel has announced that Germany will take in more refugees."
        lang = detect_lang(text)
        self.assertEqual(lang, 'en')

    def test_corpus_loaded(self):
        self.assertTrue(len(corpus_original) > 0)

if __name__ == "__main__":
    unittest.main()
