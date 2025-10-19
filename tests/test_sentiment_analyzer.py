import unittest
from core.sentiment_analyzer import analyze_sentiment

class TestSentimentAnalyzer(unittest.TestCase):
    def test_analyze_sentiment_scores(self):
        positive_text = "Stock prices surge as company reports record profits and strong growth."
        positive_score = analyze_sentiment(positive_text)
        self.assertGreater(positive_score, 0, "Positive text should have a score greater than 0.")
        
        negative_text = "Market crashes due to unexpected economic downturn and poor earnings."
        negative_score = analyze_sentiment(negative_text)
        self.assertLess(negative_score, 0, "Negative text should have a score less than 0.")
        
        neutral_text = "The company is scheduled to release its next quarterly report on Tuesday."
        neutral_score = analyze_sentiment(neutral_text)
        self.assertAlmostEqual(neutral_score, 0, delta=0.5, msg="Neutral text should have a score close to 0.")

    def test_empty_input(self):
        empty_text = ""
        empty_score = analyze_sentiment(empty_text)
        self.assertEqual(empty_score, 0, "Empty text should result in a neutral score of 0.")
        
    def test_none_input(self):
        none_text = None
        none_score = analyze_sentiment(none_text)
        self.assertEqual(none_score, 0, "None input should result in a neutral score of 0.")

if __name__ == '__main__':
    unittest.main()