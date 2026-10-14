"""Unit tests for classifier plumbing; these do not measure real-world accuracy."""
import os
import sys
import tempfile
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from main import CHUNKING


class TestVideoClassifier(unittest.TestCase):
    def test_chunking_mapping(self):
        self.assertEqual(CHUNKING["podcast"], "speech_turn_and_topic")
        self.assertEqual(CHUNKING["sports"], "event_and_play")
        self.assertIn("general", CHUNKING)

    def test_model_training_and_prediction(self):
        X = np.array([[0.0, 0.1], [0.1, 0.2], [1.0, 1.1], [1.1, 1.2]])
        y = np.array(["podcast", "podcast", "sports", "sports"])
        model = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200)),
        ])
        model.fit(X, y)
        self.assertEqual(model.predict([[0.05, 0.15]])[0], "podcast")
        self.assertAlmostEqual(
            float(model.predict_proba([[0.05, 0.15]])[0].sum()), 1.0, places=5
        )

    def test_feature_extraction_rejects_invalid_video(self):
        from features import extract_features
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with self.assertRaises(ValueError):
                extract_features(f.name)


if __name__ == "__main__":
    unittest.main()
