import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

import app_core
import train


class TrainIntegrationTests(unittest.TestCase):
    def test_train_main_writes_expected_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "artifacts.joblib"

            with patch.object(train, "ARTIFACT_PATH", artifact_path):
                train.main()

            self.assertTrue(artifact_path.exists())

            artifact = joblib.load(artifact_path)
            self.assertIn("final_model", artifact)
            self.assertIn("best_name", artifact)
            self.assertIn("results", artifact)
            self.assertIn("shap_vals", artifact)
            self.assertIn("features", artifact)

            self.assertEqual(artifact["features"], app_core.FEATURES)
            self.assertIn(artifact["best_name"], artifact["results"])
            self.assertEqual(len(artifact["results"]), 4)
            self.assertEqual(artifact["shap_vals"].shape[1], len(app_core.FEATURES))
            self.assertTrue(hasattr(artifact["final_model"], "predict_proba"))


if __name__ == "__main__":
    unittest.main()