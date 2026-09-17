from __future__ import annotations

import unittest

from model_service import MODEL_PATH, inspect_model, load_model, predict_price


@unittest.skipUnless(MODEL_PATH.is_file(), "model.joblib is not available in the expected model directory")
class ModelSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        cls.metadata = inspect_model(cls.model)

    def test_pipeline_loads_and_predicts_raw_values(self):
        values = {
            "year": 2018,
            "odometer": 65000,
            "model": "unseen-model-for-smoke-test",
            "region": "unseen-region-for-smoke-test",
            "manufacturer": "unseen-manufacturer-for-smoke-test",
            "state": "unseen-state-for-smoke-test",
            "condition": "good",
            "size": "mid-size",
            "cylinders": "unseen-cylinders-for-smoke-test",
            "fuel": "unseen-fuel-for-smoke-test",
            "title_status": "unseen-title-status-for-smoke-test",
            "transmission": "unseen-transmission-for-smoke-test",
            "drive": "unseen-drive-for-smoke-test",
            "type": "unseen-type-for-smoke-test",
            "paint_color": "unseen-paint-color-for-smoke-test",
        }
        self.assertEqual(tuple(self.metadata.feature_names), tuple(values))
        self.assertIsInstance(predict_price(self.model, values, self.metadata), float)

    def test_artifact_is_self_contained_pipeline(self):
        self.assertTrue(MODEL_PATH.is_file())
        self.assertEqual(self.metadata.estimator_name, "RandomForestRegressor")
        self.assertIn("preprocessor", self.model.named_steps)


if __name__ == "__main__":
    unittest.main()
