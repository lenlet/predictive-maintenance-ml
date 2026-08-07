import unittest
from unittest.mock import patch

import pandas as pd

import app_core


def _sample_frame():
    return pd.DataFrame([
        {
            "ID": 1,
            "Product_ID": "M1",
            "Type": "M",
            "Air_Temp": 300.0,
            "Process_Temp": 310.0,
            "Rot_Speed": 1500,
            "Torque": 40.0,
            "Tool_Wear": 50,
            "Machine_Failure": 0,
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        },
        {
            "ID": 2,
            "Product_ID": "H2",
            "Type": "H",
            "Air_Temp": 299.5,
            "Process_Temp": 309.5,
            "Rot_Speed": 1600,
            "Torque": 35.0,
            "Tool_Wear": 80,
            "Machine_Failure": 1,
            "TWF": 1,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        },
    ])


class AppCoreTests(unittest.TestCase):
    def test_engineer_features_adds_expected_columns(self):
        engineered = app_core.engineer_features(_sample_frame())

        self.assertIn("Temp_Diff", engineered.columns)
        self.assertIn("Power", engineered.columns)
        self.assertIn("Wear_Strain", engineered.columns)
        self.assertIn("Speed_Torque_Ratio", engineered.columns)
        self.assertIn("Thermal_Stress", engineered.columns)
        self.assertIn("Type_Encoded", engineered.columns)

        self.assertAlmostEqual(engineered.loc[0, "Temp_Diff"], 10.0)
        self.assertAlmostEqual(
            engineered.loc[0, "Power"], 40.0 * (1500 * 2 * 3.141592653589793 / 60)
        )
        self.assertAlmostEqual(engineered.loc[0, "Wear_Strain"], 2000.0)
        self.assertAlmostEqual(engineered.loc[0, "Speed_Torque_Ratio"], 37.5)
        self.assertEqual(engineered.loc[0, "Type_Encoded"], 1)
        self.assertEqual(engineered.loc[1, "Type_Encoded"], 2)

    def test_build_input_row_uses_same_feature_order(self):
        row = app_core.build_input_row(300.0, 310.0, 1500, 40.0, 50, "M")

        self.assertListEqual(list(row.columns), app_core.FEATURES)
        self.assertEqual(row.shape, (1, len(app_core.FEATURES)))
        self.assertAlmostEqual(row.iloc[0]["Temp_Diff"], 10.0)
        self.assertEqual(row.iloc[0]["Type_Encoded"], 1)

    def test_prepare_data_returns_expected_shapes(self):
        df, X_train, X_test, X, y, y_train, y_test, X_train_s, X_test_s, scaler = app_core.prepare_data()

        self.assertEqual(len(df), 10000)
        self.assertEqual(len(X), 10000)
        self.assertEqual(len(y), 10000)
        self.assertEqual(X_train.shape[0], 8000)
        self.assertEqual(X_test.shape[0], 2000)
        self.assertEqual(X_train_s.shape, (8000, len(app_core.FEATURES)))
        self.assertEqual(X_test_s.shape, (2000, len(app_core.FEATURES)))
        self.assertEqual(len(y_train), 8000)
        self.assertEqual(len(y_test), 2000)
        self.assertEqual(scaler.n_features_in_, len(app_core.FEATURES))

    @patch("app_core.pd.read_csv")
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_data_prefers_local_csv(self, mock_exists, mock_read_csv):
        mock_read_csv.return_value = _sample_frame()

        frame = app_core.load_data()

        mock_exists.assert_called_once()
        mock_read_csv.assert_called_once_with(app_core.DATA_PATH)
        self.assertEqual(frame.shape[0], 2)

    @patch("app_core.pd.read_csv")
    @patch("pathlib.Path.exists", return_value=False)
    def test_load_data_falls_back_to_url(self, mock_exists, mock_read_csv):
        mock_read_csv.return_value = _sample_frame()

        frame = app_core.load_data()

        mock_exists.assert_called_once()
        mock_read_csv.assert_called_once_with(app_core.DATA_URL)
        self.assertEqual(frame.shape[0], 2)


if __name__ == "__main__":
    unittest.main()