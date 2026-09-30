import unittest

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from placement_predictor import (
    canonicalize_columns,
    detect_target_column,
    encode_target,
    engineer_features,
    model_matrix,
    placement_probability,
    prepare_model_input,
)


class PlacementPredictorTests(unittest.TestCase):
    def test_canonicalizes_common_column_variants(self):
        frame = pd.DataFrame(
            {"Aptitude": [80], "Project_Count": [2], "Active Backlogs": [1]}
        )
        result = canonicalize_columns(frame)
        self.assertEqual(
            list(result.columns),
            ["Aptitude_Test_Score", "Projects", "Backlogs"],
        )

    def test_engineers_notebook_composite_features(self):
        frame = pd.DataFrame(
            {
                "CGPA": [8.0],
                "Aptitude": [75],
                "Coding": [7],
                "Projects": [2],
                "Communication": [8],
                "Soft Skills": [4],
                "Backlogs": [1],
            }
        )
        result = engineer_features(frame)
        self.assertAlmostEqual(result.loc[0, "Academic_Aptitude_Balance"], 6.0)
        self.assertAlmostEqual(result.loc[0, "Technical_Index"], 7.2)
        self.assertAlmostEqual(result.loc[0, "Employability_Rating"], 13.6)
        self.assertAlmostEqual(result.loc[0, "Academic_Risk_Score"], 1 / 8.00001)

    def test_detects_and_encodes_binary_target(self):
        frame = pd.DataFrame({"Placement_Status": ["Placed", "Not Placed"]})
        column = detect_target_column(frame)
        self.assertEqual(column, "Placement_Status")
        self.assertEqual(encode_target(frame[column]).tolist(), [1, 0])

    def test_unknown_target_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unrecognized values"):
            encode_target(pd.Series(["Placed", "Pending"]))

    def test_prediction_schema_reindexes_and_imputes(self):
        training = pd.DataFrame(
            {"CGPA": [6.0, 8.0, 7.0, 9.0], "Projects": [0, 1, 2, 3]}
        )
        matrix = model_matrix(training)
        model = RandomForestClassifier(n_estimators=5, random_state=42).fit(
            matrix, [0, 1, 0, 1]
        )
        columns = matrix.columns.tolist()
        medians = matrix.median().fillna(0).to_dict()

        model_input = prepare_model_input(
            {"CGPA": 8.5, "Projects": 2, "Unexpected_Field": 123},
            columns,
            medians,
        )
        self.assertEqual(list(model_input.columns), columns)
        self.assertEqual(model_input.shape, (1, len(columns)))
        probability = placement_probability(
            model, {"CGPA": 8.5, "Projects": 2}, columns, medians
        )
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
