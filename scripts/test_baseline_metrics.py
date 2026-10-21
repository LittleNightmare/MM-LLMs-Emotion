"""Check scoring failures and absent classes, where optimistic reporting is easy."""
import unittest
from baseline import extract_json, is_json_object, summarize


class MetricsTests(unittest.TestCase):
    def test_invalid_output_counts_against_accuracy(self):
        def row(label, prediction):
            return dict(gold_emotion=label, gold_conflict="no", prediction=prediction,
                        protocol="VT", generated=False, use_in_main=True,
                        seconds=1, format_valid=bool(prediction))
        metrics = summarize([
            row("negative", {"pred_emotion": "negative", "is_conflict": "no"}),
            row("positive", {}),
        ])["all"]
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["emotion_accuracy"], 0.5)
        self.assertEqual(metrics["emotion_per_class"]["neutral"]["support"], 0)
        self.assertAlmostEqual(metrics["emotion_macro_f1_fixed3"], 1/3)
        self.assertAlmostEqual(metrics["emotion_macro_f1_present"], 0.5)

    def test_json_extraction(self):
        self.assertEqual(extract_json('```json\n{"pred_emotion":"positive"}\n```'),
                         {"pred_emotion": "positive"})
        self.assertEqual(extract_json('{"pred_emotion":'), {})
        self.assertEqual(extract_json("positive"), {})
        self.assertFalse(is_json_object('{"pred_emotion":"positive"}\nmore text'))
        self.assertTrue(is_json_object('{"pred_emotion":"positive"}'))

    def test_wrong_value_types_are_scored_as_errors(self):
        metrics = summarize([dict(
            gold_emotion="negative", gold_conflict="yes",
            prediction={"pred_emotion": None, "is_conflict": []},
            protocol="VA", generated=True, use_in_main=True, seconds=1,
        )])["all"]
        self.assertEqual(metrics["emotion_accuracy"], 0)
        self.assertEqual(metrics["conflict_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
