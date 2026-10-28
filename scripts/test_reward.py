"""奖励函数的本地单测（纯标准库，不加载模型）。

    python scripts/test_reward.py

覆盖：JSON 解析、四维打分、过度追问惩罚、无补标时跳过追问项、组内归一化。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reward import extract_json, group_advantage, parse_prediction, reward  # noqa: E402

GOOD = '{"pred_emotion": "negative", "is_conflict": "no", "is_uncertain": "no", "clarifying_question": "none"}'


class RewardTest(unittest.TestCase):
    def test_extract_json_from_noise(self):
        text = "先判断一下：\n" + GOOD + "\n以上。"
        self.assertEqual(extract_json(text)["pred_emotion"], "negative")

    def test_invalid_json_scores_zero(self):
        result = reward("我不知道", "negative", False)
        self.assertEqual(result["total"], 0.0)
        self.assertEqual(result["format"], 0.0)

    def test_perfect_sample(self):
        result = reward(GOOD, "negative", False)
        self.assertAlmostEqual(result["total"], 0.9)  # 0.5 + 0.3 + 0.1

    def test_wrong_emotion_keeps_conflict_and_format(self):
        text = '{"pred_emotion": "positive", "is_conflict": "no", "is_uncertain": "no", "clarifying_question": "none"}'
        result = reward(text, "negative", False)
        self.assertAlmostEqual(result["total"], 0.4)

    def test_conflict_mismatch(self):
        result = reward(GOOD, "negative", True)
        self.assertAlmostEqual(result["emotion"], 1.0)
        self.assertAlmostEqual(result["conflict"], 0.0)

    def test_over_ask_penalty(self):
        text = '{"pred_emotion": "negative", "is_conflict": "no", "is_uncertain": "yes", "clarifying_question": "能再说说吗"}'
        result = reward(text, "negative", False, needs_clarification=False)
        self.assertEqual(result["over_ask"], -1.0)
        self.assertAlmostEqual(result["total"], 0.8)

    def test_no_annotation_skips_ask_term(self):
        text = '{"pred_emotion": "negative", "is_conflict": "no", "is_uncertain": "yes", "clarifying_question": "能再说说吗"}'
        result = reward(text, "negative", False, needs_clarification=None)
        self.assertEqual(result["over_ask"], 0.0)
        self.assertAlmostEqual(result["total"], 0.9)

    def test_parse_prediction_validity(self):
        pred, valid = parse_prediction(GOOD)
        self.assertTrue(valid)
        self.assertEqual(pred["is_conflict"], "no")
        _, invalid = parse_prediction('{"pred_emotion": "angry"}')
        self.assertFalse(invalid)

    def test_group_advantage(self):
        advantages = group_advantage([1.0, 1.0, 0.0, 0.0])
        self.assertGreater(advantages[0], 0)
        self.assertLess(advantages[2], 0)
        self.assertEqual(len(advantages), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
