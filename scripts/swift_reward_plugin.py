"""ms-swift GRPO 奖励插件：情感 / 冲突 / 格式 / 过度追问四维规则奖励。

注册方式：

    swift rlhf --rlhf_type grpo ... \
        --external_plugins scripts/swift_reward_plugin.py \
        --reward_funcs emotion_conflict_format_overask

数据列通过 kwargs 传入（to_swift.py --task grpo 生成的列）：

- gold_emotion:        "negative" / "positive" / "neutral"
- gold_conflict:       "yes" / "no"
- needs_clarification: 1=需要澄清，0=无需澄清，null=没有补标（追问项跳过）

奖励口径与 docs/训练方案.md 第 3 节一致；判定复用 scripts/reward.py，
不重新定义标签。总奖励范围约 [-0.1, 1.0]。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward import reward  # noqa: E402
from swift.rewards import ORM, orms  # noqa: E402


def _to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("yes", "true", "1", "y"):
        return True
    if text in ("no", "false", "0", "n", ""):
        return False
    return None


def _column(values, n):
    if values is None:
        return [None] * n
    return list(values)


class EmotionConflictFormatOverAsk(ORM):
    """四维加权总奖励，训练主奖励函数。"""

    def __init__(self, args=None, **kwargs):
        super().__init__(args)

    def __call__(self, completions, gold_emotion=None, gold_conflict=None,
                 needs_clarification=None, **kwargs):
        n = len(completions)
        emotions = _column(gold_emotion, n)
        conflicts = _column(gold_conflict, n)
        needs = _column(needs_clarification, n)
        scores = []
        for text, emotion, conflict, need in zip(completions, emotions, conflicts, needs):
            result = reward(
                text,
                gold_emotion=emotion,
                gold_conflict=_to_bool(conflict),
                needs_clarification=_to_bool(need),
            )
            scores.append(float(result["total"]))
        return scores


class EmotionOnly(ORM):
    """分量：情感是否正确（0/1），用于观察曲线。"""

    def __init__(self, args=None, **kwargs):
        super().__init__(args)

    def __call__(self, completions, gold_emotion=None, **kwargs):
        emotions = _column(gold_emotion, len(completions))
        return [float(reward(text, emotion, False)["emotion"])
                for text, emotion in zip(completions, emotions)]


class ConflictOnly(ORM):
    """分量：冲突判断是否正确（0/1）。"""

    def __init__(self, args=None, **kwargs):
        super().__init__(args)

    def __call__(self, completions, gold_conflict=None, **kwargs):
        conflicts = _column(gold_conflict, len(completions))
        return [float(reward(text, None, _to_bool(conflict))["conflict"])
                for text, conflict in zip(completions, conflicts)]


class FormatOnly(ORM):
    """分量：JSON 字段是否合法（0/1）。"""

    def __init__(self, args=None, **kwargs):
        super().__init__(args)

    def __call__(self, completions, **kwargs):
        return [float(reward(text, None, False)["format"]) for text in completions]


class OverAskPenalty(ORM):
    """分量：无需澄清却追问的惩罚（0 或 -1），没有补标时为 0。"""

    def __init__(self, args=None, **kwargs):
        super().__init__(args)

    def __call__(self, completions, needs_clarification=None, **kwargs):
        needs = _column(needs_clarification, len(completions))
        return [float(reward(text, None, False, _to_bool(need))["over_ask"])
                for text, need in zip(completions, needs)]


orms["emotion_conflict_format_overask"] = EmotionConflictFormatOverAsk
orms["emotion_only"] = EmotionOnly
orms["conflict_only"] = ConflictOnly
orms["format_only"] = FormatOnly
orms["over_ask_penalty"] = OverAskPenalty
