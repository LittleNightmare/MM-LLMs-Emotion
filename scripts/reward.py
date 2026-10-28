"""规则奖励与回答解析，训练与单测共用；不依赖 torch。

与 baseline.py 的判定口径保持一致：非法 JSON 或字段非法一律记错，
不做「扔掉格式错误再重算」的处理。
"""

import json

CLASSES = ("negative", "positive", "neutral")


def extract_json(text):
    """从自由文本里取第一个含 pred_emotion 的 JSON 对象。"""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "pred_emotion" in obj:
            return obj
    return {}


def parse_prediction(text):
    obj = extract_json(text)
    pred = {
        "pred_emotion": obj.get("pred_emotion"),
        "is_conflict": obj.get("is_conflict"),
        "is_uncertain": obj.get("is_uncertain"),
        "clarifying_question": obj.get("clarifying_question"),
    }
    valid = (
        pred["pred_emotion"] in CLASSES
        and pred["is_conflict"] in ("yes", "no")
        and pred["is_uncertain"] in ("yes", "no")
        and isinstance(pred["clarifying_question"], str)
    )
    return pred, valid


def reward(text, gold_emotion, gold_conflict, needs_clarification=None,
           weights=(0.5, 0.3, 0.1, 0.1)):
    """四维奖励：情感 / 冲突 / 格式 / 过度追问惩罚。

    needs_clarification 为 None 表示该样本没有补标，追问项跳过（不奖不罚）；
    为 False 表示人工裁定无需澄清，此时输出 is_uncertain=yes 或非 none 追问扣分。
    """
    pred, valid = parse_prediction(text)
    emotion = 1.0 if pred["pred_emotion"] == gold_emotion else 0.0
    conflict = 1.0 if pred["is_conflict"] == ("yes" if gold_conflict else "no") else 0.0
    fmt = 1.0 if valid else 0.0
    if needs_clarification is None:
        ask = 0.0
    elif needs_clarification:
        ask = 0.0
    else:
        over_ask = pred["is_uncertain"] == "yes" or pred["clarifying_question"] not in (None, "", "none")
        ask = -1.0 if over_ask else 0.0
    w_e, w_c, w_f, w_a = weights
    total = w_e * emotion + w_c * conflict + w_f * fmt + w_a * ask
    return {
        "total": total,
        "emotion": emotion,
        "conflict": conflict,
        "format": fmt,
        "over_ask": ask,
        "pred": pred,
    }


def group_advantage(rewards, eps=1e-4):
    """组内归一化：同一 prompt 的 G 个回报减去组均值、除以组标准差。"""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    return [(r - mean) / (std + eps) for r in rewards]
