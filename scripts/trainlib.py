"""SFT/RL 共用的数据与编码工具；媒体预处理直接复用 baseline.py。

训练和推理必须看到完全相同的输入：8 帧、最长边 448、16 kHz 单声道、
与 baseline 相同的 prompt 文本。这里只负责把一条样本编码成 processor 输出，
模型前向与优化在 train_sft.py / train_rl.py 里。
"""

import json
from pathlib import Path

import soundfile as sf
from transformers.video_utils import VideoMetadata

from baseline import CLASSES, SYSTEM, ensure_audio, load_clip_frames, prepare_media


def load_rows(path):
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def dataset_root(path):
    return Path(path).parent


def gold_emotion(row):
    return row["views"]["M12"]["label"]


def gold_conflict(row):
    return row["sample_type"] == "Conflict"


def target_json(row, is_uncertain=False, question="none"):
    """监督目标：与推理输出同一套 JSON 字段。"""
    return json.dumps({
        "pred_emotion": gold_emotion(row),
        "is_conflict": "yes" if gold_conflict(row) else "no",
        "is_uncertain": "yes" if is_uncertain else "no",
        "clarifying_question": question if is_uncertain else "none",
    }, ensure_ascii=False)


def build_messages(row):
    """与 baseline.py 相同的 prompt 结构；媒体用占位 path，实际张量由 processor 传入。"""
    content = [{"type": "video", "path": str(row["video"])}]
    if row["protocol"] == "VT":
        prompt = "任务：结合视频画面和以下文字判断情感。\n文字：" + (row.get("text_content") or "")
    else:
        prompt = "任务：结合视频画面和音频判断情感。"
    content.append({"type": "text", "text": prompt})
    if row["protocol"] == "VA":
        content.append({"type": "audio", "path": "audio"})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def prepare_example(row, root, cache, frames=8, max_side=448):
    """解码并缓存媒体；返回帧数组、时长和（VA 的）音频路径。"""
    original = root / row["video"]
    video, audio, meta = prepare_media(original, cache, frames, max_side)
    clip = load_clip_frames(video)
    duration = json.loads(meta.read_text(encoding="utf-8"))["duration"]
    if row["protocol"] == "VA":
        ensure_audio(original, audio)
    wave = None
    if row["protocol"] == "VA":
        wave, rate = sf.read(audio, dtype="float32")
        assert rate == 16000 and wave.ndim == 1
    return {"frames": clip, "audio": audio, "wave": wave, "duration": duration}


def _video_kwargs(example, frames):
    duration = example["duration"]
    fps = frames / max(duration, 1e-6)
    return {
        "do_sample_frames": False,
        "use_audio_in_video": False,
        "fps": fps,
        "video_metadata": [VideoMetadata(
            total_num_frames=len(example["frames"]),
            fps=fps,
            duration=duration,
            frames_indices=list(range(len(example["frames"]))),
        )],
    }


def encode_text(processor, example, text, frames):
    kwargs = dict(
        text=[text],
        videos=[example["frames"]],
        return_tensors="pt",
        videos_kwargs=_video_kwargs(example, frames),
    )
    if example["wave"] is not None:
        kwargs.update(audio=[example["wave"]], audio_kwargs={"sampling_rate": 16000})
    return processor(**kwargs)


def encode_prompt(processor, row, example, frames=8):
    messages = build_messages(row)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return encode_text(processor, example, text, frames), text


def encode_sft(processor, row, example, target, frames=8):
    """prompt + 目标 JSON + eos；labels 只保留目标段，prompt 段置 -100。"""
    messages = build_messages(row)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    eos = processor.tokenizer.eos_token or ""
    full_text = prompt_text + target + eos
    prompt = encode_text(processor, example, prompt_text, frames)
    full = encode_text(processor, example, full_text, frames)
    prompt_ids = prompt["input_ids"][0].tolist()
    ids = full["input_ids"]
    ids_list = ids[0].tolist()
    # BPE 边界偶尔变化：用最长公共前缀而不是 prompt 长度
    prefix = 0
    for a, b in zip(prompt_ids, ids_list):
        if a != b:
            break
        prefix += 1
    labels = ids.clone()
    labels[:, :prefix] = -100
    full["labels"] = labels
    return full, prefix, ids.shape[1]


__all__ = [
    "CLASSES", "SYSTEM", "build_messages", "dataset_root", "encode_prompt",
    "encode_sft", "encode_text", "gold_conflict", "gold_emotion", "load_rows",
    "prepare_example", "target_json",
]
