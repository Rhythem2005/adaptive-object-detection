"""Deterministic, in-order frame sources for evaluation."""
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# stem like "<video>_<frame>" or "<video>-<frame>"; override with --seq-regex
DEFAULT_SEQ_REGEX = r"^(?P<video>.+?)[_-](?P<frame>\d+)$"


@dataclass
class Sequence:
    name: str
    frames: List[str]
    labels: Optional[List[str]] = None


def img2label_path(p: str) -> str:
    """Same convention as ultralytics: last /images/ -> /labels/, ext -> .txt"""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    head, sep, tail = p.rpartition(sa)
    base = (head + sb + tail) if sep else p
    return os.path.splitext(base)[0] + ".txt"


def _frame_key(stem):
    nums = re.findall(r"\d+", stem)
    return (int(nums[-1]) if nums else -1, stem)


def discover_sequences(images_dir, labels_dir=None, seq_regex=DEFAULT_SEQ_REGEX):
    images_dir = Path(images_dir)
    groups = {}
    subdirs = sorted(d for d in images_dir.iterdir() if d.is_dir())
    if subdirs:                                   # one sub-directory per video
        for d in subdirs:
            fs = [str(f) for f in d.iterdir() if f.suffix.lower() in IMG_EXT]
            if fs:
                groups[d.name] = sorted(fs, key=lambda f: _frame_key(Path(f).stem))
    else:                                         # flat directory, group by regex
        rx = re.compile(seq_regex)
        tmp = {}
        for f in images_dir.iterdir():
            if f.suffix.lower() not in IMG_EXT:
                continue
            m = rx.match(f.stem)
            if not m:
                raise ValueError(f"{f.name} does not match --seq-regex {seq_regex}")
            tmp.setdefault(m.group("video"), []).append((int(m.group("frame")), str(f)))
        groups = {k: [p for _, p in sorted(v)] for k, v in tmp.items()}

    seqs = []
    for name in sorted(groups):
        frames = groups[name]
        if labels_dir:
            labels = [str(Path(labels_dir) / Path(f).relative_to(images_dir).with_suffix(".txt")) for f in frames]
        else:
            labels = [img2label_path(f) for f in frames]
        seqs.append(Sequence(name, frames, labels))
    return seqs


def iter_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(path)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield i, f
        i += 1
    cap.release()
