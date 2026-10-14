"""Lightweight video feature extraction using sparse frames and MobileNetV3-Small."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None
_TRANSFORM = None


def _encoder():
    global _MODEL, _TRANSFORM
    if _MODEL is None:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        # Keep the convolutional backbone and global pooling; discard classifier.
        _MODEL = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten()).to(_DEVICE).eval()
        _TRANSFORM = weights.transforms()
    return _MODEL, _TRANSFORM


def extract_features(video_path: str, sample_frames: int = 8) -> np.ndarray:
    """Return a fixed-length vector for a video. Frames are sampled uniformly."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    duration = frame_count / fps if fps > 0 else 0.0

    if frame_count <= 0:
        cap.release()
        raise ValueError(f"No readable frames in video: {video_path}")

    indices = np.linspace(0, max(frame_count - 1, 0), num=min(sample_frames, frame_count), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Could not decode frames from video: {video_path}")

    model, transform = _encoder()
    batch = torch.stack([transform(Image.fromarray(f)) for f in frames]).to(_DEVICE)
    with torch.inference_mode():
        embeddings = model(batch).cpu().numpy().astype(np.float32)

    # Temporal signals: normalized frame-to-frame pixel difference and coarse scene-cut rate.
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (64, 64)) for f in frames]
    diffs = [float(np.mean(cv2.absdiff(a, b)) / 255.0) for a, b in zip(small, small[1:])]
    motion_mean = float(np.mean(diffs)) if diffs else 0.0
    motion_std = float(np.std(diffs)) if diffs else 0.0
    scene_cut_rate = float(np.mean(np.array(diffs) > 0.22)) if diffs else 0.0

    # Aggregate embeddings so the classifier sees the video's overall content and variation.
    visual = np.concatenate([embeddings.mean(axis=0), embeddings.std(axis=0)])
    metadata = np.array([
        np.log1p(duration),
        np.log1p(fps),
        np.log1p(width),
        np.log1p(height),
        width / height if height > 0 else 0.0,
        motion_mean,
        motion_std,
        scene_cut_rate,
        len(frames) / float(sample_frames),
    ], dtype=np.float32)
    return np.concatenate([visual, metadata]).astype(np.float32)
