"""Train and run the video-type classifier."""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features

CHUNKING = {
    "gameplay": "state_change_and_event",
    "egocentric": "activity_and_temporal",
    "podcast": "speech_turn_and_topic",
    "wildlife": "scene_and_activity",
    "sports": "event_and_play",
    "news_lecture": "topic_and_slide",
    "music_performance": "music_segment_and_performance",
    "film_animation": "scene_and_shot",
    "surveillance": "activity_and_time_window",
    "general": "adaptive_scene_and_semantic",
}


def train(args):
    df = pd.read_csv(args.csv)
    required = {"video_path", "label"}
    if not required.issubset(df.columns):
        raise SystemExit("CSV must contain columns: video_path,label")
    df = df.dropna(subset=["video_path", "label"])
    if df["label"].nunique() < 2:
        raise SystemExit("Training needs at least two distinct labels.")
    X, y, valid_paths = [], [], []
    for row in df.itertuples(index=False):
        path, label = str(row.video_path), str(row.label).strip().lower()
        if not os.path.isfile(path):
            print(f"Skipping missing file: {path}", file=sys.stderr)
            continue
        try:
            X.append(extract_features(path, args.sample_frames))
            y.append(label)
            valid_paths.append(path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
    if len(y) < 4 or len(set(y)) < 2:
        raise SystemExit("Not enough valid labelled videos to train.")
    X = np.vstack(X)
    # Use a holdout only when every class has enough examples for a stratified split.
    counts = pd.Series(y).value_counts()
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=stratify
    )
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced", C=1.0)),
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Holdout accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, zero_division=0))
    os.makedirs(os.path.dirname(os.path.abspath(args.model)), exist_ok=True)
    joblib.dump({
        "model": model,
        "chunking": CHUNKING,
        "sample_frames": args.sample_frames,
        "feature_version": 1,
        "classes": sorted(set(y)),
    }, args.model)
    print(f"Saved model to {args.model}")


def predict(args):
    bundle = joblib.load(args.model)
    model = bundle["model"]
    paths = args.videos
    results = []
    for path in paths:
        try:
            x = extract_features(path, int(bundle.get("sample_frames", 8))).reshape(1, -1)
            probabilities = model.predict_proba(x)[0]
            idx = int(np.argmax(probabilities))
            label = str(model.classes_[idx])
            confidence = float(probabilities[idx])
            # Conservative fallback: low confidence should trigger a generic/adaptive chunker.
            final_label = label if confidence >= args.threshold else "general"
            result = {
                "video_path": path,
                "predicted_class": final_label,
                "model_top_class": label,
                "confidence": round(confidence, 4),
                "chunking_strategy": bundle["chunking"].get(final_label, CHUNKING["general"]),
                "needs_review": confidence < args.threshold,
            }
            results.append(result)
            print(json.dumps(result))
        except Exception as exc:
            print(json.dumps({"video_path": path, "error": str(exc)}))
    if args.output:
        pd.DataFrame(results).to_json(args.output, orient="records", indent=2)
        print(f"Saved predictions to {args.output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Lightweight video-type classifier for chunking strategy selection.")
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="Train from a labelled CSV with video_path,label columns.")
    tr.add_argument("--csv", required=True)
    tr.add_argument("--model", default="artifacts/video_classifier.joblib")
    tr.add_argument("--sample-frames", type=int, default=8)
    tr.add_argument("--test-size", type=float, default=0.2)
    tr.set_defaults(func=train)
    pr = sub.add_parser("predict", help="Classify one or more videos.")
    pr.add_argument("--model", default="artifacts/video_classifier.joblib")
    pr.add_argument("--threshold", type=float, default=0.45)
    pr.add_argument("--output", default="")
    pr.add_argument("videos", nargs="+")
    pr.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
