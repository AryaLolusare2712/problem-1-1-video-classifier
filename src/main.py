"""Train and run a video-type classifier that selects a chunking policy."""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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
    if not {"video_path", "label"}.issubset(df.columns):
        raise SystemExit("CSV must contain columns: video_path,label")
    df = df.dropna(subset=["video_path", "label"])
    if df.empty:
        raise SystemExit("Training CSV contains no labelled rows.")

    X, y = [], []
    for row in df.itertuples(index=False):
        path = str(row.video_path)
        label = str(row.label).strip().lower()
        if not os.path.isfile(path):
            print(f"Skipping missing file: {path}", file=sys.stderr)
            continue
        try:
            X.append(extract_features(path, args.sample_frames))
            y.append(label)
        except Exception as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)

    if len(y) < 4 or len(set(y)) < 2:
        raise SystemExit("Need at least 4 readable videos and 2 distinct classes.")

    X = np.vstack(X)
    counts = pd.Series(y).value_counts()
    # Stratification needs >=2 items per class and enough test/train slots.
    n_test = int(np.ceil(len(y) * args.test_size))
    n_train = len(y) - n_test
    can_stratify = counts.min() >= 2 and n_test >= len(counts) and n_train >= len(counts)
    stratify = y if can_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=stratify
    )

    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1200, class_weight="balanced", C=1.0
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Holdout accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

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
    results = []
    for path in args.videos:
        try:
            x = extract_features(path, int(bundle.get("sample_frames", 8))).reshape(1, -1)
            probabilities = model.predict_proba(x)[0]
            idx = int(np.argmax(probabilities))
            top_label = str(model.classes_[idx])
            confidence = float(probabilities[idx])
            label = top_label if confidence >= args.threshold else "general"
            result = {
                "video_path": path,
                "predicted_class": label,
                "model_top_class": top_label,
                "confidence": round(confidence, 4),
                "chunking_strategy": bundle["chunking"].get(
                    label, CHUNKING["general"]
                ),
                "needs_review": confidence < args.threshold,
            }
            results.append(result)
            print(json.dumps(result))
        except Exception as exc:
            result = {"video_path": path, "error": str(exc)}
            results.append(result)
            print(json.dumps(result), file=sys.stderr)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved predictions to {args.output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight video classifier for chunking strategy selection."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Train from labelled CSV.")
    train_parser.add_argument("--csv", required=True)
    train_parser.add_argument("--model", default="artifacts/video_classifier.joblib")
    train_parser.add_argument("--sample-frames", type=int, default=8)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.set_defaults(func=train)

    predict_parser = sub.add_parser("predict", help="Classify one or more videos.")
    predict_parser.add_argument("--model", default="artifacts/video_classifier.joblib")
    predict_parser.add_argument("--threshold", type=float, default=0.45)
    predict_parser.add_argument("--output", default="")
    predict_parser.add_argument("videos", nargs="+")
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    if getattr(args, "test_size", 0.2) <= 0 or getattr(args, "test_size", 0.2) >= 1:
        parser.error("--test-size must be between 0 and 1")
    if getattr(args, "threshold", 0.45) < 0 or getattr(args, "threshold", 0.45) > 1:
        parser.error("--threshold must be between 0 and 1")
    args.func(args)


if __name__ == "__main__":
    main()
