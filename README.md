# Problem 1.1 — Video Classifier

A lightweight video-type router that predicts which chunking policy should index a video.

## Quick start

```bash
pip install -r requirements.txt
python src/main.py --help
```

Create a labelled CSV with `video_path,label` columns (see `src/labels.csv.example`), then train:

```bash
python src/main.py train --csv src/labels.csv --model artifacts/video_classifier.joblib
```

Predict:

```bash
python src/main.py predict --model artifacts/video_classifier.joblib data/incoming.mp4
```

Read [`docs/design.md`](docs/design.md) for taxonomy, feature engineering, evaluation, and deployment plan.

**Important:** the repository task does not provide a labelled training manifest in the visible starter files. The classifier must be trained on labelled examples from the provided corpus before it can make meaningful predictions. The first feature-extraction run downloads pretrained MobileNetV3-Small weights.
