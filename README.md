# Problem 1.1 — Video Classifier

A lightweight video-type classifier that routes videos to appropriate chunking strategies.

## Setup
```bash
pip install -r requirements.txt
```

## Tests
```bash
python -m unittest discover -s tests -v
```
These tests verify basic code behavior only; they do not measure real-world classification accuracy.

## Train
Create `src/labels.csv` with `video_path,label` columns. Use actual video paths and labels from your corpus. See `src/labels.csv.example`.

```bash
python src/main.py train --csv src/labels.csv --model artifacts/video_classifier.joblib
```

## Predict
```bash
python src/main.py predict --model artifacts/video_classifier.joblib data/unseen_video.mp4 --output predictions.json
```

The prediction video should be excluded from the training set. First feature extraction downloads pretrained MobileNetV3-Small weights. See `docs/design.md` for feature engineering, evaluation, and deployment.
